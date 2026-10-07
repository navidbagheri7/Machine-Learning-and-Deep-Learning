"""Controlled restoration GAN objectives, including Wasserstein-GP and lazy R1.

Every variant starts from the same seed-specific, validation-selected direct model.
The direct fine-tuning control separates additional fitting from adversarial effects.
RGB-interpolant gradient penalties keep the observed condition fixed. All critic
derivatives and penalties are computed in float32, including second derivatives.
"""

import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .artifacts import Run, write_json
from .generative import (
    FeatureLoss,
    Images,
    build_model,
    evaluate,
    observation,
    reconstruct,
    save_arrays,
    save_checkpoint,
)

METHODS = ["direct_finetune", "hinge_sn", "non_saturating_r1", "least_squares", "wasserstein_gp"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Critic(nn.Module):
    def __init__(self, task, spectral=True):
        super().__init__()
        channels = 4 if task == "colorization" else 6
        layers = []
        for width, stride in [(48, 2), (96, 2), (192, 2), (192, 1)]:
            layers.extend([nn.Conv2d(channels, width, 4, stride, 1), nn.LeakyReLU(0.2)])
            channels = width
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        # Initialize all raw weights before adding SN power-iteration vectors.
        if spectral:
            layers = [
                nn.utils.spectral_norm(layer) if isinstance(layer, nn.Conv2d) else layer
                for layer in layers
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, condition, rgb):
        return self.net(torch.cat([condition, rgb], 1)).flatten(1).mean(1)


def critic_objective(real, fake, method):
    if method == "hinge_sn":
        return 0.5 * (F.relu(1 - real).mean() + F.relu(1 + fake).mean())
    if method == "non_saturating_r1":
        return F.softplus(-real).mean() + F.softplus(fake).mean()
    if method == "least_squares":
        return 0.5 * ((real - 1).square().mean() + fake.square().mean())
    if method == "wasserstein_gp":
        return fake.mean() - real.mean()
    raise ValueError("Unknown adversarial objective")


def generator_objective(fake, method):
    if method == "non_saturating_r1":
        return F.softplus(-fake).mean()
    if method == "least_squares":
        return 0.5 * (fake - 1).square().mean()
    if method in ["hinge_sn", "wasserstein_gp"]:
        return -fake.mean()
    raise ValueError("Unknown adversarial objective")


def gradient_penalty(critic, condition, real, fake, generator=None):
    alpha = torch.rand((len(real), 1, 1, 1), device=real.device, generator=generator)
    mixed = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    scores = critic(condition.detach(), mixed)
    gradient = torch.autograd.grad(scores.sum(), mixed, create_graph=True)[0]
    norm = gradient.flatten(1).norm(2, dim=1)
    return (norm - 1).square().mean(), norm.mean()


def r1_penalty(scores, real):
    gradient = torch.autograd.grad(scores.sum(), real, create_graph=True)[0]
    return gradient.flatten(1).square().sum(1).mean()


def train(config, task, method, seed, device):
    if task not in config["tasks"] or method not in METHODS or seed not in config["seeds"]:
        raise ValueError("Case must be in the declared protocol")
    if config["critic_steps"] < 1 or config["r1_interval"] < 1:
        raise ValueError("Positive critic updates and R1 interval required")
    directory = Path(config["output"]) / task / method / str(seed)
    protocol = dict(benchmark=config, task=task, method=method, seed=seed, device=device)
    record_path = directory / "run.json"
    sources = {
        name: sha256(Path(__file__).parent / name)
        for name in ["adversarial.py", "generative.py", "restoration.py", "artifacts.py"]
    }
    if record_path.exists():
        old = json.loads(record_path.read_text())
        assert old["config"] == protocol and old.get("fitting_source_sha256") == sources, (
            "Changed fitting protocol/source; preserve previous fit"
        )
        if old["status"] == "complete":
            print("Preserved complete adversarial fit", task, method, seed, flush=True)
            return
    run = Run(directory, protocol)
    run.record["fitting_source_sha256"] = sources
    root = Path(config["data_root"])
    torch.hub.set_dir(str(root / "models/torch/hub"))
    if not (Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth").exists():
        raise FileNotFoundError("Prepare LPIPS weights explicitly before training")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = build_model("l1", task, config, initialize_pretrained=False).to(device)
    initial_path = Path(config["warm_start"].format(task=task, seed=seed))
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    assert (
        initial["method"] == "l1" and initial["seed"] == seed and initial["config"]["task"] == task
    )
    model.load_state_dict(initial["state_dict"], strict=True)
    run.record["warm_start_sha256"] = sha256(initial_path)
    write_json(record_path, run.record | {"status": "running"})
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    feature = FeatureLoss(config["pretrained_weights"]).to(device)
    torch.manual_seed(100000 + seed)
    critic = (
        None
        if method == "direct_finetune"
        else Critic(task, spectral=method != "wasserstein_gp").to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999), weight_decay=1e-4
    )
    dopt = (
        None
        if critic is None
        else torch.optim.Adam(critic.parameters(), lr=config["learning_rate"] / 2, betas=(0.0, 0.9))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    loader = DataLoader(
        Images(root, "train", config["train_images"], config["crop"]),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=device.startswith("cuda"),
        generator=torch.Generator().manual_seed(seed),
    )
    validation = DataLoader(Images(root, "validation", config["validation_images"]), batch_size=4)
    penalty_rng = torch.Generator(device=device).manual_seed(3500 + seed)

    @torch.no_grad()
    def validate():
        total, count = 0.0, 0
        for rgb in validation:
            rgb = rgb.to(device)
            condition, target = observation(rgb, task)
            field = ema(condition)
            value = F.l1_loss(field, target) + config["feature_weight"] * feature(
                reconstruct(condition, field, task), rgb
            )
            total += float(value) * len(rgb)
            count += len(rgb)
        assert count == config["validation_images"]
        return total / count

    history = [dict(epoch=0, validation_loss=validate(), train={}, elapsed_seconds=0)]
    best = history[0]["validation_loss"]
    best_epoch = 0
    best_state = copy.deepcopy({key: value.cpu() for key, value in ema.state_dict().items()})
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    resume_path = checkpoint_dir / "training.pt"
    updates = 0
    start_epoch = 0
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        assert saved["protocol"] == protocol and saved["source_sha256"] == sources
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if critic is not None:
            critic.load_state_dict(saved["critic"])
            dopt.load_state_dict(saved["dopt"])
        history, best, best_epoch, best_state, updates = (
            saved["history"],
            saved["best"],
            saved["best_epoch"],
            saved["best_state"],
            saved["updates"],
        )
        start_epoch = len(history) - 1
        torch.set_rng_state(saved["rng"])
        loader.generator.set_state(saved["loader_rng"])
        penalty_rng.set_state(saved["penalty_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    for epoch in range(start_epoch, config["epochs"]):
        started = time.monotonic()
        model.train()
        totals, count = {}, 0
        for rgb in loader:
            rgb = rgb.to(device, non_blocking=True)
            condition, target = observation(rgb, task)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                field = model(condition)
                prediction = reconstruct(condition, field, task)
                pixel = F.l1_loss(field, target)
                perceptual = feature(prediction, rgb)
                loss = pixel + config["feature_weight"] * perceptual
            terms = dict(pixel_l1=pixel, feature_l1=perceptual)
            if critic is not None:
                critic.requires_grad_(True).train()
                for _ in range(config["critic_steps"]):
                    dopt.zero_grad(set_to_none=True)
                    real = (
                        rgb.detach()
                        .float()
                        .requires_grad_(
                            method == "non_saturating_r1" and updates % config["r1_interval"] == 0
                        )
                    )
                    real_scores = critic(condition.detach().float(), real)
                    fake_scores = critic(condition.detach().float(), prediction.detach().float())
                    dloss = critic_objective(real_scores, fake_scores, method)
                    terms.update(real_score=real_scores.mean(), fake_score=fake_scores.mean())
                    if method == "wasserstein_gp":
                        gp, norm = gradient_penalty(
                            critic, condition.float(), real, prediction.float(), penalty_rng
                        )
                        dloss = dloss + config["gradient_penalty"] * gp
                        terms.update(gradient_penalty=gp, interpolant_gradient_norm=norm)
                    elif real.requires_grad:
                        penalty = r1_penalty(real_scores, real)
                        dloss = dloss + 0.5 * config["r1_gamma"] * config["r1_interval"] * penalty
                        terms["lazy_r1_penalty"] = penalty
                    if not torch.isfinite(dloss):
                        raise FloatingPointError("Nonfinite critic objective")
                    dloss.backward()
                    dopt.step()
                critic.requires_grad_(False).eval()
                adversarial = generator_objective(
                    critic(condition.float(), prediction.float()), method
                )
                loss = loss + config["adversarial_weight"] * adversarial
                terms.update(critic_objective=dloss, generator_adversarial=adversarial)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite generator objective")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.parameters()):
                    a.lerp_(b, 1 - config["ema_decay"])
                for a, b in zip(ema.buffers(), model.buffers()):
                    a.copy_(b)
            for key, value in dict(
                objective=loss, generator_gradient_norm=gradient_norm, **terms
            ).items():
                totals[key] = totals.get(key, 0) + float(value.detach()) * len(rgb)
            count += len(rgb)
            updates += 1
        assert count == config["train_images"]
        validation_loss = validate()
        history.append(
            dict(
                epoch=epoch + 1,
                train={key: value / count for key, value in totals.items()},
                validation_loss=validation_loss,
                elapsed_seconds=time.monotonic() - started,
            )
        )
        if validation_loss < best:
            best, best_epoch = validation_loss, epoch + 1
            best_state = copy.deepcopy(
                {key: value.cpu() for key, value in ema.state_dict().items()}
            )
        scheduler.step()
        save_checkpoint(
            dict(
                protocol=protocol,
                source_sha256=sources,
                model=model.state_dict(),
                ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                critic=critic.state_dict() if critic is not None else None,
                dopt=dopt.state_dict() if dopt else None,
                history=history,
                best=best,
                best_epoch=best_epoch,
                best_state=best_state,
                updates=updates,
                rng=torch.get_rng_state(),
                loader_rng=loader.generator.get_state(),
                penalty_rng=penalty_rng.get_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.startswith("cuda") else None,
            ),
            resume_path,
        )
        write_json(directory / "history.json", history)
        print(
            task,
            method,
            seed,
            epoch + 1,
            "validation",
            round(validation_loss, 6),
            "seconds",
            round(history[-1]["elapsed_seconds"], 2),
            flush=True,
        )
    ema.load_state_dict(best_state)
    checkpoint = checkpoint_dir / "selected.pt"
    save_checkpoint(
        dict(
            state_dict=best_state,
            config=config | dict(task=task),
            method=method,
            seed=seed,
            best_epoch=best_epoch,
        ),
        checkpoint,
    )
    scoring = config | dict(task=task, device=device)
    with torch.no_grad():
        rows, scores, gallery = evaluate(ema, method, seed, scoring)
    public = Path(
        "results/smoke/adversarial_evidence"
        if config.get("quick")
        else "assets/source_data/adversarial_variants"
    )
    public.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_{method}_{seed}"
    save_arrays(public / (stem + "_scores.npz"), scores)
    save_arrays(public / (stem + "_gallery.npz"), gallery)
    manifest = json.loads((root / "restoration256/manifest.json").read_text())
    run.finish(
        dict(
            rows=rows,
            history=history,
            best_epoch=best_epoch,
            selection_objective=best,
            generator_updates=updates,
            critic_updates=updates * config["critic_steps"] if critic else 0,
            checkpoint_sha256=sha256(checkpoint),
            numeric_sha256={
                suffix: sha256(public / (stem + suffix + ".npz"))
                for suffix in ["_scores", "_gallery"]
            },
            data_sha256={key: value for key, value in manifest.items() if key.endswith("sha256")},
            protocol="Seed-specific validation-selected direct warm start, shared generator/data/20-epoch budget and direct fine-tuning control; all epochs completed even when initial model remains selected. Critic objective/stabilization varies; one critic step per generator. EMA validation selection includes epoch0, first tie. Deterministic same400 held-out photos, no test selection; original metric protocol3. This does not measure unconditional GAN diversity or reproduce published WGAN/StyleGAN budgets.",
        )
    )
