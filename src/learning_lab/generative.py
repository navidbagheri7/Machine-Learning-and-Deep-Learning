"""Conditional image generation with explicit training, sampling and evaluation contracts.

Local models are controlled research baselines, not reproductions of large published
training budgets. Colorization predicts Lab chroma; SR predicts an RGB residual to a
4x bicubic observation. Test-time VAE inference uses the learned conditional prior.
"""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from .artifacts import Run, write_json
from .restoration import TransferColorizer, lab_to_rgb, per_image_metrics, rgb_to_lab, summarize


def observation(rgb, task):
    if task not in ["colorization", "super_resolution"]:
        raise ValueError("Unknown restoration task")
    if task == "colorization":
        lab = rgb_to_lab(rgb)
        return lab[:, :1], lab[:, 1:]
    low = F.interpolate(rgb, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True)
    base = F.interpolate(low, size=rgb.shape[-2:], mode="bicubic", align_corners=False)
    return base, 2 * (rgb - base)


def reconstruct(condition, field, task):
    if task not in ["colorization", "super_resolution"]:
        raise ValueError("Unknown restoration task")
    if task == "colorization":
        return lab_to_rgb(torch.cat([condition, field], 1))
    return (condition + field / 2).clamp(0, 1)


class Images(Dataset):
    def __init__(self, root, split, count, crop=128):
        if split not in ["train", "validation", "test"] or not isinstance(count, int) or count < 1:
            raise ValueError("Known split and positive integer image count required")
        root = Path(root) / "restoration256"
        self.array = np.load(root / ("train.npy" if split == "train" else "val.npy"), mmap_mode="r")
        offset = 400 if split == "test" else 0
        capacity = len(self.array) if split == "train" else 400
        if count > capacity or offset + count > len(self.array):
            raise ValueError("Requested data exceeds prepared split")
        self.indices = np.arange(offset, offset + count)
        self.training, self.crop = split == "train", crop
        if self.training and (crop < 16 or crop % 8 or crop > min(self.array.shape[1:3])):
            raise ValueError("Training crop must fit the image, be at least 16 and divisible by 8")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x = torch.from_numpy(self.array[self.indices[index]].copy()).permute(2, 0, 1).float() / 255
        if self.training:
            h, w = x.shape[-2:]
            top = torch.randint(h - self.crop + 1, ()).item()
            left = torch.randint(w - self.crop + 1, ()).item()
            x = x[:, top : top + self.crop, left : left + self.crop]
            if torch.rand(()) < 0.5:
                x = x.flip(-1)
        return x


class Residual(nn.Module):
    def __init__(self, cin, cout, embedding):
        super().__init__()
        self.first = nn.Sequential(
            nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1)
        )
        self.second = nn.Sequential(
            nn.GroupNorm(8, cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1)
        )
        self.affine = nn.Linear(embedding, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, embedding):
        h = self.first(x)
        scale, shift = self.affine(embedding).chunk(2, 1)
        h = self.second(h * (1 + scale[:, :, None, None]) + shift[:, :, None, None])
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, inputs, outputs, width=48, embedding=192):
        super().__init__()
        self.head = nn.Conv2d(inputs, width, 3, padding=1)
        self.down = nn.ModuleList(
            [
                Residual(width, width, embedding),
                Residual(width, 2 * width, embedding),
                Residual(2 * width, 4 * width, embedding),
            ]
        )
        self.middle = Residual(4 * width, 4 * width, embedding)
        self.attention = nn.MultiheadAttention(4 * width, 4, batch_first=True)
        self.up = nn.ModuleList(
            [
                Residual(8 * width, 2 * width, embedding),
                Residual(4 * width, width, embedding),
                Residual(2 * width, width, embedding),
            ]
        )
        self.tail = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, outputs, 3, padding=1)
        )
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x, embedding):
        h, skips = self.head(x), []
        for block in self.down:
            h = block(h, embedding)
            skips.append(h)
            h = F.avg_pool2d(h, 2)
        h = self.middle(h, embedding)
        tokens = h.flatten(2).transpose(1, 2)
        # Attention only at 1/8 resolution; need_weights=False uses SDPA on CUDA.
        tokens = tokens + self.attention(tokens, tokens, tokens, need_weights=False)[0]
        h = tokens.transpose(1, 2).reshape_as(h)
        for block, skip in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], 1), embedding)
        return self.tail(h)


class Generator(nn.Module):
    def __init__(self, task, width=48, pretrained=None):
        super().__init__()
        self.task = task
        self.model = (
            TransferColorizer(weights=pretrained)
            if task == "colorization"
            else ConditionalUNet(3, 3, width)
        )

    def forward(self, condition):
        if self.task == "colorization":
            return self.model(condition)
        return self.model(condition, condition.new_zeros(len(condition), 192))


class GaussianEncoder(nn.Module):
    def __init__(self, channels, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 2 * latent),
        )

    def forward(self, x):
        mean, logvar = self.net(x).chunk(2, 1)
        return mean, logvar.clamp(-10, 5)


def gaussian_kl(q_mean, q_logvar, p_mean, p_logvar):
    return (
        0.5
        * (p_logvar - q_logvar + (q_logvar.exp() + (q_mean - p_mean).square()) / p_logvar.exp() - 1)
        .sum(1)
        .mean()
    )


class ConditionalVAE(nn.Module):
    def __init__(self, task, width=48, latent=32, pretrained=None, semantic=False):
        super().__init__()
        inputs, outputs = (1, 2) if task == "colorization" else (3, 3)
        self.prior, self.posterior = (
            GaussianEncoder(inputs, latent),
            GaussianEncoder(inputs + outputs, latent),
        )
        self.decoder = ConditionalUNet(inputs, outputs, width)
        self.embedding = nn.Sequential(nn.Linear(latent, 192), nn.SiLU(), nn.Linear(192, 192))
        self.semantic = (
            TransferColorizer(weights=pretrained)
            if task == "colorization" and (pretrained or semantic)
            else None
        )
        self.frozen_semantic = False

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_semantic and self.semantic is not None:
            self.semantic.eval()
        return self

    def forward(self, condition, target=None, generator=None):
        p_mean, p_logvar = self.prior(condition)
        mean, logvar = (
            (p_mean, p_logvar)
            if target is None
            else self.posterior(torch.cat([condition, target], 1))
        )
        noise = torch.randn(mean.shape, device=mean.device, generator=generator)
        z = mean + (0.5 * logvar).exp() * noise
        field = self.decoder(condition, self.embedding(z))
        if self.semantic is not None:
            field = field + self.semantic(condition)
        kl = (
            gaussian_kl(mean, logvar, p_mean, p_logvar)
            if target is not None
            else field.new_zeros(())
        )
        return field, kl


class PatchDiscriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers, cin = [], channels
        for width, stride in [(48, 2), (96, 2), (192, 2), (192, 1)]:
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(cin, width, 4, stride, 1)),
                nn.LeakyReLU(0.2),
            ]
            cin = width
        self.net = nn.Sequential(*layers, nn.Conv2d(cin, 1, 3, padding=1))

    def forward(self, condition, rgb):
        return self.net(torch.cat([condition, rgb], 1))


class FeatureLoss(nn.Module):
    """Frozen ImageNet ResNet features; this training loss is distinct from LPIPS."""

    def __init__(self, weights):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.eval().requires_grad_(False)

    def features(self, rgb):
        x = (rgb - rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / rgb.new_tensor(
            [0.229, 0.224, 0.225]
        )[None, :, None, None]
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        values = []
        for layer in [m.layer1, m.layer2, m.layer3]:
            x = layer(x)
            values.append(x)
        return values

    def forward(self, prediction, target):
        with torch.no_grad():
            target_features = self.features(target)
        return sum(F.l1_loss(a, b) for a, b in zip(self.features(prediction), target_features)) / 3


def alpha_schedule(steps=1000):
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    curve = torch.cos((x + 0.008) / 1.008 * math.pi / 2).square()
    beta = (1 - curve[1:] / curve[:-1]).clamp(1e-5, 0.999)
    return (1 - beta).cumprod(0).float()


class ConditionalDiffusion(nn.Module):
    def __init__(self, task, width=48):
        super().__init__()
        self.channels = 2 if task == "colorization" else 3
        # SR targets are 2*(RGB-bicubic); their support is wider than [-1,1].
        self.field_bound = 1.0 if task == "colorization" else 3.0
        self.model = ConditionalUNet(
            self.channels + (1 if task == "colorization" else 3), self.channels, width
        )
        self.time = nn.Sequential(nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 192))
        self.register_buffer("alpha", alpha_schedule())

    def forward(self, noisy, condition, timestep):
        frequencies = torch.exp(-math.log(10000) * torch.arange(96, device=noisy.device) / 95)
        phase = timestep[:, None].float() * frequencies[None]
        emb = self.time(torch.cat([phase.cos(), phase.sin()], 1))
        return self.model(torch.cat([noisy, condition], 1), emb)

    def objective(self, target, condition, generator=None):
        t = torch.randint(
            len(self.alpha), (len(target),), device=target.device, generator=generator
        )
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        a = self.alpha[t][:, None, None, None]
        noisy = a.sqrt() * target + (1 - a).sqrt() * noise
        # v prediction is stable at the near-zero-SNR end of the cosine schedule.
        velocity = a.sqrt() * noise - (1 - a).sqrt() * target
        return F.mse_loss(self(noisy, condition, t), velocity)

    @torch.no_grad()
    def sample(self, condition, steps=50, generator=None):
        x = torch.randn(
            (len(condition), self.channels, *condition.shape[-2:]),
            device=condition.device,
            generator=generator,
        )
        times = torch.linspace(len(self.alpha) - 1, 0, steps).round().long().tolist()
        for i, t in enumerate(times):
            a = self.alpha[t]
            v = self(x, condition, torch.full((len(x),), t, device=x.device, dtype=torch.long))
            clean = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-self.field_bound, self.field_bound)
            eps = (1 - a).sqrt() * x + a.sqrt() * v
            previous = self.alpha[times[i + 1]] if i + 1 < len(times) else x.new_tensor(1.0)
            x = previous.sqrt() * clean + (1 - previous).sqrt() * eps
        return x


def build_model(method, task, config, initialize_pretrained=True):
    weights = config.get("pretrained_weights") if initialize_pretrained else None
    if method == "cvae":
        semantic = task == "colorization" and bool(
            config.get("pretrained_weights") or config.get("semantic_decoder")
        )
        model = ConditionalVAE(task, config["width"], config["latent"], weights, semantic)
        if model.semantic is not None and config.get("mean_checkpoint"):
            if initialize_pretrained:
                state = torch.load(config["mean_checkpoint"], map_location="cpu", weights_only=True)
                model.semantic.load_state_dict(
                    {k.removeprefix("model."): v for k, v in state["state_dict"].items()},
                    strict=True,
                )
            model.frozen_semantic = True
            model.semantic.eval().requires_grad_(False)
        return model
    if method == "diffusion":
        return ConditionalDiffusion(task, config["width"])
    return Generator(task, config["width"], weights)


def save_checkpoint(value, path):
    """Publish a checkpoint atomically so interrupted writes preserve the last epoch."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    temporary.replace(path)


def validate_config(config):
    if config["task"] not in ["colorization", "super_resolution"]:
        raise ValueError("Unknown restoration task")
    methods, seeds = config["methods"], config["seeds"]
    if (
        not methods
        or len(set(methods)) != len(methods)
        or set(methods) - {"baseline", "l1", "gan", "cvae", "diffusion"}
    ):
        raise ValueError("Specify distinct supported methods")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("Specify distinct nonnegative integer training seeds")
    for key in [
        "epochs",
        "train_images",
        "validation_images",
        "test_images",
        "batch_size",
        "evaluation_batch_size",
        "latent",
        "samples",
    ]:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Positive integer {key} required")
    if config["test_images"] < 6:
        raise ValueError("At least six test images required for the fixed attributed gallery")
    if config["width"] < 8 or config["width"] % 8 or not 2 <= config["sampling_steps"] <= 1000:
        raise ValueError("Width divisible by eight and 2..1000 sampling steps required")
    if not 0 <= config["ema_decay"] < 1:
        raise ValueError("EMA decay must lie in [0,1)")
    for key in ["learning_rate", "feature_weight", "kl_weight", "adversarial_weight"]:
        if (
            not math.isfinite(config[key])
            or config[key] < 0
            or (key == "learning_rate" and config[key] == 0)
        ):
            raise ValueError(f"Invalid {key}")


def train_model(config, method, seed, run):
    device, task = torch.device(config["device"]), config["task"]
    if not (Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth").exists():
        raise FileNotFoundError(
            "LPIPS AlexNet weights missing; run prepare_external_restoration.py explicitly"
        )
    torch.manual_seed(seed)
    loader = DataLoader(
        Images(config["data_root"], "train", config["train_images"], config["crop"]),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("workers", 4 if device.type == "cuda" else 0),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    validation = DataLoader(
        Images(config["data_root"], "validation", config["validation_images"]), batch_size=8
    )
    model = build_model(method, task, config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    feature = FeatureLoss(config["pretrained_weights"]).to(device)
    discriminator = (
        PatchDiscriminator(4 if task == "colorization" else 6).to(device)
        if method == "gan"
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999), weight_decay=1e-4
    )
    dopt = (
        torch.optim.Adam(
            discriminator.parameters(), lr=config["learning_rate"] / 2, betas=(0.5, 0.999)
        )
        if discriminator
        else None
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    history, best, best_state, best_epoch = [], float("inf"), None, None
    checkpoint_dir = run.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    resume_path = checkpoint_dir / f"{method}_{seed}_training.pt"
    start_epoch = 0
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        assert saved["config"] == config, "Resume configuration differs from the original run"
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if discriminator:
            discriminator.load_state_dict(saved["discriminator"])
            dopt.load_state_dict(saved["dopt"])
        history, best, best_state, best_epoch = (
            saved["history"],
            saved["best"],
            saved["best_state"],
            saved["best_epoch"],
        )
        start_epoch = len(history)
        torch.set_rng_state(saved["rng"])
        loader.generator.set_state(saved["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.monotonic()
        model.train()
        totals, n = {}, 0
        for rgb in loader:
            rgb = rgb.to(device, non_blocking=True)
            condition, target = observation(rgb, task)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                if method == "diffusion":
                    loss = model.objective(target, condition)
                    terms = {"velocity_mse": loss}
                else:
                    field, kl = (
                        model(condition, target)
                        if method == "cvae"
                        else (model(condition), rgb.new_zeros(()))
                    )
                    prediction = reconstruct(condition, field, task)
                    pixel = F.l1_loss(field, target)
                    perceptual = feature(prediction, rgb)
                    beta = config["kl_weight"] * min(
                        1, (epoch + 1) / max(1, config["epochs"] * 0.3)
                    )
                    loss = pixel + config["feature_weight"] * perceptual + beta * kl
                    terms = {"pixel_l1": pixel, "feature_l1": perceptual, "kl": kl}
                    if discriminator:
                        discriminator.requires_grad_(True).train()
                        dopt.zero_grad(set_to_none=True)
                        real, fake = (
                            discriminator(condition, rgb),
                            discriminator(condition, prediction.detach()),
                        )
                        dloss = (F.relu(1 - real).mean() + F.relu(1 + fake).mean()) / 2
                        dloss.backward()
                        dopt.step()
                        discriminator.requires_grad_(False).eval()
                        adversarial = -discriminator(condition, prediction).mean()
                        loss = loss + config["adversarial_weight"] * adversarial
                        terms.update(discriminator_loss=dloss, generator_adversarial=adversarial)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite generator objective")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.parameters()):
                    a.lerp_(b, 1 - config["ema_decay"])
                for a, b in zip(ema.buffers(), model.buffers()):
                    a.copy_(b)
            for name, value in {"objective": loss, **terms}.items():
                totals[name] = totals.get(name, 0) + float(value.detach()) * len(rgb)
            n += len(rgb)
        ema.eval()
        rng = torch.Generator(device=device).manual_seed(812)
        values = []
        with torch.no_grad():
            for rgb in validation:
                rgb = rgb.to(device)
                condition, target = observation(rgb, task)
                if method == "diffusion":
                    value = ema.objective(target, condition, rng)
                else:
                    field = ema(condition, generator=rng)[0] if method == "cvae" else ema(condition)
                    value = F.l1_loss(field, target) + config["feature_weight"] * feature(
                        reconstruct(condition, field, task), rgb
                    )
                values.append(float(value) * len(rgb))
        validation_loss = sum(values) / config["validation_images"]
        row = {
            "epoch": epoch + 1,
            "train": {k: v / n for k, v in totals.items()},
            "validation_loss": validation_loss,
            "elapsed_seconds": time.monotonic() - epoch_start,
        }
        history.append(row)
        if validation_loss < best:
            best, best_epoch = validation_loss, epoch + 1
            best_state = copy.deepcopy({k: v.cpu() for k, v in ema.state_dict().items()})
        scheduler.step()
        save_checkpoint(
            {
                "config": config,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "discriminator": discriminator.state_dict() if discriminator else None,
                "dopt": dopt.state_dict() if dopt else None,
                "history": history,
                "best": best,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "rng": torch.get_rng_state(),
                "loader_rng": loader.generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
            resume_path,
        )
        write_json(run.output / f"{method}_{seed}_history.json", history)
        print(
            task,
            method,
            seed,
            epoch + 1,
            "train",
            totals["objective"] / n,
            "validation",
            validation_loss,
            "seconds",
            row["elapsed_seconds"],
            flush=True,
        )
    ema.load_state_dict(best_state)
    path = run.output / "checkpoints"
    path.mkdir(exist_ok=True)
    save_checkpoint(
        {
            "state_dict": best_state,
            "config": config,
            "method": method,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        path / f"{method}_{seed}.pt",
    )
    return ema, {
        "method": method,
        "seed": seed,
        "history": history,
        "best_epoch": best_epoch,
        "selection_objective": best,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def evaluate(model, method, seed, config):
    import lpips

    device, task = torch.device(config["device"]), config["task"]
    if not (Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth").exists():
        raise FileNotFoundError(
            "LPIPS AlexNet weights missing; run prepare_external_restoration.py explicitly"
        )
    metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    loader = DataLoader(
        Images(config["data_root"], "test", config["test_images"]),
        batch_size=config["evaluation_batch_size"],
    )
    samples = config["samples"] if method in ["cvae", "diffusion", "bigcolor"] else 1
    rows, arrays, gallery = [], {}, {}
    gallery_indices = [0, 1, 3, 5]
    first_draw = []
    for sample in range(samples):
        rng = torch.Generator(device=device).manual_seed(7000 + seed * 100 + sample)
        if method == "bigcolor":
            model.generator = rng
        scores = {"psnr": [], "ssim": [], "lpips": [], "data_consistency_l1": []}
        if task == "colorization":
            scores.update(delta_e76=[], chroma=[])
        if samples > 1:
            scores["sample_rgb_l1_from_first"] = []
        offset, pictures = 0, []
        start = time.monotonic()
        for rgb in loader:
            rgb = rgb.to(device)
            condition, target = observation(rgb, task)
            if method == "diffusion":
                field = model.sample(condition, config["sampling_steps"], rng)
            elif method == "cvae":
                field = model(condition, generator=rng)[0]
            elif method == "baseline":
                field = torch.zeros_like(target)
            elif method in ["swinir", "realesrgan"]:
                low = F.interpolate(
                    rgb, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True
                )
                field = 2 * (model.super_resolve(low) - condition)
            else:
                field = model(condition)
            if not torch.isfinite(field).all():
                raise FloatingPointError("Model produced a nonfinite restoration field")
            prediction = reconstruct(condition, field, task)
            if samples > 1:
                if sample == 0:
                    first_draw.append(prediction.cpu())
                    difference = prediction.new_zeros(len(rgb))
                else:
                    reference = first_draw[offset // config["evaluation_batch_size"]].to(device)
                    difference = (prediction - reference).abs().mean((1, 2, 3))
                scores["sample_rgb_l1_from_first"].extend(difference.cpu().tolist())
            fidelity = per_image_metrics(rgb, prediction, 4 if task == "super_resolution" else 0)
            for k, v in fidelity.items():
                scores[k].extend(v.tolist())
            scores["lpips"].extend(metric(rgb * 2 - 1, prediction * 2 - 1).flatten().cpu().tolist())
            if task == "colorization":
                lab = rgb_to_lab(prediction)
                scores["delta_e76"].extend(
                    ((lab - rgb_to_lab(rgb)) * lab.new_tensor([100, 110, 110])[None, :, None, None])
                    .square()
                    .sum(1)
                    .sqrt()
                    .mean((1, 2))
                    .cpu()
                    .tolist()
                )
                scores["chroma"].extend(
                    (lab[:, 1:] * 110).square().sum(1).sqrt().mean((1, 2)).cpu().tolist()
                )
                consistency = (lab[:, :1] - condition).abs().mean((1, 2, 3))
            else:
                a = F.interpolate(
                    rgb, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True
                )
                b = F.interpolate(
                    prediction,
                    scale_factor=0.25,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                consistency = (a - b).abs().mean((1, 2, 3))
            scores["data_consistency_l1"].extend(consistency.cpu().tolist())
            for i in range(len(rgb)):
                if offset + i in gallery_indices:
                    pictures.append(prediction[i].cpu().numpy())
                    if sample == 0:
                        gallery.setdefault("original", []).append(rgb[i].cpu().numpy())
                        gallery.setdefault("input", []).append(
                            reconstruct(
                                condition[i : i + 1], torch.zeros_like(target[i : i + 1]), task
                            )[0]
                            .cpu()
                            .numpy()
                        )
            offset += len(rgb)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        key = f"{method}_{seed}_sample{sample}"
        row = {
            "method": method,
            "metric_protocol_version": 3,
            "seed": seed,
            "sample": sample,
            "inference_seconds_including_metrics": elapsed,
            **{k: summarize(v) for k, v in scores.items()},
        }
        rows.append(row)
        arrays.update({key + "_" + k: np.array(v) for k, v in scores.items()})
        gallery[key] = np.stack(pictures)
    gallery["original"], gallery["input"] = (
        np.stack(gallery["original"]),
        np.stack(gallery["input"]),
    )
    return rows, arrays, gallery


def save_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def experiment(config):
    validate_config(config)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(Path(config["data_root"]) / "models" / "torch" / "hub"))
    output = Path(config["output"])
    previous = (
        json.loads((output / "run.json").read_text()) if (output / "run.json").exists() else None
    )
    evaluation_only = config.get("evaluate_only", False)
    if evaluation_only:
        if previous is None or previous.get("status") != "complete":
            raise ValueError("Evaluation-only mode requires a completed fitting record")
        clean = {k: v for k, v in config.items() if k not in ["evaluate_only", "device"]}
        original = {k: v for k, v in previous["config"].items() if k != "device"}
        if clean != original:
            raise ValueError("Evaluation-only mode must retain the original experiment protocol")
        preserved = json.loads((output / "metrics.json").read_text())
        run = None
    else:
        resuming = bool(list((output / "checkpoints").glob("*_training.pt")))
        if resuming and previous is not None and previous["config"] != config:
            raise ValueError("Resume configuration differs; the existing run record is preserved")
        run = Run(output, config)
        if resuming and previous is not None:
            run.record["prior_attempts"] = previous.get("prior_attempts", []) + [
                {k: v for k, v in previous.items() if k not in ["config", "prior_attempts"]}
            ]
            run.record["timing_scope"] = (
                "elapsed_seconds measures this invocation; complete epoch histories include prior fitting"
            )
            write_json(output / "run.json", run.record | {"status": "running"})
    scoring_source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checkpoint_hashes = {}
    if config.get("mean_checkpoint") and not config.get("evaluate_only"):
        run.record["mean_checkpoint_sha256"] = hashlib.sha256(
            Path(config["mean_checkpoint"]).read_bytes()
        ).hexdigest()
    rows, curves, scores, gallery = [], [], {}, {}
    for method in config["methods"]:
        for seed in config["seeds"] if method != "baseline" else [0]:
            if method == "baseline":
                model = None
            elif config.get("evaluate_only"):
                path = output / "checkpoints" / f"{method}_{seed}.pt"
                checkpoint_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                state = torch.load(path, map_location="cpu", weights_only=True)
                model = (
                    build_model(
                        method, config["task"], state["config"], initialize_pretrained=False
                    )
                    .to(config["device"])
                    .eval()
                )
                model.load_state_dict(state["state_dict"], strict=True)
            else:
                model, curve = train_model(config, method, seed, run)
                curves.append(curve)
            values, arrays, pictures = evaluate(model, method, seed, config)
            rows.extend(values)
            scores.update(arrays)
            gallery.update(pictures)
            if not evaluation_only:
                write_json(output / "partial_metrics.json", {"rows": rows, "curves": curves})
                save_arrays(output / "per_image_scores.npz", scores)
                save_arrays(output / "gallery.npz", gallery)
    manifest = json.loads(
        (Path(config["data_root"]) / "restoration256" / "manifest.json").read_text()
    )
    result = {
        "rows": rows,
        "curves": curves,
        "data_sha256": {k: v for k, v in manifest.items() if k.endswith("sha256")},
        "protocol": "256px COCO CC BY benchmark, fixed disjoint IDs; train-only random crops and flips. EMA checkpoints selected using validation only. CVAE test uses conditional prior, never target posterior. Diffusion uses cosine v prediction and deterministic DDIM trajectories from distinct initial noise. All samples reported, no best-of-test selection. RGB PSNR/SSIM (4px SR shave), LPIPS-Alex, observation consistency and color DeltaE/chroma. Bootstrap intervals are per-image, not training-seed uncertainty.",
    }
    if evaluation_only:
        save_arrays(output / "per_image_scores.npz", scores)
        save_arrays(output / "gallery.npz", gallery)
        write_json(
            output / "metrics.json", preserved | {k: v for k, v in result.items() if k != "curves"}
        )
        write_json(
            output / "evaluation_audit.json",
            {
                "mode": "evaluate_only",
                "device": config["device"],
                "metric_protocol_version": 3,
                "scoring_source_sha256": scoring_source,
                "checkpoint_sha256": checkpoint_hashes,
                "protocol": "Original fitting record, protocol and selected-epoch curves preserved; no initialization files required",
            },
        )
    else:
        run.finish(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    experiment(json.loads(args.config.read_text()))
