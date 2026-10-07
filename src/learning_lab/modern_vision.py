"""Fine-tune a modern pretrained encoder on the existing limited-label CIFAR protocol."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny

from .artifacts import Run, write_json
from .neural import seed_everything
from .vision import _measure, load_cifar


def input_images(x, augment=False, size=128):
    if augment:
        padded = F.pad(x, (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)
        start = torch.randint(9, (len(x), 2), device=x.device)
        rows = start[:, 0, None, None] + torch.arange(32, device=x.device)[None, :, None]
        cols = start[:, 1, None, None] + torch.arange(32, device=x.device)[None, None, :]
        x = padded[torch.arange(len(x), device=x.device)[:, None, None], rows, cols].permute(
            0, 3, 1, 2
        )
        x = torch.where(torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5, x.flip(-1), x)
    x = F.interpolate(
        x, size=(size, size), mode="bicubic", align_corners=False, antialias=True
    ).clamp(0, 1)
    return (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor(
        [0.229, 0.224, 0.225]
    )[None, :, None, None]


@torch.no_grad()
def logits(model, images, config):
    values = []
    for batch in images.split(config["batch_size"]):
        with torch.autocast(
            device_type=images.device.type,
            dtype=torch.bfloat16,
            enabled=images.device.type == "cuda",
        ):
            values.append(model(input_images(batch, size=config["input_size"])).float())
    return torch.cat(values).cpu().numpy()


def experiment(config):
    torch.set_num_threads(4)
    run = Run(config["output"], config)
    (x, y), (xt, yt) = load_cifar(config["image_root"])
    if config.get("quick"):
        x, y, xt, yt = x[:1500], y[:1500], xt[:500], yt[:500]
    indices = np.arange(len(y))
    fit, reserve = train_test_split(indices, test_size=0.1, random_state=17, stratify=y)
    val, cal = train_test_split(reserve, test_size=0.6, random_state=17, stratify=y[reserve])
    prob, conf = train_test_split(cal, test_size=0.5, random_state=17, stratify=y[cal])
    device = torch.device(config["device"])
    images = torch.tensor(x, device=device, dtype=torch.float32) / 255
    tests = torch.tensor(xt, device=device, dtype=torch.float32) / 255
    labels = torch.tensor(y, device=device)
    checkpoint = run.output / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    rows, curves = [], []
    for seed in config["seeds"]:
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        subset, _ = train_test_split(
            fit, train_size=config["label_budget"], random_state=seed, stratify=y[fit]
        )
        model = convnext_tiny(weights=None, stochastic_depth_prob=0.1)
        model.load_state_dict(
            torch.load(config["pretrained_weights"], map_location="cpu", weights_only=True),
            strict=True,
        )
        model.classifier[-1] = nn.Linear(768, 10)
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": config["learning_rate"]},
                {"params": model.classifier.parameters(), "lr": 5 * config["learning_rate"]},
            ],
            weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
        best, state, best_epoch, history = float("inf"), None, None, []
        for epoch in range(config["epochs"]):
            model.train()
            total = 0.0
            for idx in torch.tensor(subset, device=device)[
                torch.randperm(len(subset), device=device)
            ].split(config["batch_size"]):
                rgb = images[idx]
                lam = float(rng.beta(0.2, 0.2))
                partner = torch.randperm(len(idx), device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    inp = input_images(rgb, True, config["input_size"])
                    out = model(lam * inp + (1 - lam) * inp[partner])
                    loss = lam * F.cross_entropy(out, labels[idx], label_smoothing=0.1) + (
                        1 - lam
                    ) * F.cross_entropy(out, labels[idx][partner], label_smoothing=0.1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                total += float(loss.detach()) * len(idx)
            model.eval()
            v = logits(model, images[val], config)
            nll = float(F.cross_entropy(torch.from_numpy(v), torch.from_numpy(y[val])))
            history.append(
                {"epoch": epoch + 1, "train_loss": total / len(subset), "validation_nll": nll}
            )
            if nll < best:
                best, state, best_epoch = (
                    nll,
                    copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                    epoch + 1,
                )
            scheduler.step()
            write_json(run.output / f"history_{seed}.json", history)
            print("ConvNeXt", seed, epoch + 1, total / len(subset), nll, flush=True)
        model.load_state_dict(state)
        model.eval()
        ts, ps, cs = [logits(model, z, config) for z in [tests, images[prob], images[conf]]]
        row, p = _measure(ts, ps, cs, yt, y[prob], y[conf])
        noise = torch.randn(
            tests.shape, device=device, generator=torch.Generator(device=device).manual_seed(8000)
        )
        noisy = logits(model, (tests + 0.1 * noise).clamp(0, 1), config)
        row.update(
            model="convnext_tiny_finetuned",
            seed=seed,
            labels=len(subset),
            best_epoch=best_epoch,
            gaussian_noise_accuracy=float((noisy.argmax(1) == yt).mean()),
        )
        rows.append(row)
        curves.append({"seed": seed, "history": history, "best_epoch": best_epoch})
        torch.save(
            {"state_dict": state, "config": config, "seed": seed},
            checkpoint / f"convnext_{seed}.pt",
        )
        np.savez_compressed(run.output / f"predictions_{seed}.npz", p=p, y=yt)
        write_json(
            run.output / f"split_{seed}.json",
            {
                "fit": subset,
                "validation": val,
                "probability_calibration": prob,
                "conformal_calibration": conf,
            },
        )
        print("Final", seed, row["accuracy"], flush=True)
    run.finish(
        {
            "rows": rows,
            "curves": curves,
            "pretrained_sha256": hashlib.sha256(
                Path(config["pretrained_weights"]).read_bytes()
            ).hexdigest(),
            "protocol": "Official CIFAR10 test, same fixed fit/selection/probability/conformal splits and same 5000-label subsets as original vision study. ImageNet ConvNeXt-Tiny fine-tuning, ImageNet normalization, declared 128px upsampling, random native CIFAR crops/flips and mixup. Validation NLL checkpoints; temperature and conformal on separate partitions. Three fitting seeds; external exposure differs from scratch/contrastive encoders.",
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    experiment(json.loads(parser.parse_args().config.read_text()))
