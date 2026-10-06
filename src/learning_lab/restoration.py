"""Evaluated image restoration: denoising, super-resolution and uncertain colorization.

Small research baselines are implemented locally. They are inspired by residual
denoising/EDSR/color classification; they do not reproduce the original papers' budgets.
"""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from .artifacts import Run, write_json
from .project_studies import cache


def rgb_to_lab(rgb):
    """Differentiable D65 sRGB -> CIELAB, channels L/100, a/110, b/110."""
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).clamp_min(0).pow(2.4), rgb / 12.92)
    matrix = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = (
        torch.einsum("ij,njhw->nihw", matrix, linear)
        / rgb.new_tensor([0.95047, 1.0, 1.08883])[None, :, None, None]
    )
    delta = 6 / 29
    f = torch.where(xyz > delta**3, xyz.clamp_min(0).pow(1 / 3), xyz / (3 * delta**2) + 4 / 29)
    return torch.stack(
        [
            (116 * f[:, 1] - 16) / 100,
            500 * (f[:, 0] - f[:, 1]) / 110,
            200 * (f[:, 1] - f[:, 2]) / 110,
        ],
        1,
    )


def lab_to_rgb(lab):
    fy = (lab[:, 0] * 100 + 16) / 116
    f = torch.stack([fy + lab[:, 1] * 110 / 500, fy, fy - lab[:, 2] * 110 / 200], 1)
    delta = 6 / 29
    xyz = (
        torch.where(f > delta, f**3, 3 * delta**2 * (f - 4 / 29))
        * lab.new_tensor([0.95047, 1.0, 1.08883])[None, :, None, None]
    )
    matrix = lab.new_tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    linear = torch.einsum("ij,njhw->nihw", matrix, xyz)
    rgb = torch.where(
        linear > 0.0031308, 1.055 * linear.clamp_min(0).pow(1 / 2.4) - 0.055, 12.92 * linear
    )
    return rgb.clamp(0, 1)


def per_image_metrics(target, prediction, border=0):
    """RGB/grayscale PSNR and channel-mean SSIM with valid 7x7 Gaussian windows."""
    a, b = target.float(), prediction.float().clamp(0, 1)
    if border:
        a, b = a[..., border:-border, border:-border], b[..., border:-border, border:-border]
    mse = ((a - b) ** 2).mean((1, 2, 3))
    psnr = -10 * mse.clamp_min(1e-12).log10()
    grid = torch.arange(7, device=a.device, dtype=a.dtype) - 3
    g = torch.exp(-(grid**2) / (2 * 1.5**2))
    window = (g[:, None] * g[None, :]) / g.sum() ** 2
    weight = window.expand(a.shape[1], 1, 7, 7)

    def smooth(x):
        return F.conv2d(x, weight, groups=a.shape[1])

    ma, mb = smooth(a), smooth(b)
    va, vb = smooth(a * a) - ma * ma, smooth(b * b) - mb * mb
    cov = smooth(a * b) - ma * mb
    ssim = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / (
        (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
    )
    return {"psnr": psnr.cpu().numpy(), "ssim": ssim.mean((1, 2, 3)).cpu().numpy()}


def summarize(values):
    rng = np.random.default_rng(91)
    a = np.asarray(values)
    means = a[rng.integers(len(a), size=(1000, len(a)))].mean(1)
    return {
        "mean": float(a.mean()),
        "image_bootstrap_ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def measured(target, prediction, border=0):
    rows = {"psnr": [], "ssim": []}
    for i in range(0, len(target), 128):
        values = per_image_metrics(target[i : i + 128], prediction[i : i + 128], border)
        for k in rows:
            rows[k].extend(values[k].tolist())
    return {k: summarize(v) for k, v in rows.items()} | {"per_image_psnr": rows["psnr"]}


def pack_scores(run, rows):
    """Keep full paired-image evidence in lossless compressed arrays, not huge JSON."""
    arrays = {}
    for i, row in enumerate(rows):
        key = f"row_{i}"
        arrays[key] = np.asarray(row.pop("per_image_psnr"), dtype=np.float64)
        row["per_image_psnr_key"] = key
    np.savez_compressed(run.output / "per_image_scores.npz", **arrays)


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1),
        nn.ReLU(),
    )


class Autoencoder(nn.Module):
    def __init__(self, inputs=1, outputs=1, width=32):
        super().__init__()
        self.encoder1, self.encoder2 = block(inputs, width), block(width, 2 * width)
        self.middle = block(2 * width, 4 * width)
        self.decoder2, self.decoder1 = block(4 * width, 2 * width), block(2 * width, width)
        self.output = nn.Conv2d(width, outputs, 3, padding=1)

    def forward(self, x):
        h = self.encoder1(x)
        h = self.encoder2(F.avg_pool2d(h, 2))
        h = self.middle(F.avg_pool2d(h, 2))
        h = self.decoder2(F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False))
        h = self.decoder1(F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.output(h)


class ResidualDenoiser(nn.Module):
    """Eight convolution layers; noise map input and residual noise prediction."""

    def __init__(self, width=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, width, 3, padding=1),
            nn.ReLU(),
            *[
                layer
                for _ in range(6)
                for layer in (nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
            ],
            nn.Conv2d(width, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x[:, :1] - self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1)
        )

    def forward(self, x):
        return x + 0.1 * self.net(x)


class PixelShuffleSR(nn.Module):
    """Compact EDSR-inspired 4x network, no normalization, bicubic global skip."""

    def __init__(self, width=64, blocks=8):
        super().__init__()
        self.head = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(width) for _ in range(blocks)], nn.Conv2d(width, width, 3, padding=1)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(width, 4 * width, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(width, 4 * width, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x):
        h = self.head(x - 0.5)
        return F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False) + self.tail(
            h + self.body(h)
        )


class TransferColorizer(nn.Module):
    """ImageNet ResNet18 encoder with skip-connected convolution decoder."""

    def __init__(self, outputs=2, weights=None):
        super().__init__()
        self.encoder = resnet18(weights=None)
        if weights:
            self.encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.encoder.fc = nn.Identity()
        self.up3, self.up2, self.up1, self.up0 = (
            block(512 + 256, 128),
            block(128 + 128, 64),
            block(64 + 64, 32),
            block(32 + 64, 32),
        )
        self.output = nn.Sequential(block(32, 32), nn.Conv2d(32, outputs, 1))
        self.register_buffer(
            "normal_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer("normal_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, lightness):
        gray = lab_to_rgb(
            torch.cat([lightness, torch.zeros_like(lightness).expand(-1, 2, -1, -1)], 1)
        )
        h = (gray - self.normal_mean) / self.normal_std
        e = self.encoder
        a = e.relu(e.bn1(e.conv1(h)))
        b = e.layer1(e.maxpool(a))
        c, d = e.layer2(b), None
        d = e.layer3(c)
        h = e.layer4(d)
        for up, skip in [(self.up3, d), (self.up2, c), (self.up1, b), (self.up0, a)]:
            h = up(
                torch.cat(
                    [
                        F.interpolate(
                            h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        skip,
                    ],
                    1,
                )
            )
        return self.output(
            F.interpolate(h, size=lightness.shape[-2:], mode="bilinear", align_corners=False)
        )


def train(
    model, targets, validation, config, seed, make_input, loss, decode=None, learning_rate=None
):
    """Save best validation checkpoint and full curves; no access to test images."""
    device = targets.device
    model = model.to(device)
    generator = torch.Generator(device=device).manual_seed(1000 + seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate or config["learning_rate"], weight_decay=1e-4
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss, best_state, best_epoch, history = math.inf, None, None, []
    # Validation corruption is drawn exactly once with its own generator.
    val_generator = torch.Generator(device=device).manual_seed(818)
    val_input = make_input(validation, val_generator, False)
    for epoch in range(config["epochs"]):
        model.train()
        total = 0.0
        order = torch.randperm(len(targets), generator=generator, device=device)
        for idx in order.split(config["batch_size"]):
            target = targets[idx]
            if target.ndim == 4:
                flip = torch.rand(len(target), device=device, generator=generator) < 0.5
                target = torch.where(flip[:, None, None, None], target.flip(-1), target)
            inp = make_input(target, generator, True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16
            ):
                value = loss(model(inp), target)
            scaler.scale(value).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(value.detach()) * len(idx)
        model.eval()
        values = []
        with torch.no_grad():
            for i in range(0, len(validation), config["batch_size"]):
                pred = model(val_input[i : i + config["batch_size"]])
                values.append(
                    float(loss(pred, validation[i : i + config["batch_size"]])) * len(pred)
                )
        val_loss = sum(values) / len(validation)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(targets),
                "validation_loss": val_loss,
                "learning_rate": schedule.get_last_lr()[0],
            }
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        schedule.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                "epoch",
                epoch + 1,
                "train",
                total / len(targets),
                "validation",
                val_loss,
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": best_epoch, "best_validation_loss": best_loss}


@torch.no_grad()
def predict(model, inputs, batch_size=128):
    return torch.cat([model(x).float() for x in inputs.split(batch_size)])


def save_model(run, name, model, metadata):
    path = run.output / "checkpoints"
    path.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metadata": metadata,
        },
        path / f"{name}.pt",
    )
    write_json(
        run.output / f"{name}_model_card.json",
        metadata | {"parameters": sum(p.numel() for p in model.parameters())},
    )


def denoising(config):
    run = Run(config["output"], config)
    device = torch.device(config["device"])
    data = np.load(cache(config) / "mnist.npz")
    order = np.random.default_rng(42).permutation(len(data["train"]))
    if config["train_images"] + config["validation_images"] > len(order):
        raise ValueError("Training and validation image budgets would overlap")
    fit_ids, val_ids = order[: config["train_images"]], order[-config["validation_images"] :]
    clean_train = data["train"][fit_ids].astype(np.float32) / 255.0
    clean_val = data["train"][val_ids].astype(np.float32) / 255.0
    clean_test = data["test"][: config["test_images"]].astype(np.float32) / 255.0
    fit = torch.tensor(clean_train[:, None], device=device)
    val, test = (
        torch.tensor(clean_val[:, None], device=device),
        torch.tensor(clean_test[:, None], device=device),
    )
    noise_val = np.random.default_rng(818).normal(0, 30 / 255.0, clean_val.shape)
    pca = PCA(max(config["pca_ranks"]), svd_solver="randomized", random_state=0).fit(
        clean_train.reshape(len(fit), -1)
    )

    def pca_predict(images, rank, shrink):
        x = images.reshape(len(images), -1)
        v = pca.components_[:rank]
        coefficients = (x - pca.mean_) @ v.T
        if shrink:
            coefficients *= pca.explained_variance_[:rank] / (
                pca.explained_variance_[:rank] + shrink**2
            )
        return np.clip(pca.mean_ + coefficients @ v, 0, 1).reshape(images.shape)

    pca_selection = []
    for method in ["pca_projection", "pca_wiener"]:
        for rank in config["pca_ranks"]:
            prediction = pca_predict(
                clean_val + noise_val, rank, 30 / 255.0 if method.endswith("wiener") else 0
            )
            mse = np.mean((clean_val - prediction) ** 2)
            pca_selection.append({"model": method, "rank": rank, "validation_mse": mse})
    selected_ranks = {
        method: min(
            [r for r in pca_selection if r["model"] == method], key=lambda r: r["validation_mse"]
        )["rank"]
        for method in ["pca_projection", "pca_wiener"]
    }
    rows, curves = [], []
    observations = {}
    for noise in config["test_noise_levels"]:
        generator = torch.Generator(device=device).manual_seed(2000 + noise)
        observations[noise] = (
            test + torch.randn(test.shape, generator=generator, device=device) * noise / 255.0
        )
        rows.append({"model": "noisy", "sigma_255": noise, **measured(test, observations[noise])})
        for method, rank in selected_ranks.items():
            pred = pca_predict(
                observations[noise][:, 0].cpu().numpy(),
                rank,
                noise / 255.0 if method.endswith("wiener") else 0,
            )
            rows.append(
                {
                    "model": method,
                    "rank": rank,
                    "sigma_255": noise,
                    **measured(test, torch.tensor(pred[:, None], device=device)),
                }
            )
    gallery = {
        "clean": clean_test[:8],
        "observed": observations[30][:8, 0].clamp(0, 1).cpu().numpy(),
    }
    for seed in config["seeds"]:
        for name in ["autoencoder", "residual_denoiser"]:
            torch.manual_seed(seed)
            model = Autoencoder() if name == "autoencoder" else ResidualDenoiser()

            def inputs(target, generator, training):
                sigma = (
                    (
                        0.04
                        + 0.16
                        * torch.rand((len(target), 1, 1, 1), device=device, generator=generator)
                    )
                    if training
                    else torch.full((len(target), 1, 1, 1), 30 / 255.0, device=device)
                )
                noisy = (
                    target + torch.randn(target.shape, device=device, generator=generator) * sigma
                )
                return (
                    noisy
                    if name == "autoencoder"
                    else torch.cat([noisy, sigma.expand_as(noisy)], 1)
                )

            print("Training", name, seed, flush=True)
            model, curve = train(model, fit, val, config, seed, inputs, F.mse_loss)
            curves.append({"model": name, "seed": seed, **curve})
            save_model(
                run,
                f"{name}_{seed}",
                model,
                {
                    "task": "denoising",
                    "architecture": name,
                    "seed": seed,
                    "noise_range_255": [10.2, 51.0],
                    "best_epoch": curve["best_epoch"],
                },
            )
            for noise, observed in observations.items():
                inp = (
                    observed
                    if name == "autoencoder"
                    else torch.cat([observed, torch.full_like(observed, noise / 255.0)], 1)
                )
                pred = predict(model, inp).clamp(0, 1)
                rows.append(
                    {"model": name, "seed": seed, "sigma_255": noise, **measured(test, pred)}
                )
                if seed == config["seeds"][0] and noise == 30:
                    gallery[name] = pred[:8, 0].cpu().numpy()
            # Shift probe: Poisson plus Gaussian noise; the conditioning map gives only Gaussian sigma.
            gen = torch.Generator(device=device).manual_seed(33)
            shifted = (
                torch.poisson(test * 30, generator=gen) / 30
                + torch.randn(test.shape, device=device, generator=gen) * 15 / 255.0
            )
            inp = (
                shifted
                if name == "autoencoder"
                else torch.cat([shifted, torch.full_like(shifted, 15 / 255.0)], 1)
            )
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "shift": "poisson30_plus_gaussian15",
                    **measured(test, predict(model, inp)),
                }
            )
            del model
    np.savez_compressed(run.output / "gallery.npz", **gallery)
    np.savez_compressed(
        run.output / "pca.npz",
        mean=pca.mean_,
        components=pca.components_,
        variance=pca.explained_variance_,
    )
    write_json(
        run.output / "split.json",
        {"fit": fit_ids, "validation": val_ids, "test": list(range(len(test)))},
    )
    pack_scores(run, rows)
    run.finish(
        {
            "rows": rows,
            "curves": curves,
            "pca_selection": pca_selection,
            "selected_ranks": selected_ranks,
            "protocol": "Disjoint MNIST train/validation; official test. Noise generated in [0,1] units without clipping input. Checkpoints selected by validation MSE at sigma30. Bootstrap intervals describe test-image uncertainty, separate from training seeds.",
        }
    )


def natural_data(config):
    data = np.load(cache(config) / "coco96.npz")
    device = config["device"]

    def tensor(a):
        return (
            torch.tensor(a.transpose(0, 3, 1, 2).copy(), device=device, dtype=torch.float32) / 255.0
        )

    half = len(data["val"]) // 2
    if (
        config["train_images"] > len(data["train"])
        or config["validation_images"] > half
        or config["test_images"] > len(data["val"]) - half
    ):
        raise ValueError("Requested natural-image budget exceeds the prepared split")
    train = tensor(data["train"][: config["train_images"]])
    val = tensor(data["val"][:half][: config["validation_images"]])
    test = tensor(data["val"][half:][: config["test_images"]])
    return train, val, test


def degrade(images, shifted=False):
    if shifted:
        kernel = images.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        k = (kernel[:, None] * kernel[None, :]) / 256
        images = F.conv2d(
            F.pad(images, (2, 2, 2, 2), mode="reflect"), k.expand(3, 1, 5, 5), groups=3
        )
    return F.interpolate(
        images, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True
    )


def super_resolution(config):
    run = Run(config["output"], config)
    fit, val, test = natural_data(config)
    rows, curves = [], []
    gallery = {"clean": test[:8].cpu().numpy()}
    for shift in [False, True]:
        low = degrade(test, shift)
        base = F.interpolate(low, scale_factor=4, mode="bicubic", align_corners=False)
        rows.append(
            {
                "model": "bicubic",
                "degradation": "blurred_bicubic" if shift else "bicubic",
                **measured(test, base, border=4),
            }
        )
        if not shift:
            gallery["bicubic"] = base[:8].clamp(0, 1).cpu().numpy()
    for seed in config["seeds"]:
        for name in ["autoencoder", "pixelshuffle_residual"]:
            torch.manual_seed(seed)
            model = Autoencoder(3, 3) if name == "autoencoder" else PixelShuffleSR()

            def inputs(target, generator, training):
                low = degrade(target)
                return (
                    F.interpolate(low, scale_factor=4, mode="bicubic", align_corners=False)
                    if name == "autoencoder"
                    else low
                )

            print("Training", name, seed, flush=True)
            model, curve = train(model, fit, val, config, seed, inputs, F.l1_loss)
            curves.append({"model": name, "seed": seed, **curve})
            save_model(
                run,
                f"{name}_{seed}",
                model,
                {
                    "task": "super_resolution",
                    "architecture": name,
                    "seed": seed,
                    "scale": 4,
                    "best_epoch": curve["best_epoch"],
                    "degradation": "PyTorch antialiased bicubic, align_corners=False",
                },
            )
            for shift in [False, True]:
                low = degrade(test, shift)
                inp = (
                    F.interpolate(low, scale_factor=4, mode="bicubic", align_corners=False)
                    if name == "autoencoder"
                    else low
                )
                pred = predict(model, inp).clamp(0, 1)
                rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "degradation": "blurred_bicubic" if shift else "bicubic",
                        **measured(test, pred, border=4),
                    }
                )
                if seed == config["seeds"][0] and not shift:
                    gallery[name] = pred[:8].cpu().numpy()
            del model
    np.savez_compressed(run.output / "gallery.npz", **gallery)
    pack_scores(run, rows)
    run.finish(
        {
            "rows": rows,
            "curves": curves,
            "protocol": "8000 CC BY COCO training center crops; separate 400 validation / 400 test official val2017 IDs. L1 checkpoint selection, RGB [0,1] PSNR/SSIM after 4px border shave. Shift adds 5x5 blur before downsampling. This is a 96px controlled benchmark, not DIV2K/SOTA evaluation.",
        }
    )


def colorization(config):
    run = Run(config["output"], config)
    fit_rgb, val_rgb, test_rgb = natural_data(config)

    def convert(data):
        return torch.cat([rgb_to_lab(x) for x in data.split(128)])

    fit, val, test = convert(fit_rgb), convert(val_rgb), convert(test_rgb)
    # Learn the palette and balancing weights from training pixels only.
    sample = fit[: min(1000, len(fit)), 1:].permute(0, 2, 3, 1).reshape(-1, 2)[::37].cpu().numpy()
    clusters = MiniBatchKMeans(config["color_bins"], random_state=0, n_init=3, batch_size=4096).fit(
        sample
    )
    centers = torch.tensor(clusters.cluster_centers_, device=fit.device)
    counts = np.bincount(clusters.predict(sample), minlength=len(centers))
    prior = counts / counts.sum()
    weights = 1 / (0.5 * prior + 0.5 / len(prior))
    weights /= np.dot(weights, prior)
    class_weight = torch.tensor(weights, device=fit.device, dtype=torch.float32)

    def targets(ab):
        flat = ab.permute(0, 2, 3, 1).reshape(-1, 2).float()
        distance = (flat**2).sum(1, keepdim=True) + (centers**2).sum(1)[None] - 2 * flat @ centers.T
        return distance.argmin(1).reshape(len(ab), ab.shape[2], ab.shape[3])

    def decode(logits, temperature):
        probabilities = F.softmax(logits.float() / temperature, 1)
        return torch.einsum("nkhw,kc->nchw", probabilities, centers)

    def color_metrics(ab):
        prediction = lab_to_rgb(torch.cat([test[:, :1], ab], 1))
        values = measured(test_rgb, prediction)
        de = (((ab - test[:, 1:]) * 110) ** 2).sum(1).sqrt().mean((1, 2)).cpu().numpy()
        chroma = (ab**2).sum(1).sqrt().mean((1, 2)).cpu().numpy() * 110
        values["delta_e76"] = summarize(de)
        values["mean_chroma"] = summarize(chroma)
        values["true_mean_chroma"] = float((test[:, 1:] ** 2).sum(1).sqrt().mean() * 110)
        return values, prediction

    rows, curves, selections = [], [], []
    neutral_values, neutral = color_metrics(torch.zeros_like(test[:, 1:]))
    rows.append({"model": "neutral_gray", **neutral_values})
    gallery = {"clean": test_rgb[:8].cpu().numpy(), "neutral_gray": neutral[:8].cpu().numpy()}
    weights_path = config["pretrained_weights"]
    if not weights_path and not config.get("quick"):
        raise ValueError("Reference transfer study requires explicit ImageNet weights")
    if (
        weights_path
        and hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()[:8] != "f37072fd"
    ):
        raise ValueError("Unexpected pretrained ResNet18 weight checksum")
    for seed in config["seeds"]:
        for name in ["autoencoder_l2", "transfer_l2", "transfer_balanced_bins"]:
            torch.manual_seed(seed)
            model = (
                Autoencoder(1, 2)
                if name == "autoencoder_l2"
                else TransferColorizer(
                    config["color_bins"] if name.endswith("bins") else 2, weights=weights_path
                )
            )

            def loss(output, target):
                if name.endswith("bins"):
                    return F.cross_entropy(
                        output.float(), targets(target[:, 1:]), weight=class_weight
                    )
                return F.mse_loss(output, target[:, 1:])

            print("Training", name, seed, flush=True)
            model, curve = train(
                model,
                fit,
                val,
                config,
                seed,
                lambda t, g, training: t[:, :1],
                loss,
                learning_rate=config["learning_rate"]
                if name == "autoencoder_l2"
                else config["transfer_learning_rate"],
            )
            curves.append({"model": name, "seed": seed, **curve})
            temperature = 1.0
            if name.endswith("bins"):
                logits = predict(model, val[:, :1])
                candidates = []
                for t in config["temperatures"]:
                    ab = decode(logits, t)
                    de = (((ab - val[:, 1:]) * 110) ** 2).sum(1).sqrt().mean().item()
                    candidates.append({"temperature": t, "validation_delta_e76": de})
                temperature = min(candidates, key=lambda r: r["validation_delta_e76"])[
                    "temperature"
                ]
                selections.append({"seed": seed, "candidates": candidates, "selected": temperature})
            output = predict(model, test[:, :1])
            ab = decode(output, temperature) if name.endswith("bins") else output
            values, prediction = color_metrics(ab)
            row = {"model": name, "seed": seed, "temperature": temperature, **values}
            if name.endswith("bins"):
                probabilities = F.softmax(output / temperature, 1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
                row["mean_palette_entropy_nats"] = float(entropy.mean())
                if seed == config["seeds"][0]:
                    gallery["entropy"] = entropy[:8].cpu().numpy()
            rows.append(row)
            save_model(
                run,
                f"{name}_{seed}",
                model,
                {
                    "task": "colorization",
                    "architecture": name,
                    "seed": seed,
                    "best_epoch": curve["best_epoch"],
                    "temperature": temperature,
                    "centers": centers.cpu().tolist(),
                    "external_pretraining": name != "autoencoder_l2" and bool(weights_path),
                    "pretrained_weight_sha256": hashlib.sha256(
                        Path(weights_path).read_bytes()
                    ).hexdigest()
                    if name != "autoencoder_l2" and weights_path
                    else None,
                },
            )
            if seed == config["seeds"][0]:
                gallery[name] = prediction[:8].cpu().numpy()
            del model
    np.savez_compressed(run.output / "gallery.npz", **gallery)
    pack_scores(run, rows)
    run.finish(
        {
            "rows": rows,
            "curves": curves,
            "temperature_selection": selections,
            "palette_centers": centers.cpu().numpy(),
            "palette_frequency": prior,
            "protocol": "Same disjoint CC BY COCO image IDs as SR. Grayscale input is CIELAB L; two chroma channels predicted. L2 vs balanced 32-bin CE; palette/weights training-only; temperature validation-only. DeltaE76 uses target L, before RGB gamut clipping. Exact original colors cannot be identified from grayscale; no human preference or perceptual-quality claim.",
        }
    )
