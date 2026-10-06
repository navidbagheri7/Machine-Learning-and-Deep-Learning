"""CIFAR-10 supervision, contrastive representations, and pretrained transfer.

All augmentation is batched on the selected device. Test images never enter pretraining.
"""

import copy
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from learning_lab.artifacts import Run, write_json
from learning_lab.neural import seed_everything
from learning_lab.uncertainty import fit_temperature, prediction_sets

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.2435, 0.2616)


def load_cifar(root):
    root = Path(root) / "cifar-10-batches-py"

    def read(name):
        # Only load the trusted official CIFAR archive; pickle is not an untrusted upload format.
        with (root / name).open("rb") as f:
            data = pickle.load(f, encoding="bytes")
        return data[b"data"].reshape(-1, 3, 32, 32), np.array(data[b"labels"])

    batches = [read(f"data_batch_{i}") for i in range(1, 6)]
    return (np.concatenate([b[0] for b in batches]), np.concatenate([b[1] for b in batches])), read(
        "test_batch"
    )


def normalize(x):
    mean = x.new_tensor(MEAN)[None, :, None, None]
    std = x.new_tensor(STD)[None, :, None, None]
    return (x - mean) / std


def augment(x, contrastive=False):
    n = len(x)
    if contrastive:
        # Independent crops for each view, with random affine scale and translation.
        scale = torch.empty(n, device=x.device).uniform_(0.55, 1)
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 1, 1] = scale, scale
        theta[:, :, 2] = (torch.rand(n, 2, device=x.device) * 2 - 1) * (1 - scale[:, None])
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
        brightness = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.6, 1.4)
        contrast = torch.empty_like(brightness).uniform_(0.6, 1.4)
        saturation = torch.empty_like(brightness).uniform_(0.6, 1.4)
        x = x * brightness
        x = (x - x.mean((2, 3), keepdim=True)) * contrast + x.mean((2, 3), keepdim=True)
        gray = (x * x.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
        x = (x - gray) * saturation + gray
        x = torch.where((torch.rand(n, 1, 1, 1, device=x.device) < 0.2), gray, x).clamp(0, 1)
    else:
        padded = F.pad(x, (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)
        start = torch.randint(9, (n, 2), device=x.device)
        rows = start[:, 0, None, None] + torch.arange(32, device=x.device)[None, :, None]
        cols = start[:, 1, None, None] + torch.arange(32, device=x.device)[None, None, :]
        x = padded[torch.arange(n, device=x.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
    x = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < 0.5, x.flip(-1), x)
    return normalize(x)


class ImageBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outgoing),
            nn.ReLU(),
            nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outgoing),
        )
        self.shortcut = (
            nn.Identity()
            if incoming == outgoing and stride == 1
            else nn.Sequential(
                nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing)
            )
        )

    def forward(self, x):
        return F.relu(self.net(x) + self.shortcut(x))


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ImageBlock(32, 32),
            ImageBlock(32, 32),
            ImageBlock(32, 64, 2),
            ImageBlock(64, 64),
            ImageBlock(64, 128, 2),
            ImageBlock(128, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


def contrastive_loss(a, b, temperature=0.2):
    z = F.normalize(torch.cat([a, b]), dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    targets = (torch.arange(len(z), device=z.device) + len(a)) % len(z)
    return F.cross_entropy(logits, targets)


def train_encoder(images, labels, validation, validation_y, epochs, task, device, seed):
    seed_everything(seed)
    encoder = ImageEncoder().to(device)
    head = (
        nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
        if task == "contrastive"
        else nn.Linear(128, 10)
    ).to(device)
    parameters = [*encoder.parameters(), *head.parameters()]
    optimizer = (
        torch.optim.AdamW(parameters, lr=0.001, weight_decay=1e-4)
        if task == "contrastive"
        else torch.optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=5e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, state, history = np.inf, None, []
    for epoch in range(epochs):
        encoder.train()
        head.train()
        loss_sum, count = 0.0, 0
        for ix in torch.randperm(len(images), device=device).split(256):
            if len(ix) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                if task == "contrastive":
                    a, b = augment(images[ix], True), augment(images[ix], True)
                    z = head(encoder(torch.cat([a, b])))
                    loss = contrastive_loss(z[: len(a)].float(), z[len(a) :].float())
                else:
                    loss = F.cross_entropy(
                        head(encoder(augment(images[ix]))), labels[ix], label_smoothing=0.1
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(ix)
            count += len(ix)
        scheduler.step()
        encoder.eval()
        head.eval()
        if task == "supervised":
            logits = image_logits(encoder, head, validation)
            val = float(F.cross_entropy(logits, validation_y))
            if val < best:
                best, state = (
                    val,
                    (copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict())),
                )
        else:
            # Fixed-budget final checkpoint; no test-driven pretraining selection.
            val = None
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count, "validation_nll": val})
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(task, seed, "epoch", epoch + 1, "loss", round(loss_sum / count, 4), flush=True)
    if state:
        encoder.load_state_dict(state[0])
        head.load_state_dict(state[1])
    encoder.eval()
    head.eval()
    return encoder, head, history


def image_logits(encoder, head, images):
    with torch.no_grad():
        return torch.cat([head(encoder(normalize(b))).float() for b in images.split(512)])


def features(encoder, images, pretrained=False):
    out = []
    with torch.no_grad():
        for b in images.split(256):
            if pretrained:
                # Follow torchvision's pretrained resize/center-crop normalization.
                b = ResNet18_Weights.IMAGENET1K_V1.transforms()(b)
            else:
                b = normalize(b)
            out.append(encoder(b).float().cpu().numpy())
    return np.concatenate(out)


def _measure(logits, cal_logits, conf_logits, y, cal_y, conf_y):
    logits = np.asarray(logits, dtype=np.float64)
    cal_logits = np.asarray(cal_logits, dtype=np.float64)
    conf_logits = np.asarray(conf_logits, dtype=np.float64)
    temperature = fit_temperature(cal_logits, cal_y)
    raw = softmax(logits, axis=1)
    p, cp = softmax(logits / temperature, axis=1), softmax(conf_logits / temperature, axis=1)
    sets = prediction_sets(p, cp, conf_y)
    return {
        "accuracy": accuracy_score(y, p.argmax(1)),
        "raw_nll": log_loss(y, raw),
        "nll": log_loss(y, p),
        "temperature": temperature,
        "set_coverage": float(sets[np.arange(len(y)), y].mean()),
        "set_size": float(sets.sum(1).mean()),
        "per_class_accuracy": [float((p.argmax(1)[y == k] == k).mean()) for k in range(10)],
    }, p


def experiment(config):
    run = Run(config["output"], config)
    (x, y), (xt, yt) = load_cifar(config["image_root"])
    if config.get("quick"):
        x, y, xt, yt = x[:1500], y[:1500], xt[:500], yt[:500]
    ix = np.arange(len(y))
    fit, reserve = train_test_split(ix, test_size=0.1, random_state=17, stratify=y)
    val, cal = train_test_split(reserve, test_size=0.6, random_state=17, stratify=y[reserve])
    prob, conf = train_test_split(cal, test_size=0.5, random_state=17, stratify=y[cal])
    device = config["device"]
    images = torch.as_tensor(x, dtype=torch.float32, device=device) / 255
    tests = torch.as_tensor(xt, dtype=torch.float32, device=device) / 255
    labels = torch.as_tensor(y, device=device)
    rows = []
    seed_everything(0)
    random_encoder = ImageEncoder().to(device).eval()
    reps = {"random_encoder": (features(random_encoder, images), features(random_encoder, tests))}
    encoder, head, history = train_encoder(
        images[fit], None, images[val], labels[val], config["ssl_epochs"], "contrastive", device, 0
    )
    write_json(run.output / "history_contrastive.json", history)
    checkpoint = run.output / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    torch.save(encoder.state_dict(), checkpoint / "contrastive_encoder.pt")
    reps["contrastive_encoder"] = (features(encoder, images), features(encoder, tests))
    weights = Path(config["pretrained_weights"])
    if not weights.exists():
        raise FileNotFoundError(
            "Pretrained ResNet18 weights missing. Supply --pretrained-weights or prepare explicitly."
        )
    pretrained = resnet18(weights=None).to(device)
    pretrained.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    pretrained.fc = nn.Identity()
    pretrained.eval()
    reps["imagenet_resnet18"] = (
        features(pretrained, images, True),
        features(pretrained, tests, True),
    )
    for seed in config["seeds"]:
        if config["label_budget"] < len(fit):
            subset, _ = train_test_split(
                fit, train_size=config["label_budget"], random_state=seed, stratify=y[fit]
            )
        else:
            subset = fit
        for name, (fx, ft) in reps.items():
            # Freeze the encoder. Fit and tune only the linear head.
            candidates = []
            for c in [0.01, 0.1, 1]:
                model = LogisticRegression(C=c, max_iter=1000).fit(fx[subset], y[subset])
                candidates.append((log_loss(y[val], model.predict_proba(fx[val])), model, c))
            _, model, c = min(candidates, key=lambda z: z[0])
            result, p = _measure(
                model.decision_function(ft),
                model.decision_function(fx[prob]),
                model.decision_function(fx[conf]),
                yt,
                y[prob],
                y[conf],
            )
            rows.append(
                {"model": name, "seed": seed, "labels": len(subset), "selected_C": c, **result}
            )
            np.savez_compressed(run.output / f"predictions_{name}_{seed}.npz", p=p, y=yt)
            print("linear probe", name, seed, round(result["accuracy"], 4), flush=True)
        encoder, head, history = train_encoder(
            images[subset],
            labels[subset],
            images[val],
            labels[val],
            config["epochs"],
            "supervised",
            device,
            seed,
        )
        write_json(run.output / f"history_supervised_lowlabel_{seed}.json", history)
        logits = [
            image_logits(encoder, head, a).cpu().numpy()
            for a in [tests, images[prob], images[conf]]
        ]
        result, p = _measure(*logits, yt, y[prob], y[conf])
        rows.append({"model": "supervised_lowlabel", "seed": seed, "labels": len(subset), **result})
        torch.save(
            {"encoder": encoder.state_dict(), "head": head.state_dict()},
            checkpoint / f"supervised_{seed}.pt",
        )
        np.savez_compressed(run.output / f"predictions_supervised_lowlabel_{seed}.npz", p=p, y=yt)
        # Corruption evaluation uses the same frozen model and temperature, without adaptation.
        with torch.no_grad():
            torch.manual_seed(8000)
            corrupted = (tests + 0.1 * torch.randn_like(tests)).clamp(0, 1)
            pnoise = softmax(
                image_logits(encoder, head, corrupted).cpu().numpy() / result["temperature"], axis=1
            )
        rows[-1]["gaussian_noise_accuracy"] = accuracy_score(yt, pnoise.argmax(1))
    encoder, head, history = train_encoder(
        images[fit],
        labels[fit],
        images[val],
        labels[val],
        config["full_epochs"],
        "supervised",
        device,
        0,
    )
    write_json(run.output / "history_supervised_full.json", history)
    logits = [
        image_logits(encoder, head, a).cpu().numpy() for a in [tests, images[prob], images[conf]]
    ]
    result, p = _measure(*logits, yt, y[prob], y[conf])
    rows.append({"model": "supervised_full", "seed": 0, "labels": len(fit), **result})
    torch.save(
        {"encoder": encoder.state_dict(), "head": head.state_dict()},
        checkpoint / "supervised_full.pt",
    )
    np.savez_compressed(run.output / "predictions_supervised_full_0.npz", p=p, y=yt)
    # Deterministic first errors, with labels and confidence, for the error-analysis gallery.
    errors = np.flatnonzero(p.argmax(1) != yt)[:32]
    np.savez_compressed(
        run.output / "errors.npz",
        images=xt[errors],
        truth=yt[errors],
        pred=p.argmax(1)[errors],
        confidence=p.max(1)[errors],
    )
    write_json(
        run.output / "split.json",
        {
            "fit": fit,
            "validation": val,
            "probability_calibration": prob,
            "conformal_calibration": conf,
        },
    )
    run.finish(
        {
            "rows": rows,
            "test_n": len(yt),
            "pretrained_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "exposure": "contrastive: fit images without labels; transfer: external ImageNet pretraining",
            "protocol": "official CIFAR test; fixed fit/selection/probability/conformal splits; low-label subsets vary by seed",
        }
    )
