"""Gray-coded 16-QAM through a controlled nonlinear channel with circular memory.

This is a discrete baseband benchmark, not an optical-fiber propagation simulator.
"""

import copy

import numpy as np
import torch
from torch import nn

from learning_lab.artifacts import Run, write_json
from learning_lab.neural import seed_everything


def constellation():
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    points = (levels[:, None] + 1j * levels[None, :]).reshape(-1)
    gray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    bits = np.concatenate([np.repeat(gray, 4, axis=0), np.tile(gray, (4, 1))], axis=1)
    return points, bits


def channel(symbols, snr_db, gamma, rng):
    # Unit-energy FIR: power at its output is 1 for independent unit-power symbols.
    taps = np.array([1, 0.35 + 0.15j, 0.15 - 0.1j])
    taps /= np.linalg.norm(taps)
    filtered = sum(h * np.roll(symbols, j, axis=1) for j, h in enumerate(taps))
    noiseless = filtered * (1 + gamma * np.abs(filtered) ** 2)
    # SNR is Es/N0 at the normalized FIR input, not a post-nonlinearity measured SNR.
    noise = (rng.normal(size=symbols.shape) + 1j * rng.normal(size=symbols.shape)) * np.sqrt(
        10 ** (-snr_db / 10) / 2
    )
    return noiseless + noise


def generate(blocks, length, snr, gamma, seed):
    rng = np.random.default_rng(seed)
    points, _ = constellation()
    labels = rng.integers(16, size=(blocks, length))
    symbols = points[labels]
    received = channel(symbols, snr, gamma, rng)
    features = np.stack([received.real, received.imag], axis=1).astype(np.float32)
    return features, labels, symbols


def physics_inverse(received, gamma):
    radius = np.abs(received)
    amp = radius.copy()
    for _ in range(20):
        amp -= (amp + gamma * amp**3 - radius) / (1 + 3 * gamma * amp**2)
    linear = received * np.divide(amp, radius, out=np.ones_like(amp), where=radius > 0)
    taps = np.array([1, 0.35 + 0.15j, 0.15 - 0.1j])
    taps /= np.linalg.norm(taps)
    response = np.fft.fft(taps, n=received.shape[1])
    return np.fft.ifft(np.fft.fft(linear, axis=1) / response, axis=1)


def windows(x, radius=3):
    # Independent blocks are split before constructing overlapping windows.
    return np.concatenate(
        [np.roll(x, j, axis=2) for j in range(-radius, radius + 1)], axis=1
    ).transpose(0, 2, 1)


class Equalizer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "cnn":
            self.net = nn.Sequential(
                nn.Conv1d(2, 64, 7, padding=3, padding_mode="circular"),
                nn.GELU(),
                nn.Conv1d(64, 64, 5, padding=2, padding_mode="circular"),
                nn.GELU(),
                nn.Conv1d(64, 16, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(14, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 16)
            )

    def forward(self, x):
        if self.kind == "cnn":
            return self.net(x).transpose(1, 2)
        context = torch.cat([torch.roll(x, j, dims=2) for j in range(-3, 4)], dim=1).transpose(1, 2)
        return self.net(context)


def error_metrics(truth, pred):
    _, bits = constellation()
    # Treat a block, rather than an overlapping symbol window, as the bootstrap unit.
    per_block = np.mean(bits[truth] != bits[pred], axis=(1, 2))
    rng = np.random.default_rng(0)
    values = [rng.choice(per_block, len(per_block), replace=True).mean() for _ in range(500)]
    return {
        "ber": float(per_block.mean()),
        "ser": float(np.mean(truth != pred)),
        "ber_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "blocks": len(truth),
        "bits": int(truth.size * 4),
    }


def experiment(config):
    run = Run(config["output"], config)
    points, _ = constellation()
    rows = []
    for seed in config["seeds"]:
        seed_everything(seed)
        xf, yf, sf = generate(config["train_blocks"], 64, 18, 0.2, seed)
        xv, yv, _ = generate(500 if not config.get("quick") else 32, 64, 18, 0.2, 1000 + seed)
        # Ridge window equalizer predicts complex symbols from the same local information as the MLP.
        a = windows(xf).reshape(-1, 14)
        target = np.stack([sf.real, sf.imag], axis=-1).reshape(-1, 2)
        weights = np.linalg.solve(a.T @ a + 0.01 * np.eye(14), a.T @ target)
        models = {}
        for kind in ["mlp", "cnn"]:
            model = Equalizer(kind).to(config["device"])
            opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
            tx, ty = (
                torch.as_tensor(xf, device=config["device"]),
                torch.as_tensor(yf, device=config["device"]),
            )
            vx, vy = (
                torch.as_tensor(xv, device=config["device"]),
                torch.as_tensor(yv, device=config["device"]),
            )
            best, state, history = np.inf, None, []
            for epoch in range(config["epochs"]):
                model.train()
                losses = []
                for ix in torch.randperm(len(tx), device=config["device"]).split(64):
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(
                        model(tx[ix]).reshape(-1, 16), ty[ix].flatten()
                    )
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():
                    val = float(
                        nn.functional.cross_entropy(model(vx).reshape(-1, 16), vy.flatten())
                    )
                history.append(
                    {"epoch": epoch + 1, "train_loss": np.mean(losses), "validation_loss": val}
                )
                if val < best:
                    best, state = val, copy.deepcopy(model.state_dict())
            model.load_state_dict(state)
            model.eval()
            models[kind] = model
            write_json(run.output / f"history_{kind}_{seed}.json", history)
            checkpoint = run.output / "checkpoints" / f"{kind}_{seed}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            torch.save({"state_dict": state, "kind": kind, "seed": seed}, checkpoint)
        for gamma in [0.2, 0.4]:
            for snr in [10, 14, 18, 22, 26]:
                xt, yt, _ = generate(config["test_blocks"], 64, snr, gamma, 9000 + seed)
                r = xt[:, 0] + 1j * xt[:, 1]
                raw = np.argmin(abs(r[:, :, None] - points), axis=-1)
                oracle = np.argmin(abs(physics_inverse(r, gamma)[:, :, None] - points), axis=-1)
                linear = windows(xt) @ weights
                linear = linear[:, :, 0] + 1j * linear[:, :, 1]
                predictions = {
                    "nearest_raw": raw,
                    "ridge_window": np.argmin(abs(linear[:, :, None] - points), axis=-1),
                    "known_channel_inverse": oracle,
                }
                for kind, model in models.items():
                    with torch.no_grad():
                        predictions[kind] = torch.cat(
                            [
                                model(t).argmax(-1).cpu()
                                for t in torch.as_tensor(xt, device=config["device"]).split(256)
                            ]
                        ).numpy()
                for name, p in predictions.items():
                    rows.append(
                        {
                            "seed": seed,
                            "gamma": gamma,
                            "snr_db": snr,
                            "model": name,
                            **error_metrics(yt, p),
                        }
                    )
        print("equalization seed", seed, "complete", flush=True)
    run.finish(
        {
            "rows": rows,
            "training_snr_db": 18,
            "training_gamma": 0.2,
            "protocol": "independent blocks; fixed training channel; gamma=.4 is unadapted shift",
            "oracle_knowledge": "known-channel inverse knows test FIR and gamma; learned methods do not",
        }
    )
