"""Adapters for pinned, externally trained restoration references.

Official source remains in the personal cache. Every reference verifies its source
revision and checkpoint hash before inference. Each adapter runs in its own process,
avoiding collisions between upstream packages named models/utils.
"""

import argparse
import collections
import functools
import importlib.util
import io
import json
import pickle
import subprocess
import sys
import types
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from .artifacts import Run
from .generative import evaluate
from .pretrained import colorize_ddcolor, load_ddcolor
from .restoration import lab_to_rgb, rgb_to_lab


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("argparse", "Namespace"): argparse.Namespace,
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch.nn.modules.activation", "ReLU"): nn.ReLU,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported metadata global: {module}.{name}")
        return allowed[module, name]


def metadata(path):
    return MetadataReader(io.BytesIO(Path(path).read_bytes())).load()


def verified_source(config, name):
    from hashlib import sha256

    root = Path(config["data_root"]) / "models"
    cached = json.loads((root / "restoration_manifest.json").read_text())
    manifest = json.loads(
        (files("learning_lab") / "resources/restoration_manifest.json").read_text()
    )
    if cached["sources"][name]["revision"] != manifest["sources"][name]["revision"]:
        raise ValueError("Cached manifest differs from the published source pin")
    source = root / name
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != manifest["sources"][name]["revision"]:
        raise ValueError("External source revision mismatch")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD"], check=True)
    for relative, record in manifest["weights"].items():
        if relative.startswith(name + "/") and not relative.endswith(
            ("release.rar", "G_ema_256.pth")
        ):
            digest = sha256()
            with (root / relative).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"External weight checksum mismatch: {relative}")
    sys.path.insert(0, str(source))
    return source, manifest


class DDColorReference(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model, _ = load_ddcolor(config)

    def forward(self, condition):
        return colorize_ddcolor(self.model, condition, 512, 2)


class SwinIRReference(nn.Module):
    def __init__(self, config):
        super().__init__()
        source, _ = verified_source(config, "SwinIR")
        path = source / "models" / "network_swinir.py"
        spec = importlib.util.spec_from_file_location("official_swinir", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6] * 6,
            embed_dim=180,
            num_heads=[6] * 6,
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
        weight = source / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
        self.model.load_state_dict(
            torch.load(weight, map_location="cpu", weights_only=True)["params"], strict=True
        )

    def forward(self, condition):
        # condition is the benchmark's bicubic high-resolution reconstruction. Recover
        # the original LR observation without a second antialiased downsampling filter.
        # Calling with raw LR is available through super_resolve().
        raise RuntimeError(
            "SwinIR requires the actual low-resolution observation; use super_resolve"
        )

    def super_resolve(self, low):
        return self.model(low)


class RealESRGANReference(nn.Module):
    def __init__(self, config):
        super().__init__()
        source, _ = verified_source(config, "BasicSR")
        # Load only the pinned architecture, avoiding unrelated training packages.
        # Strict full checkpoint loading overwrites initialization. The forward
        # graph remains the official RRDBNet graph.
        for name, path in [
            ("basicsr", source / "basicsr"),
            ("basicsr.archs", source / "basicsr/archs"),
            ("basicsr.utils", source / "basicsr/utils"),
        ]:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        registry = types.ModuleType("basicsr.utils.registry")
        registry.ARCH_REGISTRY = types.SimpleNamespace(register=lambda: lambda cls: cls)
        sys.modules[registry.__name__] = registry
        helpers = types.ModuleType("basicsr.archs.arch_util")
        helpers.default_init_weights = lambda *args, **kwargs: None
        helpers.make_layer = lambda block, count, **kwargs: nn.Sequential(
            *[block(**kwargs) for _ in range(count)]
        )
        helpers.pixel_unshuffle = F.pixel_unshuffle
        sys.modules[helpers.__name__] = helpers
        spec = importlib.util.spec_from_file_location(
            "basicsr.archs.rrdbnet_arch", source / "basicsr/archs/rrdbnet_arch.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.RRDBNet(3, 3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
        state = torch.load(
            source / "weights/RealESRGAN_x4plus.pth", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state["params_ema"], strict=True)

    def super_resolve(self, low):
        return self.model(low)


class BigColorReference(nn.Module):
    def __init__(self, config):
        super().__init__()
        source, manifest = verified_source(config, "BigColor")
        from models.common import Colorizer

        args = metadata(source / "ckpts" / "bigcolor" / "args.pkl")
        settings = metadata(source / "pretrained" / "config.pickle")
        self.model = Colorizer(
            settings,
            None,
            args.norm_type,
            activation=args.activation,
            id_mid_layer=args.num_layer,
            use_attention=args.use_attention,
            dim_f=16,
            load_g=False,
        )
        weights = torch.load(
            source / "ckpts" / "bigcolor" / "EG_011.ckpt", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(weights, strict=True)
        ema = torch.load(
            source / "ckpts" / "bigcolor" / "EG_EMA_011.ckpt", map_location="cpu", weights_only=True
        )
        parameters = list(self.model.parameters())
        if len(parameters) != len(ema["shadow_params"]) or any(
            p.shape != v.shape for p, v in zip(parameters, ema["shadow_params"])
        ):
            raise ValueError("BigColor EMA parameter count or shape mismatch")
        for parameter, value in zip(parameters, ema["shadow_params"]):
            parameter.data.copy_(value)
        # Published inference allows classifier-dependent class selection. Use a fixed,
        # already cached official ResNet50 classifier, explicitly distinct from the
        # authors' default EfficientNet-L2. No original RGB or target labels enter it.
        self.classifier = resnet50(weights=None)
        from hashlib import sha256

        expected = manifest["weights"]["torch/hub/checkpoints/resnet50-11ad3fa6.pth"]["sha256"]
        if sha256(Path(config["classifier_weights"]).read_bytes()).hexdigest() != expected:
            raise ValueError("BigColor classifier differs from the published checkpoint")
        self.classifier.load_state_dict(
            torch.load(config["classifier_weights"], map_location="cpu", weights_only=True)
        )
        self.latent = args.dim_z
        self.seed = config.get("sampling_seed", 7000)
        self.generator = None

    def forward(self, condition):
        gray = lab_to_rgb(
            torch.cat([condition, torch.zeros_like(condition).expand(-1, 2, -1, -1)], 1)
        )
        classifier_input = F.interpolate(
            gray, size=(224, 224), mode="bilinear", align_corners=False
        )
        classifier_input = (
            classifier_input - gray.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        ) / gray.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        labels = self.classifier(classifier_input).argmax(1)
        if self.generator is None:
            self.generator = torch.Generator(device=condition.device).manual_seed(self.seed)
        z = (
            torch.randn(
                (len(condition), self.latent), device=condition.device, generator=self.generator
            )
            * 0.8
        )
        prediction = (self.model(gray[:, :1], labels, z) + 1) / 2
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("BigColor produced a nonfinite raw image")
        return rgb_to_lab(prediction.clamp(0, 1))[:, 1:]


def edge_operator(im, input_channel=2, output_channel=2):
    kernel = im.new_tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return F.conv2d(im, kernel.expand(output_channel, input_channel, 3, 3), padding=1)


class CT2Reference(nn.Module):
    def __init__(self, config):
        super().__init__()
        source, _ = verified_source(config, "CT2")
        # The upstream decoder imports the entire training engine just for this fixed
        # edge filter. Supply the identical operation, avoiding unrelated training/FID
        # packages during inference. Model code and checkpoint remain unmodified.
        engine = types.ModuleType("segm.engine")
        engine.functional_conv2d = edge_operator
        sys.modules["segm.engine"] = engine
        from segm.model import segmenter
        from segm.model.decoder import MaskTransformer
        from segm.model.vit import VisionTransformer

        checkpoint = torch.load(source / "checkpoint.pth", map_location="cpu", weights_only=True)
        weights = checkpoint.get("model", checkpoint)
        # Infer shape-bearing architecture from the checkpoint; behavior switches are
        # the official released ViT-L inference defaults, verified by strict loading.
        width = weights["encoder.cls_token"].shape[-1]
        layers = len({key.split(".")[2] for key in weights if key.startswith("encoder.blocks.")})
        encoder = VisionTransformer(
            image_size=(256, 256),
            patch_size=16,
            n_layers=layers,
            d_model=width,
            d_ff=4 * width,
            n_heads=width // 64,
            n_cls=1000,
            dropout=0,
            drop_path_rate=0,
        )
        n_layers = len(
            {key.split(".")[3] for key in weights if key.startswith("decoder.blocks.0.")}
        )
        decoder = MaskTransformer(
            n_cls=313,
            patch_size=16,
            d_encoder=width,
            n_layers=n_layers,
            n_heads=width // 64,
            d_model=width,
            d_ff=4 * width,
            drop_path_rate=0,
            dropout=0,
            add_l1_loss=True,
            color_position=any(key.startswith("decoder.pos_color.") for key in weights),
            change_mask=False,
            color_as_condition=False,
            multi_scaled=False,
            crop_size=256,
            downchannel=False,
            add_conv=True,
            before_classify=False,
            l1_conv=True,
            l1_linear=False,
            add_edge=False,
            n_blocks=1,
            without_colorattn=False,
            without_colorquery=False,
            without_classification=False,
            sin_color_pos=False,
        )
        # Upstream helper objects allocate plain tensors on default CUDA and are
        # not nn.Modules. Construct their unchanged lookups on CPU, then register
        # them here so normal .to(device) also handles CPU and nondefault GPUs.
        soft_encode, mean_decode = segmenter.SoftEncodeAB, segmenter.AnnealedMeanDecodeQ
        try:
            segmenter.SoftEncodeAB = functools.partial(soft_encode, device="cpu")
            segmenter.AnnealedMeanDecodeQ = functools.partial(mean_decode, device="cpu")
            self.model = segmenter.Segmenter(
                encoder, decoder, 313, False, "vit_large_patch16_384", False
            )
        finally:
            segmenter.SoftEncodeAB, segmenter.AnnealedMeanDecodeQ = soft_encode, mean_decode
        self.model.load_state_dict(weights, strict=True)
        self.register_buffer("encode_gamut", self.model.encode_ab.q_to_ab, persistent=False)
        self.register_buffer("decode_gamut", self.model.decode_q.q_to_ab, persistent=False)
        prior = metadata(source / "mask_prior.pickle")
        mask = np.zeros((4, 313), bool)
        for luminance in range(100):
            mask[min(luminance // 25, 3)] |= np.asarray(prior[luminance], bool)
        self.register_buffer("mask", torch.from_numpy(mask.astype(np.float32)))

    def forward(self, condition):
        self.model.encode_ab.q_to_ab = self.encode_gamut
        self.model.decode_q.q_to_ab = self.decode_gamut
        light = condition * 100
        bins = (light[:, 0] // 25).long().clamp(0, 3)
        mask = self.mask[bins].reshape(len(condition), -1, 313)
        # Ground-truth chroma is used upstream only for diagnostic q_actual. Pass zeros
        # so the adapter cannot observe test colors.
        ab, _, _, _ = self.model.inference(
            light, torch.zeros(len(condition), 2, 256, 256, device=light.device), mask, appli=True
        )
        return ab / 110


def experiment(config):
    torch.set_num_threads(4)
    torch.hub.set_dir(str(Path(config["data_root"]) / "models" / "torch" / "hub"))
    run = Run(config["output"], config)
    name = config["reference"]
    model = {
        "ddcolor": DDColorReference,
        "ct2": CT2Reference,
        "bigcolor": BigColorReference,
        "swinir": SwinIRReference,
        "realesrgan": RealESRGANReference,
    }[name](config)
    model = model.to(config["device"]).eval().requires_grad_(False)
    rows, scores, gallery = evaluate(model, name, 0, config)
    np.savez_compressed(run.output / "per_image_scores.npz", **scores)
    np.savez_compressed(run.output / "gallery.npz", **gallery)
    run.finish(
        {
            "rows": rows,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "protocol": "Frozen official external reference; same 400 high-resolution COCO test crops and observation. No local fitting or test selection. External training exposure differs and overlap is not audited. BigColor uses a declared fixed ResNet50 class-selection adapter and all four fixed random latent draws; not an exact reproduction of its default classifier. Real-ESRGAN was trained on blind synthetic degradations, differing from this bicubic test.",
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    experiment(json.loads(parser.parse_args().config.read_text()))
