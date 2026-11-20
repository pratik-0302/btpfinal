"""
AV-HuBERT in plain PyTorch (no fairseq).

Loads the official self-supervised checkpoints released by Meta
(e.g. https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt)
and reproduces AVHubertModel.extract_finetune() from facebookresearch/av_hubert
for the two single-modality passes used for lip-sync checking:

  video only:  features = LN(concat[0, video_feats]) -> proj -> transformer
  audio only:  features = LN(concat[audio_feats, 0]) -> proj -> transformer

Module names match the checkpoint, so the official state_dict loads with a
strict key check. Verified numerically against the upstream code.
"""
from __future__ import annotations

import math
import pickle
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- visual front-end
def _conv3x3(i, o, s=1):
    return nn.Conv2d(i, o, 3, s, 1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, prelu=True):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.PReLU(planes) if prelu else nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.PReLU(planes) if prelu else nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        res = self.downsample(x) if self.downsample is not None else x
        return self.relu2(out + res)


class ResNetTrunk(nn.Module):
    def __init__(self, prelu=True):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make(64, 2, 1, prelu)
        self.layer2 = self._make(128, 2, 2, prelu)
        self.layer3 = self._make(256, 2, 2, prelu)
        self.layer4 = self._make(512, 2, 2, prelu)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make(self, planes, blocks, stride, prelu):
        down = None
        if stride != 1 or self.inplanes != planes:
            down = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, down, prelu)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes, prelu=prelu) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)


class ResEncoder(nn.Module):
    def __init__(self, prelu=True):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.PReLU(64) if prelu else nn.ReLU(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))
        self.trunk = ResNetTrunk(prelu)

    def forward(self, x):                      # (B, 1, T, H, W) -> (B, 512, T)
        B = x.size(0)
        x = self.frontend3D(x)
        T = x.shape[2]
        x = x.transpose(1, 2).reshape(B * T, x.size(1), x.size(3), x.size(4))
        x = self.trunk(x)
        return x.view(B, T, -1).transpose(1, 2).contiguous()


class SubModel(nn.Module):
    def __init__(self, resnet, input_dim, dim):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, dim)

    def forward(self, x):                      # -> (B, dim, T)
        if self.resnet is not None:
            x = self.resnet(x)
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------- transformer
class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, key_padding_mask=None):   # x: (B, T, C)
        B, T, C = x.shape
        h = self.heads
        q = self.q_proj(x).view(B, T, h, C // h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, C // h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, C // h).transpose(1, 2)
        mask = None
        if key_padding_mask is not None:       # True = padding
            mask = (~key_padding_mask)[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(y.transpose(1, 2).reshape(B, T, C))


class EncoderLayer(nn.Module):
    def __init__(self, dim, ffn, heads, ln_first):
        super().__init__()
        self.self_attn = SelfAttention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.ln_first = ln_first

    def forward(self, x, pad=None):
        if self.ln_first:
            x = x + self.self_attn(self.self_attn_layer_norm(x), pad)
            return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        x = self.self_attn_layer_norm(x + self.self_attn(x, pad))
        return self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x))))


class PosConv(nn.Module):
    """Conv1d(k=128, groups=16) + SamePad + GELU. Weight norm is folded in at load time."""
    def __init__(self, dim, k=128, groups=16):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, k, padding=k // 2, groups=groups)
        self.remove = 1 if k % 2 == 0 else 0

    def forward(self, x):                      # (B, T, C)
        y = self.conv(x.transpose(1, 2))
        if self.remove:
            y = y[:, :, :-self.remove]
        return F.gelu(y).transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, ffn, heads, layers, ln_first, k=128, groups=16):
        super().__init__()
        self.pos_conv = PosConv(dim, k, groups)
        self.layers = nn.ModuleList([EncoderLayer(dim, ffn, heads, ln_first) for _ in range(layers)])
        self.layer_norm = nn.LayerNorm(dim)
        self.ln_first = ln_first

    def forward(self, x, pad=None):
        """Returns the list of every layer's output (1..L); the last entry is the
        model's final output (with the closing LayerNorm for pre-LN models)."""
        if pad is not None:
            x = x.masked_fill(pad[..., None], 0.0)
        x = x + self.pos_conv(x)
        if not self.ln_first:
            x = self.layer_norm(x)
        outs = []
        for layer in self.layers:
            x = layer(x, pad)
            outs.append(x)
        if self.ln_first:
            outs[-1] = self.layer_norm(outs[-1])
        return outs


# ---------------------------------------------------------------- model
class AVHubertLite(nn.Module):
    def __init__(self, dim=1024, ffn=4096, heads=16, layers=24, ln_first=True,
                 audio_feat_dim=104, prelu=True, conv_pos=128, conv_pos_groups=16):
        super().__init__()
        self.dim = dim
        self.feature_extractor_audio = SubModel(None, audio_feat_dim, dim)
        self.feature_extractor_video = SubModel(ResEncoder(prelu), 512, dim)
        self.layer_norm = nn.LayerNorm(2 * dim)
        self.post_extract_proj = nn.Linear(2 * dim, dim)
        self.encoder = TransformerEncoder(dim, ffn, heads, layers, ln_first, conv_pos, conv_pos_groups)

    def _run(self, fa, fv):
        x = torch.cat([fa, fv], dim=1).transpose(1, 2)        # (B, T, 2C), audio first
        x = self.post_extract_proj(self.layer_norm(x))
        return self.encoder(x)

    def forward_video(self, video):
        """video: (B, 1, T, 88, 88) normalised grey mouth crops -> list of (B, T, C)."""
        fv = self.feature_extractor_video(video)
        return self._run(torch.zeros_like(fv), fv)

    def forward_audio(self, audio):
        """audio: (B, 104, T) stacked log filterbanks -> list of (B, T, C)."""
        fa = self.feature_extractor_audio(audio)
        return self._run(fa, torch.zeros_like(fa))


# ---------------------------------------------------------------- checkpoint loading
_SAFE_MODULES = ("torch", "collections", "builtins", "copyreg", "numpy", "_codecs", "argparse")


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def __call__(self, *a, **k):
        return _Dummy()


class _Unpickler(pickle.Unpickler):
    """Loads a fairseq checkpoint without fairseq/omegaconf: config objects become
    inert placeholders; tensors and plain containers load normally."""
    def find_class(self, module, name):
        if module.split(".")[0] in _SAFE_MODULES:
            return super().find_class(module, name)
        return _Dummy


_safe_pickle = types.ModuleType("safe_pickle")
_safe_pickle.Unpickler = _Unpickler
_safe_pickle.load = lambda f, **kw: _Unpickler(f, **kw).load()
_safe_pickle.__name__ = "safe_pickle"


def _fold_weight_norm(sd, prefix):
    """weight_norm(dim=2): weight = g * v / ||v|| (norm over dims 0 and 1)."""
    g, v = sd.pop(prefix + "weight_g", None), sd.pop(prefix + "weight_v", None)
    if g is None:   # newer torch parametrizations naming
        g = sd.pop(prefix + "parametrizations.weight.original0", None)
        v = sd.pop(prefix + "parametrizations.weight.original1", None)
    if g is None:
        return
    norm = v.norm(dim=(0, 1), keepdim=True)
    sd[prefix + "weight"] = v * (g / norm)


def load_avhubert(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", pickle_module=_safe_pickle, weights_only=False)
    sd = dict(ck["model"] if "model" in ck else ck)
    if any(k.startswith("encoder.w2v_model.") for k in sd):          # fine-tuned checkpoint
        sd = {k[len("encoder.w2v_model."):]: v for k, v in sd.items() if k.startswith("encoder.w2v_model.")}
    _fold_weight_norm(sd, "encoder.pos_conv.0.")
    sd = {k.replace("encoder.pos_conv.0.", "encoder.pos_conv.conv."): v for k, v in sd.items()}
    dim = sd["post_extract_proj.weight"].shape[0]
    layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers."))
    ffn = sd["encoder.layers.0.fc1.weight"].shape[0]
    k = sd["encoder.pos_conv.conv.weight"].shape[-1]
    groups = dim // sd["encoder.pos_conv.conv.weight"].shape[1]
    prelu = "feature_extractor_video.resnet.frontend3D.2.weight" in sd
    # pre-LN for large models (as in the released configs); post-LN for base
    ln_first = _cfg_flag(ck, "layer_norm_first", default=True)   # true in every released config
    model = AVHubertLite(dim=dim, ffn=ffn, heads=dim // 64, layers=layers, ln_first=ln_first,
                         audio_feat_dim=sd["feature_extractor_audio.proj.weight"].shape[1],
                         prelu=prelu, conv_pos=k, conv_pos_groups=groups)
    own = model.state_dict()
    wanted = {k_: v for k_, v in sd.items() if k_ in own}
    missing = [k_ for k_ in own if k_ not in wanted]
    bad = [k_ for k_ in wanted if wanted[k_].shape != own[k_].shape]
    if missing or bad:
        raise RuntimeError(f"AV-HuBERT checkpoint mismatch: missing {missing[:5]}, shape {bad[:5]}")
    model.load_state_dict(wanted, strict=True)
    info = {"dim": dim, "layers": layers, "ffn": ffn, "heads": dim // 64, "ln_first": ln_first,
            "unused_keys": sorted(set(sd) - set(own))[:10]}
    return model.to(device).eval(), info


def _cfg_flag(ck, name, default):
    """Read cfg.model.<name> from the placeholder config if it survived; else default."""
    try:
        st = ck["cfg"]._state if hasattr(ck.get("cfg"), "_state") else None
        content = st.get("_content") if isinstance(st, dict) else None
        model = content["model"]._state["_content"]
        val = model[name]._state["_val"] if hasattr(model[name], "_state") else model[name]
        if isinstance(val, bool):
            return val
    except Exception:
        pass
    try:
        args = ck.get("args")
        if args is not None and hasattr(args, name):
            return bool(getattr(args, name))
    except Exception:
        pass
    return default
