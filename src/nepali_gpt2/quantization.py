"""CPU quantization variants, including a portable packed INT4 reference layer.

INT4 stores two signed weights per byte and dequantizes for each forward.
This is a storage/quality reference, not an accelerated INT4 kernel.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic


class Int4Linear(nn.Module):
    packed: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7
        quantized = (weight / scale).round().clamp(-7, 7).to(torch.int16) + 8
        if self.in_features % 2:
            quantized = torch.nn.functional.pad(quantized, (0, 1), value=8)
        self.register_buffer(
            "packed", (quantized[:, 0::2] | (quantized[:, 1::2] << 4)).to(torch.uint8)
        )
        self.register_buffer("scale", scale)
        self.register_buffer(
            "bias", linear.bias.detach().clone() if linear.bias is not None else None
        )

    def forward(self, x):
        lo = (self.packed & 15).to(torch.int16) - 8
        hi = (self.packed >> 4).to(torch.int16) - 8
        weight = (
            torch.stack((lo, hi), dim=-1).flatten(1)[:, : self.in_features] * self.scale
        )
        return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)


def convert(model, precision):
    model = copy.deepcopy(model).cpu().eval()
    if precision == "fp32":
        return model.float()
    if precision == "fp16":
        return model.half()
    if precision == "int8":
        return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    if precision != "int4":
        raise ValueError(f"Unsupported precision: {precision}")

    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, Int4Linear(child))
            else:
                replace(child)

    replace(model)
    return model
