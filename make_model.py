#!/usr/bin/env python3
"""Generates Model/AneStress.mlpackage: a stack of fp16 3x3 convolutions,
an op type the Apple Neural Engine runs natively. Weights are random (He init),
and one weight tensor is shared by every layer to keep the file small.

    pip install coremltools numpy && python3 make_model.py
"""
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

C, H, W, LAYERS = 256, 64, 64, 12

rng = np.random.default_rng(0)
weight = (rng.standard_normal((C, C, 3, 3)) * np.sqrt(2.0 / (C * 9))).astype(np.float16)
bias = np.zeros((C,), dtype=np.float16)


@mb.program(input_specs=[mb.TensorSpec(shape=(1, C, H, W), dtype=types.fp16)],
            opset_version=ct.target.macOS13)
def prog(x):
    w = mb.const(val=weight, name="w")
    b = mb.const(val=bias, name="b")
    for i in range(LAYERS):
        x = mb.conv(x=x, weight=w, bias=b, pad_type="same", name=f"conv{i}")
        x = mb.relu(x=x, name=f"relu{i}")
    return mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True, name="y")


model = ct.convert(prog, convert_to="mlprogram",
                   minimum_deployment_target=ct.target.macOS13,
                   compute_precision=ct.precision.FLOAT16,
                   skip_model_load=True)
model.short_description = "ANE load generator: %d x conv3x3 %dch @ %dx%d" % (LAYERS, C, H, W)
model.save("Model/AneStress.mlpackage")
gflop = LAYERS * 2 * C * C * 9 * H * W / 1e9
print("saved Model/AneStress.mlpackage, %.1f GFLOP per inference" % gflop)
