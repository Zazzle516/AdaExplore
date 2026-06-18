# Operator-name -> family-slug map.
#
# Covers the top operators from a KernelBench L1+L2 survey (>=90% of files).
# Long-tail ops not in the map flip the `has_unknown_ops` bit in detect.py
# (base + _default.md still apply); adding a new entry is one line.

FAMILY_MAP = {
    # conv
    "Conv1d": "conv", "Conv2d": "conv", "Conv3d": "conv",
    "ConvTranspose1d": "conv", "ConvTranspose2d": "conv", "ConvTranspose3d": "conv",
    # linear
    "Linear": "linear", "matmul": "linear", "bmm": "linear", "einsum": "linear",
    "addmm": "linear",
    # norm
    "BatchNorm1d": "norm", "BatchNorm2d": "norm", "BatchNorm3d": "norm",
    "GroupNorm": "norm", "LayerNorm": "norm",
    "InstanceNorm1d": "norm", "InstanceNorm2d": "norm", "InstanceNorm3d": "norm",
    "RMSNorm": "norm",
    # activation
    "GELU": "activation", "ReLU": "activation", "LeakyReLU": "activation",
    "Sigmoid": "activation", "Tanh": "activation", "Mish": "activation",
    "SiLU": "activation", "Softplus": "activation", "ELU": "activation",
    "gelu": "activation", "relu": "activation", "sigmoid": "activation",
    "tanh": "activation", "mish": "activation", "silu": "activation",
    # reduction
    "sum": "reduction", "mean": "reduction", "min": "reduction", "max": "reduction",
    "prod": "reduction", "softmax": "reduction", "log_softmax": "reduction",
    "logsumexp": "reduction", "argmax": "reduction", "argmin": "reduction",
    # pooling
    "MaxPool1d": "pooling", "MaxPool2d": "pooling", "MaxPool3d": "pooling",
    "AvgPool1d": "pooling", "AvgPool2d": "pooling", "AvgPool3d": "pooling",
    "AdaptiveAvgPool1d": "pooling", "AdaptiveAvgPool2d": "pooling",
    "AdaptiveAvgPool3d": "pooling",
}
