import torch
from enum import Enum

def get_device_name():
    device_name = torch.cuda.get_device_name(0)
    return "_".join(device_name.split(" "))


def get_max_flops(device_name):
    if "4090" in device_name:
        return 166
    
    if "A800" in device_name or "A100" in device_name:
        return 312
    
    assert False, "Please add your GPU number here"
    
    
class ModelType(Enum):
    LLAMA3_8B = "llama3_8b"
    LLAMA3_70B = "llama3_70"
    LLAMA3_405B = "llama3_405b"


class CheckType(Enum):
    PYTORCH_VERSION = "version"
    MARLIN_PERF = "marlin"
    SHAPE_UTILS_CHECK = "shape"


# Weight Shapes are in the format
# (name, [K, N], TP_SPLIT_DIM)
# Example:
#  A shape of ([14336, 4096], 0) indicates the following GEMM shape,
#   - TP1 : K = 14336, N = 4096
#   - TP2 : K = 7168, N = 4096
#  A shape of ([4096, 6144], 1) indicates the following GEMM shape,
#   - TP1 : K = 4096, N = 6144
#   - TP4 : K = 4096, N = 1536

# TP1 shapes
WEIGHT_SHAPES = {
    "llama3_8b": [
        ("qkv_proj", [4096, 6144], 1),
        ("o_proj", [4096, 4096], 0),
        ("gate_up_proj", [4096, 28672], 1),
        ("down_proj", [14336, 4096], 0),
    ],
    "llama3-70": [
        ("qkv_proj", [8192, 10240], 1),
        ("o_proj", [8192, 8192], 0),
        ("gate_up_proj", [8192, 57344], 1),
        ("down_proj", [28672, 8192], 0),
    ],
    "llama3-405b": [
        ("qkv_proj", [16384, 18432], 1),
        ("o_proj", [16384, 16384], 0),
        ("gate_up_proj", [16384, 106496], 1),
        ("down_proj", [53248, 16384], 0),
    ],
}
