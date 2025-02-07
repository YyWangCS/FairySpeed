import torch
from torch import Tensor
from typing import Tuple

__all__ = ["stable_sort", "stable_sort_opaque"]

def stable_sort(input: Tensor) -> Tuple[Tensor, Tensor]:
    values, indices = torch.ops.extension_cpp.stable_sort.default(input)
    return values, indices


def stable_sort_opaque(input: Tensor) -> Tuple[Tensor, Tensor]:
    values, indices = torch.ops.extension_cpp.stable_sort_opaque.default(input)
    return values, indices