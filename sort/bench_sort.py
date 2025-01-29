import random
import numpy as np
import time
import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from typing import Callable, Iterable, List
from dataclasses import dataclass

def enforce_reproduce():
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

INPUT_REPLICAS = 10 # use multiple inputs to avoid L2 cache hit 

@dataclass
class BenchmarkTensors:
    input_tensor: torch.Tensor

# bench
def bench_fns(label: str, sub_label: str, description: str,
              fns: List[Callable]):
    min_run_time = 1
    res = TBenchmark.Timer(
        stmt="""
        for fn in fns:
            fn()
        """,
        globals={
            "fns": fns
        },
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)
    return res

# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()

def sort_bench_fn(bt: BenchmarkTensors) -> Callable:
    return lambda: torch.sort(input=bt.input_tensor)

def bench_sort():
    timers = []

    label = "bench_sort"
    configs = [i * int(1e7) for i in range(1, 11)]
    for cfg in configs:
        
        bts = []
        for i in range(INPUT_REPLICAS):
            input_tensor = torch.randint(0, cfg, (cfg,), dtype=torch.int32, device="cuda")
            bt = BenchmarkTensors(input_tensor=input_tensor)
            bts.append(bt)
        sub_label = f"input_{cfg}"
        timers.append(
            bench_fns(label, sub_label, "bench_index_sort", [sort_bench_fn(bt) for bt in bts]))

    return timers


def run():
    enforce_reproduce()

    timers = bench_sort()
    print_timers(timers)

if __name__ == "__main__":
    run()