import random
import numpy as np

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from typing import Callable, Iterable, List, Tuple, Optional
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
    index_tensor: torch.Tensor
    source_tensor: Optional[torch.Tensor]

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

def index_select_bench_fn(bt: BenchmarkTensors) -> Callable:
    return lambda: torch.index_select(input=bt.input_tensor, dim=0, index=bt.index_tensor)

def bench_index_select():
    timers = []

    label = "bench_index_select"
    configs = [(1000000, 128, 307200), (1000000, 32, 307200), (1000000, 128, 204800), (1000000, 32, 204800), (128000, 4096, 4096)]
    for cfg in configs:
        num_embedding = cfg[0]
        embedding_dim = cfg[1]
        input_size = cfg[2]
        input_tensor = torch.randn(num_embedding, embedding_dim, dtype=torch.float32, device="cuda")
        bts = []
        for i in range(INPUT_REPLICAS):
            index_tensor = torch.randint(0, num_embedding, (input_size,), dtype=torch.long, device="cuda")
            bt = BenchmarkTensors(input_tensor=input_tensor, index_tensor=index_tensor, source_tensor=None)
            bts.append(bt)
        sub_label = f"[{num_embedding}, {embedding_dim}], {input_size}"
        timers.append(
            bench_fns(label, sub_label, "bench_index_select", [index_select_bench_fn(bt) for bt in bts]))
    return timers

def index_add_bench_fn(bt: BenchmarkTensors) -> Callable:
    return lambda: torch.index_add(input=bt.input_tensor, dim=0, index=bt.index_tensor, source=bt.source_tensor, out=bt.input_tensor)

def bench_index_add():
    timers = []

    label = "bench_index_add"
    configs = [(1000000, 128, 307200), (1000000, 32, 307200), (1000000, 128, 204800), (1000000, 32, 204800), (128000, 4096, 4096)]
    for cfg in configs:
        num_embedding = cfg[0]
        embedding_dim = cfg[1]
        input_size = cfg[2]
        input_tensor = torch.randn(num_embedding, embedding_dim, dtype=torch.float32, device="cuda")
        bts = []
        for i in range(INPUT_REPLICAS):
            index_tensor = torch.randint(0, num_embedding, (input_size,), dtype=torch.long, device="cuda")
            source_tensor = torch.randn(input_size, embedding_dim, dtype=torch.float32, device="cuda")
            bt = BenchmarkTensors(input_tensor=input_tensor, index_tensor=index_tensor, source_tensor=source_tensor)
            bts.append(bt)
        sub_label = f"[{num_embedding}, {embedding_dim}], {input_size}"
        timers.append(
            bench_fns(label, sub_label, "bench_index_add", [index_add_bench_fn(bt) for bt in bts]))
    return timers


def index_reduce_bench_fn(bt: BenchmarkTensors) -> Callable:
    return lambda: torch.index_reduce(input=bt.input_tensor, dim=0, index=bt.index_tensor, source=bt.source_tensor, reduce='amax', out=bt.input_tensor)

def bench_index_reduce():
    timers = []

    label = "bench_index_reduce"
    configs = [(1000000, 128, 307200), (1000000, 32, 307200), (1000000, 128, 204800), (1000000, 32, 204800)]
    for cfg in configs:
        num_embedding = cfg[0]
        embedding_dim = cfg[1]
        input_size = cfg[2]
        input_tensor = torch.randn(num_embedding, embedding_dim, dtype=torch.float32, device="cuda")
        bts = []
        for i in range(INPUT_REPLICAS):
            index_tensor = torch.randint(0, num_embedding, (input_size,), dtype=torch.long, device="cuda")
            source_tensor = torch.randn(input_size, embedding_dim, dtype=torch.float32, device="cuda")
            bt = BenchmarkTensors(input_tensor=input_tensor, index_tensor=index_tensor, source_tensor=source_tensor)
            bts.append(bt)
        sub_label = f"[{num_embedding}, {embedding_dim}], {input_size}"
        timers.append(
            bench_fns(label, sub_label, "bench_index_amax", [index_reduce_bench_fn(bt) for bt in bts]))
    return timers


def run():
    enforce_reproduce()

    timers = bench_index_select()
    print_timers(timers)
    
    timers = bench_index_add()
    print_timers(timers)

    timers = bench_index_reduce()
    print_timers(timers)

if __name__ == "__main__":
    run()