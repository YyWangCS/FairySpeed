import os
import torch
import argparse
import pandas as pd
from loguru import logger
from torch.profiler import profile, record_function, ProfilerActivity
from utils import WEIGHT_SHAPES, ModelType, get_device_name, get_max_flops




device_name = get_device_name()
torch_version = torch.__version__

profiler_dir = "./profiler"
dtype = torch.float16
RUN_ITERATIONS = 5
WARMUP_ITERATIONS = 2
MAX_BATCH = 32768
GEMM_PATTERN = ["ampere", "cutlass", "gemm", "gemv"]
BATCH_STEP = 4


def rand_weights(size_k : int, size_n : int, dtype, device : str, num : int):
    base_weight = torch.randn(size_k, size_n, dtype=dtype, device=device)
    weights = [base_weight * torch.rand(1, device=device, dtype=dtype) for i in range(num)]
    return weights


def get_kernel_total_time(event):
    if hasattr(event, 'self_device_time_total'):
        return event.self_device_time_total
    else:
        return event.self_cuda_time_total
    
def bench_torch_gemm(size_k: int, size_n: int, device: str="cuda"):
    weights = rand_weights(size_k, size_n, dtype=dtype, device=device, num=RUN_ITERATIONS)
    full_activation = torch.randn(MAX_BATCH, size_k, dtype=dtype, device=device)
    batches = [1, 2, 4, 6, 8, 10, 12, 14] + list(range(16, MAX_BATCH + 1, BATCH_STEP))
    records = []

    for batch in batches:
        activation = full_activation[0:batch].contiguous()
        # warm up
        for i in range(WARMUP_ITERATIONS):
            _ = torch.matmul(activation, weights[0])
        torch.cuda.synchronize()

        with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(RUN_ITERATIONS):
                c = torch.matmul(activation, weights[i])
        torch.cuda.synchronize()

        gemm_events = []
        for event in  prof.key_averages():
            if any(pattern in event.key for pattern in GEMM_PATTERN):
                gemm_events.append(event)
        assert len(gemm_events)==1, "only one kernel event should be found, please check GEMM_PATTERN"
        event = gemm_events[0]

        average_kernel_time = round(get_kernel_total_time(event) / RUN_ITERATIONS, 2)
        flops = round(2 * batch * size_k * size_n / average_kernel_time / 1e6, 2)
        utils = round(flops / get_max_flops(device_name), 5)
        records.append([batch, average_kernel_time, flops, utils, event.key])
        del prof
        del activation
    
    df = pd.DataFrame(records, columns=["batch", "kernel_time", "flops", "utils", "kernel_name"])

    if not os.path.exists(profiler_dir):
        os.makedirs(profiler_dir)
    dtype_str = str(dtype).split(".")[-1]
    perf_record_file = os.path.join(profiler_dir, f"torchgemm_{device_name}_torch_{torch_version}_K{size_k}_N{size_n}_dtype{str(dtype_str)}.csv")
    df.to_csv(perf_record_file, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="select the model and tp size you want to profile gemm")

    parser.add_argument(
        "--model_type", 
        type=str, 
        required=True,
        choices=[e.value for e in ModelType],
        help="model_type，like llama3-8b"
    )
    
    parser.add_argument(
        "--tp_size", 
        type=int, 
        default=1, 
        help="tensor parallel size, default to 1"
    )

    parser.add_argument(
        "--module", 
        type=str, 
        choices=["qkv_proj", "o_proj", "down_proj", "gate_up_proj"],
        default="all",
        help="model_type，like gate_proj",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    for shape_and_tp in WEIGHT_SHAPES[args.model_type]:
        module_name = shape_and_tp[0]
        if (args.module != "all") and (module_name != args.module):
            continue

        size_k, size_n = shape_and_tp[1]
        tp_dim = shape_and_tp[2]
        if tp_dim == 0:
            size_k = size_k // args.tp_size
        else:
            size_n = size_n // args.tp_size
        logger.info(f"profile gemm performance for {args.model_type}, with shape K={size_k}, shape N={size_n}")
        bench_torch_gemm(size_k=size_k, size_n=size_n)

        