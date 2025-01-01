import os
import torch
import argparse
import pandas as pd
from loguru import logger
from torch.profiler import profile, record_function, ProfilerActivity
from utils import WEIGHT_SHAPES, ModelType, get_device_name, get_max_flops, get_kernel_total_time


import vllm
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace, marlin_quantize)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N)

from torch.profiler import profile, record_function, ProfilerActivity



def rand_weights(size_k : int, size_n : int, dtype, device : str):
    weight = torch.randn(size_k, size_n, dtype=dtype, device=device)
    return weight



device_name = get_device_name()
vllm_version = vllm.__version__

profiler_dir = "./profiler"
dtype_str = "gptq_int8"
RUN_ITERATIONS = 5
WARMUP_ITERATIONS = 2
MAX_BATCH = 8192
BATCH_STEP = 4



def bench_gptq_marlin_gemm(size_k: int, size_n: int, device: str="cuda"):
    records = []
    batches = [1, 2, 4, 6, 8, 10, 12, 14] + list(range(16, MAX_BATCH + 1, BATCH_STEP))
    
    quant_type = scalar_types.uint8b128
    group_size = 128
    is_k_full = True
    act_order = False
    activation_dtype = torch.bfloat16


    full_activation = rand_weights(MAX_BATCH, size_k, dtype=activation_dtype, device=device)
    original_weight = rand_weights(size_k, size_n, dtype=activation_dtype, device=device)
   
    # Marlin quant
    (
        marlin_w_ref,
        marlin_q_w,
        marlin_s,
        marlin_g_idx,
        marlin_sort_indices,
        marlin_rand_perm,
    ) = marlin_quantize(original_weight, quant_type, group_size, act_order)

    marlin_zp = torch.zeros_like(marlin_s, dtype=torch.int)

    # Prepare
    marlin_workspace = MarlinWorkspace(size_n, GPTQ_MARLIN_MIN_THREAD_N,
                                       GPTQ_MARLIN_MAX_PARALLEL)
    
    for batch in batches:
        activation = full_activation[0:batch].contiguous()

        for i in range(WARMUP_ITERATIONS):
            ops.gptq_marlin_gemm(activation, 
                                 marlin_q_w, 
                                 marlin_s, 
                                 marlin_zp, 
                                 marlin_g_idx, 
                                 marlin_sort_indices, 
                                 marlin_workspace.scratch, 
                                 quant_type, 
                                 batch, 
                                 size_n, 
                                 size_k, 
                                 is_k_full, 
                                 False, 
                                 True, 
                                 False)

        torch.cuda.synchronize()
        with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            for i in range(RUN_ITERATIONS):
                ops.gptq_marlin_gemm(activation, 
                                     marlin_q_w, 
                                     marlin_s, 
                                     marlin_zp, 
                                     marlin_g_idx, 
                                     marlin_sort_indices, 
                                     marlin_workspace.scratch, 
                                     quant_type, 
                                     batch, 
                                     size_n, 
                                     size_k, 
                                     is_k_full, 
                                     False, 
                                     True, 
                                     False)
        torch.cuda.synchronize()
        gemm_events = []
        for event in prof.key_averages():
            if "gptq_marlin_gemm" in event.key:
                gemm_events.append(event)

        assert len(gemm_events)==1, "only one kernel event should be found"
        event = gemm_events[0]

        average_kernel_time = round(get_kernel_total_time(event) / RUN_ITERATIONS, 2)
        flops = round(2 * batch * size_k * size_n / average_kernel_time / 1e6, 2)
        utils = round(flops / get_max_flops(device_name), 5)
        records.append([batch, average_kernel_time, flops, utils])


    
    df = pd.DataFrame(records, columns=['batch', 'time', 'flops', 'utils']) 
    
    if not os.path.exists(profiler_dir):
        os.makedirs(profiler_dir)
    perf_record_file = os.path.join(profiler_dir, f"Marlin_{device_name}_vllm_{vllm_version}_K{size_k}_N{size_n}_{dtype_str}.csv")
    df.to_csv(perf_record_file, index=False)




def parse_arguments():
    parser = argparse.ArgumentParser(description="select the model and tp size you want to profile gemm")

    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=[e.value for e in ModelType],
        required=True,
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
        logger.info(f"profile Marlin int8 gemm performance for {args.model_type}, with shape K={size_k}, shape N={size_n}")
        bench_gptq_marlin_gemm(size_k=size_k, size_n=size_n)