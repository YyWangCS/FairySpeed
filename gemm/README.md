## Reproducing Experimental Results

Reproducing the experimental results from article [Inconsistency in GEMM Performance](https://yywangcs.notion.site/Inconsistency-in-GEMM-Performance-16efc9f5d80580838090dded05493014?pvs=74).

### MFU Consistency with Matrix Shapes

#### Obtaining GEMM Performance Profiler for Matrices

On a GPU machine, such as the RTX4090, execute the following commands:

```python
# Profiler GEMM performance for llama3_8b qkv_proj with tp 1, which means [batch, 4096]*[4096, 6144]
python3 bench_llm_torch_gemm.py --model_type llama3_8b --module qkv_proj --tp_size 1

# Profiler GEMM performance for llama3_8b qkv_proj with tp 2, which means [batch, 4096]*[4096, 14336]
python3 bench_llm_torch_gemm.py --model_type llama3_8b --module gate_up_proj --tp_size 2

# Profiler GEMM performance for all GEMMs of llama3_8b with shapes [4096, 4096], [4096, 6144], [4096, 28672], [14336, 4096]
python3 bench_llm_torch_gemm.py --model_type llama3_8b --module all --tp_size 1
```

For example, for the second command, the program will automatically execute a matrix computation of type float16 for [batch, 4096] * [4096, 14336], where `batch` is selected from [4, 8, 12, ..., 32768], corresponding to 8192 matrix computations. For each batch, the program will run `torch.matmul` multiple times and measure the average kernel execution time on the GPU. Based on this, the actual achieved FLOPS for the matrix will be calculated and compared to the theoretical performance of the RTX4090. In total, 8192 GEMM MFU values will be obtained. The results will be saved in the profiler directory, with the file format `torchgemm_NVIDIA_GeForce_RTX_4090_torch_2.5.1+cu124_K4096_N14336_dtypefloat16.csv`, representing GPU type, Torch version, matrix shape, precision, etc. For the [4096, 14336] shape on RTX4090, we provide two test results using Torch versions 2.3 and 2.5.1 under the profiler folder for reference.

#### Analyzing Batches with Abnormal GEMM Efficiency

Execute the following command to analyze batches with abnormal GEMM efficiency:

```python
# Analyze batches with abnormal GEMM efficiency, identifying sub-matrices with significantly lower efficiency than smaller ones
python3 shape_utils_analyze.py --profiler ./profiler/torchgemm_NVIDIA_GeForce_RTX_4090_torch_2.5.1+cu124_K4096_N14336_dtypefloat16.csv

# Analyze batches with abnormal GEMM efficiency and check if they can be optimized using split_gemm
python3 shape_utils_analyze.py --profiler ./profiler/torchgemm_NVIDIA_GeForce_RTX_4090_torch_2.5.1+cu124_K4096_N14336_dtypefloat16.csv --check_split_gemm
```

For each batch size N, if for smaller batches, the GEMM efficiency for [N, 4096] is less than the efficiency of smaller matrices [N1, 4096] (where N1 < N) multiplied by a threshold (e.g., 0.85), it is considered an anomaly in performance for the current [N, 4096] matrix shape.

If `--check_split_gemm` is set, for abnormal GEMM efficiency batches, such as N, the program will search for two smaller matrices [N1, 4096] and [N2, 4096], where N1 + N2 = N. It will then compare the GEMM performance of these two smaller matrices and sum the results. If the sum is noticeably lower than the performance for batch size N, the matrix [N, 4096] will be classified as a "splitwise_gemm" optimizable matrix. The program will automatically output all such optimizable matrices.

## MFU Consistency with Quantization Computation

#### Obtaining Marlin GEMM Performance Profiler

On a GPU machine, such as the RTX4090, execute the following commands:

```python
# source /Envs/vllm_0.6.5/bin/bash

# Profiler GEMM performance for llama3_8b qkv_proj with tp 1, which means [4096, 6144]
python3 bench_gptq_marlin_int8.py --model_type llama3_8b --module qkv_proj --tp_size 1

# Profiler GEMM performance for llama3_8b qkv_proj with tp 2, which means [4096, 14336]
python3 bench_gptq_marlin_int8.py --model_type llama3_8b --module gate_up_proj --tp_size 2

# Profiler GEMM performance for all GEMMs of llama3_8b with shapes [4096, 4096], [4096, 6144], [4096, 28672], [14336, 4096]
python3 bench_gptq_marlin_int8.py --model_type llama3_8b --module all --tp_size 1
```

For example, for the second command, the program will automatically execute Marlin int8 type matrix computation for [batch, 4096] * [4096, 14336], where `batch` is selected from [4, 8, 12, ..., 8192], corresponding to 2048 matrix computations. For each batch, the program will run `ops.gptq_marlin_gemm` multiple times, measure the average kernel execution time on the GPU, calculate the actual achieved FLOPS for the matrix, and compare it to the theoretical performance of the RTX4090. A total of 2048 GEMM efficiency values will be obtained. The results will be saved in the profiler directory with the file format `Marlin_NVIDIA_A800_80GB_PCIe_vllm_0.6.5_K4096_N28672_gptq_int8.csv`, representing GPU type, vLLM version, matrix shape, precision, etc.