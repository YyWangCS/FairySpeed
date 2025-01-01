## 复现文章实验结果

### MFU对于矩阵shape的一致性

#### 获得矩阵GEMM的性能profiler

在GPU机器上，比如RTX4090，通过如下的命令来执行。

```python
# profiler gemm performance for llama3_8b qkv_proj, with tp 1, which means [4096, 6144]
python3 bench_llm_torch_gemm.py --model_type llama3_8b --module qkv_proj --tp_size 1


# profiler gemm performance for llama3_8b qkv_proj, with tp 2, which means [4096, 14336]
python3 bench_llm_torch_gemm.py --model_type llama3_8b --module gate_up_proj --tp_size 2

# profiler gemm performance for all gemms of llama3_8b, with shape [4096, 4096], [4096, 6144], [4096, 28672], [14336, 4096]
python3 bench_llm_torch_gemm.py --model_type llama3_8b --module all --tp_size 1
```

比如对于第二个命令，程序会自动执行[batch, 4096] * [4096, 14336]的float16类型矩阵计算，其中batch从[4, 8, 12, ..., 32768]中选取，一共对应8192个矩阵计算。对于每一个batch，通过程序来自动执行torch.matmul多次，并且统计GPU上的kernel执行平均时间，基于此来计算对应矩阵实际达到的FLOPS，然后和RTX4090的理论性能进行对比，一共得到8192个GEMM计算效率的值。最终结果会存在profiler目录下，文件格式是torchgemm_NVIDIA_GeForce_RTX_4090_torch_2.5.1+cu124_K4096_N14336_dtypefloat16.csv，分别代表GPU类型、torch版本、矩阵shape、精度等。对于[4096, 14336]这个shape在RTX4090的测试，我们提供了torch2.3和torch2.5.1的两份测试结果在profiler文件下供参考。

#### 分析GEMM计算效率异常的batch

执行如下的命令来分析异常GEMM计算效率对应的batch。

```python
# 只分析GEMM计算效率异常的batch，找到所有利用率比更小的子矩阵还低比较多的
python3 shape_utils_analyze.py --profiler ./profiler/torchgemm_NVIDIA_GeForce_RTX_4090_torch_2.5.1+cu124_K4096_N14336_dtypefloat16.csv

# 分析GEMM计算效率异常的batch，同时分析是否可以用split_gemm的方式通过两个利用率
python3 shape_utils_analyze.py --profiler ./profiler/torchgemm_NVIDIA_GeForce_RTX_4090_torch_2.5.1+cu124_K4096_N14336_dtypefloat16.csv --check_split_gemm
```

对于每一个batch N来说，如果在小于N的batch里面，[N ,4096]矩阵的计算效率小于小矩阵[N1, 4096] (N1 < N)的计算效率乘以一个阈值（比如0.85），则认为当前[N ,4096] shape矩阵的性能是异常的。

如果设置了--check_split_gemm，则会对于异常GEMM计算效率的batch，比如N，我们会搜索所有N1+N2=N的两个小矩阵[N1, 4096]和[N2, 4096]，然后找到对应两个矩阵的GEMM性能，将二者求和，如果二者之和明显小于batch=N对应的矩阵性能，我们称矩阵[N, 4096]为splitwise_gemm可优化矩阵，程序会自动输出所有的splitwise_gemm可优化矩阵。

## MFU对于量化计算的一致性

#### 获得Marlin GEMM的性能profiler

在GPU机器上，比如RTX4090，通过如下的命令来执行。

```python
# source /Envs/vllm_0.6.5/bin/bash

# profiler gemm performance for llama3_8b qkv_proj, with tp 1, which means [4096, 6144]
python3 bench_gptq_marlin_int8.py --model_type llama3_8b --module qkv_proj --tp_size 1


# profiler gemm performance for llama3_8b qkv_proj, with tp 2, which means [4096, 14336]
python3 bench_gptq_marlin_int8.py --model_type llama3_8b --module gate_up_proj --tp_size 2

# profiler gemm performance for all gemms of llama3_8b, with shape [4096, 4096], [4096, 6144], [4096, 28672], [14336, 4096]
python3 bench_gptq_marlin_int8.py --model_type llama3_8b --module all --tp_size 1
```

比如对于第二个命令，程序会自动执行[batch, 4096] * [4096, 14336]的Marlin int8 类型矩阵计算，其中batch从[4, 8, 12, ..., 8192]中选取，一共对应2048个矩阵计算。对于每一个batch，通过程序来自动执行ops.gptq_marlin_gemm多次，并且统计GPU上的kernel执行平均时间，基于此来计算对应矩阵实际达到的FLOPS，然后和RTX4090的理论性能进行对比，一共得到2048个GEMM计算效率的值。最终结果会存在profiler目录下，文件格式是Marlin_NVIDIA_A800_80GB_PCIe_vllm_0.6.5_K4096_N28672_gptq_int8.csv，分别代表GPU类型、vLLM版本、矩阵shape、精度等。