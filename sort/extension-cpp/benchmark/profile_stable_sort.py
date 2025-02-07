import torch
import random
import numpy as np
import extension_cpp
from torch.profiler import profile, record_function, ProfilerActivity

def enforce_reproduce():
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_random_input(num_inputs):
    input_tensors = [torch.randint(0, num_inputs, (num_inputs, ), dtype=torch.int32).cuda() for i in range(3)]
    warmup_input = input_tensors[0]
    stable_sort_input = input_tensors[1]
    stable_sort_opaque_input = input_tensors[2]
    return warmup_input, stable_sort_input, stable_sort_opaque_input

def stable_sort(input_tensor):
    values, indices = extension_cpp.ops.stable_sort(input_tensor)
    return values, indices

def stable_sort_opaque(input_tensor):
    values, indices = extension_cpp.ops.stable_sort_opaque(input_tensor)
    return values, indices

def warmup(warmup_input):
    #warmup
    for i in range(3):
        stable_sort(warmup_input)
        stable_sort_opaque(warmup_input)
    torch.cuda.synchronize()

def profile_stable_sort(stable_sort_input):
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        stable_sort(stable_sort_input)
        torch.cuda.synchronize()

    for event in prof.key_averages():
        if "DeviceRadixSortOnesweep" in event.key:
            stable_sort_time = event.self_device_time_total / 4
            break
    print(f"DeviceRadixSortOnesweepKernel latency in stable_sort is {stable_sort_time:.2f} us")


def profile_stable_sort_opaque(stable_sort_opaque_input):
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        stable_sort_opaque(stable_sort_opaque_input)
        torch.cuda.synchronize()

    for event in prof.key_averages():
        if "DeviceRadixSortOnesweep" in event.key:
            stable_sort_opaque_time = event.self_device_time_total / 4
            break
    print(f"DeviceRadixSortOnesweepKernel latency in stable_sort_opaque is {stable_sort_opaque_time:.2f} us")


if __name__ == "__main__":
    enforce_reproduce()
    
    num_inputs = 100000000
    warmup_input, stable_sort_input, stable_sort_opaque_input = get_random_input(num_inputs)

    profile_stable_sort(stable_sort_input)
    profile_stable_sort_opaque(stable_sort_opaque_input)
