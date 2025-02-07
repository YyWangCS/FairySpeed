
#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// radix_sort_pairs doesn't interact with value_t other than to copy
// the data, so we can save template instantiations by reinterpreting
// it as an opaque type.
template <int N> struct alignas(N) OpaqueType { char data[N]; };

std::tuple<at::Tensor, at::Tensor> stable_sort_cuda(const at::Tensor& input) {
    using key_t=int32_t;
    using value_t = int64_t;
    using opaque_t = value_t;

    int64_t num_inp = input.numel();
    at::Tensor sorted = at::empty(input.sizes(), input.options());

    auto options = input.options().dtype(at::kLong);
    at::Tensor range = at::arange(0, num_inp, options);
    at::Tensor sorted_indices = at::empty({num_inp}, options);

    const opaque_t* indices_in = reinterpret_cast<const opaque_t*>(range.const_data_ptr<value_t>());
    opaque_t* indices_out = reinterpret_cast<opaque_t*>(sorted_indices.mutable_data_ptr<value_t>());

    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                    input.const_data_ptr<key_t>(), sorted.mutable_data_ptr<key_t>(), // 输入和输出key
                                    indices_in, indices_out,  // 输入和输出value
                                    num_inp);

    auto caching_allocator = c10::cuda::CUDACachingAllocator::get();
    auto temp_storage = caching_allocator->allocate(temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(temp_storage.get(), temp_storage_bytes,
                                    input.const_data_ptr<key_t>(), sorted.mutable_data_ptr<key_t>(), // 输入和输出key
                                    indices_in, indices_out,  // 输入和输出value
                                    num_inp);
    
    return std::make_tuple(sorted, sorted_indices);
}


std::tuple<at::Tensor, at::Tensor> stable_sort_opaque_cuda(const at::Tensor& input) {
    using key_t=int32_t;
    using value_t = int64_t;
    using opaque_t = OpaqueType<sizeof(value_t)>;

    int64_t num_inp = input.numel();
    at::Tensor sorted = at::empty(input.sizes(), input.options());

    auto options = input.options().dtype(at::kLong);
    at::Tensor range = at::arange(0, num_inp, options);
    at::Tensor sorted_indices = at::empty({num_inp}, options);

    const opaque_t* indices_in = reinterpret_cast<const opaque_t*>(range.const_data_ptr<value_t>());
    opaque_t* indices_out = reinterpret_cast<opaque_t*>(sorted_indices.mutable_data_ptr<value_t>());

    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                    input.const_data_ptr<key_t>(), sorted.mutable_data_ptr<key_t>(), // 输入和输出key
                                    indices_in, indices_out,  // 输入和输出value
                                    num_inp);

    auto caching_allocator = c10::cuda::CUDACachingAllocator::get();
    auto temp_storage = caching_allocator->allocate(temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(temp_storage.get(), temp_storage_bytes,
                                    input.const_data_ptr<key_t>(), sorted.mutable_data_ptr<key_t>(), // 输入和输出key
                                    indices_in, indices_out,  // 输入和输出value
                                    num_inp);
    
    return std::make_tuple(sorted, sorted_indices);
}


// Registers CUDA implementations for stable_sort
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("stable_sort", &stable_sort_cuda);
  m.impl("stable_sort_opaque", &stable_sort_opaque_cuda);
}