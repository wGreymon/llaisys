#include "argmax_nvidia.hpp"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"


namespace {

// Convert stored types to float for comparison
__device__ inline float to_float(float v) {
    return v;
}

__device__ inline float to_float(llaisys::fp16_t v) {
    union {
        __half h;
        uint16_t u;
    } x;
    x.u = v._v;
    return __half2float(x.h);
}

__device__ inline float to_float(llaisys::bf16_t v) {
    union {
        __nv_bfloat16 b;
        uint16_t u;
    } x;
    x.u = v._v;
    return __bfloat162float(x.b);
}

template <typename T>
__device__ inline T zero_value() {
    return T{0};
}

template <>
__device__ inline float zero_value<float>() {
    return 0.0f;
}

// Single-block argmax reduction over `numel` elements.
// Each thread processes a strided subset and we reduce in shared memory.
template <typename T>
__global__ void argmax_kernel(const T *vals, size_t numel, int64_t *out_idx, T *out_val) {
    extern __shared__ unsigned char smem[];
    T *s_vals = reinterpret_cast<T *>(smem);
    int64_t *s_idx = reinterpret_cast<int64_t *>(s_vals + blockDim.x);

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    if (numel == 0) {
        if (tid == 0) {
            *out_idx = 0;
            *out_val = zero_value<T>();
        }
        return;
    }

    // 1. 每个线程在自己的 strided 区间内找到局部最大值
    T best_val;
    int64_t best_idx = -1;

    size_t i = tid;
    if (i < numel) {
        best_val = vals[i];
        best_idx = static_cast<int64_t>(i);
        i += stride;
        for (; i < numel; i += stride) {
            float cur = to_float(vals[i]);
            float best = to_float(best_val);
            if (cur > best) {
                best_val = vals[i];
                best_idx = static_cast<int64_t>(i);
            }
        }
    } else {
        best_val = zero_value<T>();
    }

    s_vals[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    // 2. block 内规约
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            int64_t idx_other = s_idx[tid + offset];
            if (idx_other >= 0) {
                float v_self = to_float(s_vals[tid]);
                float v_other = to_float(s_vals[tid + offset]);
                if (s_idx[tid] < 0 || v_other > v_self) {
                    s_vals[tid] = s_vals[tid + offset];
                    s_idx[tid] = idx_other;
                }
            }
        }
        __syncthreads();
    }

    // 3. 写出结果
    if (tid == 0) {
        if (s_idx[0] < 0) {
            *out_idx = 0;
            *out_val = zero_value<T>();
        } else {
            *out_idx = s_idx[0];
            *out_val = s_vals[0];
        }
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void argmax(int64_t *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    if (numel == 0) {
        // 在 device 上直接写一个默认值
        // 这里假定 max_idx/max_val 已在 device 上分配
        switch (type) {
        case LLAISYS_DTYPE_F32: {
            float zero = 0.0f;
            CUDA_CHECK(cudaMemcpy(max_val, &zero, sizeof(float), cudaMemcpyHostToDevice));
            break;
        }
        case LLAISYS_DTYPE_F16: {
            llaisys::fp16_t zero{0};
            CUDA_CHECK(cudaMemcpy(max_val, &zero, sizeof(llaisys::fp16_t), cudaMemcpyHostToDevice));
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            llaisys::bf16_t zero{0};
            CUDA_CHECK(cudaMemcpy(max_val, &zero, sizeof(llaisys::bf16_t), cudaMemcpyHostToDevice));
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
        int64_t idx_zero = 0;
        CUDA_CHECK(cudaMemcpy(max_idx, &idx_zero, sizeof(int64_t), cudaMemcpyHostToDevice));
        return;
    }

    constexpr int block_size = 256;
    dim3 block(block_size);
    dim3 grid(1);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        size_t shmem = block_size * (sizeof(float) + sizeof(int64_t));
        argmax_kernel<<<grid, block, shmem>>>(reinterpret_cast<const float *>(vals),
                                              numel,
                                              max_idx,
                                              reinterpret_cast<float *>(max_val));
        break;
    }
    case LLAISYS_DTYPE_F16: {
        size_t shmem = block_size * (sizeof(llaisys::fp16_t) + sizeof(int64_t));
        argmax_kernel<<<grid, block, shmem>>>(reinterpret_cast<const llaisys::fp16_t *>(vals),
                                              numel,
                                              max_idx,
                                              reinterpret_cast<llaisys::fp16_t *>(max_val));
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        size_t shmem = block_size * (sizeof(llaisys::bf16_t) + sizeof(int64_t));
        argmax_kernel<<<grid, block, shmem>>>(reinterpret_cast<const llaisys::bf16_t *>(vals),
                                              numel,
                                              max_idx,
                                              reinterpret_cast<llaisys::bf16_t *>(max_val));
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace llaisys::ops::nvidia
