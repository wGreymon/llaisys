#include "linear_cpu.hpp"

#include "../../../utils.hpp"
#include "llaisys.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// 通用内核：按外积方式实现 Y = X W^T + b
// X: [M, K], W: [N, K], b: [N], Y: [M, N]
// out, in, weight, bias 都已经按类型 T 解释
template <typename T>
void linear_(T *out,
             const T *in,
             const T *weight,
             const T *bias,
             size_t M,
             size_t N,
             size_t K) {
    // 全部使用 float 做累加，最后 cast 回 T，避免 f16/bf16 精度丢失
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++){
            float sum = 0.0f;      // 为了保证精度先用float计算
            if (bias != nullptr) {
                sum += llaisys::utils::cast<float>(bias[j]);
            }
            // 对于fp16和bf16进行强转，以保证精度
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                for (size_t k = 0; k < K; k++) {
                    float data_x = llaisys::utils::cast<float>(in[i * K + k]);
                    float data_w = llaisys::utils::cast<float>(weight[j * K + k]);
                    sum += data_x * data_w;
                }
                out[i * N + j] = llaisys::utils::cast<T>(sum);
            } else {
                for (size_t k = 0; k < K; k++) {
                    sum += in[i * K + k] * weight[j * K + k];
                }
                out[i * N + j] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t type,
            size_t M,
            size_t N,
            size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<fp16_t *>(out), 
                       reinterpret_cast<const fp16_t *>(in),
                       reinterpret_cast<const fp16_t *>(weight), 
                       reinterpret_cast<const fp16_t *>(bias), 
                       M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<bf16_t *>(out), 
                       reinterpret_cast<const bf16_t*>(in), 
                       reinterpret_cast<const bf16_t *>(weight), 
                       reinterpret_cast<const bf16_t *>(bias), 
                       M, N, K);
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), 
                       reinterpret_cast<const float *>(in), 
                       reinterpret_cast<const float *>(weight), 
                       reinterpret_cast<const float *>(bias), 
                       M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

