#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "./cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 1. 检查张量所在设备
    CHECK_SAME_DEVICE(out, index, weight);
    
    // 2. 检查张量形状
    CHECK_ARGUMENT(index->ndim() == 1, "index must be a 1D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "weight must be a 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "out must be a 2D tensor");
    // 索引的数量就是输出的行数
    CHECK_ARGUMENT(index->numel() == out->shape()[0], "index must have the same number of elements as the first dimension of out");
    // 权重和输出的维度相同
    CHECK_ARGUMENT(weight->shape()[1] == out->shape()[1], "weight must have the same number of rows as the second dimension of out");
    // 索引的类型设为int64，与pytorch对齐
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "index must be a 64-bit integer tensor");
    // 检测 index 的值是否在权重范围内 [0, weight->shape()[0])
    {
        const auto *idx_data = reinterpret_cast<const int64_t *>(index->data());
        size_t idx_numel = index->numel();
        size_t vocab_size = weight->shape()[0];
        for (size_t i = 0; i < idx_numel; ++i) {
            CHECK_ARGUMENT(idx_data[i] >= 0
                               && static_cast<size_t>(idx_data[i]) < vocab_size,
                           "index must be in the range of weight");
        }
    }
    // 权重和输出的数据类型相同
    CHECK_ARGUMENT(weight->dtype() == out->dtype(), "weight and out must have the same data type");
    // 索引、权重和输出必须连续
    ASSERT(index->isContiguous() && weight->isContiguous() && out->isContiguous(), "index, weight and out must be contiguous");

    // 3. 设置设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 4. 设备分发
    size_t index_numel = index->numel();
    size_t embedding_dim = weight->shape()[1];

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 需要传入index_numel和embedding_dim，因为传入类型为std::byte*，丢失shape信息
        return cpu::embedding(out->data(),
                              index->data(),
                              weight->data(),
                              out->dtype(),
                              index_numel,
                              embedding_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
