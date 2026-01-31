#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 1. 参数校验
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_ARGUMENT(out->ndim() == 2, "out must be a 2D tensor");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "out, gate and up must be contiguous");

    // 2. 设置设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    const size_t numel = out->numel();

    // 3. 设备分发
    switch(out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
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
