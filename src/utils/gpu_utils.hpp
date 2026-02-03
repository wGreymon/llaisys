#if defined(ENABLE_NVIDIA_API) && defined(__CUDACC__)

#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define LOAD_FLOAT4(value) *(reinterpret_cast<const float4*>(&value))
#define STORE_FLOAT4(value) *(reinterpret_cast<float4*>(&value))
#define LOAD_HALF2(value) *(reinterpret_cast<const half2*>(&value))
#define STORE_HALF2(value) *(reinterpret_cast<half2 *>(&(value)))
#define LOAD_BFLOAT2(value) *(reinterpret_cast<const __nv_bfloat162*>(&value))
#define STORE_BFLOAT2(value) *(reinterpret_cast<__nv_bfloat162*>(&value))

#define CEIL(x, y) ((x + y - 1) / y)

#define CUDA_CHECK(err) _cudaCheck(err, __FILE__, __LINE__)
inline void _cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

#endif // ENABLE_NVIDIA_API