#include "linear_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"
#include <cstddef>

namespace {

// cpu_time:
// Torch time: 30.81158 ms
// LLAISYS time: 401.65733 ms
// Torch time: 140.67506 ms
// LLAISYS time: 3028.21840 ms
// Torch time: 142.86126 ms
// LLAISYS time: 2105.92961 ms

// naive：使用global memory实现
// in[M, K], weight[N, K], bias[N], out[M, N]
// v1_time:
// Torch time: 2.06076 ms
// LLAISYS time: 82.52521 ms
// Torch time: 0.58656 ms
// LLAISYS time: 82.01252 ms
// Torch time: 0.59076 ms
// LLAISYS time: 82.44525 ms
template <typename T>
__global__ void sgemm_v1(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {
  int midx = blockIdx.y * blockDim.y + threadIdx.y;
  int nidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (midx >= M || nidx >= N) {
    return;
  }

  float sum = 0.0f;
  if (bias != nullptr) {
    sum += to_float(bias[nidx]);
  }

  for (int k = 0; k < K; k++) {
    sum += to_float(in[midx * K + k]) * to_float(weight[nidx * K + k]);
  }

  out[midx * N + nidx] = from_float<T>(sum);
}

// v2：使用sharead memory实现，显著降低对global memory的访问次数
// v2_time:
// Torch time: 5.63606 ms
// LLAISYS time: 43.84619 ms
// Torch time: 0.60475 ms
// LLAISYS time: 49.69251 ms
// Torch time: 0.60049 ms
// LLAISYS time: 50.35990 ms
template <typename T>
__global__ void sgemm_v2(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {
  constexpr int bm = 16;
  constexpr int bn = 16;
  constexpr int bk = 16;

  // NVIDIA GeForce GTX 4060 sharedMemPerBlock is 48KB = 48*1024B =
  // 49152B(0xc000) 1 float takes 4 Bytes, so (BM*BK + BK*BN) should <=
  // 48*1024/4 = 12288
  __shared__ float in_shared[bm * bk];
  __shared__ float weight_shared[bn * bk];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * bm + ty;
  int col = bx * bn + tx;

  float sum = 0.0f;
  if (bias != nullptr && col < N) {
    sum += to_float(bias[col]);
  }

  for (int k = 0; k < K; k += bk) {
    // 加载in：global memory -> shared memory
    if (row < M && (k + tx) < K) {
      in_shared[ty * bk + tx] = to_float(in[row * K + k + tx]);
    } else {
      in_shared[ty * bk + tx] = 0.0f;
    }

    // 加载weight
    if (col < N && (k + ty) < K) {
      weight_shared[tx * bk + ty] = to_float(weight[col * K + k + ty]);
    } else {
      weight_shared[tx * bk + ty] = 0.0f;
    }

    __syncthreads();

    // 在shared mem上进行当前bk的累加
    //// C[row, col] += sum_{k=0..bk-1} A[row, k+i] * W[col, k0+i]
    for (int i = 0; i < bk; i++) {
      sum += to_float(in_shared[ty * bk + i]) *
             to_float(weight_shared[tx * bk + i]);
    }
    __syncthreads();
  }

  if (by * bm + ty < M && bx * bn + tx < N) {
    out[row * N + col] = from_float<T>(sum);
  }
}

// v3：block tile 32x32 + thread tile 4x4，block 内 (8,8)=64 线程
template <typename T>
__global__ void sgemm_v3(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {
  constexpr int bm = 32;
  constexpr int bn = 32;
  constexpr int bk = 16;
  constexpr int TM = 4;
  constexpr int TN = 4;

  __shared__ float in_shared[bm * bk];
  __shared__ float weight_shared[bn * bk];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;  // [0, bn/TN) = [0, 8)
  int ty = threadIdx.y;  // [0, bm/TM) = [0, 8)

  float sum[TM][TN];
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      int col = bx * bn + tx * TN + j;
      sum[i][j] = (bias != nullptr && col < (int)N) ? to_float(bias[col]) : 0.0f;
    }
  }

  for (int k = 0; k < (int)K; k += bk) {
    // 64 线程协作加载 in_shared[32][16]：每线程 8 个，coalesced
    int linear = ty * (bn / TN) + tx;
    int r = (linear * 8) / bk;
    int c = (linear * 8) % bk;
    for (int j = 0; j < 8; j++) {
      int gr = by * bm + r;
      int gc = k + c + j;
      if (gr < (int)M && gc < (int)K) {
        in_shared[r * bk + c + j] = to_float(in[gr * (int)K + gc]);
      } else {
        in_shared[r * bk + c + j] = 0.0f;
      }
    }
    // 协作加载 weight_shared[32][16]
    for (int j = 0; j < 8; j++) {
      int wc = (linear * 8 + j) / bk;
      int wr = (linear * 8 + j) % bk;
      int gr = bx * bn + wc;
      int gc = k + wr;
      if (gr < (int)N && gc < (int)K) {
        weight_shared[wc * bk + wr] = to_float(weight[gr * (int)K + gc]);
      } else {
        weight_shared[wc * bk + wr] = 0.0f;
      }
    }

    __syncthreads();

    for (int kk = 0; kk < bk; kk++) {
      for (int i = 0; i < TM; i++) {
        float a = in_shared[(ty * TM + i) * bk + kk];
        for (int j = 0; j < TN; j++) {
          sum[i][j] += a * weight_shared[(tx * TN + j) * bk + kk];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      int row = by * bm + ty * TM + i;
      int col = bx * bn + tx * TN + j;
      if (row < (int)M && col < (int)N) {
        out[row * (int)N + col] = from_float<T>(sum[i][j]);
      }
    }
  }
}

template <typename T>
__global__ void sgemm_v4(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {}

template <typename T>
__global__ void sgemm_v5(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {}

template <typename T>
__global__ void sgemm_v6(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {}

template <typename T>
__global__ void sgemm_v7(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {}

} // namespace

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type, size_t M, size_t N,
            size_t K) {
  // v3: block tile 32x32, thread tile 4x4 -> (8,8) threads per block
  constexpr dim3 block_size(8, 8);
  dim3 grid_size(CEIL(N, 32), CEIL(M, 32));

  switch (type) {
  case LLAISYS_DTYPE_F32:
    sgemm_v3<float><<<grid_size, block_size>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
        reinterpret_cast<const float *>(weight),
        reinterpret_cast<const float *>(bias), M, N, K);
    break;
  case LLAISYS_DTYPE_F16:
    sgemm_v3<half><<<grid_size, block_size>>>(
        reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
        reinterpret_cast<const half *>(weight),
        reinterpret_cast<const half *>(bias), M, N, K);
    break;
  case LLAISYS_DTYPE_BF16:
    sgemm_v3<__nv_bfloat16><<<grid_size, block_size>>>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        reinterpret_cast<const __nv_bfloat16 *>(in),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        reinterpret_cast<const __nv_bfloat16 *>(bias), M, N, K);
    break;
  default:
    EXCEPTION_UNSUPPORTED_DATATYPE(type);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace llaisys::ops::nvidia