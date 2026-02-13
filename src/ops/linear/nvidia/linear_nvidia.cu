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

// v2：使用sharead memory实现，显著降低对global memory的访问次数实现加速
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
    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int BK = 16;

    // NVIDIA GeForce GTX 4060 sharedMemPerBlock is 48KB = 48*1024B =
    // 49152B(0xc000) 1 float takes 4 Bytes, so (BM*BK + BK*BN) should <=
    // 48*1024/4 = 12288
    __shared__ float in_shared[BM * BK];
    __shared__ float weight_shared[BN * BK];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BM + ty;
    int col = bx * BN + tx;

    float sum = 0.0f;
    if (bias != nullptr && col < N) {
        sum += to_float(bias[col]);
    }

    for (int k = 0; k < K; k += BK) {
        // 加载in：global memory -> shared memory
        if (row < M && (k + tx) < K) {
            in_shared[ty * BK + tx] = to_float(in[row * K + k + tx]);
        } else {
            in_shared[ty * BK + tx] = 0.0f;
        }

        // 加载weight
        if (col < N && (k + ty) < K) {
            weight_shared[tx * BK + ty] = to_float(weight[col * K + k + ty]);
        } else {
            weight_shared[tx * BK + ty] = 0.0f;
        }

        __syncthreads();

        // 在shared mem上进行当前bk的累加
        //// C[row, col] += sum_{k=0..BK-1} A[row, k+i] * W[col, k0+i]
        for (int i = 0; i < BK; i++) {
            sum += to_float(in_shared[ty * BK + i]) * to_float(weight_shared[tx * BK + i]);
        }
        __syncthreads();
    }

    if (by * BM + ty < M && bx * BN + tx < N) {
        out[row * N + col] = from_float<T>(sum);
    }
}

// v3：block tile 32x32 + thread tile 4x4，block 内 (8,8)=64 线程
// 每个线程计算一小块(4*4)，且数据复用加强，能显著增加计算强度
// v3_time:
// Torch time: 2.00178 ms
// LLAISYS time: 20.16289 ms
// Torch time: 0.56751 ms
// LLAISYS time: 20.26551 ms
// Torch time: 0.56799 ms
// LLAISYS time: 20.25749 ms
template <typename T>
__global__ void sgemm_v3(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    __shared__ float in_shared[BM * BK];
    __shared__ float weight_shared[BN * BK];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum[TM][TN];
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int col = bx * BN + tx * TN + j;
            sum[i][j] = (bias != nullptr && col < (int)N) ? to_float(bias[col]) : 0.0f;
        }
    }

    for (int k = 0; k < K; k += BK) {
        int tid = ty * blockDim.x + tx;
        int nthread = blockDim.x * blockDim.y;
        // 64 线程协作加载 in_shared[32][16]：每线程 8 个，coalesced
        for (int e = tid; e < BM * BK; e += nthread) {
            int r = e / BK;
            int c = e % BK;

            int global_r = by * BM + r;
            int global_c = k + c;

            in_shared[r * BK + c] = (global_r < M && global_c < K) ? to_float(in[global_r * K + global_c]) : 0.0f;
        }

        // load weight_shared[32][16]
        for (int e = tid; e < BN * BK; e += nthread) {
            int r = e / BK;
            int c = e % BK;

            int global_r = bx * BN + r;
            int global_c = k + c;

            weight_shared[r * BK + c] = (global_r < N && global_c < K) ? to_float(weight[global_r * K + global_c]) : 0.0f;
        }

        __syncthreads();

        // compute
        for (int kk = 0; kk < BK; kk++) {
            for (int i = 0; i < TM; i++) {
                float x = in_shared[(ty * TM + i) * BK + kk];
                for (int j = 0; j < TN; j++) {
                    sum[i][j] += x * weight_shared[(tx * TN + j) * BK + kk];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = by * BM + ty * TM + i;
            int col = bx * BN + tx * TN + j;
            if (row < (int)M && col < (int)N) {
                out[row * (int)N + col] = from_float<T>(sum[i][j]);
            }
        }
    }
}

// v4:将shared_mem上的数据搬运到reg上，计算时减少对shared_mem的访问
// v4_time:
// Torch time: 2.00347 ms
// LLAISYS time: 14.46333 ms
// Torch time: 0.56831 ms
// LLAISYS time: 14.59107 ms
// Torch time: 0.56920 ms
// LLAISYS time: 14.59146 ms
template <typename T>
__global__ void sgemm_v4(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int block_row_base = by * BM;
    int block_col_base = bx * BN;
    int out_row_base = by * BM + ty * TM;
    int out_col_base = bx * BN + tx * TN;
    int nthread = blockDim.x * blockDim.y;

    __shared__ float in_shared[BM][BK];
    __shared__ float weight_shared[BN][BK];

    float sum[TM][TN] = {0.0f};
    float a_frag[TM];
    float b_frag[TN];

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            sum[i][j] = (bias != nullptr && out_col_base + j < N) ? to_float(bias[out_col_base + j]) : 0.0f;
        }
    }

    for (int k = 0; k < K; k += BK) {
        // load in
        for (int i = tid; i < BM * BK; i += nthread) {
            int r = i / BK;
            int c = i % BK;
            in_shared[r][c] = ((block_row_base + r) < M && (k + c) < K) ? to_float(in[(block_row_base + r) * K + (k + c)]) : 0.0f;
        }

        // load weight
        for (int i = tid; i < BN * BK; i += nthread) {
            int r = i / BK;
            int c = i % BK;
            weight_shared[r][c] = ((block_col_base + r) < N && (k + c) < K) ? to_float(weight[(block_col_base + r) * K + (k + c)]) : 0.0f;
        }

        __syncthreads();

        for (int kk = 0; kk < BK; kk++) {
            // load：shared_mem to reg
            for (int i = 0; i < TM; i++) {
                a_frag[i] = in_shared[ty * TM + i][kk];
            }

            for (int j = 0; j < TN; j++) {
                b_frag[j] = weight_shared[tx * TN + j][kk];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    sum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int r = by * BM + ty * TM + i;
            int c = bx * BN + tx * TN + j;
            if (r < (int)M && c < (int)N) {
                out[r * (int)N + c] = from_float<T>(sum[i][j]);
            }
        }
    }
}

// v5（借鉴 matmul4/matmul5 思路）：
// 1) global->shared 使用 float4 向量化加载
// 2) shared 中转置存储为 [BK, BM]/[BK, BN]，便于 thread-tile 连续读取
// 3) shared->register 用 float4 一次取 4 个元素，继续提高复用
// 4) 保留边界检查与尾块标量回退，保证通用输入尺寸正确
// Torch time: 2.01833 ms
// LLAISYS time: 4.00644 ms
__global__ void sgemm_v5_float(float *out, const float *in, const float *weight, const float *bias,
                               size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int VEC = 4;
    constexpr int BKV = (BK + VEC - 1) / VEC; // number of float4 groups along K in one BK-tile

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int nthread = blockDim.x * blockDim.y;

    const int block_row_base = by * BM;
    const int block_col_base = bx * BN;
    const int out_row_base = by * BM + ty * TM;
    const int out_col_base = bx * BN + tx * TN;

    // Transposed shared tiles:
    // A_tile[BM, BK] -> As_t[BK, BM], W_tile[BN, BK] -> Ws_t[BK, BN]
    __shared__ float As_t[BK][BM];
    __shared__ float Ws_t[BK][BN];

    float sum[TM][TN] = {0.0f};

    // Initialize accumulators with bias.
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            const int out_c = out_col_base + j;
            sum[i][j] = (bias != nullptr && out_c < static_cast<int>(N)) ? bias[out_c] : 0.0f;
        }
    }

    for (int k0 = 0; k0 < static_cast<int>(K); k0 += BK) {
        // Step-1: vectorized load A tile + transpose into As_t.
        for (int idx = tid; idx < BM * BKV; idx += nthread) {
            const int r = idx / BKV;
            const int vc = idx % BKV;
            const int c = vc * VEC;
            const int gr = block_row_base + r;
            const int gc = k0 + c;

            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < static_cast<int>(M)) {
                const size_t base = static_cast<size_t>(gr) * K + static_cast<size_t>(gc);
                if (gc + (VEC - 1) < static_cast<int>(K) && (base % VEC) == 0) {
                    v = *reinterpret_cast<const float4 *>(in + base);
                } else {
                    if (gc + 0 < static_cast<int>(K)) {
                        v.x = in[base + 0];
                    }
                    if (gc + 1 < static_cast<int>(K)) {
                        v.y = in[base + 1];
                    }
                    if (gc + 2 < static_cast<int>(K)) {
                        v.z = in[base + 2];
                    }
                    if (gc + 3 < static_cast<int>(K)) {
                        v.w = in[base + 3];
                    }
                }
            }
            if (c + 0 < BK) {
                As_t[c + 0][r] = v.x;
            }
            if (c + 1 < BK) {
                As_t[c + 1][r] = v.y;
            }
            if (c + 2 < BK) {
                As_t[c + 2][r] = v.z;
            }
            if (c + 3 < BK) {
                As_t[c + 3][r] = v.w;
            }
        }

        // Step-2: vectorized load W tile + transpose into Ws_t.
        for (int idx = tid; idx < BN * BKV; idx += nthread) {
            const int r = idx / BKV;
            const int vc = idx % BKV;
            const int c = vc * VEC;
            const int gr = block_col_base + r;
            const int gc = k0 + c;

            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < static_cast<int>(N)) {
                const size_t base = static_cast<size_t>(gr) * K + static_cast<size_t>(gc);
                if (gc + (VEC - 1) < static_cast<int>(K) && (base % VEC) == 0) {
                    v = *reinterpret_cast<const float4 *>(weight + base);
                } else {
                    if (gc + 0 < static_cast<int>(K)) {
                        v.x = weight[base + 0];
                    }
                    if (gc + 1 < static_cast<int>(K)) {
                        v.y = weight[base + 1];
                    }
                    if (gc + 2 < static_cast<int>(K)) {
                        v.z = weight[base + 2];
                    }
                    if (gc + 3 < static_cast<int>(K)) {
                        v.w = weight[base + 3];
                    }
                }
            }
            if (c + 0 < BK) {
                Ws_t[c + 0][r] = v.x;
            }
            if (c + 1 < BK) {
                Ws_t[c + 1][r] = v.y;
            }
            if (c + 2 < BK) {
                Ws_t[c + 2][r] = v.z;
            }
            if (c + 3 < BK) {
                Ws_t[c + 3][r] = v.w;
            }
        }

        __syncthreads();

        // Step-3: compute using float4 shared->register fetch.
#pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            const float4 a4 = *reinterpret_cast<const float4 *>(&As_t[kk][ty * TM]); // TM=4
            const float4 b4 = *reinterpret_cast<const float4 *>(&Ws_t[kk][tx * TN]); // TN=4
            const float a_frag[TM] = {a4.x, a4.y, a4.z, a4.w};
            const float b_frag[TN] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
            for (int i = 0; i < TM; i++) {
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    sum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // Step-4: guarded write-back.
    for (int i = 0; i < TM; i++) {
        const int out_r = out_row_base + i;
        if (out_r >= static_cast<int>(M)) {
            continue;
        }
        for (int j = 0; j < TN; j++) {
            const int out_c = out_col_base + j;
            if (out_c < static_cast<int>(N)) {
                out[out_r * static_cast<int>(N) + out_c] = sum[i][j];
            }
        }
    }
}

__global__ void sgemm_v5_half(half *out, const half *in, const half *weight, const half *bias,
                              size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
}

__global__ void sgemm_v5_bfloat16(__nv_bfloat16 *out, const __nv_bfloat16 *in, const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
                                  size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
}

template <typename T>
__global__ void sgemm_v6(T *out, const T *in, const T *weight, const T *bias,
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
        sgemm_v5_float<<<grid_size, block_size>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias), M, N, K);
        break;
    case LLAISYS_DTYPE_F16:
        sgemm_v4<half><<<grid_size, block_size>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
            reinterpret_cast<const half *>(weight),
            reinterpret_cast<const half *>(bias), M, N, K);
        break;
    case LLAISYS_DTYPE_BF16:
        sgemm_v4<__nv_bfloat16><<<grid_size, block_size>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            reinterpret_cast<const __nv_bfloat16 *>(bias), M, N, K);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
