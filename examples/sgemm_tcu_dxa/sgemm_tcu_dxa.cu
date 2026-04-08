// sgemm_tcu_dxa.cu — SGEMM with cp.async (→DXA on Vortex) + wmma (→TCU)
//
// Standard CUDA APIs only:
//   cp.async: global → shared memory async copy
//     - NVIDIA GPU: hardware async copy unit
//     - Vortex: CuPBoP translates to DXA hardware DMA
//   wmma: tensor core matrix multiply
//     - NVIDIA GPU: Tensor Core
//     - Vortex: CuPBoP translates to TCU instruction
//
// A: M×K (fp16, row-major), B: K×N (fp16, row-major), C: M×N (fp32)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// cp.async: copy 16 bytes from global to shared memory
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
#ifdef __CUDA_ARCH__
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#ifdef __CUDA_ARCH__
    asm volatile("cp.async.commit_group;\n" :::);
#endif
}

__device__ __forceinline__ void cp_async_wait() {
#ifdef __CUDA_ARCH__
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// Kernel: cp.async tiles into shared memory, wmma computes
__global__ void sgemm_tcu_dxa_kernel(const half* __restrict__ A,
                                      const half* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
    int tile_row = blockIdx.y * WMMA_M;
    int tile_col = blockIdx.x * WMMA_N;

    // Shared memory for one A tile and one B tile
    // A tile: 16×16 halfs = 512 bytes, B tile: 16×16 halfs = 512 bytes
    __shared__ half shA[WMMA_M * WMMA_K];
    __shared__ half shB[WMMA_K * WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    wmma::fill_fragment(fragC, 0.0f);

    int tid = threadIdx.x;
    // 16×16 = 256 halfs = 512 bytes. 32 threads, each copies 16 bytes (8 halfs)
    int elems_per_thread = (WMMA_M * WMMA_K) / blockDim.x; // 8

    for (int k = 0; k < K; k += WMMA_K) {
        // cp.async: load A tile [tile_row..+16, k..+16] into shA
        for (int i = 0; i < elems_per_thread; i += 8) {
            int elem_base = tid * elems_per_thread + i;
            int r = elem_base / WMMA_K;
            int c = elem_base % WMMA_K;
            int src_row = tile_row + r;
            int src_col = k + c;
            if (src_row < M && src_col + 7 < K) {
                cp_async_16(&shA[elem_base], &A[src_row * K + src_col]);
            }
        }

        // cp.async: load B tile [k..+16, tile_col..+16] into shB
        for (int i = 0; i < elems_per_thread; i += 8) {
            int elem_base = tid * elems_per_thread + i;
            int r = elem_base / WMMA_N;
            int c = elem_base % WMMA_N;
            int src_row = k + r;
            int src_col = tile_col + c;
            if (src_row < K && src_col + 7 < N) {
                cp_async_16(&shB[elem_base], &B[src_row * N + src_col]);
            }
        }

        cp_async_commit();
        cp_async_wait();
        __syncthreads();

        // Load from shared memory and compute
        wmma::load_matrix_sync(fragA, shA, WMMA_K);
        wmma::load_matrix_sync(fragB, shB, WMMA_N);
        wmma::mma_sync(fragC, fragA, fragB, fragC);

        __syncthreads();
    }

    // Store result
    float* tileC = C + tile_row * N + tile_col;
    wmma::store_matrix_sync(tileC, fragC, N, wmma::mem_row_major);
}

// CPU reference
void gemm_cpu(const half* A, const half* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++)
                sum += __half2float(A[i * K + kk]) * __half2float(B[kk * N + j]);
            C[i * N + j] = sum;
        }
}

int main(int argc, char** argv) {
    int M = WMMA_M, N = WMMA_N, K = WMMA_K;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            int n = atoi(argv[++i]);
            if (n > 0 && n % WMMA_M == 0) { M = N = K = n; }
            else { fprintf(stderr, "Size must be multiple of %d\n", WMMA_M); return -1; }
        }
    }
    printf("SGEMM TCU+DXA (cp.async + wmma): M=%d, N=%d, K=%d\n", M, N, K);

    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(float);

    half* h_A = (half*)malloc(sizeA);
    half* h_B = (half*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)(rand() % 5) / 5.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half((float)(rand() % 5) / 5.0f);
    memset(h_C, 0, sizeC);
    gemm_cpu(h_A, h_B, h_C_ref, M, N, K);

    half *d_A, *d_B;
    float* d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeC);

    dim3 grid(N / WMMA_N, M / WMMA_M);
    dim3 block(32, 1);
    sgemm_tcu_dxa_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(h_C[i] - h_C_ref[i]);
        if (diff / fmaxf(fabsf(h_C_ref[i]), 1.0f) > 1e-2f) {
            if (errors < 10) printf("Mismatch [%d]: got %f, expected %f\n", i, h_C[i], h_C_ref[i]);
            errors++;
        }
    }
    printf("%s: %d/%d\n", errors ? "FAILED!" : "PASSED!", M * N - errors, M * N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return errors ? 1 : 0;
}
