// sgemm_tcu_dxa.cu — SGEMM with DXA tile loading + TCU WGMMA compute
//
// Full hardware-accelerated GEMM pipeline:
//   DXA: global → shared memory tile transfer (hardware DMA)
//   WGMMA: TCU reads data from memory using descriptor metadata (not register loads)
//
// A: M×K (fp16), B: K×N (fp16), C: M×N (fp32)
// Tile: 16×16×16 (wmma m16n16k16)

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

// Host-side DXA descriptor programming
extern "C" void cudaDxaProgramDesc2D(unsigned int slot, unsigned long long base_addr,
                                      unsigned int size0, unsigned int size1,
                                      unsigned int stride0_bytes,
                                      unsigned int tile0, unsigned int tile1,
                                      unsigned int elem_bytes);

// DXA device-side API stubs
__device__ void __vx_dxa_issue_2d(unsigned int, unsigned int, void*, unsigned int, unsigned int) {}
__device__ void __vx_barrier_arrive_and_wait(unsigned int, unsigned int) {}
__device__ unsigned int __vx_dxa_barrier_id(unsigned int) { return 0; }
__device__ void* __vx_local_mem_offset(unsigned int) { return nullptr; }
__device__ unsigned int __vx_warps_per_group(void) { return 1; }

// Note: TCU compute now uses standard wmma API (mma_sync).
// CuPBoP translator handles wmma → Vortex TCU conversion automatically.

#define DXA_DESC_A  0
#define DXA_DESC_B  1

// TCU+DXA kernel using WGMMA path (SS mode: both A,B from descriptors)
// Double-buffered: two DXA barrier IDs overlap prefetch with compute
__global__ void sgemm_tcu_dxa_kernel(const half *A, const half *B, float *C,
                                      int M, int N, int K) {
    int tile_row = blockIdx.y * WMMA_M;
    int tile_col = blockIdx.x * WMMA_N;

    // Standard WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    wmma::fill_fragment(fragC, 0.0f);

    // Double-buffer setup
    unsigned int bar[2];
    bar[0] = __vx_dxa_barrier_id(0);
    bar[1] = __vx_dxa_barrier_id(1);
    unsigned int num_warps = __vx_warps_per_group();

    // Double-buffer shared memory: 2 stages, each holds A tile + B tile
    half* shmem = (half*)__vx_local_mem_offset(2 * 2 * WMMA_M * WMMA_K * sizeof(half));
    half* prefetch_buf[2];
    prefetch_buf[0] = shmem;
    prefetch_buf[1] = shmem + 2 * WMMA_M * WMMA_K;

    // Prologue: issue DXA prefetch for first tile on bar[0]
    unsigned int cur = 0;
    unsigned int nxt = 1;
    __vx_dxa_issue_2d(DXA_DESC_A, bar[cur], prefetch_buf[cur], 0, tile_row);
    __vx_dxa_issue_2d(DXA_DESC_B, bar[cur], prefetch_buf[cur] + WMMA_M*WMMA_K, tile_col, 0);

    for (int k = 0; k < K; k += WMMA_K) {
        int next_k = k + WMMA_K;

        // Issue DXA prefetch for NEXT tile on bar[nxt] (overlaps with compute)
        if (next_k < K) {
            __vx_dxa_issue_2d(DXA_DESC_A, bar[nxt], prefetch_buf[nxt], next_k, tile_row);
            __vx_dxa_issue_2d(DXA_DESC_B, bar[nxt], prefetch_buf[nxt] + WMMA_M*WMMA_K, tile_col, next_k);
        }

        // Wait for CURRENT tile prefetch to complete
        __vx_barrier_arrive_and_wait(bar[cur], num_warps);

        // Load A and B from DXA-prefetched shared memory using standard wmma API
        half* tileA = prefetch_buf[cur];
        half* tileB = prefetch_buf[cur] + WMMA_M * WMMA_K;
        wmma::load_matrix_sync(fragA, tileA, WMMA_K);
        wmma::load_matrix_sync(fragB, tileB, WMMA_N);

        // Standard wmma compute — CuPBoP translates to Vortex TCU instruction
        wmma::mma_sync(fragC, fragA, fragB, fragC);

        // Post-compute sync on current barrier
        __vx_barrier_arrive_and_wait(bar[cur], num_warps);

        // Swap cur/nxt for next iteration
        unsigned int tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    // Store result
    float *tileC = C + tile_row * N + tile_col;
    wmma::store_matrix_sync(tileC, fragC, N, wmma::mem_row_major);
}

// CPU reference
void gemm_cpu(const half *A, const half *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            C[i * N + j] = sum;
        }
}

int main(int argc, char **argv) {
    int M = WMMA_M, N = WMMA_N, K = WMMA_K;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            int n = atoi(argv[++i]);
            if (n > 0 && n % WMMA_M == 0) { M = N = K = n; }
            else { fprintf(stderr, "Size must be multiple of %d\n", WMMA_M); return -1; }
        }
    }
    printf("SGEMM TCU+DXA (WGMMA SS): M=%d, N=%d, K=%d\n", M, N, K);

    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(float);

    half *h_A = (half *)malloc(sizeA);
    half *h_B = (half *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)(rand() % 5) / 5.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half((float)(rand() % 5) / 5.0f);
    memset(h_C, 0, sizeC);
    gemm_cpu(h_A, h_B, h_C_ref, M, N, K);

    half *d_A, *d_B; float *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeC);

    dim3 grid(N / WMMA_N, M / WMMA_M);
    dim3 block(32, 1);
    // Shared memory for double-buffered DXA prefetch: 2 stages * (A tile + B tile)
    size_t sharedMem = 2 * 2 * WMMA_M * WMMA_K * sizeof(half);
    sgemm_tcu_dxa_kernel<<<grid, block, sharedMem>>>(d_A, d_B, d_C, M, N, K);

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
