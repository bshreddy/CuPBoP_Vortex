#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

__global__ void wmma_test_kernel(half *A, half *B, float *C) {
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // wmma::load_matrix_sync(a_frag, A, K);
    // wmma::load_matrix_sync(b_frag, B, N);

    // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}

int main() {
    half *A, *B;
    float *C;

    cudaMalloc(&A, M*K*sizeof(half));
    cudaMalloc(&B, K*N*sizeof(half));
    cudaMalloc(&C, M*N*sizeof(float));

    for (int i = 0; i < M*K; i++)
        A[i] = __float2half(1.0f);

    for (int i = 0; i < K*N; i++)
        B[i] = __float2half(1.0f);

    cudaDeviceSynchronize();

    wmma_test_kernel<<<1, 32>>>(A, B, C);
    cudaDeviceSynchronize();

    printf("C[0] = %f\n", C[0]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
