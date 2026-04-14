#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__ void shfl_test(int *output) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int val = warp_id * 1000 + lane + 1;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (threadIdx.x == 0) {
        output[0] = val;
    }
}

int main() {
    int h_out = -1;
    int *d_out;
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));

    shfl_test<<<1, 256>>>(d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: %d, Expected: 528\n", h_out);
    printf("%s\n", (h_out == 528) ? "PASSED!" : "FAILED!");

    cudaFree(d_out);
    return 0;
}
