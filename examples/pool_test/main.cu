#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// Warp reduction with multiple shfl — similar to wedford pattern
__global__ void shfl_test(int *output) {
    // warp 0: val=1..32 (sum=528), warp 1: val=1001..1032 (sum=33528), etc.
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int val = warp_id * 1000 + lane + 1;

    // Warp reduction: manually unrolled
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (threadIdx.x == 0) {
        output[0] = val;  // expected: 1+2+...+32 = 528
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
    // warp 0: sum(1..32)=528, warp 1: sum(1001..1032)=33528, etc.
    printf("Result: %d\n", h_out);
    printf("warp0_expected=528, warp1=33528, warp7=225528\n");
    printf("%s\n", (h_out == 528) ? "PASSED!" : "FAILED!");

    cudaFree(d_out);
    return 0;
}
