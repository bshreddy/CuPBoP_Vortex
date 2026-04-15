#include <stdio.h>
#include <cuda.h>

// Minimal inter-warp reduction — same structure as wedford.
// Tests: alloca at merge point replicated per-thread or not.
// block=128 (4 warps), expected = sum(1..128) = 8256

template<typename T>
__device__ __forceinline__
void warp_reduce(T &val) {
  #pragma unroll
  for (int i = warpSize/2; i > 0; i >>= 1)
    val += __shfl_down_sync(0xffffffff, val, i);
}

template<typename T>
__device__ void block_reduce(T *smem, T &val, int block_size, int tid) {
  int lane = tid % warpSize;
  int wid  = tid / warpSize;

  if (block_size > warpSize) {
    warp_reduce(val);
    if (lane == 0)
      smem[wid] = val;
    __syncthreads();
    if (wid == 0)
      val = (tid < block_size / warpSize) ? smem[lane] : T(0);
  }
  if (wid == 0)
    warp_reduce(val);
}

template<typename T>
__global__ void reduce_kernel(T *out) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int block_size = blockDim.x * blockDim.y;
  T val = T(tid + 1);

  static __shared__ T smem[32];
  block_reduce(smem, val, block_size, tid);

  if (tid == 0)
    out[0] = val;
}

int main() {
  int *d_out, h_out;
  cudaMalloc(&d_out, sizeof(int));

  // Use 2D block like wedford: (32, 4) = 128 threads = 4 warps
  dim3 block(32, 4);
  reduce_kernel<int><<<1, block>>>(d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  int n = 32 * 4;
  int expected = n * (n + 1) / 2;  // sum(1..128) = 8256
  printf("Result: %d, Expected: %d\n", h_out, expected);
  printf("%s\n", (h_out == expected) ? "PASSED!" : "FAILED!");
  return (h_out != expected);
}
