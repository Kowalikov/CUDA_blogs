#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void kernel(int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = idx + 1; // first integers above zero: 1..n
        out[idx] = val * val;
    }
}

int main() {
    const int N = 100;

    // Host array allocation
    int *h_out = (int*)malloc(N * sizeof(int));
    if (!h_out) return EXIT_FAILURE;
    
    // Device array allocation
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

    int threadsPerBlock = 128;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocks, threadsPerBlock>>>(d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        printf("%d^2 = %d\n", i + 1, h_out[i]);
    }

    CUDA_CHECK(cudaFree(d_out));
    std::free(h_out);
    return 0;
}