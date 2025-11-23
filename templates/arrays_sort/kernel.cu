// /d:/Projects/CUDA/CG_decomposition/kernel.cu
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using std::cout;
using std::endl;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void squareKernel(int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = idx + 1; // first integers above zero: 1..n
        out[idx] = val * val;
    }
}

void print2DArray(int** arr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << arr[i][j] << '\t';
        }
        cout << endl;
    }
}

int main() {
    const int N = 10;

    int** x = new int*[N];
    for (int i=0; i<N; i++) {
        x[i] = new int[N]; 
    }

    print2DArray(x, N, N);


    // Free memory
    for (int i = 0; i < N; i++) {
        delete[] x[i];
    }
    delete[] x;


    return 0;
}