#include <iostream>
#include <cuda_runtime.h>

    
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0)
        printf("CUDA device available\n");
    else
        printf("No CUDA device!\n");

    return 0;
}