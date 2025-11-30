#include <iostream>
#include <cuda_runtime.h>

int main() {
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max block dimensions: "
              << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << "\n";
    std::cout << "Max grid dimensions: "
              << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x "
              << prop.maxGridSize[2] << "\n";

    return 0;
}