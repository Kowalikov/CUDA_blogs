#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Helper function to check CUDA errors
int checkCuda(cudaError_t result, const char* msg) {

    if (result != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(result) << "\n";
        return 1; 
    }
    return 0;
}


int main() {

    return 0;
}