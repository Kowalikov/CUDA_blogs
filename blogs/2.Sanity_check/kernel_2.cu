#include <iostream>
#include <cuda_runtime.h>


void deviceSpecification(
    int& maxThreadsPerBlock,
    int& maxBlocksPerGrid,
    int& maxThreadsPerSM,
    int& multiProcessorCount
) {

    // Maximum number of threads per block
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

    // Maximum number of blocks in 1D (X axis)
    cudaDeviceGetAttribute(&maxBlocksPerGrid, cudaDevAttrMaxGridDimX, 0);

    // Maximum number of threads per multiprocessor
    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    // Number of multiprocessors
    cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, 0);

    std::cout << "*** Your GPU specifications ***\n";
    std::cout << "Max threads per block: " << maxThreadsPerBlock << "\n";
    std::cout << "Max blocks per grid (X): " << maxBlocksPerGrid << "\n";
    std::cout << "Max threads per multiprocessor: " << maxThreadsPerSM << "\n";
    std::cout << "Number of multiprocessors: " << multiProcessorCount << "\n";
}

int main() {

    int maxThreads = -1, maxBlocks = -1, maxThreadsPerSM = -1, multiProcessorCount = -1;

    // Show GPU specification
    deviceSpecification(maxThreads, maxBlocks, maxThreadsPerSM, multiProcessorCount);
    
    return 0;
}