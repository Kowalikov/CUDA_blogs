#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>


// Helper function to check CUDA errors
int checkCuda(cudaError_t result, const char* msg) {

    if (result != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(result) << "\n";
        return 1; 
    }
    return 0;
}

// Kernel GPU - Array Multiplication
__global__ void matMulKernel(float* A, float* B, float* C, int N) {
    // TODO
}


// CPU – Array Multiplication
void matMulCPU(float* A, float* B, float* C, int N) {
    // TODO
}

// Benchmark GPU
int runBenchmarkGPU(int N, int threadsPerBlock, int blocksPerGrid) {
    // TODO

    return 0;
}

// Benchmark CPU
void runBenchmarkCPU(int N) {
    // TODO
}


void deviceSpecification(int& maxThreadsPerBlock, int& maxBlocksPerGrid) {
    // TODO
}


bool getConfiguration(int& size, int& blocks, int& threads, int maxThreadsPerBlock, int maxBlocksPerGrid) {
    // TODO
}

int main() {

    int N, threads, blocks, maxThreads, maxBlocks;

    std::cout << "=== CUDA Benchmark: 2D Array Multiplication [NxN] ===\n";
    std::cout << "Arrays are filled with 1s and 2s for benchmarking purposes.\n\n\n";
    
    // Show GPU specification
    deviceSpecification(maxThreads, maxBlocks);
    
    // Main loop: user can run multiple benchmarks until they enter 0
    int loop_limit = 10;
    int loop_count = 0;
    while (loop_count < loop_limit)
    {
        if (!getConfiguration(N, blocks, threads, maxThreads, maxBlocks))
            break;
        if (runBenchmarkGPU(N, threads, blocks) == 0) {
            runBenchmarkCPU(N);
        }
        else {
            std::cerr << "GPU benchmark finished with an error.\n";
            return 1;
        }

        loop_count++;
    }
    return 0;
}