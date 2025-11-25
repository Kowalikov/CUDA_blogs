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
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N * N) {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


// CPU – Array Multiplication
void matMulCPU(float* A, float* B, float* C, int N) {
    
    for(int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }

}

// Benchmark GPU
int runBenchmarkGPU(int N, int threadsPerBlock, int blocksPerGrid) {
    
    size_t bytes = N * N * sizeof(float);

    // Alocate memory on CPU
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Alocate memory on GPU
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    if (checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A") || checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B") || checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C")) {
        delete[] h_A; 
        delete[] h_B; 
        delete[] h_C;
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        return 1;
    }

    // copy data CPU → GPU
    if (checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "Memcpy h_A -> d_A") || checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "Memcpy h_B -> d_B")) {
        delete[] h_A; 
        delete[] h_B; 
        delete[] h_C;
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C);
        return 1;
    }

    // configuration threads and blocks
    dim3 threads(threadsPerBlock);
    dim3 blocks(blocksPerGrid);

    // clocking GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulKernel <<< blocks, threads >>> (d_A, d_B, d_C, N); 
    cudaEventRecord(stop);

    if (checkCuda(cudaGetLastError(), "Kernel execution") || checkCuda(cudaDeviceSynchronize(), "Device synchronize")) {
        
        //free memory
        delete[] h_A; 
        delete[] h_B; 
        delete[] h_C;
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C);

        return 1;
    }
    
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // copy results GPU → CPU
    if (checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "Memcpy d_C -> h_C")) {
        delete[] h_A; 
        delete[] h_B; 
        delete[] h_C;
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C);
        return 1;
    }

    // Results of benchmark
    std::cout << "\nConfiguration: " << blocks.x << " blocks, " << threads.x << " threads per block\n";
    std::cout << "GPU TIME: " << gpuTime << " ms\n";

    // free memory
    delete[] h_A; 
    delete[] h_B; 
    delete[] h_C;
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);

    return 0;
}

// Benchmark CPU
void runBenchmarkCPU(int N) {

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();

    matMulCPU(h_A, h_B, h_C, N);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU TIME: " << cpu_duration.count() << " ms\n";
    
    //free memory
    delete[] h_A; 
    delete[] h_B; 
    delete[] h_C;
}

void deviceSpecification(int& maxThreadsPerBlock, int& maxBlocksPerGrid) {

    int maxThreadsPerSM, multiProcessorCount;

    // Maximum number of threads per block
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

    // Maximum number of threads per multiprocessor
    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    // Number of multiprocessors
    cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, 0);

    // Maximum number of blocks in 1D (X axis)
    cudaDeviceGetAttribute(&maxBlocksPerGrid, cudaDevAttrMaxGridDimX, 0);

    std::cout << "*** Your GPU specifications ***\n";
    std::cout << "Max threads per block: " << maxThreadsPerBlock << "\n";
    std::cout << "Max blocks per grid (X): " << maxBlocksPerGrid << "\n";
    std::cout << "Max threads per multiprocessor: " << maxThreadsPerSM << "\n";
    std::cout << "Number of multiprocessors: " << multiProcessorCount << "\n";
}


bool getConfiguration(int& size, int& blocks, int& threads, int maxThreadsPerBlock, int maxBlocksPerGrid) {
    
    while (true) {

        int menu;
        std::cout << "\nChoose configuration mode (enter 0 as a number of threads, blocks, size or choice to exit):\n";
        std::cout << "1 - Enter number of blocks\n";
        std::cout << "2 - Enter number of threads\n";
        std::cout << "Your choice: ";
        std::cin >> menu;

        if (menu == 0)
            return false;

        std::cout << "\nEnter array size [N]: ";
        std::cin >> size;
        if (size == 0) 
            return false;

        if (menu == 1) {

            std::cout << "Enter number of blocks: ";
            std::cin >> blocks;
            if (blocks == 0) 
                return false;
            if (blocks > maxBlocksPerGrid) {
                std::cerr << "Error. Number of blocks exceeds GPU limit. Try again\n";
                continue;
            }
            threads = (size * size + blocks - 1) / blocks;
            if (threads > maxThreadsPerBlock) {
                std::cerr << "Calculated number of threads (" << threads << ") exceeds GPU limit (" << maxThreadsPerBlock << ")\n";
                continue;
            }
        }
        else if (menu == 2) {

            std::cout << "Enter number of threads per block: ";
            std::cin >> threads;
            if (threads == 0) return false;
            if (threads > maxThreadsPerBlock) { 
                std::cerr << "Error. Number of threads exceeds GPU limit. Try again\n";
                continue;
            }
            blocks = (size * size + threads - 1) / threads;
            if (blocks > maxBlocksPerGrid) {
                std::cerr << "Calculated number of blocks (" << blocks << ") exceeds GPU limit (" << maxBlocksPerGrid << ")\n";
                continue;
            }
        }
        else {

            std::cerr << "Unknown configuration mode. Choose again\n";
            continue;
        }

        int totalThreads = blocks * threads;
        if (totalThreads > size * size)
            std::cout << "\nWarning: " << totalThreads - size * size << " threads will not process any data.\n\n";
             
        return true; 
    }
}

void clearConsole() {

    std::cout << "\033[2J\033[1;1H";

}



int main() {

    int N, threads, blocks, maxThreads, maxBlocks;

    std::cout << "=== CUDA Benchmark: 2D Array Multiplication [NxN] ===\n";
    std::cout << "Arrays are filled with 1s and 2s for benchmarking purposes.\n\n\n";
    
    // Show GPU specification
    deviceSpecification(maxThreads, maxBlocks);
    
    // Main loop: user can run multiple benchmarks until they enter 0
    while (true)
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
        clearConsole();
    }
    return 0;
}