#include <iostream>                      
#include <cuda_runtime.h>               // Główna biblioteka CUDA do zarządzania pamięcią i urządzeniem
#include <device_launch_parameters.h>   // Parametry uruchamiania kernela (np. blockIdx, threadIdx)


// Kernel uruchamiany na GPU - każdy wątek odkłada lokalnie indeksy bloku i wątku jaki go wykonuje
__global__ void kernel() {
    int thr_idx_x = threadIdx.x;
    int thr_idx_y = threadIdx.y;
    int thr_idx_z = threadIdx.z;

    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;
    int block_idx_z = blockIdx.z;

    int block_dim_x = blockDim.x;
    int block_dim_y = blockDim.y;
    int block_dim_z = blockDim.z;

    int grid_dim_x = gridDim.x;
    int grid_dim_y = gridDim.y;
    int grid_dim_z = gridDim.z;
}


int main() {
    // Uruchomienie kernela: 2 bloki po 4 wątki = 8 wątków
    kernel <<< 2, 4 >>> ();

    // Synchronizacja — czekamy aż GPU zakończy pracę
    cudaDeviceSynchronize();

    // Sprawdzenie, czy kernel wykonał się poprawnie
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Blad kernela: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    return 0;
}