#include <iostream>                      
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Helper do radzenia sobie z błędami CUDA
int checkCuda(cudaError_t result, const char* msg) {

    if (result != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(result) << "\n";
        return 1; 
    }
    return 0;
}

// Kernel uruchamiany na GPU — każdy wątek podnosi jeden element tablicy do kwadratu
__global__ void squareKernel(int* x, int n) {
    // Obliczenie globalnego indeksu wątku w siatce
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Sprawdzenie, czy indeks mieści się w zakresie tablicy
    if (i < n) {
        x[i] = x[i] * x[i];  // Podniesienie wartości do kwadratu
    }
}

int main() {
    const int size = 100;  // Rozmiar tablicy
    int* host_x = new int[size];  // Dynamiczna alokacja pamięci na CPU (na stercie)

    // Inicjalizacja tablicy wartościami od 0 do 99
    for (int i = 0; i < size; ++i)
        host_x[i] = i;

    // Alokacja pamięci na GPU (device)
    int* device_x;
    if (!checkCuda(cudaMalloc(&device_x, size * sizeof(int)), "cudaMalloc device_x")) {
        delete[] host_x; 
        if (device_x) cudaFree(device_x);

        return 1;
    }

    // Kopiowanie danych z CPU (host) do GPU (device)
    if (checkCuda(cudaMemcpy(device_x, host_x, size * sizeof(int), cudaMemcpyHostToDevice), "Memcpy host_array -> device_array") ) {
        delete[] host_x; 
        cudaFree(device_x);
        
        return 1;
    }

    // Uruchomienie kernela: 20 bloków po 5 wątków = 100 wątków
    squareKernel <<< 20, 5 >>> (device_x, size);

    // Synchronizacja - czekamy, aż GPU zakończy pracę, zanim przejdziemy dalej
    if (checkCuda(cudaDeviceSynchronize(), "Device synchronize")) {
        delete[] host_x;
        cudaFree(device_x);
        
        return 1;
    }

    // Sprawdzenie, czy kernel wykonał się poprawnie
    if (checkCuda(cudaGetLastError(), "Kernel execution")){
        delete[] host_x;
        cudaFree(device_x);
        
        return 1;
    }
    

    // Kopiowanie wyników z GPU z powrotem do CPU
    if (checkCuda(cudaMemcpy(host_x, device_x, size * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy device_array -> host_array")) {
        delete[] host_x;
        cudaFree(device_x); 
        
        return 1;
    }

    // Wypisanie wyników na konsolę
    for (int i = 0; i < size; ++i)
        std::cout << "x[" << i << "] = " << host_x[i] << "\n";

    // Zwolnienie pamięci na GPU i CPU
    delete[] host_x;
    cudaFree(device_x);

    return 0;
}