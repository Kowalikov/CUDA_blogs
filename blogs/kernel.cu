#include <iostream>
#include <cuda_runtime.h>   // Główna biblioteka CUDA do zarządzania pamięcią i urządzeniem
#include <device_launch_parameters.h>   // Parametry uruchamiania kernela (np. blockIdx, threadIdx)
#include <chrono> //bilioteka do mierzenia czasu w C++

void squareCPU(long long int* x, int size) {
    for (int i = 0; i < size; ++i)
        x[i] = x[i] * x[i];
}


// Kernel uruchamiany na GPU — każdy wątek podnosi jeden element tablicy do kwadratu
__global__ void squareKernel(long long int* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Obliczenie globalnego indeksu wątku
    if (i < size) {
        x[i] = x[i] * x[i]; // Podniesienie wartości do kwadratu
    }
}

// Funkcja benchmarkująca zadaną konfigurację bloków i wątków
int runBenchmarkGPU(int numBlocks, int numThreads, int size) {
    
    long long int* host_x = new long long int[size]; // Alokacja pamięci na CPU

    // Inicjalizacja tablicy wartościami od 0 do size - 1
    for (int i = 0; i < size; ++i)
        host_x[i] = i;

    // Alokacja pamięci na GPU
    long long int* device_x;
    cudaMalloc(&device_x, size * sizeof(long long int));

    // Kopiowanie danych z CPU do GPU
    cudaMemcpy(device_x, host_x, size * sizeof(long long int), cudaMemcpyHostToDevice);

    // Utworzenie zdarzeń do pomiaru czasu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Rozpoczęcie pomiaru czasu
    cudaEventRecord(start);

    // Uruchomienie kernela z podaną konfiguracją
    squareKernel <<< numBlocks, numThreads >>> (device_x, size);

    // Zakończenie pomiaru czasu
    cudaEventRecord(stop);
    cudaDeviceSynchronize(); // Czekamy na zakończenie działania kernela

    // Sprawdzenie, czy kernel wykonał się poprawnie
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Blad kernela: " << cudaGetErrorString(err) << "\n";
        delete[] host_x;  // Zwolnienie pamięci na CPU
        return 1;
    }

    // Obliczenie czasu wykonania
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Kopiowanie wyników z GPU do CPU
    cudaMemcpy(host_x, device_x, size * sizeof(long long int), cudaMemcpyDeviceToHost);

    // Wypisanie wyników na konsolę
    /*for (int i = 0; i < size; ++i)
        std::cout << "x[" << i << "] = " << host_x[i] << "\n";*/

    // Wyświetlenie wyników benchmarku
    std::cout << "\nKonfiguracja: " << numBlocks << " blokow x " << numThreads << " watkow\n";
    std::cout << "Czas GPU: " << milliseconds << " ms\n";

    // Zwolnienie pamięci na GPU i CPU
    cudaFree(device_x);
    delete[] host_x;
   
    return 0;
}

//Funkcja benchmarkująca CPU
void runBenchmarkCPU(int size) {
    
    long long int* cpu_x = new long long int[size];
    for (int i = 0; i < size; ++i)
        cpu_x[i] = i;

    // Pomiar czasu CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    squareCPU(cpu_x, size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

    std::cout << "Czas CPU: " << cpu_duration.count() << " ms\n";

    delete[] cpu_x;
}

bool getConfiguration(int &size, int &blocks, int &threads, int maxThreadsPerBlock, int maxBlocksPerGrid) {

    int menu;

    std::cout << "\nWybierz tryb konfiguracji:\n";
    std::cout << "1 - Podaj liczbe blokow\n";
    std::cout << "2 - Podaj liczbe watkow\n";
    std::cout << "Twoj wybor: ";

    std::cin >> menu;

    std::cout << "\nPodaj rozmiar tablicy: ";
    std::cin >> size;
    if (size == 0)
        return false;

    if (menu == 1) {
        std::cout << "Podaj liczbe blokow: ";
        std::cin >> blocks;
        if (blocks == 0)
            return false;
        if (blocks > maxBlocksPerGrid) {
            std::cerr << "Blad. Liczba blokow przekracza limit GPU. Wprowadz ponownie\n";
            return getConfiguration(size, blocks, threads, maxThreadsPerBlock, maxBlocksPerGrid);
        }
        else {
            threads = (size + blocks - 1) / blocks; //obliczanie liczby wątków
            if (threads > maxThreadsPerBlock) {
                std::cerr << "Obliczona liczba wątków (" << threads << ") przekracza limit GPU (" << maxThreadsPerBlock << ")\n";
                return getConfiguration(size, blocks, threads, maxThreadsPerBlock, maxBlocksPerGrid);
            }
        }
            
    }
    else if (menu == 2) {
        std::cout << "Podaj liczbe watkow na blok: ";
        std::cin >> threads;
        if (threads == 0)
            return false;
        if (threads > maxThreadsPerBlock) {
            std::cerr << "Blad. Liczba blokow przekracza limit GPU. Wprowadz ponownie\n";
            return getConfiguration(size, blocks, threads, maxThreadsPerBlock, maxBlocksPerGrid);
        }
        else
        {
            blocks = (size + threads - 1) / threads; //obliczanie liczby bloków
            if (blocks > maxBlocksPerGrid) {
                std::cerr << "Obliczona liczba blokow (" << blocks << ") przekracza limit GPU (" << maxBlocksPerGrid << ")\n";
                return getConfiguration(size, blocks, threads, maxThreadsPerBlock, maxBlocksPerGrid);
            }
        }
            
    }
    else {
        std::cerr << "Nieznany tryb konfiguracji. Wybierz ponownie\n";
        return getConfiguration(size, blocks, threads, maxThreadsPerBlock, maxBlocksPerGrid);
    }
    
    int totalThreads = blocks * threads;
    if (totalThreads > size) {
        std::cout << "\nUwaga: " << totalThreads - size << " watkow nie przetworzy zadnych danych.\n\n";
        system("pause");
    }

    return true;
}

void deviceSpecifcation(int &maxThreadsPerBlock, int &maxBlocksPerGrid) {

    int maxThreadsPerSM, multiProcessorCount;
    
    // Maksymalna liczba wątków na blok
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

    // Maksymalna liczba wątków na multiprocesor
    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    // Liczba multiprocesorów
    cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, 0);

    // Maksymalna liczba bloków w 1D (oś X)
    cudaDeviceGetAttribute(&maxBlocksPerGrid, cudaDevAttrMaxGridDimX, 0);

    std::cout << "Maksymalna liczba watkow na blok: " << maxThreadsPerBlock << "\n";
    std::cout << "Maksymalna liczba blokow w siatce (X): " << maxBlocksPerGrid << "\n";
    std::cout << "Maksymalna liczba watkow na multiprocesor: " << maxThreadsPerSM << "\n";
    std::cout << "Liczba multiprocesorow: " << multiProcessorCount << "\n";
}

int main() {

    int blocks, threads, size;
    int maxThreadsPerBlock, maxBlocksPerGrid;

    std::cout << "=== CUDA Benchmark na przykladzie podnoszenia do kwadratu indeksow tablicy ===\n\n";
    std::cout << "---Specyfikacja Twojej karty graficznej---\n\n";
    deviceSpecifcation(maxThreadsPerBlock, maxBlocksPerGrid);
    std::cout << "\nWpisz 0 jako rozmiar tablicy albo liczbe blokow lub watkow, aby zakonczyc.\n";
    
    // Pętla umożliwiająca wielokrotne testowanie bez restartu programu
    while (true) {
        
        if (!getConfiguration(size, blocks, threads, maxThreadsPerBlock, maxBlocksPerGrid))
            break;

        if (runBenchmarkGPU (blocks, threads, size) != 0) 
            break;

        runBenchmarkCPU(size);
    }

    return 0;
}