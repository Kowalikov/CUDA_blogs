---
layout: default
title: Prosty benchmark GPU
permalink: /blog4
---

<!-- blog4 content here -->
<h1>IV. Prosty benchmark GPU </h1>  

<author> <b>Autor:</b> Natan Jarzyński</author>  
<!-- Key words lineup -->
<meta keywords="GPU programming, programming, coding, CUDA toolkit, CUDA, C++, GPU, programowanie równoległe, instalacja CUDA, pierwszy program CUDA, alokacja pamięci GPU, kernel CUDA, nvcc, nvidia-smi">
<!-- display keywords interactive way -->
<p> <b>Słowa kluczowe:</b>  
  <span class="keyword" style="color:cornflowerblue">GPU programming</span>,
  <span class="keyword" style="color:cornflowerblue">programming</span>,
  <span class="keyword" style="color:cornflowerblue">coding</span>,
  <span class="keyword" style="color:cornflowerblue">CUDA toolkit</span>,
  <span class="keyword" style="color:cornflowerblue">CUDA</span>,
  <span class="keyword" style="color:cornflowerblue">C++</span>,
  <span class="keyword" style="color:cornflowerblue">GPU</span>,
  <span class="keyword" style="color:cornflowerblue">programowanie równoległe</span>,
  <span class="keyword" style="color:cornflowerblue">instalacja CUDA</span>,
  <span class="keyword" style="color:cornflowerblue">pierwszy program CUDA</span>,
  <span class="keyword" style="color:cornflowerblue">alokacja pamięci GPU</span>,
  <span class="keyword" style="color:cornflowerblue">kernel CUDA</span>,
  <span class="keyword" style="color:cornflowerblue">nvcc</span>,
  <span class="keyword" style="color:cornflowerblue">nvidia-smi</span>
</p>

-----------
<h3>CUDA w C++ - jak zrobić prosty program do benchmarku GPU</h3>

W czwartej odsłonie tej serii pokażemy, jak sprawdzić specyfikację GPU z CUDA w C++, i na jej podstawie następnie napiszemy i uruchomimy prosty benchmark, żeby pokazać jej możliwości. Użyjemy do tego celu wszystkich elementów, które poznaliśmy do tej pory, a więc: alokacji pamięci na GPU, kopiowania danych między CPU a GPU, pisania funkcji na GPU (kernel), i uruchamiania jej z odpowiednią konfiguracją bloków i wątków. Rutyna do benchmarku to mnożenie dwóch kwadratowych macierzy (`N`x`N`) i pomiar czasu wykonania tej operacji na CPU i GPU, żeby zobaczyć różnicę. Do dzieła!


<h4>🧰 Krok 1: Zadania programu i szkielet implementacji </h4>

Nasz program powinien sprawdzić czas mnożenia dwóch macierzy na CPU i GPU, to na pewno. Potrzebujemy doprecyzować rozmiar tych macierzy, a także konfigurację uruchomienia kernela na GPU (ilość wątków na blok). Możemy to zrobić na sztywno, ale fajniej będzie, jeśli użytkownik będzie mógł wprowadzić te dane samodzielnie. Dlatego nasz program będzie miał następujące kroki:
1. Sprawdź specyfikację GPU, a w szczególności maksymalną liczbę wątków na blok i maksymalną liczbę bloków na siatkę.
2. Poproś użytkownika o wprowadzenie liczby wątków na blok, (bloki zostaną obliczone automatycznie) i rozmiaru macierzy (`N` dla `N`x`N`), które zwalidujemy pod limitami GPU. 
3. Puść benchmarki na CPU i GPU.

Ujmijmy to w mainie:

```cpp
int main() {

    int N, threads, blocks, maxThreads, maxBlocks;

    std::cout << "=== CUDA Benchmark: 2D Array Multiplication [NxN] ===\n";
    
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
```

Potrzebujemy 5 zmiennych dla danych i limitów. Dalej pobieramy limity i w pętli z max 10 iteracjami, pobieramy i walidujemy konfigurację od użytkownika, a następnie uruchamiamy benchmarki. Jeśli walidacja się nie powiedzie, wracamy do wprowadzania danych. Na poprawnych danych uruchamiamy benchmark GPU, a jeśli się powiedzie, to benchmark CPU.

Przygotujmy szkielety funkcji, które będziemy implementować w kolejnych krokach:

```cpp
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
```

`checkCuda` to no-brainer, zawsze jej będziemy potrzebować. `matMulKernel` i `matMulCPU` to nasze rutyny mnożenia macierzy. Każda będzie pobierać wskaźniki do tablic A, B, C i rozmiar N. Jedne wskaźniki będą na GPU, a drugie na CPU.
`runBenchmarkGPU` i `runBenchmarkCPU` to pełne benchmarkowe funkcje, które będą alokować pamięć, inicjalizować dane, uruchamiać mnożenie i mierzyć czas wykonania. Potrzebują one tylko rozmiar macierzy (i konfigurację kernela, dla GPU).  `deviceSpecification` to funkcja, która pobierze limity GPU i zwróci je przez referencję (w domyśle do maina). `getConfiguration` to funkcja, która będzie pobierać dane od użytkownika i walidować je. Dane od użytkowanika to rozmiar macierzy (N), liczba wątków na blok i liczba bloków, które przekażemy przez referencję, żeby zmienić je globalnie w maine, a do walidacji potrzebujemy znać limity GPU, które przekazujemy już normalnie jako kopie.

<h4>🧰 Krok 2: Piszemy funkcje do mnożenia macierzy</h4>

Zaczynamy od `matMulCPU` do mnożenia macierzy na CPU, a w zasadzie dwóch tablic typu float, które będą interpretowane jako macierze, a ich iloczyn odłożymy do tablicy `C`. Dwie pętle, po wszystkich elementów wyjściowych `C` do wypełnienia, i wewnętrzna pętla do obliczenia sumy iloczynów odpowiednich elementów z `A` i `B`. To jest klasyczna implementacja mnożenia macierzy, która jest prosta, ale nie jest zoptymalizowana (nie korzysta z cache, nie jest blokowa itp.), ale do naszego benchmarku będzie wystarczająca.

```cpp
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
```

`matMulKernel` to nasz kernel do mnożenia macierzy na GPU. Każdy wątek będzie odpowiedzialny za obliczenie jednego elementu wyjściowej macierzy `C`. Indeks wątku (`idx`) jest obliczany na podstawie indeksu bloku i indeksu wątku w bloku. Następnie, jeśli `idx` jest mniejszy niż liczba elementów w macierzy (`N`*`N`), to obliczamy odpowiedni wiersz i kolumnę dla tego `idx`, i wykonujemy sumę iloczynów odpowiednich elementów z `A` i `B`, zapisując wynik do `C`. 

```cpp
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
```

<h4>🧰 Krok 3: Piszemy funkcje do puszczania benchmarków</h4>

Benchmark CPU to prosta sprawa. Alokujemy trzy tablice na CPU, inicjalizujemy `A` i `B` jakimiś wartościami (np. 1.0f i 2.0f), a `C` zostawiamy pustą. Następnie mierzymy czas przed i po wywołaniu `matMulCPU`, i obliczamy różnicę, którą wypisujemy w milisekundach. Na koniec pamiętamy o zwolnieniu pamięci.

```cpp
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
```

Benchmark do GPU rozszerza to o tablice alokowone na GPU, oraz, na nich, transfer danych z CPU do GPU i na odwrót, oraz zwolnienie pamięci. Od razu opakujemy to wszystko w odpowiednie sprawdzanie błędów, które po poprzednim wpisie, nie powininny być niczym nowym.

Nowością tutaj, jest pomiar czasu wykonania kernela na GPU, korzystając z `cudaEvent_t` do dokładnego pomiaru czasu GPU. Zasadniczo do dokładnego pomiaru czasu wykonania kernela, musimy użyć `cudaEvent_t`, ponieważ `std::chrono` mierzy czas na CPU, a kernel jest wykonywany asynchronicznie na GPU. Dlatego tworzymy dwa zdarzenia (za pomocą funkcji `cudaEventCreate` do `start` i `stop`), rejestrujemy `start` przed wywołaniem kernela, a `stop` zaraz po. Następnie synchronizujemy GPU, żeby mieć pewność, że kernel się zakończył, i obliczamy różnicę między tymi dwoma zdarzeniami, za pomocą funkcji `cudaEventElapsedTime`. Mamy analog funkcji do benchmarku na CPU!

```cpp
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
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
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
```

<h4>🧰 Krok 4: Piszemy funkcje do pobierania konfiguracji i specyfikacji urządzenia</h4>

Specyfikacja GPU to skrócona wersja z poprzedniego wpisu.

```cpp
void deviceSpecification(int& maxThreadsPerBlock, int& maxBlocksPerGrid) {

    // Maximum number of threads per block
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

    // Maximum number of blocks in 1D (X axis)
    cudaDeviceGetAttribute(&maxBlocksPerGrid, cudaDevAttrMaxGridDimX, 0);

    std::cout << "*** Your GPU specifications ***\n";
    std::cout << "Max threads per block: " << maxThreadsPerBlock << "\n";
    std::cout << "Max blocks per grid (X): " << maxBlocksPerGrid << "\n";
}
```

Pobieranie konfiguracji od użytkowniak obędzie się w 5 próbach. Mamy kilka warunków do wejścia:
- rozmiar tablicy (`N`) musi być dodatni i mniejszy niż 1000, żeby uniknąć długich czasów wykonania i problemów z pamięcią
- liczba wątków na blok musi być mniejsza lub równa limitowi GPU
- liczba bloków, obliczona dla rozmiaru tablicy, musi być mniejsza lub równa limitowi GPU

Oprócz tego użytkownik ma opcję wyjścia z programu, wpisując 0 dla liczby wątków na blok. Jeśli wszystkie warunki są spełnione, funkcja zwraca true, a dane są przekazywane przez referencję. Jeśli którykolwiek warunek nie jest spełniony, użytkownik jest informowany o błędzie i proszony o ponowne wprowadzenie danych. Po 5 próbach, funkcja zwraca false, co może być sygnałem do zakończenia programu.

UWAGA! Liczba bloków jest obliczana na podstawie rozmiaru tablicy i liczby wątków na blok, zakładając, że każdy wątek przetwarza jeden element tablicy wyjściowej. Dlatego `blocks = (size * size + threads - 1) / threads` to sposób na obliczenie liczby bloków potrzebnych do pokrycia całej tablicy, z uwzględnieniem sytuacji, gdy liczba elementów nie jest idealnie podzielna przez liczbę wątków.

```cpp
bool getConfiguration(int& size, int& blocks, int& threads, int maxThreadsPerBlock, int maxBlocksPerGrid) {
    int loop_limit = 5;
    int loop_count = 0;
    while (loop_count < loop_limit) {
        loop_count++;

        std::cout << "Enter number of threads per block (enter 0 to exit):\n";
        std::cin >> threads;
        if (threads == 0) return false;
        
        std::cout << "\nEnter array size (N for NxN), smaller than 1000:\n";
        std::cin >> size;
        if (size <= 0) {
            std::cerr << "Error. Array size must be a positive integer. Try again.\n";
            continue;
        } else if (size > 1000) {
            std::cerr << "Warning. Array size is very large and may cause long execution times or memory issues. Try again with a smaller size.\n";
            continue;
        }

        if (threads > maxThreadsPerBlock) { 
            std::cerr << "Error. Number of threads exceeds GPU limit. Try again\n";
            continue;
        }
        blocks = (size * size + threads - 1) / threads;
        if (blocks > maxBlocksPerGrid) {
            std::cerr << "Calculated number of blocks (" << blocks << ") exceeds GPU limit (" << maxBlocksPerGrid << ")\n";
            continue;
        }

        int totalThreads = blocks * threads;
        if (totalThreads > size * size) {
            std::cout << "\nWarning: " << totalThreads - size * size << " threads will not process any data.\n\n";
        }
        
        return true; 
    }

    return false;
}
```

Nasz program jest gotowy! Pełen skrypt pobierzesz z [repo](https://github.com/kowalikov/CUDA_blogs/blob/main/4.Benchmark_GPU/kernel_2.cu).

<h4> 🔍 Podsumowanie</h4>

W tym wpisie stworzyliśmy prosty program do benchmarku GPU, który mnoży dwie macierze na CPU i GPU, mierząc czas wykonania obu operacji. Program jest interaktywny, pozwalając użytkownikowi wprowadzić rozmiar macierzy i konfigurację kernela, z odpowiednią walidacją danych. Dzięki temu możemy zobaczyć różnicę w czasie wykonania między CPU a GPU dla różnych rozmiarów macierzy i konfiguracji. To świetny sposób, żeby zobaczyć moc obliczeniową GPU w praktyce!

-------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>

<!-- Previous blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog3">Poprzedni wpis: III. Pisanie funkcji na GPU</a></p>

<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog5">Następny wpis: V. Prosty Ray Tracing</a></p>
