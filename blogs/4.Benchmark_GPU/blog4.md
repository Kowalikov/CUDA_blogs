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

W drugiej odsłonie tej serii pokażemy, jak sprawdzić specyfikację GPU z CUDA w C++, a następnie napiszemy i uruchomimy prosty benchmark, żeby pokazać jej możliwości. 



<h4>🧰 Krok 3: Piszemy funkcję do pobierania konfiguracji benchmarku </h4>

```cpp
bool getConfiguration(int& size, int& blocks, int& threads, int maxThreadsPerBlock, int maxBlocksPerGrid) {
    
    while (true) {

        int menu;
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
        if (totalThreads > size * size)
            std::cout << "\nWarning: " << totalThreads - size * size << " threads will not process any data.\n\n";
             
        return true; 
    }
}
```
Ten kod ma strukturę dość liniową, ale zwróćmy uwagę na kilka rzeczy:
- Podstawą jest ilość wątków na blok, którą użytkownik może ustawić. Jeśli wpisze 0, funkcja zwróci false, co może być sygnałem do zakończenia programu.
- Następnie użytkownik jest proszony o rozmiar tablicy (N dla NxN). Wprowadzamy ograniczenie, że N musi być mniejsze niż 1000, aby uniknąć długich czasów wykonania i problemów z pamięcią.
- Sprawdzamy, czy liczba wątków na blok nie przekracza limitu GPU. Jeśli tak, prosimy użytkownika o ponowne wprowadzenie.
- Obliczamy liczbę bloków potrzebnych do przetworzenia całej tablicy, zakładając, że każdy wątek przetwarza jeden element. Jeśli liczba bloków przekracza limit GPU, prosimy o ponowne wprowadzenie.
- Na koniec, jeśli liczba wątków (blocks * threads) jest większa rozmiarem tablicy (size * size), wyświetlamy ostrzeżenie, że część wątków nie będzie przetwarzać żadnych danych - jesteśmy coraz lepsi w rozumieniu sposobu działania GPU.

Efektywnie, po wprowadzeniu tych danych, dla macierzy o rozmiarze NxN, będziemy mieli `size*size` elementów do przetworzenia. Jeśli użytkownik ustawi `threads` na 256, a `size` na 512, to potrzebujemy `blocks = (512*512 + 256 - 1) / 256 = 1024` bloków, co jest w granicach mojego GPU. Wtedy będziemy mieli `totalThreads = blocks * threads = 1024 * 256 = 262144` wątków, a rozmiar tablicy to `size*size = 512*512 = 262144`, więc każdy wątek będzie miał dokładnie jeden element do przetworzenia.

<h4>🧰 Krok 4: Funkcja mnożenia macierzy na CPU </h4>

Jako, że piszemy benchmark, potrzebujemy mieć coś do porównania. Napiszmy prostą funkcję mnożenia macierzy na CPU, a w zasadzie dwóch tablic typu float, które będą interpretowane jako macierze, a ich iloczyn (Hadamarda, każdy element macierzy `A` z każdym elementem macierzy `B`) odłożymy do tablicy `C`.

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

<h4>🧰 Krok 5: Funkcja mnożenia macierzy na GPU </h4>


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


<h4> 🔍 Podsumowanie</h4>

[TBD]

-------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>

<!-- Previous blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog3">Poprzedni wpis: III. Pisanie funkcji na GPU</a></p>

<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog5">Następny wpis: V. Prosty Ray Tracing</a></p>
