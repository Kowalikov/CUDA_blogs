---
layout: default
title: Prosty benchmark GPU
permalink: /blog2
---

<!-- blog2 content here -->
<h1>Prosty benchmark GPU </h1>  

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

📌 Przypominajka, jak skompilować i uruchomić program CUDA na Linuxie (na Windowsie będzie podobnie, ale z rozszerzeniem `.exe`):
```bash
nvcc kernel.cu -o kernel && chmod u+x ./kernel && ./kernel
```

<h4>🧰 Krok 1: Sprawdzamy, czy CUDA widzi GPU</h4>


<!-- make a code snippet in cpp -->
```cpp
#include <iostream>
#include <cuda_runtime.h>

    
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0)
        printf("CUDA device available\n");
    else
        printf("No CUDA device!\n");

    return 0;
}
```

<h4>🧰 Krok 2: Sprawdzamy, jakie obliczenia możemy puścić na GPU</h4>

Do zaznajomienia się ze specyfikacją GPU użyjemy funkcji `cudaDeviceGetAttribute`, która pozwala nam odczytać różne parametry dotyczące możliwości naszego GPU. Zwraca ona wartość artybutu, o który pytamy w drugim argumencie funkcji, pod zmienną, której adres podajemy w pierwszym argumencie. Ostatni argument to numer urządzenia (zazwyczaj 0, jeśli mamy tylko jedno GPU). Teraz, my będziemy pytać o:
- `cudaDevAttrMaxThreadsPerBlock` - maksymalna liczba wątków, które mogą być uruchomione w jednym bloku.
- `cudaDevAttrMaxGridDimX` - maksymalna liczba bloków, które mogą być uruchomione w jednym wymiarze siatki (grid).
- `cudaDevAttrMaxThreadsPerMultiProcessor` - maksymalna liczba wątków, które mogą być uruchomione na jednym multiprocesorze.
- `cudaDevAttrMultiProcessorCount` - liczba multiprocesorów w GPU.

Trochę tego dużo, ale efektywnie pozwoli nam to zrozumieć, jak duże obliczenia możemy uruchomić na naszym GPU i jak je najlepiej zorganizować.
Rozpakujmy ten żargon. Wątki, są efektywnie tym samym co wątki na CPU, ale są organizowane w bloki, a bloki są organizowane w siatkę (grid). Multiprocesory to jednostki wykonawcze w GPU, które obsługują te bloki i wątki. Znając te parametry, możemy zoptymalizować nasze obliczenia, aby najlepiej wykorzystać zasoby GPU.
Puśćmy teraz kod, który odczyta te parametry i wyświetli je na ekranie:

<!-- make a code snippet in cpp -->
```cpp
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
```

Parę komentarzy co do tego kodu:
- Funkcja `deviceSpecification` jest odpowiedzialna za odczytanie i wyświetlenie specyfikacji GPU. Przyjmuje ona referencje ( `int&` ) do zmiennych, które zostaną wypełnione odpowiednimi wartościami. Chcę odłożyć wyniki `cudaDeviceGetAttribute` do zmiennych w mainie, gdyby potem miałyby mi posłużyć do ustawienia rozmiaru bloków i siatki w benchmarku (albo jej limitów).
- W `main`ie deklarujemy zmienne, które będą przechowywać te wartości, i wywołujemy funkcję `deviceSpecification`, która je wypełni i wyświetli.
- W `cudaDeviceGetAttribute` podajemy zmienne z operatorem `&` wydobywającym adres, ponieważ ta funkcja oczekuje wskaźnika do zmiennej, w której ma zapisać wynik. Uwaga! Ten sam operator `&` jest używany zarówno do deklaracji referencji ( `int&` ), jak i do pobierania adresu zmiennej ( `&maxThreadsPerBlock` ). To może być mylące, ale w kontekście funkcji `cudaDeviceGetAttribute`, `&` jest używane do przekazania adresu zmiennej, aby funkcja mogła zapisać wynik bezpośrednio w tej zmiennej, a w przypadku deklaracji `int&` jest to sposób na stworzenie referencji, która jest aliasem dla innej zmiennej. Efektywnie, upewnieniem się, że funkcja będzie pracować z prawdziwym obiektem, a nie jego kopią.

W moim przypadku, GPU pokazał następujące parametry:
```
*** Your GPU specifications ***
Max threads per block: 1024
Max blocks per grid (X): 2147483647
Max threads per multiprocessor: 1024
Number of multiprocessors: 14
```
To oznacza, że mogę uruchomić do 1024 wątków w jednym bloku, mogę mieć ogromną liczbę bloków w siatce (2^31-1), a na jednym multiprocesorze mogę mieć do 1024 wątków. Mój GPU ma 14 multiprocesorów, więc teoretycznie może obsłużyć do 14 * 1024 = 14336 wątków jednocześnie, ale to zależy od wielu czynników, takich jak zasoby pamięci i harmonogramowanie. Wykorzystajmy te informacje, aby napisać funkcję pobierającą od użytkownika rozmiar benchmarku z odpowiednimi limitami.


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



<h4> 🔍 Podsumowanie</h4>

[TBD]

-------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>

<!-- Previous blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog1">Poprzedni wpis</a></p>

<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog3">Następny wpis</a></p>
