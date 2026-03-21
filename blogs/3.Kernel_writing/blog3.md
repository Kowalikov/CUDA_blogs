---
layout: default
title: Pisanie funkcji na GPU
permalink: /blog3
---

<!-- blog3 content here -->
<h1>Pisanie funkcji na GPU</h1>  

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

W trzeciej odsłonie tej serii pokażemy, jak od podstaw napisać funkcję wykonywaną na GPU z CUDA w C++ - tzw. kernel. Pokażemy, jak alokować pamięć na GPU, kopiować dane między CPU a GPU, a następnie uruchomić prosty kernel mnożący macierze.

<h4>🧰 Funkcje na GPU</h4>

<h3>Dostęp do danych w kernelu: identyfikator wątku i bloku</h3>

Na początek, napiszmy prosty kernel, który będzie wykonywany na GPU a jego funkcja, to zorientować się jaki blok i wątek wykonuje go w danym momencie. Kernel to funkcja oznaczona specjalnym modyfikatorem `__global__`, która jest wywoływana z CPU, ale wykonywana na GPU.

W tym celu potrzebujemy dwóch importów:
- `#include <cuda_runtime.h>` - główna biblioteka CUDA do zarządzania pamięcią i urządzeniem. Bez niej nie napiszemy kernela ani nie będziemy mogli alokować pamięci na GPU.
- `#include <device_launch_parameters.h>` - zawiera definicje parametrów uruchamiania kernela, takich jak `blockIdx`, `threadIdx`, `blockDim`, które pozwalają nam zorientować się, który wątek i blok wykonuje dany fragment kodu.


<!-- make a code snippet in cpp -->
```cpp
#include <iostream>                      
#include <cuda_runtime.h>               // Główna biblioteka CUDA do zarządzania pamięcią i urządzeniem
#include <device_launch_parameters.h>   // Parametry uruchamiania kernela (np. blockIdx, threadIdx)


// Kernel uruchamiany na GPU — każdy wątek podnosi jeden element tablicy do kwadratu
__global__ void kernel() {
    // Obliczenie globalnego indeksu wątku w siatce
    int thr_idx_x = blockIdx.x;
    int thr_idx_y = blockIdx.y;
    int thr_idx_z = blockIdx.z;

    int block_idx_x = blockDim.x;
    int block_idx_y = blockDim.y; 
}
```

Okej, kernel jest gotowy. Teraz go uruchomimy. W tym celu musimy go wywołać z CPU, używając specjalnej składni `<<<...>>>`, która pozwala nam określić, ile bloków i wątków chcemy uruchomić. Na przykład, jeśli chcemy uruchomić 2 bloki po 4 wątki każdy, napiszemy:

```cpp
int main() {
    // Uruchomienie kernela: 2 bloki po 4 wątki = 8 wątków
    kernel <<< 2, 4 >>> ();

    // Synchronizacja — czekamy aż GPU zakończy pracę
    cudaDeviceSynchronize();

    // Sprawdzenie, czy kernel wykonał się poprawnie
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Blad kernela: " << cudaGetErrorString(err) << "\n";
        delete[] host_x;  // Zwolnienie pamięci na CPU
        return 1;
    }

    return 0;
}
```

Dość egzotyczna składnia, prawda? Nazwa oddzielona od nawiasów do wywołania, a w nich liczby określające rozmiar siatki i bloku. Ale dzięki temu CUDA jest tak wyraźna i charakterystyczna.  

Oprócz wywołania, mamy też synchronizację, która jest ważna, ponieważ kernel jest wykonywany asynchronicznie. Musimy poczekać, aż GPU zakończy pracę, zanim przejdziemy dalej w kodzie. Będzie to ważne, gdy będziemy chcieli odczytać wyniki z GPU i przenieść je na CPU, po zakończeniu pracy wszystkich wątków.  

Na koniec sprawdzamy, czy kernel wykonał się poprawnie, używając `cudaGetLastError()`. Oprócz sprawdzenia, możemy też wypisać ewentualny błąd, jeśli coś poszło nie tak.  

Teraz, wróćmy do samego kernela. Nie jest to najbardziej użyteczna funkcja. Żeby była praktyczna, potrzebujemy rozszerzyć ją o zwracanie, modyfikowanie i wyświetlanie danych. Co można i jak to zrobić? Rozważmy parę przykładów.

<h3>Dostęp do danych w kernelu: jak przekazywać argumenty do kernela</h3>

Na początek przekażmy do kernela informacyjnie jeden integer i tablicę floatów do modyfikacji. Ten sam przykład co z bloga 1. 


<h4>🧰 Krok 2: Sprawdzamy, jakie obliczenia możemy puścić na GPU</h4>


<h4> 🔍 Podsumowanie</h4>

[TBD]

-------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>

<!-- Previous blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog2">Poprzedni wpis</a></p>

<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog4">Następny wpis</a></p>
