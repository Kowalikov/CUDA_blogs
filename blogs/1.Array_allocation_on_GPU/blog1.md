---
layout: default
title: Strona główna
permalink: /blog1
---

<!-- blog1 content here -->
<h1>Alokacja tablicy na GPU </h1>  

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

-----------
<h3>CUDA w C++ - od instalacji do pierwszego programu</h3>

W tym artykule pokażę, jak przygotować środowisko programistyczne do pracy z CUDA w C++, a następnie napiszemy i uruchomimy prosty program, który podnosi liczby do kwadratu na karcie graficznej NVIDIA. Jeśli dopiero zaczynasz przygodę z programowaniem równoległym, ten wpis jest dla Ciebie.


<h4>🧰 Krok 1: Instalacja CUDA Toolkit</h4>

1. Przejdź na stronę pobierania CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
2. Wybierz swój system operacyjny (Windows, Linux) i wersję
3. Pobierz i zainstaluj:
    - CUDA Toolkit (zawiera kompilator nvcc, biblioteki, przykłady)
    - Sterowniki NVIDIA (upewnij się, że są aktualne)

📌 Po instalacji sprawdź wersję CUDA:


```bash
nvcc --version
```

📌 Sprawdź, czy karta graficzna jest widoczna:

```bash
nvidia-smi
```

<h4>🧱 Krok 2: Konfiguracja środowiska</h4>

Możesz używać dowolnego edytora kodu, np.:

- **Visual Studio (Windows)** — CUDA integruje się automatycznie
- **CLion** — wygodny dla CMake, wymaga ustawienia toolchaina z `nvcc`

<h4>🧪 Krok 3: Pierwszy program CUDA</h4>

Poniżej znajduje się kod, który:

- Tworzy tablicę 100 liczb
- Przesyła ją na GPU
- Uruchamia 100 wątków, każdy podnosi jedną liczbę do kwadratu
- Kopiuje wynik z powrotem na CPU i wypisuje go


<h4>📦 Kod źródłowy z komentarzami</h4>


<!-- make a code snippet in cpp -->
```cpp
#include <iostream>                      
#include <cuda_runtime.h>               // Główna biblioteka CUDA do zarządzania pamięcią i urządzeniem
#include <device_launch_parameters.h>   // Parametry uruchamiania kernela (np. blockIdx, threadIdx)

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
    cudaMalloc(&device_x, size * sizeof(int));

    // Kopiowanie danych z CPU (host) do GPU (device)
    cudaMemcpy(device_x, host_x, size * sizeof(int), cudaMemcpyHostToDevice);

    // Uruchomienie kernela: 20 bloków po 5 wątków = 100 wątków
    squareKernel <<< 20, 5 >>> (device_x, size);

    // Synchronizacja — czekamy aż GPU zakończy pracę
    cudaDeviceSynchronize();

    // Sprawdzenie, czy kernel wykonał się poprawnie
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Blad kernela: " << cudaGetErrorString(err) << "\n";
        delete[] host_x;  // Zwolnienie pamięci na CPU
        return 1;
    }

    // Kopiowanie wyników z GPU z powrotem do CPU
    cudaMemcpy(host_x, device_x, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Wypisanie wyników na konsolę
    for (int i = 0; i < size; ++i)
        std::cout << "x[" << i << "] = " << host_x[i] << "\n";

    // Zwolnienie pamięci na GPU i CPU
    cudaFree(device_x);
    delete[] host_x;

    return 0;
}

```

<h4> 🧠 Co warto zapamiętać? </h4>

- `__global__` oznacza funkcję uruchamianą na GPU
- `>>` to sposób uruchamiania kernela — tutaj 100 wątków
- `cudaMalloc`, `cudaMemcpy`, `cudaFree` — to podstawowe operacje na pamięci GPU
- `cudaDeviceSynchronize()` — czeka, aż GPU zakończy pracę
- `cudaGetLastError()` — pozwala wykryć błędy wykonania kernela

-------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>