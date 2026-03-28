---
layout: default
title: Ustawienie środowiska CUDA na Windowsie i Linuxie
permalink: /blog1
---

<!-- blog1 content here -->
<h1>I. CUDA C++: Ustawienie środowiska CUDA na Windowsie i Linuxie </h1>  

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
<h3>CUDA w C++ - od instalacji do pierwszego programu</h3>

W pierwszej części tej serii pokażemy, jak przygotować środowisko programistyczne do pracy z CUDA w C++, a następnie napiszemy i uruchomimy prosty program, który podnosi liczby do kwadratu na karcie graficznej NVIDIA. 


<h4>🧰 Krok 1: Instalacja CUDA Toolkit</h4>

>Upewnij się, że masz kompatybilną kartę graficzną NVIDIA (np. z serii GeForce, Quadro, Tesla).

1. Przejdź na stronę pobierania CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
2. Wybierz swój system operacyjny (Windows, Linux):
    - **Windows**: Najprostszym sposobem w naszej opinii, jest skorzystanie z pakietu CUDA dla Windows, który zawiera wszystkie niezbędne komponenty, w tym sterowniki i narzędzia programistyczne. 
    - **Linux**: Na Linuxie możesz skorzystać z menedżera pakietów (np. `apt` na Ubuntu) lub pobrać instalator ze strony NVIDIA.
    
        >Pamiętaj, że na Linuxie może być konieczne ręczne zainstalowanie sterowników NVIDIA przed instalacją CUDA Toolkit. 
        
        Z naszego doświadczenia, na linuxie najlepiej jest użyć Ubuntu 22.04/24.04 LTS, które są szeroko wspierane przez NVIDIA i mają dobre wsparcie dla sterowników i narzędzi CUDA. 

        > Możesz na swoim Windowsie postawić Ubuntu 24.04 LTS, przez WSL (Windows Subsystem for Linux), co pozwoli Ci korzystać z narzędzi Linuxowych i jednocześnie mieć dostęp do GPU NVIDIA. Zazwyczaj GPU jest dostępna w WSL automatycznie w nowszych Windowsach. W przypadku braku automatycznej integracji, konfiguracja WSL z obsługą GPU może być nieco bardziej skomplikowana niż tradycyjna instalacja na natywnym Linuxie, ale jest to świetna opcja dla programistów, którzy chcą korzystać z obu środowisk, mimo wszystko. Tutaj [instrukcja](https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/) do postawienia Ubuntu 24.04 LTS przez WSL.
3. Pobierz i zainstaluj:
    - [Jeżeli nie są zainstalowane, **jeżeli są, nie ruszaj!**] Sterowniki NVIDIA (upewnij się, że są aktualne)
    - CUDA Toolkit (zawiera kompilator nvcc, biblioteki, przykłady)

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

- [Łatwa opcja] **Visual Studio (Windows)** — CUDA integruje się automatycznie, oprócz tego, [wspiera WSL](https://code.visualstudio.com/docs/remote/wsl), więc możesz pisać kod w Linuxie ze swojego WSL, mając graficzne IDE! 
- [Trudniejsza opcja] **CLion** — wygodny dla CMake, wymaga ustawienia toolchaina z `nvcc`

<h4>🧪 Krok 3: Test z pierwszym programem CUDA</h4>

Poniżej znajduje się kod, który:

- Tworzy tablicę 100 liczb
- Przesyła ją na GPU
- Uruchamia 100 wątków, każdy podnosi jedną liczbę do kwadratu
- Kopiuje wynik z powrotem na CPU i wypisuje go


<h4>📦 Kod źródłowy z komentarzami</h4>

Jest to prosty skrypt w CUDA C++, który pokazuje, jak alokować pamięć na GPU, kopiować dane, uruchamiać kernel i synchronizować wyniki. Potraktujemy go jako "sanity check" — czyli podstawowy test, który pozwoli nam upewnić się, że środowisko CUDA jest poprawnie skonfigurowane i że możemy uruchamiać programy na GPU. W kolejnych częściach będziemy omawiać już określone funkcjonalności w szczególe. Rzuć okiem na kod i jego komentarze, a następnie wklej go do pliku `kernel.cu` i uruchom go na swoim komputerze z GPU NVIDIA, za pomocą komendy `nvcc`. 

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

Na linuxie możesz użyć terminala (kompilując program, dodając uprawnienia do wykonania skompilowanego pliku i uruchamiając go):
```bash
nvcc kernel.cu -o kernel 
chmod u+x ./kernel
./kernel
```
a na Windowsie PowerShell lub CMD:
```powershell
nvcc kernel.cu -o kernel.exe
.\kernel.exe
```

<span style="color: red;">**Zapisz sobie te komendy!**</span> Będziesz z nich korzystać co sekundę, gdy będziesz pisać i testować programy CUDA.

<h4> 🧠 Najważniejsze elementy skryptu ⬆️ </h4>

- `__global__` oznacza funkcję uruchamianą na GPU
- `>>` to sposób uruchamiania kernela — tutaj 100 wątków
- `cudaMalloc`, `cudaMemcpy`, `cudaFree` — to podstawowe operacje na pamięci GPU
- `cudaDeviceSynchronize()` — czeka, aż GPU zakończy pracę
- `cudaGetLastError()` — pozwala wykryć błędy wykonania kernela

<h4> 🔍 Podsumowanie</h4>
W tym tutorialu w 3 krokach pokazaliśmy, jak zacząć używać CUDA Toolkit i przetestować z prostym programem w CUDA C++. W kolejnych częściach zagłębimy się w szczegóły programowania równoległego z CUDA. 

-------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>


<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog2">Następny wpis: II. Sprawdzanie środowiska CUDA</a></p>