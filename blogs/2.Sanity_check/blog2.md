---
layout: default
title: Prosty benchmark GPU
permalink: /blog2
---

<!-- blog2 content here -->
<h1>Sprawdzanie środowiska CUDA</h1>  

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
<h3>CUDA w C++ - pierwsze testy</h3>

W drugiej odsłonie tej serii sprawdzimy, jak działa GPU z CUDA w C++.

<h4>🧰 Kompilowanie skryptów CUDA C++ `.cu` z `nvcc`</h4>

📌 Przypominajka, jak skompilować i uruchomić program CUDA na Linuxie (na Windowsie będzie podobnie, ale z rozszerzeniem `.exe`):
```bash
nvcc kernel.cu -o kernel && chmod u+x ./kernel && ./kernel
```

<h4>🧰 Test 1: Sprawdzamy, czy CUDA widzi GPU</h4>

Bez GPU ani rusz. Najpierw sprawdzimy, czy CUDA jest w stanie wykryć naszą kartę graficzną NVIDIA. W tym celu wywołamy funkcję `cudaGetDeviceCount`, która zwraca liczbę dostępnych urządzeń CUDA. Jeśli wynik jest większy niż 0, oznacza to, że GPU jest dostępne i gotowe do pracy z CUDA. Alternatywnie, środowisko CUDA mogło zostać niepoprawnie skonfigurowane, i sama funkcja zwróci błąd - ten przypadek też obsługujemy.

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

<h4>🧰 Test 2: Sprawdzamy, jakie obliczenia możemy puścić na GPU</h4>

Jeżeli widzimy GPU, to sprawdźmy, jakie daje nam ona możliwości. Do zaznajomienia się ze specyfikacją GPU użyjemy funkcji `cudaDeviceGetAttribute`, która pozwala nam odczytać różne parametry dotyczące możliwości naszego GPU. Zwraca ona wartość artybutu, o który pytamy w drugim argumencie funkcji, pod zmienną, której adres podajemy w pierwszym argumencie. Ostatni argument to numer urządzenia (zazwyczaj 0, jeśli mamy tylko jedno GPU). Teraz, my będziemy pytać o:
- `cudaDevAttrMaxThreadsPerBlock` - maksymalna liczba wątków, które mogą być uruchomione w jednym bloku.
- `cudaDevAttrMaxGridDimX` - maksymalna liczba bloków, które mogą być uruchomione w jednym wymiarze siatki (grid).
- `cudaDevAttrMaxThreadsPerMultiProcessor` - maksymalna liczba wątków, które mogą być uruchomione na jednym multiprocesorze.
- `cudaDevAttrMultiProcessorCount` - liczba multiprocesorów w GPU.

Trochę tego dużo, ale efektywnie pozwoli nam to zrozumieć, jak duże obliczenia możemy uruchomić na naszym GPU i jak je najlepiej zorganizować.
Rozpakujmy ten żargon. 

<h3> Hierarchia wykonania CUDA — wątki, bloki, siatki i multiprocesory w pigułce </h3>

1) Wątki: podstawowy element pracy równoległej
W CUDA wątek jest najmniejszą jednostką wykonawczą. Każdy wątek uruchamia tę samą funkcję jądra (kernel), ale zazwyczaj przetwarza inny element danych (np. jeden piksel, jeden indeks tablicy, jedną cząsteczkę). Wątki mają własne rejestry i prywatną pamięć lokalną, a także dostęp do pamięci współdzielonej w obrębie bloku oraz pamięci globalnej urządzenia. Na GPU NVIDII wątki są wykonywane w grupach po 32, zwanych warpami. Warp pracuje synchronicznie na jednym Streaming Multiprocessorze (SM). Zachowanie spójnego przepływu sterowania i koaleskowalnych dostępów do pamięci w obrębie warpa jest kluczowe dla wysokiej wydajności.  

2) Bloki wątków: współpraca, pamięć współdzielona i synchronizacja
Blok wątków to grupa wątków, które mogą ze sobą współpracować. Wątki w tym samym bloku mogą synchronizować się za pomocą `__syncthreads()` oraz komunikować poprzez szybką, niskolatencyjną pamięć współdzieloną. Dzięki temu bloki świetnie nadają się do takich wzorców obliczeń jak tiling, redukcje, skany czy operacje stencylowe. Bloki mogą być jednowymiarowe, dwuwymiarowe lub trójwymiarowe (w sensie zbierają w sobie wątki, które można indeksować po wymiarach X, Y, Z) — zależnie od sposobu odwzorowania danych (np. obrazów). Są też niezależne od siebie: sprzęt może wykonywać je w dowolnej kolejności i na różnych SM-ach. Z tego powodu kernel nie powinien zakładać istnienia synchronizacji pomiędzy blokami (poza oddzielnymi wywołaniami kernela lub użyciem wyspecjalizowanych API).  

3) Siatki (grids): uruchamianie wielu bloków, aby pokryć cały problem
Siatka (grid) to zbiór wszystkich bloków uruchomionych w pojedynczym wywołaniu kernela. Podobnie jak bloki, siatki mogą mieć 1–3 wymiary i są tak definiowane, aby pokryć cały przetwarzany zakres danych. Siatka określa całkowity poziom równoległości, jaki będzie dostępny dla GPU. Wątek oblicza swoje globalne indeksy za pomocą wbudowanych zmiennych (`blockIdx`, `blockDim`, `threadIdx`, `gridDim`). Dzięki temu, że bloki są niezależne, ten sam kernel może skalować się na różne rozmiary danych i różne modele GPU — runtime po prostu rozdziela bloki na dostępne SM-y aż do ukończenia pracy.  

4) Multiprocesory strumieniowe (SM): miejsce, gdzie dzieje się wykonanie
*Streaming Multiprocessor* (SM) to jednostka sprzętowa, która faktycznie wykonuje warpy. Każdy SM posiada rdzenie obliczeniowe, pliki rejestrów, pamięć współdzieloną/pamięć L1 oraz harmonogramy warpów. Po uruchomieniu kernela GPU mapuje bloki na dostępne SM-y; pojedynczy SM może utrzymywać wiele bloków jednocześnie, jeżeli pozwalają na to zasoby (rejestry na wątek, pamięć współdzielona na blok, maksymalna liczba aktywnych wątków/warpów/bloków). *Occupancy* — czyli ile warpów jest aktywnych w stosunku do maksymalnych możliwości SM — wpływa na zdolność ukrywania opóźnień pamięci. Jednak maksymalna occupancy nie zawsze oznacza maksymalną wydajność: kluczowe jest znalezienie balansu między użyciem zasobów a liczbą aktywnych warpów.  

5) Relacje między nimi: planowanie, wydajność i decyzje projektowe
Podsumowując: host uruchamia siatkę złożoną z bloków; GPU przypisuje te bloki do SM-ów; SM wykonuje je jako warpy zbudowane z wątków. Wynikają z tego ważne implikacje dla wydajności. Bloki powinny mieć rozmiary będące wielokrotnością 32, aby unikać częściowych warpów. Pamięci współdzielonej należy używać do ponownego wykorzystania danych i optymalizacji lokalności — z uwzględnieniem tego, że jej użycie ogranicza liczbę bloków rezydujących na SM. Dostępy do pamięci globalnej powinny być koaleskowalne w obrębie warpa. I wreszcie, logika kernela powinna opierać się na niezależności bloków, co umożliwia schedulerowi elastyczny podział pracy i naturalne skalowanie między różnymi GPU.


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
To oznacza, że mogę uruchomić do 1024 wątków w jednym bloku, mogę mieć ogromną liczbę bloków w siatce (2^31-1), a na jednym multiprocesorze mogę mieć do 1024 wątków. Mój GPU ma 14 multiprocesorów, więc teoretycznie może obsłużyć do 14 * 1024 = 14336 wątków jednocześnie, ale to zależy od wielu czynników, takich jak zasoby pamięci i harmonogramowanie. Te informacje są kluczowe, gdy będziemy projektować nasze obliczenia na GPU, aby upewnić się, że wykorzystujemy jego możliwości w sposób efektywny.  

Po więcej informacji o hierarchii wykonania CUDA i jej implikacjach dla projektowania kerneli, polecam zajrzeć do dokumentacji NVIDII oraz przewodnika [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), gdzie jest to omówione bardzo szczegółowo.


<h4> 🔍 Podsumowanie</h4>

W tym wpisie sprawdziliśmy, czy nasze środowisko CUDA jest poprawnie skonfigurowane i czy możemy wykryć nasz GPU. Następnie odczytaliśmy specyfikację naszego GPU, co pozwoli nam lepiej zrozumieć, jakie obliczenia możemy na nim uruchomić i jak je zoptymalizować. W kolejnych częściach tej serii będziemy już pisać konkretne programy CUDA, które będą wykorzystywać te możliwości do wykonywania różnych zadań obliczeniowych.

-------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>

<!-- Previous blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog1">Poprzedni wpis</a></p>

<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog3">Następny wpis</a></p>
