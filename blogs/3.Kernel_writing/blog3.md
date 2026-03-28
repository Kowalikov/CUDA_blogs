---
layout: default
title: Pisanie funkcji na GPU
permalink: /blog3
---

<!-- blog3 content here -->
<h1>III. Pisanie funkcji na GPU</h1>  

<author> <b>Autor:</b> Natan Jarzyński, Marek Kowalik </author>  
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
<h3>Jak zrobić prostą funkcję wykonywaną na GPU</h3>

W trzeciej odsłonie tej serii pokażemy, jak od podstaw napisać funkcję wykonywaną na GPU z CUDA w C++ - tzw. kernel. Pokażemy, jak alokować pamięć na GPU, kopiować dane między CPU a GPU, a następnie uruchomić prosty kernel mnożący macierze.

<h4>🧰 Funkcje na GPU</h4>

<h3>Dostęp do danych w kernelu: identyfikator wątku i bloku</h3>

Na początek, napiszmy prosty kernel, który będzie wykonywany na GPU a jego funkcja, to zorientować się jaki blok i wątek wykonuje go w danym momencie. Kernel to funkcja oznaczona specjalnym modyfikatorem `__global__`, która jest wywoływana z CPU, ale wykonywana na GPU.

W tym celu potrzebujemy dwóch importów:
- `#include <cuda_runtime.h>` - główna biblioteka CUDA do zarządzania pamięcią i urządzeniem. Bez niej nie napiszemy kernela ani nie będziemy mogli alokować pamięci na GPU.
- `#include <device_launch_parameters.h>` - zawiera definicje parametrów uruchamiania kernela, takich jak `blockIdx`, `threadIdx`, `blockDim`, które pozwalają nam zorientować się, który wątek i blok wykonuje dany fragment kodu.

Nasz przykładowy kernel bedzie prosty - wyłuskamy z niego informacje, jak jest wykonywany. Zacznijmy od krótkiego wyjaśnienia jak działają kernele, po co się je wywołuje i jak są organizowane.

Najważniejszym celem angażowania GPU w obliczenia jest masowe zrównoleglenie pracy. GPU składa się z tysięcy rdzeni, które mogą wykonywać obliczenia równolegle. Aby to wykorzystać, musimy napisać kod, który będzie wykonywany przez wiele wątków jednocześnie. Kernel jest właśnie takim kodem - jest to funkcja, która jest uruchamiana na GPU i może być wykonywana przez wiele wątków równocześnie.

Możesz je traktować jako wnętrze funkcji `for`. W prostym scenariuszu, weźmy pętlę z 20 iteracjami, które za każdym razem podnoszą iterator do kwadratu. Możemy napisać to jako kernel, który jest uruchamiany przez 20 wątków, a każdy wątek będzie odpowiedzialny za jedną iterację tej pętli. Tylko, chwila, skąd dane wywołanie ma wiedzieć, która iteracja jest jego zadaniem? Zacznijmy od organizacji wątków, bloków i siatek, a potem wrócimy do tego pytania.


Rozmawialiśmy o wątkach, a tutaj nagle dochodzą bloki i siatki. Wyjaśnijmy to. Wątki są organizowane w bloki, a bloki są organizowane w siatki. Blok to grupa wątków, które mogą współdzielić pamięć i synchronizować się ze sobą. Siatka to grupa bloków, które mogą być uruchamiane na GPU. Kiedy uruchamiamy kernel, musimy określić, ile bloków i ile wątków na blok chcemy uruchomić - to tworzy naszą siatkę. Na przykład, jeśli chcemy uruchomić 2 bloki po 4 wątki każdy, napiszemy `kernel <<< 2, 4 >>> ();`. Oznacza to, że uruchomimy 2 bloki, a każdy blok będzie miał 4 wątki, co daje nam łącznie 8-wątkową siatkę wykonującą nasz kernel. W prostej analogii, siatka to wybudowany budynek fabryki, składający się z bloków (pięter i ich sekcji), a wątki, to pracownicy wykonujący zadania w tych blokach. Każdy pracownik (wątek) wie, w którym bloku (piętrze) pracuje i jakie zadanie ma wykonać, dzięki wbudowanym zmiennym, które mówią mu, gdzie się znajduje w strukturze siatki. Jeżeli mamy 5 bloków po 10 wątków, to mamy 50 pracowników, którzy mogą wykonywać zadania równolegle, a każdy z nich wie, które zadanie ma wykonać dzięki swojej pozycji w strukturze siatki.

Teraz, wątki i bloki, możemy organizować w trzech wymiarach (x, y, z), co pozwala nam tworzyć bardziej złożone struktury siatki. Na przykład, możemy mieć 2 bloki w wymiarze x i 3 bloki w wymiarze y, co daje nam 6 bloków w sumie. Każdy blok może mieć 4 wątki w wymiarze x i 5 wątki w wymiarze y, co daje nam 20 wątków na blok. W ten sposób, możemy tworzyć siatki o różnych kształtach i rozmiarach, które są dostosowane do naszych potrzeb obliczeniowych. Nie oznacza to `fizycznej` organizacji wątków na GPU, a jedynie ułatwienie pracy na wielowymiarowych danych, takich jak obrazy czy macierze, gdzie naturalnie pasuje nam organizacja w trzech wymiarach.  

W naszej analogii, możemy pomyśleć o danym bloku z indeksami x, y, z jako bloku w budynku x, piętrze y, sekcji z, a wątki w tym bloku jako pracownicy wykonujący zadania w tej konkretnej lokalizacji. Dalej, pracownicy mogą być pogrupowani w trzy wymiary, i tak, każdy z nich pracuje w jednosce x, dziale y (jednostki x), z numerem pracownika z. 

Wróćmy do pytania, skąd w kernelu wziąć te indeksy? Odpowiedź jest prosta - każdy wątek ma wbudowane zmienne, które mówią mu, jakim jest wykonaniem. Te zmienne to `blockIdx` i `threadIdx`. Można z nich wyłuskać indeksy x, y i z. Na przykład, `blockIdx.x` daje nam indeks bloku w wymiarze x, `threadIdx.x` daje nam indeks wątku w wymiarze x.
Oprócz indeksów, wymiary bloków i siatki mogą być pomocne i są one dostępne przez `blockDim` i `gridDim`. 

Dzięki tym zmiennym, każdy wątek może obliczyć swoją unikalną pozycję w siatce i wiedzieć, które zadanie ma wykonać, oraz w jakich ramach i limitach pracuje.

<!-- make a code snippet in cpp -->
```cpp
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
        return 1;
    }

    return 0;
}
```

Dość egzotyczna składnia, prawda? Nazwa oddzielona od nawiasów do wywołania, a w nich liczby określające rozmiar siatki i bloku. Ale dzięki temu składnia CUDA C++ jest tak wyraźna i charakterystyczna.  

Co z przypadkiem gdy potrzebujemy siatkę wątków albo bloków? Możemy użyć dim3 do określenia wymiarów siatki i bloków. Na przykład, `dim3 blockSize(4, 4, 1);` tworzy blok o wymiarach 4x4x1, a `dim3 gridSize(2, 2, 1);` tworzy siatkę o wymiarach 2x2x1. Następnie możemy uruchomić kernel używając tych zmiennych: `kernel <<< gridSize, blockSize >>> ();`.

Oprócz wywołania, mamy też synchronizację, która jest ważna, ponieważ kernel jest wykonywany asynchronicznie. Musimy poczekać, aż GPU zakończy pracę, zanim przejdziemy dalej w kodzie. Będzie to ważne, gdy będziemy chcieli odczytać wyniki z GPU i przenieść je na CPU, po zakończeniu pracy wszystkich wątków.  

Na koniec sprawdzamy, czy kernel wykonał się poprawnie, używając `cudaGetLastError()`. Oprócz sprawdzenia, możemy też wypisać ewentualny błąd, jeśli coś poszło nie tak.  

Teraz, wróćmy do samego kernela. Nie jest to najbardziej użyteczna funkcja. Żeby była praktyczna, potrzebujemy rozszerzyć ją o zwracanie, modyfikowanie i wyświetlanie danych. Jak to zrobić? Rozważmy parę przykładów.

<h3>Dostęp do danych w kernelu: jak przekazywać argumenty do kernela</h3>

Na początek przekażmy do kernela informacyjnie jeden integer i tablicę floatów do modyfikacji. Ten sam przykład co z bloga 1. 
Zwróćmy uwagę na argumenty funkcji kernela. Nie ma żadnej różnicy między argumentami funkcji kernela a zwykłej funkcji C++. Możemy przekazywać dowolne typy danych, w tym wskaźniki do tablic. Jednakże, gdy przekazujemy wskaźnik do tablicy, musimy pamiętać, że ta tablica musi być alokowana na GPU, a nie na CPU.

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
```
Przejdźmy więc do maina i przykładowego użycia. 

<h3>Kernel w użyciu: prosta demonstracja</h3>

Pierwsze 5 linijek to czysty C++. Zabawa zaczyna się od alokacji pamięci na GPU.  

Używamy do tego wrappera `cudaMalloc`, który jest funkcją CUDA do alokacji pamięci na urządzeniu. Musimy podać wskaźnik do wskaźnika, aby `cudaMalloc` mógł zwrócić adres alokowanej pamięci. Uwaga! Wskaźnik, który przekazujemy do `cudaMalloc`, musi być typu wskaźnika do typu danych, który chcemy alokować. W naszym przypadku jest to `int*`, ponieważ chcemy alokować tablicę integerów. Oprócz tego, wskaźnik ten znajduje się fizycznie po stronie CPU (a w zasadzie hosta, żeby być precyzyjnym), ale `cudaMalloc` alokuje pamięć na GPU (device) i zwraca adres tej pamięci, który jest przypisywany do naszego wskaźnika.  

Teraz, `cudaMalloc` potrzebuje rozmiaru pamięci do alokacji, który jest obliczany jako liczba elementów tablicy pomnożona przez rozmiar pojedynczego elementu (w naszym przypadku `sizeof(int)`). 

Jeżeli wszystko poszło dobrze, `cudaMalloc` zwróci `cudaSuccess`, a nasz wskaźnik będzie wskazywał na alokowaną pamięć na GPU. Jeśli coś poszło nie tak, możemy sprawdzić błąd za pomocą `cudaGetLastError()` i wypisać odpowiedni komunikat. Tak, CUDA nie rzuca wyjątków, więc musimy ręcznie sprawdzać błędy po każdej operacji CUDA. Jest to jeden z powodów, dlaczego tak cenieni są specjaliści od CUDA - muszą być bardzo skrupulatni w sprawdzaniu błędów i zarządzaniu pamięcią. 

```cpp
int main() {
    const int size = 100;  // Rozmiar tablicy
    int* host_x = new int[size];  // Dynamiczna alokacja pamięci na CPU (na stercie)

    // Inicjalizacja tablicy wartościami od 0 do 99
    for (int i = 0; i < size; ++i)
        host_x[i] = i;

    // Alokacja pamięci na GPU (device)
    int* device_x;
    cudaMalloc(&device_x, size * sizeof(int));
```

Miejsce na GPU zaalokowane, ale co z danymi? Musimy je przenieść z CPU na GPU, aby kernel mógł je przetwarzać. Do tego służy funkcja `cudaMemcpy`, która kopiuje dane między hostem a device'em. W naszym przypadku, chcemy skopiować dane z `host_x` (CPU) do `device_x` (GPU). `cudaMemcpy` potrzebuje czterech argumentów: docelowego wskaźnika, źródłowego wskaźnika, rozmiaru danych do skopiowania oraz kierunku kopiowania (w naszym przypadku `cudaMemcpyHostToDevice`, ponieważ kopiujemy z hosta na device).

Ponownie, po wywołaniu `cudaMemcpy`, powinniśmy sprawdzić, czy operacja zakończyła się sukcesem, używając `cudaGetLastError()`. Jeśli wszystko poszło dobrze, dane będą teraz dostępne na GPU i gotowe do przetwarzania przez kernel.

```cpp
    // Kopiowanie danych z CPU (host) do GPU (device)
    cudaMemcpy(device_x, host_x, size * sizeof(int), cudaMemcpyHostToDevice);
```

Teraz możemy uruchomić kernel, który podniesie każdy element tablicy do kwadratu. W tym celu używamy składni `<<<...>>>`, gdzie pierwsza liczba (20) to liczba bloków, a druga liczba (5) to liczba wątków na blok. Oznacza to, że uruchomimy 20 bloków, a każdy blok będzie miał 5 wątków, co daje nam łącznie 100 wątków. Każdy wątek będzie odpowiedzialny za przetworzenie jednego elementu tablicy.

Ale moment, co ze zmienną size? Przecież jest na hoście. Zgadza się, ale jeżeli przekazujemy int, albo ogólniej typ prosty (int, float, double, char, itp.) jako argument do kernela, jest przekazywana jego wartość. Oznacza to, że kernel otrzyma kopię tej wartości i będzie mógł z niej korzystać bez problemu. W przypadku bardziej złożonych struktur danych, takich jak tablice czy struktury, musielibyśmy zadbać o to, aby były one alokowane na GPU i przekazywane jako wskaźniki.

Zanim pójdziemy dalej, kładziemy blocker `cudaDeviceSynchronize()` czekający, aż GPU zakończy pracę. Musimy z niego skorzystać, ponieważ interesują nas wyniki obliczeń kernela, a bez synchronizacji moglibyśmy próbować odczytać wyniki z GPU zanim kernel zakończy swoje działanie, co prowadziłoby do błędów i nieprzewidywalnych wyników.

Na koniec sprawdzamy, czy kernel wykonał się poprawnie, używając `cudaGetLastError()`. Jeśli kernel zakończył się błędem, wypisujemy komunikat o błędzie i zwalniamy pamięć zarówno na GPU, jak i na CPU, zanim zakończymy program z kodem błędu.

```cpp
    // Uruchomienie kernela: 20 bloków po 5 wątków = 100 wątków
    squareKernel <<< 20, 5 >>> (device_x, size);

    // Synchronizacja — czekamy aż GPU zakończy pracę
    cudaDeviceSynchronize();

// Sprawdzenie, czy kernel wykonał się poprawnie
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Blad kernela: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_x);  // Zwolnienie pamięci na GPU
        delete[] host_x;  // Zwolnienie pamięci na CPU
        return 1;
    }
```

Jak wszystko jest ok, to czas na odczyt wyników. Ponownie używamy `cudaMemcpy`, ale tym razem kopiujemy dane z GPU (device) z powrotem na CPU (host). Kierunek kopiowania to `cudaMemcpyDeviceToHost`. Sprawdzamy, czy operacja zakończyła się sukcesem, i wypisujemy wyniki na konsolę, aby zobaczyć wyniki działania kernela.

Nie zapomnijmy o zwolnieniu pamięci zarówno na GPU, jak i na CPU, aby uniknąć wycieków pamięci. Na GPU używamy `cudaFree`, a na CPU `delete[]` dla tablicy dynamicznej.

```cpp
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

Nasz program ma wszystkie podstawowe funkcjonalności, ale sprawdzi się tylko, jeżeli wszystko pójdzie dobrze. W praktyce, podczas pracy z CUDA, często napotykamy na różne błędy, takie jak błędy alokacji pamięci, błędy kopiowania danych, czy błędy w kernelu. Dlatego tak ważne jest, aby zawsze sprawdzać wyniki operacji CUDA i być przygotowanym na obsługę błędów. To do dzieła! 

<h3>Kernel w użyciu: demonstracja kuloodporna</h3>

Zaczynamy od handlera błędów CUDY. Będzie to jedna funkcja, bo CUDA umożliwia nam sprawdzenie, czy kod CUDA wykonał się poprawnie, zwracając `cudaSuccess` z funkcji `cudaGetLastError()`. Ogólnie, to co zwraca `cudaGetLastError()` jest typu `cudaError_t`, i może być zarówno `cudaSuccess`, jak i różnymi kodami błędów, które możemy sprawdzić za pomocą `cudaGetErrorString()`, aby uzyskać czytelny komunikat o błędzie. Teraz, nasz handler będzie drukował komunikat, który przekażemy i zwróci wartość 1 w przypadku błędu, albo 0 w przypadku sukcesu; pod integrację z warunkowym przerwaniem maina.

```cpp
// Helper do radzenia sobie z błędami CUDA
int checkCuda(cudaError_t result, const char* msg) {

    if (result != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(result) << "\n";
        return 1; 
    }
    return 0;
}
```

Teraz zmiany w maine. Po pierwsze, po alokacji pamięci na GPU, sprawdzamy, czy `cudaMalloc` zwrócił `cudaSuccess`. Jeżeli nie, to zwalniamy pamięć na CPU (i GPU, jeśli została alokowana) i kończymy program z kodem błędu.

```cpp
    // Alokacja pamięci na GPU (device)
    int* device_x;
    if (checkCuda(cudaMalloc(&device_x, size * sizeof(int)), "cudaMalloc device_x")) {
        delete[] host_x; 
        if (device_x) cudaFree(device_x);

        return 1;
    }
```

Dalej, obsługujemy kopiowanie danych. Tutaj alokacja tablicy na GPU została zakończona sukcesem, więc na pewno musimy ją zwolnić w razie błedu.

```cpp
    // Kopiowanie danych z CPU (host) do GPU (device)
    if (checkCuda(cudaMemcpy(device_x, host_x, size * sizeof(int), cudaMemcpyHostToDevice), "Memcpy host_array -> device_array") ) {
        delete[] host_x; 
        cudaFree(device_x);
        
        return 1;
    }
```

Kolejno, potrzebujemy sprawdzić synchronizację i wykonanie kernela. Jeżeli kernel zakończy się błędem, to również musimy zwolnić pamięć i zakończyć program z kodem błędu. Zwróć uwagę, że poprawność wykonania kernela sprawdzaliśmy wcześniej, ale z nowym handlerem mamy bardziej elegancką formę.

```cpp
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
```

Ostatnia zamiana to obudowanie kopiowania wyników z GPU z powrotem do CPU. 

```cpp
    // Kopiowanie wyników z GPU z powrotem do CPU
    if (checkCuda(cudaMemcpy(host_x, device_x, size * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy device_array -> host_array")) {
        delete[] host_x;
        cudaFree(device_x); 
        
        return 1;
    }
```

Pamiętaj, że pełne skrypty CUDA powinny zawsze zawierać obsługę błędów, aby zapewnić stabilność i łatwość debugowania. Dzięki temu, nawet jeśli coś pójdzie nie tak, będziemy mieli jasny komunikat o błędzie i nasz program nie będzie się zawieszał lub zachowywał nieprzewidywalnie.

Co do kodu, masz pełne skrypty na źródłowym GitHbubie, więc możesz je skopiować bezpośrednio stamtąd, bez update'owania wyrywków z bloga.

------------------------------

<h4> 🔍 Podsumowanie</h4>

W tym wpisie pokazaliśmy, jak napisać prosty kernel CUDA. Stworzyliśmy, pełen skrypt, używający danych z hosta i przeprowadzący obliczenia na GPU. Omówiliśmy, jak alokować pamięć na GPU, kopiować dane między CPU a GPU, uruchamiać kernel i sprawdzać jego poprawność. Pokazaliśmy też, jak obsługiwać błędy CUDA, aby nasz program był bardziej stabilny i łatwiejszy do debugowania, i obudowaliśmy nasz skrypt pod obsługę wszystkich błedów i bezpieczne wykonanie od początku do końca. 


------------------------------
<h2> Pytania kontrolne </h2>

1. Jaki modyfikator `__<...>___` służy do zdefiniowania funkcji kernela w CUDA?
2. Będąc w kernelu, możemy zidentyfikować kilka wbudowanych zmiennych, które dostarczają informacji o pozycji wątku w siatce i bloku.  
    (a) Jak wywołać indeksy bloku?  
    (b) Jakie identykiatory wywołania kernela znasz, które możesz odczytać?

3. Dlaczego musimy używać `cudaDeviceSynchronize()` po uruchomieniu kernela, zanim przejdziemy dalej w kodzie?

4. Sprawdzanie błędów:  
    (a) Jak sprawdzić ostatni błąd CUDA?  
    (b) Jak wypisać jego komunikat?  
    (c) Co się stanie, jeżeli sprawdzimy ostatni błąd CUDA po udanym wywołaniu kernela?

------------------------------
<h2>Ćwiczenia:</h2>

1. Zmodyfikuj kernel i jego wywołanie, żeby przyjmował w dalszym ciągu tablicę jednowymiarową, ale tym razem o rozmiarze 512, i wywołaj jeden blok i 512 wątków zorganizowanych jako {8, 8, 8}. Nie zapomnij o zaktualizowaniu indeksowania w kernelu, aby poprawnie obliczyć globalny indeks wątku w siatce.

2. Napisz nowy kernel, który najpierw podniesie każdy element tablicy do kwadratu, odłoży dane z GPU do pamięci hosta, a następnie doda 10 do każdego elementu i odłoży nowe wyniki do innej tablicy na hoście. Nie zapomnij o synchronizacji i sprawdzaniu błędów.

3. Napisz program, który doda dwie małe tablice. Zdefiniuj kernel, który przyjmuje dwie tablice wejściowe i jedną tablicę wyjściową, i dodaje elementy z dwóch tablic wejściowych, zapisując wynik w tablicy wyjściowej. Uruchom kernel z odpowiednią konfiguracją bloków i wątków, a następnie skopiuj wyniki z GPU z powrotem na CPU i wypisz je.

------------------------------
<break></break>
------------------------------
<h2>Odpowiedzi do pytań kontrolnych:</h2>

1. Modyfikator `__global__` służy do zdefiniowania funkcji kernela w CUDA. Oznacza on, że funkcja jest uruchamiana na GPU i może być wywoływana z CPU. Składniowo, kernel jest definiowany jako `__global__ <typ_zwracany> nazwa_kernela(...) { ... }`.
2. (a) Indeksy bloku można wywołać za pomocą wbudowanych zmiennych `blockIdx.x`, `blockIdx.y`, i `blockIdx.z` dla odpowiednio wymiarów x, y, i z.  
   (b) Oprócz `blockIdx`, mamy też `threadIdx` do identyfikacji indeksów wątku w bloku, `blockDim` do określenia rozmiaru bloku, oraz `gridDim` do określenia rozmiaru siatki.

3. Musimy używać `cudaDeviceSynchronize()` po uruchomieniu kernela, ponieważ kernel jest wykonywany asynchronicznie. Oznacza to, że CPU może kontynuować wykonywanie kodu, zanim GPU zakończy pracę nad kernelem. Jeśli chcemy mieć pewność, że kernel zakończył się przed przejściem dalej w kodzie (na przykład, jeśli chcemy odczytać wyniki z GPU), musimy zsynchronizować CPU z GPU, używając `cudaDeviceSynchronize()`. Bez tej synchronizacji, moglibyśmy próbować odczytać dane z GPU zanim kernel zakończy swoje działanie, co prowadziłoby do błędów i nieprzewidywalnych wyników.

4. (a) Aby sprawdzić ostatni błąd CUDA, używamy funkcji `cudaGetLastError()`, która zwraca kod błędu typu `cudaError_t`.  
   (b) Aby wypisać komunikat błędu, możemy użyć funkcji `cudaGetErrorString(cudaError_t error)`, która zwraca czytelny komunikat opisujący błąd.  
   (c) Jeżeli sprawdzimy ostatni błąd CUDA po udanym wywołaniu kernela, `cudaGetLastError()` zwróci `cudaSuccess`, co oznacza, że nie wystąpił żaden błąd. W takim przypadku, `cudaGetErrorString(cudaSuccess)` zwróci komunikat "no error", co potwierdzi, że kernel wykonał się poprawnie.

------------------------------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>

<!-- Previous blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog2">Poprzedni wpis: II. Sprawdzanie środowiska CUDA</a></p>

<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog4">Następny wpis: IV. Prosty benchmark GPU</a></p>
