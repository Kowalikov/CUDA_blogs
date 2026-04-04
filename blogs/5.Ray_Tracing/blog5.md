---
layout: default
title: Prosty Ray Tracing
permalink: /blog5
---

<!-- blog5 content here -->
<h1> V. Prosty Ray Tracing </h1>  

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
<h3> Render oświetlenia kuli z różnych stron </h3>

W tym wpisie, użyjemy zdobytej wiedzy z poprzednich wpisów, żeby użyć GPU to stworzenia animacji prostej sceny 3D, z kulą oświetloną z różnych stron. Będzie to bardzo uproszczona wersja ray tracingu, ale pozwoli nam na praktyczne zastosowanie kerneli CUDA do renderowania grafiki - czyli tego, do czego GPU zostały stworzone. Wszystkie wersje skryptu, które będziemy omawiać, są dostępne w [repozytorium GitHub](https://github.com/Kowalikov/CUDA_blogs/tree/main/blogs/5.Ray_Tracing).


<h4>Setup renderowanego środowiska</h4>

Do renderowania sceny, potrzebujemy zdefiniować kilka elementów:
- układ współrzędnych, w którym będziemy pracować (kartezjański, z osią Y skierowaną do góry,
  osią X i Z rozpinającymi płaszczyznę poziomą)
- tło, które potraktujemy jako bardzo odległą płaszczyznę, która będzie miała stały kolor 
  (blady granatowo-fioletowy, RGB(30, 30, 50))
- kulę, która będzie umieszczona blisko centrum sceny na współrzędnych (0, 0, -3), i będzie miała promień
  1 jednostki. Kula będzie miała kolor RGB(255, 50, 50) przy maksymalnym świetle. Będzie ona ciemnieć 
  w zależności od kąta padania światła i odległości od źródła światła. Dla każdego piksela, będziemy
  operowali parametrem `intensity`, który będzie określał, jak jasny jest dany punkt na kuli, i będzie się
  wahał od 0 (czarny) do 1 (pełna jasność). Kolor piksela (jeżeli trafi na kulę) będzie wtedy obliczany jako
  `color = base_color * intensity`, gdzie `base_color` to właśnie kolor kuli RGB(255, 50, 50).
- kamerę, która będzie umieszczona na współrzędnych (5, 0, -3) i będzie patrzeć na wprost na kulkę.
  Kamera będzie miała pole widzenia 90 stopni, widząc obszar od `-1 * z` do `1 * z` w osi X i Y. Czyli efektywnie,
  tuż przed kamerą, będziemy projektować wszystkie punkty w odległości `z_i` z ich współrzędnymi `x_i` i `y_i`
  obliczanymi jako `x_i' = x_i / z_i` i `y_i' = y_i / z_i`. Polecam sprawdzić ten [filmik od Tsoding](https://www.youtube.com/watch?v=qjWkNZ0SXfo).
  Najważniejsza informacja dla nas, to że odległość od kamery do najbliższego punktu na kuli będzie wynosić 5-1=4.
  To pozycja startowa. Będziemy animować kamerę, obracając ją wokół kuli, więc jej pozycja będzie się zmieniać
  w czasie. Kamera będzie poruszać się po okręgu o promieniu 5 jednostek wokół kuli, w płaszczyźnie XZ, 
  ze stałą prędkością kątową. Chcemy mieć pełny obrót kamery na wszystkie klatki, więc jeśli mamy 120 klatek,
  przy 60 fpsach to kamera będzie obracać się o 360/120=3 stopnie na klatkę, i 260/120/60=180 stopni na sekundę. 
- źródło światła, które będzie umieszczone blisko kamery, na współrzędnych (0, 1.0, -3.05). 
  Będzie to punktowe źródło światła, które będzie emitować światło we wszystkich kierunkach.
  Jego pozycja jest blisko kuli, powyżej niej i lekko na bok, co oznacza, że będzie oświetlać kulę z góry, delikatnie asymetrycznie.
  Światło będzie miało stałą jasność (`intensity`=1.0) i kolor (białe światło RGB(255, 255, 255)).

Mamy podstawowe elementy sceny, teraz musimy je zaimplementować w kodzie. Będziemy używać kernela
CUDA do renderowania każdego piksela obrazu, i będziemy generować serię klatek, klatka po klatce,
w pętli `for`, które potem połączymy w animację. Zacznijmy od szkieletu maina.

<h4>Szkielet maina</h4>

Pełen skrypt z tej części znajdziesz [tutaj](https://github.com/Kowalikov/CUDA_blogs/blob/main/blogs/5.Ray_Tracing/kernel_1.cu). 
Zaczynamy od importów. Dzisiaj potrzebne nam: 

<!-- make a code snippet in cpp -->
```cpp
#include <cuda_runtime.h> // zarządzanie pamięcią GPU i funkcje CUDA
#include <device_launch_parameters.h> // identyfikatory bloków i wątków w kernelach
#include <fstream> // zapisywanie klatek do plików
#include <iostream> // logowanie postępu i informacji o scenie
#include <cmath> // funkcje matematyczne (sqrt, atan2, sin, cos)
#include <cstdlib> // komendy do tworzenia i usuwania folderów i plików
```

typy `float3` i funkcje wektorowe (dot, cross, normalize) są dostępne w `cuda_runtime.h`, więc nie musimy ich definiować sami.

Dalej zaczynamy od najważniejszych graficznych szczegółów sceny, czyli ilość klatek i rozdzielczość obrazu. 
W tym przypadku, chcemy wygenerować 120 klatek animacji, w rozdzielczości 800x600 pikseli.
Każdy piksel będzie reprezentowany przez 3 bajty (RGB), więc rozmiar pojedynczej klatki w bajtach to `width * height * 3`.
Przy okazji, tworzymy folder `frames`, w którym będziemy zapisywać poszczególne klatki animacji. 
Używamy do tego komendy systemowej `mkdir`, która jest kompatybilna z Windows. 
Komenda `> nul 2>&1` przekierowuje standardowe wyjście i błędy do "czarnej dziury", więc nie zobaczymy komunikatu o błędzie,
jeśli folder już istnieje.

```cpp
int main()
{
    int frames_to_render = 120;

    const int width = 800;
    const int height = 600;
    const int imageSize = width * height * 3;

    // Setup working directory (create 'frames' folder if not exists)
    // Windows command to create folder quietly
    system("mkdir frames > nul 2>&1");
```

Dalej dodajmy dane kuli, światła i kamery, a następnie je wyświetlamy w stylizowanym nagłówku programu. 

```cpp
    // --- SCENE CONFIGURATION ---
    float3 center = make_float3(0, 0, -3);
    float sphereRadius = 1.0f;
    float3 camPosition = make_float3(5.0f, 0, -3.0f);
    float3 lightSource = make_float3(0, 1.0f, -3.05f);

    // Display Application Header
    std::cout << "========================================\n";
    std::cout << "   CUDA RAY TRACER - ORBIT DEMO    \n";
    std::cout << "========================================\n";
    std::cout << "Sphere: Center (" << center.x << ", " << center.y << ", " << center.z;
    std::cout << "), Radius " << sphereRadius << "\n";
    std::cout << "Light position (" << lightSource.x << ", " << lightSource.y << ", " << lightSource.z << ")\n";
    std::cout << "System: Generating " << frames_to_render << " frames of ";
    std::cout << width << "x" << height << " resolution.\n";
    std::cout << "Output: frames/*.ppm -> output.mp4\n";
    std::cout << "----------------------------------------\n";
```

Nasz układ sceny jest gotowy. Przygotujmy utilsy, które będą postawą obliczeń zmian,
klatka po klatce. Całe wideo, ma być animacją obrotu kamery wokół kuli, więc
musimy obliczyć, jak będzie się zmieniać pozycja kamery w czasie. W tym celu,
będziemy pracować z okrągłą orbitą kamery wokół centrum kuli. Potrzebujemy do tego 
promień orbity i początkowy kąt. Dalej, w każdej klatce, będziemy aktualizować kąt
o stałą wartość, co spowoduje obrót kamery. Zaczynamy od dystansów na poziomej
płaszczyźnie (dx i dz) między kamerą a środkiem kuli, i na ich podstawie 
obliczamy promień i kąt startowy.

```cpp
    // --- ORBIT CALCULATION ---
    float dx = camPosition.x - center.x;
    float dz = camPosition.z - center.z;

    // Calculate initial radius and angle based on user input
    float orbitRadius = sqrtf(dx * dx + dz * dz);
    float startAngle = atan2f(dx, dz);
```

Jesteśmy gotowi do renderowania. Najpierw, alokujemy pamięć na hosta (RAM) i na urządzeniu (GPU)
dla obrazu, który będziemy renderować.

```cpp
    // Memory Allocation
    unsigned char* h_image = new unsigned char[imageSize]; // Host
    unsigned char* d_image;
    cudaMalloc(&d_image, imageSize); // Device
```

Teraz zrównoleglenie dla renderowania pojedyńczej klatki. Każdy piksel obrazu będzie renderowany 
przez jeden wątek GPU, więc musimy zdefiniować konfigurację bloków i siatki. Wątki zorganizujemy w bloki
32x16, co daje 512 wątków na blok, co jest dość optymalne dla większości GPU, pod kątem szerokiego zrównoleglenia
bez wąskich gardeł żyłowania GPU pod limity.  

Bloki wątków to podstawa naszego zrównoleglenia. Pozostało nam spiąć bloki w siatkę, żeby każdy piksel
obrazu miał swój wątek. Siatka będzie miała tyle bloków, ile potrzeba, żeby pokryć cały obraz. 

```cpp
    // Konfiguracja wykonania kernela
    // 32x16 = 512 wątków na blok (dość optymalne obłożenie dla większości GPU)
    dim3 block(32, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
```

Przeanalizuj na spokojnie linijkę od definicji `grid`, żeby zrozumieć jej logikę. Nie spiesz się. 
Niektóre wzory w programowaniu mogą wydawać się niepotrzebnie skomplikowane, zwłaszcza w przypadku dość 
generycznych albo rutynowych operacji. Cóż, jest tak. Z tego powodu, daj sobie czasu na ich zrozumienie. 
Dzięki temu, szybciej do nich przywykniesz i gruntownie zrozumiesz, co robi twój kod. 
To prosta, chodź powolna droga, do zostania bardzo dobrym programistą.

Zaczynamy pętlę! Renderujemy klatkę po klatce, przesuwając kamerę wokół kuli o stały kąt `angleStep`.
Chcemy mieć pełen obrót kamery na dystansie wszystkich klatek, stąd wzór $2 \pi / n$ gdzie $n$ to 
liczba klatek.

Na początku ciała pętli, zawsze obliczamy aktualną pozycję kamery, na podstawie jej startowej pozycji i aktualnego kąta.
Dalej, wywołujemy kernel, który renderuje aktualną klatkę, kopiujemy wynik z powrotem na hosta i zapisujemy klatkę do pliku.
Kernel jeszcze nie jest napisany, ale na pewno będzie potrzebował:

- tablicy z pikselami obrazu
- rozmiarów obrazu
- pozycji światła
- aktualnej pozycji kamery
- centrum kuli i jej promienia  

Dalej, tradycyjnie, synchronizacja GPU, kopiowanie danych z powrotem na hosta
i zapis klatki do pliku. Nazwa pliku będzie miała format `frame_XXX.ppm`, 
gdzie `XXX` to numer klatki z zerami wiodącymi, co ułatwi późniejsze łączenie klatek w wideo.
Wywołujemy jeszcze nie zaimplementowaną funkcję zapisu klatki `savePPM`, ale na pewno 
użyje ona obrazu na hoście, wymiarów obrazu i nazwy pliku. Dodajmy na koniec log postępu.

```cpp
    std::cout << "\nStarting rendering loop...\n";

    // --- RENDER LOOP ---
    float angleStep = (2.0f * 3.14159f) / frames_to_render;
    for (int frame = 0; frame < frames_to_render; frame++) {
        // Calculate new camera angle (Full circle in 120 frames)
        float currentAngle = startAngle + (frame * angleStep);

        // Update Camera Position
        float3 currentCamPos;
        currentCamPos.x = center.x + orbitRadius * sinf(currentAngle);
        currentCamPos.y = camPosition.y; // Maintain constant height
        currentCamPos.z = center.z + orbitRadius * cosf(currentAngle);

        // Launch Kernel
        render <<< grid, block >>> (d_image, width, height, lightSource, currentCamPos, center, sphereRadius);
        cudaDeviceSynchronize();

        // Copy back to Host
        cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

        // Save Frame
        char filename[64];
        sprintf(filename, "frames/frame_%03d.ppm", frame);
        savePPM(filename, h_image, width, height);

        // Progress Log
        if (frame % 10 == 0) std::cout << "Rendered frame " << frame << "/" << frames_to_render << "\n";
    }
```

Po udanym przebiegu pętli, będziemy mieli wszystkie klatki zapisane w folderze `frames`.
Użyjemy na nich FFmpeg, z poziomu komendy w terminalu, żeby połączyć te klatki w animację MP4.

Sprzątamy po sobie - usuwamy klatki, zwalniamy pamięć i kończymy program z komunikatem. 

```cpp
    // FFmpeg Video Generation
    std::cout << "Rendering complete. Generating MP4...\n";
    system("ffmpeg -y -framerate 60 -i frames/frame_%03d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4");

    // Cleanup
    system("rm frames/frame_*.ppm"); // Remove individual frames

    cudaFree(d_image);
    delete[] h_image;

    std::cout << "Done! Saved output.mp4\n";
    return 0;
}
```

<h4>Szkielet maina</h4>


<h4> 🔍 Podsumowanie</h4>

[TBD]

------------------------------
<h2> Pytania kontrolne </h2>

1. ...
2. Co oznaczają argumenty i flagi od ffmpega? Jak zmienić framerate i nazwę pliku wynikowego?
Jak zmienić kodek i format wyjściowy?

------------------------------
<h2>Ćwiczenia:</h2>

1. Dostosuj program, żeby użytkownik mógł wprowadzić pozycję kamery i źródła światła z klawiatury, zamiast mieć je na sztywno w kodzie. 
   Upewnij się, że program nadal działa poprawnie z nowymi wartościami. Pobaw się zachowaniem outputu:  
    (a) Czy zauważysz obrót, jeżeli światło będzie dokładnie nad kulą?
    (b) Co się stanie, jeżeli światło będzie dokładnie z tyłu kamery?
    (c) Co się stanie, jeżeli kamera będzie bliżej środka kuli, niż promień kuli (np. na współrzędnych (0, 0, -2))?
    (d) Co się stanie jeżeli kamera będzie bardzo daleko od kuli (np. na współrzędnych (1'000'000, 0, -3))?
2. ...

------------------------------
<break></break>
------------------------------
<h2>Odpowiedzi do pytań kontrolnych:</h2>

1. ...

------------------------------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>

<!-- Previous blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog4">Poprzedni wpis: IV. Prosty benchmark GPU</a></p>

<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog6">Następny wpis</a></p>
