---
layout: default
title: Prosty Ray Tracing
permalink: /blog5
---

<!-- blog5 content here -->
<h1> V. Prosty Ray Tracing</h1>  

<author> <b>Autor:</b> Natan Jarzyński, Marek Kowalik</author>  
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
  wahał od 0.1 (prawie czarny, pełny cień) do 1 (pełna jasność). Kolor piksela (jeżeli trafi na kulę) będzie wtedy obliczany jako
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

<h3>Szkielet maina</h3>

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

Jesteśmy gotowi do renderowania. Najpierw, alokujemy pamięć na hosta (RAM) i na urządzeniu (GPU) dla obrazu, który będziemy renderować. UWAGA! Są to dane w formacie tablicy `char`'ów,przygotowanym do zapisania do formatu pliku PPM, używając natywnych handlerów zapisu plików. 
Efektywnie, wszystkie obliczenia do renderowania, również na GPU, będą przeprowadzane oczywiście na liczbach, jednak za każdym razem, gdzie zapisujemy dane do `h_image` albo `d_image`, musimy odpowiednio je zrzutowować. 

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
liczba klatek. Korzystamy z radianów, nie kątów, ze względu na kompatybilność z wbudowanymi funkcjami C++ i prostotę obliczeń.

Na początku ciała pętli, zawsze obliczamy aktualną pozycję kamery, na podstawie jej startowej pozycji i aktualnego kąta.
Dalej, wywołujemy kernel, który renderuje aktualną klatkę, kopiujemy wynik z powrotem na hosta i zapisujemy klatkę do pliku.
Kernel jeszcze nie jest napisany, ale na pewno będzie potrzebował:

- tablicy z pikselami obrazu
- rozmiarów obrazu
- pozycji światła
- aktualnej pozycji kamery
- centrum kuli i jej promienia  

Dalej, tradycyjnie, synchronizacja GPU, kopiowanie danych z powrotem na hosta
i zapis klatki do pliku. Klatki będziemy zapisywać w formacie w postaci Portable Fixmap (PPM), szeroko używanego do podręcznego przechowywania klatek przed sparsowaniem.
Nazwa pliku będzie miała format `frame_XXX.ppm`, gdzie `XXX` to numer klatki z zerami wiodącymi, co ułatwi późniejsze łączenie klatek w wideo.
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
Tutaj [manual do obsługi komendy ffmpeg](https://linux.die.net/man/1/ffmpeg).
> **UWAGA!** Do uruchomienia komendy ffmpeg, musisz mieć go zainstalowanego na swojej maszynie. Na linuxie wystarczy `sudo apt install ffmpeg`.  

```cpp
    // FFmpeg Video Generation
    std::cout << "Rendering complete. Generating MP4...\n";
    system("ffmpeg -y -framerate 60 -i frames/frame_%03d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4");
```

Na koniec, sprzątamy po sobie - usuwamy klatki, zwalniamy pamięć i kończymy program z komunikatem. 

```cpp
    // Cleanup
    system("rm frames/frame_*.ppm"); // Remove individual frames

    cudaFree(d_image);
    delete[] h_image;

    std::cout << "Done! Saved output.mp4\n";
    return 0;
}
```

<h3>Kernel i reszta funkcji</h3>

Pełen skrypt z tej części znajdziesz [tutaj](https://github.com/Kowalikov/CUDA_blogs/blob/main/blogs/5.Ray_Tracing/kernel_2.cu).  
Mamy dwie główne rutyny do napisania - kernel `render` i `savePPM`. Zacznijmy od prostszej i bardziej kompaktowej `savePPM`.

Argumenty to będą:
- nazwa pliku (koniecznie bez zmian dla pojedyńczego wywołania - stąd `const`)
- dane - tablica `char`'ów, dla wygody zapisywania
- szerokość
- wysokość

W takiej formie, funkcja ma bardzo proste przebieg. Otwieramy plik w binarnym trybie.
Jeżeli się nie powiodła, konczymy funkcję błędęm. Dalej dodajemy nagłówek, determinujący format danych i sposób ich zapisu/odczytu. W naszym przypadku jest to P6, a więc format binarny, RGB umożliwiający zapis pikseli z wartościami kanałów kolorów w zakresie 0-255. Więcej szczegółów możesz znaleźć na [Wikipedii](https://en.wikipedia.org/wiki/Netpbm#File_formats).  
Linijkę niżej zapisujemy szerokość i wysokość. Na ostatek, precyzujemy maksymalny zakres wartości piksela kanału - 255. Dalej, zrzucamy hurtem, bezpośrednio do pliku binarne wartości z tablicy `data`. Użyjemy do tego `reinterpret_cast`, który biorąc surowy wskaźnik do znaków, zrzuca podany pod nim ciąg bajtów wielkości `szerkość * wysokość * 3 kanały` jako zwykły ciąg znaków, bez zmiany ich zawartości. Zamykamy plik i klatka jest zapisana. 

```cpp
// --------------------
// SAVE IMAGE (BINARY P6)
// --------------------
void savePPM(const char* filename, unsigned char* data, int width, int height)
{
    // Open file in binary mode (crucial for performance and Windows compatibility)
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for writing: " << filename << "\n";
        std::cerr << "Hint: Does the 'frames' directory exist?\n";
        return;
    }

    // P6 Header
    file << "P6\n" << width << " " << height << "\n255\n";

    // Fast binary dump of the memory buffer
    file.write(reinterpret_cast<const char*>(data), width * height * 3);

    file.close();
}
```

Przejdźmy teraz do kernela. Zaczynając od nagłówka, funkcja przyjmie:
- tablicę znaków na GPU, przekazaną przez wskaźnik na GPU
- szerokość i wysokość obrazu
- 3 `float`'y na źródło światła
- 3 `float`'y na wyjściowy punkt promienia (w naszym kontekście lokalizację kamery w klatce)
- 3 `float`'y na lokalizację centrum kuli
- `float` na promień kuli

Dalej identyfikujemy, który piksel jest obsługiwany w klatce, przez index x i y. 
Nie wykonujemy kerneli, które obsługują indeksy spoza rozmiarów obrazu, więc obsługujemy
ten przypadek ifem. 

```cpp
__global__ void render(unsigned char* image, int width, int height, float3 lightSource, float3 rayOrigin, float3 center, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
```

Dalej dla wygody obliczeń, indeksy x i y, znormalizujemy i dostosujemy do następujących wzorów. Pod nowymi zmiennymi `u` i `v`, w obu przypadkach zmieniamy zakres indeksu od 0 do `width`-1 albo `height`-1, na położenie centrum piksela, czyli piksel 0, leży w zakresie 0 do 1, więc jego centrum jest w 0.5. Stąd do każdego indeksu dodajemy 0.5. Dalej zmieniamy zakres na od -1 do 1, więc:
- dzielimy odpowiednio x przez szerokość, a y przez wysokość - zmieni to zakresy od 0 do 1 (obu wartości)
- mnożymy razy 2 - teraz zakresy to 0 ... 2. 
- odejmujemy 1 - to kończy transformację obu wartości do zakresu -1 ... 1. 
 
Teraz problem jest taki, że gdy dla tego kwadratu wyrenderujemy kulę, po odwrotnej transformacie, dostaniemy spłaszczony obraz. Wracamy do prostokąta z oryginalnym ratio, przez mnożnik `aspect_ratio`. 

```cpp
    // 1. Calculate aspect ratio to fix image distortion
    float aspect_ratio = (float)width / (float)height;

    // 2. Map pixel coordinates (x,y) to 3D Viewport (u,v)
    // 
    // Correct 'u' for aspect ratio
    float u = ((x + 0.5f) / width * 2.0f - 1.0f) * aspect_ratio;
    // Invert 'v' so that y=0 is the top of the image
    float v = -(y + 0.5f) / height * 2.0f + 1.0f;
```

Po ustaleniu współrzędnych u i v, musimy jeszcze skonstruować układ współrzędnych kamery, bo przecież kamera w każdej klatce patrzy w inną stronę (orbitowanie). W ray tracingu najczęstszą praktyką jest zbudowanie trzech wektorów:
- `forward` – kierunek patrzenia kamery
- `right` – kierunek w prawo od kamery
- `up` – kierunek w górę

Tymi trzema wektorami opisujemy płaszczyznę obrazu, z której wypuszczamy promienie. Z racji tego,
że są to wektory układu współrzędnych, chcemy je mieć z długością 1, więc je znormalizujemy (czyli właśnie sprowadzimy wielowymiarowy wektor do wektora skierowanego w tym samym kierunku, ale o długości 1). 

Zajmijmy się tymi 3-ma wektorami jeden po drugim.

**Wektor forward**. Mówi on „gdzie kamera patrzy”.
To po prostu znormalizowany wektor od kamery do centrum kuli.

**Wektor Right**. Aby wiedzieć, gdzie jest „prawo”, wykonujemy iloczyn wektorowy `cross(forward, worldUp)`:
- `worldUp = (0,1,0)` – zakładamy, że globalna oś Y to góra sceny
- `cross(forward, worldUp)` - daje wektor prostopadły do obu, czyli „prawo"

Po więcej informacji o iloczynie wektorowym, zobacz [Wikipedię](https://pl.wikipedia.org/wiki/Iloczyn_wektorowy). 

Bez tego kamera kręciłaby się razem z orbitą, jak kamera przymocowana do samochodu.

**Wektor Up**. Jest to kierunek w górę. Aby dostać „górę kamery”, znów używamy iloczynu wektorowego.
Teraz mamy kompletny ortonormalny ("prostopadły" i o wektorach o długości 1) układ kamerowy.

```cpp
    // 3. Camera System (LookAt Logic)
    // Forward vector: Direction from camera to the target (sphere center)
    float3 forward = normalize(center - rayOrigin);

    // Right vector: Perpendicular to Forward and World Up (0,1,0)
    float3 worldUp = make_float3(0, 1, 0);
    float3 right = normalize(cross(forward, worldUp));

    // Up vector: Perpendicular to Right and Forward
    float3 up = cross(right, forward);
```

Po ustabilizowaniu układu kamery, możemy zbudować promień wychodzący z kamery przez dany piksel.
Robimy to tak:

- bierzemy `forward` – punkt centralny widoku
- dodajemy `u * right` – przesunięcie poziome
- dodajemy `v * up` – przesunięcie pionowe
- normalizujemy wynik

```cpp
    // 4. Ray Direction
    float fov_scale = 1.0f; // Zoom factor
    float3 rayDir = normalize(forward + u * right * fov_scale + v * up * fov_scale);
```

W efekcie każdy piksel dostaje swój własny promień skierowany dokładnie tam, gdzie powinien.

Teraz zajmujemy się logiką rozpoznania, czy trafiliśmy na kulę, czy tło. A jeżeli trafi, to w którym miejscu i pod jakim kątem? 

Model naszego sprawdzenia, to metaforyczny promień `ray`, wypuszczony z piksela `rayOrigin` w kierunku patrzenia piksela `rayDir`. Leci on w tej linii, zaczynając od czasu `t=0`. Daje nam to wzór:
$$
ray(t) = rayOrigin + rayDir \cdot t
$$

na trajektorię promienia. Czy w tej trajektorii, promień przecina kulę? Kula ma powierzchnię na punktach:
$$
((x, y, z) - center)^2 = r^2,
$$
a więc rozpisując:
$$
((x - center.x)^2 + (y - center.y)^2 + (z-center.z)^2) = r^2
$$
Więc podłożymy `ray(t)` za `x`, `y`, i `z`. Wszystkie wektory, których używamy, mają 3 wymiary, ale dla skróconego zapisu, nie będziemy ich rozpisywać. Więc:
$$
(ray(t) - center)^2 = r^2,  
$$
$$
(rayDir \cdot t + rayOrigin - center)^2 = r^2,
$$
Trzy człony do kwadratu będą trudne do rozwinięcia, wprowadźmy zmienną `L`:
$$
L = rayOrigin - center
$$
$$
(rayDir \cdot t + L)^2 = r^2,
$$
Teraz rozwinięcie, jest znacznie prostsze do pogrupowania:
$$
(rayDir^2)t^2 + (2 \cdot L \cdot rayDir)t + (L^2 - r^2) = 0
$$
I w ten sposób utrzymaliśmy równanie kwadratowe, sprawdzające, czy istnieje jakiś moment w czasie `t`, faktycznie dotykający kuli. Policzmy więc deltę równania kwadratowego, implementując od razu tradycyjne oznaczenia `a`, `b` i `c`:

```cpp
    // 5. Sphere Intersection (Quadratic Formula) (t*rayDir + L)^2 = r^2
    float3 L = rayOrigin - center;
    float a = dot(rayDir, rayDir); // Always 1.0 if normalized
    float b = 2.0f * dot(L, rayDir);
    float c = dot(L, L) - radius * radius;
    float delta = b * b - 4.0f * a * c;
```

Mając deltę, możemy określić, czy istnieje taki moment, gdzie promień pada na kulę, czy nie, i piksel "patrzy" na tło. 
Zaczynamy od defaultowego założenia, że piksel właśnie "patrzy" na tło, i to kolory tła umieszczamy w definicji
wartości kanałów RGB.

```cpp
    // Default background color (Dark Navy)
    unsigned char red = 30, green = 30, blue = 50;
```

Jeżeli jest inaczej, to delta jest większa/równa zero. Liczymy więc dla tego przypadku `t`, i to biorąc wzór zwracający mniejszą wartośc. Dla zerowej delty, będzie to jedyne miejsce zerowe, a dla większej od 0, będzie to mniejsza wartość. Bierzemy mniejszą wartość, żeby upewnić się, że wyrenderujemy pierwszy punkt, na który padnie światło. Jaki jest sens renderowania piksela z tyłu kuli?

```cpp
    if (delta >= 0.0f)
    {
        // Calculate the nearest intersection distance 't'
        float t = (-b - sqrtf(delta)) / (2.0f * a);
```

Jeżeli obiekt jest przed kamerą, to pokażemy kulę. Jeżeli jest na kamerze, albo za nią (czas zero, albo ujemny), nie pokażemy nic. 

W przypadku kuli, obliczmy punkt dotknięcia kuli, a następnie znormalizowany wektor kierunku styku do centrum kuli - jest to tzw. normalna punktu przecięcia.
```cpp
        // Render only if the object is IN FRONT of the camera (t > 0)
        if (t > 0.0f) {
            float3 hit = rayOrigin + t * rayDir;
            float3 normal = normalize(hit - center);
```

Teraz jesteśmy tylko o krok do wyrenderowania odpowiedniego koloru! Potrzebujemy intensywności, która zależy tylko od kąta normalnej punktu przecięcia i punktu padania światła na ten punkt. Policzmy go więc. 

```cpp
            // Lighting calculation (Lambertian)
            // Ensure light direction is normalized
            lightSource = normalize(lightSource);

            // Calculate intensity based on the angle between normal and light
            float intensity = fmaxf(0.1f, dot(normal, lightSource));
```
Jak widzisz, intensywność jest na koniec poddana korekcie, wybierając wartośc 0.1, jeżeli zostanie ona policzona w zakresie 0 - 0.1. Jest tak, bo założyliśmy, że pełen cień, to intensywność 0.1 - nie chcemy mieć zupełnie czarnej kuli, po nieoświetlonej stronie. 

Na koniec ciała ifa (naszego przypadku trafienia na kulę), odkładamy odcień czerwieni kuli do kanałów RGB, zeskalowane przez obliczoną intensywność.

```cpp
            red = (unsigned char)(255 * intensity);
            green = (unsigned char)(50 * intensity);
            blue = (unsigned char)(50 * intensity);
        }
    }
```

Konkludujemy kernel przez odkłożenie do obrazu obliczone wartości piksela. 
> **UWAGA!** Nasz obraz jest w tablicy jednowymiarowej, więc zanim to zrobimy, musimy znaleźć spłaszczony indeks, iterujący tę jednowymiarową tablicę.


```cpp
    int idx = (y * width + x) * 3;

    // Write final pixel color to Global Memory
    image[idx + 0] = red;
    image[idx + 1] = green;
    image[idx + 2] = blue;
}
```

To tyle co do kodu kernela. Zostały nam jednak operacje na `float3`, które nie są zaimplementowane dla kernela, mimo że sam typ, został zaciągniety z CUDY i jest obsługiwany na GPU. Porzebujemy przeciążyć operatory `*`, `+`, i `-`, a także dopisać funkcje normalizacji, mnożeń, liczenia długości, i operatora wektorowego. Nie będą się one różniły od klasycznej implementacji w kodzie, ale żeby działy w obrębie kernela, potrzebujemy słowa kluczowego `__device__`, przed każdą definicją. 

> Specyfikatory `__device__`, `__host__`, i `__global__`:  
  Każdy z nich precyzuje, gdzie można wykonać daną funkcję i skąd można ją wywołać. Poniżej specyfikajcja, a co ciekawe, `__device__` i `__host__` można łączyć:
> | Specyfikator | Gdzie działa | Skąd wywoływana | Zastosowanie |
> | --------- | ---- | ---- | ---- |
> | `__global__` | GPU | CPU | Kernel |
> | `__device__` | GPU | GPU | Funkcje pomocnicze kernela |
> | `__host__` | CPU | CPU | Normalne funkcje C++ |
> | `__device__ __host__` | GPU i CPU | GPU i CPU | Funkcje wspólne (wykonujące ten sam kod, zarówno > przez) hosta i urządzenia |  
> 
> **UWAGA!** `__device__ __host__` pozwala wykonywać kod na CPU, wywołany z CPU, i na GPU wywołany przez GPU. Nie pozwala on puścić kodu na GPU przez wywołanie z CPU (jak kernel), ani wykonać kodu na CPU z wywołania z GPU. 

Tutaj implementacja wszystkich:

```cpp
__device__ float3 operator*(float b, const float3& a)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3& a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float length(const float3& v)
{
    return sqrtf(dot(v, v));
}

__device__ float3 normalize(const float3& v)
{
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
```

Nasz skrypt jest gotowy, od początku do końca! 

<h3> 🔍 Podsumowanie</h3>

W tym wpisie zaimplementowaliśmy od początku do końca render kuli z punktowym oświetleniem. Zaanimowaliśmy stały obrót kamery wokół kuli, i klatka, po klatce, wykonaliśmy zrównoleglony redner całego obrazu naraz na GPU (jeden wątek na jeden piksel). Wykorzystaliśmy funkcje pomocnicze na GPU dla kernela, wykorzystujące specyfikator `__device__`, które m.in. przeciążały operatory do obsługi trójwymiarowego `float`a. 

------------------------------
<h2> Pytania kontrolne </h2>

Do dzisiejszych pytań kontrolnych, przyda się sprawdzenie linków do dodatkowych materiałów z tekstu. 

1. Jakie inne formaty z rodziny PPM możemy użyć? Czy P1 jest możliwe? A P3?
2. Co oznaczają argumenty i flagi od ffmpega? Jak zmienić framerate i nazwę pliku wynikowego?
Jak zmienić kodek i format wyjściowy?
3. Jakie znasz specyfikatory funkcji, do zmiany urządzeń, skąd można je wywołać i na czym je wywołać?

------------------------------
<h2>Ćwiczenia:</h2>

1. Dostosuj program, żeby użytkownik mógł wprowadzić pozycję kamery i źródła światła z klawiatury, zamiast mieć je na sztywno w kodzie. 
   Upewnij się, że program nadal działa poprawnie z nowymi wartościami. Pobaw się zachowaniem outputu:  
    (a) Czy zauważysz obrót, jeżeli światło będzie dokładnie nad kulą?  
    (b) Co się stanie, jeżeli światło będzie dokładnie z tyłu kamery?  
    (c) Co się stanie, jeżeli kamera będzie bliżej środka kuli, niż promień kuli (np. na współrzędnych (0, 0, -2))?  
    (d) Co się stanie jeżeli kamera będzie bardzo daleko od kuli (np. na współrzędnych (1'000'000, 0, -3))?  
2. Nasz program ma zahardkodowane dane. Przepisz go w efektywniejszy/czystszy sposób: przenieś dane jako stałe globalne, albo wartości do kernela i zmniejsz ilość argumentów w funkcjach. 


------------------------------
<break></break>
------------------------------
<h2>Odpowiedzi do pytań kontrolnych:</h2>

1. Formaty rodziny *PBM/PGM/PPM* (zestaw P1–P6):

| Format | Typ obrazu                          | ASCII/binarny | Możliwy w ray tracerze?                                      |
| ------ | ----------------------------------- | ------------- | ------------------------------------------------------------ |
| **P1** | PBM – obraz 1‑bitowy (czarno‑biały) | ASCII     | nie — tylko piksel czarny albo biały |
| **P2** | PGM – skala szarości | ASCII | nie |
| **P3** | PPM – RGB | ASCII | tak, ale **dużo wolniejszy i większe pliki**               |
| **P4** | PBM | binarny   | nie |
| **P5** | PGM | binarny   | nie |
| **P6** | PPM – RGB | binarny   | tak |

P1 jest niekompatybilny, bo możesz mieć tylko czerń i biel.
P3 jest i możliwy i w pełni zgodny z P6 - różni się tylko tym, że dane RGB zapisujesz pełnym tekstem w ASCII.


2. Nasza komenda:

```bash
ffmpeg -y -framerate 60 -i frames/frame_%03d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4
```

zawiera flagi:

*   **`-y`** — nadpisz plik bez pytania
*   **`-framerate 60`** — liczba klatek na sekundę wejścia
*   **`-i frames/frame_%03d.ppm`** — wejście: sekwencja `frame_000.ppm`, `frame_001.ppm` itd.
*   **`-c:v libx264`** — kodek wideo (H.264)
*   **`-pix_fmt yuv420p`** — format pikseli kompatybilny z większością odtwarzaczy
*   **`output.mp4`** — nazwa pliku wynikowego

Co do zmian:  
(a) Jak zmienić framerate?

```bash
ffmpeg -framerate 30 -i frames/frame_%03d.ppm ...
```

(b) Jak zmienić nazwę wyjścia?

Po prostu nowa nazwa na końcu:

```bash
... my_animation.mkv
```

(c) Jak zmienić kodek?

Np. H.265:

```bash
-c:v libx265
```

WebM (VP9):

```bash
-c:v libvpx-vp9
```

GIF:

```bash
ffmpeg -framerate 30 -i frames/frame_%03d.ppm output.gif
```

***

3. `__global__`:
-   działa na: **GPU**
-   wywoływana z: **CPU**
-   tworzy kernel
-   musi zwracać `void`

`__device__`:
-   działa na: **GPU**
-   wywoływana z: **GPU** (kernel lub inna device)


`__host__`:
-   działa na: **CPU**
-   wywoływana z: **CPU**
-   normalna funkcja C++

`__host__ __device__` (oba naraz):

-   generuje DWIE wersje funkcji:
    -   jedną dla CPU
    -   jedną dla GPU
-   ale:
    - funkcja nie przełącza CPU→GPU  
    - nie jest kernelem

Używane do operatorów matematycznych, `make_float3`, itp.


------------------------------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>

<!-- Previous blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog4">Poprzedni wpis: IV. Prosty benchmark GPU</a></p>

<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog6">Następny wpis</a></p>
