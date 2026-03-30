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

W tym wpisie, użyjemy zdobytej wiedzy z poprzednich wpisów, żeby użyć GPU to stworzenia animacji prostej sceny 3D, z kulą oświetloną z różnych stron. Będzie to bardzo uproszczona wersja ray tracingu, ale pozwoli nam na praktyczne zastosowanie kerneli CUDA do renderowania grafiki - czyli tego, do czego GPU zostały stworzone.


<h4>Setup renderowanego środowiska</h4>

Do renderowania sceny, potrzebujemy zdefiniować kilka elementów:
- układ współrzędnych, w którym będziemy pracować (kartezjański, z osią Y skierowaną do góry, osią X i Z rozpinającymi płaszczyznę poziomą)
- tło, które potraktujemy jako bardzo odległą płaszczyznę, która będzie miała stały kolor ( blady granatowo-fioletowy, RGB(30, 30, 50))
- kulę, która będzie umieszczona blisko centrum sceny na współrzędnych (0, 0, -3), i będzie miała promień 1 jednostki. Kula będzie miała kolor RGB(255, 50, 50) przy maksymalnym świetle. Będzie ona ciemnieć w zależności od kąta padania światła i odległości od źródła światła. Dla każdego piksela, będziemy operowali parametrem `intensity`, który będzie określał, jak jasny jest dany punkt na kuli, i będzie się wahał od 0 (czarny) do 1 (pełna jasność). Kolor piksela (jeżeli trafi na kulę) będzie wtedy obliczany jako `color = base_color * intensity`, gdzie `base_color` to właśnie kolor kuli RGB(255, 50, 50).
- kamerę, która będzie umieszczona na współrzędnych (5, 0, -3) i będzie patrzeć na wprost na kulkę. Kamera będzie miała pole widzenia 90 stopni, co oznacza, że będzie widzieć obszar od -1 do 1 w osi X i Y na płaszczyźnie Z=0. Zasadniczo, odległość od kamery do najbliższego punktu na kuli będzie wynosić 5-1=4. To pozycja startowa. Będziemy animować kamerę, obracając ją wokół kuli, więc jej pozycja będzie się zmieniać w czasie. Kamera będzie poruszać się po okręgu o promieniu 5 jednostek wokół kuli, w płaszczyźnie XZ, ze stałą prędkością kątową. Chcemy mieć pełny obrót kamery na wszystkie klatki, więc jeśli mamy 120 klatek, przy 60 fpsach to kamera będzie obracać się o 360/120=3 stopnie na klatkę, i 260/120/60=180 stopni na sekundę. 

Mamy podstawowe elementy sceny, teraz musimy je zaimplementować w kodzie. Będziemy używać kernela CUDA do renderowania każdego piksela obrazu, i będziemy generować serię klatek, klatka po klatce, w pętli `for`, które potem połączymy w animację. Zacznijmy od szkieletu maina.

<h4>Szkielet maina</h4>



<!-- make a code snippet in cpp -->
```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib> // For system()

// --------------------
// VECTOR MATH HELPERS
// --------------------

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

// --------------------
// CUDA KERNEL
// --------------------
__global__ void render(unsigned char* image, int width, int height, float3 lightDir, float3 rayOrigin, float3 center, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // 1. Calculate aspect ratio to fix image distortion
    float aspect_ratio = (float)width / (float)height;

    // 2. Map pixel coordinates (x,y) to 3D Viewport (u,v)
    // 
    // Correct 'u' for aspect ratio
    float u = ((x + 0.5f) / width * 2.0f - 1.0f) * aspect_ratio;
    // Invert 'v' so that y=0 is the top of the image
    float v = -(y + 0.5f) / height * 2.0f + 1.0f;


    // 3. Camera System (LookAt Logic)
    // Forward vector: Direction from camera to the target (sphere center)
    float3 forward = normalize(center - rayOrigin);

    // Right vector: Perpendicular to Forward and World Up (0,1,0)
    float3 worldUp = make_float3(0, 1, 0);
    float3 right = normalize(cross(forward, worldUp));

    // Up vector: Perpendicular to Right and Forward
    float3 up = cross(right, forward);

    // 4. Ray Direction
    float fov_scale = 1.0f; // Zoom factor
    float3 rayDir = normalize(forward + u * right * fov_scale + v * up * fov_scale);

    // 5. Sphere Intersection (Quadratic Formula) (t*rayDir + L)^2 = r^2
    float3 L = rayOrigin - center;
    float a = dot(rayDir, rayDir); // Always 1.0 if normalized
    float b = 2.0f * dot(L, rayDir);
    float c = dot(L, L) - radius * radius;
    float delta = b * b - 4.0f * a * c;

    // Default background color (Dark Navy)
    unsigned char red = 30, green = 30, blue = 50;

    if (delta >= 0.0f)
    {
        // Calculate the nearest intersection distance 't'
        float t = (-b - sqrtf(delta)) / (2.0f * a);

        // Render only if the object is IN FRONT of the camera (t > 0)
        if (t > 0.0f) {
            float3 hit = rayOrigin + t * rayDir;
            float3 normal = normalize(hit - center);

            // Lighting calculation (Lambertian)
            // Ensure light direction is normalized
            lightDir = normalize(lightDir);

            // Calculate intensity based on the angle between normal and light
            float intensity = fmaxf(0.1f, dot(normal, lightDir));

            red = (unsigned char)(255 * intensity);
            green = (unsigned char)(50 * intensity);
            blue = (unsigned char)(50 * intensity);
        }
    }

    int idx = (y * width + x) * 3;

    // Write final pixel color to Global Memory
    image[idx + 0] = red;
    image[idx + 1] = green;
    image[idx + 2] = blue;
}

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

// --------------------
// MAIN APPLICATION
// --------------------
int main()
{
    int frames_to_render = 120;

    const int width = 800;
    const int height = 600;
    const int imageSize = width * height * 3;

    // --- SCENE CONFIGURATION ---
    float3 center = make_float3(0, 0, -3);
    float sphereRadius = 1.0f;

    float3 lightDir = { 0, 0, 1.0f };
    float x_cam, y_cam, z_cam;
    float x_light, y_light, z_light;

    // Display Application Header
    std::cout << "========================================\n";
    std::cout << "   CUDA RAY TRACER - ORBIT DEMO    \n";
    std::cout << "========================================\n";
    std::cout << "Sphere: Center (" << center.x << ", " << center.y << ", " << center.z;
    std::cout << "), Radius " << sphereRadius << "\n";
    std::cout << "System: Generating " << frames_to_render << " frames of ";
    std::cout << width << "x" << height << " resolution.\n";
    std::cout << "Output: frames/*.ppm -> output.mp4\n";
    std::cout << "----------------------------------------\n";

    // Setup working directory (create 'frames' folder if not exists)
    // Windows command to create folder quietly
    system("mkdir frames > nul 2>&1");

    // User Input
    std::cout << "Enter camera starting position (x, y, z): ";
    // std::cin >> x_cam >> y_cam >> z_cam;
    x_cam = 5.0f; y_cam = 0.0f; z_cam = -3.0f;

    std::cout << "Enter light source position (x, y, z): ";
    // std::cin >> x_light >> y_light >> z_light;
    x_light = 0.0f; y_light = 1.0f; z_light = -3.05f;
    lightDir = { x_light, y_light, z_light };

    // --- ORBIT CALCULATION ---
    float dx = x_cam - center.x;
    float dz = z_cam - center.z;

    // Calculate initial radius and angle based on user input
    float orbitRadius = sqrtf(dx * dx + dz * dz);
    float startAngle = atan2f(dx, dz);

    // Memory Allocation
    unsigned char* h_image = new unsigned char[imageSize]; // Host
    unsigned char* d_image;
    cudaMalloc(&d_image, imageSize); // Device

    // Execution Configuration
    // 32x16 = 512 threads per block (optimal for occupancy)
    dim3 block(32, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    std::cout << "\nStarting rendering loop...\n";

    // --- RENDER LOOP ---
    for (int frame = 0; frame < frames_to_render; frame++) {
        // Calculate new camera angle (Full circle in 120 frames)
        float angleStep = (2.0f * 3.14159f) / frames_to_render;
        float currentAngle = startAngle + (frame * angleStep);

        // Update Camera Position
        float3 currentCamPos;
        currentCamPos.x = center.x + orbitRadius * sinf(currentAngle);
        currentCamPos.y = y_cam; // Maintain constant height
        currentCamPos.z = center.z + orbitRadius * cosf(currentAngle);

        // Launch Kernel
        render <<< grid, block >>> (d_image, width, height, lightDir, currentCamPos, center, sphereRadius);
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


<h4> 🔍 Podsumowanie</h4>

[TBD]

------------------------------
<h2> Pytania kontrolne </h2>

1. ...
2. ...

------------------------------
<h2>Ćwiczenia:</h2>

1. ...
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
