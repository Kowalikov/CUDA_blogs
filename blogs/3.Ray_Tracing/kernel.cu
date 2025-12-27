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
    // Display Application Header
    std::cout << "========================================\n";
    std::cout << "   CUDA RAY TRACER - ORBIT DEMO    \n";
    std::cout << "========================================\n";
    std::cout << "Sphere: Center (0, 0, -3), Radius 1.0\n";
    std::cout << "System: Generating 600 frames of 800x600 resolution.\n";
    std::cout << "Output: frames/*.ppm -> output.mp4\n";
    std::cout << "----------------------------------------\n";

    // Setup working directory (create 'frames' folder if not exists)
    // Windows command to create folder quietly
    system("mkdir frames > nul 2>&1");

    const int width = 800;
    const int height = 600;
    const int imageSize = width * height * 3;

    // --- SCENE CONFIGURATION ---
    float3 center = make_float3(0, 0, -3);
    float sphereRadius = 1.0f;

    float3 lightDir = { 0, 0, 1.0f };
    float x_cam, y_cam, z_cam;
    float x_light, y_light, z_light;

    // User Input
    std::cout << "Enter camera starting position (x, y, z): ";
    std::cin >> x_cam >> y_cam >> z_cam;

    std::cout << "Enter light source position (x, y, z): ";
    std::cin >> x_light >> y_light >> z_light;
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
    for (int frame = 0; frame < 600; frame++) {
        // Calculate new camera angle (Full circle in 120 frames)
        float angleStep = (2.0f * 3.14159f) / 600.0f;
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
        if (frame % 10 == 0) std::cout << "Rendered frame " << frame << "/600\n";
    }

    // FFmpeg Video Generation
    std::cout << "Rendering complete. Generating MP4...\n";
    system("ffmpeg -y -framerate 30 -i frames/frame_%03d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4");

    // Cleanup
    cudaFree(d_image);
    delete[] h_image;

    std::cout << "Done! Saved output.mp4\n";
    return 0;
}