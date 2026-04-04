#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib> // For system()


// --------------------
// CUDA KERNEL
// --------------------
__global__ void render(unsigned char* image, int width, int height, float3 lightSource, float3 rayOrigin, float3 center, float radius)
{
    // TODO: implement ray tracing logic to compute pixel colors based on sphere intersection and lighting
}

// --------------------
// SAVE IMAGE (BINARY P6)
// --------------------
void savePPM(const char* filename, unsigned char* data, int width, int height)
{
    // TODO: implement PPM saving (P6 format)
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

    // Setup working directory (create 'frames' folder if not exists)
    // Windows command to create folder quietly
    system("mkdir frames > nul 2>&1");
    
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
    
    // --- ORBIT CALCULATION ---
    float dx = camPosition.x - center.x;
    float dz = camPosition.z - center.z;

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