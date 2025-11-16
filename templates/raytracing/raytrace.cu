#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <math.h>

// ---- Vector Helpers ----
__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 operator*(const float3 &a, float t) {
    return make_float3(a.x * t, a.y * t, a.z * t);
}
__device__ float3 operator*(float t, const float3 &a) {
    return make_float3(a.x * t, a.y * t, a.z * t);
}
__device__ float dot(const float3 &a, const float3 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ float3 normalize(const float3 &v) {
    float len = sqrtf(dot(v,v));
    return make_float3(v.x/len, v.y/len, v.z/len);
}


// ---- Ray Tracing Functions ----
__device__ float hit_sphere(const float3& center, float radius, const float3& ray_origin, const float3& ray_dir) {
    float3 oc = ray_origin - center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b*b - 4*a*c;
    return (discriminant < 0) ? -1.0f : (-b - sqrtf(discriminant)) / (2.0f * a);
}

__device__ float3 ray_color(const float3& ray_origin, const float3& ray_dir) {
    float t = hit_sphere(make_float3(0,0,-1), 0.5f, ray_origin, ray_dir);
    if (t > 0.0f) {
        float3 N = normalize(ray_origin + ray_dir * t - make_float3(0,0,-1));
        return 0.5f * make_float3(N.x+1, N.y+1, N.z+1);
    }
    float3 unit_dir = normalize(ray_dir);
    t = 0.5f*(unit_dir.y + 1.0f);
    return (1.0f - t)*make_float3(1.0f, 1.0f, 1.0f) + t*make_float3(0.5f, 0.7f, 1.0f);
}

// ---- CUDA Kernel ----
__global__ void render(float3* fb, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;
    int idx = j * width + i;

    float u = float(i) / (width - 1);
    float v = float(j) / (height - 1);
    float3 origin = make_float3(0,0,0);
    float3 lower_left = make_float3(-2,-1,-1);
    float3 horizontal = make_float3(4,0,0);
    float3 vertical = make_float3(0,2,0);
    float3 dir = lower_left + horizontal * u + vertical * v - origin;

    fb[idx] = ray_color(origin, dir);
}

int main() {
    const int width = 400;
    const int height = 200;
    size_t fb_size = width * height * sizeof(float3);

    float3* fb;
    cudaMallocManaged(&fb, fb_size);

    dim3 threads(8,8);
    dim3 blocks((width+threads.x-1)/threads.x, (height+threads.y-1)/threads.y);
    render<<<blocks, threads>>>(fb, width, height);
    cudaDeviceSynchronize();

    std::ofstream out("raytraced_frame.ppm");
    out << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height-1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            int idx = j * width + i;
            int ir = int(255.99f * fb[idx].x);
            int ig = int(255.99f * fb[idx].y);
            int ib = int(255.99f * fb[idx].z);
            out << ir << " " << ig << " " << ib << "\n";
        }
    }
    out.close();

    cudaFree(fb);
    return 0;
}

