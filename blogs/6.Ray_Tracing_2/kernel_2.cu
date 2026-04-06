#include <cuda_runtime.h>
#include <iostream>


const int width = 400;
const int height = 400;

__global__
void fillSurface(cudaSurfaceObject_t surface) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uchar4 color = make_uchar4(
        (unsigned char)( (x) % 256),
        (unsigned char)( (y) % 256),
        128,
        255
    );

    surf2Dwrite(color, surface, x * sizeof(uchar4), y, cudaBoundaryModeTrap);
}

int main()
{
    // Create CUDA array that can be used as a surface
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t array;
    cudaMallocArray(&array, &desc, width, height, cudaArraySurfaceLoadStore);

    // Describe resource (surface writes to cudaArray)
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    // Create the surface object
    cudaSurfaceObject_t surfObj = 0;
    cudaCreateSurfaceObject(&surfObj, &resDesc);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    fillSurface<<<grid, block>>>(surfObj);
    cudaDeviceSynchronize();

    // Copy result to host
    uchar4* host = new uchar4[width * height];
    cudaMemcpy2DFromArray(
        host,
        width * sizeof(uchar4),
        array,
        0,
        0,
        width * sizeof(uchar4),
        height,
        cudaMemcpyDeviceToHost
    );

    // Write PPM
    FILE* f = fopen("gradient.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        fprintf(f, "%d %d %d\n", host[i].x, host[i].y, host[i].z);
    }
    fclose(f);

    delete[] host;
    cudaDestroySurfaceObject(surfObj);
    cudaFreeArray(array);

    return 0;
}