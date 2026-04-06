#include <cuda_runtime.h>
#include <iostream>
#include "include/savePPM.h"


const int width = 300;
const int height = 300;

__global__
void fillSurface(cudaSurfaceObject_t surface) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uchar4 color = make_uchar4(
        (unsigned char)( (255 - abs(abs(x)%510 - 255))%256),
        (unsigned char)( (255 - abs(abs(y)%510 - 255))%256),
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

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

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

    savePPM("./gradient.ppm", host, width, height);

    delete[] host;
    cudaDestroySurfaceObject(surfObj);
    cudaFreeArray(array);

    return 0;
}