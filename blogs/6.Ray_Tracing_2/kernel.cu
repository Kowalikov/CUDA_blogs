#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>


surface<void, cudaSurfaceType2D> surf;

const int width = 400;
const int height = 400;

__global__ void fillSurface() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    
    uchar4 color = make_uchar4(
        (unsigned char)(x % 256),
        (unsigned char)(y % 256),
        128, 255
    );

    surf2Dwrite(color, surf, x * sizeof(uchar4), y);
}

int main() {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t array;
    cudaMallocArray(&array, &desc, width, height, cudaArraySurfaceLoadStore);

    cudaBindSurfaceToArray(surf, array);

    dim3 block(16,16);
    dim3 grid((width+15)/16, (height+15)/16);

    fillSurface<<<grid,block>>>();
    cudaDeviceSynchronize();

    // Copy back to host:
    uchar4* host = new uchar4[width*height];
    cudaMemcpy2DFromArray(
        host, width*sizeof(uchar4),
        array, 0,0,
        width*sizeof(uchar4), height,
        cudaMemcpyDeviceToHost
    );

    // Save to PPM
    FILE* f = fopen("gradient.ppm","w");
    fprintf(f,"P3\n%d %d\n255\n",width,height);
    for(int i=0;i<width*height;i++)
        fprintf(f,"%d %d %d\n", host[i].x, host[i].y, host[i].z);
    fclose(f);

    delete[] host;
    cudaFreeArray(array);

    return 0;
}