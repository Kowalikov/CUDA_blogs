#pragma once
#include <cuda_runtime.h>

void savePPM(const char* filename, unsigned char* data, int width, int height);
void savePPM(const char* filename, uchar4* data, int width, int height);