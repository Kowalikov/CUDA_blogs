// savePPM.cpp
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>


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

void savePPM(const char* filename, uchar4* data, int width, int height)
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

    for (int i = 0; i < width * height; ++i) {
        file.put(data[i].x);
        file.put(data[i].y);
        file.put(data[i].z);
    }

    file.close();
}