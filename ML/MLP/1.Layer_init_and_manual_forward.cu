#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <optional>


const int VERBOSE = 1;

class Layer {
public:
    int in_size, out_size;
    std::vector<float> weights;
    std::vector<float> biases;

    // Initializer with random parameters
    Layer(int i = 5, int o = 5, std::optional<unsigned> seed = std::nullopt):
        in_size(i), out_size(o), weights(o * i, 0.0f), biases(o, 0.0f) {
            std::mt19937 gen = seed
                ? std::mt19937(*seed)
                : std::mt19937(std::random_device{}());

            std::uniform_real_distribution<float> dist(-1.0f/sqrt((float)i), 1.0f/sqrt((float)i));

            for (auto& x : weights) x = dist(gen);
            for (auto& x : biases)  x = dist(gen);
        }
    
    // Weights setters through the operator () overload
    float& operator()(int i, int j) {        
        if (i < 0 || i >= out_size || j < 0 || j >= in_size)
            throw std::out_of_range("Layer::operator(): index out of bounds");

        return weights[i * in_size + j];
    }
    const float& operator()(int i, int j) const {
        if (i < 0 || i >= out_size || j < 0 || j >= in_size)
            throw std::out_of_range("Layer::operator(): index out of bounds");

        return weights[i * in_size + j];
    }

    // Biases setter through the operator [] overload
    float& operator[](int i) {
        if (i < 0 || i >= out_size)
            throw std::out_of_range("Layer::operator(): index out of bounds");

        return biases[i];
    }
    const float& operator[](int i) const {
        if (i < 0 || i >= out_size)
            throw std::out_of_range("Layer::operator(): index out of bounds");
        return biases[i];
    }

    // Forward pass through the operator () overload (no gradients calculation)
    std::vector<float> operator()(std::vector<float> x) {
        std::vector<float> y(out_size, 0.0);
        
        if ( x.size() != in_size ) 
            throw std::out_of_range("Layer::operator(): input vector size mismatch with layer input size");
        
        for ( int i = 0; i < out_size; ++i ) {
            if ( VERBOSE > 0 ) std::cout << "i=" << i << "\n";
            for (int j = 0; j < in_size; ++j ) {
                if ( VERBOSE > 0 ) std::cout << "\tj=" << j << "\n";
                if ( VERBOSE > 0 ) std::cout << "\t\ty[" << i << "]=" << y[i] << ", x[" << j << "]=" << x[j] << ", w[" << i * in_size + j << "]=" << weights[i * in_size + j] << '\n';
                y[i] += x[j] * weights[i * in_size + j];
                if ( VERBOSE > 0 ) std::cout << "\t\ty[" << i << "]=" << y[i] << '\n';
            }
            if ( VERBOSE > 0 ) std::cout << "\tb[" << i << "]=" << biases[i] << "\n";
            y[i] += biases[i];
            if ( VERBOSE > 0 ) std::cout << "\ty[" << i << "]=" << y[i] << '\n';
        }

        return y;
    }
    
    // Print weights and biases
    void print() const {
        std::cout << "W = [";
        for (int i = 0; i < out_size; ++i) {
            if (i > 0) { std::cout << "    ["; }
            for (int j = 0; j < in_size; ++j) {
                if ( j > 0 ) { std::cout << ", "; }
                std::cout << std::fixed << std::setprecision(3) << std::setw(6) << operator()(i, j);
            }
            std::cout << "]\n";
        }
        std::cout << "B = [";
        
        for (int i = 0; i < out_size; ++i) {
            if (i > 0) { std::cout << "    ["; }
            std::cout << std::fixed << std::setprecision(3) <<  std::setw(6) << operator[](i) << "]\n";
        }        
    }
};

int main() {
    const int in_size = 3, out_size = 4;
    Layer l(in_size, out_size, 2026); // construct layer with 2026 seed
    
    // test setting up single weight parameter
    l(1, 1) = 9.0; 
    // test setting up single bias parameter
    l[1] = 9.0;

    // see the change
    l.print();

    // Forward pass
    std::vector<float> x = {1, 2, 3};
    std::vector<float> y_hat = l(x);

    std::cout << "y_hat = [";
    for (const auto& y_i : y_hat) {
        std::cout << y_i << ", ";
    }
    std::cout << "\033[1D\033[1P\033[1D\033[1P]\n";

    return 0;
}