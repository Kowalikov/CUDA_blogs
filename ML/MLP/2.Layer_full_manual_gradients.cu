#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <optional>


const int VERBOSE = 0;

std::vector<float> operator-(std::vector<float> x1, std::vector<float> x2) {
    if ( x1.size() != x2.size() ) 
            throw std::out_of_range("operator-: input vector sizes mismatch");
        
    std::vector<float> y(x1.size(), 0);
    for (int i = 0; i < y.size(); ++i) {
        y[i] = x1[i] - x2[i];
    }

    return y;
}

std::vector<float> operator*(std::vector<float> x1, std::vector<float> x2) {
    if ( x1.size() != x2.size() ) 
            throw std::out_of_range("operator-: input vector sizes mismatch");
        
    std::vector<float> y(x1.size(), 0);
    for (int i = 0; i < y.size(); ++i) {
        y[i] = x1[i] * x2[i];
    }

    return y;
}

std::vector<float> operator*(int a, std::vector<float> x) {
    std::vector<float> y(x.size(), 0);
    for (int i = 0; i < y.size(); ++i) {
        y[i] = a * x[i];
    }

    return y;
}

std::vector<float> operator*(float a, std::vector<float> x) {
    std::vector<float> y(x.size(), 0);
    for (int i = 0; i < y.size(); ++i) {
        y[i] = a * x[i];
    }

    return y;
}

std::vector<float> operator*(std::vector<float> x, int a) {
    std::vector<float> y(x.size(), 0);
    for (int i = 0; i < y.size(); ++i) {
        y[i] = a * x[i];
    }

    return y;
}

std::vector<float> operator*(std::vector<float> x, float a) {
    std::vector<float> y(x.size(), 0);
    for (int i = 0; i < y.size(); ++i) {
        y[i] = a * x[i];
    }

    return y;
}

std::vector<float> operator/(std::vector<float> x, int a) {
    std::vector<float> y(x.size(), 0);
    for (int i = 0; i < y.size(); ++i) {
        y[i] = x[i] / (a + 1e-8f);
    }

    return y;
}

std::vector<float> pow(std::vector<float> x, int e) {
    std::vector<float> y(x.size(), 0);
    for (int i = 0; i < y.size(); ++i) {
        y[i] = pow(x[i], e);
    }

    return y;
}

float mean(std::vector<float> x) {
    return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
}


class Layer {
public:
    int in_size, out_size;
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<float> weights_grad;
    std::vector<float> biases_grad;

    // Initializer with random parameters
    Layer(int i = 5, int o = 5, std::optional<unsigned> seed = std::nullopt):
        in_size(i), out_size(o), weights(o * i, 0.0f), biases(o, 0.0f), weights_grad(o * i, 0.0f), biases_grad(o, 0.0f) {
            std::mt19937 gen = seed
                ? std::mt19937(*seed)
                : std::mt19937(std::random_device{}());

            std::uniform_real_distribution<float> dist(-1.0f/sqrt((float)i), 1.0f/sqrt((float)i));

            for (auto& x : weights) x = dist(gen);
            for (auto& x : biases)  x = dist(gen);
        }
    
    // Forward pass through the operator () overload (no gradients calculation)
    std::vector<float> operator()(std::vector<float> x) {
        std::vector<float> y(out_size, 0.0);
        
        if ( x.size() != in_size ) 
            throw std::out_of_range("Layer::operator(): input vector size mismatch with layer input size");
        
        for ( int i = 0; i < out_size; ++i ) {
            for (int j = 0; j < in_size; ++j ) {
                y[i] += x[j] * weights[i * in_size + j];
            }
            y[i] += biases[i];
        }

        return y;
    }
    
    // Gradient zeroing
    void grad_zero() {
        std::fill(weights_grad.begin(), weights_grad.end(), 0.0f);
        std::fill(biases_grad.begin(), biases_grad.end(), 0.0f);
    }

    // Backward pass computing the gradients 
    void backward(std::vector<float> y_grads, std::vector<float> y, std::vector<float> x) {
        //d loss / d W =  (d loss / d y) * (d y / d W)
        // y = W * x + b
        // d y / d W = x
        // d loss / d W = (d loss / d y) * x
        for ( int i = 0; i < out_size; ++i ) {
            for (int j = 0; j < in_size; ++j ) {   
                weights_grad[i * in_size + j] += y_grads[i] * x[j];
            }
        }
        
        //d loss / d b =  (d loss / d y) * (d y / d b)
        // y = W * x + b
        // d y / d b = 1  
        // d loss / d b = (d loss / d y)
        biases_grad = y_grads;
    }

    // Updating the weights through the gradients 
    void step(float lr) {
        weights = weights - lr * weights_grad;
        biases = biases - lr * biases_grad;
    }
    
    // Print weights and biases
    void print() const {
        std::cout << "W = [";
        for (int i = 0; i < out_size; ++i) {
            if (i > 0) { std::cout << "    ["; }
            for (int j = 0; j < in_size; ++j) {
                if ( j > 0 ) { std::cout << ", "; }
                std::cout << std::fixed << std::setprecision(3) << std::setw(6) << weights[i * in_size + j];
            }
            std::cout << "]\n";
        }
        std::cout << "B = [";
        
        for (int i = 0; i < out_size; ++i) {
            if (i > 0) { std::cout << "    ["; }
            std::cout << std::fixed << std::setprecision(3) <<  std::setw(6) << biases[i] << "]\n";
        }        
    }

    // Print weights and biases gradients
    void print_grad() const {
        std::cout << "W_grad = [";
        for (int i = 0; i < out_size; ++i) {
            if (i > 0) { std::cout << "         ["; }
            for (int j = 0; j < in_size; ++j) {
                if ( j > 0 ) { std::cout << ", "; }
                std::cout << std::fixed << std::setprecision(3) << std::setw(6) << weights_grad[i * in_size + j];
            }
            std::cout << "]\n";
        }
        std::cout << "B_grad = [";
        
        for (int i = 0; i < out_size; ++i) {
            if (i > 0) { std::cout << "         ["; }
            std::cout << std::fixed << std::setprecision(3) <<  std::setw(6) << biases_grad[i] << "]\n";
        }        
    }
};

int main() {
    const int in_size = 3, out_size = 4;
    Layer l(in_size, out_size, 2026); // construct layer with 2026 seed
    
    // see the change
    if ( VERBOSE > 0 ) {
        l.print();
        l.print_grad();
    }

    // Training setup
    std::vector<float> x = {1, 2, 3};
    std::vector<float> y_true = {1, 4, 6, -0.5};
    std::vector<float> y_hat;

    float learning_rate = 5e-2;
    int n_epochs = 20;

    for (int epoch=0; epoch < n_epochs; ++epoch) {
        float lr = learning_rate * (pow(0.999, epoch));

        // Forward pass
        y_hat = l(x);

        // Compute the loss (Mean Squared Error)
        float loss = mean(pow(y_hat - y_true, 2));

        // loss = (E_{i=0}^{k} (y_hat_i - y_true_i)^2)/k = (E_{i=0}^{k} (y_hat_i^2 - 2*y_hat_i*y_true_i + y_true_i^2))/k
        // d(loss)/d(y_hat_i) = (2*y_hat_i - 2*y_true_i)/k
        // d(loss)/d(y_hat) = (2*y_hat - 2*y_true)/k // element-wise
        std::vector<float> grad_loss_y_hat = (2 * y_hat - 2 * y_true) / y_hat.size();

        // Zero gradients before backward pass
        l.grad_zero();

        // Backward pass
        l.backward(grad_loss_y_hat, y_hat, x);

        // Update weights and biases manually using gradients
        l.step(lr);

        if ( (epoch + 1) % 5 == 0 ) {
            if ( VERBOSE > 0 ) {
                std::cout << "Final y_pred:\n\t[ ";
                for (const auto& y_i : y_hat) {
                    std::cout << y_i << ", ";
                }
                l.print();
                l.print_grad();
            }
            std::cout << "Epoch [" << epoch+1 << "/" << n_epochs << "], LR: ";
            std::cout << std::fixed << std::setprecision(3) << lr << ", Loss: " << loss;
            std::cout << "\n";
        }
    }

    std::cout << "\nTraining complete!\n";
    std::cout << "Final y_pred:\n\t[ ";
    for (const auto& y_i : y_hat) {
        std::cout << y_i << ", ";
    }
    std::cout << "\033[1D\033[1P\033[1D\033[1P]\n";

    std::cout << "Y_true:\n\t[ ";
    for (const auto& y_i : y_true) {
        std::cout << y_i << ", ";
    }
    std::cout << "\033[1D\033[1P\033[1D\033[1P]\n";
    
    // see the change
    if ( VERBOSE > 0 ) {
        l.print();
        l.print_grad();
    }

    return 0;
}