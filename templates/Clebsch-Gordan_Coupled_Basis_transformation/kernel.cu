#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void kernel(int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = idx + 1; // first integers above zero: 1..n
        out[idx] = val * val;
    }
}

struct QN_pair {
    int twoS; // 1 for S=0.5
    int twoMs; // 1 for Ms=0.5, -1 for Ms=-0.5

    bool operator<(const QN_pair& other) const {
        if (twoS != other.twoS) return twoS < other.twoS;
        return twoMs < other.twoMs;
    }
    bool operator>(const QN_pair& other) const {
        if (twoS != other.twoS) return twoS > other.twoS;
        return twoMs > other.twoMs;
    }
    bool operator==(const QN_pair& other) const {
        return twoS == other.twoS && twoMs == other.twoMs;
    }

}; 


// Input: Basis QNs for Block L and Site s (example values for spin-1/2)
const std::vector<QN_pair> basis_L_qn = {{1, 1}, {1, -1}}; // Indices 0, 1
const std::vector<QN_pair> basis_s_qn = {{1, 1}, {1, -1}}; // Indices 0, 1

const size_t dim_L = basis_L_qn.size();
const size_t dim_s = basis_s_qn.size();
const size_t dim_uncoupled = dim_L * dim_s; // 4 in our case

// Factorial function helper
double factorial(int n) {
    double res = 1.0;
    for (int i = 2; i <= n; ++i) res *= i;
    return res;
}

// Helper function for the Wigner 3j symbol calculation
double wigner_3j(double j1, double j2, double J, double m1, double m2, double M) {
    // Selection rules
    if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(M) > J) return 0.0;
    if (std::abs(j1 - j2) > J || j1 + j2 < J) return 0.0;
    if (std::abs(m1 + m2 + M) > 1e-9) return 0.0; // Ensure M = - (m1 + m2) for 3j

    // Factorials/gamma function parts
    double pre_factor = std::sqrt(
        (factorial(j1 + j2 - J) * factorial(j1 - j2 + J) * factorial(-j1 + j2 + J)) /
        factorial(j1 + j2 + J + 1)
    ) * std::sqrt(
        factorial(j1 + m1) * factorial(j1 - m1) * factorial(j2 + m2) * factorial(j2 - m2) *
        factorial(J + M) * factorial(J - M)
    );

    // Summation over k
    double sum_val = 0.0;
    for (int k = 0; k <= 100; ++k) { // Max 100 iterations should be enough for small J values
        // Denominators
        double den = factorial(k) * factorial(j1 + j2 - J - k) * factorial(j1 - m1 - k) *
                     factorial(j2 + m2 - k) * factorial(J - j2 + k + m1) * factorial(J - j1 + k - m2);
        
        // Skip if denominator is zero due to factorial of negative number
        if (std::isinf(den) || den == 0.0) continue; 

        // Alternating sign
        double term = (k % 2 == 0 ? 1.0 : -1.0) / den;
        sum_val += term;
    }
    
    return pre_factor * sum_val;
}

// Function to calculate a single Clebsch-Gordan coefficient <j1, m1; j2, m2 | J, M>
// Inputs use 'twoJ' and 'twoM' integers for exact arithmetic
double calculate_clebsch_gordan(int twoJ1, int twoM1, int twoJ2, int twoM2, int twoJ, int twoM) {
    if (twoM != twoM1 + twoM2) return 0.0; // Selection rule M = m1 + m2

    double j1 = twoJ1 / 2.0;
    double m1 = twoM1 / 2.0;
    double j2 = twoJ2 / 2.0;
    double m2 = twoM2 / 2.0;
    double J = twoJ / 2.0;
    double M = twoM / 2.0;

    // Use the 3j symbol relation:
    // <j1 m1; j2 m2 | J M> = (-1)^(j1 - j2 + M) * sqrt(2J + 1) * (j1 j2 J / m1 m2 -M)
    
    double sign_factor = std::pow(-1.0, j1 - j2 + M);
    double root_factor = std::sqrt(2.0 * J + 1.0);
    double three_j_val = wigner_3j(j1, j2, J, m1, m2, -M); // Note the -M

    return sign_factor * root_factor * three_j_val;
}

// Function to build the unitary transformation matrix U
std::vector<double> build_transformation_matrix_U(
    const std::vector<QN_pair>& basis_L, 
    const std::vector<QN_pair>& basis_s,
    const std::vector<QN_pair>& basis_J // the sorted coupled list from Step 2
) {
    size_t dim_L = basis_L.size();
    size_t dim_s = basis_s.size();
    size_t dim_uncoupled = dim_L * dim_s;
    size_t dim_coupled = basis_J.size();

    // The transformation matrix U, stored row-major (for now on host)
    std::vector<double> U(dim_uncoupled * dim_coupled, 0.0);

    for (size_t idx_L = 0; idx_L < dim_L; ++idx_L) {
        for (size_t idx_s = 0; idx_s < dim_s; ++idx_s) {
            // The index in the uncoupled, flat vector
            size_t uncoupled_idx = idx_L * dim_s + idx_s;

            // Get the QNs of the L and s parts
            QN_pair qn_L = basis_L[idx_L];
            QN_pair qn_s = basis_s[idx_s];

            // Now iterate over the *output* coupled basis states (J, M)
            for (size_t idx_J = 0; idx_J < dim_coupled; ++idx_J) {
                QN_pair qn_J = basis_J[idx_J];

                // Calculate the specific Clebsch-Gordan coefficient for this transition
                double cg_coeff = calculate_clebsch_gordan(
                    qn_L.twoS, qn_L.twoMs, 
                    qn_s.twoS, qn_s.twoMs, 
                    qn_J.twoS, qn_J.twoMs
                );

                if (std::abs(cg_coeff) > 1e-12) {
                    // Place the coefficient into the matrix U
                    U[uncoupled_idx * dim_coupled + idx_J] = cg_coeff;
                }
            }
        }
    }
    return U;
}

// Example usage placeholder (requires the definition of coupled_qn_list from Step 2)
int main() {
    // coupled_qn_list generated from Step 2
    std::vector<QN_pair> coupled_qn_list = { {0, 0}, {2, -2}, {2, 0}, {2, 2} }; // Example order

    std::vector<double> U_matrix = build_transformation_matrix_U(
        basis_L_qn, basis_s_qn, coupled_qn_list
    );

    std::cout << "Transformation Matrix U (dim " << dim_uncoupled << "x" << dim_coupled << "):\n";
    // Print U (requires formatting function)
    return 0;
}

// int main() {

//     const int N = 100;

//     // Host array allocation
//     int *h_out = (int*)malloc(N * sizeof(int));
//     if (!h_out) return EXIT_FAILURE;
    
//     // Device array allocation
//     int *d_out = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

//     int threadsPerBlock = 128;
//     int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
//     kernel<<<blocks, threadsPerBlock>>>(d_out, N);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

//     for (int i = 0; i < N; ++i) {
//         printf("%d^2 = %d\n", i + 1, h_out[i]);
//     }

//     CUDA_CHECK(cudaFree(d_out));
//     std::free(h_out);
//     return 0;
// }