#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <map>
#include <algorithm>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <iomanip> 
#include <limits>


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

// Function log-factorial (log(n!))
double log_factorial(double n) {
    if (n < 0) return -std::numeric_limits<double>::infinity();
    if (n == 0) return 0.0;
    // We use the lgamma function, which is log(Gamma(x)) and Gamma(n+1) = n!
    // lgamma(n+1) = log(n!)
    return std::lgamma(n + 1);
}

// Helper function for the Wigner 3j symbol calculation
double wigner_3j(double j1, double j2, double J, double m1, double m2, double M) {
    // Selection rules - te zostają bez zmian
    if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(M) > J) return 0.0;
    if (std::abs(j1 - j2) > J + 1e-9 || j1 + j2 < J - 1e-9) return 0.0;
    if (std::abs(m1 + m2 + M) > 1e-9) return 0.0;
    
    // We use log factorials to calculate the pre-factor
    double log_pre_factor = 0.5 * (
        log_factorial(j1 + j2 - J) + log_factorial(j1 - j2 + J) + 
        log_factorial(-j1 + j2 + J) - log_factorial(j1 + j2 + J + 1) +
        log_factorial(j1 + m1) + log_factorial(j1 - m1) + 
        log_factorial(j2 + m2) + log_factorial(j2 - m2) +
        log_factorial(J + M) + log_factorial(J - M)
    );

    // Summation - now we need to dynamically determine the range of k, 
    // to avoid factorials of negative numbers
    double sum_val = 0.0;
    int k_min = std::max({0, (int)std::ceil(j2 - J - m1), (int)std::ceil(j1 - J + m2)});
    int k_max = std::min({(int)std::floor(j1 + j2 - J), (int)std::floor(j1 - m1), (int)std::floor(j2 + m2)});

    for (int k = k_min; k <= k_max; ++k) {
        // Calculate the log of the denominator factorials
        double log_den = log_factorial(k) + log_factorial(j1 + j2 - J - k) + 
                         log_factorial(j1 - m1 - k) + log_factorial(j2 + m2 - k) + 
                         log_factorial(J - j2 + k + m1) + log_factorial(J - j1 + k - m2);
        
        // Alternating sign: (-1)^k
        double term = std::exp(-log_den);
        if (k % 2 != 0) term = -term;
        sum_val += term;
    }
    
    // Final results: exp(log_pre_factor) * sum_val
    return std::exp(log_pre_factor) * sum_val;
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

void print_matrix_U(
    const std::vector<double>& U_matrix, 
    size_t dim_uncoupled, 
    size_t dim_coupled
) {
    std::cout << "Transformation Matrix U (" << dim_uncoupled << "x" << dim_coupled << "):" << std::endl;
    std::cout << std::fixed << std::setprecision(3); // Set precision for floating point output

    // Iterate over rows (uncoupled index)
    for (size_t i = 0; i < dim_uncoupled; ++i) {
        // Iterate over columns (coupled index)
        for (size_t j = 0; j < dim_coupled; ++j) {
            // Calculate the 1D index for the 2D element U[i][j]
            size_t index_1D = i * dim_coupled + j;
            double value = U_matrix[index_1D];
            
            std::cout << std::setw(8) << value;
        }
        std::cout << std::endl; // Newline at the end of each row
    }
}


// Example usage placeholder (requires the definition of coupled_qn_list from Step 2)
int main() {
    // coupled_qn_list generated from Step 2
    std::vector<QN_pair> coupled_qn_list = { {0, 0}, {2, -2}, {2, 0}, {2, 2} }; // Example order
    const size_t dim_coupled = coupled_qn_list.size();

    std::vector<double> U_matrix = build_transformation_matrix_U(
        basis_L_qn, basis_s_qn, coupled_qn_list
    );

    print_matrix_U(U_matrix, dim_uncoupled, coupled_qn_list.size());

    return 0;
}