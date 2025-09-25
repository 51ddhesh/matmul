#include <iostream>
#include <vector>
#include "random_.hpp"

// Size of matrix -> N * N
constexpr size_t N = 2'000;

// Namespace to hold the limits of getInt() in random_.hpp
namespace limits_ {
    constexpr int MIN = -1'000;
    constexpr int MAX = 1'000;
}

int main() {
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 0));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 0));
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    std::cout << "Initializing the matrices..." << std::endl;

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i][j] = random_::getInt(limits_::MIN, limits_::MAX);       
        }
    }
    
    std::cout << "Matrix A initialized." << std::endl;
    
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            B[i][j] = random_::getInt(limits_::MIN, limits_::MAX);       
        }
    }
 
    std::cout << "Matrix B initialized." << std::endl;

    std::cout << "Beginning the multiplication." << std::endl;

    auto matmul_start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    auto matmul_end = std::chrono::high_resolution_clock::now();

    std::cout << "Matrix Multiplication completed." << std::endl;

    auto matmul_duration_micro = std::chrono::duration_cast<std::chrono::microseconds> (matmul_end - matmul_start);
    auto matmul_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds> (matmul_end - matmul_start);
    auto matmul_duration_s = matmul_duration_ms / 1000.0;

    std::cout << "\033[32m[EXECUTION TIME]\033[0m " << matmul_duration_micro.count() << " μs (" << matmul_duration_ms.count() << " ms == " << matmul_duration_s.count() << " s)" << std::endl;

    return 0;
}

