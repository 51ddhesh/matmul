#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "random_.hpp"

constexpr size_t N = 10'000;

__global__ void matMulKernel(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


namespace limits_ {
    constexpr int MIN = -1000;
    constexpr int MAX = 1000;
}

int main() {
    size_t num_elements = N * N;
    size_t size = num_elements * sizeof(int);

    std::vector<int> h_A(num_elements);
    std::vector<int> h_B(num_elements);
    std::vector<int> h_C(num_elements, 0);

    std::cout << "Initializing the matrices on the host..." << std::endl;
    for (size_t i = 0; i < num_elements; i++) {
        h_A[i] = random_::getInt(limits_::MIN, limits_::MAX);
        h_B[i] = random_::getInt(limits_::MIN, limits_::MAX);
    }
    std::cout << "Matrices initialized." << std::endl;

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    std::cout << "Copying data from host to device..." << std::endl;
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks( (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Beginning the multiplication on the GPU." << std::endl;
    auto matmul_start = std::chrono::high_resolution_clock::now();

    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    auto matmul_end = std::chrono::high_resolution_clock::now();
    std::cout << "Matrix Multiplication completed." << std::endl;

    std::cout << "Copying result from device to host..." << std::endl;
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    auto matmul_duration_micro = std::chrono::duration_cast<std::chrono::microseconds>(matmul_end - matmul_start);
    auto matmul_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(matmul_end - matmul_start);
    auto matmul_duration_s = std::chrono::duration<double>(matmul_end - matmul_start);

    std::cout << "\033[32m[GPU EXECUTION TIME]\033[0m " << matmul_duration_micro.count() << " μs ("
              << matmul_duration_ms.count() << " ms == " << matmul_duration_s.count() << " s)" << std::endl;

    return 0;
}