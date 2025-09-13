// github.com/51ddhesh
// MIT License

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_DIM 16

// * Helper Utilities
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0) 


/*
*   Kernel 1: Transpose Matrix
* @brief This kernel transposes a matrix
* @note Transpose K before the first matrix multiplication.
*       Transposing K to K.T allows for coalasced memory access
*       in Q * K.T, which is more efficient than strided access
*/
__global__ void transpose(const float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Calculate global thread corrdinates
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load data into shared memory tile if within the matrix bounds
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    // Synchronize all the threads to avoid 
    __syncthreads();

    // Calculate the transposed global coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write the transposed matrix from shared memory to the output matrix
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

/*
*   Kernel 2: Fused Matmul (Q * K.T) and Scaling
* @brief This kernel performs the first major step: C = (Q * K.T) / (sqrt(d_k))
* @note Uses tiled matrix multiplication
*       Each thread block calculates one tile of the output matrix C 
*/
__global__ void matmul_scale(const float* A, const float* B, float* C, int M, int N, int K, float scale) {
    // A: Q (M * K), B: K.T (K * N), C: Output (M * N)

    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float acc = 0.0f; // accumulator for dot product

    // Loop over the tiles of A and B to compute C
    for (int i = 0; i < (K + TILE_DIM - 1) / TILE_DIM; i++) {
        // Load the tile of A into shared memory
        if (row < M && (i * TILE_DIM + tx) < K) {
            tileA[ty][tx] = A[row * K + (i * TILE_DIM + tx)];
        } else tileA[ty][tx] = 0.0f;

        // Load the tile of B into shared memory
        if (col < N && (i * TILE_DIM + ty) < K) {
            tileB[ty][tx] = B[(i * TILE_DIM + ty) * N + col];
        } else tileB[ty][tx] = 0.0f;

        // Synchronize threads
        __syncthreads();

        // Perform dot product for the current tiles
        for (int i = 0; i < TILE_DIM; i++) {
            acc += tileA[ty][i] * tileB[i][tx];
        }

        // Synchronize threads
        __syncthreads();
    }

    // Write the final scaled result to global memory
    if (row < M && col < N) {
        C[row * N + col] = acc * scale;
    }
}

// driver 
int main() {
    constexpr int batch_size = 4;
    constexpr int seq_len    = 1024; // Sequence length
    constexpr int d_k        = 64;   // Dimension of keys and queries
    constexpr int d_v        = 64;   // Dimension of values
    assert(d_k == d_v); // sanity check
}
