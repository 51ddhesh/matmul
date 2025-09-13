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

/*
*   Kernel 3: Row-wise softmax
* @brief Calculates the softmax for each row of the input matrix
* @note Softmax = exp(x_i) / sum(exp(x_j))
*       Implemented in a numerically stable way by subtracting the
*       max value of the row from each element
*       Each thread processes one row
* @param M: number of rows, N: length of rows
*/

__global__ void softmax(float* data, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    // Shared memory to hold current row and intermediate values
    extern __shared__ float sdata[];

    // Find the maximum value in a row
    float max_val = -1e20f; // init with a very small number
    // Each thread loads one element from the row and finds its local max
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, data[row * N + i]);
    }
    sdata[threadIdx.x] = max_val;

    __syncthreads();

    // Reduce within the block to find the true max
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + i]);
        }
        __syncthreads();
    }

    max_val = sdata[0];

    // Calculate the sum of exponentials (parallel reduction)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i++) {
        sum += expf(data[row * N + i] - max_val);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within the block to get the total sum
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    } 

    sum = sdata[0];

    // Normalize and write the block
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        data[row * N + i] = expf(data[row * N + i] - max_val) / sum;
    }
} 



/*
*   Kernel 4: Final Matmul (P_softmax * V)
* @brief Second tiled matmul, similar to kernel 2
* @note Has no scaling factor unlike the previous matmul kernel
*       Computes P_softmax * V
*/
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // A: P_softmax (M * K)
    // B: V (K * N)
    // C: Output (M * N)

    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float acc = 0.0f;

    for (int i = 0; i < (K + TILE_DIM - 1) / TILE_DIM; i++) {
        if (row < M && (i * TILE_DIM + tx) < K) {
            tileA[ty][tx] = A[row * K + (i * TILE_DIM + tx)];
        } else tileA[ty][tx] = 0.0f;

        if ((i * TILE_DIM + ty) < K && col < N) {
            tileB[ty][tx] = B[(i * TILE_DIM + ty) * N + col];
        } else tileB[ty][tx] = 0.0f;
        
        __syncthreads();

        for (int i = 0; i < TILE_DIM; ++i) {
            acc += tileA[ty][i] * tileB[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }

}

/*
* @brief CPU implementation of attention for verification 
*/
void cpu_attn(const float* q, const float* k, const float* v, float* output,
        int batch_size, int seq_len, int d_k, int d_v
        ) {
    std::cout << "\nRunning CPU attention for verification..." << std::endl;
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));

    for (int b = 0; b < batch_size; ++b) {
        // P = Q * K^T
        std::vector<float> p(seq_len * seq_len);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (int l = 0; l < d_k; ++l) {
                    dot += q[b * seq_len * d_k + i * d_k + l] * k[b * seq_len * d_k + j * d_k + l];
                }
                p[i * seq_len + j] = dot * scale;
            }
        }
        
        // Softmax(P)
        for (int i = 0; i < seq_len; ++i) {
            float max_val = -1e20f;
            for (int j = 0; j < seq_len; ++j) {
                max_val = fmaxf(max_val, p[i * seq_len + j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                p[i * seq_len + j] = expf(p[i * seq_len + j] - max_val);
                sum += p[i * seq_len + j];
            }
            for (int j = 0; j < seq_len; ++j) {
                p[i * seq_len + j] /= sum;
            }
        }

        // Output = P * V
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_v; ++j) {
                float acc = 0.0f;
                for (int l = 0; l < seq_len; ++l) {
                    acc += p[i * seq_len + l] * v[b * seq_len * d_v + l * d_v + j];
                }
                output[b * seq_len * d_v + i * d_v + j] = acc;
            }
        }
    }
    std::cout << "CPU attention finished." << std::endl;
}




// driver 
int main() {
    constexpr int batch_size = 4;
    constexpr int seq_len    = 1024; // Sequence length
    constexpr int d_k        = 64;   // Dimension of keys and queries
    constexpr int d_v        = 64;   // Dimension of values
    assert(d_k == d_v); // sanity check

    return 0;
}
