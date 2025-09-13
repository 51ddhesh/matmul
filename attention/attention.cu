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
#define CUDA_CHECK(call)                                                              \
do {                                                                                  \
    cudaError_t err = call;                                                           \
    if (err != cudaSuccess) {                                                         \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,              \
                cudaGetErrorString(err));                                             \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
} while (0)


/*
* Kernel 1: Transpose Matrix
* @brief This kernel transposes a matrix
* @note Transpose K before the first matrix multiplication.
* Transposing K to K.T allows for coalesced memory access
* in Q * K.T, which is more efficient than strided access
*/
__global__ void transpose(const float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Calculate global thread coordinates
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load data into shared memory tile if within the matrix bounds
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    // Synchronize all the threads to avoid race conditions
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
* Kernel 2: Fused Matmul (Q * K.T) and Scaling
* @brief This kernel performs the first major step: C = (Q * K.T) / (sqrt(d_k))
* @note Uses tiled matrix multiplication
* Each thread block calculates one tile of the output matrix C 
*/
__global__ void matmul_scale(const float* A, const float* B, float* C, int M, int N, int K, float scale) {
    // A: Q (M * K), B: K.T (K * N), C: Output (M * N)

    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f; // accumulator for dot product

    // Loop over the tiles of A and B to compute C
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        // Load the tile of A into shared memory
        if (row < M && (t * TILE_DIM + tx) < K) {
            tileA[ty][tx] = A[row * K + (t * TILE_DIM + tx)];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Load the tile of B into shared memory
        if (col < N && (t * TILE_DIM + ty) < K) {
            tileB[ty][tx] = B[(t * TILE_DIM + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        // Synchronize threads
        __syncthreads();

        // Perform dot product for the current tiles
        for (int k = 0; k < TILE_DIM; k++) {
            acc += tileA[ty][k] * tileB[k][tx];
        }

        // Synchronize threads before the next tile
        __syncthreads();
    }

    // Write the final scaled result to global memory
    if (row < M && col < N) {
        C[row * N + col] = acc * scale;
    }
}

/*
* Kernel 3: Row-wise softmax
* @brief Calculates the softmax for each row of the input matrix
* @note Softmax = exp(x_i) / sum(exp(x_j))
* Implemented in a numerically stable way by subtracting the
* max value of the row from each element
* Each thread block processes one row
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
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];

    // Calculate the sum of exponentials (parallel reduction)
    float sum = 0.0f;
    // FIX: Changed i++ to i += blockDim.x for correct parallel striding
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += expf(data[row * N + i] - max_val);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within the block to get the total sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    } 
    sum = sdata[0];

    // Normalize and write back
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        data[row * N + i] = expf(data[row * N + i] - max_val) / sum;
    }
} 

/*
* Kernel 4: Final Matmul (P_softmax * V)
* @brief Second tiled matmul, similar to kernel 2
* @note Has no scaling factor unlike the previous matmul kernel
* Computes P_softmax * V
*/
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // A: P_softmax (M * K)
    // B: V (K * N)
    // C: Output (M * N)

    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        if (row < M && (t * TILE_DIM + tx) < K) {
            tileA[ty][tx] = A[row * K + (t * TILE_DIM + tx)];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if ((t * TILE_DIM + ty) < K && col < N) {
            tileB[ty][tx] = B[(t * TILE_DIM + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            acc += tileA[ty][k] * tileB[k][tx];
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

/*
* @brief function to verify result
*/
void verify_results(const float* cpu_res, const float* gpu_res, int size) {
    double total_error = 0.0;
    float max_error = 0.0f;
    for (int i = 0; i < size; ++i) {
        float error = fabsf(cpu_res[i] - gpu_res[i]);
        total_error += error;
        if (error > max_error) {
            max_error = error;
        }
    }
    double avg_error = total_error / size;
    std::cout << "Verification Results:" << std::endl;
    std::cout << "  Max Error: " << max_error << std::endl;
    std::cout << "  Average Error: " << avg_error << std::endl;

    if (avg_error < 1e-5 && max_error < 1e-4) {
        std::cout << "  ✅ Verification PASSED" << std::endl;
    } else {
        std::cout << "  ❌ Verification FAILED" << std::endl;
    }
}


// driver 
int main() {
    const int batch_size = 4;
    const int seq_len    = 1024; // Sequence length
    const int d_k        = 64;   // Dimension of keys and queries
    const int d_v        = 64;   // Dimension of values
    assert(d_k == d_v); // sanity check

    std::cout << "Attention Parameters:" << std::endl;
    std::cout << "  Batch Size: " << batch_size << std::endl;
    std::cout << "  Sequence Length: " << seq_len << std::endl;
    std::cout << "  Head Dimension (d_k, d_v): " << d_k << std::endl;

    // --- 2. Allocate and initialize host memory ---
    const int qk_size = batch_size * seq_len * d_k;
    const int v_size = batch_size * seq_len * d_v;
    const int output_size = v_size;

    std::vector<float> h_q(qk_size);
    std::vector<float> h_k(qk_size);
    std::vector<float> h_v(v_size);
    std::vector<float> h_output_gpu(output_size);
    std::vector<float> h_output_cpu(output_size);

    std::mt19937 gen(1337);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < qk_size; ++i) {
        h_q[i] = dis(gen);
        h_k[i] = dis(gen);
    }
    for (int i = 0; i < v_size; ++i) h_v[i] = dis(gen);
    
    // --- 3. Allocate device memory ---
    // FIX: Renamed float* d_v to d_v_ptr to avoid conflict with const int d_v
    float *d_q = nullptr, *d_k_ptr = nullptr, *d_v_ptr = nullptr, *d_k_t = nullptr, *d_p1 = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_q, qk_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_k_ptr, qk_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_v_ptr, v_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_k_t, qk_size * sizeof(float))); // For K transposed
    CUDA_CHECK(cudaMalloc((void**)&d_p1, batch_size * seq_len * seq_len * sizeof(float))); // For Q*K^T result
    CUDA_CHECK(cudaMalloc((void**)&d_output, output_size * sizeof(float)));

    // --- 4. Copy data from Host to Device ---
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), qk_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_ptr, h_k.data(), qk_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_ptr, h_v.data(), v_size * sizeof(float), cudaMemcpyHostToDevice));

    // --- 5. Prepare for kernel launches and benchmarking ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Main loop over the batch ---
    std::cout << "\nStarting GPU computation and benchmarking..." << std::endl;
    // Warm-up run
    for (int b = 0; b < 1; ++b) {
        const float* q_batch = d_q + b * seq_len * d_k;
        const float* k_batch = d_k_ptr + b * seq_len * d_k;
        float* k_t_batch = d_k_t + b * seq_len * d_k;
        // Transpose K -> K^T
        dim3 transpose_grid((d_k + TILE_DIM - 1) / TILE_DIM, (seq_len + TILE_DIM - 1) / TILE_DIM);
        dim3 transpose_block(TILE_DIM, TILE_DIM);
        transpose<<<transpose_grid, transpose_block>>>(k_batch, k_t_batch, d_k, seq_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing loop
    float total_ms = 0;
    int iterations = 100;
    for (int iter = 0; iter < iterations; ++iter) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int b = 0; b < batch_size; ++b) {
            // Get pointers to the current batch slice
            const float* q_batch = d_q + b * seq_len * d_k;
            const float* k_batch = d_k_ptr + b * seq_len * d_k;
            // FIX: Use d_v_ptr for pointer arithmetic
            const float* v_batch = d_v_ptr + b * seq_len * d_v;
            float* k_t_batch = d_k_t + b * seq_len * d_k;
            float* p1_batch = d_p1 + b * seq_len * seq_len;
            // FIX: Use d_v for the integer dimension in offset calculation
            float* output_batch = d_output + b * seq_len * d_v;

            // --- Step 1: Transpose K -> K^T ---
            dim3 transpose_grid((d_k + TILE_DIM - 1) / TILE_DIM, (seq_len + TILE_DIM - 1) / TILE_DIM);
            dim3 transpose_block(TILE_DIM, TILE_DIM);
            transpose<<<transpose_grid, transpose_block>>>(k_batch, k_t_batch, d_k, seq_len);
            
            // --- Step 2: P = (Q * K^T) / sqrt(d_k) ---
            float scale = 1.0f / sqrtf(static_cast<float>(d_k));
            dim3 matmul_grid((seq_len + TILE_DIM - 1) / TILE_DIM, (seq_len + TILE_DIM - 1) / TILE_DIM);
            dim3 matmul_block(TILE_DIM, TILE_DIM);
            matmul_scale<<<matmul_grid, matmul_block>>>(q_batch, k_t_batch, p1_batch, seq_len, seq_len, d_k, scale);

            // --- Step 3: Softmax(P) ---
            int softmax_block_size = 256; // Common choice, can be tuned
            size_t smem_size = softmax_block_size * sizeof(float);
            softmax<<<seq_len, softmax_block_size, smem_size>>>(p1_batch, seq_len, seq_len);
            
            // --- Step 4: Output = P * V ---
            // FIX: Pass the integer d_v as the dimension argument
            matmul<<<matmul_grid, matmul_block>>>(p1_batch, v_batch, output_batch, seq_len, d_v, seq_len);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nGPU Benchmarking Results:" << std::endl;
    std::cout << "  Total iterations: " << iterations << std::endl;
    std::cout << "  Average time per batch of " << batch_size << ": " << total_ms / iterations << " ms" << std::endl;
    std::cout << "  Throughput: " << (iterations * batch_size) / (total_ms / 1000.0) << " sequences/sec" << std::endl;

    // --- 6. Copy result from Device to Host ---
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // --- 7. Verify results against CPU implementation ---
    // FIX: Pass the integer d_v as the dimension argument
    cpu_attn(h_q.data(), h_k.data(), h_v.data(), h_output_cpu.data(), batch_size, seq_len, d_k, d_v);
    verify_results(h_output_cpu.data(), h_output_gpu.data(), output_size);

    // --- 8. Free memory ---
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k_ptr));
    // FIX: Free the renamed pointer
    CUDA_CHECK(cudaFree(d_v_ptr));
    CUDA_CHECK(cudaFree(d_k_t));
    CUDA_CHECK(cudaFree(d_p1));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
