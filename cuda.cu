#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Kernel mnożący macierze z użyciem pamięci shared (tiled multiplication)
__global__ void matrixMultiplyTiled(const double* A, const double* B, double* C, int n) {
    __shared__ double ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    double Pvalue = 0.0;

    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < numTiles; ++ph) {
        if (Row < n && ph * TILE_WIDTH + tx < n)
            ds_A[ty][tx] = A[Row * n + ph * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0;

        if (Col < n && ph * TILE_WIDTH + ty < n)
            ds_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * n + Col];
        else
            ds_B[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < n && Col < n)
        C[Row * n + Col] = Pvalue;
}

// Funkcja wywołująca kernel mnożenia macierzy
void multiplyMatricesCUDA(const double* d_A, const double* d_B, double* d_C, int size) {
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((size + TILE_WIDTH - 1) / TILE_WIDTH,
                  (size + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();
}

// Kernel inicjalizujący macierz jednostkową na GPU
__global__ void initIdentityMatrix(double* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        matrix[idx] = (row == col) ? 1.0 : 0.0;
    }
}

// Funkcja potęgowania macierzy z użyciem CUDA
void matrixPowerCUDA(const double* h_base, double* h_result, int size, int power) {
    size_t bytes = size * size * sizeof(double);

    // Alokacja pamięci na GPU
    double *d_acc, *d_result, *d_temp;
    cudaMalloc(&d_acc, bytes);
    cudaMalloc(&d_result, bytes);
    cudaMalloc(&d_temp, bytes);

    // Skopiuj bazową macierz na GPU
    cudaMemcpy(d_acc, h_base, bytes, cudaMemcpyHostToDevice);

    // Inicjalizacja macierzy wynikowej jako jednostkowa na GPU
    int totalThreads = size * size;
    int block = 256;
    int grid = (totalThreads + block - 1) / block;
    initIdentityMatrix<<<grid, block>>>(d_result, size);
    cudaDeviceSynchronize();

    // Realizacja szybkiego potęgowania macierzy na GPU
    while (power > 0) {
        if (power & 1) {
            multiplyMatricesCUDA(d_result, d_acc, d_temp, size);
            std::swap(d_result, d_temp);
        }
        multiplyMatricesCUDA(d_acc, d_acc, d_temp, size);
        std::swap(d_acc, d_temp);
        power >>= 1;
    }

    // Skopiuj wynik z GPU na CPU
    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);

    // Zwolnij pamięć
    cudaFree(d_acc);
    cudaFree(d_result);
    cudaFree(d_temp);
}

int main(int argc, char* argv[]) {
    int size = 200;
    int power = 100; // do testów

    if (argc >= 3) {
        size = std::atoi(argv[1]);
        power = std::atoi(argv[2]);
    }

    srand(42);
    double* A = new double[size * size];
    for (int i = 0; i < size * size; ++i)
        A[i] = rand() % 10;

    double* result = new double[size * size];

    auto start = std::chrono::high_resolution_clock::now();
    matrixPowerCUDA(A, result, size, power);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time (CUDA tiled): " << duration.count() << " ms\n";

    delete[] A;
    delete[] result;
    return 0;
}
