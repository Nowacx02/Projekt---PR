#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring> // memcpy
#include <omp.h>   // OpenMP

double* allocateMatrix(int size, bool randomize = false) {
    double* matrix = new double[size * size];
    for (int i = 0; i < size * size; ++i)
        matrix[i] = randomize ? (rand() % 10) : 0.0;
    return matrix;
}

// Równoległe mnożenie dwóch macierzy A i B
void multiplyMatricesOMP(const double* A, const double* B, double* result, int size) {
    #pragma omp parallel for collapse(2) default(none) shared(A, B, result, size)
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k)
                sum += A[i * size + k] * B[k * size + j];
            result[i * size + j] = sum;
        }
}

void matrixPowerOMP(const double* base, double* result, int size, int power) {
    for (int i = 0; i < size * size; ++i)
        result[i] = (i / size == i % size) ? 1.0 : 0.0;

    double* temp = allocateMatrix(size);
    double* acc = allocateMatrix(size);
    memcpy(acc, base, sizeof(double) * size * size);

    while (power > 0) {
        if (power % 2 == 1) {
            multiplyMatricesOMP(result, acc, temp, size);
            std::swap(result, temp);
        }
        multiplyMatricesOMP(acc, acc, temp, size);
        std::swap(acc, temp);
        power /= 2;
    }

    delete[] temp;
    delete[] acc;
}

int main(int argc, char* argv[]) {
    int size = 100;
    int power = 100;

    if (argc >= 3) {
        size = std::atoi(argv[1]);
        power = std::atoi(argv[2]);
    }

    omp_set_num_threads(8);

    srand(42);
    double* A = allocateMatrix(size, true);
    double* result = allocateMatrix(size);

    auto start = std::chrono::high_resolution_clock::now();
    matrixPowerOMP(A, result, size, power);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time (OpenMP): " << duration.count() << " ms\n";

    delete[] A;
    delete[] result;
    return 0;
}
