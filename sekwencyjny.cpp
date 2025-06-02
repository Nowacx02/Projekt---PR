#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring> // dla memcpy

// Alokacja i inicjalizacja macierzy kwadratowej
double* allocateMatrix(int size, bool randomize = false) {
    double* matrix = new double[size * size];
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = randomize ? (rand() % 10) : 0.0;
    }
    return matrix;
}

void multiplyMatrices(const double* A, const double* B, double* result, int size) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            result[i * size + j] = 0.0;
            for (int k = 0; k < size; ++k)
                result[i * size + j] += A[i * size + k] * B[k * size + j];
        }
}

void matrixPower(const double* base, double* result, int size, int power) {
    for (int i = 0; i < size * size; ++i)
        result[i] = (i / size == i % size) ? 1.0 : 0.0;

    double* temp = allocateMatrix(size);
    double* acc = allocateMatrix(size);
    memcpy(acc, base, sizeof(double) * size * size);

    while (power > 0) {
        if (power % 2 == 1) {
            multiplyMatrices(result, acc, temp, size);
            std::swap(result, temp);
        }
        multiplyMatrices(acc, acc, temp, size);
        std::swap(acc, temp);
        power /= 2;
    }

    delete[] temp;
    delete[] acc;
}

// Wypisanie macierzy
void printMatrix(const double* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j)
            std::cout << matrix[i * size + j] << " ";
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    int size = 100;
    int power = 100;

    if (argc >= 3) {
        size = std::atoi(argv[1]);
        power = std::atoi(argv[2]);
    }

    srand(42); // ustalenie ziarna

    double* A = allocateMatrix(size, true);
    double* result = allocateMatrix(size);

    auto start = std::chrono::high_resolution_clock::now();
    matrixPower(A, result, size, power);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Time (sequential): " << duration.count() << " ms\n";


    //printMatrix(A, size);
    //std::cout << "A^" << power << ":\n";
    //printMatrix(result, size);

    delete[] A;
    delete[] result;
    return 0;
}
