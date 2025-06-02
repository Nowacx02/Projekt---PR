#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring> // dla memcpy
#include <CL/cl.h> // OpenCL

// Alokacja i inicjalizacja macierzy kwadratowej
double* allocateMatrix(int size, bool randomize = false) {
    double* matrix = new double[size * size];
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = randomize ? (rand() % 10) : 0.0;
    }
    return matrix;
}

// Kod OpenCL do potęgowania macierzy (kernel)
const char* kernelSource = R"(
__kernel void matrixMultiply(__global const double* A, __global const double* B, __global double* C, const int size) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    double sum = 0.0;

    for (int k = 0; k < size; ++k) {
        sum += A[row * size + k] * B[k * size + col];
    }

    C[row * size + col] = sum;
}
)";

// Funkcja potęgowania macierzy (OpenCL)
void matrixPowerOpenCL(const double* base, double* result, int size, int power) {
    // OpenCL: Inicjalizacja platformy i urządzenia
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // OpenCL: Kontekst i kolejka
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // OpenCL: Kompilacja jądra
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", nullptr);

    // OpenCL: Bufory pamięci
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * size * size, (void*)base, nullptr);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * size * size, nullptr, nullptr);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * size * size, nullptr, nullptr);

    // Kopiowanie macierzy wejściowej
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, sizeof(double) * size * size, base, 0, nullptr, nullptr);

    for (int i = 0; i < power; ++i) {
        // Ustawienie argumentów jądra
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferB);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
        clSetKernelArg(kernel, 3, sizeof(int), &size);

        size_t globalSize[] = { (size_t)size, (size_t)size };
        clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);

        clFinish(queue);
        std::swap(bufferB, bufferC);
    }

    clEnqueueReadBuffer(queue, bufferB, CL_TRUE, 0, sizeof(double) * size * size, result, 0, nullptr, nullptr);

    // OpenCL: Sprzątanie
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// Funkcja główna
int main(int argc, char* argv[]) {
    int size = 100;
    int power = 100;

    if (argc >= 3) {
        size = std::atoi(argv[1]);
        power = std::atoi(argv[2]);
    }

    srand(42);

    double* A = allocateMatrix(size, true);
    double* result = allocateMatrix(size);

    auto start = std::chrono::high_resolution_clock::now();
    matrixPowerOpenCL(A, result, size, power);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time (OpenCL): " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    delete[] A;
    delete[] result;
    return 0;
}
