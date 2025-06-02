cmake_minimum_required(VERSION 3.29)
project(Projekt)

set(CMAKE_CXX_STANDARD 20)

# Wymuś użycie OpenMP – działa z MinGW
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Ręczne ustawienie ścieżek do OpenCL (CUDA)
set(OpenCL_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include")
set(OpenCL_LIBRARY "C:/Windows/System32/OpenCL.dll")

# Dodanie ścieżki do nagłówków i biblioteki
include_directories(${OpenCL_INCLUDE_DIR})
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64")
link_libraries(${OpenCL_LIBRARY})

# Wybierz odpowiedni plik do kompilacji
add_executable(Projekt opencl.cpp)
