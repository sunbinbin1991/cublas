cmake_minimum_required(VERSION 3.9)
project(OpenblasBenchmark)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -lpthread")

include_directories(C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.0/include/include)
link_directories(C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.0/include/lib/x64)

include_directories(D:/soft/OpenBLAS/include)
link_directories(D:/soft/OpenBLAS/lib)

file(GLOB srcs ./*.cpp)
add_executable(test_cublas test_cublas.cpp)
target_link_libraries(test_cublas openblas cublas cudart)
