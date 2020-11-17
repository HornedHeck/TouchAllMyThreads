cmake_minimum_required(VERSION 3.17)
project(ACS_20)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(ACS_20 main.cpp)

target_link_libraries(ACS_20 PRIVATE OpenMP::OpenMP_CXX)