# CMakeList.txt : CMake project for pybind11_tensor_test.
#
cmake_minimum_required (VERSION 3.8)

# Python dll output: pybind11_tensor_test

# External durectories
find_package(pybind11)
find_package(Eigen3)

add_subdirectory ("transfer")
add_subdirectory ("slice")