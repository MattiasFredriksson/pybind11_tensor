# CMakeList.txt : CMake project for pybind11_tensor_test.
#
cmake_minimum_required (VERSION 3.8)

# Python dll output: pybind11_tensor_test
add_subdirectory ("pybind11")
add_subdirectory ("transfer")
add_subdirectory ("slice")