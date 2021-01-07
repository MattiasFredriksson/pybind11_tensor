# CMakeList.txt : CMake project for pybind11_tensor_test.
#
cmake_minimum_required (VERSION 3.8)

# Python dll output: pybind11_tensor_test
add_subdirectory ("pybind11")
pybind11_add_module(pybind11_tensor_test "")

# Preprocessor defines
target_compile_options(pybind11_tensor_test PRIVATE -O2)

# Header libs
target_include_directories(pybind11_tensor_test PRIVATE "../include/")
target_include_directories(pybind11_tensor_test PRIVATE ${EigenPath})

# Preprocessor defines
target_compile_definitions(pybind11_tensor_test PRIVATE "")


# Sources
target_sources(pybind11_tensor_test 
	PRIVATE 
	   "main.cpp"
	   "../include/pybind11_eigen_tensor.h"
	   "../include/pybind11_eigen_quat.h")