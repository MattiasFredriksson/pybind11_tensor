# CMakeList.txt : CMake project for pybind11_tensor_test.
#
cmake_minimum_required (VERSION 3.8)

# Python dll output: pybind11_tensor_test/transfer
pybind11_add_module(transfer "")

# Preprocessor options
target_compile_options(transfer PRIVATE -O2)

# Preprocessor defines
target_compile_definitions(transfer PRIVATE "EIGEN_HAS_RVALUE_REFERENCES")

# Header libs
target_include_directories(transfer PRIVATE "../../include/")
target_include_directories(transfer PRIVATE ${EigenPath})


# Sources
target_sources(transfer 
	PRIVATE 
	   "../../include/pybind11_eigen_tensor.h"
	   "main.cpp"
	   )