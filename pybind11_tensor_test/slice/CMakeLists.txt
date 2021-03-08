# CMakeList.txt : CMake project for pybind11_tensor_test.
#
cmake_minimum_required (VERSION 3.8)

# Python dll output: pybind11_tensor_test/slice
pybind11_add_module(slice "")

# Preprocessor options
target_compile_options(slice PRIVATE -openmp)
target_compile_options(slice PRIVATE -O2)

# Compilation options
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

# Preprocessor defines
target_compile_definitions(slice PRIVATE "EIGEN_HAS_RVALUE_REFERENCES")

# Header libs
target_include_directories(slice PRIVATE "../../include/")
target_include_directories(slice PRIVATE ${EigenPath})


# Sources
target_sources(slice 
	PRIVATE 
	   "../../include/pybind11_eigen_tensor.h"
	   "slice.cpp"
	   )