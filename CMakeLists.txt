﻿# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("pybind11_tensor_test")
set (CMAKE_CXX_STANDARD 20)

# Include sub-projects.
add_subdirectory ("pybind11_tensor_test")
