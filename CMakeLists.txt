# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("pybind11_tensor_test")

# Path to eigen library goes here:
set(EigenPath "${CMAKE_CURRENT_LIST_DIR}/../../external/eigen/")

# Include sub-projects.
add_subdirectory ("pybind11_tensor_test")
