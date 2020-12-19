// DTW_python.cpp : Defines the entry point for the application.
//
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "pybind11_eigen_tensor.h"

namespace py = pybind11;
namespace pyd = pybind11::detail; 

 
void print_const(const Eigen::TensorRef<const Eigen::Tensor<float, 3, 1>>& tensor) {
    std::cout << tensor << std::endl;
}
void print_nonconst(Eigen::TensorRef<Eigen::Tensor<float, 3, 1>>& tensor) {
    std::cout << tensor << std::endl;
}
Eigen::TensorRef<Eigen::Tensor<float, 3, 1>> double_ref(Eigen::TensorRef<Eigen::TensorRef<Eigen::Tensor<float, 3, 1>>> tensor) {
    return tensor + tensor;
}
Eigen::TensorRef<Eigen::Tensor<float, 3, 1>> add_self(Eigen::TensorRef< Eigen::Tensor<float, 3, 1>> tensor) {
    return tensor + tensor;
}

void add_self_map(Eigen::TensorMap<Eigen::Tensor<float, 3, 1>>& tensor) {
    tensor = tensor + tensor;
}

PYBIND11_MODULE(pybind11_tensor_test, m) {
    m.doc() = R"pbdoc(
        Python module to implement test functions for pybind11_eigen_tensor.h
        -----------------------

        .. currentmodule:: pybind11_tensor_test

        .. autosummary::
           :toctree: _generate

    )pbdoc";
    m.def("print_const", &print_const, py::return_value_policy::copy, R"pbdoc(
        Print a const entity.
    )pbdoc");
    m.def("print", &print_nonconst, R"pbdoc(
        Print a non-const entity.
    )pbdoc");
    m.def("add_self", &add_self, R"pbdoc(
        Add self to self.
    )pbdoc");
    m.def("add_self_map", &add_self_map, R"pbdoc(
        Add self to self.
    )pbdoc");
    m.def("double_ref", &double_ref, R"pbdoc(
        Pass ref ref.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "test";
#endif
}

