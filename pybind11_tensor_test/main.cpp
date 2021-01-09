// DTW_python.cpp : Defines the entry point for the application.
//
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "pybind11_eigen_tensor.h"

namespace py = pybind11;
namespace pyd = pybind11::detail; 


template<typename FP>
void print_const(const Eigen::TensorRef<const Eigen::Tensor<FP, 3, 1>> tensor) {
    std::cout << tensor << std::endl;
}
template<typename FP>
void print_nonconst(Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> tensor) {
    std::cout << tensor << std::endl;
}

template<typename FP>
Eigen::Tensor<FP, 3, 1> add_self(Eigen::Tensor<FP, 3, 1> tensor) {
    return tensor + tensor;
}

template<typename FP>
Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> add_self_ref(Eigen::TensorRef< Eigen::Tensor<FP, 3, 1>> tensor) {
    // Return tensor evaluated tensor packed in ref argument
    tensor = Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>>(Eigen::Tensor<FP, 3, 1>(tensor + tensor));
    auto c(tensor);
    new(&c)Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>>(Eigen::Tensor<FP, 3, 1>(tensor + tensor + tensor));
    
    std::cout << tensor.data() << std::endl;

    Eigen::Tensor<FP, 3, 1> e(tensor + tensor + tensor);
    std::cout << e.data() << std::endl;
    Eigen::Tensor<FP, 3, 1> f(tensor + tensor + tensor);
    std::cout << f.data() << std::endl;
    Eigen::Tensor<FP, 3, 1> g(tensor + tensor + tensor);
    std::cout << g.data() << std::endl;
    Eigen::Tensor<FP, 3, 1> h(tensor + tensor + tensor);
    std::cout << h.data() << std::endl;

    return tensor;
}

template<typename FP>
Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> add_self_ref_expr(Eigen::TensorRef< Eigen::Tensor<FP, 3, 1>> tensor) {
    // Return tensor expression as ref
    return tensor + tensor;
}


template<typename FP>
Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> double_ref(Eigen::TensorRef<Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>>> tensor) {
    return tensor + tensor;
}


template<typename FP>
void add_self_map(Eigen::TensorMap<Eigen::Tensor<FP, 3, 1>>& tensor) {
    tensor = tensor + tensor;
}

template<typename FP>
Eigen::TensorMap<Eigen::Tensor<FP, 3, 1>> add_self_map_ret(Eigen::TensorMap<Eigen::Tensor<FP, 3, 1>>& tensor) {
    tensor = tensor + tensor;

    return tensor;
}

PYBIND11_MODULE(pybind11_tensor_test, m) {
    m.doc() = R"pbdoc(
        Python module to implement test functions for pybind11_eigen_tensor.h
        -----------------------

        .. currentmodule:: pybind11_tensor_test

        .. autosummary::
           :toctree: _generate

    )pbdoc";
    m.def("print_const", &print_const<double>, py::return_value_policy::copy, R"pbdoc(
        Print a const entity.
    )pbdoc");
    m.def("print", &print_nonconst<double>, R"pbdoc(
        Print a non-const entity.
    )pbdoc");
    m.def("add_self", &add_self<double>, R"pbdoc(
        Add self to self.
    )pbdoc");
    m.def("add_self_map", &add_self_map<double>, R"pbdoc(
        Add self to self.
    )pbdoc");
    m.def("add_self_map_ret", &add_self_map_ret<double>, R"pbdoc(
        Add self to self and return self.
    )pbdoc");

    m.def("add_self_ref", &add_self_ref<double>, R"pbdoc(
        Add self to self.
    )pbdoc");
    m.def("add_self_ref_expr", &add_self_ref_expr<double>, R"pbdoc(
        Add self to self.
    )pbdoc");
    m.def("double_ref", &double_ref<double>, R"pbdoc(
        Pass ref ref.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "test";
#endif
}

