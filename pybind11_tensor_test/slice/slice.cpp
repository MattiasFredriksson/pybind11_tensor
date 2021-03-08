// DTW_python.cpp : Defines the entry point for the application.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

//#include "tensor_traits.h"
#include "pybind11_eigen_tensor.h"
#include "eigen_tensor_ext.h"


namespace py = pybind11;
namespace pyd = pybind11::detail;
using namespace tensorial;


void check_eigen_row_col_mult() {
    Eigen::Matrix<float, 3, 3, Eigen::ColMajor> c(Eigen::Matrix<float, 3, 3, Eigen::ColMajor>::Identity());
    c(1, 0) = 1.0;
    c(2, 0) = 1.0;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> r(Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Identity());
    r(1, 0) = 1.0;
    r(2, 0) = 1.0;
    Eigen::Matrix<float, 3, 1, Eigen::ColMajor> cv(1.0, 0.0, 0.0);
    Eigen::Matrix<float, 1, 3, Eigen::RowMajor> rv(1.0, 0.0, 0.0);
    std::cout << "Col matrix:\n" << c << std::endl;
    std::cout << "Col vector:\n" << cv << std::endl;

    std::cout << "Row matrix:\n" << r << std::endl;
    std::cout << "Row vector:\n" << rv << std::endl;

    std::cout << "Colmajor mult m * cv:\n" << c * cv << std::endl;

    // std::cout << "Rowmajor mult: " << r * rv << std::endl; // << INVALID
    std::cout << "Rowmajor mult rv * m: " << rv * r << std::endl;
    std::cout << "Rowmajor mult m * cv:\n" << r * cv << std::endl;

}


template<typename FP = double>
Tensor<FP, 3> slice(const TensorMapC<FP, 3>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 3> result(dims);
    
    // Update
    tensoriterator<TensorMap<FP, 3>> T(result);
    for (long long int i = 0; i < (long long int)dims[0]; i++) {
        MatrixNN<FP> mat(slice_matrix(tensor, i));

        T(i) = TensorMap<FP, 2>(mat.data(), dims[1], dims[2]);
        //result.chip(i, 0) = TensorMap<FP, 2>(mat.data(), dims[1], dims[2]);
    }

    return result;
}

template<typename FP = double>
Tensor<FP, 4> slice(const TensorMapC<FP, 4>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 4> result(dims);

    // Update
    tensoriterator<TensorMap<FP, 4>> T(result);
    for (long long int j = 0; j < (long long int)dims[0]; j++) {
        auto J = T(j);
        tensoriterator<TensorMap<FP, 3>> subtensor(J);
        for (long long int i = 0; i < (long long int)dims[1]; i++) {
            MatrixNN<FP> mat(slice_matrix(tensor, j, i));

            subtensor[i].tensor() = TensorMap<FP, 2>(mat.data(), dims[2], dims[3]);
            //subtensor[i].ref() = TensorMap<FP, 2>(mat.data(), dims[2], dims[3]);
            //subtensor(i) = TensorMap<FP, 2>(mat.data(), dims[2], dims[3]);
        }
    }

    // Chip implementation (slower)
    
    /*
    for (long long int j = 0; j < (long long int)dims[0]; j++) {
        auto subtensor = result.chip(j, 0);
        for (long long int i = 0; i < (long long int)dims[1]; i++) {
            MatrixNN<FP> mat(slice_matrix(tensor, j, i));

            subtensor.chip(i, 0) = TensorMap<FP, 2>(mat.data(), dims[2], dims[3]);
        }
    }
    */

    return result;
}
template<typename FP = double>
Tensor<FP, 5> slice(const TensorMapC<FP, 5>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 5> result(dims);

    // Update
    tensoriterator<TensorMap<FP, 5>> T(result);
    for (long long int k = 0; k < (long long int)dims[0]; k++) {
        auto K = T(k);
        tensoriterator<TensorMap<FP, 4>> subtensor(K);
        for (long long int j = 0; j < (long long int)dims[1]; j++) {
            auto J = subtensor(j);
            tensoriterator<TensorMap<FP, 3>> subsubtensor(J);
            for (long long int i = 0; i < (long long int)dims[2]; i++) {
                MatrixNN<FP> mat(slice_matrix(tensor, k, j, i));

                subsubtensor(i) = TensorMap<FP, 2>(mat.data(), dims[3], dims[4]);
            }
        }
    }

    return result;
}

/**
* <summary>Utilize recursive chip instead of iterators.</summary>
*/
template<typename FP = double>
Tensor<FP, 5> chip(const TensorMapC<FP, 5>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 5> result(dims);

    // Chip version:
    for (long long int k = 0; k < (long long int)dims[0]; k++) {
        auto subtensor = result.chip(k, 0);
        for (long long int j = 0; j < (long long int)dims[1]; j++) {
            auto subsubtensor = subtensor.chip(j, 0);
            for (long long int i = 0; i < (long long int)dims[2]; i++) {
                MatrixNN<FP> mat(slice_matrix(tensor, k, j, i));

                subsubtensor.chip(i, 0) = TensorMap<FP, 2>(mat.data(), dims[3], dims[4]);
            }
        }
    }
    return result;
}

PYBIND11_MODULE(slice, m) {
    m.doc() = R"pbdoc(
        slice
        -----------------------

        .. currentmodule:: slice

        .. autosummary::
           :toctree: _generate
           
           check_eigen_row_col_mult
           slice
           chip
    )pbdoc";


    m.def("check_eigen_row_col_mult", &check_eigen_row_col_mult, R"pbdoc(
        ....
    )pbdoc");

    m.def("slice", py::overload_cast<const TensorMapC<float, 3>&>(&slice<float>), R"pbdoc(
        slice
    )pbdoc");
    m.def("slice", py::overload_cast<const TensorMapC<double, 3>&>(&slice<double>), R"pbdoc(
        slice
    )pbdoc");

    m.def("slice", py::overload_cast<const TensorMapC<float, 4>&>(&slice<float>), R"pbdoc(
        slice
    )pbdoc");
    m.def("slice", py::overload_cast<const TensorMapC<double, 4>&>(&slice<double>), R"pbdoc(
        slice
    )pbdoc");

    m.def("slice", py::overload_cast<const TensorMapC<float, 5>&>(&slice<float>), R"pbdoc(
        slice
    )pbdoc");
    m.def("slice", py::overload_cast<const TensorMapC<double, 5>&>(&slice<double>), R"pbdoc(
        slice
    )pbdoc");

    m.def("chip", py::overload_cast<const TensorMapC<float, 5>&>(&chip<float>), R"pbdoc(
        chip
    )pbdoc");
    m.def("chip", py::overload_cast<const TensorMapC<double, 5>&>(&chip<double>), R"pbdoc(
        chip
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}