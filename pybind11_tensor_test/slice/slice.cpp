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

#pragma region slice_matrix/chip_matrix

template<typename FP = double>
Tensor<FP, 3> slice_matrix(const TensorMapC<FP, 3>& tensor) {

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
Tensor<FP, 4> slice_matrix(const TensorMapC<FP, 4>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 4> result(dims);

    // Update
    tensoriterator<TensorMap<FP, 4>> T(result);
    for (long long int j = 0; j < (long long int)dims[0]; j++) {
        tensoriterator<TensorMap<FP, 3>> subtensor(T(j));
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
Tensor<FP, 5> slice_matrix(const TensorMapC<FP, 5>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 5> result(dims);

    // Update
    tensoriterator<TensorMap<FP, 5>> T(result);
    for (long long int k = 0; k < (long long int)dims[0]; k++) {
        tensoriterator<TensorMap<FP, 4>> subtensor(T(k));
        for (long long int j = 0; j < (long long int)dims[1]; j++) {
            tensoriterator<TensorMap<FP, 3>> subsubtensor(subtensor(j));
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
Tensor<FP, 5> chip_matrix(const TensorMapC<FP, 5>& tensor) {

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

#pragma endregion


#pragma region slice_vector

template<typename FP = double>
Tensor<FP, 2> slice_vector(const TensorMapC<FP, 2>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 2> result(dims);

    // Update
    tensoriterator<TensorMap<FP, 2>> T(result);
    for (long long int i = 0; i < (long long int)dims[0]; i++) {
        Vector<FP> vec(slice_vector(tensor, i));
        T(i) = vector2tensor(vec);
    }

    return result;
}
/*
template<typename FP = double>
Tensor<FP, 3> slice_vector(const TensorMapC<FP, 3>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 3> result(dims);

    // Update
    tensoriterator<TensorMap<FP, 3>> T(result);
    for (long long int i = 0; i < (long long int)dims[0]; i++) {
        Vector<FP> mat(slice_matrix(tensor, i));

        T(i) = TensorMap<FP, 2>(mat.data(), dims[1], dims[2]);
    }

    return result;
}
template<typename FP = double>
Tensor<FP, 4> slice_vector(const TensorMapC<FP, 4>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 4> result(dims);

    // Update
    tensoriterator<TensorMap<FP, 4>> T(result);
    for (long long int j = 0; j < (long long int)dims[0]; j++) {
        tensoriterator<TensorMap<FP, 3>> subtensor(T(j));
        for (long long int i = 0; i < (long long int)dims[1]; i++) {
            MatrixNN<FP> mat(slice_matrix(tensor, j, i));

            subtensor[i].tensor() = TensorMap<FP, 2>(mat.data(), dims[2], dims[3]);
            //subtensor[i].ref() = TensorMap<FP, 2>(mat.data(), dims[2], dims[3]);
            //subtensor(i) = TensorMap<FP, 2>(mat.data(), dims[2], dims[3]);
        }
    }
    return result;
}
template<typename FP = double>
Tensor<FP, 5> slice_vector(const TensorMapC<FP, 5>& tensor) {

    auto dims = tensor.dimensions();
    Tensor<FP, 5> result(dims);

    // Update
    tensoriterator<TensorMap<FP, 5>> T(result);
    for (long long int k = 0; k < (long long int)dims[0]; k++) {
        tensoriterator<TensorMap<FP, 4>> subtensor(T(k));
        for (long long int j = 0; j < (long long int)dims[1]; j++) {
            tensoriterator<TensorMap<FP, 3>> subsubtensor(subtensor(j));
            for (long long int i = 0; i < (long long int)dims[2]; i++) {
                MatrixNN<FP> mat(slice_matrix(tensor, k, j, i));

                subsubtensor(i) = TensorMap<FP, 2>(mat.data(), dims[3], dims[4]);
            }
        }
    }

    return result;
}
*/

#pragma endregion

PYBIND11_MODULE(slice, m) {
    m.doc() = R"pbdoc(
        slice
        -----------------------

        .. currentmodule:: slice

        .. autosummary::
           :toctree: _generate
           
           check_eigen_row_col_mult
           slice_matrix
           slice_vector
           chip
    )pbdoc";


    m.def("check_eigen_row_col_mult", &check_eigen_row_col_mult, R"pbdoc(
        ....
    )pbdoc");


    m.def("slice_vector", py::overload_cast<const TensorMapC<float, 2>&>(&slice_vector<float>), R"pbdoc(
        slice_matrix
    )pbdoc");
    m.def("slice_vector", py::overload_cast<const TensorMapC<double, 2>&>(&slice_vector<double>), R"pbdoc(
        slice_vector
    )pbdoc");
    

    m.def("slice_matrix", py::overload_cast<const TensorMapC<float, 3>&>(&slice_matrix<float>), R"pbdoc(
        slice_matrix
    )pbdoc");
    m.def("slice_matrix", py::overload_cast<const TensorMapC<double, 3>&>(&slice_matrix<double>), R"pbdoc(
        slice_matrix
    )pbdoc");


    m.def("slice_matrix", py::overload_cast<const TensorMapC<float, 4>&>(&slice_matrix<float>), R"pbdoc(
        slice_matrix
    )pbdoc");
    m.def("slice_matrix", py::overload_cast<const TensorMapC<double, 4>&>(&slice_matrix<double>), R"pbdoc(
        slice_matrix
    )pbdoc");

    m.def("slice_matrix", py::overload_cast<const TensorMapC<float, 5>&>(&slice_matrix<float>), R"pbdoc(
        slice_matrix
    )pbdoc");
    m.def("slice_matrix", py::overload_cast<const TensorMapC<double, 5>&>(&slice_matrix<double>), R"pbdoc(
        slice_matrix
    )pbdoc");

    m.def("chip_matrix", py::overload_cast<const TensorMapC<float, 5>&>(&chip_matrix<float>), R"pbdoc(
        chip_matrix
    )pbdoc");
    m.def("chip_matrix", py::overload_cast<const TensorMapC<double, 5>&>(&chip_matrix<double>), R"pbdoc(
        chip_matrix
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}