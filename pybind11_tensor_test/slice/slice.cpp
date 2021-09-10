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
Tensor<FP, 3> slice_matrix_for_range(const TensorMapC<FP, 3>& tensor) {

	auto dims = tensor.dimensions();
	Tensor<FP, 3> result(dims);

	// Update
	int i = 0;
	for (auto st0 : tensoriterator<TensorMap<FP, 3>>(result)) {
		MatrixNN<FP> mat(slice_matrix(tensor, i++));
		st0 = matrix2tensor(mat);
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
Tensor<FP, 4> slice_matrix_for_range(const TensorMapC<FP, 4>& tensor) {

	auto dims = tensor.dimensions();
	Tensor<FP, 4> result(dims);


	// Update
	int j = 0;
	tensoriterator<TensorMap<FP, 4>> outer(result);
	for (auto t0 : tensoriterator<TensorMap<FP, 4>>(result)) {
		int i = 0;
		if (t0.data() != (*outer[j]).data()) {
			throw std::runtime_error("Missmatching pointers at outer iteration " + std::to_string(j));
		}

		tensoriterator<TensorMap<FP, 3>> inner(t0);
		for (auto t1 : tensoriterator<TensorMap<FP, 3>>(t0)) {
			if (t1.data() != (*inner[i]).data()) {
				throw std::runtime_error("Missmatching pointers at inner iteration " + std::to_string(i));
			}

			MatrixNN<FP> mat(slice_matrix(tensor, j, i));
			t1 = matrix2tensor(mat);
			i++;
		}
		j++;
	}

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
template<typename FP = double>
Tensor<FP, 3> slice_vector(const TensorMapC<FP, 3>& tensor) {

	auto dims = tensor.dimensions();
	Tensor<FP, 3> result(dims);

	// Update
	tensoriterator<TensorMap<FP, 3>> T(result);
	for (long long int j = 0; j < (long long int)dims[0]; j++) {
		tensoriterator<TensorMap<FP, 2>> subtensor(T(j));
		for (long long int i = 0; i < (long long int)dims[1]; i++) {
			Vector<FP> vec(slice_vector(tensor, j, i));
			subtensor(i) = vector2tensor(vec);
		}
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
			tensoriterator<TensorMap<FP, 2>> subsubtensor(subtensor(i));
			for (long long int l = 0; l < (long long int)dims[2]; l++) {
				Vector<FP> vec(slice_vector(tensor, j, i, l));
				subsubtensor(l) = vector2tensor(vec);
			}
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
				tensoriterator<TensorMap<FP, 2>> subx3tensor(subsubtensor(i));
				for (long long int l = 0; l < (long long int)dims[3]; l++) {
					Vector<FP> vec(slice_vector(tensor, k, j, i, l));
					subx3tensor(l) = vector2tensor(vec);
				}
			}
		}
	}

	return result;
}


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


	m.def("slice_vector", py::overload_cast<const TensorMapC<float, 2>&>(&slice_vector<float>), R"pbdoc(
        slice_matrix
    )pbdoc");
	m.def("slice_vector", py::overload_cast<const TensorMapC<double, 2>&>(&slice_vector<double>), R"pbdoc(
        slice_vector
    )pbdoc");


	m.def("slice_vector", py::overload_cast<const TensorMapC<float, 3>&>(&slice_vector<float>), R"pbdoc(
        slice_matrix
    )pbdoc");
	m.def("slice_vector", py::overload_cast<const TensorMapC<double, 3>&>(&slice_vector<double>), R"pbdoc(
        slice_vector
    )pbdoc");


	m.def("slice_vector", py::overload_cast<const TensorMapC<float, 4>&>(&slice_vector<float>), R"pbdoc(
        slice_matrix
    )pbdoc");
	m.def("slice_vector", py::overload_cast<const TensorMapC<double, 4>&>(&slice_vector<double>), R"pbdoc(
        slice_vector
    )pbdoc");


	m.def("slice_vector", py::overload_cast<const TensorMapC<float, 5>&>(&slice_vector<float>), R"pbdoc(
        slice_matrix
    )pbdoc");
	m.def("slice_vector", py::overload_cast<const TensorMapC<double, 5>&>(&slice_vector<double>), R"pbdoc(
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


	m.def("slice_matrix_for_range", py::overload_cast<const TensorMapC<float, 3>&>(&slice_matrix_for_range<float>), R"pbdoc(
        slice_matrix_for_range
    )pbdoc");
	m.def("slice_matrix_for_range", py::overload_cast<const TensorMapC<double, 3>&>(&slice_matrix_for_range<double>), R"pbdoc(
        slice_matrix_for_range
    )pbdoc");

	m.def("slice_matrix_for_range", py::overload_cast<const TensorMapC<float, 4>&>(&slice_matrix_for_range<float>), R"pbdoc(
        slice_matrix_for_range
    )pbdoc");
	m.def("slice_matrix_for_range", py::overload_cast<const TensorMapC<double, 4>&>(&slice_matrix_for_range<double>), R"pbdoc(
        slice_matrix_for_range
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