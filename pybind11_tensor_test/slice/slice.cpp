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

template<typename FP, int rank>
void verify_typedefs() {
	/* TODO:
	* 1) Check all type_traits.h
	* 2) Make type traits consistent std::bool_constant<>
	* 3) Move to own cpp file
	*/

	// Any tensor type
	static_assert(is_any_eigen_tensor<Tensor<FP, rank>> == true);
	static_assert(is_any_eigen_tensor<TensorMap<FP, rank>> == true);
	static_assert(is_any_eigen_tensor<TensorMapC<FP, rank>> == true);
	static_assert(is_any_eigen_tensor<TensorRef<FP, rank>> == true);
	static_assert(is_any_eigen_tensor<TensorRefC<FP, rank>> == true);
	static_assert(is_any_eigen_tensor<const Tensor<FP, rank>> == true);
	static_assert(is_any_eigen_tensor<const TensorMap<FP, rank>> == true);
	static_assert(is_any_eigen_tensor<const TensorMapC<FP, rank>> == true);
	static_assert(is_any_eigen_tensor<const TensorRef<FP, rank>> == true);
	static_assert(is_any_eigen_tensor<const TensorRefC<FP, rank>> == true);
	// Not tensor
	static_assert(is_any_eigen_tensor<int> == false);
	static_assert(is_any_eigen_tensor<float> == false);
	static_assert(is_any_eigen_tensor<MatrixNN<double>> == false);
	static_assert(is_any_eigen_tensor<MatrixMapNN<double>> == false);
	static_assert(is_any_eigen_tensor<MatrixRefNN<double>> == false);

	// Dense type check
	static_assert(is_eigen_tensor<Tensor<FP, rank>> == true);
	static_assert(is_eigen_tensor<const Tensor<FP, rank>> == true);
	// Nested mutable
	static_assert(is_eigen_nested_mutable<TensorRef<FP, rank>>::value == true);
	static_assert(is_eigen_nested_mutable<TensorMap<FP, rank>>::value == true);
	static_assert(is_eigen_nested_mutable<TensorRefC<FP, rank>>::value == false);
	static_assert(is_eigen_nested_mutable<TensorMapC<FP, rank>>::value == false);
	// Mutable (inner or outer)
	static_assert(is_eigen_mutable_tensor<Tensor<FP, rank>>::value == true);
	static_assert(is_eigen_mutable_tensor<TensorMap<FP, rank>>::value == true);
	static_assert(is_eigen_mutable_tensor<TensorRef<FP, rank>>::value == true);
	static_assert(is_eigen_mutable_tensor<TensorMapC<FP, rank>>::value == false);
	static_assert(is_eigen_mutable_tensor<TensorRefC<FP, rank>>::value == false);
	static_assert(is_eigen_mutable_tensor<const Tensor<FP, rank>>::value == false);
	static_assert(is_eigen_mutable_tensor<const TensorMap<FP, rank>>::value == false);
	static_assert(is_eigen_mutable_tensor<const TensorRef<FP, rank>>::value == false);
	static_assert(is_eigen_mutable_tensor<const TensorMapC<FP, rank>>::value == false);
	static_assert(is_eigen_mutable_tensor<const TensorRefC<FP, rank>>::value == false);
	// Row-major
	static_assert(is_eigen_row_major_tensor<const Tensor<FP, rank>>::value == true);
	static_assert(is_eigen_mappable_tensor<const Tensor<FP, rank>>::value == true);
	static_assert(is_eigen_row_major_tensor<Tensor<FP, rank>>::value == true);
	static_assert(is_eigen_mappable_tensor<Tensor<FP, rank>>::value == true);
	static_assert(is_eigen_row_major_tensor<TensorMap<FP, rank>>::value == true);
	static_assert(is_eigen_mappable_tensor<TensorMap<FP, rank>>::value == true);
	static_assert(is_eigen_row_major_tensor<const TensorMap<FP, rank>>::value == true);
	static_assert(is_eigen_mappable_tensor<const TensorMap<FP, rank>>::value == true);
	static_assert(is_eigen_row_major_tensor<TensorMapC<FP, rank>>::value == true);
	static_assert(is_eigen_mappable_tensor<TensorMapC<FP, rank>>::value == true);
	static_assert(is_eigen_row_major_tensor<const TensorMapC<FP, rank>>::value == true);
	static_assert(is_eigen_mappable_tensor<const TensorMapC<FP, rank>>::value == true);
}
template void verify_typedefs<float, 2>();
template void verify_typedefs<double, 2>();
template void verify_typedefs<float, 3>();
template void verify_typedefs<double, 3>();


/* Rank 2 tensors: special case as they are matrices.
*/
template<typename FP>
void verify_slice2_types() {
	Tensor<FP, 2> tensor({ 3,3 });
	TensorMap<FP, 2> tensorm(tensor);
	const Tensor<FP, 2> tensorc({ 3,3 });
	const TensorMapC<FP, 2> tensormc(const_cast<FP*>(tensorc.data()), tensorc.dimensions());

	// Const alloc checks
	//tensormc(0) = 2; // Is invalid
	tensormc.data()[0] = 2; // Should be invalid but isn't

	auto mat_from_tens = slice_matrix(tensor);
	auto mat_from_map = slice_matrix(tensorm);
	auto cmat_from_tens = slice_matrix(tensorc);
	auto cmat_from_map = slice_matrix(tensormc);

	// TODO: Check equality...
}
template void verify_slice2_types<float>();
template void verify_slice2_types<double>();


template<typename FP>
void verify_slice3_types() {
	constexpr int rank = 3;
	Tensor<FP, rank> tensor({ 3, 3, 3 });
	TensorMap<FP, rank> tensorm(tensor);
	const Tensor<FP, rank> tensorc({ 3, 3, 3 });
	const TensorMapC<FP, rank> tensormc(const_cast<FP*>(tensorc.data()), tensorc.dimensions());

	auto mat_from_tens = slice_matrix(tensor, 0);
	auto mat_from_map = slice_matrix(tensorm, 0);
	//auto cmat_from_tens = slice_matrix(tensorc, 0);
	auto cmat_from_map = slice_matrix(tensormc, 0);

	// TODO: Check equality...
}
template void verify_slice3_types<float>();
template void verify_slice3_types<double>();


#pragma region slice_matrix/chip_matrix

template<typename FP = double>
Tensor<FP, 3> slice_matrix(const TensorMapC<FP, 3>& tensor) {

	auto dims = tensor.dimensions();
	Tensor<FP, 3> result(dims);

	// Update
	tensoriterator T(result);
	for (long long int i = 0; i < (long long int)dims[0]; i++) {
		MatrixNN<FP> mat(slice_matrix(tensor, i));

		T(i) = TensorMap<FP, 2>(mat.data(), dims[1], dims[2]);
		//result.chip(i, 0) = TensorMap<FP, 2>(mat.data(), dims[1], dims[2]);
	}

	return result;
}

template<typename FP = double>
Tensor<FP, 3> slice_matrix_for_range(const TensorMapC<FP, 3>& tensor) {
	Tensor<FP, 3> result(tensor.dimensions());

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
	tensoriterator T(result);
	for (long long int j = 0; j < (long long int)dims[0]; j++) {
		tensoriterator subtensor(T(j));
		for (long long int i = 0; i < (long long int)dims[1]; i++) {
			MatrixNN<FP> mat(slice_matrix(tensor, j, i));

			subtensor[i] = TensorMap<FP, 2>(mat.data(), dims[2], dims[3]);
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
	tensoriterator outer(result);
	for (auto t0 : tensoriterator(result)) {
		int i = 0;
		if (t0.data() != (*outer[j]).data()) {
			throw std::runtime_error("Missmatching pointers at outer iteration " + std::to_string(j));
		}

		tensoriterator inner{ t0.iter() };
		for (auto t1 : t0.iter()) {
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
	tensoriterator T(result);
	for (long long int k = 0; k < (long long int)dims[0]; k++) {
		tensoriterator subtensor(T(k));
		for (long long int j = 0; j < (long long int)dims[1]; j++) {
			tensoriterator subsubtensor(subtensor(j));
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
	tensoriterator T(result);
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
	tensoriterator T(result);
	for (long long int j = 0; j < (long long int)dims[0]; j++) {
		tensoriterator subtensor(T(j));
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
	tensoriterator T(result);
	for (long long int j = 0; j < (long long int)dims[0]; j++) {
		tensoriterator subtensor(T(j));
		for (long long int i = 0; i < (long long int)dims[1]; i++) {
			tensoriterator subsubtensor(subtensor(i));
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
	tensoriterator T(result);
	for (long long int k = 0; k < (long long int)dims[0]; k++) {
		tensoriterator subtensor(T(k));
		for (long long int j = 0; j < (long long int)dims[1]; j++) {
			tensoriterator subsubtensor(subtensor(j));
			for (long long int i = 0; i < (long long int)dims[2]; i++) {
				tensoriterator subx3tensor(subsubtensor(i));
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

	verify_typedefs<double, 2>();
	verify_typedefs<double, 3>();
	verify_slice2_types<double>();
	verify_slice3_types<double>();

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