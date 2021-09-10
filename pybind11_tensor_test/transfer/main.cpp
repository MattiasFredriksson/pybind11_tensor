// DTW_python.cpp : Defines the entry point for the application.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "pybind11_eigen_tensor.h"
#include "eigen_tensor_ext_pyb.h"

namespace py = pybind11;
namespace pyd = pybind11::detail;


template<typename FP>
void print_const(const Eigen::TensorRef<const Eigen::Tensor<FP, 3, 1>> tensor) {
	std::stringstream st;
	st << tensor;
	py::print(st.str());
}
template<typename FP>
void print_nonconst(Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> tensor) {
	std::stringstream st;
	st << tensor;
	py::print(st.str());
}

template<typename FP>
void check_eigen_move_operator() {
	Eigen::Tensor<FP, 3, 1> tensor(4, 4, 4);
	tensor.setRandom();

	FP* ptr = tensor.data();
	Eigen::Tensor<FP, 3, 1> t(std::move(tensor));
	if (t.data() == ptr)
		py::print("Tensor moved.");
	else
		py::print("Tensor copied.");


	Eigen::Matrix<FP, 12, 12> a;
	a.setRandom();
	ptr = a.data();

	Eigen::Matrix<FP, 12, 12> b(std::move(a));
	if (b.data() == ptr)
		py::print("Matrix moved.");
	else
		py::print("Matrix copied.");
}

template<typename FP>
Eigen::Tensor<FP, 3, 1> add_self(Eigen::Tensor<FP, 3, 1> tensor) {
	return tensor + tensor;
}

template<typename FP>
Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> add_self_ref(Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> tensor) {
	return tensor + tensor;
}

template<typename FP>
Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> add_self_ref_ref(Eigen::TensorRef<Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>>> tensor) {
	return tensor + tensor;
}

template<typename FP = double>
void tensor_dummy_ops(int N) {
	static Eigen::Tensor<double, 3, 1> dummy;
	for (int i = 0; i < N; i++) {
		Eigen::Tensor<double, 3, 1> tensor(10, 10, 10);
		tensor.setRandom();
		dummy = tensor;
	}
	std::stringstream os;
	os << dummy;
	os.clear();
	std::cout << os.str()[0] << '\r' << " \r";
}

template<typename FP>
Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> add_self_ref_undef(Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> tensor) {
	// Returning an evaluated tensor expression as rvalue yields undefined behavior

	Eigen::Tensor<FP, 3, 1> tmp_ten = tensor;
	Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> tmp_ref = tmp_ten + tmp_ten;
	tmp_ten = tensor + tensor;
	Eigen::TensorRef<Eigen::Tensor<FP, 3, 1>> valid_ref = tensor + tensor;

	//return tmp_ten;       // Undefined behavior
	return tmp_ref;         // Undefined behavior
	//return valid_ref;     // Valid. Consist of an expression over the input argument which persists until after variable python cast is complete.
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

/* Adds the tensor to itself 10 times.
*/
template<typename FP>
std::vector<Eigen::Tensor<FP, 3, 1>*> add_self_repeat10(Eigen::TensorMap<Eigen::Tensor<FP, 3, 1>>& tensor) {
	std::vector<Eigen::Tensor<FP, 3, 1>*> arr(10);
	for (int i = 0; i < 10; i++)
		arr[i] = new Eigen::Tensor<FP, 3, 1>(tensor + tensor);
	return arr;
}


template<typename FP = double>
using MatrixNN = Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/* Adds the matrix to itself 10 times.
*/
template<typename FP>
std::vector<MatrixNN<FP>*> add_self_repeat10(MatrixNN<FP> matrix) {
	std::vector<MatrixNN<FP>*> arr(10);
	for (int i = 0; i < 10; i++)
		arr[i] = new MatrixNN<FP>(matrix + matrix);
	return arr;
}
/* Adds each matrix to itself.
*/
template<typename FP>
std::vector<MatrixNN<FP>*> add_self_vector(const std::vector<MatrixNN<FP>>& matrices) {
	std::vector<MatrixNN<FP>*> arr(matrices.size());
	for (int i = 0; i < matrices.size(); i++)
		arr[i] = new MatrixNN<FP>(matrices[i] + matrices[i]);
	return arr;
}
/* Adds each tensor to itself.
*/
template<typename FP>
std::vector<Eigen::Tensor<FP, 3, 1>*> add_self_vector(const std::vector<Eigen::TensorMap<Eigen::Tensor<FP, 3, 1>>>& tensors) {
	std::vector<Eigen::Tensor<FP, 3, 1>*> arr(tensors.size());
	for (int i = 0; i < tensors.size(); i++)
		arr[i] = new Eigen::Tensor<FP, 3, 1>(tensors[i] + tensors[i]);
	return arr;
}

PYBIND11_MODULE(transfer, m) {
	m.doc() = R"pbdoc(
        Python module to implement test functions for pybind11_eigen_tensor.h
        -----------------------

        .. currentmodule:: pybind11_tensor_test

        .. autosummary::
           :toctree: _generate

    )pbdoc";

	m.def("check_eigen_move_operator", &check_eigen_move_operator<double>, R"pbdoc(
        Check if eigen support move constructors.
    )pbdoc");

	tensorial::pyb_define_tensors_rank<2>(m);


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
        Add self to self using a TensorRef<> argument.
    )pbdoc");
	m.def("add_self_ref_ref", &add_self_ref_ref<double>, R"pbdoc(
        Add self to self using a TensorRef<TensorRef<>> argument.
    )pbdoc");
	m.def("add_self_ref_undef", &add_self_ref_undef<double>, R"pbdoc(
        Add self to self using a TensorRef return argument (undefined behavior). 
    )pbdoc");

	m.def("add_self_repeat10", py::overload_cast<Eigen::TensorMap<Eigen::Tensor<double, 3, 1>>&>(&add_self_repeat10<double>),
		py::return_value_policy::take_ownership, R"pbdoc(
        Add self to self 10 times.
    )pbdoc");
	m.def("add_self_repeat10", py::overload_cast<MatrixNN<double>>(&add_self_repeat10<double>),
		py::return_value_policy::take_ownership, R"pbdoc(
        Add self to self 10 times.
    )pbdoc");
	m.def("add_self_vector", py::overload_cast<const std::vector<Eigen::TensorMap<Eigen::Tensor<double, 3, 1>>>&>(&add_self_vector<double>),
		py::return_value_policy::take_ownership, R"pbdoc(
        Adds each tensor to itself.
    )pbdoc");
	m.def("add_self_vector", py::overload_cast<const std::vector<MatrixNN<double>>&>(&add_self_vector<double>),
		py::return_value_policy::take_ownership, R"pbdoc(
        Adds each matrix to itself.
    )pbdoc");


#ifdef VERSION_INFO
	m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
	m.attr("__version__") = "test";
#endif
}

