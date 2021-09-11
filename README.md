# pybind11_tensor
Simple header-only library for exposing Eigen::Tensors to python within C++ pybind11 modules. Implementation is similar to the Eigen::Matrix type_casters available in the pybind11 distribution but handles tensor Map and Ref differently. Relevant type_caster implementations are available in [include/pybind11_eigen_tensor.h](../include/pybind11_eigen_tensor.h). The alpha version support the (latest) stable eigen release 3.3.9.

# Alpha version

Code supplied is WIP and tests are limited, but it should provide a simple pybind11 extension that can serve as a starting point when using the pybind11 and Eigen libraries. Following sections contain a a short description over how provided type_casters behave when casting between eigen and python types.

# Implemented type_caster<>

## Eigen::Tensor

Intended behavior is to be identical to the dense matrix type_caster. Implementation is based on the dense matrix type_caster distributed with the pybind11 release version 2.6.1. Passing an Eigen::Tensor to C++ side ensure the data buffer is copied, passing tensor arguments to python is flexible and depend on the return_value_policy argument specified for the function binding.
  
Passing dense tensor types back to python without copying requires a pointer argument with return_value_policy::take_ownership to avoid implicit copies. Options such as rvalues with move is only valid once move operators is properly supported in the eigen library. Behavior can be assumed to follow the pattern:

**Python** -> **Copy** -> **C++** -> **(Copy)** -> **Python**

Passing dense tensor arguments provide clear separations between Python and C++ but the behavior is not always preferable. To avoid copying data use the TensorMap<> or TensorRef<> types instead.

## Eigen::TensorMap

Supports binding input arguments passed to C++ side for native types. Intended use for a TensorMap is to pass a mutable view over a dense Python buffer to a C++ function:

**Python** -> **C++** -> **Python**

Since TensorMap<> expects either a C or Fortran contiguous (dense) buffer of a specific native type, it is not always possible to provide direct access to the underlying buffer. Therefor TensorMaps should only be used to perform inplace calculations, or to expose buffers that is of a known native type and expected to be memory continous. In all other cases it should be preferable to return a dense Eigen::Tensor to make functions safer and behave consistently. 

### Python -> C++

Attempts to pass the argument without copying by wrapping the data pointer using a TensorMap. If the argument passed to the function is not an exact match, data is copied and a callback is created to map changes back to the original buffer when returning to Python. To prevent the callback from updating the original buffer ensure the function argument is readonly using the following type: const Eigen::TensorMap<const Tensor<...>>.

### C++ -> Python

Default behavior is to return a TensorMap<> view over the buffer with no regard to its lifetime, return policy reference_internal is used. Under the define PYBIND11_ET_STRICT conversion of TensorMap to numpy arrays is disallowed.

## Eigen::TensorRef

Supports binding of both input and output arguments. The underlying idea for using the TensorRef type when the data pattern for a C++ function call is:

**Python** -> **C++** -> **Copy** -> **Python**

Specifics of how TensorRef arguments are converted when passed between Python and C++ are discussed below. 

### Python -> C++

Attempts to pass the argument without copying the underlying buffer by mapping it using a TensorMap slice. If mapping the buffer is not possible, the data will be copied and converted to match the input argument unless copying is explictly dissallowed. Thus, TensorRef arguments allow both numpy slices and numpy arrays with mismatching scalar types to be passed from python and provides the most general option when copying is not prefered. Since the underlying buffer can be directly accessed from the C++ function, using the input argument type: const Eigen::TensorRef<const Tensor<>>>& is advisable unless other behaviors are specifically prefered.

### C++ -> Python

Preferable behavior when returning TensorRef arguments to the Python side is to copy the underlying buffer rather then attempt to wrap it in a numpy array object. While not dissalowing to return a direct reference to the buffer unless PYBIND11_ET_STRICT is defined, the type_caster does not attempt to ensure the data sticks around once the argument is returned back to python. 

## Eigen::Quaternion

Type caster for Eigen::Quaternion is also provided by pybind11_eigen_tensor.h.
