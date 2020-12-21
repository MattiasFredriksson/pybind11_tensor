# pybind11_tensor
Simple header-only library for exposing Eigen::Tensors to python within C++ pybind11 modules. Implementation is similar to the Eigen::Matrix type_casters available in the pybind11 distribution but handles tensor Map and Ref differently.

# Alpha version

The supplied code is a poorly tested alpha version, but it should provide a simple pybind11 extension that can serve as a starting point when using the pybind11 and Eigen libraries. Below is a short description over the intended use and specifics regarding the type converions.

# Implemented type_caster<>

## Eigen::TensorRef

Supports binding of both input and output arguments. The underlying idea for using the TensorRef type when the data pattern for a C++ function call is:

**Python** -> **C++** -> **Copy** -> **Python**

Specifics of how TensorRef arguments are converted when passed between Python and C++ are discussed below. 

### Python -> C++

Attempts to pass the argument without copying the underlying buffer by mapping it using a TensorMap slice. If mapping the buffer is not possible, the data will be copied and converted to match the input argument unless copying is explictly dissallowed. Thus, TensorRef arguments allow both numpy slices and numpy arrays with mismatching scalar types to be passed from python and provides the most general option when copying is not prefered. Since the underlying buffer can be directly accessed from the C++ function, using the input argument type: const Eigen::TensorRef<const Tensor>>& is advisable unless other behaviors are specifically prefered.

### C++ -> Python

Preferable behavior when returning TensorRef arguments to the Python side is to copy the underlying buffer rather then attempt to wrap it in a numpy array object. While not dissalowing to return a direct reference to the buffer unless PYBIND11_ET_STRICT is defined, unless a copy is made, the type_caster does not attempt to ensure the data sticks around once the argument is returned back to python. The primary purpose for allowing TensorRef return arguments is then to support it as a return type similar to returning a Tensor<> but retaining the generallity of the of the TensorRef<> type.

## Eigen::TensorMap

Supports binding of input arguments. The underlying idea for using the TensorMap type when the data pattern for calling a C++ function is:

**Python** -> **C++** -> **Python**

Changes are then applied directly to the exposed data buffer. The use case is then to provide a function which perform inplace calculations or when exposing a buffer that is either enforced to be memory continous or expected to be so.

### Python -> C++

Attempts to pass the argument without copying the underlying buffer by mapping it to a TensorMap. For mapping to be possible however, the numpy array must be memory continous and exactly match the input argument (i.e. scalar type missmatch is not possible). If the argument is not a match, the data is copied and a callback is called when the the C++ function terminates mapping any changes back to the original buffer. To avoid updating the original buffer ensure the input tensor type is: const Eigen::TensorMap<const Tensor<>>.

### C++ -> Python

Under the define PYBIND11_ET_STRICT the intended behavior is to not allow conversion of TensorMap to numpy arrays. Other options are not implemented, intention is to allow return of the original buffer or a Tensor buffer managed on the C++ side.
