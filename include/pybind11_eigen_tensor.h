/*
*/

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

#pragma region Helper functions

template <int rank>
std::ostream& operator <<(std::ostream& stream, const Eigen::array<EigenIndex, rank>& arr) {
    for (int i = 0; i < rank; i++)
        std::cout << arr[i] << ", ";
    return stream;
}
// Divide Eigen::array
template <typename Index, int rank>
Eigen::array<Index, rank> operator/(const Eigen::array<Index, rank>& numerator, const Eigen::array<Index, rank>& denom){
    Eigen::array<Index, rank> res;
    for (int i = 0; i < rank; i++)
        res[i] = numerator[i] / denom[i];
    return res;
}
// Mult Eigen::array
template <typename Index, int rank>
Eigen::array<Index, rank> operator*(const Eigen::array<Index, rank>& numerator, const Eigen::array<Index, rank>& denom) {
    Eigen::array<Index, rank> res;
    for (int i = 0; i < rank; i++)
        res[i] = numerator[i] * denom[i];
    return res;
}


#pragma endregion

template<typename T>
struct void_ { typedef void type; };

// Determine nested PlainObjectType from an EigenTensor type 
template <typename Type> struct eigen_nested_type { using type = Type; };
template <typename PlainObjectType>
struct eigen_nested_type<Eigen::TensorMap<PlainObjectType>> { using type = PlainObjectType; };
template <typename PlainObjectType>
struct eigen_nested_type<Eigen::TensorRef<PlainObjectType>> { using type = PlainObjectType; };

// Expression verifying if T is a valid Eigen::Tensor... by checking if T::Base is a valid expression, 
//  will generate compiler garbage if the expression is valid but not a valid tensor type.
template <typename T>
using EigenTensorIdentifier = typename void_<typename T::Base>::type;

// Check if type T inherits from TensorBase<T>
template <typename T, typename = void>
struct check_if_tensor_base { static constexpr bool value = false;  };
template <typename T> // Specialization verifying if the EigenTensorIdentifier expression is valid
struct check_if_tensor_base<T, EigenTensorIdentifier<T>>{ static constexpr bool value = std::is_base_of<Eigen::TensorBase<T>, T>::value;  };

// Check if type T is an Eigen::Tensor<>
template <typename T, typename = void>
struct check_if_eigen_tensor { static constexpr bool value = false; };
template <typename T>   // Specialization verifying if the EigenTensorIdentifier expression is valid
struct check_if_eigen_tensor<T, EigenTensorIdentifier<T>> { static constexpr bool value = std::is_same<typename eigen_nested_type<T>::type, T>::value; };

// True if type T is an Eigen::Tensor (accepts any Type)
template <typename T>
constexpr bool is_eigen_tensor = check_if_eigen_tensor<T>::value;
// True if type T inherits from TensorBase (accepts any Type)
template <typename T>
constexpr bool is_any_eigen_tensor = check_if_tensor_base<T>::value;



template <typename T> using is_eigen_mutable_tensor = all_of<
    std::negation<std::is_const<T>>, 
    std::negation<std::is_const<typename eigen_nested_type<T>::type>>>;
template <typename T> using is_eigen_row_major_tensor = std::bool_constant<T::Layout == 1>;

// Determine if T == TensorRef<Tensor>
template <typename T> using is_eigen_tensor_ref = any_of<
    std::is_same<Eigen::TensorRef<Eigen::Tensor<typename T::Scalar, T::NumIndices, T::Layout, typename T::Index>>, typename std::remove_const<typename std::remove_reference<T>::type>::type>,
    std::is_same<Eigen::TensorRef<const Eigen::Tensor<typename T::Scalar, T::NumIndices, T::Layout, typename T::Index>>, typename std::remove_const<typename std::remove_reference<T>::type>::type>>;
// Determine if T == TensorMap<Tensor> (Does not handle MakePointer)
template <typename T> using is_eigen_tensor_map = any_of <
    std::is_same<Eigen::TensorMap<Eigen::Tensor<typename T::Scalar, T::NumIndices, T::Layout, typename T::Index>, T::Options>, typename std::remove_const<typename std::remove_reference<T>::type>::type>,
    std::is_same<Eigen::TensorMap<const Eigen::Tensor<typename T::Scalar, T::NumIndices, T::Layout, typename T::Index>, T::Options>, typename std::remove_const<typename std::remove_reference<T>::type>::type>>;



#pragma region Conformable

template <int rank, bool row_major> struct TensorShape {
public:
    using EigenArray = Eigen::array<EigenIndex, rank>;
    static constexpr EigenIndex
        dims = rank;

    // Members
    EigenArray
        shape,
        stride;

    TensorShape()
        : shape(), stride() {}
    TensorShape(EigenArray shape)
        : shape(shape), stride() {
        for (int i = 0; i < rank; i++)
            stride[i] = 1;
    }
    TensorShape(EigenArray shape, EigenArray stride)
        : shape(shape), stride(stride) {}
    TensorShape(const TensorShape& shape)
        : shape(shape.shape), stride(shape.stride) {}

    // Number of elements associated with the strided shape.
    size_t size() const {
        size_t total = 1;
        for (int i = 0; i < rank; i++)
            total *= (shape[i] / stride[i]);
        return total;
    }

    template <bool isrow = row_major, enable_if_t<std::bool_constant<isrow>::value, int> = 1>
    EigenArray cum_stride() const {
        Eigen::array<EigenIndex, rank> cum_stride;
        cum_stride[rank - 1] = 1;
        for (int i = rank - 2; 0 <= i; i--)
            cum_stride[i] = cum_stride[i + 1] * shape[i + 1];
        return cum_stride * stride;
    }
    template <bool isrow = row_major, enable_if_t<std::bool_constant<!isrow>::value, int> = 0>
    EigenArray cum_stride() const {
        Eigen::array<EigenIndex, rank> cum_stride;
        cum_stride[0] = 1;
        for (int i = 1; i < rank; i++)
            cum_stride[i] = cum_stride[i - 1] * shape[i - 1];
        return cum_stride * stride;
    }

    // Shape after dividing the rank dim with the stride along the dimension (i.e. shape / stride).
    EigenArray shape_reduced() const {
        return shape / stride;
    }
    // Get the reduced shape representation (i.e. shape / stride).
    TensorShape reduced() const {
        return TensorShape(shape_reduced());
    }

    // True if the shape represent a continous block in memory (must be non-cumul shape)
    bool mem_continous() const  {
        EigenIndex sum = 0;
        for (int i = 0; i < rank; i++)
            sum += stride[i];
        return sum == rank;
    }

    // Convert byte stride to item stride.
    static TensorShape<rank, row_major> item_shape(const array& pyshape) {

        ssize_t elem_size = pyshape.itemsize();

        // Determine shape and stride from numpy array description
        Eigen::array<EigenIndex, rank>
            shape,
            stride;
        for (long long int i = rank - 1; i >= 0; i--) {
            shape[i] = pyshape.shape()[i];
            stride[i] = pyshape.strides()[i] / elem_size;
        }

        return TensorShape<rank, row_major>(shape, stride);
    }

    // Reconstructs the full shape (except dimension of the last rank) from the byte strided shape.
    template<bool row_major, typename DataT, typename ShapeT>
    static void reconstruct_byte_shape(const DataT* view_shape, const DataT* byte_stride, ShapeT* shape, ShapeT* stride, size_t ndim, size_t item_byte_size) {
        constexpr int incr = row_major ? -1 : 1;
        constexpr int limit = row_major ? -1 : (int)ndim;
        int offset = row_major ? (int)ndim - 1 : 0;

        // Base case
        int cuml_shape = 1;
        int S = (int)(*(byte_stride + offset) / item_byte_size);
        *(stride + offset) = S;

        // Unravel the shape
        for (int i = offset + incr; i != limit; i += incr) {
            int i_p = i - incr;
            int B = (int)(*(byte_stride + i) / (item_byte_size * cuml_shape));
            int y_p = S * (int)*(view_shape + i_p);
            int k = B % y_p;

            if (k == 0 && S != 1)       // if S == 1 => D = *(view_shape + i_p)
                S = y_p;
            else {
                if (k * y_p != B)
                    k = k * y_p;
                S = (B - k) / y_p;
            }
            int D = B / S;
            *(shape + i_p) = D;
            *(stride + i) = S;
            cuml_shape *= D;
        }
        // More information is required to determine the true shape in the last rank dimension. 
        // Set the rank dimension as the minimum size required to satisfy the determined stride.
        *(shape + limit - incr) = S * *(view_shape + limit - incr);
        return;
    }

    // Verify if order matches
    template <bool isrow = row_major, enable_if_t<std::bool_constant<isrow>::value, int> = 1>
    static bool order_match(const array& pyshape) {
        // Check row major
        return pyshape.strides()[0] > pyshape.strides()[pyshape.ndim() - 1];
    }
    template <bool isrow = row_major, enable_if_t<std::bool_constant<!isrow>::value, int> = 0>
    static bool order_match(const array& pyshape) {
        // Check col major
        return pyshape.strides()[0] < pyshape.strides()[pyshape.ndim() - 1];
    }
    static bool rank_match(const array& pyshape) {
        return pyshape.ndim() > 0 && pyshape.ndim() == rank;
    }
    static bool conform(const array& pyshape) {
        return rank_match(pyshape) && order_match<row_major>(pyshape);
    }

    // Convert the numpy shape to an eigen compatible shape.
    template <bool isrow = row_major>
    static TensorShape<rank, row_major> numpy_to_eigen_shape(const array& pyshape) {

        // Determine shape and stride from numpy array description
        Eigen::array<EigenIndex, rank>
            shape,
            stride;

        reconstruct_byte_shape<isrow>(pyshape.shape(), pyshape.strides(), shape.data(), stride.data(), rank, pyshape.itemsize());

        return TensorShape<rank, row_major>(shape, stride);
    }
};
template <int rank, bool row_major>
std::ostream& operator<<(std::ostream& stream, const TensorShape<rank, row_major>& shape) {
    stream << shape.shape << std::endl << shape.stride;
    return stream;
}
template <int rank, bool EigenRowMajor> struct ShapeConformable 
    : TensorShape<rank, EigenRowMajor> {

    // Members
    bool conformable = false;

    ShapeConformable(bool isconformable = false) 
        : TensorShape(), conformable(isconformable) {}
    ShapeConformable(bool conformable, TensorShape shape)
        : TensorShape(shape), conformable(conformable) {}

    operator bool() const { return conformable; }
};



// Helper struct for extracting information from an Eigen tensor type
template <typename Type_> 
struct EigenTensorProps { 
    using Type = Type_;
    using Scalar = typename Type::Scalar;
    //using IndexType = typename Type::IndexType;
    static constexpr EigenIndex
        rank = Type::NumDimensions;
    static constexpr bool
        row_major = Type::Layout;
    static constexpr bool is_ref = is_eigen_tensor_ref<Type>::value;

    // Takes an input array and determines whether we can make it fit into the Eigen type.
    static ShapeConformable<rank, row_major> conformable(const array & arr, bool mem_continous) {
        using TShape = TensorShape<rank, row_major>;
        // Verify rank and memory order match
        if (!TShape::conform(arr))
            return { false };

        TShape shape = TShape::numpy_to_eigen_shape(arr);
        return ShapeConformable<rank, row_major>(!mem_continous || shape.mem_continous(), shape);
    }

    static constexpr bool show_writeable = is_eigen_mutable_tensor<Type>::value;
    static constexpr bool show_c_contiguous = row_major;
    static constexpr bool show_f_contiguous = !row_major;

    static constexpr auto descriptor =
        _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name +
        _("<") + _<(size_t)rank>() + _(">") +
        // Express array flags required by the input
        _<show_writeable>(", flags.writeable", "") +
        _<show_c_contiguous>(", flags.c_contiguous", "") +
        _<show_f_contiguous>(", flags.f_contiguous", "") +
        _("]");
};

#pragma endregion


template <typename props, typename Type>
handle eigen_encapsulate_tensor(Type* src);
// Casts an Eigen tensor type to numpy array.  If given a base, the numpy array references the src data,
// otherwise it'll make a copy.  writeable lets you turn off the writeable flag for the array.
template <typename props> handle eigen_tensor_array_cast(typename props::Type const& src, handle base = handle(), bool writeable = true) {
    constexpr ssize_t elem_size = sizeof(typename props::Scalar);

    array arr = array(src.dimensions(), src.data(), base);

    if (!writeable)
        array_proxy(arr.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;

    return arr.release();
}
// Takes an lvalue ref to some Eigen type and a (python) base object, creating a numpy array that
// reference the Eigen object's data with `base` as the python-registered base class (if omitted,
// the base will be set to None, and lifetime management is up to the caller).  The numpy array is
// non-writeable if the given type is const.
template <typename props, typename Type>
handle eigen_ref_array_tensor(Type& src, handle parent = none()) {
    // none here is to get past array's should-we-copy detection, which currently always
    // copies when there is no base.  Setting the base to None should be harmless.
    return eigen_tensor_array_cast<props>(src, parent, !std::is_const<Type>::value);
}

// Takes a pointer to some dense, plain Eigen type, builds a capsule around it, then returns a numpy
// array that references the encapsulated data with a python-side reference to the capsule to tie
// its destruction to that of any dependent python objects.  Const-ness is determined by whether or
// not the Type of the pointer given is const.
template <typename props, typename Type>
handle eigen_encapsulate_tensor(Type* src) {
    capsule base(src, [](void* o) { delete static_cast<Type*>(o); });
    return eigen_ref_array_tensor<props>(*src, base);
}

template <int ndim, typename ScalarR, typename ScalarW, typename Shape>
static void copy_buf_row(ScalarR* rbuf, ScalarW* wbuf, const Shape& cshape, const Shape& rcstride, const Shape& wcstride, size_t dim) {
    if (dim == ndim - 1) {
        for (EigenIndex i = 0; i < cshape[dim]; i++) {
            *wbuf = *rbuf;
            wbuf += wcstride[dim];
            rbuf += rcstride[dim];
        }
    }
    else {
        for (EigenIndex i = 0; i < cshape[dim]; i++) {
            copy_buf_row<ndim>(rbuf, wbuf, cshape, rcstride, wcstride, dim + 1);
            wbuf += wcstride[dim];
            rbuf += rcstride[dim];
        }
    }
}

// TensorMap<...> type_caster.
template <typename TensorType>
struct type_caster<
    Eigen::TensorMap<TensorType>,
    std::enable_if_t<is_eigen_tensor_map<Eigen::TensorMap<TensorType>>::value>>
{
private:
    using Type = Eigen::TensorMap<TensorType>;
    using RefType = Eigen::TensorRef<TensorType>;
    using props = EigenTensorProps<Type>;
    using Scalar = typename props::Scalar;
    using TShape = TensorShape<props::rank, props::row_major>;

    static constexpr bool is_writeable = is_eigen_mutable_tensor<TensorType>::value;
    static constexpr int array_flags =
        (props::row_major ? array::c_style : array::f_style) |              // Memory continous (row/col) 
        (is_writeable ? npy_api::constants::NPY_ARRAY_WRITEABLE_ : false);  // Write flag

    using Array = array_t<Scalar, array_flags>;

    // Callback using the destructor to update any changes done to the copy
    // back to the original data source.
    struct UpdateCallback {
        bool update;
        array src_ref;      // Write buffer (original data source, needs to be updated).
        Array src_copy;     // Read buffer  (presumably changed during function call).

        UpdateCallback()
            : update(false), src_ref(), src_copy() {}

        ~UpdateCallback() {
            if (update)
                return;

            // Update source buffer
            py::buffer_info src = src_ref.request(true);
            if (!src.ptr)
                pybind11::print("UpdateCallback failed to acquire write buffer when updating the source buffer.");

            TShape rshape = TShape::item_shape(src_copy);
            TShape wshape = TShape::item_shape(src_ref);
            const Scalar* rbuf = reinterpret_cast<const Scalar*>(src_copy.data());
            Scalar* wbuf = reinterpret_cast<Scalar*>(src_ref.mutable_data());

            copy_buf_row<props::rank>(rbuf, wbuf, rshape.shape, rshape.stride, wshape.stride, 0);
        }
    };

    // Callback updating the input data src after python call ends.
    UpdateCallback callback;
    // TensorMap mapping the input data either directly or the copy.
    std::unique_ptr<Type> map;

public:
    bool load(handle src, bool convert) {
        //Check array_t argument matches the scalar type.
        bool array_type_mismatch = !isinstance<Array>(src);

        // No support for row major (c_style) tensors
        if (check_flags(src.ptr(), array::f_style))
            return false;
        // Verify we are allowed to write to the source buffer
        if (!is_writeable && !check_flags(src.ptr(), npy_api::constants::NPY_ARRAY_WRITEABLE_))
            return false;

        // c_style & f_style flags only represent continous row/col major arrays.
        // If neither flag is set we assume the array is a row major but non-continous (strided).
        bool has_stride = !check_flags(src.ptr(), array::c_style);

        ShapeConformable<props::rank, props::row_major> shape_conform;
        if (!array_type_mismatch && !has_stride) {
            // We don't need a converting copy, but we need to ensure it conforms.
            auto aref = reinterpret_borrow<Array>(src);

            if (aref) {
                shape_conform = props::conformable(aref, true);
                if (!shape_conform)
                    return false; // Incompatible
                map.reset(new Type(const_cast<props::Scalar*>(data(aref)), shape_conform.shape));
            }
            else
                return false; // Incompatible?
        }
        else {
            // Copy is required, fail if `py::arg().noconvert()` flag is set.
            if (!convert) return false;

            // Create a copy matching the tensor map cast, ensuring the allocation is:
            //  - Memory continous (of row/col)
            //  - Correct scalar type
            Array src_cpy = Array::ensure(src);
            if (!src_cpy)
                return false;

            // Get the shape
            shape_conform = props::conformable(src_cpy, true);
            if (!shape_conform)
                return false; // Incompatible, should not occur

            // Create a TensorMapping
            map.reset(new Type(src_cpy.mutable_data(), shape_conform.shape));

            callback.src_ref = reinterpret_borrow<array>(src);
            callback.src_copy = std::move(src_cpy);
            loader_life_support::add_patient(callback.src_copy);
            // Only run update the source buffer if the expecting changes to the copied buffer.
            callback.update = is_writeable;
        }

#ifdef DEBUG_OUT
        std::cout << typeid(Type).name() << std::endl;
        std::cout << "Write access: " << need_writeable << std::endl;
        std::cout << "Shape: " << shape_conform.shape << std::endl;
        std::cout << "Stride: " << shape_conform.stride << std::endl;
#endif

        return true;
    }

    static handle cast_impl(const Type* src, return_value_policy policy, handle parent) {

#ifndef PYBIND11_ET_STRICT
        static_assert(false, "exposing an Eigen::TensorMap to python is disallowed under PYBIND11_ET_STRICT rules.");
#endif
        static_assert(false, "exposing an Eigen::TensorMap to python is not implemented.");
    }

#pragma region cast implementations
    // non-reference, non-const type:
    static handle cast(Type&& src, return_value_policy policy, handle parent) {
        return cast_impl(&src, policy, parent);
    }
    // non-reference, non-const type:
    static handle cast(const Type&& src, return_value_policy policy, handle parent) {
        return cast_impl(&src, policy, parent);
    }
    // const pointer return; disallowed return type with STRICT defined
    static handle cast(const Type* src, return_value_policy policy, handle parent) {
#ifdef PYBIND11_ET_STRICT
        static_assert(false, "exposing a pointer of Eigen::TensorMap to python is disallowed under PYBIND11_ET_STRICT compilation rules.");
#else
        return cast_impl(src, policy, parent);
#endif
    }
    // non-const pointer return
    static handle cast(Type* src, return_value_policy policy, handle parent) {
        return cast(const_cast<const Type*>(src), policy, parent);
    }
    // lvalue reference return; disallowed return type without PERMISSIVE defined
    static handle cast(const Type& src, return_value_policy policy, handle parent) {
#ifdef PYBIND11_ET_PERMISSIVE
        return cast_impl(src, policy, parent);
#else
        static_assert(false, "exposing an lvalue reference of Eigen::TensorMap to python is disallowed.");
#endif
    }
    static handle cast(Type& src, return_value_policy policy, handle parent) {
        return cast(const_cast<const Type&>(src), policy, parent);
    }
#pragma endregion

    static constexpr auto name = props::descriptor;

    operator Type* () { return map.get(); }
    operator Type& () { return *map; }
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;

private:
    template <typename T = TensorType, enable_if_t<is_eigen_mutable_tensor<T>::value, int> = 0>
    Scalar* data(Array& a) { return a.mutable_data(); }

    template <typename T = TensorType, enable_if_t<!is_eigen_mutable_tensor<T>::value, int> = 0>
    const Scalar* data(Array& a) const { return a.data(); }


};


// TensorRef<...> type_caster.
template <typename TensorType>
struct type_caster<
    Eigen::TensorRef<TensorType>,
    std::enable_if_t<is_eigen_tensor_ref<Eigen::TensorRef<TensorType>>::value>>
{
private:
    using Type = Eigen::TensorRef<TensorType>;
    using props = EigenTensorProps<Type>;
    using Scalar = typename props::Scalar;
    using MapType = Eigen::TensorMap<TensorType>;
    static constexpr bool need_writeable = is_eigen_mutable_tensor<Type>::value;

    using Array = array_t<Scalar>;

    // Delay construction (no default constructor)
    std::unique_ptr<MapType> map;
    std::unique_ptr<Type> ref;

    // Copy or reference to the numpy data.
    Array copy_or_ref;
public:
    bool load(handle src, bool convert) {
        //Check array_t argument matches the scalar type.
        bool scalar_type_mismatch = !isinstance<Array>(src);

        // Only support row major (c_style) tensors
        if (check_flags(src.ptr(), array::f_style))
            return false;
        if (need_writeable && !check_flags(src.ptr(), npy_api::constants::NPY_ARRAY_WRITEABLE_))
            return false;   // Need writeable arg.

        ShapeConformable<props::rank, props::row_major> conform;
        if (!scalar_type_mismatch) {
            // We don't need a converting copy, but we need to ensure it conforms.
            auto aref = reinterpret_borrow<Array>(src);

            if (aref) {
                conform = props::conformable(aref, false);
                if (!conform)
                    return false; // Incompatible
                copy_or_ref = std::move(aref);
            }
            else
                return false; // Incompatible?
        }
        else {
            // Copy is required, fail if `py::arg().noconvert()` flag is set or writeable is required.
            if (!convert || need_writeable) return false;

            Array copy = Array::ensure(src);
            if (!copy) 
                return false;
            conform = props::conformable(copy, false);
            if (!conform)
                return false;
            copy_or_ref = std::move(copy);
            loader_life_support::add_patient(copy_or_ref);
        }

#ifdef DEBUG_OUT
        std::cout << typeid(TensorType).name() << std::endl;
        std::cout << "Write access: " << need_writeable << std::endl;
        std::cout << "Shape: " << conform.shape << std::endl;
        std::cout << "Stride: " << conform.stride << std::endl;
#endif

        // Map the python data, cast away the const classifier to map it to a non-const ref internally.
        ref.reset();
        map.reset(new MapType(const_cast<props::Scalar*>(data(copy_or_ref)), conform.shape));

        if (conform.mem_continous()) {
            ref.reset(new Type(*map));
        }
        else
            ref.reset(new Type(map->stride(conform.stride)));

        return true;
    }

    /* Under strict rules always return an evaluated copy of the tensor reffered to.
    *  Under more permissive rules other return types are allowed under the
    *   assumption that the implementer understand the consequences (see eigen matrix impl.).
    *  If the tensor argument is a tensor expression, 
    *   the expression is evaluated into a new tensor and returned.
    */
    static handle cast_impl(Type* src, return_value_policy policy, handle parent) {

        const props::Scalar* ptr = src->data();

        // If no data ptr, assume the argument is an expression which requires evaluation.
        if (!ptr) 
            policy = return_value_policy::copy;

        switch (policy) {
        case return_value_policy::reference_internal:
#ifndef PYBIND11_ET_STRICT
            return eigen_tensor_array_cast<props>(*src, parent, is_eigen_mutable_tensor<Type>::value);
#endif
        case return_value_policy::reference:
        case return_value_policy::automatic_reference:
#ifndef PYBIND11_ET_STRICT
            return eigen_tensor_array_cast<props>(*src, none(), is_eigen_mutable_tensor<Type>::value);
#endif
        case return_value_policy::copy:
        case return_value_policy::automatic:
        default:
        {
            // Create a new dense tensor copy to pass to python.
            TensorType* out = new TensorType();
            *out = *src; // evaluate/copy
            return eigen_encapsulate_tensor<EigenTensorProps<TensorType>>(out);
        }
        }
    }

#pragma region cast implementations

    // non-reference, non-const type:
    static handle cast(Type&& src, return_value_policy policy, handle parent) {
        return cast_impl(&src, policy, parent);
    }
    // non-reference, non-const type:
    static handle cast(const Type&& src, return_value_policy policy, handle parent) {
        return cast_impl(&src, policy, parent);
    }
    // const pointer return; disallowed return type with STRICT defined
    static handle cast(const Type* src, return_value_policy policy, handle parent) {
#ifdef PYBIND11_ET_STRICT
        static_assert(false, "exposing a pointer of Eigen::TensorRef to python is disallowed under PYBIND11_ET_STRICT compilation rules.");
#else
        return cast_impl(src, policy, parent);
#endif
    }
    // non-const pointer return
    static handle cast(Type* src, return_value_policy policy, handle parent) {
        return cast(const_cast<const Type*>(src), policy, parent);
    }
    // lvalue reference return; disallowed return type without PERMISSIVE defined
    static handle cast(const Type& src, return_value_policy policy, handle parent) {
#ifdef PYBIND11_ET_PERMISSIVE
        return cast_impl(src, policy, parent);
#else
        static_assert(false, "exposing an lvalue reference of Eigen::TensorRef to python is disallowed.");
#endif
    }
    static handle cast(Type& src, return_value_policy policy, handle parent) {
        return cast(const_cast<const Type&>(src), policy, parent);
    }

#pragma endregion

    static constexpr auto name = props::descriptor;

    operator Type* () { return ref.get(); }
    operator Type& () { return *ref; }
    //operator const Type& () = delete;
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;

private:
    template <typename T = TensorType, enable_if_t<is_eigen_mutable_tensor<T>::value, int> = 0>
    Scalar* data(Array& a) { return a.mutable_data(); }

    template <typename T = TensorType, enable_if_t<!is_eigen_mutable_tensor<T>::value, int> = 0>
    const Scalar* data(Array& a) const { return a.data(); }


};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)