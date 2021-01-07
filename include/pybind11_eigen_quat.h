/*
*/

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Geometry>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)


// Casts an Eigen quaternion type to a numpy array. Writeable lets you turn off the writeable flag for the array.
template <typename Scalar, int Options> handle eigen_quat_array_cast(Eigen::Quaternion<Scalar, Options> src, bool writeable = true) {
    constexpr ssize_t elem_size = sizeof(Scalar);

    array arr = array(4, &src.x(), handle());

    if (!writeable)
        array_proxy(arr.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;

    return arr.release();
}

template <typename Scalar, int Options>
struct type_caster<Eigen::Quaternion<Scalar, Options>>
{
private:
    using Type = Eigen::Quaternion<Scalar, Options>;
    using Array = array_t<Scalar>;


    Type ref;

public:
    bool load(handle src, bool convert) {

        Array copy = Array::ensure(src);
        if (!copy)
            return false;

        // Copy
        ref = Type(copy.data()[3], copy.data()[0], copy.data()[1], copy.data()[2]);

        return true;
    }

    static handle cast_impl(Type src, return_value_policy policy, handle parent) {


        switch (policy) {
        case return_value_policy::reference_internal:
        case return_value_policy::reference:
        case return_value_policy::automatic_reference:
        case return_value_policy::copy:
        case return_value_policy::automatic:
        default:
        {
            // Copy
            return eigen_quat_array_cast<Scalar, Options>(src);
        }
        }
    }

    static handle cast(Type src, return_value_policy policy, handle parent) {
        return cast_impl(src, policy, parent);
    }
    static handle cast(Type* src, return_value_policy policy, handle parent) {
        return cast_impl(*src, policy, parent);
    }

    static constexpr auto name = _("Quat");

    operator Type* () { return &ref; }
    operator Type& () { return ref; }
    //operator const Type& () = delete;
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;



};
template struct type_caster<Eigen::Quaternion<float, 0>>;
template struct type_caster<Eigen::Quaternion<double, 0>>;
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)