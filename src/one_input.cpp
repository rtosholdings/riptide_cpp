#include "one_input.h"
#include "one_input_impl.h"
#include "overloaded.h"

#include "MathWorker.h"
#include "RipTide.h"
#include "basic_ops.h"
#include "ndarray.h"

#include "simd/avx2.h"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>

PyObject * process_one_input(PyArrayObject const * in_array, PyArrayObject * out_object_1, int32_t function_num,
                             int32_t numpy_intype, int32_t numpy_outtype)
{
    int32_t ndim{};
    int64_t stride{};

    int32_t direction{ GetStridesAndContig(in_array, ndim, stride) };
    npy_intp len{ CALC_ARRAY_LENGTH(ndim, PyArray_DIMS(const_cast<PyArrayObject *>(in_array))) };

    auto [opt_op_trait, opt_type_trait] = riptable_cpp::set_traits(function_num, numpy_intype);

    if (opt_op_trait && opt_type_trait)
    {
        if (direction == 0 && numpy_outtype == -1)
        {
            numpy_outtype = riptable_cpp::get_active_value_return(
                                *opt_op_trait, std::make_index_sequence<std::variant_size_v<riptable_cpp::operation_t>>{}) ?
                                numpy_intype :
                                NPY_BOOL;
            PyArrayObject * result_array{ (ndim <= 1) ? AllocateNumpyArray(1, &len, numpy_outtype) :
                                                        AllocateLikeNumpyArray(in_array, numpy_outtype) };

            if (result_array)
            {
                char const * in_p = PyArray_BYTES(const_cast<PyArrayObject *>(in_array));
                char * out_p{ PyArray_BYTES(const_cast<PyArrayObject *>(result_array)) };

                riptable_cpp::walk_data_array(1, len, stride, stride, in_p, out_p, *opt_op_trait, *opt_type_trait);
            }
            else
            {
                Py_INCREF(Py_None);
                return Py_None;
            }

            return reinterpret_cast<PyObject *>(result_array);
        }
        else
        {
            int wanted_outtype = riptable_cpp::get_active_value_return(
                                     *opt_op_trait, std::make_index_sequence<std::variant_size_v<riptable_cpp::operation_t>>{}) ?
                                     numpy_intype :
                                     NPY_BOOL;

            if (numpy_outtype != -1 && numpy_outtype != wanted_outtype)
            {
                LOGGING("Wanted output type %d does not match output type %d\n", wanted_outtype, numpy_outtype);
                Py_INCREF(Py_None);
                return Py_None;
            }

            PyArrayObject * result_array{ numpy_outtype == -1 ? AllocateLikeNumpyArray(in_array, wanted_outtype) : out_object_1 };

            if ((result_array == nullptr) || ((result_array == out_object_1) && (len != ArrayLength(result_array))))
            {
                Py_INCREF(Py_None);
                return Py_None;
            }

            if (result_array == out_object_1)
            {
                Py_INCREF(result_array);
            }

            char const * in_p{ PyArray_BYTES(const_cast<PyArrayObject *>(in_array)) };
            char * out_p{ PyArray_BYTES(const_cast<PyArrayObject *>(result_array)) };

            int num_dims_out{};
            int64_t stride_out{};
            int direction_out = GetStridesAndContig(result_array, num_dims_out, stride_out);

            if (direction_out == 0)
            {
                switch (direction)
                {
                case 0:
                    riptable_cpp::walk_data_array(ndim, len, stride, stride_out, in_p, out_p, *opt_op_trait, *opt_type_trait);
                    break;
                case 1:
                    riptable_cpp::walk_row_major(in_p, out_p, ndim, in_array, stride_out, *opt_op_trait, *opt_type_trait);
                    break;
                case -1:
                    riptable_cpp::walk_column_major(in_p, out_p, ndim, in_array, stride_out, *opt_op_trait, *opt_type_trait);
                    break;
                }
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

namespace riptable_cpp
{
    chosen_traits_t set_traits(int32_t const function_num, int32_t const numpy_intype)
    {
        chosen_traits_t retval{};

        switch (numpy_intype)
        {
        case NPY_INT8:
            retval.second = int8_traits{};
            break;
        case NPY_INT16:
            retval.second = int16_traits{};
            break;
#if RT_COMPILER_MSVC
        case NPY_INT:
#endif
        case NPY_INT32:
            retval.second = int32_traits{};
            break;
#if (RT_COMPILER_CLANG || RT_COMPILER_GCC)
        case NPY_LONGLONG:
#endif
        case NPY_INT64:
            retval.second = int64_traits{};
            break;
        case NPY_UINT8:
            retval.second = uint8_traits{};
            break;
        case NPY_UINT16:
            retval.second = uint16_traits{};
            break;
#if RT_COMPILER_MSVC
        case NPY_UINT:
#endif
        case NPY_UINT32:
            retval.second = uint32_traits{};
            break;
#if (RT_COMPILER_CLANG || RT_COMPILER_GCC)
        case NPY_ULONGLONG:
#endif
        case NPY_UINT64:
            retval.second = uint64_traits{};
            break;
        case NPY_FLOAT:
            retval.second = float_traits{};
            break;
        case NPY_DOUBLE:
            retval.second = double_traits{};
            break;
        }

        switch (function_num)
        {
        case MATH_OPERATION::ABS:
            retval.first = abs_op{};
            break;
        case MATH_OPERATION::ISNAN:
            retval.first = isnan_op{};
            break;
        case MATH_OPERATION::ISNOTNAN:
            retval.first = isnotnan_op{};
            break;
        case MATH_OPERATION::ISFINITE:
            retval.first = isfinite_op{};
            break;
        case MATH_OPERATION::ISNOTFINITE:
            retval.first = isnotfinite_op{};
            break;
        case MATH_OPERATION::NEG:
            retval.first = bitwise_not_op{};
            break;
        case MATH_OPERATION::INVERT:
            retval.first = bitwise_not_op{};
            break;
        case MATH_OPERATION::FLOOR:
            retval.first = floor_op{};
            break;
        case MATH_OPERATION::CEIL:
            retval.first = ceil_op{};
            break;
        case MATH_OPERATION::TRUNC:
            retval.first = trunc_op{};
            break;
        case MATH_OPERATION::ROUND:
            retval.first = round_op{};
            break;
        case MATH_OPERATION::SQRT:
            retval.first = sqrt_op{};
            break;
        }

        return retval;
    }
} // namespace riptable_cpp
