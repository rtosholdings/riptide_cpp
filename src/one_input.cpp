#include "one_input.h"
#include "one_input_impl.h"
#include "overloaded.h"
#include "platform_detect.h"

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

    std::optional<riptable_cpp::operation_t> opt_op_trait = riptable_cpp::get_op_trait(function_num);
    std::optional<riptable_cpp::array_content_t> opt_type_trait = riptable_cpp::get_type_trait(numpy_intype);

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
    std::optional<array_content_t> get_type_trait(int32_t const numpy_intype)
    {
        switch (numpy_intype)
        {
        case NPY_INT8:
            return int8_traits{};
            break;
        case NPY_INT16:
            return int16_traits{};
            break;
#if RT_COMPILER_MSVC
        case NPY_INT:
#endif
        case NPY_INT32:
            return int32_traits{};
            break;
#if (RT_COMPILER_CLANG || RT_COMPILER_GCC)
        case NPY_LONGLONG:
#endif
        case NPY_INT64:
            return int64_traits{};
            break;
        case NPY_UINT8:
            return uint8_traits{};
            break;
        case NPY_UINT16:
            return uint16_traits{};
            break;
#if RT_COMPILER_MSVC
        case NPY_UINT:
#endif
        case NPY_UINT32:
            return uint32_traits{};
            break;
#if (RT_COMPILER_CLANG || RT_COMPILER_GCC)
        case NPY_ULONGLONG:
#endif
        case NPY_UINT64:
            return uint64_traits{};
            break;
        case NPY_FLOAT:
            return float_traits{};
            break;
        case NPY_DOUBLE:
            return double_traits{};
            break;
        case NPY_STRING:
            return string_traits{};
            break;
        case NPY_UNICODE:
            return unicode_traits{};
            break;
        default:
            throw(std::runtime_error("Invalid type traits requested"));
        }

        return {};
    }

    std::optional<operation_t> get_op_trait(int32_t const function_num)
    {
        switch (function_num)
        {
        case MATH_OPERATION::ABS:
            return abs_op{};
            break;
        case MATH_OPERATION::ISNAN:
            return isnan_op{};
            break;
        case MATH_OPERATION::ISNOTNAN:
            return isnotnan_op{};
            break;
        case MATH_OPERATION::ISFINITE:
            return isfinite_op{};
            break;
        case MATH_OPERATION::ISNOTFINITE:
            return isnotfinite_op{};
            break;
        case MATH_OPERATION::NEG:
            return bitwise_not_op{};
            break;
        case MATH_OPERATION::INVERT:
            return bitwise_not_op{};
            break;
        case MATH_OPERATION::FLOOR:
            return floor_op{};
            break;
        case MATH_OPERATION::CEIL:
            return ceil_op{};
            break;
        case MATH_OPERATION::TRUNC:
            return trunc_op{};
            break;
        case MATH_OPERATION::ROUND:
            return round_op{};
            break;
        case MATH_OPERATION::SQRT:
            return sqrt_op{};
            break;
        default:
            throw(std::runtime_error("Invalid unary operation requested"));
        }

        return {};
    }
} // namespace riptable_cpp
