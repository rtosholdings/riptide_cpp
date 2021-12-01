#pragma once
#include <cstdint>
#include <limits>
#include <typeinfo>

#if defined(_WIN32) && ! defined(__GNUC__)
    #include <../Lib/site-packages/numpy/core/include/numpy/ndarraytypes.h>
#else
    #include <numpy/ndarraytypes.h>
#endif

// TODO: Remove these once users of this header have switched over.
#include "missing_values.h"

namespace riptide
{
    /**
     * @brief Template-based, compile-time mapping between C++ types and numpy type
     * codes (e.g. NPY_FLOAT64).
     *
     * @tparam T A C++ primitive type.
     */
    template <typename T>
    struct numpy_type_code
    {
    };

    template <>
    struct numpy_type_code<bool>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_BOOL;
    };

    template <>
    struct numpy_type_code<npy_int8>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT8;
    };

    template <>
    struct numpy_type_code<npy_int16>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT16;
    };

    template <>
    struct numpy_type_code<npy_int32>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT32;
    };

    template <>
    struct numpy_type_code<npy_int64>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT64;
    };

    template <>
    struct numpy_type_code<npy_uint8>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT8;
    };

    template <>
    struct numpy_type_code<npy_uint16>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT16;
    };

    template <>
    struct numpy_type_code<npy_uint32>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT32;
    };

    template <>
    struct numpy_type_code<npy_uint64>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT64;
    };

    template <>
    struct numpy_type_code<npy_float32>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_FLOAT32;
    };

    template <>
    struct numpy_type_code<npy_float64>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_FLOAT64;
    };

#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE

    template <>
    struct numpy_type_code<npy_longdouble>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_LONGDOUBLE;
    };

#endif

    /**
     * @brief Template-based, compile-time mapping between C++ types and numpy type
     * codes (e.g. NPY_FLOAT64).
     *
     * @tparam T A C++ primitive type.
     */
    template <typename T>
    struct numpy_ctype_code
    {
    };

    template <>
    struct numpy_ctype_code<bool>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_BOOL;
    };

    template <>
    struct numpy_ctype_code<int8_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT8;
    };

    template <>
    struct numpy_ctype_code<int16_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT16;
    };

    template <>
    struct numpy_ctype_code<int32_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT32;
    };

    template <>
    struct numpy_ctype_code<int64_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT64;
    };

    template <>
    struct numpy_ctype_code<uint8_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT8;
    };

    template <>
    struct numpy_ctype_code<uint16_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT16;
    };

    template <>
    struct numpy_ctype_code<uint32_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT32;
    };

    template <>
    struct numpy_ctype_code<uint64_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT64;
    };

    template <>
    struct numpy_ctype_code<float>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_FLOAT32;
    };

    template <>
    struct numpy_ctype_code<double>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_FLOAT64;
    };

#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE

    template <>
    struct numpy_ctype_code<long double>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_LONGDOUBLE;
    };

#endif

    /**
     * @brief Template-based, compile-time mapping between C++ types and numpy type
     * codes (e.g. NPY_FLOAT64).
     *
     * @tparam ndarraytype An NPY_TYPES value.
     */
    template <NPY_TYPES ndarraytype>
    struct numpy_cpp_type
    {
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_BOOL>
    {
        using type = bool;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_INT8>
    {
        using type = npy_int8;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_INT16>
    {
        using type = npy_int16;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_INT32>
    {
        using type = npy_int32;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_INT64>
    {
        using type = npy_int64;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_UINT8>
    {
        using type = npy_uint8;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_UINT16>
    {
        using type = npy_uint16;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_UINT32>
    {
        using type = npy_uint32;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_UINT64>
    {
        using type = npy_uint64;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_FLOAT32>
    {
        using type = npy_float32;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_FLOAT64>
    {
        using type = npy_float64;
    };

#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_LONGDOUBLE>
    {
        using type = npy_longdouble;
    };

#endif

    /**
     * @brief Type trait for getting the cutoff (as a number of elements).
     *
     * @tparam ndarraytype
     */
    template <NPY_TYPES ndarraytype>
    struct arrlen_index_cutoff
    {
    };

    template <>
    struct arrlen_index_cutoff<NPY_TYPES::NPY_INT8>
    {
        // N.B. The 'value' field here should really use ssize_t as the type, but
        // since we've hard-coded int64_t throughout riptide, we need to use that for
        // compatibility (at least for now).
        static constexpr int64_t value = 100;
    };

    template <>
    struct arrlen_index_cutoff<NPY_TYPES::NPY_INT16>
    {
        static constexpr int64_t value = 30000;
    };

    template <>
    struct arrlen_index_cutoff<NPY_TYPES::NPY_INT32>
    {
        static constexpr int64_t value = 2000000000LL;
    };

    /**
     * @brief Get the NPY_TYPES value for the smallest integer type
     * which can be used as a fancy index for an array with the
     * given number of elements.
     *
     * @param array_length The number of elements in an array.
     * @return constexpr NPY_TYPES
     */
    static constexpr NPY_TYPES index_size_type(int64_t array_length)
    {
        // VS2015 says it supports C++14, but doesn't fully -- so it's explicitly
        // excluded here even when the language version checks pass.
#if __cplusplus >= 201402L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201402L && _MSC_VER > 1900)
        if (array_length < 0)
        {
            return NPY_TYPES::NPY_NOTYPE;
        }
        else if (array_length < arrlen_index_cutoff<NPY_TYPES::NPY_INT8>::value)
        {
            return NPY_TYPES::NPY_INT8;
        }
        else if (array_length < arrlen_index_cutoff<NPY_TYPES::NPY_INT16>::value)
        {
            return NPY_TYPES::NPY_INT16;
        }
        else if (array_length < arrlen_index_cutoff<NPY_TYPES::NPY_INT32>::value)
        {
            return NPY_TYPES::NPY_INT32;
        }
        else
        {
            return NPY_TYPES::NPY_INT64;
        }
#else
        // Fall back to the C++11 constexpr style.
        return array_length < 0                                                ? NPY_TYPES::NPY_NOTYPE :
               array_length < arrlen_index_cutoff<NPY_TYPES::NPY_INT8>::value  ? NPY_TYPES::NPY_INT8 :
               array_length < arrlen_index_cutoff<NPY_TYPES::NPY_INT16>::value ? NPY_TYPES::NPY_INT16 :
               array_length < arrlen_index_cutoff<NPY_TYPES::NPY_INT32>::value ? NPY_TYPES::NPY_INT32 :
                                                                                 NPY_TYPES::NPY_INT64;
#endif
    }

    // The maximum number of rows (non-inclusive) for which we'll use a 32-bit
    // integer array to hold a fancy index.
    static constexpr int64_t int32_index_cutoff = arrlen_index_cutoff<NPY_INT32>::value;
} // namespace riptide
