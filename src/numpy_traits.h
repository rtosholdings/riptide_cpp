#pragma once
#include <cstdint>
#include <limits>
#include <typeinfo>

#include <numpy/ndarraytypes.h>

namespace riptide
{
    // Naming conventions:
    // type_code is the NPY_TYPES enumeration (e.g. NPY_INT16).
    // cpp_type is the fixed C++ types (e.g. int16_t).
    // c_type is the NumPy C storage types (e.g. npy_int16)

    /// @brief Mapping between C++ fixed types and numpy type codes (e.g. int16_t -> NPY_FLOAT64).
    ///
    /// @tparam T A C++ fixed primitive type.
    template <typename T>
    struct numpy_type_code;

    template <typename T>
    inline constexpr NPY_TYPES numpy_type_code_v = numpy_type_code<T>::value;

    /// @brief Mapping between numpy type codes and C++ fixed types (e.g. NPY_INT16 -> int16_t)
    ///
    /// @tparam ndarraytype An NPY_TYPES value.
    template <NPY_TYPES ndarraytype>
    struct numpy_cpp_type;

    template <NPY_TYPES TypeCode>
    using numpy_cpp_type_t = typename numpy_cpp_type<TypeCode>::type;

    /// @brief Mapping between C storage types and numpy type codes (e.g. npy_int16 -> NPY_INT16).
    ///
    /// @tparam T A NumPy C storage type.
    template <typename T>
    struct numpy_c_type_code;

    template <typename T>
    inline constexpr NPY_TYPES numpy_c_type_code_v = numpy_c_type_code<T>::value;

    /// @brief Mapping between numpy type codes and C storage types (e.g. NPY_INT16 -> npy_int16).
    ///
    /// @tparam ndarraytype An NPY_TYPES value.
    template <NPY_TYPES ndarraytype>
    struct numpy_c_type;

    template <NPY_TYPES TypeCode>
    using numpy_c_type_t = typename numpy_c_type<TypeCode>::type;

    /// @brief Indicates if InT is storable in OutT (same representation and size).
    /// @tparam InT Fixed C++ type.
    /// @tparam OutT NumPy C storage type.
    template <typename InT, typename OutT>
    inline constexpr bool numpy_is_storable_v =
        sizeof(InT) == sizeof(OutT) && std::is_arithmetic_v<OutT> && std::is_arithmetic_v<InT> &&
        std::numeric_limits<OutT>::is_integer == std::numeric_limits<InT>::is_integer &&
        std::numeric_limits<OutT>::is_signed == std::numeric_limits<InT>::is_signed &&
        std::numeric_limits<OutT>::digits >= std::numeric_limits<InT>::digits;
}

namespace riptide
{
    template <>
    struct numpy_c_type_code<bool>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_BOOL;
    };

    template <>
    struct numpy_c_type_code<npy_int8>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT8;
    };

    template <>
    struct numpy_c_type_code<npy_int16>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT16;
    };

    template <>
    struct numpy_c_type_code<npy_int32>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT32;
    };

    template <>
    struct numpy_c_type_code<npy_int64>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT64;
    };

    template <>
    struct numpy_c_type_code<npy_uint8>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT8;
    };

    template <>
    struct numpy_c_type_code<npy_uint16>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT16;
    };

    template <>
    struct numpy_c_type_code<npy_uint32>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT32;
    };

    template <>
    struct numpy_c_type_code<npy_uint64>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT64;
    };

    template <>
    struct numpy_c_type_code<npy_float32>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_FLOAT32;
    };

    template <>
    struct numpy_c_type_code<npy_float64>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_FLOAT64;
    };

#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE

    template <>
    struct numpy_c_type_code<npy_longdouble>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_LONGDOUBLE;
    };

#endif
}

namespace riptide
{
    template <>
    struct numpy_type_code<bool>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_BOOL;
    };

    template <>
    struct numpy_type_code<int8_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT8;
    };

    template <>
    struct numpy_type_code<int16_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT16;
    };

    template <>
    struct numpy_type_code<int32_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT32;
    };

    template <>
    struct numpy_type_code<int64_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_INT64;
    };

    template <>
    struct numpy_type_code<uint8_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT8;
    };

    template <>
    struct numpy_type_code<uint16_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT16;
    };

    template <>
    struct numpy_type_code<uint32_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT32;
    };

    template <>
    struct numpy_type_code<uint64_t>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_UINT64;
    };

    template <>
    struct numpy_type_code<float>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_FLOAT32;
    };

    template <>
    struct numpy_type_code<double>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_FLOAT64;
    };

#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE

    template <>
    struct numpy_type_code<long double>
    {
        static constexpr NPY_TYPES value = NPY_TYPES::NPY_LONGDOUBLE;
    };

#endif
}

namespace riptide
{
    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_BOOL>
    {
        using type = bool;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_INT8>
    {
        using type = int8_t;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_INT16>
    {
        using type = int16_t;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_INT32>
    {
        using type = int32_t;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_INT64>
    {
        using type = int64_t;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_UINT8>
    {
        using type = uint8_t;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_UINT16>
    {
        using type = uint16_t;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_UINT32>
    {
        using type = uint32_t;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_UINT64>
    {
        using type = uint64_t;
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_FLOAT32>
    {
        using type = float; // float32_t
        static_assert(sizeof(type) == 32 / 8);
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_FLOAT64>
    {
        using type = double; // float64_t
        static_assert(sizeof(type) == 64 / 8);
    };

    template <>
    struct numpy_cpp_type<NPY_TYPES::NPY_LONGDOUBLE>
    {
#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE
        using type = long double;
#else
        using type = double;
#endif
        static_assert(sizeof(type) >= 64 / 8);
    };
}

namespace riptide
{
    template <>
    struct numpy_c_type<NPY_TYPES::NPY_BOOL>
    {
        using type = npy_bool;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_INT8>
    {
        using type = npy_int8;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_INT16>
    {
        using type = npy_int16;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_INT32>
    {
        using type = npy_int32;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_INT64>
    {
        using type = npy_int64;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_UINT8>
    {
        using type = npy_uint8;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_UINT16>
    {
        using type = npy_uint16;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_UINT32>
    {
        using type = npy_uint32;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_UINT64>
    {
        using type = npy_uint64;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_FLOAT32>
    {
        using type = npy_float32;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_FLOAT64>
    {
        using type = npy_float64;
    };

    template <>
    struct numpy_c_type<NPY_TYPES::NPY_LONGDOUBLE>
    {
        using type = npy_longdouble;
    };
}

namespace riptide
{
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
        if (array_length < 0)
        {
            return NPY_TYPES::NPY_NOTYPE;
        }

        if (array_length < arrlen_index_cutoff<NPY_TYPES::NPY_INT8>::value)
        {
            return NPY_TYPES::NPY_INT8;
        }

        if (array_length < arrlen_index_cutoff<NPY_TYPES::NPY_INT16>::value)
        {
            return NPY_TYPES::NPY_INT16;
        }

        if (array_length < arrlen_index_cutoff<NPY_TYPES::NPY_INT32>::value)
        {
            return NPY_TYPES::NPY_INT32;
        }

        return NPY_TYPES::NPY_INT64;
    }

    // The maximum number of rows (non-inclusive) for which we'll use a 32-bit
    // integer array to hold a fancy index.
    static constexpr int64_t int32_index_cutoff = arrlen_index_cutoff<NPY_INT32>::value;
} // namespace riptide
