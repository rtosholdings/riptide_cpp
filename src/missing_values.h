#pragma once
#include <cstdint>
#include <limits>
#include <type_traits>

namespace riptide
{
    namespace details
    {
#ifdef __cpp_lib_remove_cvref
        template <typename T>
        using remove_cvref_t = std::remove_cvref_t<T>;
#else
        template <typename T>
        using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
#endif

        template <typename TExpected, typename TActual>
        using bool_if_same = std::enable_if_t<std::is_same_v<remove_cvref_t<TExpected>, remove_cvref_t<TActual>>, bool>;
    }

    /**
     * Type trait for getting an invalid value (by the riptide definition) for a
     * given C++ type.
     */
    template <typename T>
    struct invalid_for_type;

    template <>
    struct invalid_for_type<bool>
    {
        /* TODO: 'bool' does not have an invalid value in riptide; remove this field?
         */
        static constexpr bool value = false;

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            // Return true (always) here until/if we really support invalid/null values
            // for bools.
            return true;
        }
    };

    template <>
    struct invalid_for_type<int8_t>
    {
        // NOTE: We don't use std::numeric_limits<int8_t>::min() here, because if the
        // value
        //       differs between platforms (for whatever reason), it'd mean that
        //       different systems would interpret the same data values differently.
        //       Hard-coding the constant value here avoids that.
        static constexpr int8_t value = -128; // 0x80

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            return x != value;
        }
    };

    template <>
    struct invalid_for_type<int16_t>
    {
        static constexpr int16_t value = -32768; // 0x8000

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            return x != value;
        }
    };

    template <>
    struct invalid_for_type<int32_t>
    {
        static constexpr int32_t value = 0x80000000;

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            return x != value;
        }
    };

    template <>
    struct invalid_for_type<int64_t>
    {
        static constexpr int64_t value = 0x8000000000000000;

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            return x != value;
        }
    };

    template <>
    struct invalid_for_type<uint8_t>
    {
        // NOTE: We don't use std::numeric_limits<int8_t>::min() here, because if the
        // value
        //       differs between platforms (for whatever reason), it'd mean that
        //       different systems would interpret the same data values differently.
        //       Hard-coding the constant value here avoids that.
        static constexpr uint8_t value = 0xff;

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            return x != value;
        }
    };

    template <>
    struct invalid_for_type<uint16_t>
    {
        static constexpr uint16_t value = 0xffff;

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            return x != value;
        }
    };

    template <>
    struct invalid_for_type<uint32_t>
    {
        static constexpr uint32_t value = 0xffffffff;

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            return x != value;
        }
    };

    template <>
    struct invalid_for_type<uint64_t>
    {
        static constexpr uint64_t value = 0xffffffffffffffff;

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            return x != value;
        }
    };

    template <>
    struct invalid_for_type<float>
    {
        static constexpr auto value = std::numeric_limits<float>::quiet_NaN();

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            // There are multiple NaN values, so the check for floating-point types
            // needs to be done differently than for integers.
            return x == x;
        }
    };

    template <>
    struct invalid_for_type<double>
    {
        static constexpr auto value = std::numeric_limits<double>::quiet_NaN();

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            // There are multiple NaN values, so the check for floating-point types
            // needs to be done differently than for integers.
            return x == x;
        }
    };

    template <>
    struct invalid_for_type<long double>
    {
        static constexpr auto value = std::numeric_limits<long double>::quiet_NaN();

        /**
         * @brief Determine if a given value is considered 'valid'.
         *
         * @param x The value to check.
         * @return true The value is 'valid'.
         * @return false The value is invalid / NaN.
         */
        template <typename T>
        static constexpr details::bool_if_same<T, decltype(value)> is_valid(const T x)
        {
            // There are multiple NaN values, so the check for floating-point types
            // needs to be done differently than for integers.
            return x == x;
        }
    };

    /**
     * @brief Casting function which understands riptide invalid/NA values and
     * preserves them when converting.
     *
     * @tparam TargetType The type to convert the source value.
     * @tparam SourceType The type of the source value.
     */
    template <typename TargetType, typename SourceType>
    inline constexpr TargetType cast_nan_aware(SourceType const value)
    {
        return invalid_for_type<SourceType>::is_valid(value)
                   // The value is valid, so perform the conversion normally.
                   // NOTE: Because of the need to preserve values as closely as
                   // possible, this function
                   //       isn't able to handle the case where a valid value (such as
                   //       255.f) is converted to another type and happens to be the
                   //       invalid value for that type. This problem isn't really
                   //       solvable without having a flag at the numpy C array level
                   //       indicating whether the array should be considered as
                   //       having sentinel values or not. Having that flag (or a
                   //       bitmask indicating which values are valid) would allow the
                   //       user to decide how to handle these values.
                   ?
                   static_cast<TargetType>(value) :
                   // The input value is invalid, so return the/an
                   // invalid value for the target type.
                   invalid_for_type<TargetType>::value;
    }
} // namespace riptide
