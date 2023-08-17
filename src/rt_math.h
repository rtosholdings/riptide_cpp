#pragma once
#ifndef RIPTABLE_CPP_RT_MATH_H
    #define RIPTABLE_CPP_RT_MATH_H

    #include "missing_values.h"

    #include <cstdint>

namespace riptide::math
{
    /**
     * @brief Function like @c std::min but which always propagates NaN values.
     *
     * @tparam T The element type.
     * @param x The left element.
     * @param y The right element.
     * @return T The min value, or invalid if any are invalid.
     */
    template <typename T>
    T min_with_nan_passthru(T const x, T const y)
    {
        using invalid_type = invalid_for_type<T>;

        // floating point can take advantage of intrinsic nan-propagation
        if constexpr (std::is_floating_point_v<T>)
        {
            auto const blended = ! invalid_type::is_valid(x) ? x : y;
            return x < blended ? x : blended;
        }
        // integral types with invalid as min value can just use min
        else if constexpr (std::is_integral_v<T> && invalid_type::value == std::numeric_limits<T>::min())
        {
            return (std::min)(x, y);
        }
        // otherwise test for and return invalid, else min
        else
        {
            return invalid_type::is_valid(x) && invalid_type::is_valid(y) ? (std::min)(x, y) : invalid_type::value;
        }
    }

    /**
     * @brief Function like @c std::max but which always propagates NaN values.
     *
     * @tparam T The element type.
     * @param x The left element.
     * @param y The right element.
     * @return T The max value, or invalid if any are invalid.
     */
    template <typename T>
    T max_with_nan_passthru(T const x, T const y)
    {
        using invalid_type = invalid_for_type<T>;

        // floating point can take advantage of intrinsic nan-propagation
        if constexpr (std::is_floating_point_v<T>)
        {
            auto const blended = ! invalid_type::is_valid(x) ? x : y;
            return x > blended ? x : blended;
        }
        // integral types with invalid as max value can just use max
        else if constexpr (std::is_integral_v<T> && invalid_type::value == std::numeric_limits<T>::max())
        {
            return (std::max)(x, y);
        }
        // otherwise test for and return invalid, else max
        else
        {
            return invalid_type::is_valid(x) && invalid_type::is_valid(y) ? (std::max)(x, y) : invalid_type::value;
        }
    }

} // namespace math

#endif // RIPTABLE_CPP_RT_MATH_H