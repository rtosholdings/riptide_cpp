#pragma once
#pragma once
#ifndef RIPTABLE_CPP_RT_MATH_H
    #define RIPTABLE_CPP_RT_MATH_H

    #include "missing_values.h"

    #include <cstdint>

namespace riptide::math
{
    /**
     * @brief Function like @c std::min but which always propagates NaN values (for
     * floating-point types).
     *
     * @tparam T The element type.
     * @param x The left element.
     * @param y The right element.
     * @return T const& The result of the operation.
     */
    template <typename T>
    T const & min_with_nan_passthru(T const & x, T const & y)
    {
        return (std::min)(x, y);
    }

    template <>
    float const & min_with_nan_passthru(float const & x, float const & y)
    {
        const auto & blended = (x != x) ? x : y;
        return x < blended ? x : blended;
    }

    template <>
    double const & min_with_nan_passthru(double const & x, double const & y)
    {
        const auto & blended = (x != x) ? x : y;
        return x < blended ? x : blended;
    }

    /**
     * @brief Function like @c std::max but which always propagates NaN values (for
     * floating-point types).
     *
     * @tparam T The element type.
     * @param x The left element.
     * @param y The right element.
     * @return T const& The result of the operation.
     */
    template <typename T>
    T const & max_with_nan_passthru(T const & x, T const & y)
    {
        return (std::max)(x, y);
    }

    template <>
    float const & max_with_nan_passthru(float const & x, float const & y)
    {
        const auto & blended = (x != x) ? x : y;
        return x > blended ? x : blended;
    }

    template <>
    double const & max_with_nan_passthru(double const & x, double const & y)
    {
        const auto & blended = (x != x) ? x : y;
        return x > blended ? x : blended;
    }
} // namespace math

#endif // RIPTABLE_CPP_RT_MATH_H