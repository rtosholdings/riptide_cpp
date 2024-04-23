#pragma once

#include "simd/avx2.h"
#include "missing_values.h"

#include <immintrin.h>
#include <span>

namespace riptide::simd
{
    template <typename T>
    bool contains(const std::span<T> & array, T value)
    {
        using vec256 = avx2::vec256<T>;
        const auto broadcasted = vec256::broadcast(value);

        size_t i = 0;
        constexpr size_t N = sizeof(typename vec256::reg_type) / sizeof(T);

        for (; i + N <= array.size(); i += N)
        {
            bool any_equal = false;

            auto x = vec256::load_unaligned(&array[i]);
            auto compare = vec256::isequal(x, broadcasted);

            if constexpr (std::is_same_v<T, float>)
            {
                any_equal = _mm256_movemask_ps(compare) != 0;
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                any_equal = _mm256_movemask_pd(compare) != 0;
            }
            else
            {
                any_equal = _mm256_movemask_epi8(compare) != 0;
            }

            if (any_equal)
            {
                return true;
            }
        }

        for (; i < array.size(); i++)
        {
            if (array[i] == value)
            {
                return true;
            }
        }

        return false;
    }

    template <typename T, std::enable_if_t<! std::is_floating_point_v<T>, bool> = true>
    bool contains_invalid(const std::span<T> & array)
    {
        return contains(array, riptide::invalid_for_type<T>::value);
    }

    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    bool contains_invalid(const std::span<T> & array)
    {
        using vec256 = avx2::vec256<T>;

        size_t i = 0;
        constexpr size_t N = sizeof(typename vec256::reg_type) / sizeof(T);

        for (; i + N <= array.size(); i += N)
        {
            bool any_invalid = false;

            auto x = vec256::load_unaligned(&array[i]);
            auto compare = vec256::isequal(x, x);

            if constexpr (std::is_same_v<T, float>)
            {
                any_invalid = _mm256_movemask_ps(compare) != 0xff;
            }
            else
            {
                any_invalid = _mm256_movemask_pd(compare) != 0xf;
            }

            if (any_invalid)
            {
                return true;
            }
        }

        for (; i < array.size(); i++)
        {
            if (array[i] != array[i])
            {
                return true;
            }
        }

        return false;
    }
}