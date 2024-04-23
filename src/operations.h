#pragma once

namespace riptide
{
    template <typename T, typename U>
    inline T cast(U x)
    {
        if (! riptide::invalid_for_type<U>::is_valid(x))
        {
            return riptide::invalid_for_type<T>::value;
        }

        return static_cast<T>(x);
    }

    template <typename T>
    inline T max(T x, T y)
    {
        if (! riptide::invalid_for_type<T>::is_valid(x) || ! riptide::invalid_for_type<T>::is_valid(y))
        {
            return riptide::invalid_for_type<T>::value;
        }

        return std::max(x, y);
    }

    template <typename T>
    inline T min(T x, T y)
    {
        if (! riptide::invalid_for_type<T>::is_valid(x) || ! riptide::invalid_for_type<T>::is_valid(y))
        {
            return riptide::invalid_for_type<T>::value;
        }

        return std::min(x, y);
    }

    template <typename T>
    inline T add(T x, T y)
    {
        if (! riptide::invalid_for_type<T>::is_valid(x) || ! riptide::invalid_for_type<T>::is_valid(y))
        {
            return riptide::invalid_for_type<T>::value;
        }

        return x + y;
    }
}