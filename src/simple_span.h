#ifndef RIPTIDE_SIMPLE_SPAN_H
#define RIPTIDE_SIMPLE_SPAN_H

#include <type_traits>
#include <functional>
#include <string_view>

namespace riptide_cpp
{
    // Simple string span, a trivial type modeling a std::basic_string_view<T>.
    template <typename T>
    struct simple_span
    {
        T const * pointer;
        size_t length;

        bool operator==(simple_span<T> const & b) const noexcept
        {
            return std::basic_string_view<T>{ pointer, length } == std::basic_string_view<T>{ b.pointer, b.length };
        }
    };
}

namespace std
{
    template <typename T>
    struct hash<typename riptide_cpp::simple_span<T>>
    {
        std::size_t operator()(typename riptide_cpp::simple_span<T> const & s) const noexcept
        {
            return std::hash<std::basic_string_view<T>>{}({ s.pointer, s.length });
        }
    };
}

#endif
