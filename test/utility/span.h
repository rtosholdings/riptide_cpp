#pragma once

#include <cstddef>
#ifdef __cpp_lib_span
    #include <span>
#endif

namespace riptide_utility::internal
{
#ifdef __cpp_lib_span
    inline constexpr auto dynamic_extent = std::dynamic_extent;

    template <typename T, std::size_t Extent = dynamic_extent>
    using span = std::span<T, Extent>;
#else
    inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

    /// @brief Partial implementation of std::span<T>
    /// @tparam T element type
    /// @tparam Extent number of elements, or dynamic_extent if dynamic
    template <typename T, std::size_t Extent = dynamic_extent>
    class span
    {
    public:
        using element_type = T;
        using value_type = std::remove_cv_t<T>;
        using size_type = std::size_t;
        using pointer = T *;
        using const_pointer = T const *;
        using reference = T &;
        using const_reference = T const &;

        static constexpr std::size_t extent = Extent;

        constexpr span() noexcept {}

        template <typename It>
        explicit constexpr span(It const first, size_type const count)
            : data_{ first }
            , size_{ count }
        {
        }

        constexpr pointer data() const noexcept
        {
            return data_;
        }

        constexpr size_type size() const noexcept
        {
            return size_;
        }

        constexpr reference operator[](size_type idx) const
        {
            return data_[idx];
        }

    private:
        pointer data_{ nullptr };
        size_type size_{ 0 };
    };
#endif
}
