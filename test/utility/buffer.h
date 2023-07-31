#pragma once

#include <cstddef>
#include <type_traits>
#include <array>

namespace riptide_utility::internal
{
    namespace details::buffer
    {
        template <typename Cont>
        using enable_if_container_t =
            std::void_t<decltype(std::data(std::declval<Cont>())), decltype(std::size(std::declval<Cont>()))>;

    }

    /// @brief Const contiguous buffer of T.
    /// @tparam T value type.
    template <typename T>
    class const_buffer
    {
    public:
        const_buffer() noexcept = default;

        explicit const_buffer(T const * const data, size_t const size) noexcept
            : data_{ data }
            , size_{ size }
        {
        }

        template <template <typename> typename Cont, typename = details::buffer::enable_if_container_t<Cont<T>>>
        const_buffer(Cont<T> const & cont) noexcept(noexcept(std::data(cont)) && noexcept(std::size(cont)))
            : const_buffer(std::data(cont), std::size(cont))
        {
        }

        [[nodiscard]] T const * begin() const noexcept
        {
            return data_;
        }

        [[nodiscard]] T const * data() const noexcept
        {
            return data_;
        }

        [[nodiscard]] T const * end() const noexcept
        {
            return data_ + size_;
        }

        [[nodiscard]] size_t size() const noexcept
        {
            return size_;
        }

        [[nodiscard]] T const & operator[](size_t const pos) const
        {
            return data_[pos];
        }

    private:
        T const * data_{ nullptr };
        size_t size_{ 0 };
    };

    /// @brief Mutable contiguous buffer of T.
    /// @tparam T value type.
    template <typename T>
    class mutable_buffer : public const_buffer<T>
    {
        using base_t = const_buffer<T>;

    public:
        mutable_buffer() = default;

        explicit mutable_buffer(T * const data, size_t const size) noexcept
            : base_t(const_cast<T const *>(data), size)
        {
        }

        template <template <typename> typename Cont, typename = details::buffer::enable_if_container_t<Cont<T>>>
        mutable_buffer(Cont<T> const & cont) noexcept(noexcept(std::data(cont)) && noexcept(std::size(cont)))
            : mutable_buffer(std::data(cont), std::size(cont))
        {
        }

        [[nodiscard]] T * begin() noexcept
        {
            return const_cast<T *>(base_t::begin());
        }

        [[nodiscard]] T * data() noexcept
        {
            return const_cast<T *>(base_t::data());
        }

        [[nodiscard]] T * end() noexcept
        {
            return const_cast<T *>(base_t::end());
        }

        [[nodiscard]] T & operator[](size_t const pos)
        {
            return begin()[pos];
        }
    };
}
