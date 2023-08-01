#pragma once

#include "buffer.h"
#include "mem_buffer.h"

#include <type_traits>
#include <variant>

namespace riptide_utility::internal
{
    /// Holds any kind of buffer and provides a const_buffer.
    template <typename T>
    class any_const_buffer
    {
    public:
        // Cannot use variant ctor because it will match types based on conversions.
        // So mem_buffer will be constructed as const_buffer and result in dangling ptrs.
        // Thus, we explicitly emplace by index based on type.
        // In addition, GCC10 parses emplace<I> as conditional so it must be explicitly told it's fn template.

        template <typename BufT>
        any_const_buffer(BufT && buf)
        {
            using namespace riptide_utility::internal;
            using test_t = std::remove_cvref_t<BufT>;
            constexpr size_t idx{ std::is_same_v<test_t, const_buffer<T>>   ? 0U :
                                  std::is_same_v<test_t, mutable_buffer<T>> ? 1U :
                                  std::is_same_v<test_t, mem_buffer<T>>     ? 2U :
                                                                              std::variant_npos };
            static_assert(idx != std::variant_npos);
            storage_.template emplace<idx>(std::forward<BufT>(buf));
        }

        [[nodiscard]] operator riptide_utility::internal::const_buffer<T>() const
        {
            using namespace riptide_utility::internal;

            if (std::holds_alternative<const_buffer<T>>(storage_))
            {
                return std::get<const_buffer<T>>(storage_);
            }
            if (std::holds_alternative<mutable_buffer<T>>(storage_))
            {
                return const_buffer<T>{ std::get<mutable_buffer<T>>(storage_) };
            }
            if (std::holds_alternative<mem_buffer<T>>(storage_))
            {
                return const_buffer<T>{ std::get<mem_buffer<T>>(storage_) };
            }
            throw std::runtime_error("Unexpected buffer alternative");
        }

        template <typename Visitor, typename... Variants>
        void visit(Visitor && visitor, Variants &&... variants)
        {
            std::visit(visitor, variants..., storage_);
        }

    private:
        using storage_t = std::variant<riptide_utility::internal::const_buffer<T>, riptide_utility::internal::mutable_buffer<T>,
                                       riptide_utility::internal::mem_buffer<T>>;
        storage_t storage_;
    };
}