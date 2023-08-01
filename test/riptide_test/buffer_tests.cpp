#include "any_const_buffer.h"
#include "buffer.h"
#include "mem_buffer.h"

#include "ut_extensions.h"

#include "boost/ut.hpp"

#include <optional>
#include <tuple>
#include <type_traits>

using namespace riptide_utility::internal;
using namespace boost::ut;
using boost::ut::suite;

namespace
{
    static_assert(std::is_nothrow_default_constructible_v<const_buffer<int8_t>>);
    static_assert(std::is_nothrow_constructible_v<const_buffer<int8_t>, int8_t const *, size_t>);
    static_assert(std::is_nothrow_constructible_v<const_buffer<int8_t>, int8_t *, size_t>);
    static_assert(std::is_nothrow_copy_assignable_v<const_buffer<int8_t>>);
    static_assert(std::is_nothrow_move_assignable_v<const_buffer<int8_t>>);

    static_assert(std::is_nothrow_default_constructible_v<mutable_buffer<int8_t>>);
    static_assert(! std::is_constructible_v<mutable_buffer<int8_t>, int8_t const *, size_t>);
    static_assert(std::is_nothrow_constructible_v<mutable_buffer<int8_t>, int8_t *, size_t>);
    static_assert(std::is_nothrow_copy_assignable_v<mutable_buffer<int8_t>>);
    static_assert(std::is_nothrow_move_assignable_v<mutable_buffer<int8_t>>);
    static_assert(std::is_nothrow_assignable_v<const_buffer<int8_t>, mutable_buffer<int8_t>>);
    static_assert(std::is_nothrow_constructible_v<const_buffer<int8_t>, mutable_buffer<int8_t> &&>);

    static_assert(std::is_nothrow_default_constructible_v<mem_buffer<int8_t>>);
    static_assert(std::is_constructible_v<mem_buffer<int8_t>, size_t>);
    static_assert(std::is_constructible_v<mem_buffer<int8_t>, size_t, std::pmr::memory_resource *>);
    static_assert(! std::is_copy_assignable_v<mem_buffer<int8_t>>);
    static_assert(std::is_nothrow_move_assignable_v<mem_buffer<int8_t>>);

#if defined(_MSC_VER) && _MSC_VER < 1936
    // MSVC < 1936 seems to lose noexcept.
    static_assert(std::is_assignable_v<const_buffer<int8_t>, mem_buffer<int8_t> const &>);
    static_assert(std::is_assignable_v<const_buffer<int8_t>, mem_buffer<int8_t> &&>);
    static_assert(std::is_constructible_v<const_buffer<int8_t>, mem_buffer<int8_t> const &>);
    static_assert(std::is_constructible_v<const_buffer<int8_t>, mem_buffer<int8_t> &&>);
    static_assert(std::is_assignable_v<mutable_buffer<int8_t>, mem_buffer<int8_t> &&>);
    static_assert(std::is_constructible_v<mutable_buffer<int8_t>, mem_buffer<int8_t> &&>);
#else
    static_assert(std::is_nothrow_assignable_v<const_buffer<int8_t>, mem_buffer<int8_t> const &>);
    static_assert(std::is_nothrow_assignable_v<const_buffer<int8_t>, mem_buffer<int8_t> &&>);
    static_assert(std::is_nothrow_constructible_v<const_buffer<int8_t>, mem_buffer<int8_t> const &>);
    static_assert(std::is_nothrow_constructible_v<const_buffer<int8_t>, mem_buffer<int8_t> &&>);
    static_assert(std::is_nothrow_assignable_v<mutable_buffer<int8_t>, mem_buffer<int8_t> &&>);
    static_assert(std::is_nothrow_constructible_v<mutable_buffer<int8_t>, mem_buffer<int8_t> &&>);
#endif

    enum class buffer_kind
    {
        CONST_BUFFER,
        MUTABLE_BUFFER,
        MEM_BUFFER,
    };

    template <typename T, typename V>
    struct to_kind;

    template <typename V>
    struct to_kind<const_buffer<V>, V>
    {
        static constexpr buffer_kind value{ buffer_kind::CONST_BUFFER };
    };

    template <typename V>
    struct to_kind<mutable_buffer<V>, V>
    {
        static constexpr buffer_kind value{ buffer_kind::MUTABLE_BUFFER };
    };

    template <typename V>
    struct to_kind<mem_buffer<V>, V>
    {
        static constexpr buffer_kind value{ buffer_kind::MEM_BUFFER };
    };

    template <typename T, typename V>
    inline constexpr buffer_kind to_kind_v{ to_kind<T, V>::value };

    template <typename V, buffer_kind K>
    struct to_buffer;

    template <typename V>
    struct to_buffer<V, buffer_kind::CONST_BUFFER>
    {
        using type = const_buffer<V>;
    };

    template <typename V>
    struct to_buffer<V, buffer_kind::MUTABLE_BUFFER>
    {
        using type = mutable_buffer<V>;
    };

    template <typename V>
    struct to_buffer<V, buffer_kind::MEM_BUFFER>
    {
        using type = mem_buffer<V>;
    };

    template <typename V, buffer_kind K>
    using to_buffer_t = typename to_buffer<V, K>::type;

    template <buffer_kind K>
    using buffer_kind_t = std::integral_constant<buffer_kind, K>;

    using SupportedTypes = std::tuple<buffer_kind_t<buffer_kind::CONST_BUFFER>, buffer_kind_t<buffer_kind::MUTABLE_BUFFER>,
                                      buffer_kind_t<buffer_kind::MEM_BUFFER>>;

    struct buffer_ctor_tester
    {
        template <typename KindT>
        void operator()()
        {
            using buffer_type = to_buffer_t<int, KindT::value>;
            if constexpr (std::is_same_v<buffer_type, mem_buffer<int>>)
            {
                buffer_type buf{};
                buffer_type buf2{ std::move(buf) };
                typed_expect<KindT>(buf2.data() == buf.data());
                typed_expect<KindT>(buf2.size() == buf.size());
            }
            else if constexpr (std::is_same_v<buffer_type, const_buffer<int>>)
            {
                buffer_type buf{ reinterpret_cast<int const *>(0x1ee7feedULL), 246U };
                buffer_type buf2{ buf };
                typed_expect<KindT>(buf2.data() == buf.data());
                typed_expect<KindT>(buf2.size() == buf.size());
            }
            else if constexpr (std::is_same_v<buffer_type, mutable_buffer<int>>)
            {
                buffer_type buf{ reinterpret_cast<int *>(0xdeadbeefULL), 12345U };
                buffer_type buf2{ buf };
                typed_expect<KindT>(buf2.data() == buf.data());
                typed_expect<KindT>(buf2.size() == buf.size());
            }
            else
            {
                throw std::runtime_error("Unexpected type");
            }
        }
    };

    struct buffer_empty_tester
    {
        template <typename KindT>
        void operator()()
        {
            using buffer_type = to_buffer_t<int, KindT::value>;
            buffer_type buf{};
            typed_expect<KindT>(buf.data() == nullptr);
            typed_expect<KindT>(buf.size() == 0);
        }
    };

    struct any_const_buffer_ctor_tester
    {
        template <typename KindT>
        void operator()()
        {
            using buffer_type = to_buffer_t<int, KindT::value>;
            any_const_buffer<int> any{ buffer_type{} };
            std::optional<buffer_kind> actual;
            any.visit(
                [&actual](auto && arg)
                {
                    using actual_t = std::decay_t<decltype(arg)>;
                    actual = to_kind_v<actual_t, int>;
                });
            typed_expect<KindT>(actual.value() == KindT::value)
                << "expected:" << (int)KindT::value << "actual:" << (int)actual.value();
        }
    };

    suite invalids_compatibility = []
    {
        "buffer_ctor"_test = buffer_ctor_tester{} | SupportedTypes{};
        "buffer_empty"_test = buffer_empty_tester{} | SupportedTypes{};

        "any_const_buffer_ctor"_test = any_const_buffer_ctor_tester{} | SupportedTypes{};
    };
}
