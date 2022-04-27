#ifndef RIPTIDE_CPP_FLAT_HASH_MAP_H
    #define RIPTIDE_CPP_FLAG_HASH_MAP_H

    #include "CommonInc.h"
    #include "overload.h"
    #include "absl/container/flat_hash_map.h"
#include "oneapi/tbb/concurrent_hash_map.h"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"

    #include <cstdint>
    #include <variant>
    #include <type_traits>
    #include <utility>
#include <memory_resource>
#include <vector>

    #if defined(_WIN32) && ! defined(__GNUC__)
        #define dll_export __declspec(dllexport)
    #else
        #define dll_export
    #endif

namespace
{
    using is_member_allowed_types_t =
        std::variant<uint8_t, uint16_t, uint32_t, uint64_t, float, double, int8_t, int16_t, int32_t, int64_t>;

    template< typename KeyT >
    struct hash_function
    {
        size_t const operator()(KeyT const & key) const
        {
            if constexpr( std::is_integral_v<KeyT>)
            {
                return static_cast<size_t const>(key) % 3145739;
            }
            else
            {
                size_t retval{};
                if constexpr( sizeof( KeyT ) > sizeof(size_t))
                {
                    memcpy( &retval, &key, sizeof(size_t) );
                    return retval % 3145739;
                }
                else
                {
                    memcpy(&retval, &key, sizeof(KeyT));
                    return retval % 3245739;
                }
            }
        }
    };

    std::vector< char > backing(128ULL * 1024ULL * 1024ULL);
}

template <typename T>
struct type_deducer;


template <typename KeyT, typename IndexT>
struct fhm_hasher
{
//    std::pmr::monotonic_buffer_resource mono{ backing.data(), backing.capacity() };
//    std::pmr::unsynchronized_pool_resource pool{ &mono };
//    std::pmr::memory_resource * resource{&mono};
    
    oneapi::tbb::concurrent_hash_map<KeyT, IndexT> hasher;
//    absl::flat_hash_map<KeyT, IndexT> hasher{};
//    absl::flat_hash_map<KeyT, IndexT,
//                        absl::container_internal::hash_default_hash<KeyT>,
//                        absl::container_internal::hash_default_eq<KeyT>,
//                        std::pmr::polymorphic_allocator<std::pair<KeyT, IndexT>>> hasher{resource};
//    std::pmr::unordered_map<KeyT, IndexT, hash_function<KeyT>> hasher{resource};
//    std::unordered_map<KeyT, IndexT> hasher{};
    KeyT const * data_series_p{};

    dll_export fhm_hasher() {}

    dll_export void make_hash(size_t array_size, char const * hash_list_p, size_t hint_size)
    {
        do_make_hash(array_size, reinterpret_cast<KeyT const *>(hash_list_p), hint_size);
    }

    void do_make_hash(size_t array_size, KeyT const * hash_list_p, size_t hint_size)
    {
        if (not hint_size)
        {
            hint_size = array_size;
        }

//        hasher.reserve(hint_size);

        data_series_p = hash_list_p;

        for (IndexT i{ 0 }; i != hint_size; ++i)
        {
//            hasher.emplace(hash_list_p[i], i);
            typename oneapi::tbb::concurrent_hash_map<KeyT, IndexT>::accessor a;
            hasher.insert(a, hash_list_p[i]);
            a->second = i;
        }
    }

    int64_t find(KeyT const * key) const noexcept
    {
        typename oneapi::tbb::concurrent_hash_map<KeyT, IndexT>::const_accessor finder;
        bool found = hasher.find(finder, *key);
        //        bool found = hasher.contains(*key);
        //        bool found = hasher.find(*key) != hasher.end();

        //return found ? hasher.at(*key) : -1ll;
        return found ? finder->second : -1ll;
    }
};

template <typename KeyT, typename IndexT>
struct is_member_check
{
    fhm_hasher<KeyT, IndexT> mutable hash{};

    int operator()(size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p, IndexT * output_p,
                   int8_t * bool_out_p)
    {
        KeyT const * typed_needles_p{ reinterpret_cast<KeyT const *>(needles_p) };

        hash.make_hash(haystack_size, haystack_p, 0);

        for (ptrdiff_t elem{ 0 }; elem != needles_size; ++elem)
        {
            IndexT found_at = static_cast<IndexT>(hash.find(typed_needles_p + elem));
            if (found_at >= 0)
            {
                *(output_p + elem) = found_at;
                *(bool_out_p + elem) = 1;
            }
            else
            {
                *(output_p + elem) = std::numeric_limits<IndexT>::min();
                *(bool_out_p + elem) = 0;
            }
        }
        return 0;
    }
};

template<typename out_t>
dll_export inline int is_member(size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p,
                         out_t * output_p, int8_t * bool_out_p, is_member_allowed_types_t sample_value)
{
    auto hasher_calls = overload{
        [=](uint8_t) -> int
        { return is_member_check<uint8_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](uint16_t) -> int
        { return is_member_check<uint16_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](uint32_t) -> int
        { return is_member_check<uint32_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](uint64_t) -> int
        { return is_member_check<uint64_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](int8_t) -> int
        { return is_member_check<int8_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](int16_t) -> int
        { return is_member_check<int16_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](int32_t) -> int
        { return is_member_check<int32_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](int64_t) -> int
        { return is_member_check<int64_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](float) -> int
        { return is_member_check<float, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](double) -> int
        { return is_member_check<double, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
    };

    return std::visit(hasher_calls, sample_value);
}

template <typename data_trait_t, typename out_t>
inline int is_member_shim(size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p,
                          out_t * output_p, int8_t * bool_out_p, data_trait_t const * data_p)
{
    if ( data_p )
    {
        using T = typename data_trait_t::data_type;
        T const sample_value{};
        return is_member<out_t>( needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p, sample_value );
    }
    else
    {
        return 0;
    }
}

template< typename variant_t, typename out_t, size_t... Is>
dll_export inline int is_member_for_type( size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p,
                                   out_t * output_p, int8_t * bool_out_p, variant_t data_type_traits, std::index_sequence<Is...>)
{
    return (is_member_shim(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p, std::get_if<Is>(&data_type_traits)) + ...);
}
#endif
