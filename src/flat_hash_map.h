#ifndef RIPTIDE_CPP_FLAT_HASH_MAP_H
#define RIPTIDE_CPP_FLAG_HASH_MAP_H

#include "CommonInc.h"
#include "overload.h"
#include "absl/container/flat_hash_map.h"

#include <cstdint>
#include <variant>
#include <type_traits>
#include <utility>

#if defined(_WIN32) && ! defined(__GNUC__)
#define dll_export __declspec(dllexport)
#else
#define dll_export
#endif

namespace
{
    using is_member_allowed_types_t = std::variant< uint8_t, uint16_t, uint32_t, uint64_t, float, double, int8_t, int16_t, int32_t, int64_t>;
}

template< typename T >
struct type_deducer;

template <typename KeyT>
struct fhm_hasher
{
    absl::flat_hash_map<KeyT, int64_t> hasher{};
    KeyT const * data_series_p{};

    dll_export fhm_hasher() {}

    dll_export void make_hash(size_t array_size, char const * hash_list_p, size_t hint_size)
    {
        do_make_hash(array_size, reinterpret_cast< KeyT const * >(hash_list_p), hint_size);
    }

    void do_make_hash(size_t array_size, KeyT const * hash_list_p, size_t hint_size)
    {
        if (not hint_size)
        {
            hint_size = array_size;
        }

        hasher.reserve(hint_size);

        data_series_p = hash_list_p;
        
        for( size_t i{0}; i != hint_size; ++i )
        {
            hasher.emplace(hash_list_p[i], i);
        }
    }

    int64_t find( KeyT const * key  ) const noexcept
    {
        bool found = hasher.contains(*key);
        
        return found ? hasher.at( *key ): -1ll;        
    }
};

template< typename KeyT >
struct is_member_check
{
    fhm_hasher<KeyT> hash{};

    int operator()(size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p, int64_t * output_p, int8_t * bool_out_p)
    {
//        memset( output_p, -1, sizeof( KeyT ) * needles_size );
//        memset( bool_out_p, 0, needles_size );
        
        KeyT const * typed_needles_p{ reinterpret_cast< KeyT const *>(needles_p) };
        
        hash.make_hash(haystack_size, haystack_p, 0);

        for( ptrdiff_t elem{0}; elem != needles_size; ++elem )
        {
            int64_t found_at = hash.find(typed_needles_p + elem);
            if ( found_at >= 0 )
            {
                *(output_p + elem) = found_at;
                *(bool_out_p + elem) = 1;
            }
            else
            {
                *(output_p + elem) = -1;
                *(bool_out_p + elem) = 0;
            }
        }
        return 0;
    }
};

dll_export int
is_member(size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p, int64_t * output_p,
          int8_t * bool_out_p, is_member_allowed_types_t sample_value)
{
    auto hasher_calls = overload{
        [=](uint8_t)->int{ return is_member_check<uint8_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](uint16_t)->int{ return is_member_check<uint16_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](uint32_t)->int { return is_member_check<uint32_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](uint64_t)->int{ return is_member_check<uint64_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](int8_t)->int{ return is_member_check<int8_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](int16_t)->int{ return is_member_check<int16_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](int32_t)->int{ return is_member_check<int32_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](int64_t)->int{ return is_member_check<int64_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](float)->int{ return is_member_check<float>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
        [=](double)->int{ return is_member_check<double>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p); },
    };

    return std::visit(hasher_calls, sample_value);
}

#endif
