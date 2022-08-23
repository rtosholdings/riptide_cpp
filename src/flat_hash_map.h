#ifndef RIPTIDE_CPP_FLAT_HASH_MAP_H
    #define RIPTIDE_CPP_FLAT_HASH_MAP_H

    #include "CommonInc.h"
    #include "Defs.h"

    #include "absl/container/flat_hash_map.h"
    #define __TBB_NO_IMPLICIT_LINKAGE 1 // don't import tbb libs by default
    #include "oneapi/tbb/concurrent_hash_map.h"
    #include "oneapi/tbb/blocked_range.h"
    #include "oneapi/tbb/parallel_for.h"

    #include <cstdint>
    #include <variant>
    #include <type_traits>
    #include <utility>

namespace
{
    using is_member_allowed_types_t =
        std::variant<unsigned char, uint16_t, uint32_t, uint64_t, float, double, char, int16_t, int32_t, int64_t>;

    template<typename HasherT, typename KeyT, typename IndexT>
    static constexpr bool is_tbb_v = std::is_same_v<HasherT, typename oneapi::tbb::concurrent_hash_map<KeyT, IndexT>>;

    template<typename HasherT, typename KeyT, typename IndexT>
    static constexpr bool is_absl_v = std::is_same_v<HasherT, typename absl::flat_hash_map<KeyT, IndexT>>;

    template<typename HasherT, typename KeyT, typename IndexT>
    static constexpr bool is_stl_v = std::is_same_v<HasherT, typename std::unordered_map<KeyT, IndexT>>;
}

enum struct hash_choice_t
{
    hash_linear,
    tbb,
    absl,
    stl
};

extern DllExport hash_choice_t runtime_hash_choice;

template <typename KeyT, typename IndexT>
struct fhm_hasher
{
    oneapi::tbb::concurrent_hash_map<KeyT, IndexT> tbb_hasher;
    absl::flat_hash_map<KeyT, IndexT> absl_hasher{};
    std::unordered_map<KeyT, IndexT> std_hasher{};
    KeyT const * data_series_p{};

    DllExport fhm_hasher() {}

    DllExport void make_hash(size_t array_size, char const * hash_list_p, size_t hint_size)
    {
        switch (runtime_hash_choice)
        {
        case hash_choice_t::tbb:
            do_make_hash(array_size, reinterpret_cast<KeyT const *>(hash_list_p), hint_size, tbb_hasher);
            break;
        case hash_choice_t::absl:
            do_make_hash(array_size, reinterpret_cast<KeyT const *>(hash_list_p), hint_size, absl_hasher);
            break;
        case hash_choice_t::stl:
            do_make_hash(array_size, reinterpret_cast<KeyT const *>(hash_list_p), hint_size, std_hasher);
            break;
        default:
        case hash_choice_t::hash_linear:
            throw std::runtime_error("RT_NEW_HASH should be 1, 2, or 3");
        }
    }

    DllExport void clear_all()
    {
        tbb_hasher.clear();
        absl_hasher.clear();
        std_hasher.clear();
    }

    template <typename hasher_t>
    void do_make_hash(size_t array_size, KeyT const * hash_list_p, size_t hint_size, hasher_t & hasher)
    {
        if (not hint_size)
        {
            hint_size = array_size;
        }

        if constexpr (not is_tbb_v<hasher_t, KeyT, IndexT>)
        {
            hasher.reserve(hint_size);
        }

        data_series_p = hash_list_p;

        using LoopT = std::conditional_t<std::is_unsigned_v<IndexT>, IndexT, std::make_unsigned_t<IndexT>>;

        for (LoopT i(0); i != hint_size; ++i)
        {
            if constexpr (not is_tbb_v<hasher_t, KeyT, IndexT>)
            {
                hasher.emplace(hash_list_p[i], i);
            }
            else
            {
                typename oneapi::tbb::concurrent_hash_map<KeyT, IndexT>::accessor a;
                hasher.insert(a, hash_list_p[i]);
                a->second =
                IndexT((reinterpret_cast<char const *>(&hash_list_p[i]) - reinterpret_cast<char const *>(hash_list_p)) /
                       sizeof(KeyT));
            }
        }
    }

    template <typename hasher_t>
    int64_t find(KeyT const * key) const noexcept
    {
        if constexpr (is_tbb_v<hasher_t, KeyT, IndexT>)
        {
            typename oneapi::tbb::concurrent_hash_map<KeyT, IndexT>::const_accessor finder;
            bool found = tbb_hasher.find(finder, *key);

            return found ? finder->second : -1ll;
        }
        if constexpr (is_absl_v<hasher_t, KeyT, IndexT>)
        {
            bool found = absl_hasher.contains(*key);
            return found ? absl_hasher.at(*key) : -1ll;
        }
        if constexpr (is_stl_v<hasher_t, KeyT, IndexT>)
        {
            bool found = std_hasher.find(*key) != std_hasher.end();
            return found ? std_hasher.at(*key) : -1ll;
        }
    }
};

template <typename KeyT, typename IndexT>
struct is_member_check
{
    fhm_hasher<KeyT, IndexT> mutable hash{};

    void operator()(size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p, IndexT * output_p,
                    int8_t * bool_out_p)
    {
        KeyT const * typed_needles_p{ reinterpret_cast<KeyT const *>(needles_p) };

        hash.make_hash(haystack_size, haystack_p, 0);

        switch (runtime_hash_choice)
        {
        case hash_choice_t::tbb:
            for (size_t elem{ 0 }; elem != needles_size; ++elem)
            {
                IndexT found_at = static_cast<IndexT>(
                    hash.template find<typename oneapi::tbb::concurrent_hash_map<KeyT, IndexT>>(typed_needles_p + elem));
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
            break;
        case hash_choice_t::absl:
            for (size_t elem{ 0 }; elem != needles_size; ++elem)
            {
                IndexT found_at =
                    static_cast<IndexT>(hash.template find<typename absl::flat_hash_map<KeyT, IndexT>>(typed_needles_p + elem));
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
            break;
        case hash_choice_t::stl:
            for (size_t elem{ 0 }; elem != needles_size; ++elem)
            {
                IndexT found_at =
                    static_cast<IndexT>(hash.template find<typename std::unordered_map<KeyT, IndexT>>(typed_needles_p + elem));
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
            break;
        default:
        case hash_choice_t::hash_linear:
            throw(std::runtime_error("RT_NEW_HASH / runtime_hash_choice invalid!"));
        }
    }
};

template <typename out_t>
DllExport inline void is_member(size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p,
                                 out_t * output_p, int8_t * bool_out_p, is_member_allowed_types_t sample_value)
{
    std::visit(
        [=](auto && arg)
        {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<unsigned char, T>)
            {
                is_member_check<unsigned char, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
            if constexpr (std::is_same_v<char, T>)
            {
                is_member_check<char, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
            if constexpr (std::is_same_v<int16_t, T>)
            {
                is_member_check<int16_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
            if constexpr (std::is_same_v<uint16_t, T>)
            {
                is_member_check<uint16_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
            if constexpr (std::is_same_v<int32_t, T>)
            {
                is_member_check<int32_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
            if constexpr (std::is_same_v<uint32_t, T>)
            {
                is_member_check<uint32_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
            if constexpr (std::is_same_v<int64_t, T>)
            {
                is_member_check<int64_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
            if constexpr (std::is_same_v<uint64_t, T>)
            {
                is_member_check<uint64_t, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
            if constexpr (std::is_same_v<float, T>)
            {
                is_member_check<float, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
            if constexpr (std::is_same_v<double, T>)
            {
                is_member_check<double, out_t>{}(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p);
            }
        },
        sample_value);
}

template <typename data_trait_t, typename out_t>
inline void is_member_shim(size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p,
                           out_t * output_p, int8_t * bool_out_p, data_trait_t const * data_p)
{
    if (data_p)
    {
        using T = typename data_trait_t::data_type;
        T const sample_value{};
        is_member<out_t>(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p, sample_value);
    }
}

template <typename variant_t, typename out_t, size_t... Is>
DllExport inline void is_member_for_type(size_t needles_size, char const * needles_p, size_t haystack_size,
                                          char const * haystack_p, out_t * output_p, int8_t * bool_out_p,
                                          variant_t data_type_traits, std::index_sequence<Is...>)
{
    (is_member_shim(needles_size, needles_p, haystack_size, haystack_p, output_p, bool_out_p, std::get_if<Is>(&data_type_traits)),
     ...);
}
#endif
