#ifndef RIPTIDE_CPP_IS_MEMBER_TG_H
#define RIPTIDE_CPP_IS_MEMBER_TG_H

#include "CommonInc.h"
#include "operation_traits.h"
#include "Defs.h"
#include "missing_values.h"
#include "simple_span.h"

#define TBB_PREVIEW_CONCURRENT_HASH_MAP_EXTENSIONS 1
#define __TBB_NO_IMPLICIT_LINKAGE 1 // don't import tbb libs by default
#define __TBB_PREVIEW_MUTEXES 1
#include "oneapi/tbb/concurrent_hash_map.h"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/scalable_allocator.h"
#include "oneapi/tbb/rw_mutex.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/global_control.h"
#include "oneapi/tbb/flow_graph.h"
#include "oneapi/tbb.h"

#include "absl/container/flat_hash_map.h"

#include <cstdint>
#include <variant>
#include <type_traits>
#include <utility>
#include <thread>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>

namespace
{
    template <typename index_t>
    using data_span_t = std::pair<index_t, index_t>;

    static constexpr ptrdiff_t grainsize_v{ 1000000 };

    template <typename T>
    riptide_cpp::simple_span<T> simple_span_from_fixed_cstring(T const * const string, size_t const fixed_length)
    {
        size_t length{ 0 };
        while (length < fixed_length && string[length])
        {
            ++length;
        }
        return riptide_cpp::simple_span<T>{ string, length };
    }

    template <typename key_t>
    constexpr bool is_string_type()
    {
        return std::is_same_v<std::remove_const_t<key_t>, char> || std::is_same_v<std::remove_const_t<key_t>, wchar_t> ||
               std::is_same_v<std::remove_const_t<key_t>, riptide_cpp::simple_span<char>> ||
               std::is_same_v<std::remove_const_t<key_t>, riptide_cpp::simple_span<wchar_t>>;
    }
}

namespace riptide_cpp
{
    template <typename index_t>
    data_span_t<index_t> rangify_func(index_t const start_pos, index_t const num_elems, size_t const elem_length) noexcept
    {
        ptrdiff_t const range_size = (grainsize_v / elem_length) * elem_length;
        index_t end_pos{ start_pos + range_size >= num_elems ? static_cast<index_t>(num_elems * elem_length) :
                                                               static_cast<index_t>(start_pos + range_size) };
        return { start_pos, end_pos };
    }

    template <typename key_t>
    size_t our_hash_function(key_t value, size_t num_buckets = 1) noexcept
    {
        if constexpr (is_string_type<key_t>())
        {
            return num_buckets == 1 ? 0 : std::hash<key_t>{}(value);
        }
        else
        {
            return num_buckets == 1 ? 0 : static_cast<size_t>(value) % num_buckets;
        }
    }
}

enum struct hash_choice_t
{
    hash_linear,
    tbb,
};

extern DllExport hash_choice_t runtime_hash_choice;

namespace
{
    template <typename key_t, typename Enable = void>
    struct hash_details
    {
        ptrdiff_t index{ -1 };
        key_t value{};
        size_t hasher_index{};
    };

    template <typename key_t>
    struct hash_details<key_t, std::enable_if_t<std::is_same_v<std::remove_const_t<key_t>, char>>>
    {
        ptrdiff_t index{ -1 };
        riptide_cpp::simple_span<key_t> value{};
        size_t hasher_index{};
    };

    template <typename key_t>
    struct hash_details<key_t, std::enable_if_t<std::is_same_v<std::remove_const_t<key_t>, wchar_t>>>
    {
        ptrdiff_t index{ -1 };
        riptide_cpp::simple_span<key_t> value{};
        size_t hasher_index{};
    };

    template <typename key_t>
    using hash_details_container_t = std::vector<hash_details<key_t>>;
    template <typename key_t>
    using mt_hash_details_container_t = oneapi::tbb::concurrent_vector<hash_details<key_t>>;
    template <typename key_t>
    using segmented_hash_details_container_t = std::vector<mt_hash_details_container_t<key_t>>;

    using is_member_allowed_types_t =
        std::variant<uint8_t, uint16_t, uint32_t, uint64_t, float, double, int8_t, int16_t, int32_t, int64_t, char, wchar_t>;

// TODO: When moving the whole thread model to TBB, consider how to configure things like NUMA node, etc. HWLOC?
// These should be explicitly instantiated/shut-down in order to avoid strange deadlocks (see riptide_cpp#33).
#if 0
    inline static oneapi::tbb::global_control const global_limit{ oneapi::tbb::global_control::max_allowed_parallelism, 32 };
    inline static oneapi::tbb::task_arena::constraints const arena_setters{ oneapi::tbb::numa_node_id{ 0 }, 8 };
    inline static oneapi::tbb::task_arena arena{ arena_setters };
#endif

    template <typename key_t, typename index_t, typename enable = void>
    struct my_hasher_types
    {
        using map_key_t = key_t;

        using std_type = typename std::map<map_key_t, index_t>;
        using absl_type = typename absl::flat_hash_map<map_key_t, index_t>;
        using my_hasher_t = absl_type;
        using local_hashes_t = std::vector<my_hasher_t>;
    };

    template <typename key_t, typename index_t>
    struct my_hasher_types<key_t, index_t, std::enable_if_t<std::is_same_v<std::remove_const_t<key_t>, char>>>
    {
        using map_key_t = riptide_cpp::simple_span<key_t>;

        using std_type = typename std::map<map_key_t, index_t>;
        using absl_type = typename absl::flat_hash_map<map_key_t, index_t>;
        using my_hasher_t = absl_type;
        using local_hashes_t = std::vector<my_hasher_t>;
    };

    template <typename key_t, typename index_t>
    struct my_hasher_types<key_t, index_t, std::enable_if_t<std::is_same_v<std::remove_const_t<key_t>, wchar_t>>>
    {
        using map_key_t = riptide_cpp::simple_span<key_t>;

        using std_type = typename std::map<map_key_t, index_t>;
        using absl_type = typename absl::flat_hash_map<map_key_t, index_t>;
        using my_hasher_t = absl_type;
        using local_hashes_t = std::vector<my_hasher_t>;
    };

    template <typename key_t, typename index_t>
    struct hashing_graph
    {
        oneapi::tbb::flow::graph g{};
        size_t const num_elems;
        size_t const num_buckets;
        size_t const string_length;
        key_t const * start_p{};
        typename my_hasher_types<key_t, index_t>::local_hashes_t grouped_hashes{};

        index_t range_start{};
        bool has_ranger_run{ false };

        static constexpr size_t num_per_bucket_v = 1024;

        using buckets_t = std::vector<hash_details_container_t<key_t>>;

        struct ranging_node_t : public oneapi::tbb::flow::input_node<data_span_t<index_t>>
        {
            template <typename body_t>
            ranging_node_t(oneapi::tbb::flow::graph & graph, body_t body)
                : oneapi::tbb::flow::input_node<data_span_t<index_t>>(graph, body)
            {
            }
        };

        struct splitter_node_t : public oneapi::tbb::flow::function_node<data_span_t<index_t>, hash_details_container_t<key_t>>
        {
            template <typename body_t>
            splitter_node_t(oneapi::tbb::flow::graph & graph, size_t concurrency, body_t body)
                : oneapi::tbb::flow::function_node<data_span_t<index_t>, hash_details_container_t<key_t>>(graph, concurrency, body)
            {
            }
        };

        struct storing_node_t : public oneapi::tbb::flow::function_node<hash_details_container_t<key_t>>
        {
            template <typename body_t>
            storing_node_t(oneapi::tbb::flow::graph & graph, size_t concurrency, body_t body)
                : oneapi::tbb::flow::function_node<hash_details_container_t<key_t>>(graph, concurrency, body)
            {
            }
        };

        std::vector<splitter_node_t> splitters{};
        std::vector<storing_node_t> storers{};

        hashing_graph(key_t const * data_p, size_t const num_elements, size_t const data_size, size_t const buckets = 0)
            : num_elems{ num_elements }
            , num_buckets{ buckets ? buckets : (num_elements + grainsize_v - 1) / grainsize_v }
            , string_length{ data_size }
            , start_p{ data_p }
        {
            for (size_t i{}; i != num_buckets; ++i)
            {
                grouped_hashes.push_back(typename my_hasher_types<key_t, index_t>::my_hasher_t{});
                grouped_hashes[i].reserve(num_elems / num_buckets + 1);
            }
        }

        void operator()()
        {
            auto data_rangifier = [&](oneapi::tbb::flow_control & fc) -> data_span_t<index_t>
            {
                if (has_ranger_run && range_start == 0)
                {
                    fc.stop();
                    return {};
                }
                has_ranger_run = true;
                auto [start_pos, end_pos] = riptide_cpp::rangify_func(range_start, static_cast<index_t>(num_elems), string_length);
                range_start = (start_pos > (end_pos - grainsize_v + 1) ? 0 : end_pos);

                return data_span_t<index_t>{ start_pos, end_pos };
            };

            auto storing_func = [&](hash_details_container_t<key_t> const & data) -> oneapi::tbb::flow::continue_msg
            {
                for (hash_details<key_t> const & value : data)
                {
                    if (value.index != -1)
                    {
                        grouped_hashes[value.hasher_index].insert({ value.value, value.index });
                    }
                }
                return oneapi::tbb::flow::continue_msg{};
            };

            auto splitter_func = [&](data_span_t<index_t> data_span) -> hash_details_container_t<key_t>
            {
                if (data_span == data_span_t<index_t>{})
                {
                    return {};
                }

                buckets_t buckets(num_buckets);

                for (index_t i{ data_span.first }; i != data_span.second; ++i)
                {
                    size_t target_hasher{};
                    if constexpr (is_string_type<key_t>())
                    {
                        if (i % string_length)
                        {
                            continue;
                        }
                        riptide_cpp::simple_span<key_t> temp{ simple_span_from_fixed_cstring(&start_p[i], string_length) };
                        target_hasher = riptide_cpp::our_hash_function(temp, num_buckets);

                        buckets[target_hasher].push_back({ i / static_cast<index_t>(string_length), temp, target_hasher });
                    }
                    else
                    {
                        target_hasher = riptide_cpp::our_hash_function(start_p[i], num_buckets);

                        buckets[target_hasher].push_back({ i, start_p[i], target_hasher });
                    }
                    if (buckets[target_hasher].size() == num_per_bucket_v)
                    {
                        storers[target_hasher].try_put(buckets[target_hasher]);
                        buckets[target_hasher].clear();
                    }
                }

                for (auto const & bucket : buckets)
                {
                    if (bucket[0].index != -1)
                    {
                        storers[bucket[0].hasher_index].try_put(bucket);
                    }
                }

                return {};
            };

            for (size_t i{}; i != num_buckets; ++i)
            {
                storers.push_back(storing_node_t(g, 1, storing_func));
            }

            ranging_node_t ranger(g, data_rangifier);
            splitter_node_t splitter(g, num_buckets, splitter_func);

            oneapi::tbb::flow::make_edge(ranger, splitter);
            for (size_t i{}; i != num_buckets; ++i)
            {
                oneapi::tbb::flow::make_edge(splitter, storers[i]);
            }

            ranger.activate();
            g.wait_for_all();
        }

        index_t find(typename my_hasher_types<key_t, index_t>::map_key_t const * key, size_t bucket = 0) const noexcept
        {
            typename my_hasher_types<key_t, index_t>::my_hasher_t const & hashes{ grouped_hashes[bucket] };
            typename my_hasher_types<key_t, index_t>::my_hasher_t::const_iterator found = hashes.find(*key);

            return (found != std::end(hashes) && riptide::invalid_for_type<index_t>::is_valid(found->second)) ?
                       found->second :
                       riptide::invalid_for_type<index_t>::value;
        }
    };

    template <typename key_t, typename index_t>
    struct is_member_pre_hash_needles
    {
        key_t const * needles_base_p{};
        size_t num_needles{};
        size_t num_hashers{};
        size_t string_length{};
        segmented_hash_details_container_t<key_t> & hashes;

        is_member_pre_hash_needles(size_t const needles, key_t const * needles_p, size_t data_len, size_t const hashers,
                                   segmented_hash_details_container_t<key_t> & hasher_ref)
            : needles_base_p{ needles_p }
            , num_needles{ needles }
            , num_hashers{ hashers }
            , string_length{ data_len }
            , hashes{ hasher_ref }
        {
        }

        void operator()(oneapi::tbb::blocked_range<size_t> const & slice) const
        {
            // Input is a blocked_range of the needles space
            // Prescan the needles and populate a set of vectors by task, containing the value we hash, and it's index in the
            // needles Then iterate over those containers, at which point we should be looking at local-only values, and build
            // the output arrays

            for (size_t i{ slice.begin() }; i != slice.end(); ++i)
            {
                if constexpr (is_string_type<key_t>())
                {
                    typename my_hasher_types<key_t, index_t>::map_key_t temp{ &needles_base_p[i * string_length], string_length };
                    size_t hasher_index{ riptide_cpp::our_hash_function(temp, num_hashers) };

                    hashes[hasher_index].push_back(
                        hash_details<key_t>{ static_cast<ptrdiff_t>(i),
                                             typename my_hasher_types<key_t, index_t>::map_key_t{ simple_span_from_fixed_cstring(
                                                 &needles_base_p[i * string_length], string_length) },
                                             hasher_index });
                }
                else
                {
                    size_t hasher_index{ riptide_cpp::our_hash_function(needles_base_p[i], num_hashers) };

                    hashes[hasher_index].push_back(
                        hash_details<key_t>{ static_cast<ptrdiff_t>(i), needles_base_p[i], hasher_index });
                }
            }
        }
    };

    template <typename key_t, typename index_t>
    struct is_member_lookup_needles
    {
        mt_hash_details_container_t<key_t> const * hashers_base_p{};
        index_t * output_base_p{};
        int8_t * bool_base_p{};
        size_t num_hashers{};
        hashing_graph<key_t, index_t> * hasher_p{};

        is_member_lookup_needles(mt_hash_details_container_t<key_t> const * hashers_p, index_t * output_p, int8_t * bool_p,
                                 size_t const hashers, hashing_graph<key_t, index_t> * hashed_keys_p)
            : hashers_base_p{ hashers_p }
            , output_base_p{ output_p }
            , bool_base_p{ bool_p }
            , num_hashers{ hashers }
            , hasher_p{ hashed_keys_p }
        {
        }

        void operator()(oneapi::tbb::blocked_range<size_t> const & slice) const
        {
            for (size_t i{ slice.begin() }; i != slice.end(); ++i)
            {
                mt_hash_details_container_t<key_t> const * my_needles_p{ hashers_base_p + i };

                for (hash_details<key_t> const & d : *my_needles_p)
                {
                    static_assert(std::is_signed_v<index_t>, "The index value needs to be signed");
                    index_t const found_at = hasher_p->find(&d.value, i);
                    if (found_at >= 0)
                    {
                        *(output_base_p + d.index) = found_at;
                        *(bool_base_p + d.index) = 1;
                    }
                    else
                    {
                        *(output_base_p + d.index) = riptide::invalid_for_type<index_t>::value;
                        *(bool_base_p + d.index) = 0;
                    }
                }
            }
        }
    };
}

template <typename index_t>
inline void is_member_tg(size_t const needles_size, char const * needles_p, size_t const needles_type_size,
                         size_t const haystack_size, char const * haystack_p, size_t const haystack_type_size, index_t * output_p,
                         int8_t * bool_out_p, is_member_allowed_types_t const sample_value, int max_cpus = 8)
{
    // TODO: global_limit belongs with the existing app-level thread startup/shutdown logic.
    oneapi::tbb::global_control const global_limit{ oneapi::tbb::global_control::max_allowed_parallelism, 32 };

    oneapi::tbb::task_arena::constraints const local_arena_setters{ oneapi::tbb::numa_node_id{ 0 }, max_cpus };
    oneapi::tbb::task_arena local_arena{ local_arena_setters };

    std::visit(
        [=, &local_arena](auto && arg)
        {
            using key_t = std::decay_t<decltype(arg)>;

            size_t num_slices = (haystack_size + grainsize_v - 1) / grainsize_v;

            local_arena.execute(
                [=]()
                {
                    hashing_graph<key_t, index_t> hasher{ reinterpret_cast<key_t const *>(haystack_p), haystack_size,
                                                          haystack_type_size, num_slices };
                    hasher();

                    segmented_hash_details_container_t<key_t> index_hashes{ num_slices };
                    key_t const * typed_needles_p{ reinterpret_cast<key_t const *>(needles_p) };

                    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, needles_size),
                                              is_member_pre_hash_needles<key_t, index_t>(
                                                  needles_size, typed_needles_p, needles_type_size, num_slices, index_hashes),
                                              oneapi::tbb::static_partitioner{});

                    oneapi::tbb::parallel_for(
                        oneapi::tbb::blocked_range<size_t>(0, num_slices),
                        is_member_lookup_needles(index_hashes.data(), output_p, bool_out_p, num_slices, &hasher),
                        oneapi::tbb::simple_partitioner{});
                });
        },
        sample_value);
}

template <typename data_trait_t, typename out_t>
inline void is_member_shim(size_t const needles_size, char const * needles_p, size_t const needles_type_size,
                           size_t const haystack_size, char const * haystack_p, size_t const haystack_type_size, out_t * output_p,
                           int8_t * bool_out_p, data_trait_t const * data_p, int max_cpus = 8)
{
    if (data_p)
    {
        using T = typename data_trait_t::data_type;
        T const sample_value{};
        is_member_tg<out_t>(needles_size, needles_p, needles_type_size, haystack_size, haystack_p, haystack_type_size, output_p,
                            bool_out_p, sample_value, max_cpus);
    }
}

template <typename variant_t, typename out_t, size_t... Is>
inline void is_member_for_type(size_t const needles_size, char const * needles_p, size_t const needles_type_size,
                               size_t const haystack_size, char const * haystack_p, size_t const haystack_type_size,
                               out_t * output_p, int8_t * bool_out_p, variant_t const data_type_traits, int max_cpus = 8,
                               std::index_sequence<Is...> const = std::make_index_sequence<std::variant_size_v<variant_t>>{})
{
    (is_member_shim(needles_size, needles_p, needles_type_size, haystack_size, haystack_p, haystack_type_size, output_p,
                    bool_out_p, std::get_if<Is>(&data_type_traits), max_cpus),
     ...);
}

#endif
