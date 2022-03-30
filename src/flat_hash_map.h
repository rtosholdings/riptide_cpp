#ifndef RIPTIDE_CPP_FLAT_HASH_MAP_H
    #define RIPTIDE_CPP_FLAG_HASH_MAP_H

    #include "CommonInc.h"
    #include "absl/container/flat_hash_map.h"

    #include <cstdint>

    #if defined(_WIN32) && ! defined(__GNUC__)
        #define dll_export __declspec(dllexport)
    #else
        #define dll_export
    #endif

namespace
{
    enum class HASH_MODE
    {
        HASH_MODE_PRIME = 1,
        HASH_MODE_MASK = 2
    };
}

dll_export int is_member(size_t needles_size, char const * needles_p, size_t haystack_size, char const * haystack_p,
                         int64_t * output_p, int8_t * bool_out_p, int32_t size_type);

template <typename Key>
struct fhm_hasher
{
    absl::flat_hash_map<Key, uint64_t> hasher{};

    dll_export fhm_hasher() = default;
    dll_export fhm_hasher(HASH_MODE mode = HASH_MODE_PRIME, bool deallocate = true);

    dll_export void make_hash_location(size_t array_size, uint64_t * hash_list, size_t hint_size);
};

#endif
