#pragma once

#include "Defs.h"

#include <algorithm>
#include <atomic>
#include <functional>

namespace riptide
{
    namespace internal
    {
        // Shouldn't be used in the signal handler if this isn't true
        static_assert(std::atomic<bool>::is_always_lock_free);
        RT_DLLEXPORT extern std::atomic<bool> interrupted_;
    }

    RT_FORCEINLINE bool is_interrupted()
    {
        return internal::interrupted_.load(std::memory_order_relaxed);
    }

    RT_DLLEXPORT bool interruptible_section(std::function<void(void)> function);

    template <typename F, int64_t chunk_size = 0x4000>
    RT_FORCEINLINE void interruptible_for(int64_t start, int64_t end, int64_t increment, F && loop_body)
    {
        for (int64_t chunk_start = start; chunk_start < end && ! is_interrupted(); chunk_start += chunk_size)
        {
            int64_t chunk_end = std::min(chunk_start + chunk_size, end);

            for (int64_t i = chunk_start; i < chunk_end; i += increment)
            {
                loop_body(i);
            }
        }
    }
}