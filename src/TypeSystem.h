#pragma once
#include <cstdint>

#if defined(_WIN32) && ! defined(__GNUC__)
    #include <../Lib/site-packages/numpy/core/include/numpy/ndarraytypes.h>
#else
    #include <numpy/ndarraytypes.h>
#endif

namespace riptide
{
    // Linux: long = 64 bits
    // Windows: long = 32 bits
    static /*constexpr*/ NPY_TYPES normalize_dtype(const NPY_TYPES dtype, const int64_t itemsize)
    {
        switch (dtype)
        {
        case NPY_TYPES::NPY_LONG:
            // types 7 and 8 are ambiguous because they map to different concrete types
            // depending on the platform being targeted; specifically, the size of a
            // "long" on that platform.
            return itemsize == 4 ? NPY_TYPES::NPY_INT : NPY_TYPES::NPY_LONGLONG;

        case NPY_TYPES::NPY_ULONG:
            // types 7 and 8 are ambiguous
            return itemsize == 4 ? NPY_TYPES::NPY_UINT : NPY_TYPES::NPY_ULONGLONG;

        default:
            // No adjustment needed, return the original dtype.
            return dtype;
        }
    }
} // namespace riptide
