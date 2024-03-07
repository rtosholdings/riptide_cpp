#include "zstd.h"
#include "zstd_errors.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define LOGGING(...)
//#define LOGGING printf

namespace
{
    bool enable_zstd_checksumming = true;
}

//----------------------------------------------------
// Enable Zstandard checksumming. This is only for compression.
// Decompression will automatically verify checksum if it's present
void ZstdSetChecksumming(bool enable)
{
    enable_zstd_checksumming = enable;
}

//-------------------------------------------------------
// Compress with Zstd, returning size or zstd error
// Called from multiple threads
size_t ZstdCompressData(void * dst, size_t dstCapacity, const void * src, size_t srcSize, int32_t compressionLevel)
{
    if (enable_zstd_checksumming)
    {
        // Note: although ZSTD_CCtx can be reused, we need a separate instance per thread, and ZstdCompressData
        // is only called once per thread. So allocating/freeing here for simplicity.
        ZSTD_CCtx * cctx = ZSTD_createCCtx();
        if (! cctx)
        {
            LOGGING("Error creating ZSTD_CCtx\n");
            return ZSTD_error_memory_allocation;
        }
        else
        {
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1);
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, compressionLevel);
            LOGGING("Compressing with checksum\n");
            size_t result = ZSTD_compress2(cctx, dst, dstCapacity, src, srcSize);
            ZSTD_freeCCtx(cctx);
            return result;
        }
    }
    else
    {
        return ZSTD_compress(dst, dstCapacity, src, srcSize, compressionLevel);
    }
}
