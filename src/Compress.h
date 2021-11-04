#pragma once
#include "RipTide.h"

/*-=====  Pre-defined compression levels  =====-*/
#ifndef ZSTD_CLEVEL_DEFAULT
    #define ZSTD_CLEVEL_DEFAULT 3
#endif

#ifndef ZSTD_MAX_CLEVEL
    #define ZSTD_MAX_CLEVEL 22
#endif

#ifndef ZSTD_MIN_CLEVEL
    #define ZSTD_MIN_CLEVEL -5
#endif

#define DISCARD_PARAMETER (void)

PyObject * CompressString(PyObject * self, PyObject * args);
PyObject * DecompressString(PyObject * self, PyObject * args);
PyObject * CompressDecompressArrays(PyObject * self, PyObject * args);

#define COMPRESSION_TYPE_NONE 0
#define COMPRESSION_TYPE_ZSTD 1

#define COMPRESSION_MODE_COMPRESS 0
#define COMPRESSION_MODE_DECOMPRESS 1
#define COMPRESSION_MODE_COMPRESS_FILE 2
#define COMPRESSION_MODE_DECOMPRESS_FILE 3
#define COMPRESSION_MODE_SHAREDMEMORY 4
#define COMPRESSION_MODE_INFO 5

#define COMPRESSION_MAGIC 253

#define HEADER_TAG_ARRAY 1

struct NUMPY_HEADERSIZE
{
    uint8_t magic;
    int8_t compressiontype;
    int8_t dtype;
    int8_t ndim;

    int32_t itemsize;
    int32_t flags;

    // no more than 3 dims
    int64_t dimensions[3];
    size_t compressedSize;

    void * pCompressedArray;

    int64_t get_arraylength()
    {
        int64_t arraylength = dimensions[0];

        for (int i = 1; i < ndim; i++)
        {
            arraylength *= dimensions[0];
        }

        arraylength *= itemsize;
        return arraylength;
    }
};

struct COMPRESS_NUMPY_TO_NUMPY
{
    struct ArrayInfo * aInfo;

    int64_t totalHeaders;

    int16_t compMode;
    int16_t compType;
    int32_t compLevel;

    // Per core allocations
    void * pCoreMemory[64];

    // See: compMode value -- COMPRESSION_MODE_SHAREDMEMORY
    union
    {
        // used when compressing to arrays
        NUMPY_HEADERSIZE * pNumpyHeaders[1];

        // used when compressing to file
        PyArrayObject * pNumpyArray[1];
    };
};

//--------------- COMPRESSION ROUTINES --------------------
// size_t CompressGetBound(int compMode, size_t srcSize);

// size_t CompressData(int compMode, void* dst, size_t dstCapacity,
//   const void* src, size_t srcSize,
//   int compressionLevel);
//
// size_t DecompressData(int compMode, void* dst, size_t dstCapacity, const
// void* src, size_t srcSize);
//
// BOOL CompressIsError(int compMode, size_t code);
