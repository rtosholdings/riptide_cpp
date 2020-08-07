#pragma once
#include "RipTide.h"


/*-=====  Pre-defined compression levels  =====-*/
#ifndef ZSTD_CLEVEL_DEFAULT
#define ZSTD_CLEVEL_DEFAULT 3
#endif

#ifndef ZSTD_MAX_CLEVEL
#define ZSTD_MAX_CLEVEL     22
#endif

#ifndef ZSTD_MIN_CLEVEL
#define ZSTD_MIN_CLEVEL     -5
#endif

#define DISCARD_PARAMETER (void)

PyObject *CompressString(PyObject* self, PyObject *args);
PyObject *DecompressString(PyObject* self, PyObject *args);
PyObject *CompressDecompressArrays(PyObject* self, PyObject *args);

#define COMPRESSION_TYPE_NONE    0
#define COMPRESSION_TYPE_ZSTD    1

#define COMPRESSION_MODE_COMPRESS 0
#define COMPRESSION_MODE_DECOMPRESS 1
#define COMPRESSION_MODE_COMPRESS_FILE 2
#define COMPRESSION_MODE_DECOMPRESS_FILE 3
#define COMPRESSION_MODE_SHAREDMEMORY 4
#define COMPRESSION_MODE_INFO 5

#define COMPRESSION_MAGIC  253

#define HEADER_TAG_ARRAY 1

struct NUMPY_HEADERSIZE {
   UINT8    magic;
   INT8     compressiontype;
   INT8     dtype;
   INT8     ndim;

   INT32    itemsize;
   INT32    flags;

   // no more than 3 dims
   INT64    dimensions[3];
   size_t   compressedSize;

   void*    pCompressedArray;

   INT64    get_arraylength() {
      INT64 arraylength = dimensions[0];

      for (int i = 1; i < ndim; i++) {
         arraylength *= dimensions[0];
      }

      arraylength *= itemsize;
      return arraylength;
   }

};



struct COMPRESS_NUMPY_TO_NUMPY {
   struct ArrayInfo*   aInfo;

   INT64               totalHeaders;

   INT16               compMode;
   INT16               compType;
   INT32               compLevel;

   // Per core allocations
   void*                pCoreMemory[64];

   // See: compMode value -- COMPRESSION_MODE_SHAREDMEMORY
   union {
      // used when compressing to arrays
      NUMPY_HEADERSIZE*   pNumpyHeaders[1];

      // used when compressing to file
      PyArrayObject*      pNumpyArray[1];

   };

};


//--------------- COMPRESSION ROUTINES --------------------
//size_t CompressGetBound(int compMode, size_t srcSize);

//size_t CompressData(int compMode, void* dst, size_t dstCapacity,
//   const void* src, size_t srcSize,
//   int compressionLevel);
//
//size_t DecompressData(int compMode, void* dst, size_t dstCapacity, const void* src, size_t srcSize);
//
//BOOL CompressIsError(int compMode, size_t code);
