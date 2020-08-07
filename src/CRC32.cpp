#include "RipTide.h"
#include "ndarray.h"

#include "CommonInc.h"

//#include "crc32c/crc32c.h"
// NOTE: The crc algorithm was adapted from Google Go.
// Parts Copyright 2009 The Go Authors. 
// The size of a CRC-64 checksum in bytes.
const int Size = 8;

// The ISO polynomial, defined in ISO 3309 and used in HDLC.
const UINT64 ISO = 0xD800000000000000;

// The ISO polynomial, defined in ISO 3309 and used in HDLC.
const UINT64 ECMA = 0xC96C5795D7870F42;

// returns array of 256 UINT64
static UINT64* makeTable(UINT64 poly)
{
   UINT64* t = new UINT64[256];
   for (int i = 0; i < 256; i++)
   {
      UINT64 crc = (UINT64)i;
      for (int j = 0; j < 8; j++)
      {
         if ((crc & 1) == 1)
            crc = (crc >> 1) ^ poly;
         else
            crc >>= 1;
      }

      t[i] = crc;
   }

   return t;
}

// returns array of 8 PUINT64 ->  256
static UINT64** createTables(int sizeInner, int sizeOuter)
{
   PUINT64* l = new PUINT64[sizeOuter];
   for (int i = 0; i < sizeOuter; i++)
   {
      l[i] = new UINT64[sizeInner];
   }

   return l;
}

// returns array of 8 PUINT64 ->  256
static const UINT64** makeSlicingBy8Table(UINT64* t)
{
   UINT64** helperTable = createTables(256, 8);
   helperTable[0] = t;
   for (int i = 0; i < 256; i++)
   {
      UINT64 crc = t[i];
      for (int j = 1; j < 8; j++)
      {
         crc = t[crc & 0xff] ^ (crc >> 8);
         helperTable[j][i] = crc;
      }
   }

   return (const UINT64**)helperTable;
}

static const UINT64** slicing8TableISO = makeSlicingBy8Table(makeTable(ISO));
static const UINT64** slicing8TableECMA =  makeSlicingBy8Table(makeTable(ECMA));


// Call with crc=0 to init or previous crc to chain
// et tab to slicing8TableECMA
static UINT64 CalculateCRC64(UINT64 crc, const UINT64* tab, UINT8* p, INT64 Length)
{
   crc = ~crc;
   UINT8*  pEnd = p + Length;
   while (Length >= 64)
   {
      const UINT64** helperTable;
      if (tab == slicing8TableECMA[0])
         helperTable = slicing8TableECMA;
      else if (tab == slicing8TableISO[0])
         helperTable = slicing8TableISO;
      else return 0;  // error

      // Update using slicing by 8
      while (Length >= 8)
      {
         // TODO: this can be vectorized
         crc ^= ((UINT64)p[0]) | ((UINT64)p[1]) << 8 | ((UINT64)p[2]) << 16 | ((UINT64)p[3]) << 24 |
            ((UINT64)p[4]) << 32 | ((UINT64)p[5]) << 40 | ((UINT64)p[6]) << 48 | ((UINT64)p[7]) << 56;
         crc = helperTable[7][crc & 0xff] ^
            helperTable[6][(crc >> 8) & 0xff] ^
            helperTable[5][(crc >> 16) & 0xff] ^
            helperTable[4][(crc >> 24) & 0xff] ^
            helperTable[3][(crc >> 32) & 0xff] ^
            helperTable[2][(crc >> 40) & 0xff] ^
            helperTable[1][(crc >> 48) & 0xff] ^
            helperTable[0][crc >> 56];

         p = p + 8;
         Length = Length - 8;
      }
   }

   // For smaller sizes
   while (p < pEnd)
   {
      crc = tab[((UINT8)crc) ^ *p] ^ (crc >> 8);
      p++;
   }

   return ~crc;
}


// Returns a 64bit CRC value using ECMA
// Could return UINT64 value instead, but for equality comparison it does not matter
PyObject *
CalculateCRC(PyObject *self, PyObject *args) {
   PyArrayObject *inArr1 = NULL;

   if (!PyArg_ParseTuple(
      args, "O!",
      &PyArray_Type, &inArr1)) {

      return NULL;
   }

   if (!PyArray_ISCONTIGUOUS(inArr1)) {
      PyErr_Format(PyExc_ValueError, "CalculateCRC array is not contiguous");
      return NULL;
   }

   void* pDataIn1 = PyArray_BYTES(inArr1);
   INT64 arraySize = ArrayLength(inArr1) * PyArray_ITEMSIZE(inArr1);

   // TOOD: in future can pass ISO table as well
   UINT64 crc_value = CalculateCRC64(0, slicing8TableECMA[0], (UINT8*)pDataIn1, arraySize);
   return PyLong_FromLongLong(crc_value);
}



