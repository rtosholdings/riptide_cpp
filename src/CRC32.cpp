#include "RipTide.h"
#include "ndarray.h"

#include "CommonInc.h"

//#include "crc32c/crc32c.h"
// NOTE: The crc algorithm was adapted from Google Go.
// Parts Copyright 2009 The Go Authors.
// The size of a CRC-64 checksum in bytes.
const int Size = 8;

// The ISO polynomial, defined in ISO 3309 and used in HDLC.
const uint64_t ISO = 0xD800000000000000;

// The ISO polynomial, defined in ISO 3309 and used in HDLC.
const uint64_t ECMA = 0xC96C5795D7870F42;

// returns array of 256 uint64_t
static uint64_t * makeTable(uint64_t poly)
{
    uint64_t * t = new uint64_t[256];
    for (int i = 0; i < 256; i++)
    {
        uint64_t crc = (uint64_t)i;
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

// returns array of 8 uint64_t* ->  256
static uint64_t ** createTables(int sizeInner, int sizeOuter)
{
    uint64_t ** l = new uint64_t *[sizeOuter];
    for (int i = 0; i < sizeOuter; i++)
    {
        l[i] = new uint64_t[sizeInner];
    }

    return l;
}

// returns array of 8 uint64_t* ->  256
static const uint64_t ** makeSlicingBy8Table(uint64_t * t)
{
    uint64_t ** helperTable = createTables(256, 8);
    helperTable[0] = t;
    for (int i = 0; i < 256; i++)
    {
        uint64_t crc = t[i];
        for (int j = 1; j < 8; j++)
        {
            crc = t[crc & 0xff] ^ (crc >> 8);
            helperTable[j][i] = crc;
        }
    }

    return (const uint64_t **)helperTable;
}

static const uint64_t ** slicing8TableISO = makeSlicingBy8Table(makeTable(ISO));
static const uint64_t ** slicing8TableECMA = makeSlicingBy8Table(makeTable(ECMA));

// Call with crc=0 to init or previous crc to chain
// et tab to slicing8TableECMA
static uint64_t CalculateCRC64(uint64_t crc, const uint64_t * tab, uint8_t * p, int64_t Length)
{
    crc = ~crc;
    uint8_t * pEnd = p + Length;
    while (Length >= 64)
    {
        const uint64_t ** helperTable;
        if (tab == slicing8TableECMA[0])
            helperTable = slicing8TableECMA;
        else if (tab == slicing8TableISO[0])
            helperTable = slicing8TableISO;
        else
            return 0; // error

        // Update using slicing by 8
        while (Length >= 8)
        {
            // TODO: this can be vectorized
            crc ^= ((uint64_t)p[0]) | ((uint64_t)p[1]) << 8 | ((uint64_t)p[2]) << 16 | ((uint64_t)p[3]) << 24 |
                   ((uint64_t)p[4]) << 32 | ((uint64_t)p[5]) << 40 | ((uint64_t)p[6]) << 48 | ((uint64_t)p[7]) << 56;
            crc = helperTable[7][crc & 0xff] ^ helperTable[6][(crc >> 8) & 0xff] ^ helperTable[5][(crc >> 16) & 0xff] ^
                  helperTable[4][(crc >> 24) & 0xff] ^ helperTable[3][(crc >> 32) & 0xff] ^ helperTable[2][(crc >> 40) & 0xff] ^
                  helperTable[1][(crc >> 48) & 0xff] ^ helperTable[0][crc >> 56];

            p = p + 8;
            Length = Length - 8;
        }
    }

    // For smaller sizes
    while (p < pEnd)
    {
        crc = tab[((uint8_t)crc) ^ *p] ^ (crc >> 8);
        p++;
    }

    return ~crc;
}

// Returns a 64bit CRC value using ECMA
// Could return uint64_t value instead, but for equality comparison it does not
// matter
PyObject * CalculateCRC(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;

    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &inArr1))
    {
        return NULL;
    }

    if (! PyArray_ISCONTIGUOUS(inArr1))
    {
        PyErr_Format(PyExc_ValueError, "CalculateCRC array is not contiguous");
        return NULL;
    }

    void * pDataIn1 = PyArray_BYTES(inArr1);
    int64_t arraySize = ArrayLength(inArr1) * PyArray_ITEMSIZE(inArr1);

    // TOOD: in future can pass ISO table as well
    uint64_t crc_value = CalculateCRC64(0, slicing8TableECMA[0], (uint8_t *)pDataIn1, arraySize);
    return PyLong_FromLongLong(crc_value);
}
