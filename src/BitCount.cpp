#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "BitCount.h"
#include <nmmintrin.h>

PyObject * BitCount(PyObject * self, PyObject * args)
{
    static constexpr int8_t nibble_bitcount[] = { 0,   // 0b0000
                                                  1,   // 0b0001
                                                  1,   // 0b0010
                                                  2,   // 0b0011
                                                  1,   // 0b0100
                                                  2,   // 0b0101
                                                  2,   // 0b0110
                                                  3,   // 0b0111
                                                  1,   // 0b1000
                                                  2,   // 0b1001
                                                  2,   // 0b1010
                                                  3,   // 0b1011
                                                  2,   // 0b1100
                                                  3,   // 0b1101
                                                  3,   // 0b1110
                                                  4 }; // 0b1111
    PyArrayObject * inArr = NULL;
    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &inArr))
        return NULL;

    const int ndim = PyArray_NDIM(inArr);
    // N.B. AllocateNumpyArray doesn't accept const pointer for dims
    npy_intp * dims = PyArray_DIMS(inArr);
    PyArrayObject * returnObject = AllocateNumpyArray(ndim, dims, NPY_INT8);
    if (returnObject == NULL)
        return NULL;
    int8_t * pDataOut = (int8_t *)PyArray_BYTES(returnObject);

    const npy_intp length = CalcArrayLength(ndim, dims);
    if (PyArray_TYPE(inArr) == NPY_BOOL)
    {
        auto pDataIn = (uint8_t *)PyArray_BYTES(inArr);
        for (npy_intp i(0); i < length; ++i, ++pDataIn, ++pDataOut)
            *pDataOut = *pDataIn == 0 ? 0 : 1;
    }
    else
    {
        npy_intp itemsize = PyArray_ITEMSIZE(inArr);
        switch (itemsize)
        {
        case 1:
            {
                auto pDataIn = (uint8_t *)PyArray_BYTES(inArr);
                for (npy_intp i(0); i < length; ++i, ++pDataIn, ++pDataOut)
                {
                    const auto n(*pDataIn);
                    *pDataOut = nibble_bitcount[n >> 4] + nibble_bitcount[n & 0xf];
                }
                break;
            }
        case 2:
            // N.B. __builtin_popcount from GCC works for only unsigned int, can't use
            // for short.
            {
                auto pDataIn = (uint16_t *)PyArray_BYTES(inArr);
                for (npy_intp i(0); i < length; ++i, ++pDataIn, ++pDataOut)
                {
                    const auto n(*pDataIn);
                    *pDataOut = nibble_bitcount[n >> 12] + nibble_bitcount[(n >> 8) & 0xf] + nibble_bitcount[(n >> 4) & 0xf] +
                                nibble_bitcount[n & 0xf];
                }
                break;
            }
        case 4:
            {
                auto pDataIn = (uint32_t *)PyArray_BYTES(inArr);
                for (npy_intp i(0); i < length; ++i, ++pDataIn, ++pDataOut)
                    *pDataOut = _mm_popcnt_u32(*pDataIn);
                break;
            }
        case 8:
            {
                auto pDataIn = (uint64_t *)PyArray_BYTES(inArr);
                for (npy_intp i(0); i < length; ++i, ++pDataIn, ++pDataOut)
                    *pDataOut = (int8_t)_mm_popcnt_u64(*pDataIn);
                break;
            }
        default:
            PyArray_XDECREF(returnObject);
            PyErr_Format(PyExc_ValueError, "%s %i", "Unsupported itemsize", itemsize);
            returnObject = NULL;
        }
    }

    return (PyObject *)returnObject;
}
