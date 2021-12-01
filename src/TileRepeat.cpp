#include "MathWorker.h"
#include "RipTide.h"
#include "ndarray.h"

#include "CommonInc.h"

//#define LOGGING printf
#define LOGGING(...)

PyObject * ToBeDone(PyObject * self, PyObject * args, PyObject * kwargs)
{
    return NULL;
}

struct stOffsets
{
    char * pData;
    int64_t itemsize;
};
static const int64_t CHUNKSIZE = 16384;

// This is used to multiply the strides
SFW_ALIGN(64)
const union
{
    int32_t i[8];
    __m256i m;
    //} __vindex8_strides = { 7, 6, 5, 4, 3, 2, 1, 0 };
} __vindex8_strides = { 0, 1, 2, 3, 4, 5, 6, 7 };

//-----------------------------------
//
void ConvertRecArray(char * pStartOffset, int64_t startRow, int64_t totalRows, stOffsets * pstOffset, int64_t * pOffsets,
                     int64_t numArrays, int64_t itemSize)
{
    // Try to keep everything in L1Cache
    const int64_t L1CACHE = 32768;
    int64_t CHUNKROWS = L1CACHE / (itemSize * 2);
    if (CHUNKROWS < 1)
    {
        CHUNKROWS = 1;
    }

    __m256i vindex = _mm256_mullo_epi32(_mm256_set1_epi32((int32_t)itemSize), _mm256_loadu_si256(&__vindex8_strides.m));
    __m128i vindex128 = _mm256_extracti128_si256(vindex, 0);

    while (startRow < totalRows)
    {
        // Calc how many rows to process in this pass
        int64_t endRow = startRow + CHUNKROWS;
        if (endRow > totalRows)
        {
            endRow = totalRows;
        }

        int64_t origRow = startRow;

        // printf("processing %lld\n", startRow);
        for (int64_t i = 0; i < numArrays; i++)
        {
            startRow = origRow;

            // Calculate place to read
            char * pRead = pStartOffset + pOffsets[i];
            char * pWrite = pstOffset[i].pData;

            int64_t arrItemSize = pstOffset[i].itemsize;

            // printf("processing  start:%lld  end:%lld   pRead:%p  %p  itemsize:
            // %lld\n", startRow, endRow, pRead, pWrite, arrItemSize);

            switch (pstOffset[i].itemsize)
            {
            case 1:
                while (startRow < endRow)
                {
                    int8_t data = *(int8_t *)(pRead + (startRow * itemSize));
                    *(int8_t *)(pWrite + startRow) = data;
                    startRow++;
                }
                break;
            case 2:
                while (startRow < endRow)
                {
                    int16_t data = *(int16_t *)(pRead + (startRow * itemSize));
                    *(int16_t *)(pWrite + startRow * arrItemSize) = data;
                    startRow++;
                }
                break;
            case 4:
                // ??? use _mm256_i32gather_epi32 to speed up
                {
                    int64_t endSubRow = endRow - 8;
                    while (startRow < endSubRow)
                    {
                        __m256i m0 = _mm256_i32gather_epi32(reinterpret_cast<int32_t *>(pRead + (startRow * itemSize)), vindex, 1);
                        _mm256_storeu_si256((__m256i *)(pWrite + (startRow * arrItemSize)), m0);
                        startRow += 8;
                    }
                    while (startRow < endRow)
                    {
                        int32_t data = *(int32_t *)(pRead + (startRow * itemSize));
                        *(int32_t *)(pWrite + startRow * arrItemSize) = data;
                        startRow++;
                    }
                }
                break;
            case 8:
                {
                    int64_t endSubRow = endRow - 4;
                    while (startRow < endSubRow)
                    {
#if defined(__linux__) && defined(__GNUC__)
                        // on Linux _mm256_i32gather_epi64() accepts a long long *, which is
                        // the same size but not identical to long * (a.k.a int64_t *) so GCC
                        // complains.
                        static_assert(sizeof(long long) == sizeof(int64_t));
                        __m256i m0 = _mm256_i32gather_epi64(reinterpret_cast<long long const *>(pRead + (startRow * itemSize)),
                                                            vindex128, 1);
#else
                        __m256i m0 =
                            _mm256_i32gather_epi64(reinterpret_cast<int64_t const *>(pRead + (startRow * itemSize)), vindex128, 1);
#endif
                        _mm256_storeu_si256((__m256i *)(pWrite + (startRow * arrItemSize)), m0);
                        startRow += 4;
                    }
                    while (startRow < endRow)
                    {
                        int64_t data = *(int64_t *)(pRead + (startRow * itemSize));
                        *(int64_t *)(pWrite + startRow * arrItemSize) = data;
                        startRow++;
                    }
                }
                break;
            default:
                while (startRow < endRow)
                {
                    char * pSrc = pRead + (startRow * itemSize);
                    char * pDest = pWrite + (startRow * arrItemSize);
                    char * pEnd = pSrc + arrItemSize;
                    while ((pSrc + 8) < pEnd)
                    {
                        *(int64_t *)pDest = *(int64_t *)pSrc;
                        pDest += 8;
                        pSrc += 8;
                    }
                    while (pSrc < pEnd)
                    {
                        *pDest++ = *pSrc++;
                    }
                    startRow++;
                }
                break;
            }
        }
    }
}

//-----------------------------------
// Input1: the recordarray to convert
// Input2: int64 array of offsets
// Input3: list of arrays pre allocated
PyObject * RecordArrayToColMajor(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr = NULL;
    PyArrayObject * offsetArr = NULL;
    PyArrayObject * arrArr = NULL;

    if (! PyArg_ParseTuple(args, "O!O!O!:RecordArrayToColMajor", &PyArray_Type, &inArr, &PyArray_Type, &offsetArr, &PyArray_Type,
                           &arrArr))
    {
        return NULL;
    }

    int64_t itemSize = PyArray_ITEMSIZE(inArr);

    if (itemSize != PyArray_STRIDE(inArr, 0))
    {
        PyErr_Format(PyExc_ValueError, "RecordArrayToColMajor cannot handle strides");
        return NULL;
    }

    int64_t length = ArrayLength(inArr);
    int64_t numArrays = ArrayLength(arrArr);

    if (numArrays != ArrayLength(offsetArr))
    {
        PyErr_Format(PyExc_ValueError, "RecordArrayToColMajor inputs do not match");
        return NULL;
    }

    int64_t totalRows = length;
    int64_t * pOffsets = (int64_t *)PyArray_BYTES(offsetArr);
    PyArrayObject ** ppArrays = (PyArrayObject **)PyArray_BYTES(arrArr);

    stOffsets * pstOffset;

    pstOffset = (stOffsets *)WORKSPACE_ALLOC(sizeof(stOffsets) * numArrays);

    for (int64_t i = 0; i < numArrays; i++)
    {
        pstOffset[i].pData = PyArray_BYTES(ppArrays[i]);
        pstOffset[i].itemsize = PyArray_ITEMSIZE(ppArrays[i]);
    }

    // printf("chunkrows is %lld\n", CHUNKROWS);

    char * pStartOffset = PyArray_BYTES(inArr);
    int64_t startRow = 0;

    if (totalRows > 16384)
    {
        // Prepare for multithreading
        struct stConvertRec
        {
            char * pStartOffset;
            int64_t startRow;
            int64_t totalRows;
            stOffsets * pstOffset;
            int64_t * pOffsets;
            int64_t numArrays;
            int64_t itemSize;
            int64_t lastRow;
        } stConvert;

        int64_t items = (totalRows + (CHUNKSIZE - 1)) / CHUNKSIZE;

        stConvert.pStartOffset = pStartOffset;
        stConvert.startRow = startRow;
        stConvert.totalRows = totalRows;
        stConvert.pstOffset = pstOffset;
        stConvert.pOffsets = pOffsets;
        stConvert.numArrays = numArrays;
        stConvert.itemSize = itemSize;
        stConvert.lastRow = items - 1;

        auto lambdaConvertRecCallback = [](void * callbackArgT, int core, int64_t workIndex) -> bool
        {
            stConvertRec * callbackArg = (stConvertRec *)callbackArgT;
            int64_t startRow = callbackArg->startRow + (workIndex * CHUNKSIZE);
            int64_t totalRows = startRow + CHUNKSIZE;

            if (totalRows > callbackArg->totalRows)
            {
                totalRows = callbackArg->totalRows;
            }

            ConvertRecArray(callbackArg->pStartOffset, startRow, totalRows, callbackArg->pstOffset, callbackArg->pOffsets,
                            callbackArg->numArrays, callbackArg->itemSize);

            LOGGING("[%d] %lld completed\n", core, workIndex);
            return true;
        };

        g_cMathWorker->DoMultiThreadedWork((int)items, lambdaConvertRecCallback, &stConvert);
    }
    else
    {
        ConvertRecArray(pStartOffset, startRow, totalRows, pstOffset, pOffsets, numArrays, itemSize);
    }

    WORKSPACE_FREE(pstOffset);

    RETURN_NONE;
}
