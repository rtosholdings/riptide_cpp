#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "Merge.h"
#include "Convert.h"

#include <algorithm>

// for _pext_u64
#if defined(__GNUC__) || defined(__clang__)
    //#include <bmi2intrin.h>
    #include <x86intrin.h>
#endif

//#define LOGGING printf
#define LOGGING(...)

/**
 * Count the number of 'True' (nonzero) 1-byte bool values in an array,
 * using an AVX2-based implementation.
 *
 * @param pData Array of 1-byte bool values.
 * @param length The number of elements in the array.
 * @return The number of nonzero 1-byte bool values in the array.
 */
// TODO: When we support runtime CPU detection/dispatching, bring back the
// original popcnt-based implementation
//       of this function for systems that don't support AVX2. Also consider
//       implementing an SSE-based version of this function for the same reason
//       (logic will be very similar, just using __m128i instead).
// TODO: Consider changing `length` to uint64_t here so it agrees better with
// the result of sizeof().
int64_t SumBooleanMask(const int8_t * const pData, const int64_t length, const int64_t strideBoolean)
{
    // Basic input validation.
    if (! pData)
    {
        return 0;
    }
    else if (length < 0)
    {
        return 0;
    }
    // Holds the accumulated result value.
    int64_t result = 0;

    if (strideBoolean == 1)
    {
        // Now that we know length is >= 0, it's safe to convert it to unsigned so
        // it agrees with the sizeof() math in the logic below. Make sure to use
        // this instead of 'length' in the code below to avoid signed/unsigned
        // arithmetic warnings.
        const size_t ulength = length;

        // YMM (32-byte) vector packed with 32 byte values, each set to 1.
        // NOTE: The obvious thing here would be to use _mm256_set1_epi8(1),
        //       but many compilers (e.g. MSVC) store the data for this vector
        //       then load it here, which unnecessarily wastes cache space we could
        //       be using for something else. Generate the constants using a few
        //       intrinsics, it's faster than even an L1 cache hit anyway.
        const auto zeros_ = _mm256_setzero_si256();
        // compare 0 to 0 returns 0xFF; treated as an int8_t, 0xFF = -1, so abs(-1)
        // = 1.
        const auto ones = _mm256_abs_epi8(_mm256_cmpeq_epi8(zeros_, zeros_));

        //
        // Convert each byte in the input to a 0 or 1 byte according to C-style
        // boolean semantics.
        //

        // This first loop does the bulk of the processing for large vectors -- it
        // doesn't use popcount instructions and instead relies on the fact we can
        // sum 0/1 values to acheive the same result, up to CHAR_MAX. This allows us
        // to use very inexpensive instructions for most of the accumulation so
        // we're primarily limited by memory bandwidth.
        const size_t vector_length = ulength / sizeof(__m256i);
        const auto pVectorData = (__m256i *)pData;
        for (size_t i = 0; i < vector_length;)
        {
            // Determine how much we can process in _this_ iteration of the loop.
            // The maximum number of "inner" iterations here is CHAR_MAX (255),
            // because otherwise our byte-sized counters would overflow.
            auto inner_loop_iters = vector_length - i;
            if (inner_loop_iters > 255)
                inner_loop_iters = 255;

            // Holds the current per-vector-lane (i.e. per-byte-within-vector)
            // popcount. PERF: If necessary, the loop below can be manually unrolled
            // to ensure we saturate memory bandwidth.
            auto byte_popcounts = _mm256_setzero_si256();
            for (size_t j = 0; j < inner_loop_iters; j++)
            {
                // Use an unaligned load to grab a chunk of data;
                // then call _mm256_min_epu8 where one operand is the register we set
                // earlier containing packed byte-sized 1 values (e.g. 0x01010101...).
                // This effectively converts each byte in the input to a 0 or 1 byte
                // value.
                const auto cstyle_bools = _mm256_min_epu8(ones, _mm256_loadu_si256(&pVectorData[i + j]));

                // Since each byte in the converted vector now contains either a 0 or 1,
                // we can simply add it to the running per-byte sum to simulate a
                // popcount.
                byte_popcounts = _mm256_add_epi8(byte_popcounts, cstyle_bools);
            }

            // Sum the per-byte-lane popcounts, then add them to the overall result.
            // For the vectorized partial sums, it's important the 'zeros' argument is
            // used as the second operand so that the zeros are 'unpacked' into the
            // high byte(s) of each packed element in the result.
            const auto zeros = _mm256_setzero_si256();

            // Sum 32x 1-byte counts -> 16x 2-byte counts
            const auto byte_popcounts_8a = _mm256_unpacklo_epi8(byte_popcounts, zeros);
            const auto byte_popcounts_8b = _mm256_unpackhi_epi8(byte_popcounts, zeros);
            const auto byte_popcounts_16 = _mm256_add_epi16(byte_popcounts_8a, byte_popcounts_8b);

            // Sum 16x 2-byte counts -> 8x 4-byte counts
            const auto byte_popcounts_16a = _mm256_unpacklo_epi16(byte_popcounts_16, zeros);
            const auto byte_popcounts_16b = _mm256_unpackhi_epi16(byte_popcounts_16, zeros);
            const auto byte_popcounts_32 = _mm256_add_epi32(byte_popcounts_16a, byte_popcounts_16b);

            // Sum 8x 4-byte counts -> 4x 8-byte counts
            const auto byte_popcounts_32a = _mm256_unpacklo_epi32(byte_popcounts_32, zeros);
            const auto byte_popcounts_32b = _mm256_unpackhi_epi32(byte_popcounts_32, zeros);
            const auto byte_popcounts_64 = _mm256_add_epi64(byte_popcounts_32a, byte_popcounts_32b);

            // perform the operation horizontally in m0
            union
            {
                volatile int64_t horizontal[4];
                __m256i mathreg[1];
            };

            mathreg[0] = byte_popcounts_64;
            for (int j = 0; j < 4; j++)
            {
                result += horizontal[j];
            }

            // Increment the outer loop counter by the number of inner iterations we
            // performed.
            i += inner_loop_iters;
        }

        // Handle the last few bytes, if any, that couldn't be handled with the
        // vectorized loop.
        const size_t vectorized_length = vector_length * sizeof(__m256i);
        for (size_t i = vectorized_length; i < ulength; i++)
        {
            if (pData[i])
            {
                result++;
            }
        }
    }
    else
    {
        for (int64_t i = 0; i < length; i++)
        {
            if (pData[i * strideBoolean])
            {
                result++;
            }
        }
        LOGGING("sum bool  %p   len:%I64d  vs  true:%I64d  stride:%I64d\n", pData, length, result, strideBoolean);
    }
    return result;
}

//===================================================
// Input: boolean array
// Output: chunk count and ppChunkCount
// NOTE: CALLER MUST FREE pChunkCount
//
int64_t BooleanCount(PyArrayObject * aIndex, int64_t ** ppChunkCount, int64_t strideBoolean)
{
    // Pass one, count the values
    // Eight at a time
    const int64_t lengthBool = ArrayLength(aIndex);
    const int8_t * const pBooleanMask = (int8_t *)PyArray_BYTES(aIndex);

    // Count the number of chunks (of boolean elements).
    // It's important we handle the case of an empty array (zero length) when
    // determining the number of per-chunk counts to return; the behavior of
    // malloc'ing zero bytes is undefined, and the code below assumes there's
    // always at least one entry in the count-per-chunk array. If we don't handle
    // the empty array case we'll allocate an empty count-per-chunk array and end
    // up doing an out-of-bounds write.
    const int64_t chunkSize = g_cMathWorker->WORK_ITEM_CHUNK;
    int64_t chunks = lengthBool > 1 ? lengthBool : 1;

    chunks = (chunks + (chunkSize - 1)) / chunkSize;

    // TOOD: try to allocate on stack when possible
    int64_t * const pChunkCount = (int64_t *)WORKSPACE_ALLOC(chunks * sizeof(int64_t));

    if (pChunkCount)
    {
        // MT callback
        struct BSCallbackStruct
        {
            int64_t * pChunkCount;
            const int8_t * pBooleanMask;
            int64_t strideBoolean;
        };

        // This is the routine that will be called back from multiple threads
        // t64_t(*MTCHUNK_CALLBACK)(void* callbackArg, int core, int64_t start,
        // int64_t length);
        auto lambdaBSCallback = [](void * callbackArgT, int core, int64_t start, int64_t length) -> bool
        {
            BSCallbackStruct * callbackArg = (BSCallbackStruct *)callbackArgT;

            const int8_t * pBooleanMask = callbackArg->pBooleanMask;
            int64_t * pChunkCount = callbackArg->pChunkCount;

            // Use the single-threaded implementation to sum the number of
            // 1-byte boolean true values in the current chunk.
            // This means the current function is just responsible for parallelizing
            // over the chunks but doesn't do any real "math" itself.
            int64_t strides = callbackArg->strideBoolean;
            int64_t total = SumBooleanMask(&pBooleanMask[start * strides], length, strides);

            pChunkCount[start / g_cMathWorker->WORK_ITEM_CHUNK] = total;
            return true;
        };

        BSCallbackStruct stBSCallback;
        stBSCallback.pChunkCount = pChunkCount;
        stBSCallback.pBooleanMask = pBooleanMask;
        stBSCallback.strideBoolean = strideBoolean;

        bool didMtWork = g_cMathWorker->DoMultiThreadedChunkWork(lengthBool, lambdaBSCallback, &stBSCallback);

        *ppChunkCount = pChunkCount;
        // if multithreading turned off...
        return didMtWork ? chunks : 1;
    }
    // out of memory
    return 0;
}

//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aValues (can be any array)
// Arg2: numpy array aIndex (must be bool)
//
PyObject * BooleanIndexInternal(PyArrayObject * aValues, PyArrayObject * aIndex)
{
    if (PyArray_TYPE(aIndex) != NPY_BOOL)
    {
        PyErr_Format(PyExc_ValueError, "Second argument must be a boolean array");
        return NULL;
    }

    int ndimValue = 0;
    int ndimBoolean = 0;
    int64_t strideValue = 0;
    int64_t strideBoolean = 0;

    int result1 = GetStridesAndContig(aValues, ndimValue, strideValue);
    int result2 = GetStridesAndContig(aIndex, ndimBoolean, strideBoolean);

    // This logic is not quite correct, if the strides on all dimensions are the
    // same, we can use this routine
    if (result1 != 0)
    {
        if (! PyArray_ISCONTIGUOUS(aValues))
        {
            PyErr_Format(PyExc_ValueError,
                         "Dont know how to handle multidimensional "
                         "value array for boolean index.");
            return NULL;
        }
    }
    if (result2 != 0)
    {
        if (! PyArray_ISCONTIGUOUS(aIndex))
        {
            PyErr_Format(PyExc_ValueError,
                         "Dont know how to handle multidimensional "
                         "array used as boolean index.");
            return NULL;
        }
    }

    // if ( strideBoolean != 1) {
    //   PyErr_Format(PyExc_ValueError, "Dont know how to handle multidimensional
    //   array to be indexed."); return NULL;
    //}
    // Pass one, count the values
    // Eight at a time
    int64_t lengthBool = ArrayLength(aIndex);
    int64_t lengthValue = ArrayLength(aValues);

    if (lengthBool != lengthValue)
    {
        PyErr_Format(PyExc_ValueError, "Array lengths must match %lld vs %lld", lengthBool, lengthValue);
        return NULL;
    }

    int64_t * pChunkCount = NULL;
    int64_t chunks = BooleanCount(aIndex, &pChunkCount, strideBoolean);

    if (chunks == 0)
    {
        PyErr_Format(PyExc_ValueError, "Out of memory");
        return NULL;
    }

    int64_t totalTrue = 0;

    // Store the offset
    for (int64_t i = 0; i < chunks; i++)
    {
        int64_t temp = totalTrue;
        totalTrue += pChunkCount[i];

        // reassign to the cumulative sum so we know the offset
        pChunkCount[i] = temp;
    }

    LOGGING("boolean index total: %I64d  length: %I64d  type:%d   chunks:%I64d\n", totalTrue, lengthBool, PyArray_TYPE(aValues),
            chunks);

    int8_t * pBooleanMask = (int8_t *)PyArray_BYTES(aIndex);

    // Now we know per chunk how many true there are... we can allocate the new
    // array
    PyArrayObject * pReturnArray = AllocateLikeResize(aValues, totalTrue);

    if (pReturnArray)
    {
        // If the resulting array is empty there is no work to do
        if (totalTrue > 0)
        {
            // MT callback
            struct BICallbackStruct
            {
                int64_t * pChunkCount;
                int8_t * pBooleanMask;
                int64_t strideBoolean;
                char * pValuesIn;
                int64_t strideValues;
                char * pValuesOut;
                int64_t itemSize;
            };

            //-----------------------------------------------
            //-----------------------------------------------
            // This is the routine that will be called back from multiple threads
            auto lambdaBICallback2 = [](void * callbackArgT, int core, int64_t start, int64_t length) -> bool
            {
                BICallbackStruct * callbackArg = (BICallbackStruct *)callbackArgT;

                int8_t * pBooleanMask = callbackArg->pBooleanMask;
                int64_t chunkCount = callbackArg->pChunkCount[start / g_cMathWorker->WORK_ITEM_CHUNK];
                int64_t itemSize = callbackArg->itemSize;
                int64_t strideBoolean = callbackArg->strideBoolean;
                int64_t * pData = (int64_t *)&pBooleanMask[start * strideBoolean];
                int64_t strideValues = callbackArg->strideValues;
                char * pValuesIn = &callbackArg->pValuesIn[start * strideValues];

                // output is assumed contiguous
                char * pValuesOut = &callbackArg->pValuesOut[chunkCount * itemSize];

                // process 8 booleans at a time in the loop
                int64_t blength = length / 8;

                if (strideBoolean == 1)
                {
                    switch (itemSize)
                    {
                    case 1:
                        {
                            // NOTE: This routine can be improved further by
                            // Loading 32 booleans at time in math register
                            // Storing the result in a math register (shifting over new values)
                            // until full
                            int8_t * pVOut = (int8_t *)pValuesOut;
                            int8_t * pVIn = (int8_t *)pValuesIn;

                            for (int64_t i = 0; i < blength; i++)
                            {
                                uint64_t bitmask = *(uint64_t *)pData;

                                // NOTE: the below can be optimized with vector intrinsics
                                // little endian, so the first value is low bit (not high bit)
                                if (bitmask != 0)
                                {
                                    for (int j = 0; j < 8; j++)
                                    {
                                        if (bitmask & 0xff)
                                        {
                                            *pVOut++ = *pVIn;
                                        }
                                        pVIn += strideValues;
                                        bitmask >>= 8;
                                    }
                                }
                                else
                                {
                                    pVIn += 8 * strideValues;
                                }
                                pData++;
                            }

                            // Get last
                            pBooleanMask = (int8_t *)pData;

                            blength = length & 7;
                            for (int64_t i = 0; i < blength; i++)
                            {
                                if (*pBooleanMask++)
                                {
                                    *pVOut++ = *pVIn;
                                }
                                pVIn += strideValues;
                            }
                        }
                        break;
                    case 2:
                        {
                            int16_t * pVOut = (int16_t *)pValuesOut;
                            int16_t * pVIn = (int16_t *)pValuesIn;

                            for (int64_t i = 0; i < blength; i++)
                            {
                                uint64_t bitmask = *(uint64_t *)pData;
                                uint64_t mask = 0xff;
                                // little endian, so the first value is low bit (not high bit)
                                if (bitmask != 0)
                                {
                                    for (int j = 0; j < 8; j++)
                                    {
                                        if (bitmask & 0xff)
                                        {
                                            *pVOut++ = *pVIn;
                                        }
                                        pVIn = STRIDE_NEXT(int16_t, pVIn, strideValues);
                                        bitmask >>= 8;
                                    }
                                }
                                else
                                {
                                    pVIn = STRIDE_NEXT(int16_t, pVIn, 8 * strideValues);
                                }
                                pData++;
                            }

                            // Get last
                            pBooleanMask = (int8_t *)pData;

                            blength = length & 7;
                            for (int64_t i = 0; i < blength; i++)
                            {
                                if (*pBooleanMask++)
                                {
                                    *pVOut++ = *pVIn;
                                }
                                pVIn = STRIDE_NEXT(int16_t, pVIn, strideValues);
                            }
                        }
                        break;
                    case 4:
                        {
                            int32_t * pVOut = (int32_t *)pValuesOut;
                            int32_t * pVIn = (int32_t *)pValuesIn;

                            for (int64_t i = 0; i < blength; i++)
                            {
                                // little endian, so the first value is low bit (not high bit)
                                uint64_t bitmask = *(uint64_t *)pData;
                                if (bitmask != 0)
                                {
                                    for (int j = 0; j < 8; j++)
                                    {
                                        if (bitmask & 0xff)
                                        {
                                            *pVOut++ = *pVIn;
                                        }
                                        pVIn = STRIDE_NEXT(int32_t, pVIn, strideValues);
                                        bitmask >>= 8;
                                    }
                                }
                                else
                                {
                                    pVIn = STRIDE_NEXT(int32_t, pVIn, 8 * strideValues);
                                }
                                pData++;
                            }

                            // Get last
                            pBooleanMask = (int8_t *)pData;

                            blength = length & 7;
                            for (int64_t i = 0; i < blength; i++)
                            {
                                if (*pBooleanMask++)
                                {
                                    *pVOut++ = *pVIn;
                                }
                                pVIn = STRIDE_NEXT(int32_t, pVIn, strideValues);
                            }
                        }
                        break;
                    case 8:
                        {
                            int64_t * pVOut = (int64_t *)pValuesOut;
                            int64_t * pVIn = (int64_t *)pValuesIn;

                            for (int64_t i = 0; i < blength; i++)
                            {
                                // little endian, so the first value is low bit (not high bit)
                                uint64_t bitmask = *(uint64_t *)pData;
                                if (bitmask != 0)
                                {
                                    for (int j = 0; j < 8; j++)
                                    {
                                        if (bitmask & 0xff)
                                        {
                                            *pVOut++ = *pVIn;
                                        }
                                        pVIn = STRIDE_NEXT(int64_t, pVIn, strideValues);
                                        bitmask >>= 8;
                                    }
                                }
                                else
                                {
                                    pVIn = STRIDE_NEXT(int64_t, pVIn, 8 * strideValues);
                                }
                                pData++;
                            }

                            // Get last
                            pBooleanMask = (int8_t *)pData;

                            blength = length & 7;
                            for (int64_t i = 0; i < blength; i++)
                            {
                                if (*pBooleanMask++)
                                {
                                    *pVOut++ = *pVIn;
                                }
                                pVIn = STRIDE_NEXT(int64_t, pVIn, strideValues);
                            }
                        }
                        break;

                    default:
                        {
                            for (int64_t i = 0; i < blength; i++)
                            {
                                // little endian, so the first value is low bit (not high bit)
                                uint64_t bitmask = *(uint64_t *)pData;
                                if (bitmask != 0)
                                {
                                    int counter = 8;
                                    while (counter--)
                                    {
                                        if (bitmask & 0xff)
                                        {
                                            memcpy(pValuesOut, pValuesIn, itemSize);
                                            pValuesOut += itemSize;
                                        }
                                        pValuesIn += strideValues;
                                        bitmask >>= 8;
                                    }
                                }
                                else
                                {
                                    pValuesIn += (strideValues * 8);
                                }
                                pData++;
                            }

                            // Get last
                            pBooleanMask = (int8_t *)pData;

                            blength = length & 7;
                            for (int64_t i = 0; i < blength; i++)
                            {
                                if (*pBooleanMask++)
                                {
                                    memcpy(pValuesOut, pValuesIn, itemSize);
                                    pValuesOut += strideValues;
                                }
                                pValuesIn += strideValues;
                            }
                        }
                        break;
                    }
                }
                else
                {
                    // The boolean mask is strided
                    // FUTURE OPTIMIZATION: We can use the gather command to speed this
                    // path
                    int8_t * pBool = (int8_t *)pData;
                    switch (itemSize)
                    {
                    case 1:
                        {
                            int8_t * pVOut = (int8_t *)pValuesOut;
                            int8_t * pVIn = (int8_t *)pValuesIn;
                            for (int64_t i = 0; i < length; i++)
                            {
                                if (*pBool)
                                {
                                    *pVOut++ = *pVIn;
                                }
                                pBool += strideBoolean;
                                pVIn += strideValues;
                            }
                        }
                        break;
                    case 2:
                        {
                            int16_t * pVOut = (int16_t *)pValuesOut;
                            int16_t * pVIn = (int16_t *)pValuesIn;
                            for (int64_t i = 0; i < length; i++)
                            {
                                if (*pBool)
                                {
                                    *pVOut++ = *pVIn;
                                }
                                pBool += strideBoolean;
                                pVIn = STRIDE_NEXT(int16_t, pVIn, strideValues);
                            }
                        }
                        break;
                    case 4:
                        {
                            int32_t * pVOut = (int32_t *)pValuesOut;
                            int32_t * pVIn = (int32_t *)pValuesIn;
                            for (int64_t i = 0; i < length; i++)
                            {
                                if (*pBool)
                                {
                                    *pVOut++ = *pVIn;
                                }
                                pBool += strideBoolean;
                                pVIn = STRIDE_NEXT(int32_t, pVIn, strideValues);
                            }
                        }
                        break;
                    case 8:
                        {
                            int64_t * pVOut = (int64_t *)pValuesOut;
                            int64_t * pVIn = (int64_t *)pValuesIn;
                            for (int64_t i = 0; i < length; i++)
                            {
                                if (*pBool)
                                {
                                    *pVOut++ = *pVIn;
                                }
                                pBool += strideBoolean;
                                pVIn = STRIDE_NEXT(int64_t, pVIn, strideValues);
                            }
                        }
                        break;
                    default:
                        {
                            char * pVOut = (char *)pValuesOut;
                            char * pVIn = (char *)pValuesIn;
                            for (int64_t i = 0; i < length; i++)
                            {
                                if (*pBool)
                                {
                                    memcpy(pVOut, pVIn, itemSize);
                                    pVOut += itemSize;
                                }
                                pBool += strideBoolean;
                                pVIn += strideValues;
                            }
                        }
                        break;
                    }
                }
                return true;
            };

            BICallbackStruct stBICallback;
            stBICallback.pChunkCount = pChunkCount;
            stBICallback.pBooleanMask = pBooleanMask;
            stBICallback.pValuesIn = (char *)PyArray_BYTES(aValues);
            stBICallback.pValuesOut = (char *)PyArray_BYTES(pReturnArray);
            stBICallback.itemSize = PyArray_ITEMSIZE(aValues);

            stBICallback.strideBoolean = strideBoolean;
            stBICallback.strideValues = strideValue;

            g_cMathWorker->DoMultiThreadedChunkWork(lengthBool, lambdaBICallback2, &stBICallback);
        }
    }
    else
    {
        // ran out of memory
        PyErr_Format(PyExc_ValueError, "Out of memory");
    }

    WORKSPACE_FREE(pChunkCount);
    return (PyObject *)pReturnArray;
}

//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aValues (can be anything)
// Arg2: numpy array aIndex (must be bool)
//
PyObject * BooleanIndex(PyObject * self, PyObject * args)
{
    PyArrayObject * aValues = NULL;
    PyArrayObject * aIndex = NULL;

    if (! PyArg_ParseTuple(args, "O!O!:BooleanIndex", &PyArray_Type, &aValues, &PyArray_Type, &aIndex))
    {
        return NULL;
    }
    return BooleanIndexInternal(aValues, aIndex);
}

//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aIndex (must be bool)
// Returns: how many true values there are
// NOTE: faster than calling sum
PyObject * BooleanSum(PyObject * self, PyObject * args)
{
    PyArrayObject * aIndex = NULL;

    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &aIndex))
    {
        return NULL;
    }

    if (PyArray_TYPE(aIndex) != NPY_BOOL)
    {
        PyErr_Format(PyExc_ValueError, "First argument must be boolean array");
        return NULL;
    }

    int ndimBoolean;
    int64_t strideBoolean;

    int result1 = GetStridesAndContig(aIndex, ndimBoolean, strideBoolean);

    int64_t * pChunkCount = NULL;
    int64_t chunks = BooleanCount(aIndex, &pChunkCount, strideBoolean);

    int64_t totalTrue = 0;
    for (int64_t i = 0; i < chunks; i++)
    {
        totalTrue += pChunkCount[i];
    }

    WORKSPACE_FREE(pChunkCount);
    return PyLong_FromSize_t(totalTrue);
}

//----------------------------------------------------
// Consider:  C=A[B]   where A is a value array
// C must be the same type as A (and is also a value array)
// B is an integer that indexes into A
// The length of B is the length of the output C
// valSize is the length of A
// aValues  : remains constant (pointer to A)
// aIndex   : incremented each call (pIndex) traverses B
// aDataOut : incremented each call (pDataOut) traverses C
// NOTE: The output CANNOT be strided
template <typename VALUE, typename INDEX>
static void GetItemInt(void * aValues, void * aIndex, void * aDataOut, int64_t valLength, int64_t itemSize, int64_t len,
                       int64_t strideIndex, int64_t strideValue, void * pDefault)
{
    const VALUE * pValues = (VALUE *)aValues;
    const INDEX * pIndex = (INDEX *)aIndex;
    VALUE * pDataOut = (VALUE *)aDataOut;
    VALUE defaultVal = *(VALUE *)pDefault;

    LOGGING("getitem sizes %lld  len: %lld   def: %I64d  or  %lf\n", valLength, len, (int64_t)defaultVal, (double)defaultVal);
    LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valLength);

    VALUE * pDataOutEnd = pDataOut + len;
    if (sizeof(VALUE) == strideValue && sizeof(INDEX) == strideIndex)
    {
        while (pDataOut != pDataOutEnd)
        {
            const INDEX index = *pIndex;
            *pDataOut =
                // Make sure the item is in range; if the index is negative -- but
                // otherwise still in range -- mimic Python's negative-indexing
                // support.
                index >= -valLength && index < valLength ? pValues[index >= 0 ? index : index + valLength]

                                                           // Index is out of range -- assign the invalid value.
                                                           :
                                                           defaultVal;
            pIndex++;
            pDataOut++;
        }
    }
    else
    {
        // Either A or B or both are strided
        while (pDataOut != pDataOutEnd)
        {
            const INDEX index = *pIndex;
            // Make sure the item is in range; if the index is negative -- but
            // otherwise still in range -- mimic Python's negative-indexing support.
            if (index >= -valLength && index < valLength)
            {
                int64_t newindex = index >= 0 ? index : index + valLength;
                newindex *= strideValue;
                *pDataOut = *(VALUE *)((char *)pValues + newindex);
            }
            else
            {
                // Index is out of range -- assign the invalid value.
                *pDataOut = defaultVal;
            }

            pIndex = STRIDE_NEXT(const INDEX, pIndex, strideIndex);
            pDataOut++;
        }
    }
}

//----------------------------------------------------
// Consider:  C=A[B]   where A is a value array
// C must be the same type as A (and is also a value array)
// B is an integer that indexes into A
// The length of B is the length of the output C
// valSize is the length of A
// aValues  : remains constant (pointer to A)
// aIndex   : incremented each call (pIndex) traverses B
// aDataOut : incremented each call (pDataOut) traverses C
// NOTE: The output CANNOT be strided
template <typename VALUE, typename INDEX>
static void GetItemUInt(void * aValues, void * aIndex, void * aDataOut, int64_t valLength, int64_t itemSize, int64_t len,
                        int64_t strideIndex, int64_t strideValue, void * pDefault)
{
    const VALUE * pValues = (VALUE *)aValues;
    const INDEX * pIndex = (INDEX *)aIndex;
    VALUE * pDataOut = (VALUE *)aDataOut;
    VALUE defaultVal = *(VALUE *)pDefault;

    LOGGING("getitem sizes %lld  len: %lld   def: %I64d  or  %lf\n", valLength, len, (int64_t)defaultVal, (double)defaultVal);
    LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valLength);

    VALUE * pDataOutEnd = pDataOut + len;
    if (sizeof(VALUE) == strideValue && sizeof(INDEX) == strideIndex)
    {
        while (pDataOut != pDataOutEnd)
        {
            const INDEX index = *pIndex;
            *pDataOut =
                // Make sure the item is in range
                index < valLength ? pValues[index] : defaultVal;
            pIndex++;
            pDataOut++;
        }
    }
    else
    {
        // Either A or B or both are strided
        while (pDataOut != pDataOutEnd)
        {
            const INDEX index = *pIndex;
            // Make sure the item is in range; if the index is negative -- but
            // otherwise still in range -- mimic Python's negative-indexing support.
            if (index < valLength)
            {
                *pDataOut = *(VALUE *)((char *)pValues + (strideValue * index));
            }
            else
            {
                // Index is out of range -- assign the invalid value.
                *pDataOut = defaultVal;
            }

            pIndex = STRIDE_NEXT(const INDEX, pIndex, strideIndex);
            pDataOut++;
        }
    }
}

//----------------------------------------------------
// This routine is for strings or NPY_VOID (variable length)
// Consider:  C=A[B]   where A is a value array
// C must be the same type as A (and is also a value array)
// B is an integer that indexes into A
// The length of B is the length of the output C
// valSize is the length of A
template <typename INDEX>
static void GetItemIntVariable(void * aValues, void * aIndex, void * aDataOut, int64_t valLength, int64_t itemSize, int64_t len,
                               int64_t strideIndex, int64_t strideValue, void * pDefault)
{
    const char * pValues = (char *)aValues;
    const INDEX * pIndex = (INDEX *)aIndex;
    char * pDataOut = (char *)aDataOut;

    LOGGING("getitem sizes %I64d  len: %I64d   itemsize:%I64d\n", valLength, len, itemSize);
    LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valLength);

    char * pDataOutEnd = pDataOut + (len * itemSize);
    if (itemSize == strideValue && sizeof(INDEX) == strideIndex)
    {
        while (pDataOut != pDataOutEnd)
        {
            const INDEX index = *pIndex;
            const char * pSrc;
            if (index >= -valLength && index < valLength)
            {
                int64_t newindex = index >= 0 ? index : index + valLength;
                newindex *= itemSize;
                pSrc = pValues + newindex;
            }
            else
            {
                pSrc = (const char *)pDefault;
            }

            char * pEnd = pDataOut + itemSize;

            while (pDataOut < (pEnd - 8))
            {
                *(int64_t *)pDataOut = *(int64_t *)pSrc;
                pDataOut += 8;
                pSrc += 8;
            }
            while (pDataOut < pEnd)
            {
                *pDataOut++ = *pSrc++;
            }
            //    memcpy(pDataOut, pSrc, itemSize);

            pIndex++;
        }
    }
    else
    {
        // Either A or B or both are strided
        while (pDataOut != pDataOutEnd)
        {
            const INDEX index = *pIndex;
            const char * pSrc;
            if (index >= -valLength && index < valLength)
            {
                int64_t newindex = index >= 0 ? index : index + valLength;
                newindex *= strideValue;
                pSrc = pValues + newindex;
            }
            else
            {
                pSrc = (const char *)pDefault;
            }

            char * pEnd = pDataOut + itemSize;

            while (pDataOut < (pEnd - 8))
            {
                *(int64_t *)pDataOut = *(int64_t *)pSrc;
                pDataOut += 8;
                pSrc += 8;
            }
            while (pDataOut < pEnd)
            {
                *pDataOut++ = *pSrc++;
            }
            pIndex = STRIDE_NEXT(const INDEX, pIndex, strideIndex);
        }
    }
}

template <typename INDEX>
static void GetItemUIntVariable(void * aValues, void * aIndex, void * aDataOut, int64_t valLength, int64_t itemSize, int64_t len,
                                int64_t strideIndex, int64_t strideValue, void * pDefault)
{
    const char * pValues = (char *)aValues;
    const INDEX * pIndex = (INDEX *)aIndex;
    char * pDataOut = (char *)aDataOut;

    LOGGING("getitem sizes %I64d  len: %I64d   itemsize:%I64d\n", valLength, len, itemSize);
    LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valLength);

    char * pDataOutEnd = pDataOut + (len * itemSize);
    if (itemSize == strideValue && sizeof(INDEX) == strideIndex)
    {
        while (pDataOut != pDataOutEnd)
        {
            const INDEX index = *pIndex;
            const char * pSrc;
            if (index < valLength)
            {
                pSrc = pValues + (itemSize * index);
            }
            else
            {
                pSrc = (const char *)pDefault;
            }

            char * pEnd = pDataOut + itemSize;

            while (pDataOut < (pEnd - 8))
            {
                *(int64_t *)pDataOut = *(int64_t *)pSrc;
                pDataOut += 8;
                pSrc += 8;
            }
            while (pDataOut < pEnd)
            {
                *pDataOut++ = *pSrc++;
            }
            //    memcpy(pDataOut, pSrc, itemSize);

            pIndex++;
            pDataOut += itemSize;
        }
    }
    else
    {
        // Either A or B or both are strided
        while (pDataOut != pDataOutEnd)
        {
            const INDEX index = *pIndex;
            const char * pSrc;
            if (index < valLength)
            {
                pSrc = pValues + (strideValue * index);
            }
            else
            {
                pSrc = (const char *)pDefault;
            }

            char * pEnd = pDataOut + itemSize;

            while (pDataOut < (pEnd - 8))
            {
                *(int64_t *)pDataOut = *(int64_t *)pSrc;
                pDataOut += 8;
                pSrc += 8;
            }
            while (pDataOut < pEnd)
            {
                *pDataOut++ = *pSrc++;
            }
            pIndex = STRIDE_NEXT(const INDEX, pIndex, strideIndex);
            pDataOut += itemSize;
        }
    }
}

typedef void (*GETITEM_FUNC)(void * pDataIn, void * pDataIn2, void * pDataOut, int64_t valLength, int64_t itemSize, int64_t len,
                             int64_t strideIndex, int64_t strideValue, void * pDefault);
struct MBGET_CALLBACK
{
    GETITEM_FUNC GetItemCallback;

    void * pValues;  // value array or A in the equation C=A[B]
    void * pIndex;   // index array or B in the equation C=A[B]
    void * pDataOut; // output array or C in the equation C=A[B]
    int64_t aValueLength;
    int64_t aIndexLength;
    int64_t aValueItemSize;
    int64_t aIndexItemSize;
    int64_t strideValue;
    int64_t strideIndex;
    void * pDefault;

} stMBGCallback;

//---------------------------------------------------------
// Used by GetItem
//  Concurrent callback from multiple threads
static bool GetItemCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    int64_t didSomeWork = 0;
    MBGET_CALLBACK * Callback = &stMBGCallback; // (MBGET_CALLBACK*)&pstWorkerItem->WorkCallbackArg;

    char * aValues = (char *)Callback->pValues;
    char * aIndex = (char *)Callback->pIndex;

    int64_t valueItemSize = Callback->aValueItemSize;
    int64_t strideValue = Callback->strideValue;
    int64_t strideIndex = Callback->strideIndex;

    LOGGING("check2 ** %lld %lld\n", typeSizeValues, typeSizeIndex);

    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
    {
        // Do NOT move aValues
        // Move aIndex
        // Move pDataOut (same type as Values)
        // move starting position

        // Calculate how much to adjust the pointers to get to the data for this
        // work block
        int64_t blockStart = workBlock * pstWorkerItem->BlockSize;

        int64_t valueAdj = blockStart * strideValue;
        int64_t indexAdj = blockStart * strideIndex;

        LOGGING(
            "%d : workBlock %lld   blocksize: %lld    lenx: %lld  %lld  %lld  "
            "%lld %lld\n",
            core, workBlock, pstWorkerItem->BlockSize, lenX, typeSizeValues, typeSizeIndex, valueAdj, indexAdj);

        Callback->GetItemCallback(aValues, aIndex + indexAdj, (char *)Callback->pDataOut + valueAdj, Callback->aValueLength,
                                  valueItemSize, lenX, strideIndex, strideValue, Callback->pDefault);

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
    }

    return didSomeWork > 0;
}

//------------------------------------------------------------
// itemSize is Values itemSize
// indexType is Index type
static GETITEM_FUNC GetItemFunction(int64_t itemSize, int indexType)
{
    switch (indexType)
    {
    case NPY_INT8:
        switch (itemSize)
        {
        case 1:
            return GetItemInt<int8_t, int8_t>;
        case 2:
            return GetItemInt<int16_t, int8_t>;
        case 4:
            return GetItemInt<int32_t, int8_t>;
        case 8:
            return GetItemInt<int64_t, int8_t>;
        case 16:
            return GetItemInt<__m128, int8_t>;
        default:
            return GetItemIntVariable<int8_t>;
        }
        break;
    case NPY_UINT8:
        switch (itemSize)
        {
        case 1:
            return GetItemUInt<int8_t, int8_t>;
        case 2:
            return GetItemUInt<int16_t, int8_t>;
        case 4:
            return GetItemUInt<int32_t, int8_t>;
        case 8:
            return GetItemUInt<int64_t, int8_t>;
        case 16:
            return GetItemUInt<__m128, int8_t>;
        default:
            return GetItemUIntVariable<int8_t>;
        }
        break;

    case NPY_INT16:
        switch (itemSize)
        {
        case 1:
            return GetItemInt<int8_t, int16_t>;
        case 2:
            return GetItemInt<int16_t, int16_t>;
        case 4:
            return GetItemInt<int32_t, int16_t>;
        case 8:
            return GetItemInt<int64_t, int16_t>;
        case 16:
            return GetItemInt<__m128, int16_t>;
        default:
            return GetItemIntVariable<int16_t>;
        }
        break;
    case NPY_UINT16:
        switch (itemSize)
        {
        case 1:
            return GetItemUInt<int8_t, int16_t>;
        case 2:
            return GetItemUInt<int16_t, int16_t>;
        case 4:
            return GetItemUInt<int32_t, int16_t>;
        case 8:
            return GetItemUInt<int64_t, int16_t>;
        case 16:
            return GetItemUInt<__m128, int16_t>;
        default:
            return GetItemUIntVariable<int16_t>;
        }
        break;

    CASE_NPY_INT32:
        switch (itemSize)
        {
        case 1:
            return GetItemInt<int8_t, int32_t>;
        case 2:
            return GetItemInt<int16_t, int32_t>;
        case 4:
            return GetItemInt<int32_t, int32_t>;
        case 8:
            return GetItemInt<int64_t, int32_t>;
        case 16:
            return GetItemInt<__m128, int32_t>;
        default:
            return GetItemIntVariable<int32_t>;
        }
        break;
    CASE_NPY_UINT32:
        switch (itemSize)
        {
        case 1:
            return GetItemUInt<int8_t, int32_t>;
        case 2:
            return GetItemUInt<int16_t, int32_t>;
        case 4:
            return GetItemUInt<int32_t, int32_t>;
        case 8:
            return GetItemUInt<int64_t, int32_t>;
        case 16:
            return GetItemUInt<__m128, int32_t>;
        default:
            return GetItemUIntVariable<int32_t>;
        }
        break;

    CASE_NPY_INT64:
        switch (itemSize)
        {
        case 1:
            return GetItemInt<int8_t, int64_t>;
        case 2:
            return GetItemInt<int16_t, int64_t>;
        case 4:
            return GetItemInt<int32_t, int64_t>;
        case 8:
            return GetItemInt<int64_t, int64_t>;
        case 16:
            return GetItemInt<__m128, int64_t>;
        default:
            return GetItemIntVariable<int64_t>;
        }
        break;
    CASE_NPY_UINT64:
        switch (itemSize)
        {
        case 1:
            return GetItemUInt<int8_t, int64_t>;
        case 2:
            return GetItemUInt<int16_t, int64_t>;
        case 4:
            return GetItemUInt<int32_t, int64_t>;
        case 8:
            return GetItemUInt<int64_t, int64_t>;
        case 16:
            return GetItemUInt<__m128, int64_t>;
        default:
            return GetItemUIntVariable<int64_t>;
        }
        break;
    }

    return NULL;
}

//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aValues (can be anything)
// Arg2: numpy array aIndex (must be int8_t/int16_t/int32_t or int64_t)
// Arg3: default value
//
// def fixMbget(aValues, aIndex, result, default) :
//   """
//   A proto routine.
//   """
//   N = aIndex.shape[0]
//   valSize = aValues.shape[0]
//   for i in range(N) :
//      if (aIndex[i] >= 0 and aIndex[i] < valSize) :
//         result[i] = aValues[aIndex[i]]
//      else :
//         result[i] = default  (OR RETURN ERROR)
PyObject * MBGet(PyObject * self, PyObject * args)
{
    PyArrayObject * aValues = NULL;
    PyArrayObject * aIndex = NULL;
    PyObject * defaultValue = NULL;

    if (PyTuple_Size(args) == 2)
    {
        if (! PyArg_ParseTuple(args, "O!O!:getitem", &PyArray_Type, &aValues, &PyArray_Type, &aIndex))
        {
            return NULL;
        }
        defaultValue = Py_None;
    }
    else if (! PyArg_ParseTuple(args, "O!O!O:getitem", &PyArray_Type, &aValues, &PyArray_Type, &aIndex, &defaultValue))
    {
        return NULL;
    }

    int32_t numpyValuesType = PyArray_TYPE(aValues);
    int32_t numpyIndexType = PyArray_TYPE(aIndex);

    // TODO: For boolean call
    if (numpyIndexType > NPY_LONGDOUBLE)
    {
        PyErr_Format(PyExc_ValueError, "Dont know how to convert these types %d using index dtype: %d", numpyValuesType,
                     numpyIndexType);
        return NULL;
    }

    if (numpyIndexType == NPY_BOOL)
    {
        // special path for boolean
        return BooleanIndexInternal(aValues, aIndex);
    }

    int ndimValue;
    int ndimIndex;
    int64_t strideValue = 0;
    int64_t strideIndex = 0;

    int result1 = GetStridesAndContig(aValues, ndimValue, strideValue);
    int result2 = GetStridesAndContig(aIndex, ndimIndex, strideIndex);

    // This logic is not quite correct, if the strides on all dimensions are the
    // same, we can use this routine
    if (result1 != 0)
    {
        if (! PyArray_ISCONTIGUOUS(aValues))
        {
            PyErr_Format(PyExc_ValueError,
                         "Dont know how to handle multidimensional array %d using "
                         "index dtype: %d",
                         numpyValuesType, numpyIndexType);
            return NULL;
        }
    }
    if (result2 != 0)
    {
        if (! PyArray_ISCONTIGUOUS(aIndex))
        {
            PyErr_Format(PyExc_ValueError,
                         "Dont know how to handle multidimensional array %d using "
                         "index dtype: %d",
                         numpyValuesType, numpyIndexType);
            return NULL;
        }
    }

    // printf("numpy types %d %d\n", numpyValuesType, numpyIndexType);

    void * pValues = PyArray_BYTES(aValues);
    void * pIndex = PyArray_BYTES(aIndex);

    int64_t aValueLength = ArrayLength(aValues);
    int64_t aValueItemSize = PyArray_ITEMSIZE(aValues);
    int64_t aIndexLength = ArrayLength(aIndex);

    // Get the proper function to call
    GETITEM_FUNC pFunction = GetItemFunction(aValueItemSize, numpyIndexType);

    if (pFunction != NULL || aIndexLength == 0)
    {
        PyArrayObject * outArray = (PyArrayObject *)Py_None;

        // Allocate the size of aIndex but the type is the value
        outArray = AllocateLikeResize(aValues, aIndexLength);

        if (outArray)
        {
            if (aIndexLength != 0)
            {
                void * pDataOut = PyArray_BYTES(outArray);
                void * pDefault = GetDefaultForType(numpyValuesType);

                // reserve a full 16 bytes for default in case we have oneS
                _m256all tempDefault;

                // Check if a default value was passed in as third parameter
                if (defaultValue != Py_None)
                {
                    bool result;
                    int64_t itemSize;
                    void * pTempData = NULL;
                    // Try to convert the scalar
                    result = ConvertScalarObject(defaultValue, &tempDefault, numpyValuesType, &pTempData, &itemSize);
                    if (result)
                    {
                        // Assign the new default for out of range indexes
                        pDefault = &tempDefault;
                    }
                }

                stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(aIndexLength);

                if (pWorkItem == NULL)
                {
                    // Threading not allowed for this work item, call it directly from
                    // main thread
                    // typedef void(*GETITEM_FUNC)(void* pDataIn, void* pDataIn2, void*
                    // pDataOut, int64_t valSize, int64_t itemSize, int64_t len, int64_t
                    // strideIndex, int64_t strideValue, void* pDefault);
                    pFunction(pValues, pIndex, pDataOut, aValueLength, aValueItemSize, aIndexLength, strideIndex, strideValue,
                              pDefault);
                }
                else
                {
                    // Each thread will call this routine with the callbackArg
                    // typedef int64_t(*DOWORK_CALLBACK)(struct stMATH_WORKER_ITEM*
                    // pstWorkerItem, int core, int64_t workIndex);
                    pWorkItem->DoWorkCallback = GetItemCallback;

                    pWorkItem->WorkCallbackArg = &stMBGCallback;

                    stMBGCallback.GetItemCallback = pFunction;
                    stMBGCallback.pValues = pValues;
                    stMBGCallback.pIndex = pIndex;
                    stMBGCallback.pDataOut = pDataOut;

                    // arraylength of values input array -- used to check array bounds
                    stMBGCallback.aValueLength = aValueLength;
                    stMBGCallback.aIndexLength = aIndexLength;
                    stMBGCallback.pDefault = pDefault;

                    //
                    stMBGCallback.aValueItemSize = aValueItemSize;
                    stMBGCallback.aIndexItemSize = PyArray_ITEMSIZE(aIndex);
                    stMBGCallback.strideIndex = strideIndex;
                    stMBGCallback.strideValue = strideValue;

                    // printf("**check %p %p %p %lld %lld\n", pValues, pIndex, pDataOut,
                    // stMBGCallback.TypeSizeValues, stMBGCallback.TypeSizeIndex);

                    // This will notify the worker threads of a new work item
                    g_cMathWorker->WorkMain(pWorkItem, aIndexLength, 0);
                    // g_cMathWorker->WorkMain(pWorkItem, aIndexLength);
                }
            }
            return (PyObject *)outArray;
        }
        PyErr_Format(PyExc_ValueError, "GetItem ran out of memory %d %d", numpyValuesType, numpyIndexType);
        return NULL;
    }

    PyErr_Format(PyExc_ValueError, "Dont know how to convert these types %d %d.  itemsize: %lld", numpyValuesType, numpyIndexType,
                 aValueItemSize);
    return NULL;
}

//===============================================================================
// checks for kwargs 'both'
// if exists, and is True return True
bool GetKwargBoth(PyObject * kwargs)
{
    // Check for cutoffs kwarg to see if going into parallel mode
    if (kwargs && PyDict_Check(kwargs))
    {
        PyObject * pBoth = NULL;
        // Borrowed reference
        // Returns NULL if key not present
        pBoth = PyDict_GetItemString(kwargs, "both");

        if (pBoth != NULL && pBoth == Py_True)
        {
            return true;
        }
    }
    return false;
}

//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aIndex (must be BOOL)
// Kwarg: "both"
//
// Returns: fancy index array where the true values are
// if 'both' is set to True it returns an index array which has both True and
// False if 'both' is set, the number of True values is also returned
PyObject * BooleanToFancy(PyObject * self, PyObject * args, PyObject * kwargs)
{
    PyArrayObject * aIndex = NULL;

    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &aIndex))
    {
        return NULL;
    }

    if (PyArray_TYPE(aIndex) != NPY_BOOL)
    {
        PyErr_Format(PyExc_ValueError, "First argument must be boolean array");
        return NULL;
    }

    int ndimBoolean;
    int64_t strideBoolean;

    int result1 = GetStridesAndContig(aIndex, ndimBoolean, strideBoolean);

    // if bothMode is set, will return fancy index for both True and False
    bool bothMode = GetKwargBoth(kwargs);

    int64_t * pChunkCount = NULL;
    int64_t * pChunkCountFalse = NULL;
    int64_t chunks = BooleanCount(aIndex, &pChunkCount, strideBoolean);
    int64_t indexLength = ArrayLength(aIndex);

    int64_t totalTrue = 0;

    if (bothMode)
    {
        // now count up the chunks
        // TJD: April 2019 note -- when the chunk size is between 65536 and 128000
        // it is really one chunk, but the code still works
        int64_t chunkSize = g_cMathWorker->WORK_ITEM_CHUNK;
        int64_t chunks = (indexLength + (chunkSize - 1)) / chunkSize;

        // Also need false count
        pChunkCountFalse = (int64_t *)WORKSPACE_ALLOC(chunks * sizeof(int64_t));
        int64_t totalFalse = 0;

        // Store the offset
        for (int64_t i = 0; i < chunks; i++)
        {
            int64_t temp = totalFalse;

            // check for last chunk
            totalFalse += (chunkSize - pChunkCount[i]);

            // reassign to the cumulative sum so we know the offset
            pChunkCountFalse[i] = temp;
        }

        // printf("both mode - chunks: %lld  totalTrue: %lld   toatlFalse: %lld\n",
        // chunks, pChunkCount[0], pChunkCountFalse[0]);
    }

    // Store the offset
    for (int64_t i = 0; i < chunks; i++)
    {
        int64_t temp = totalTrue;
        totalTrue += pChunkCount[i];

        // reassign to the cumulative sum so we know the offset
        pChunkCount[i] = temp;
    }

    PyArrayObject * returnArray = NULL;
    int dtype = NPY_INT64;
    // INT32 or INT64
    if (indexLength < 2000000000)
    {
        dtype = NPY_INT32;
    }

    if (bothMode)
    {
        // Allocate for both True and False
        returnArray = AllocateNumpyArray(1, (npy_intp *)&indexLength, dtype);
    }
    else
    {
        // INT32 or INT64
        returnArray = AllocateNumpyArray(1, (npy_intp *)&totalTrue, dtype);
    }

    CHECK_MEMORY_ERROR(returnArray);

    if (returnArray)
    {
        // MT callback
        struct BTFCallbackStruct
        {
            int64_t * pChunkCount;
            int64_t * pChunkCountFalse;
            int8_t * pBooleanMask;
            void * pValuesOut;
            int64_t totalTrue;
            int dtype;
            bool bothMode;
        };

        // This is the routine that will be called back from multiple threads
        auto lambdaCallback = [](void * callbackArgT, int core, int64_t start, int64_t length) -> bool
        {
            BTFCallbackStruct * callbackArg = (BTFCallbackStruct *)callbackArgT;

            int64_t chunkCount = callbackArg->pChunkCount[start / g_cMathWorker->WORK_ITEM_CHUNK];
            int8_t * pBooleanMask = callbackArg->pBooleanMask;
            bool bothMode = callbackArg->bothMode;

            if (bothMode)
            {
                int64_t chunkCountFalse = callbackArg->pChunkCountFalse[start / g_cMathWorker->WORK_ITEM_CHUNK];
                // printf("[%lld] ccf %lld  length %lld\n", start, chunkCountFalse,
                // length);

                if (callbackArg->dtype == NPY_INT64)
                {
                    int64_t * pOut = (int64_t *)callbackArg->pValuesOut;
                    pOut = pOut + chunkCount;

                    int64_t * pOutFalse = (int64_t *)callbackArg->pValuesOut;
                    pOutFalse = pOutFalse + callbackArg->totalTrue + chunkCountFalse;

                    for (int64_t i = start; i < (start + length); i++)
                    {
                        if (pBooleanMask[i])
                        {
                            *pOut++ = i;
                        }
                        else
                        {
                            *pOutFalse++ = i;
                        }
                    }
                }
                else
                {
                    int32_t * pOut = (int32_t *)callbackArg->pValuesOut;
                    pOut = pOut + chunkCount;

                    int32_t * pOutFalse = (int32_t *)callbackArg->pValuesOut;
                    pOutFalse = pOutFalse + callbackArg->totalTrue + chunkCountFalse;

                    for (int64_t i = start; i < (start + length); i++)
                    {
                        if (pBooleanMask[i])
                        {
                            *pOut++ = (int32_t)i;
                        }
                        else
                        {
                            *pOutFalse++ = (int32_t)i;
                        }
                    }
                }
            }
            else
            {
                if (callbackArg->dtype == NPY_INT64)
                {
                    int64_t * pOut = (int64_t *)callbackArg->pValuesOut;
                    pOut = pOut + chunkCount;

                    for (int64_t i = start; i < (start + length); i++)
                    {
                        if (pBooleanMask[i])
                        {
                            *pOut++ = i;
                        }
                    }
                }
                else
                {
                    int32_t * pOut = (int32_t *)callbackArg->pValuesOut;
                    pOut = pOut + chunkCount;

                    for (int64_t i = start; i < (start + length); i++)
                    {
                        if (pBooleanMask[i])
                        {
                            *pOut++ = (int32_t)i;
                        }
                    }
                }
            }

            return true;
        };

        BTFCallbackStruct stBTFCallback;
        stBTFCallback.pChunkCount = pChunkCount;
        stBTFCallback.pChunkCountFalse = pChunkCountFalse;
        stBTFCallback.pBooleanMask = (int8_t *)PyArray_BYTES(aIndex);
        stBTFCallback.pValuesOut = (int64_t *)PyArray_BYTES(returnArray);
        stBTFCallback.dtype = dtype;
        stBTFCallback.totalTrue = totalTrue;
        stBTFCallback.bothMode = bothMode;

        g_cMathWorker->DoMultiThreadedChunkWork(indexLength, lambdaCallback, &stBTFCallback);
    }

    //_mm_i32gather_epi32

    WORKSPACE_FREE(pChunkCount);
    if (pChunkCountFalse)
    {
        WORKSPACE_FREE(pChunkCountFalse);
    }
    if (bothMode)
    {
        // also return the true count so user knows cutoff
        PyObject * returnTuple = PyTuple_New(2);
        PyTuple_SET_ITEM(returnTuple, 0, (PyObject *)returnArray);
        PyTuple_SET_ITEM(returnTuple, 1, PyLong_FromSize_t(totalTrue));
        return returnTuple;
    }
    return (PyObject *)returnArray;
}

//
//
//#--------- START OF C++ ROUTINE -------------
//#based on how many uniques we have, allocate the new ikey
//# do we have a routine for this ?
// uikey_length = len(uikey)
// if uikey_length < 100 :
//   dtype = np.int8
//   elif uikey_length < 30_000 :
//   dtype = np.int16
//   elif uikey_length < 2_000_000_000 :
//   dtype = np.int32
// else:
// dtype = np.int64
//
// newikey = empty((len(ikey), ), dtype = dtype)
//
// start = 0
// starti = 0
// for i in range(len(u_cutoffs)) :
//   stop = u_cutoffs[i]
//   stopi = i_cutoffs[i]
//   uikey_slice = uikey[start:stop]
//   oldikey_slice = ikey[starti:stopi]
//
//   if verbose:
//      print("fixing ", starti, stopi)
//      print("newikey ", newikey)
//      print("oldikey_slice ", oldikey_slice)
//
//   if base_index == 1 :
//      # write a routine for this in C++
//      # if 0 and base_index=1, then keep the 0
//      filtermask = oldikey_slice == 0
//      newikey[starti:stopi] = uikey_slice[oldikey_slice - 1]
//      if filtermask.sum() > 0:
//         newikey[starti:stopi][filtermask] = 0
//   else:
//         newikey[starti:stopi] = uikey_slice[oldikey_slice]
//
// start = stop
// starti = stopi
//#END C++ ROUTINE-------------------------------- -
//

struct stReIndex
{
    int64_t * pUCutOffs;
    int64_t * pICutOffs;
    int32_t * pUKey;
    void * pIKey;

    int64_t ikey_length;
    int64_t uikey_length;
    int64_t u_cutoffs_length;
};

//
// t is the partition/cutoff index
template <typename KEYTYPE>
bool ReIndexGroupsMT(void * preindexV, int core, int64_t t)
{
    stReIndex * preindex = (stReIndex *)preindexV;

    int64_t * pUCutOffs = preindex->pUCutOffs;
    int64_t * pICutOffs = preindex->pICutOffs;
    int32_t * pUKey = preindex->pUKey;
    KEYTYPE * pIKey = (KEYTYPE *)preindex->pIKey;

    // Base 1 loop
    int64_t starti = 0;
    int64_t start = 0;
    if (t > 0)
    {
        starti = pICutOffs[t - 1];
        start = pUCutOffs[t - 1];
    }

    int64_t stopi = pICutOffs[t];
    int32_t * pUniques = &pUKey[start];

    // Check for out of bounds when indexing uniques
    int64_t uKeyLength = preindex->uikey_length - start;
    if (uKeyLength < 0)
        uKeyLength = 0;

    LOGGING("Start %lld  Stop %lld  Len:%lld\n", starti, stopi, preindex->ikey_length);

    for (int64_t j = starti; j < stopi; j++)
    {
        KEYTYPE index = pIKey[j];
        if (index <= 0 || index > uKeyLength)
        {
            // preserve filtered out or mark as filtered if out of range
            pIKey[j] = 0;
        }
        else
        {
            // reindex ikey inplace
            pIKey[j] = (KEYTYPE)pUniques[index - 1];
        }
    }

    return true;
}

//---------------------------------------------------------------------------
// Input:
// Arg1: ikey numpy array of old ikey after hstack or multistack load
// Arg2: uikey numpy array unique index
// Arg3: u_cutoffs
// Arg3: i_cutoffs
//
PyObject * ReIndexGroups(PyObject * self, PyObject * args)
{
    PyArrayObject * ikey = NULL;
    PyArrayObject * uikey = NULL;
    PyArrayObject * u_cutoffs = NULL;
    PyArrayObject * i_cutoffs = NULL;

    if (! PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &ikey, &PyArray_Type, &uikey, &PyArray_Type, &u_cutoffs, &PyArray_Type,
                           &i_cutoffs))
    {
        return NULL;
    }
    if (PyArray_ITEMSIZE(u_cutoffs) != 8)
    {
        PyErr_Format(PyExc_ValueError, "u-cutoffs must be int64");
        return NULL;
    }
    if (PyArray_ITEMSIZE(i_cutoffs) != 8)
    {
        PyErr_Format(PyExc_ValueError, "i-cutoffs must be int64");
        return NULL;
    }

    if (PyArray_ITEMSIZE(uikey) != 4)
    {
        PyErr_Format(PyExc_ValueError, "uikey must be int32");
        return NULL;
    }

    int64_t u_cutoffs_length = ArrayLength(u_cutoffs);

    stReIndex preindex;

    preindex.pUCutOffs = (int64_t *)PyArray_BYTES(u_cutoffs);
    preindex.pICutOffs = (int64_t *)PyArray_BYTES(i_cutoffs);
    preindex.pUKey = (int32_t *)PyArray_BYTES(uikey);
    preindex.pIKey = PyArray_BYTES(ikey);

    preindex.ikey_length = ArrayLength(ikey);
    preindex.u_cutoffs_length = u_cutoffs_length;
    preindex.uikey_length = ArrayLength(uikey);

    switch (PyArray_ITEMSIZE(ikey))
    {
    case 1:
        g_cMathWorker->DoMultiThreadedWork((int)u_cutoffs_length, ReIndexGroupsMT<int8_t>, &preindex);
        break;
    case 2:
        g_cMathWorker->DoMultiThreadedWork((int)u_cutoffs_length, ReIndexGroupsMT<int16_t>, &preindex);
        break;
    case 4:
        g_cMathWorker->DoMultiThreadedWork((int)u_cutoffs_length, ReIndexGroupsMT<int32_t>, &preindex);
        break;
    case 8:
        g_cMathWorker->DoMultiThreadedWork((int)u_cutoffs_length, ReIndexGroupsMT<int64_t>, &preindex);
        break;
    default:
        PyErr_Format(PyExc_ValueError, "ikey must be int8/16/32/64");
        return NULL;
    }

    Py_IncRef((PyObject *)ikey);
    return (PyObject *)ikey;
}

struct stReverseIndex
{
    void * pIKey;
    void * pOutKey;
    int64_t ikey_length;
};

// This routine is parallelized
// Algo: out[in[i]] = i
template <typename KEYTYPE>
bool ReverseShuffleMT(void * preindexV, int core, int64_t start, int64_t length)
{
    stReverseIndex * preindex = (stReverseIndex *)preindexV;

    KEYTYPE * pIn = (KEYTYPE *)preindex->pIKey;
    KEYTYPE * pOut = (KEYTYPE *)preindex->pOutKey;
    int64_t maxindex = preindex->ikey_length;

    for (int64_t i = start; i < (start + length); i++)
    {
        KEYTYPE index = pIn[i];
        if (index >= 0 && index < maxindex)
        {
            pOut[index] = (KEYTYPE)i;
        }
    }
    return true;
}

//---------------------------------------------------------------------------
// Input:
// Arg1: ikey numpy array from lexsort or grouping.iGroup
//       array must be integers
//       array must have integers only from 0 to len(arr)-1
//       all values must be unique, then it can be reversed quickly
//
//       if "in" is the input array and "out" is the output array
//       out[in[i]] = i
// Output:
//      Returns index array with indexes reversed back prior to lexsort
PyObject * ReverseShuffle(PyObject * self, PyObject * args)
{
    PyArrayObject * ikey = NULL;

    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &ikey))
    {
        return NULL;
    }

    stReverseIndex preindex;

    int dtype = PyArray_TYPE(ikey);

    // check for only signed ints
    if (dtype >= 10 || (dtype & 1) == 0)
    {
        PyErr_Format(PyExc_ValueError, "ReverseShuffle: ikey must be int8/16/32/64");
        return NULL;
    }

    PyArrayObject * pReturnArray = AllocateLikeNumpyArray(ikey, dtype);

    if (pReturnArray)
    {
        preindex.pIKey = PyArray_BYTES(ikey);
        preindex.pOutKey = PyArray_BYTES(pReturnArray);

        int64_t arrlength = ArrayLength(ikey);
        preindex.ikey_length = arrlength;

        switch (PyArray_ITEMSIZE(ikey))
        {
        case 1:
            g_cMathWorker->DoMultiThreadedChunkWork(arrlength, ReverseShuffleMT<int8_t>, &preindex);
            break;
        case 2:
            g_cMathWorker->DoMultiThreadedChunkWork(arrlength, ReverseShuffleMT<int16_t>, &preindex);
            break;
        case 4:
            g_cMathWorker->DoMultiThreadedChunkWork(arrlength, ReverseShuffleMT<int32_t>, &preindex);
            break;
        case 8:
            g_cMathWorker->DoMultiThreadedChunkWork(arrlength, ReverseShuffleMT<int64_t>, &preindex);
            break;
        default:
            PyErr_Format(PyExc_ValueError, "ReverseShuffle: ikey must be int8/16/32/64");
            return NULL;
        }

        return (PyObject *)pReturnArray;
    }

    PyErr_Format(PyExc_ValueError, "ReverseShuffle: ran out of memory");
    return NULL;
}

/*
//-----------------------------------------------------
PyObject *
MergeBinnedCutoffs(PyObject *self, PyObject *args) {

   //#--------- START OF C++ ROUTINE -------------
   //   #based on how many uniques we have, allocate the new ikey
   //      # do we have a routine for this ?
   //      uikey_length = len(uikey)
   //      if uikey_length < 100 :
   //         dtype = np.int8
   //         elif uikey_length < 30_000 :
   //         dtype = np.int16
   //         elif uikey_length < 2_000_000_000 :
   //         dtype = np.int32
   //      else:
   //   dtype = np.int64
   //
   //      newikey = empty((len(ikey), ), dtype = dtype)
   //
   //      start = 0
   //      starti = 0
   //      for i in range(len(u_cutoffs)) :
   //         stop = u_cutoffs[i]
   //         stopi = i_cutoffs[i]
   //         uikey_slice = uikey[start:stop]
   //         oldikey_slice = ikey[starti:stopi]
   //
   //         if verbose:
   //   print("fixing ", starti, stopi)
   //      print("newikey ", newikey)
   //      print("oldikey_slice ", oldikey_slice)
   //
   //      # write a routine for this in C++
   //# if 0 and base_index=1, then keep the 0
   //      newikey[starti:stopi] = uikey_slice[oldikey_slice - 1]
   //      start = stop
   //      starti = stopi
   //      #END C++ ROUTINE-------------------------------- -

   Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

   if (argTupleSize < 3) {
      PyErr_Format(PyExc_ValueError, "SetItem requires three args instead of
%llu args", argTupleSize); return NULL;
   }

   PyArrayObject* arr = (PyArrayObject*)PyTuple_GetItem(args, 0);
   PyArrayObject* mask = (PyArrayObject*)PyTuple_GetItem(args, 1);

   // Try to convert value if we have to
   PyObject* value = PyTuple_GetItem(args, 2);
   if (!PyArray_Check(value)) {
      value = PyArray_FromAny(value, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
   }

   if (PyArray_Check(arr) && PyArray_Check(mask) && PyArray_Check(value)) {
      PyArrayObject* inValues = (PyArrayObject*)value;

      if (PyArray_TYPE(mask) == NPY_BOOL) {

         int64_t itemSizeOut = PyArray_ITEMSIZE(arr);
         int64_t itemSizeIn = PyArray_ITEMSIZE(inValues);

         // check for strides... ?
         int64_t arrayLength = ArrayLength(arr);
         if (arrayLength == ArrayLength(mask) && itemSizeOut ==
PyArray_STRIDE(arr, 0)) { int64_t valLength = ArrayLength(inValues);

            if (arrayLength == valLength) {
               int outDType = PyArray_TYPE(arr);
               int inDType = PyArray_TYPE(inValues);
               MASK_CONVERT_SAFE maskSafe = GetConversionPutMask(inDType,
outDType);

               if (maskSafe) {

                  // MT callback
                  struct MASK_CALLBACK_STRUCT {
                     MASK_CONVERT_SAFE maskSafe;
                     char* pIn;
                     char* pOut;
                     int64_t itemSizeOut;
                     int64_t itemSizeIn;
                     int8_t* pMask;
                     void* pBadInput1;
                     void* pBadOutput1;

                  };

                  MASK_CALLBACK_STRUCT stMask;

                  // This is the routine that will be called back from multiple
threads auto lambdaMaskCallback = [](void* callbackArgT, int core, int64_t
start, int64_t length) -> bool { MASK_CALLBACK_STRUCT* callbackArg =
(MASK_CALLBACK_STRUCT*)callbackArgT;

                     //printf("[%d] Mask %lld %lld\n", core, start, length);
                     //maskSafe(pIn, pOut, (int8_t*)pMask, length, pBadInput1,
pBadOutput1);
                     // Auto adjust pointers
                     callbackArg->maskSafe(
                        callbackArg->pIn + (start * callbackArg->itemSizeIn),
                        callbackArg->pOut + (start * callbackArg->itemSizeOut),
                        callbackArg->pMask + start,
                        length,
                        callbackArg->pBadInput1,
                        callbackArg->pBadOutput1);

                     return true;
                  };

                  stMask.itemSizeIn = itemSizeIn;
                  stMask.itemSizeOut = itemSizeOut;
                  stMask.pBadInput1 = GetDefaultForType(inDType);
                  stMask.pBadOutput1 = GetDefaultForType(outDType);

                  stMask.pIn = (char*)PyArray_BYTES(inValues, 0);
                  stMask.pOut = (char*)PyArray_BYTES(arr, 0);
                  stMask.pMask = (int8_t*)PyArray_BYTES(mask, 0);
                  stMask.maskSafe = maskSafe;

                  g_cMathWorker->DoMultiThreadedChunkWork(arrayLength,
lambdaMaskCallback, &stMask);

                  Py_IncRef(Py_True);
                  return Py_True;
               }
            }
         }
      }
   }

   // punt to numpy
   Py_IncRef(Py_False);
   return Py_False;



}

*/
