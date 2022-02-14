#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "Convert.h"
#include "MultiKey.h"
#include "Recycler.h"
#include "Reduce.h"

#include <algorithm>

#define LOGGING(...)
//#define LOGGING printf

#if defined(_MSC_VER) && _MSC_VER < 1910
    // The VS2015 compiler doesn't provide the _mm256_extract_epi64() intrinsic,
    // even though that intrinsic is supposed to be available as part of AVX
    // support. Define a compatible version of the function using intrinsics that
    // _are_ available in that compiler.

    // Extract a 64-bit integer from a, selected with imm8.
    #define _mm256_extract_epi64(a, imm8) _mm_extract_epi64(_mm256_extracti128_si256((a), imm8 / 2), imm8 % 2)

#endif // _MSC_VER

// SIMD conversion functions for integers
// extern __m256i __cdecl _mm256_cvtepi8_epi16(__m128i);
// extern __m256i __cdecl _mm256_cvtepi8_epi32(__m128i);
// extern __m256i __cdecl _mm256_cvtepi8_epi64(__m128i);
// extern __m256i __cdecl _mm256_cvtepi16_epi32(__m128i);
// extern __m256i __cdecl _mm256_cvtepi16_epi64(__m128i);
// extern __m256i __cdecl _mm256_cvtepi32_epi64(__m128i);
//
// extern __m256i __cdecl _mm256_cvtepu8_epi16(__m128i);
// extern __m256i __cdecl _mm256_cvtepu8_epi32(__m128i);
// extern __m256i __cdecl _mm256_cvtepu8_epi64(__m128i);
// extern __m256i __cdecl _mm256_cvtepu16_epi32(__m128i);
// extern __m256i __cdecl _mm256_cvtepu16_epi64(__m128i);
// extern __m256i __cdecl _mm256_cvtepu32_epi64(__m128i);
typedef void (*CONVERT_SAFE)(void * pDataIn, void * pDataOut, int64_t len, void * pBadInput1, void * pBadOutput1, int64_t strideIn,
                             int64_t strideOut);
typedef void (*MASK_CONVERT_SAFE)(void * pDataIn, void * pDataOut, int8_t * pMask, int64_t len, void * pBadInput1,
                                  void * pBadOutput1);
typedef void (*CONVERT_SAFE_STRING)(void * pDataIn, void * pDataOut, int64_t len, int64_t inputItemSize, int64_t outputItemSize);

static void ConvertSafeStringCopy(void * pDataIn, void * pDataOut, int64_t len, int64_t inputItemSize, int64_t outputItemSize)
{
    LOGGING("String convert %lld %lld\n", inputItemSize, outputItemSize);
    if (inputItemSize == outputItemSize)
    {
        // straight memcpy
        memcpy(pDataOut, pDataIn, len * inputItemSize);
    }
    else
    {
        if (inputItemSize < outputItemSize)
        {
            char * pOut = (char *)pDataOut;
            char * pIn = (char *)pDataIn;
            int64_t remain = outputItemSize - inputItemSize;

            if (inputItemSize >= 8)
            {
                for (int64_t i = 0; i < len; i++)
                {
                    memcpy(pOut, pIn, inputItemSize);
                    pOut += inputItemSize;
                    for (int64_t j = 0; j < remain; j++)
                    {
                        pOut[j] = 0;
                    }

                    pOut += remain;
                    pIn += inputItemSize;
                }
            }
            else
            {
                for (int64_t i = 0; i < len; i++)
                {
                    for (int64_t j = 0; j < inputItemSize; j++)
                    {
                        pOut[j] = pIn[j];
                    }
                    pOut += inputItemSize;

                    // consider memset
                    for (int64_t j = 0; j < remain; j++)
                    {
                        pOut[j] = 0;
                    }

                    pOut += remain;
                    pIn += inputItemSize;
                }
            }
        }
        else
        {
            // currently not possible (clipping input)
            char * pOut = (char *)pDataOut;
            char * pIn = (char *)pDataIn;

            for (int64_t i = 0; i < len; i++)
            {
                memcpy(pOut, pIn, outputItemSize);
                pOut += outputItemSize;
                pIn += inputItemSize;
            }
        }
    }
}

//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// thus <float, int32> converts a float to an int32
template <typename T, typename U>
class ConvertBase
{
public:
    ConvertBase(){};
    ~ConvertBase(){};

    static void PutMaskFast(void * pDataIn, void * pDataOut, int8_t * pMask, int64_t len, void * pBadInput1, void * pBadOutput1)
    {
        T * pIn = (T *)pDataIn;
        T * pOut = (T *)pDataOut;

        // TODO can be made faster by pulling 8 bytes at once
        for (int i = 0; i < len; i++)
        {
            if (pMask[i])
            {
                pOut[i] = pIn[i];
            }
        }
    }

    static void PutMaskCopy(void * pDataIn, void * pDataOut, int8_t * pMask, int64_t len, void * pBadInput1, void * pBadOutput1)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        T pBadValueIn = *(T *)pBadInput1;
        U pBadValueOut = *(U *)pBadOutput1;

        for (int i = 0; i < len; i++)
        {
            if (pMask[i])
            {
                if (pIn[i] != pBadValueIn)
                {
                    pOut[i] = (U)pIn[i];
                }
                else
                {
                    pOut[i] = pBadValueOut;
                }
            }
        }
    }

    static void PutMaskCopyBool(void * pDataIn, void * pDataOut, int8_t * pMask, int64_t len, void * pBadInput1,
                                void * pBadOutput1)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        T pBadValueIn = *(T *)pBadInput1;
        U pBadValueOut = *(U *)pBadOutput1;

        for (int i = 0; i < len; i++)
        {
            if (pMask[i])
            {
                pOut[i] = pIn[i] != 0;
            }
        }
    }

    static void PutMaskCopyFloat(void * pDataIn, void * pDataOut, int8_t * pMask, int64_t len, void * pBadInput1,
                                 void * pBadOutput1)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        U pBadValueOut = *(U *)pBadOutput1;

        for (int i = 0; i < len; i++)
        {
            if (pMask[i])
            {
                if (pIn[i] == pIn[i])
                {
                    pOut[i] = (U)pIn[i];
                }
                else
                {
                    pOut[i] = pBadValueOut;
                }
            }
        }
    }

    // Pass in one vector and returns converted vector
    // Used for operations like C = A + B
    // typedef void(*ANY_TWO_FUNC)(void* pDataIn, void* pDataIn2, void* pDataOut,
    // int64_t len, int32_t scalarMode); typedef void(*ANY_ONE_FUNC)(void* pDataIn,
    // void* pDataOut, int64_t len);
    static void OneStubConvert(void * pDataIn, void * pDataOut, int64_t len, int64_t strideIn, int64_t strideOut)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        if (strideIn == sizeof(T) && strideOut == sizeof(U))
        {
            // How to handle nan conversions?
            // NAN converts to MININT (for float --> int conversion)
            // then the reverse, MIININT converts to NAN (for int --> float
            // conversion) convert from int --> float check for NPY_MIN_INT64_t
            for (int64_t i = 0; i < len; i++)
            {
                pOut[i] = (U)pIn[i];
            }
        }
        else
        {
            // Strided loop
            U * pEndOut = (U *)((char *)pOut + (len * strideOut));
            while (pOut != pEndOut)
            {
                *pOut = (U)*pIn;
                pIn = STRIDE_NEXT(T, pIn, strideIn);
                pOut = STRIDE_NEXT(U, pOut, strideOut);
            }
        }
    }

    static void OneStubConvertSafeCopy(void * pDataIn, void * pDataOut, int64_t len, void * pBadInput1, void * pBadOutput1,
                                       int64_t strideIn, int64_t strideOut)
    {
        // include memcpy with stride
        if (strideIn == sizeof(T) && strideOut == sizeof(U))
        {
            memcpy(pDataOut, pDataIn, len * sizeof(U));
        }
        else
        {
            T * pIn = (T *)pDataIn;
            U * pOut = (U *)pDataOut;

            // Strided loop
            U * pEndOut = (U *)((char *)pOut + (len * strideOut));
            while (pOut != pEndOut)
            {
                *pOut = *pIn;
                pIn = STRIDE_NEXT(T, pIn, strideIn);
                pOut = STRIDE_NEXT(U, pOut, strideOut);
            }
        }
    }

    //------------------------------------------------------------------------
    // Designed NOT to preserve sentinels
    static void OneStubConvertUnsafe(void * pDataIn, void * pDataOut, int64_t len, void * pBadInput1, void * pBadOutput1,
                                     int64_t strideIn, int64_t strideOut)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        if (strideIn == sizeof(T) && strideOut == sizeof(U))
        {
            for (int64_t i = 0; i < len; i++)
            {
                pOut[i] = (U)pIn[i];
            }
        }
        else
        {
            // Strided loop
            U * pEndOut = (U *)((char *)pOut + (len * strideOut));
            while (pOut != pEndOut)
            {
                *pOut = (U)*pIn;
                pIn = STRIDE_NEXT(T, pIn, strideIn);
                pOut = STRIDE_NEXT(U, pOut, strideOut);
            }
        }
    }

    //------------------------------------------------------------------------
    // Designed to preserve sentinels
    // NOTE: discussion on on what happens with uint8_t conversion (may or may not
    // ignore 0xFF on conversion since so common)
    static void OneStubConvertSafe(void * pDataIn, void * pDataOut, int64_t len, void * pBadInput1, void * pBadOutput1,
                                   int64_t strideIn, int64_t strideOut)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        T pBadValueIn = *(T *)pBadInput1;
        U pBadValueOut = *(U *)pBadOutput1;

        // How to handle nan conversions?
        // NAN converts to MININT (for float --> int conversion)
        // then the reverse, MIININT converts to NAN (for int --> float conversion)
        // convert from int --> float
        // check for NPY_MIN_INT64_t
        if (strideIn == sizeof(T) && strideOut == sizeof(U))
        {
            for (int64_t i = 0; i < len; i++)
            {
                if (pIn[i] != pBadValueIn)
                {
                    pOut[i] = (U)pIn[i];
                }
                else
                {
                    pOut[i] = pBadValueOut;
                }
            }
        }
        else
        {
            // Strided loop
            U * pEndOut = (U *)((char *)pOut + (len * strideOut));
            while (pOut != pEndOut)
            {
                if (*pIn != pBadValueIn)
                {
                    *pOut = (U)*pIn;
                }
                else
                {
                    *pOut = pBadValueOut;
                }
                pIn = STRIDE_NEXT(T, pIn, strideIn);
                pOut = STRIDE_NEXT(U, pOut, strideOut);
            }
        }
    }

    static void OneStubConvertSafeFloatToDouble(void * pDataIn, void * pDataOut, int64_t len, void * pBadInput1,
                                                void * pBadOutput1, int64_t strideIn, int64_t strideOut)
    {
        float * pIn = (float *)pDataIn;
        double * pOut = (double *)pDataOut;

        if (strideIn == sizeof(float) && strideOut == sizeof(double))
        {
            const double * pEndOut = (double *)((char *)pOut + (len * strideOut));
            const double * pEndOut8 = pEndOut - 8;
            while (pOut <= pEndOut8)
            {
                __m256 m0 = _mm256_loadu_ps(pIn);
                _mm256_storeu_pd(pOut, _mm256_cvtps_pd(_mm256_extractf128_ps(m0, 0)));
                _mm256_storeu_pd(pOut + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(m0, 1)));
                pOut += 8;
                pIn += 8;
            }
            while (pOut != pEndOut)
            {
                *pOut++ = (double)*pIn++;
            }
        }
        else
        {
            // Strided loop
            double * pEndOut = (double *)((char *)pOut + (len * strideOut));
            while (pOut != pEndOut)
            {
                *pOut = (double)*pIn;
                pIn = STRIDE_NEXT(float, pIn, strideIn);
                pOut = STRIDE_NEXT(double, pOut, strideOut);
            }
        }
    }

    static void OneStubConvertSafeDoubleToFloat(void * pDataIn, void * pDataOut, int64_t len, void * pBadInput1,
                                                void * pBadOutput1, int64_t strideIn, int64_t strideOut)
    {
        double * pIn = (double *)pDataIn;
        float * pOut = (float *)pDataOut;

        if (strideIn == sizeof(double) && strideOut == sizeof(float))
        {
            for (int64_t i = 0; i < len; i++)
            {
                pOut[i] = (float)pIn[i];
            }
        }
        else
        {
            // Strided loop
            float * pEndOut = (float *)((char *)pOut + (len * strideOut));
            while (pOut != pEndOut)
            {
                *pOut = (float)*pIn;
                pIn = STRIDE_NEXT(double, pIn, strideIn);
                pOut = STRIDE_NEXT(float, pOut, strideOut);
            }
        }
    }

    static void OneStubConvertSafeFloat(void * pDataIn, void * pDataOut, int64_t len, void * pBadInput1, void * pBadOutput1,
                                        int64_t strideIn, int64_t strideOut)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        T pBadValueIn = *(T *)pBadInput1;
        U pBadValueOut = *(U *)pBadOutput1;

        // How to handle nan conversions?
        // NAN converts to MININT (for float --> int conversion)
        // then the reverse, MIININT converts to NAN (for int --> float conversion)
        // convert from int --> float
        // check for NPY_MIN_INT64_t
        if (strideIn == sizeof(T) && strideOut == sizeof(U))
        {
            for (int64_t i = 0; i < len; i++)
            {
                if (std::isfinite(pIn[i]) && (pIn[i] != pBadValueIn))
                {
                    pOut[i] = (U)pIn[i];
                }
                else
                {
                    pOut[i] = pBadValueOut;
                }
            }
        }
        else
        {
            // Strided loop
            U * pEndOut = (U *)((char *)pOut + (len * strideOut));
            while (pOut != pEndOut)
            {
                if (std::isfinite(*pIn) && (*pIn != pBadValueIn))
                {
                    *pOut = (U)*pIn;
                }
                else
                {
                    *pOut = pBadValueOut;
                }
                pIn = STRIDE_NEXT(T, pIn, strideIn);
                pOut = STRIDE_NEXT(U, pOut, strideOut);
            }
        }
    }

    static void OneStubConvertSafeBool(void * pDataIn, void * pDataOut, int64_t len, void * pBadInput1, void * pBadOutput1,
                                       int64_t strideIn, int64_t strideOut)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        // How to handle nan conversions?
        // NAN converts to MININT (for float --> int conversion)
        // then the reverse, MIININT converts to NAN (for int --> float conversion)
        // convet from float --> int
        if (strideIn == sizeof(T) && strideOut == sizeof(U))
        {
            for (int64_t i = 0; i < len; i++)
            {
                pOut[i] = (U)(pIn[i] != 0);
            }
        }
        else
        {
            // Strided loop
            U * pEndOut = (U *)((char *)pOut + (len * strideOut));
            while (pOut != pEndOut)
            {
                *pOut = (U)(*pIn != 0);
                pIn = STRIDE_NEXT(T, pIn, strideIn);
                pOut = STRIDE_NEXT(U, pOut, strideOut);
            }
        }
    }

    static void OneStubConvertBool(void * pDataIn, void * pDataOut, int64_t len, int64_t strideIn, int64_t strideOut)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        if (strideIn == sizeof(T) && strideOut == sizeof(U))
        {
            // How to handle nan conversions?
            // NAN converts to MININT (for float --> int conversion)
            // then the reverse, MIININT converts to NAN (for int --> float
            // conversion) convet from float --> int
            for (int64_t i = 0; i < len; i++)
            {
                pOut[i] = (U)(pIn[i] != 0);
            }
        }
        else
        {
            // Strided loop
            U * pEndOut = (U *)((char *)pOut + (len * strideOut));
            while (pOut != pEndOut)
            {
                *pOut = (U)(*pIn != 0);
                pIn = STRIDE_NEXT(T, pIn, strideIn);
                pOut = STRIDE_NEXT(U, pOut, strideOut);
            }
        }
    }
};

template <typename T>
static UNARY_FUNC GetConversionStep2(int outputType)
{
    switch (outputType)
    {
    case NPY_BOOL:
        return ConvertBase<T, bool>::OneStubConvertBool;
    case NPY_FLOAT:
        return ConvertBase<T, float>::OneStubConvert;
    case NPY_DOUBLE:
        return ConvertBase<T, double>::OneStubConvert;
    case NPY_LONGDOUBLE:
        return ConvertBase<T, long double>::OneStubConvert;
    case NPY_BYTE:
        return ConvertBase<T, int8_t>::OneStubConvert;
    case NPY_INT16:
        return ConvertBase<T, int16_t>::OneStubConvert;
    CASE_NPY_INT32:
        return ConvertBase<T, int32_t>::OneStubConvert;
    CASE_NPY_INT64:

        return ConvertBase<T, int64_t>::OneStubConvert;
    case NPY_UBYTE:
        return ConvertBase<T, uint8_t>::OneStubConvert;
    case NPY_UINT16:
        return ConvertBase<T, uint16_t>::OneStubConvert;
    CASE_NPY_UINT32:
        return ConvertBase<T, uint32_t>::OneStubConvert;
    CASE_NPY_UINT64:

        return ConvertBase<T, uint64_t>::OneStubConvert;
    }
    return NULL;
}

template <typename T>
static CONVERT_SAFE GetConversionStep2Safe(int outputType)
{
    switch (outputType)
    {
    case NPY_BOOL:
        return ConvertBase<T, bool>::OneStubConvertSafeBool;
    case NPY_FLOAT:
        return ConvertBase<T, float>::OneStubConvertSafe;
    case NPY_DOUBLE:
        return ConvertBase<T, double>::OneStubConvertSafe;
    case NPY_LONGDOUBLE:
        return ConvertBase<T, long double>::OneStubConvertSafe;
    case NPY_BYTE:
        return ConvertBase<T, int8_t>::OneStubConvertSafe;
    case NPY_INT16:
        return ConvertBase<T, int16_t>::OneStubConvertSafe;
    CASE_NPY_INT32:
        return ConvertBase<T, int32_t>::OneStubConvertSafe;
    CASE_NPY_INT64:

        return ConvertBase<T, int64_t>::OneStubConvertSafe;
    case NPY_UBYTE:
        return ConvertBase<T, uint8_t>::OneStubConvertSafe;
    case NPY_UINT16:
        return ConvertBase<T, uint16_t>::OneStubConvertSafe;
    CASE_NPY_UINT32:
        return ConvertBase<T, uint32_t>::OneStubConvertSafe;
    CASE_NPY_UINT64:

        return ConvertBase<T, uint64_t>::OneStubConvertSafe;
    }
    return NULL;
}

//-----------------------------------------------------------
// Used when converting from a uint8_t which has no sentinel (discussion point)
template <typename T>
static CONVERT_SAFE GetConversionStep2Unsafe(int outputType)
{
    switch (outputType)
    {
    case NPY_BOOL:
        return ConvertBase<T, bool>::OneStubConvertSafeBool;
    case NPY_FLOAT:
        return ConvertBase<T, float>::OneStubConvertUnsafe;
    case NPY_DOUBLE:
        return ConvertBase<T, double>::OneStubConvertUnsafe;
    case NPY_LONGDOUBLE:
        return ConvertBase<T, long double>::OneStubConvertUnsafe;
    case NPY_BYTE:
        return ConvertBase<T, int8_t>::OneStubConvertUnsafe;
    case NPY_INT16:
        return ConvertBase<T, int16_t>::OneStubConvertUnsafe;
    CASE_NPY_INT32:
        return ConvertBase<T, int32_t>::OneStubConvertUnsafe;
    CASE_NPY_INT64:

        return ConvertBase<T, int64_t>::OneStubConvertUnsafe;
    case NPY_UBYTE:
        return ConvertBase<T, uint8_t>::OneStubConvertUnsafe;
    case NPY_UINT16:
        return ConvertBase<T, uint16_t>::OneStubConvertUnsafe;
    CASE_NPY_UINT32:
        return ConvertBase<T, uint32_t>::OneStubConvertUnsafe;
    CASE_NPY_UINT64:

        return ConvertBase<T, uint64_t>::OneStubConvertUnsafe;
    }
    return NULL;
}

template <typename T>
static CONVERT_SAFE GetConversionStep2SafeFromFloat(int outputType)
{
    switch (outputType)
    {
    case NPY_BOOL:
        return ConvertBase<T, bool>::OneStubConvertSafeBool;
    case NPY_FLOAT:
        return ConvertBase<T, float>::OneStubConvertSafeFloat;
    case NPY_DOUBLE:
        return ConvertBase<T,
                           double>::OneStubConvertSafeFloatToDouble; // very common
    case NPY_LONGDOUBLE:
        return ConvertBase<T, long double>::OneStubConvertSafeFloat;
    case NPY_BYTE:
        return ConvertBase<T, int8_t>::OneStubConvertSafeFloat;
    case NPY_INT16:
        return ConvertBase<T, int16_t>::OneStubConvertSafeFloat;
    CASE_NPY_INT32:
        return ConvertBase<T, int32_t>::OneStubConvertSafeFloat;
    CASE_NPY_INT64:

        return ConvertBase<T, int64_t>::OneStubConvertSafeFloat;
    case NPY_UBYTE:
        return ConvertBase<T, uint8_t>::OneStubConvertSafeFloat;
    case NPY_UINT16:
        return ConvertBase<T, uint16_t>::OneStubConvertSafeFloat;
    CASE_NPY_UINT32:
        return ConvertBase<T, uint32_t>::OneStubConvertSafeFloat;
    CASE_NPY_UINT64:

        return ConvertBase<T, uint64_t>::OneStubConvertSafeFloat;
    }
    return NULL;
}

template <typename T>
static CONVERT_SAFE GetConversionStep2SafeFromDouble(int outputType)
{
    switch (outputType)
    {
    case NPY_BOOL:
        return ConvertBase<T, bool>::OneStubConvertSafeBool;
    case NPY_FLOAT:
        return ConvertBase<T,
                           float>::OneStubConvertSafeDoubleToFloat; // very common
    case NPY_DOUBLE:
        return ConvertBase<T, double>::OneStubConvertSafeFloat;
    case NPY_LONGDOUBLE:
        return ConvertBase<T, long double>::OneStubConvertSafeFloat;
    case NPY_BYTE:
        return ConvertBase<T, int8_t>::OneStubConvertSafeFloat;
    case NPY_INT16:
        return ConvertBase<T, int16_t>::OneStubConvertSafeFloat;
    CASE_NPY_INT32:
        return ConvertBase<T, int32_t>::OneStubConvertSafeFloat;
    CASE_NPY_INT64:

        return ConvertBase<T, int64_t>::OneStubConvertSafeFloat;
    case NPY_UBYTE:
        return ConvertBase<T, uint8_t>::OneStubConvertSafeFloat;
    case NPY_UINT16:
        return ConvertBase<T, uint16_t>::OneStubConvertSafeFloat;
    CASE_NPY_UINT32:
        return ConvertBase<T, uint32_t>::OneStubConvertSafeFloat;
    CASE_NPY_UINT64:

        return ConvertBase<T, uint64_t>::OneStubConvertSafeFloat;
    }
    return NULL;
}

template <typename T>
static CONVERT_SAFE GetConversionStep2SafeFloat(int outputType)
{
    switch (outputType)
    {
    case NPY_BOOL:
        return ConvertBase<T, bool>::OneStubConvertSafeBool;
    case NPY_FLOAT:
        return ConvertBase<T, float>::OneStubConvertSafeFloat;
    case NPY_DOUBLE:
        return ConvertBase<T, double>::OneStubConvertSafeFloat;
    case NPY_LONGDOUBLE:
        return ConvertBase<T, long double>::OneStubConvertSafeFloat;
    case NPY_BYTE:
        return ConvertBase<T, int8_t>::OneStubConvertSafeFloat;
    case NPY_INT16:
        return ConvertBase<T, int16_t>::OneStubConvertSafeFloat;
    CASE_NPY_INT32:
        return ConvertBase<T, int32_t>::OneStubConvertSafeFloat;
    CASE_NPY_INT64:

        return ConvertBase<T, int64_t>::OneStubConvertSafeFloat;
    case NPY_UBYTE:
        return ConvertBase<T, uint8_t>::OneStubConvertSafeFloat;
    case NPY_UINT16:
        return ConvertBase<T, uint16_t>::OneStubConvertSafeFloat;
    CASE_NPY_UINT32:
        return ConvertBase<T, uint32_t>::OneStubConvertSafeFloat;
    CASE_NPY_UINT64:

        return ConvertBase<T, uint64_t>::OneStubConvertSafeFloat;
    }
    return NULL;
}

static CONVERT_SAFE GetConversionFunctionSafeCopy(int inputType)
{
    switch (inputType)
    {
    case NPY_BYTE:
    case NPY_UBYTE:
    case NPY_BOOL:
        return ConvertBase<int8_t, int8_t>::OneStubConvertSafeCopy;

    case NPY_INT16:
    case NPY_UINT16:
        return ConvertBase<int16_t, int16_t>::OneStubConvertSafeCopy;

    CASE_NPY_INT32:
    CASE_NPY_UINT32:
    case NPY_FLOAT:
        return ConvertBase<int32_t, int32_t>::OneStubConvertSafeCopy;

    CASE_NPY_INT64:

    CASE_NPY_UINT64:

    case NPY_DOUBLE:
        return ConvertBase<int64_t, int64_t>::OneStubConvertSafeCopy;

    case NPY_LONGDOUBLE:
        return ConvertBase<long double, long double>::OneStubConvertSafeCopy;
    }
    return NULL;
}

static CONVERT_SAFE GetConversionFunctionSafe(int inputType, int outputType)
{
    // check for same type -- which is shorthand for copy
    if (inputType == outputType)
    {
        return GetConversionFunctionSafeCopy(inputType);
    }

    switch (inputType)
    {
    // case NPY_BOOL:   return GetConversionStep2Safe<bool>(outputType);
    case NPY_BOOL:
        return GetConversionStep2Safe<int8_t>(outputType);
    case NPY_FLOAT:
        return GetConversionStep2SafeFromFloat<float>(outputType);
    case NPY_DOUBLE:
        return GetConversionStep2SafeFromDouble<double>(outputType);
    case NPY_LONGDOUBLE:
        return GetConversionStep2SafeFloat<long double>(outputType);
    case NPY_BYTE:
        return GetConversionStep2Safe<int8_t>(outputType);
    case NPY_INT16:
        return GetConversionStep2Safe<int16_t>(outputType);
    CASE_NPY_INT32:
        return GetConversionStep2Safe<int32_t>(outputType);
    CASE_NPY_INT64:

        return GetConversionStep2Safe<int64_t>(outputType);

    // DISCUSSION -- uint8_t and the value 255 or 0xFF will not be a sentinel
    // case NPY_UBYTE:  return GetConversionStep2Unsafe<uint8_t>(outputType);
    case NPY_UBYTE:
        return GetConversionStep2Safe<uint8_t>(outputType);

    case NPY_UINT16:
        return GetConversionStep2Safe<uint16_t>(outputType);
    CASE_NPY_UINT32:
        return GetConversionStep2Safe<uint32_t>(outputType);
    CASE_NPY_UINT64:

        return GetConversionStep2Safe<uint64_t>(outputType);
    }
    return NULL;
}

static CONVERT_SAFE GetConversionFunctionUnsafe(int inputType, int outputType)
{
    // check for same type -- which is shorthand for copy
    if (inputType == outputType)
    {
        return GetConversionFunctionSafeCopy(inputType);
    }

    switch (inputType)
    {
        // case NPY_BOOL:   return GetConversionStep2Safe<bool>(outputType);
    case NPY_BOOL:
        return GetConversionStep2Unsafe<int8_t>(outputType);
    case NPY_FLOAT:
        return GetConversionStep2Unsafe<float>(outputType);
    case NPY_DOUBLE:
        return GetConversionStep2Unsafe<double>(outputType);
    case NPY_LONGDOUBLE:
        return GetConversionStep2Unsafe<long double>(outputType);
    case NPY_BYTE:
        return GetConversionStep2Unsafe<int8_t>(outputType);
    case NPY_INT16:
        return GetConversionStep2Unsafe<int16_t>(outputType);
    CASE_NPY_INT32:
        return GetConversionStep2Unsafe<int32_t>(outputType);
    CASE_NPY_INT64:

        return GetConversionStep2Unsafe<int64_t>(outputType);
    case NPY_UBYTE:
        return GetConversionStep2Unsafe<uint8_t>(outputType);
    case NPY_UINT16:
        return GetConversionStep2Unsafe<uint16_t>(outputType);
    CASE_NPY_UINT32:
        return GetConversionStep2Unsafe<uint32_t>(outputType);
    CASE_NPY_UINT64:

        return GetConversionStep2Unsafe<uint64_t>(outputType);
    }
    return NULL;
}

template <typename T>
static MASK_CONVERT_SAFE GetConversionPutMask2Float(int outputType)
{
    switch (outputType)
    {
    case NPY_BOOL:
        return ConvertBase<T, bool>::PutMaskCopyBool;
    case NPY_FLOAT:
        return ConvertBase<T, float>::PutMaskCopyFloat;
    case NPY_DOUBLE:
        return ConvertBase<T, double>::PutMaskCopyFloat;
    case NPY_LONGDOUBLE:
        return ConvertBase<T, long double>::PutMaskCopyFloat;
    case NPY_BYTE:
        return ConvertBase<T, int8_t>::PutMaskCopyFloat;
    case NPY_INT16:
        return ConvertBase<T, int16_t>::PutMaskCopyFloat;
    CASE_NPY_INT32:
        return ConvertBase<T, int32_t>::PutMaskCopyFloat;
    CASE_NPY_INT64:

        return ConvertBase<T, int64_t>::PutMaskCopyFloat;
    case NPY_UBYTE:
        return ConvertBase<T, uint8_t>::PutMaskCopyFloat;
    case NPY_UINT16:
        return ConvertBase<T, uint16_t>::PutMaskCopyFloat;
    CASE_NPY_UINT32:
        return ConvertBase<T, uint32_t>::PutMaskCopyFloat;
    CASE_NPY_UINT64:

        return ConvertBase<T, uint64_t>::PutMaskCopyFloat;
    }
    return NULL;
}

template <typename T>
static MASK_CONVERT_SAFE GetConversionPutMask2(int outputType)
{
    switch (outputType)
    {
    case NPY_BOOL:
        return ConvertBase<T, bool>::PutMaskCopyBool;
    case NPY_FLOAT:
        return ConvertBase<T, float>::PutMaskCopy;
    case NPY_DOUBLE:
        return ConvertBase<T, double>::PutMaskCopy;
    case NPY_LONGDOUBLE:
        return ConvertBase<T, long double>::PutMaskCopy;
    case NPY_BYTE:
        return ConvertBase<T, int8_t>::PutMaskCopy;
    case NPY_INT16:
        return ConvertBase<T, int16_t>::PutMaskCopy;
    CASE_NPY_INT32:
        return ConvertBase<T, int32_t>::PutMaskCopy;
    CASE_NPY_INT64:

        return ConvertBase<T, int64_t>::PutMaskCopy;
    case NPY_UBYTE:
        return ConvertBase<T, uint8_t>::PutMaskCopy;
    case NPY_UINT16:
        return ConvertBase<T, uint16_t>::PutMaskCopy;
    CASE_NPY_UINT32:
        return ConvertBase<T, uint32_t>::PutMaskCopy;
    CASE_NPY_UINT64:

        return ConvertBase<T, uint64_t>::PutMaskCopy;
    }
    return NULL;
}

static MASK_CONVERT_SAFE GetConversionPutMask(int inputType, int outputType)
{
    // check for same type -- which is shorthand for copy
    if (inputType == outputType)
    {
        switch (inputType)
        {
        case NPY_BYTE:
        case NPY_UBYTE:
        case NPY_BOOL:
            return ConvertBase<int8_t, int8_t>::PutMaskFast;

        case NPY_INT16:
        case NPY_UINT16:
            return ConvertBase<int16_t, int16_t>::PutMaskFast;

        CASE_NPY_INT32:
        CASE_NPY_UINT32:
        case NPY_FLOAT:
            return ConvertBase<int32_t, int32_t>::PutMaskFast;

        CASE_NPY_INT64:

        CASE_NPY_UINT64:

        case NPY_DOUBLE:
            return ConvertBase<int64_t, int64_t>::PutMaskFast;

        case NPY_LONGDOUBLE:
            return ConvertBase<long double, long double>::PutMaskFast;
        }
    }

    switch (inputType)
    {
        // case NPY_BOOL:   return GetConversionStep2Safe<bool>(outputType);
    case NPY_BOOL:
        return GetConversionPutMask2<int8_t>(outputType);
    case NPY_FLOAT:
        return GetConversionPutMask2Float<float>(outputType);
    case NPY_DOUBLE:
        return GetConversionPutMask2Float<double>(outputType);
    case NPY_LONGDOUBLE:
        return GetConversionPutMask2Float<long double>(outputType);
    case NPY_BYTE:
        return GetConversionPutMask2<int8_t>(outputType);
    case NPY_INT16:
        return GetConversionPutMask2<int16_t>(outputType);
    CASE_NPY_INT32:
        return GetConversionPutMask2<int32_t>(outputType);
    CASE_NPY_INT64:

        return GetConversionPutMask2<int64_t>(outputType);

        // DISCUSSION -- uint8_t and the value 255 or 0xFF will not be a sentinel
        // case NPY_UBYTE:  return GetConversionStep2Unsafe<uint8_t>(outputType);
    case NPY_UBYTE:
        return GetConversionPutMask2<uint8_t>(outputType);

    case NPY_UINT16:
        return GetConversionPutMask2<uint16_t>(outputType);
    CASE_NPY_UINT32:
        return GetConversionPutMask2<uint32_t>(outputType);
    CASE_NPY_UINT64:

        return GetConversionPutMask2<uint64_t>(outputType);
    }
    return NULL;
}

//=====================================================================================
//--------------------------------------------------------------------
struct CONVERT_CALLBACK
{
    CONVERT_SAFE anyConvertCallback;
    char * pDataIn;
    char * pDataOut;
    void * pBadInput1;
    void * pBadOutput1;

    int64_t typeSizeIn;
    int64_t typeSizeOut;

} stConvertCallback;

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static bool ConvertThreadCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    bool didSomeWork = false;
    CONVERT_CALLBACK * Callback = (CONVERT_CALLBACK *)pstWorkerItem->WorkCallbackArg;

    char * pDataIn = (char *)Callback->pDataIn;
    char * pDataOut = (char *)Callback->pDataOut;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
    {
        int64_t inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeIn;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;

        Callback->anyConvertCallback(pDataIn + inputAdj, pDataOut + outputAdj, lenX, Callback->pBadInput1, Callback->pBadOutput1,
                                     Callback->typeSizeIn, Callback->typeSizeOut);

        // Indicate we completed a block
        didSomeWork = true;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        // printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}

//=====================================================================================
//
//
void * GetInvalid(int dtype)
{
    void * pBadInput = GetDefaultForType(dtype);
    if (dtype == NPY_BOOL)
    {
        // We do not want false to become a sentinel
        pBadInput = GetDefaultForType(NPY_INT8);
    }
    return pBadInput;
}

//=====================================================================================
// Input: Two parameters
// Arg1: array to convert
// Arg2: dtype.num of the output array
//
// Returns converted array (or NULL, if an error occurs).
// The returned array is expected to have the same shape as the input array;
// when the input array has either or both the C_CONTIGUOUS or F_CONTIGUOUS
// flags values set, the returned array is expected to have the same flags set.
// NOTE: if they are the same type, special fast routine called
PyObject * ConvertSafeInternal(PyArrayObject * const inArr1, const int64_t out_dtype)
{
    const int32_t numpyOutType = (int32_t)out_dtype;
    const int32_t numpyInType = PyArray_TYPE(inArr1);

    if (numpyOutType < 0 || numpyInType > NPY_LONGDOUBLE || numpyOutType > NPY_LONGDOUBLE)
    {
        return PyErr_Format(PyExc_ValueError, "ConvertSafe: Don't know how to convert these types %d %d", numpyInType,
                            numpyOutType);
    }

    // TODO: Do we still need the check above? Or can we just rely on
    // GetConversionFunctionSafe() to do any necessary checks?
    const CONVERT_SAFE pFunction = GetConversionFunctionSafe(numpyInType, numpyOutType);
    if (! pFunction)
    {
        return PyErr_Format(PyExc_ValueError, "ConvertSafe: Don't know how to convert these types %d %d", numpyInType,
                            numpyOutType);
    }

    LOGGING("ConvertSafe converting type %d to type %d\n", numpyInType, numpyOutType);

    void * const pDataIn = PyArray_BYTES(inArr1);
    int ndim = PyArray_NDIM(inArr1);
    npy_intp * const dims = PyArray_DIMS(inArr1);

    // Make sure array lengths match
    const int64_t arraySize1 = CalcArrayLength(ndim, dims);
    const int64_t len = arraySize1;

    // Allocate the output array.
    // TODO: Consider using AllocateLikeNumpyArray here instead for simplicity.
    PyArrayObject * outArray = AllocateNumpyArray(ndim, dims, numpyOutType, 0, PyArray_IS_F_CONTIGUOUS(inArr1));
    CHECK_MEMORY_ERROR(outArray);

    // Check if we're out of memory.
    if (! outArray)
    {
        return PyErr_Format(PyExc_MemoryError, "ConvertSafe out of memory");
    }

    void * pDataOut = PyArray_BYTES(outArray);
    void * pBadInput1 = GetInvalid(numpyInType);

    // if output is boolean, bad means false
    void * pBadOutput1 = GetDefaultForType(numpyOutType);

    // Check the strides of both the input and output to make sure we can handle
    int64_t strideIn;
    int directionIn = GetStridesAndContig(inArr1, ndim, strideIn);

    int ndimOut;
    int64_t strideOut;
    int directionOut = GetStridesAndContig(outArray, ndimOut, strideOut);

    // If the input is C and/or F-contiguous, the output should have
    // the same flag(s) set.
    if (directionIn != 0 || directionOut != 0)
    {
        // non-contiguous loop
        // Walk the input, dimension by dimension, getting the stride
        // Check if we can process, else punt to numpy
        if (directionIn == 1 && directionOut == 0)
        {
            // Row Major 2dim like array with output being fully contiguous
            int64_t innerLen = 1;
            for (int i = directionIn; i < ndim; i++)
            {
                innerLen *= PyArray_DIM(inArr1, i);
            }
            // TODO: consider dividing the work over multiple threads if innerLen is
            // large enough
            const int64_t outerLen = PyArray_DIM(inArr1, 0);
            const int64_t outerStride = PyArray_STRIDE(inArr1, 0);

            LOGGING("Row Major  innerLen:%lld  outerLen:%lld  outerStride:%lld\n", innerLen, outerLen, outerStride);

            for (int64_t j = 0; j < outerLen; j++)
            {
                pFunction((char *)pDataIn + (j * outerStride), (char *)pDataOut + (j * innerLen * strideOut), innerLen, pBadInput1,
                          pBadOutput1, strideIn, strideOut);
            }
        }
        else if (directionIn == -1 && directionOut == 0)
        {
            // Col Major 2dim like array with output being fully contiguous
            int64_t innerLen = 1;
            directionIn = -directionIn;
            for (int i = 0; i < directionIn; i++)
            {
                innerLen *= PyArray_DIM(inArr1, i);
            }
            // TODO: consider dividing the work over multiple threads if innerLen is
            // large enough
            const int64_t outerLen = PyArray_DIM(inArr1, (ndim - 1));
            const int64_t outerStride = PyArray_STRIDE(inArr1, (ndim - 1));

            LOGGING("Col Major  innerLen:%lld  outerLen:%lld  outerStride:%lld\n", innerLen, outerLen, outerStride);

            for (int64_t j = 0; j < outerLen; j++)
            {
                pFunction((char *)pDataIn + (j * outerStride), (char *)pDataOut + (j * innerLen * strideOut), innerLen, pBadInput1,
                          pBadOutput1, strideIn, strideOut);
            }
        }
        else
        {
            // Don't leak the memory we allocated -- free it before raising the Python
            // error and returning.
            RecycleNumpyInternal(outArray);
            // have numpy do the work
            outArray = (PyArrayObject *)PyArray_FROM_OT((PyObject *)inArr1, numpyOutType);
            // return PyErr_Format(PyExc_RuntimeError, "ConvertSafe allocated an
            // output array whose *_CONTIGUOUS flags were not set even though the input
            // array was contiguous. %d %d   out:%d %d  strideIn:%lld  strideOut:%lld",
            // contigIn, ndim, contigOut, ndimOut, strideIn, strideOut);
        }
    }
    else
    {
        stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(len);

        if (! pWorkItem)
        {
            // Threading not allowed for this work item, call it directly from main
            // thread
            pFunction(pDataIn, pDataOut, len, pBadInput1, pBadOutput1, strideIn, strideOut);
        }
        else
        {
            // Each thread will call this routine with the callbackArg
            pWorkItem->DoWorkCallback = ConvertThreadCallback;
            pWorkItem->WorkCallbackArg = &stConvertCallback;

            stConvertCallback.anyConvertCallback = pFunction;
            stConvertCallback.pDataOut = (char *)pDataOut;
            stConvertCallback.pDataIn = (char *)pDataIn;
            stConvertCallback.pBadInput1 = pBadInput1;
            stConvertCallback.pBadOutput1 = pBadOutput1;

            stConvertCallback.typeSizeIn = strideIn;
            stConvertCallback.typeSizeOut = strideOut;

            // This will notify the worker threads of a new work item
            // TODO: Calc how many threads we need to do the conversion (possibly just
            // 3 worker threads is enough)
            g_cMathWorker->WorkMain(pWorkItem, len, 0);
        }
    }
    return (PyObject *)outArray;
}

//=====================================================================================
PyObject * ConvertSafe(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    int64_t out_dtype = 0;

    if (Py_SIZE(args) > 1)
    {
        PyArrayObject * inObject = (PyArrayObject *)PyTuple_GET_ITEM(args, 0);
        PyObject * inNumber = PyTuple_GET_ITEM(args, 1);
        if (PyLong_CheckExact(inNumber))
        {
            int64_t dtypeNum = PyLong_AsLongLong(inNumber);

            if (IsFastArrayOrNumpy(inObject))
            {
                PyObject * result = ConvertSafeInternal(inObject, dtypeNum);
                return result;
            }
            else
            {
                PyErr_Format(PyExc_ValueError, "ConvertSafe first argument must be an array not type %s",
                             ((PyObject *)inObject)->ob_type->tp_name);
            }
        }
        else
        {
            PyErr_Format(PyExc_ValueError, "ConvertSafe second argument must be an integer not type %s",
                         inNumber->ob_type->tp_name);
        }
    }
    else
    {
        PyErr_Format(PyExc_ValueError, "ConvertSafe must have at least two arguments");
    }

    return NULL;
}

//=====================================================================================
// Input: Two parameters
// Arg1: array to convert
// Arg2: dtype.num of the output array
//
// Returns converted array or nullptr on error.
// NOTE: if they are the same type, special fast routine called
// TODO: Combine ConvertSafeInternal and ConvertUnsafeInternal into a single,
// templated function --
//       they only differ in whether they use GetConversionFunctionSafe /
//       GetConversionFunctionUnsafe.
PyObject * ConvertUnsafeInternal(PyArrayObject * inArr1, int64_t out_dtype)
{
    const int32_t numpyOutType = (int32_t)out_dtype;
    const int32_t numpyInType = ObjectToDtype(inArr1);

    if (numpyOutType < 0 || numpyInType < 0 || numpyInType > NPY_LONGDOUBLE || numpyOutType > NPY_LONGDOUBLE)
    {
        return PyErr_Format(PyExc_ValueError, "ConvertUnsafe: Don't know how to convert these types %d %d", numpyInType,
                            numpyOutType);
    }

    // TODO: Do we still need the check above? Or can we just rely on
    // GetConversionFunctionUnsafe() to do any necessary checks?
    CONVERT_SAFE pFunction = GetConversionFunctionUnsafe(numpyInType, numpyOutType);
    if (! pFunction)
    {
        return PyErr_Format(PyExc_ValueError, "ConvertUnsafe: Don't know how to convert these types %d %d", numpyInType,
                            numpyOutType);
    }

    LOGGING("ConvertUnsafe converting type %d to type %d\n", numpyInType, numpyOutType);

    void * pDataIn = PyArray_BYTES(inArr1);

    const int ndim = PyArray_NDIM(inArr1);
    npy_intp * const dims = PyArray_DIMS(inArr1);
    // auto* const inArr1_strides = PyArray_STRIDES(inArr1);
    // TODO: CalcArrayLength probably needs to be fixed to account for any
    // non-default striding
    const int64_t arraySize1 = CalcArrayLength(ndim, dims);
    const int64_t len = arraySize1;

    // Allocate the output array.
    // TODO: Consider using AllocateLikeNumpyArray here instead for simplicity.
    PyArrayObject * outArray = AllocateNumpyArray(ndim, dims, numpyOutType, 0, PyArray_IS_F_CONTIGUOUS(inArr1));
    CHECK_MEMORY_ERROR(outArray);
    if (! outArray)
    {
        return PyErr_Format(PyExc_MemoryError, "ConvertUnsafe out of memory");
    }

    void * pDataOut = PyArray_BYTES(outArray);

    void * pBadInput1 = GetInvalid(numpyInType);

    // if output is boolean, bad means false
    void * pBadOutput1 = GetDefaultForType(numpyOutType);

    stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(len);

    if (! pWorkItem)
    {
        // Threading not allowed for this work item, call it directly from main
        // thread
        pFunction(pDataIn, pDataOut, len, pBadInput1, pBadOutput1, PyArray_STRIDE(inArr1, 0), PyArray_STRIDE(outArray, 0));
    }
    else
    {
        // Each thread will call this routine with the callbackArg
        pWorkItem->DoWorkCallback = ConvertThreadCallback;
        pWorkItem->WorkCallbackArg = &stConvertCallback;

        stConvertCallback.anyConvertCallback = pFunction;
        stConvertCallback.pDataOut = (char *)pDataOut;
        stConvertCallback.pDataIn = (char *)pDataIn;
        stConvertCallback.pBadInput1 = pBadInput1;
        stConvertCallback.pBadOutput1 = pBadOutput1;

        stConvertCallback.typeSizeIn = PyArray_STRIDE(inArr1, 0);
        stConvertCallback.typeSizeOut = PyArray_STRIDE(outArray, 0);

        // This will notify the worker threads of a new work item
        g_cMathWorker->WorkMain(pWorkItem, len, 0);
    }

    return (PyObject *)outArray;
}

//=====================================================================================
// Input: Two parameters
// Arg1: array to convert
// Arg2: dtype.num of the output array
//
// Returns converted array
// NOTE: if they are the same type, special fast routine called
PyObject * ConvertUnsafe(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    int64_t out_dtype = 0;

    if (! PyArg_ParseTuple(args, "O!L:ConvertUnsafe", &PyArray_Type, &inArr1, &out_dtype))
    {
        return NULL;
    }
    return ConvertUnsafeInternal(inArr1, out_dtype);
}

//=====================================================================================
// COMBINE
// MASK------------------------------------------------------------------------
//=====================================================================================
typedef void (*COMBINE_MASK)(void * pDataIn, void * pDataOut, int64_t len, int8_t * pFilter);

template <typename T>
static void CombineMask(void * pDataInT, void * pDataOutT, int64_t len, int8_t * pFilter)
{
    T * pDataIn = (T *)pDataInT;
    T * pDataOut = (T *)pDataOutT;

    for (int64_t i = 0; i < len; i++)
    {
        pDataOut[i] = pDataIn[i] * (T)pFilter[i];
    }
}

static COMBINE_MASK GetCombineFunction(int outputType)
{
    switch (outputType)
    {
    case NPY_INT8:
        return CombineMask<int8_t>;
    case NPY_INT16:
        return CombineMask<int16_t>;
    CASE_NPY_INT32:
        return CombineMask<int32_t>;
    CASE_NPY_INT64:

        return CombineMask<int64_t>;
    }
    return NULL;
}

//--------------------------------------------------------------------
struct COMBINE_CALLBACK
{
    COMBINE_MASK anyCombineCallback;
    char * pDataIn;
    char * pDataOut;
    int8_t * pFilter;

    int64_t typeSizeOut;

} stCombineCallback;

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static bool CombineThreadCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    bool didSomeWork = false;
    COMBINE_CALLBACK * Callback = (COMBINE_CALLBACK *)pstWorkerItem->WorkCallbackArg;

    char * pDataIn = (char *)Callback->pDataIn;
    char * pDataOut = (char *)Callback->pDataOut;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
    {
        int64_t inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;
        int64_t filterAdj = pstWorkerItem->BlockSize * workBlock;

        Callback->anyCombineCallback(pDataIn + inputAdj, pDataOut + inputAdj, lenX, Callback->pFilter + filterAdj);

        // Indicate we completed a block
        didSomeWork = true;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        // printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}

//=====================================================================================
// Input: Two parameters
// Arg1: Index array
// Arg2: Boolean array to merge
//
// Returns new index array with 0
PyObject * CombineFilter(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    PyArrayObject * inFilter = NULL;

    int64_t out_dtype = 0;

    if (! PyArg_ParseTuple(args, "O!O!:CombineFilter", &PyArray_Type, &inArr1, &PyArray_Type, &inFilter))
    {
        return NULL;
    }

    int32_t numpyOutType = PyArray_TYPE(inArr1);
    void * pDataIn = PyArray_BYTES(inArr1);
    int ndim = PyArray_NDIM(inArr1);
    npy_intp * dims = PyArray_DIMS(inArr1);
    int64_t arraySize1 = CalcArrayLength(ndim, dims);
    int64_t len = arraySize1;

    if (arraySize1 != ArrayLength(inFilter))
    {
        PyErr_Format(PyExc_ValueError, "CombineFilter: Filter size not the same %lld", arraySize1);
        return NULL;
    }

    if (PyArray_TYPE(inFilter) != NPY_BOOL)
    {
        PyErr_Format(PyExc_ValueError, "CombineFilter: Filter is not type NPY_BOOL");
        return NULL;
    }

    // SWTICH
    COMBINE_MASK pFunction = NULL;

    switch (numpyOutType)
    {
    case NPY_INT8:
        pFunction = GetCombineFunction(numpyOutType);
        break;

    case NPY_INT16:
        pFunction = GetCombineFunction(numpyOutType);
        break;

    CASE_NPY_INT32:
        pFunction = GetCombineFunction(numpyOutType);
        break;

    CASE_NPY_INT64:

        pFunction = GetCombineFunction(numpyOutType);
        break;
    }

    if (pFunction != NULL)
    {
        PyArrayObject * outArray = AllocateNumpyArray(ndim, dims, numpyOutType);
        CHECK_MEMORY_ERROR(outArray);

        if (outArray)
        {
            void * pDataOut = PyArray_BYTES(outArray);
            int8_t * pFilter = (int8_t *)PyArray_BYTES(inFilter);

            stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(len);

            if (pWorkItem == NULL)
            {
                // Threading not allowed for this work item, call it directly from main
                // thread
                pFunction(pDataIn, pDataOut, len, pFilter);
            }
            else
            {
                // Each thread will call this routine with the callbackArg
                pWorkItem->DoWorkCallback = CombineThreadCallback;
                pWorkItem->WorkCallbackArg = &stCombineCallback;

                stCombineCallback.anyCombineCallback = pFunction;
                stCombineCallback.pDataOut = (char *)pDataOut;
                stCombineCallback.pDataIn = (char *)pDataIn;
                stCombineCallback.pFilter = pFilter;
                stCombineCallback.typeSizeOut = PyArray_ITEMSIZE(inArr1);

                // This will notify the worker threads of a new work item
                g_cMathWorker->WorkMain(pWorkItem, len, 0);
            }

            return (PyObject *)outArray;
        }
        PyErr_Format(PyExc_ValueError, "Combine out of memory");
        return NULL;
    }

    PyErr_Format(PyExc_ValueError, "Dont know how to combine these types %d", numpyOutType);
    return NULL;
}

//=====================================================================================
// COMBINE ACCUM 2
// MASK----------------------------------------------------------------
//=====================================================================================
typedef void (*COMBINE_ACCUM2_MASK)(void * pDataIn1, void * pDataIn2, void * pDataOut, const int64_t multiplier,
                                    const int64_t maxbin, int64_t len, void * pCountOut2, int8_t * pFilter);

template <typename T, typename U, typename V>
static void CombineAccum2Mask(void * pDataIn1T, void * pDataIn2T, void * pDataOutT, const int64_t multiplier,
                              const int64_t maxbinT, int64_t len, void * pCountOut2, int8_t * pFilter)
{
    T * pDataIn1 = (T *)pDataIn1T;
    U * pDataIn2 = (U *)pDataIn2T;
    V * pDataOut = (V *)pDataOutT;

    const V maxbin = (V)maxbinT;

    if (pCountOut2)
    {
        // TODO: handle int64_t also
        int32_t * pCountOut = (int32_t *)pCountOut2;

        if (pFilter)
        {
            for (int64_t i = 0; i < len; i++)
            {
                if (pFilter[i])
                {
                    V bin = (V)(pDataIn2[i] * multiplier + pDataIn1[i]);
                    if (bin >= 0 && bin < maxbin)
                    {
                        pCountOut[bin]++;
                        pDataOut[i] = bin;
                    }
                    else
                    {
                        pCountOut[0]++;
                        pDataOut[i] = 0;
                    }
                }
                else
                {
                    pCountOut[0]++;
                    pDataOut[i] = 0;
                }
            }
        }
        else
        {
            for (int64_t i = 0; i < len; i++)
            {
                V bin = (V)(pDataIn2[i] * multiplier + pDataIn1[i]);
                if (bin >= 0 && bin < maxbin)
                {
                    pCountOut[bin]++;
                    pDataOut[i] = bin;
                }
                else
                {
                    pCountOut[0]++;
                    pDataOut[i] = 0;
                }
            }
        }
    }
    else
    {
        // NO COUNT
        if (pFilter)
        {
            for (int64_t i = 0; i < len; i++)
            {
                if (pFilter[i])
                {
                    V bin = (V)(pDataIn2[i] * multiplier + pDataIn1[i]);
                    if (bin >= 0 && bin < maxbin)
                    {
                        pDataOut[i] = bin;
                    }
                    else
                    {
                        pDataOut[i] = 0;
                    }
                }
                else
                {
                    pDataOut[i] = 0;
                }
            }
        }
        else
        {
            for (int64_t i = 0; i < len; i++)
            {
                V bin = (V)(pDataIn2[i] * multiplier + pDataIn1[i]);
                pDataOut[i] = bin;
                // if (bin >= 0 && bin < maxbin) {
                //   pDataOut[i] = bin;
                //}
                // else {
                //   pDataOut[i] = 0;
                //}
            }
        }
    }
}

template <typename T, typename U>
static COMBINE_ACCUM2_MASK GetCombineAccum2Function(int outputType)
{
    // printf("GetCombine -- %lld %lld\n", sizeof(T), sizeof(U));

    switch (outputType)
    {
    case NPY_INT8:
        return CombineAccum2Mask<T, U, int8_t>;
    case NPY_INT16:
        return CombineAccum2Mask<T, U, int16_t>;
    CASE_NPY_INT32:
        return CombineAccum2Mask<T, U, int32_t>;
    CASE_NPY_INT64:

        return CombineAccum2Mask<T, U, int64_t>;
    }
    return NULL;
}

//--------------------------------------------------------------------
struct COMBINE_ACCUM2_CALLBACK
{
    COMBINE_ACCUM2_MASK anyCombineCallback;
    char * pDataIn1;
    char * pDataIn2;
    char * pDataOut;
    int8_t * pFilter;

    void * pCountOut; // int32 or int64
    int64_t typeSizeIn1;
    int64_t typeSizeIn2;
    int64_t typeSizeOut;
    int64_t multiplier;
    int64_t maxbin;
    void * pCountWorkSpace; // int32 or int64

} stCombineAccum2Callback;

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static bool CombineThreadAccum2Callback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    bool didSomeWork = false;
    COMBINE_ACCUM2_CALLBACK * Callback = (COMBINE_ACCUM2_CALLBACK *)pstWorkerItem->WorkCallbackArg;

    char * pDataIn1 = (char *)Callback->pDataIn1;
    char * pDataIn2 = (char *)Callback->pDataIn2;
    char * pDataOut = (char *)Callback->pDataOut;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
    {
        int64_t inputAdj1 = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeIn1;
        int64_t inputAdj2 = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeIn2;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;
        int64_t filterAdj = pstWorkerItem->BlockSize * workBlock;

        // pFunction(pDataIn1, pDataIn2, pDataOut, inArr1Max, hashSize, arraySize1,
        // pCountArray, pFilterIn);

        Callback->anyCombineCallback(
            pDataIn1 + inputAdj1, pDataIn2 + inputAdj2, pDataOut + outputAdj, Callback->multiplier, Callback->maxbin, lenX,
            // based on core, pick a counter
            // TODO: 64bit code also
            Callback->pCountWorkSpace ? &((int32_t *)(Callback->pCountWorkSpace))[(core + 1) * Callback->maxbin] : NULL,
            Callback->pFilter ? (Callback->pFilter + filterAdj) : NULL);

        // Indicate we completed a block
        didSomeWork = true;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        // printf("|%d %d %lld", core, (int)workBlock, lenX);
    }

    return didSomeWork;
}

//=====================================================================================
// Input: Five parameters
// Arg1: First Index array
// Arg2: Second Index array
// Arg3: Max value first index array
// Arg4: Max value second index array
// Arg5: Boolean array to merge <optional: can set to none>
//
// Returns new index array and unique count array
PyObject * CombineAccum2Filter(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    PyArrayObject * inArr2 = NULL;
    int64_t inArr1Max = 0;
    int64_t inArr2Max = 0;
    PyObject * inFilter = NULL;

    int64_t out_dtype = 0;

    if (! PyArg_ParseTuple(args, "O!O!LLO:CombineAccum2Filter", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &inArr1Max,
                           &inArr2Max, &inFilter))
    {
        return NULL;
    }

    inArr1Max++;
    inArr2Max++;

    int64_t hashSize = inArr1Max * inArr2Max;
    // printf("Combine hashsize is %lld     %lld x %lld\n", hashSize, inArr1Max,
    // inArr2Max);

    void * pDataIn1 = PyArray_BYTES(inArr1);
    void * pDataIn2 = PyArray_BYTES(inArr2);
    int8_t * pFilterIn = NULL;
    int ndim = PyArray_NDIM(inArr1);
    npy_intp * dims = PyArray_DIMS(inArr1);
    int64_t arraySize1 = CalcArrayLength(ndim, dims);
    int64_t arraySize2 = ArrayLength(inArr2);

    if (arraySize1 != arraySize2)
    {
        PyErr_Format(PyExc_ValueError, "CombineAccum2Filter: array sizes not the same %lld", arraySize1);
        return NULL;
    }

    if (PyArray_Check(inFilter))
    {
        if (arraySize1 != ArrayLength((PyArrayObject *)inFilter))
        {
            PyErr_Format(PyExc_ValueError, "CombineAccum2Filter: Filter size not the same %lld", arraySize1);
            return NULL;
        }
        if (PyArray_TYPE((PyArrayObject *)inFilter) != NPY_BOOL)
        {
            PyErr_Format(PyExc_ValueError, "CombineAccum2Filter: Filter is not type NPY_BOOL");
            return NULL;
        }
        pFilterIn = (int8_t *)PyArray_BYTES((PyArrayObject *)inFilter);
    }
    else if (inFilter != Py_None)
    {
        PyErr_Format(PyExc_ValueError, "CombineAccum2Filter: unsupported filter type, %s", Py_TYPE(inFilter)->tp_name);
        return NULL;
    }

    if (hashSize < 0)
    {
        PyErr_Format(PyExc_ValueError, "CombineAccum2Filter: Index sizes are negative %lld", hashSize);
        return NULL;
    }

    int32_t numpyOutType = NPY_INT64;
    int64_t typeSizeOut = 8;

    if (hashSize < 2000000000)
    {
        numpyOutType = NPY_INT32;
        typeSizeOut = 4;
    }
    if (hashSize < 32000)
    {
        numpyOutType = NPY_INT16;
        typeSizeOut = 2;
    }
    if (hashSize < 120)
    {
        numpyOutType = NPY_INT8;
        typeSizeOut = 1;
    }

    // SWTICH
    COMBINE_ACCUM2_MASK pFunction = NULL;

    int type2 = PyArray_TYPE(inArr2);

    switch (PyArray_TYPE(inArr1))
    {
    case NPY_INT8:
        switch (type2)
        {
        case NPY_INT8:
            pFunction = GetCombineAccum2Function<int8_t, int8_t>(numpyOutType);
            break;
        case NPY_INT16:
            pFunction = GetCombineAccum2Function<int8_t, int16_t>(numpyOutType);
            break;
        CASE_NPY_INT32:
            pFunction = GetCombineAccum2Function<int8_t, int32_t>(numpyOutType);
            break;
        CASE_NPY_INT64:

            pFunction = GetCombineAccum2Function<int8_t, int64_t>(numpyOutType);
            break;
        }
        break;

    case NPY_INT16:
        switch (type2)
        {
        case NPY_INT8:
            pFunction = GetCombineAccum2Function<int16_t, int8_t>(numpyOutType);
            break;
        case NPY_INT16:
            pFunction = GetCombineAccum2Function<int16_t, int16_t>(numpyOutType);
            break;
        CASE_NPY_INT32:
            pFunction = GetCombineAccum2Function<int16_t, int32_t>(numpyOutType);
            break;
        CASE_NPY_INT64:

            pFunction = GetCombineAccum2Function<int16_t, int64_t>(numpyOutType);
            break;
        }
        break;

    CASE_NPY_INT32:
        switch (type2)
        {
        case NPY_INT8:
            pFunction = GetCombineAccum2Function<int32_t, int8_t>(numpyOutType);
            break;
        case NPY_INT16:
            pFunction = GetCombineAccum2Function<int32_t, int16_t>(numpyOutType);
            break;
        CASE_NPY_INT32:
            pFunction = GetCombineAccum2Function<int32_t, int32_t>(numpyOutType);
            break;
        CASE_NPY_INT64:

            pFunction = GetCombineAccum2Function<int32_t, int64_t>(numpyOutType);
            break;
        }
        break;

    CASE_NPY_INT64:

        switch (type2)
        {
        case NPY_INT8:
            pFunction = GetCombineAccum2Function<int64_t, int8_t>(numpyOutType);
            break;
        case NPY_INT16:
            pFunction = GetCombineAccum2Function<int64_t, int16_t>(numpyOutType);
            break;
        CASE_NPY_INT32:
            pFunction = GetCombineAccum2Function<int64_t, int32_t>(numpyOutType);
            break;
        CASE_NPY_INT64:

            pFunction = GetCombineAccum2Function<int64_t, int64_t>(numpyOutType);
            break;
        }
        break;
    }

    bool bWantCount = false;

    if (pFunction != NULL)
    {
        PyArrayObject * outArray = AllocateNumpyArray(ndim, dims, numpyOutType, 0, PyArray_IS_F_CONTIGUOUS(inArr1));
        CHECK_MEMORY_ERROR(outArray);

        if (outArray)
        {
            void * pDataOut = PyArray_BYTES(outArray);
            bool is64bithash = false;
            int64_t sizeofhash = 4;

            // 32 bit count limitation here
            PyArrayObject * countArray = NULL;

            if (hashSize > 2147480000)
            {
                is64bithash = true;
                sizeofhash = 8;
            }

            void * pCountArray = NULL;

            if (bWantCount)
            {
                countArray = AllocateNumpyArray(1, (npy_intp *)&hashSize, is64bithash ? NPY_INT64 : NPY_INT32);
                CHECK_MEMORY_ERROR(countArray);
                if (countArray)
                {
                    pCountArray = (int64_t *)PyArray_BYTES(countArray);
                    memset(pCountArray, 0, hashSize * sizeofhash);
                }
            }

            stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(arraySize1);

            if (pWorkItem == NULL)
            {
                // Threading not allowed for this work item, call it directly from main
                // thread
                pFunction(pDataIn1, pDataIn2, pDataOut, inArr1Max, hashSize, arraySize1, pCountArray, pFilterIn);
            }
            else
            {
                // TODO: steal from hash
                int32_t numCores = g_cMathWorker->WorkerThreadCount + 1;
                int64_t sizeToAlloc = numCores * hashSize * sizeofhash;
                void * pWorkSpace = 0;

                if (bWantCount)
                {
                    pWorkSpace = WORKSPACE_ALLOC(sizeToAlloc);
                    memset(pWorkSpace, 0, sizeToAlloc);
                }

                // Each thread will call this routine with the callbackArg
                pWorkItem->DoWorkCallback = CombineThreadAccum2Callback;
                pWorkItem->WorkCallbackArg = &stCombineAccum2Callback;

                stCombineAccum2Callback.anyCombineCallback = pFunction;
                stCombineAccum2Callback.pDataOut = (char *)pDataOut;
                stCombineAccum2Callback.pDataIn1 = (char *)pDataIn1;
                stCombineAccum2Callback.pDataIn2 = (char *)pDataIn2;
                stCombineAccum2Callback.pFilter = pFilterIn;
                stCombineAccum2Callback.pCountOut = pCountArray;

                stCombineAccum2Callback.typeSizeIn1 = PyArray_ITEMSIZE(inArr1);
                stCombineAccum2Callback.typeSizeIn2 = PyArray_ITEMSIZE(inArr2);
                stCombineAccum2Callback.typeSizeOut = typeSizeOut;
                stCombineAccum2Callback.multiplier = inArr1Max;
                stCombineAccum2Callback.maxbin = hashSize;
                stCombineAccum2Callback.pCountWorkSpace = pWorkSpace;

                LOGGING("**array: %lld      out sizes: %lld %lld %lld\n", arraySize1, stCombineAccum2Callback.typeSizeIn1,
                        stCombineAccum2Callback.typeSizeIn2, stCombineAccum2Callback.typeSizeOut);

                // This will notify the worker threads of a new work item
                g_cMathWorker->WorkMain(pWorkItem, arraySize1, 0);

                if (bWantCount && pCountArray)
                {
                    if (is64bithash)
                    {
                        // Collect the results
                        int64_t * pCoreCountArray = (int64_t *)pWorkSpace;
                        int64_t * pCountArray2 = (int64_t *)pCountArray;

                        for (int j = 0; j < numCores; j++)
                        {
                            for (int i = 0; i < hashSize; i++)
                            {
                                pCountArray2[i] += pCoreCountArray[i];
                            }

                            // go to next core
                            pCoreCountArray += hashSize;
                        }
                    }
                    else
                    {
                        // Collect the results
                        int32_t * pCoreCountArray = (int32_t *)pWorkSpace;
                        int32_t * pCountArray2 = (int32_t *)pCountArray;

                        for (int j = 0; j < numCores; j++)
                        {
                            for (int i = 0; i < hashSize; i++)
                            {
                                pCountArray2[i] += pCoreCountArray[i];
                            }

                            // go to next core
                            pCoreCountArray += hashSize;
                        }
                    }
                }

                if (bWantCount)
                {
                    WORKSPACE_FREE(pWorkSpace);
                }
            }

            if (bWantCount)
            {
                PyObject * retObject = Py_BuildValue("(OO)", outArray, countArray);
                Py_DecRef((PyObject *)outArray);
                Py_DecRef((PyObject *)countArray);
                return (PyObject *)retObject;
            }
            else
            {
                Py_INCREF(Py_None);
                PyObject * retObject = Py_BuildValue("(OO)", outArray, Py_None);
                Py_DecRef((PyObject *)outArray);
                return (PyObject *)retObject;
            }
        }
        PyErr_Format(PyExc_ValueError, "CombineFilter out of memory");
        return NULL;
    }

    PyErr_Format(PyExc_ValueError,
                 "Dont know how to combine filter these types %d.  Please make "
                 "sure all bins are int8_t, int16_t, int32_t, or int64_t.",
                 numpyOutType);
    return NULL;
}

//==========================================
// Old  Filter First  ==> NewIndex NewFirst
// 1      T     0           1        0
// 1      F     2           0        4
// 2      F     3           0
// 3      F                 0
// 3      T                 2
// 3      T                 2
//
// Input:  InputIndex
//         Filter
// Output: OutputIndex (the new iKey)
//         NewFirst    (the new iFirstKey)
//
//
typedef int64_t (*COMBINE_1_FILTER)(void * pInputIndex,
                                    void * pOutputIndex, // newly allocated
                                    int32_t * pNewFirst, // newly allocated
                                    int8_t * pFilter,    // may be null
                                    int64_t arrayLength, // index array size
                                    int64_t hashLength); // max uniques + 1 (for 0 bin)

template <typename INDEX>
int64_t Combine1Filter(void * pInputIndex,
                       void * pOutputIndex, // newly allocated
                       int32_t * pNewFirst, // newly allocated NOTE: for > 2e9 should be int64_t
                       int8_t * pFilter,    // may be null
                       int64_t arrayLength, int64_t hashLength)
{
    INDEX * pInput = (INDEX *)pInputIndex;
    INDEX * pOutput = (INDEX *)pOutputIndex;

    // WORKSPACE_ALLOC
    int64_t allocSize = hashLength * sizeof(int32_t);

    int32_t * pHash = (int32_t *)WorkSpaceAllocLarge(allocSize);
    memset(pHash, 0, allocSize);

    int32_t uniquecount = 0;
    if (pFilter)
    {
        for (int64_t i = 0; i < arrayLength; i++)
        {
            if (pFilter[i])
            {
                INDEX index = pInput[i];
                // printf("[%lld] got index for %lld\n", (int64_t)index, i);

                if (index != 0)
                {
                    // Check hash
                    if (pHash[index] == 0)
                    {
                        // First time, assign FirstKey
                        pNewFirst[uniquecount] = (int32_t)i;
                        uniquecount++;

                        // printf("reassign index:%lld to bin:%d\n", (int64_t)index,
                        // uniquecount);

                        // ReassignKey
                        pHash[index] = uniquecount;
                        pOutput[i] = (INDEX)uniquecount;
                    }
                    else
                    {
                        // Get reassigned key
                        // printf("exiting  index:%lld to bin:%d\n", (int64_t)index,
                        // (int32_t)pHash[index]);
                        pOutput[i] = (INDEX)pHash[index];
                    }
                }
                else
                {
                    // was already 0 bin
                    pOutput[i] = 0;
                }
            }
            else
            {
                // filtered out
                pOutput[i] = 0;
            }
        }
    }
    else
    {
        // When no filter provided
        for (int64_t i = 0; i < arrayLength; i++)
        {
            INDEX index = pInput[i];
            // printf("[%lld] got index\n", (int64_t)index);

            if (index != 0)
            {
                // Check hash
                if (pHash[index] == 0)
                {
                    // First time, assign FirstKey
                    pNewFirst[uniquecount] = (int32_t)i;
                    uniquecount++;

                    // ReassignKey
                    pHash[index] = uniquecount;
                    pOutput[i] = (INDEX)uniquecount;
                }
                else
                {
                    // Get reassigned key
                    pOutput[i] = (INDEX)pHash[index];
                }
            }
            else
            {
                // was already 0 bin
                pOutput[i] = 0;
            }
        }
    }

    void * pHashVoid = pHash;
    WorkSpaceFreeAllocLarge(pHashVoid, allocSize);
    return uniquecount;
}

//=====================================================================================
// Input:
// Arg1: Index array
// Arg2: Max uniques
// Arg3: Boolean array to filter on (or None)
//
// Output:
// New Index Array
// New First Array (can use to pull in key names)
// UniqueCount (should be size of FirstArray)... possibly 0 if everything
// removed Returns new index array and unique count array
PyObject * CombineAccum1Filter(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    int64_t inArr1Max = 0;
    PyObject * inFilter = NULL;

    int64_t out_dtype = 0;

    if (! PyArg_ParseTuple(args, "O!LO:CombineAccum1Filter", &PyArray_Type, &inArr1, &inArr1Max, &inFilter))
    {
        return NULL;
    }

    void * pDataIn1 = PyArray_BYTES(inArr1);
    int8_t * pFilterIn = NULL;

    int ndim = PyArray_NDIM(inArr1);
    npy_intp * dims = PyArray_DIMS(inArr1);
    int64_t arraySize1 = CalcArrayLength(ndim, dims);

    if (PyArray_Check(inFilter))
    {
        if (arraySize1 != ArrayLength((PyArrayObject *)inFilter))
        {
            PyErr_Format(PyExc_ValueError, "CombineAccum1Filter: Filter size not the same %lld", arraySize1);
            return NULL;
        }
        if (PyArray_TYPE((PyArrayObject *)inFilter) != NPY_BOOL)
        {
            PyErr_Format(PyExc_ValueError, "CombineAccum1Filter: Filter is not type NPY_BOOL");
            return NULL;
        }
        pFilterIn = (int8_t *)PyArray_BYTES((PyArrayObject *)inFilter);
    }

    inArr1Max++;
    int64_t hashSize = inArr1Max;
    // printf("Combine hashsize is %lld     %lld\n", hashSize, inArr1Max);
    if (hashSize < 0 || hashSize > 2000000000)
    {
        PyErr_Format(PyExc_ValueError,
                     "CombineAccum1Filter: Index sizes are either 0, negative, or "
                     "produce more than 2 billion results %lld",
                     hashSize);
        return NULL;
    }

    int dtype = PyArray_TYPE(inArr1);

    COMBINE_1_FILTER pFunction = NULL;
    switch (dtype)
    {
    case NPY_INT8:
        pFunction = Combine1Filter<int8_t>;
        break;
    case NPY_INT16:
        pFunction = Combine1Filter<int16_t>;
        break;
    CASE_NPY_INT32:
        pFunction = Combine1Filter<int32_t>;
        break;
    CASE_NPY_INT64:

        pFunction = Combine1Filter<int64_t>;
        break;
    }

    if (pFunction != NULL)
    {
        PyArrayObject * outArray = AllocateNumpyArray(ndim, dims, dtype, 0, PyArray_IS_F_CONTIGUOUS(inArr1));
        CHECK_MEMORY_ERROR(outArray);

        PyArrayObject * firstArray = AllocateNumpyArray(1, (npy_intp *)&arraySize1,
                                                        NPY_INT32); // TODO: bump up to int64_t for large arrays
        CHECK_MEMORY_ERROR(firstArray);

        if (outArray && firstArray)
        {
            int32_t * pFirst = (int32_t *)PyArray_BYTES(firstArray);

            int64_t uniqueCount = pFunction(pDataIn1, PyArray_BYTES(outArray), pFirst, pFilterIn, arraySize1, hashSize);

            if (uniqueCount < arraySize1)
            {
                // fixup first to hold only the uniques
                PyArrayObject * firstArrayReduced = AllocateNumpyArray(1, (npy_intp *)&uniqueCount,
                                                                       NPY_INT32); // TODO: bump up to int64_t for large arrays
                CHECK_MEMORY_ERROR(firstArrayReduced);

                if (firstArrayReduced)
                {
                    int32_t * pFirstReduced = (int32_t *)PyArray_BYTES(firstArrayReduced);

                    memcpy(pFirstReduced, pFirst, uniqueCount * sizeof(int32_t));
                }
                Py_DecRef((PyObject *)firstArray);
                firstArray = firstArrayReduced;
            }

            PyObject * returnObject = PyList_New(3);
            PyList_SET_ITEM(returnObject, 0, (PyObject *)outArray);
            PyList_SET_ITEM(returnObject, 1, (PyObject *)firstArray);
            PyList_SET_ITEM(returnObject, 2, (PyObject *)PyLong_FromLongLong(uniqueCount));
            return returnObject;
        }
    }

    return NULL;
}

typedef int64_t (*IFIRST_FILTER)(void * pInputIndex,
                                 void * pNewFirstIndex, // newly allocated
                                 int8_t * pFilter,      // may be null
                                 int64_t arrayLength,   // index array size
                                 int64_t hashLength);   // max uniques + 1 (for 0 bin)

template <typename INDEX>
int64_t iFirstFilter(void * pInputIndex,
                     void * pNewFirstIndex, // newly allocated NOTE: for > 2e9 should be int64_t
                     int8_t * pFilter,      // may be null
                     int64_t arrayLength, int64_t hashLength)
{
    INDEX * pInput = (INDEX *)pInputIndex;
    int64_t * pNewFirst = (int64_t *)pNewFirstIndex;
    int64_t invalid = (int64_t)(1LL << (sizeof(int64_t) * 8 - 1));

    // Fill with invalid
    for (int64_t i = 0; i < hashLength; i++)
    {
        pNewFirst[i] = invalid;
    }

    // NOTE: the uniquecount is currently not used
    int32_t uniquecount = 0;
    if (pFilter)
    {
        for (int64_t i = 0; i < arrayLength; i++)
        {
            if (pFilter[i])
            {
                INDEX index = pInput[i];
                // printf("[%lld] got index for %lld\n", (int64_t)index, i);

                if (index > 0 && index < hashLength)
                {
                    // Check hash
                    if (pNewFirst[index] == invalid)
                    {
                        // First time, assign FirstKey
                        pNewFirst[index] = i;
                        uniquecount++;
                    }
                }
            }
        }
    }
    else
    {
        // When no filter provided
        for (int64_t i = 0; i < arrayLength; i++)
        {
            INDEX index = pInput[i];

            if (index > 0 && index < hashLength)
            {
                // Check hash
                if (pNewFirst[index] == invalid)
                {
                    // First time, assign FirstKey
                    pNewFirst[index] = i;
                    uniquecount++;
                }
            }
        }
    }
    return uniquecount;
}

template <typename INDEX>
int64_t iLastFilter(void * pInputIndex,
                    void * pNewLastIndex, // newly allocated NOTE: for > 2e9 should be int64_t
                    int8_t * pFilter,     // may be null
                    int64_t arrayLength, int64_t hashLength)
{
    INDEX * pInput = (INDEX *)pInputIndex;
    int64_t * pNewLast = (int64_t *)pNewLastIndex;
    int64_t invalid = (int64_t)(1LL << (sizeof(int64_t) * 8 - 1));

    // Fill with invalid
    for (int64_t i = 0; i < hashLength; i++)
    {
        pNewLast[i] = invalid;
    }

    if (pFilter)
    {
        for (int64_t i = 0; i < arrayLength; i++)
        {
            if (pFilter[i])
            {
                INDEX index = pInput[i];
                if (index > 0 && index < hashLength)
                {
                    // assign current LastKey
                    pNewLast[index] = i;
                }
            }
        }
    }
    else
    {
        // When no filter provided
        for (int64_t i = 0; i < arrayLength; i++)
        {
            INDEX index = pInput[i];

            if (index > 0 && index < hashLength)
            {
                // assign current LastKey
                pNewLast[index] = i;
            }
        }
    }
    // last does not keep track of uniquecount
    return 0;
}

//=====================================================================================
// Input:
// Arg1: Index array
// Arg2: Max uniques
// Arg3: Boolean array to filter on (or None)
// Arg4: integer set to 0 for first, 1 for last
//
// Output:
// New First Array (can use to pull in key names)
// UniqueCount (should be size of FirstArray)... possibly 0 if everything
// removed
PyObject * MakeiFirst(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    int64_t inArr1Max = 0;
    PyObject * inFilter = NULL;
    int64_t isLast = 0;

    int64_t out_dtype = 0;

    if (! PyArg_ParseTuple(args, "O!LOL:MakeiFirst", &PyArray_Type, &inArr1, &inArr1Max, &inFilter, &isLast))
    {
        return NULL;
    }

    void * pDataIn1 = PyArray_BYTES(inArr1);
    int8_t * pFilterIn = NULL;

    int ndim = PyArray_NDIM(inArr1);
    npy_intp * dims = PyArray_DIMS(inArr1);
    int64_t arraySize1 = CalcArrayLength(ndim, dims);

    if (PyArray_Check(inFilter))
    {
        if (arraySize1 != ArrayLength((PyArrayObject *)inFilter))
        {
            PyErr_Format(PyExc_ValueError, "MakeiFirst: Filter size not the same %lld", arraySize1);
            return NULL;
        }
        if (PyArray_TYPE((PyArrayObject *)inFilter) != NPY_BOOL)
        {
            PyErr_Format(PyExc_ValueError, "MakeiFirst: Filter is not type NPY_BOOL");
            return NULL;
        }
        pFilterIn = (int8_t *)PyArray_BYTES((PyArrayObject *)inFilter);
    }

    inArr1Max++;
    int64_t hashSize = inArr1Max;
    // printf("Combine hashsize is %lld     %lld\n", hashSize, inArr1Max);
    if (hashSize < 0 || hashSize > 20000000000LL)
    {
        PyErr_Format(PyExc_ValueError,
                     "MakeiFirst: Index sizes are either 0, negative, or produce "
                     "more than 20 billion results %lld",
                     hashSize);
        return NULL;
    }

    int dtype = PyArray_TYPE(inArr1);

    IFIRST_FILTER pFunction = NULL;

    if (isLast)
    {
        switch (dtype)
        {
        case NPY_INT8:
            pFunction = iLastFilter<int8_t>;
            break;
        case NPY_INT16:
            pFunction = iLastFilter<int16_t>;
            break;
        CASE_NPY_INT32:
            pFunction = iLastFilter<int32_t>;
            break;
        CASE_NPY_INT64:
            pFunction = iLastFilter<int64_t>;
            break;
        }
    }
    else
    {
        switch (dtype)
        {
        case NPY_INT8:
            pFunction = iFirstFilter<int8_t>;
            break;
        case NPY_INT16:
            pFunction = iFirstFilter<int16_t>;
            break;
        CASE_NPY_INT32:
            pFunction = iFirstFilter<int32_t>;
            break;
        CASE_NPY_INT64:
            pFunction = iFirstFilter<int64_t>;
            break;
        }
    }

    if (pFunction != NULL)
    {
        PyArrayObject * firstArray = AllocateNumpyArray(1, (npy_intp *)&hashSize, NPY_INT64);
        CHECK_MEMORY_ERROR(firstArray);

        if (firstArray)
        {
            void * pFirst = PyArray_BYTES(firstArray);

            int64_t uniqueCount = pFunction(pDataIn1, pFirst, pFilterIn, arraySize1, hashSize);

            return (PyObject *)firstArray;
        }
    }
    return NULL;
}

//=====================================================================================
//
void TrailingSpaces(char * pStringArray, int64_t length, int64_t itemSize)
{
    for (int64_t i = 0; i < length; i++)
    {
        char * pStart = pStringArray + (i * itemSize);
        char * pEnd = pStart + itemSize - 1;
        while (pEnd >= pStart && (*pEnd == ' ' || *pEnd == 0))
        {
            *pEnd-- = 0;
        }
    }
}

//=====================================================================================
//
void TrailingSpacesUnicode(uint32_t * pUnicodeArray, int64_t length, int64_t itemSize)
{
    itemSize = itemSize / 4;

    for (int64_t i = 0; i < length; i++)
    {
        uint32_t * pStart = pUnicodeArray + (i * itemSize);
        uint32_t * pEnd = pStart + itemSize - 1;
        while (pEnd >= pStart && (*pEnd == 32 || *pEnd == 0))
        {
            *pEnd-- = 0;
        }
    }
}

//=====================================================================================
// Arg1: array to strip trailing spaces
//
// Returns converted array or NULL
PyObject * RemoveTrailingSpaces(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;

    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &inArr1))
    {
        return NULL;
    }

    int dtype = PyArray_TYPE(inArr1);

    if (dtype == NPY_STRING || dtype == NPY_UNICODE)
    {
        void * pDataIn1 = PyArray_BYTES(inArr1);
        int64_t arraySize1 = ArrayLength(inArr1);
        int64_t itemSize = PyArray_ITEMSIZE(inArr1);

        if (dtype == NPY_STRING)
        {
            TrailingSpaces((char *)pDataIn1, arraySize1, itemSize);
        }
        else
        {
            TrailingSpacesUnicode((uint32_t *)pDataIn1, arraySize1, itemSize);
        }

        Py_IncRef((PyObject *)inArr1);
        return (PyObject *)inArr1;
    }

    PyErr_Format(PyExc_ValueError,
                 "Dont know how to convert these types %d.  Please make sure to "
                 "pass a string.",
                 dtype);
    return NULL;
}

int GetUpcastDtype(ArrayInfo * aInfo, int64_t tupleSize)
{
    int maxfloat = -1;
    int maxint = -1;
    int maxuint = -1;
    int maxstring = -1;
    int maxobject = -1;
    int abort = 0;

    for (int t = 0; t < tupleSize; t++)
    {
        int tempdtype = aInfo[t].NumpyDType;
        if (tempdtype <= NPY_LONGDOUBLE)
        {
            if (tempdtype >= NPY_FLOAT)
            {
                if (tempdtype > maxfloat)
                {
                    maxfloat = tempdtype;
                }
            }
            else
            {
                if (tempdtype & 1 || tempdtype == 0)
                {
                    if (tempdtype > maxint)
                    {
                        maxint = tempdtype;
                    }
                }
                else
                {
                    if (tempdtype > maxuint)
                    {
                        maxuint = tempdtype;
                    }
                }
            }
        }
        else
        {
            if (tempdtype == NPY_OBJECT)
            {
                maxobject = NPY_OBJECT;
            }
            else if (tempdtype == NPY_UNICODE)
            {
                maxstring = NPY_UNICODE;
            }
            else if (tempdtype == NPY_STRING)
            {
                if (maxstring < NPY_STRING)
                {
                    maxstring = NPY_STRING;
                }
            }
            else
            {
                abort = tempdtype;
            }
        }
    }

    if (abort > 0)
    {
        return -1;
    }

    // Return in this order:
    // OBJECT
    // UNICODE
    // STRING
    if (maxobject == NPY_OBJECT)
    {
        return NPY_OBJECT;
    }

    if (maxstring > 0)
    {
        // return either NPY_UNICODE or NPY_STRING
        return maxstring;
    }

    if (maxfloat > 0)
    {
        // do we have a float?
        if (maxfloat > NPY_FLOAT)
        {
            return maxfloat;
        }

        // we have a float... see if we have integers that force a double
        if (maxint > NPY_INT16 || maxuint > NPY_UINT16)
        {
            return NPY_DOUBLE;
        }

        return maxfloat;
    }
    else
    {
        if (maxuint > 0)
        {
            // Do we have a uint and no floats?
            if (maxint > maxuint)
            {
                // we can safely upcast the uint to maxint
                return maxint;
            }

            // check if any ints
            if (maxint == -1)
            {
                // no integers and no floats
                return maxuint;
            }

            if (sizeof(long) == 8)
            {
                // gcc/linux path
                // if maxuint is hit and we have integers, force to go to double
                if (maxuint == NPY_ULONGLONG || maxuint == NPY_ULONG)
                {
                    return NPY_DOUBLE;
                }
                if (maxint == NPY_LONG || maxint == NPY_LONGLONG)
                {
                    return NPY_DOUBLE;
                }

                // we have ints, go to next higher int
                return (maxuint + 1);
            }
            else
            {
                if (maxuint == NPY_ULONGLONG)
                {
                    return NPY_DOUBLE;
                }
                if (maxint == NPY_LONG)
                {
                    return NPY_DOUBLE;
                }

                // we have ints, go to next higher int
                return (maxuint + 1);
            }

            // should not get here
            return maxuint;
        }
        else
        {
            // we have just ints or bools
            return maxint;
        }
    }
}

//----------------------------------------------------
// Arg1: Pass in list of arrays
//
// Returns: dtype num to upcast to
//          may return -1 on impossible
PyObject * GetUpcastNum(PyObject * self, PyObject * args)
{
    PyObject * inList1 = NULL;

    if (! PyArg_ParseTuple(args, "O", &inList1))
    {
        return NULL;
    }

    int64_t totalItemSize = 0;
    int64_t tupleSize = 0;

    // Allow jagged rows
    // Do not copy
    ArrayInfo * aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, false, false);

    if (aInfo)
    {
        int dtype = GetUpcastDtype(aInfo, tupleSize);
        FreeArrayInfo(aInfo);
        return PyLong_FromLong(dtype);
    }
    return NULL;
}

//----------------------------------------------------
// Arg1: Pass in list of arrays
// Arg2: (optional) default dtype num (for possible upcast)
// Returns single array concatenatedessed
//
PyObject * HStack(PyObject * self, PyObject * args)
{
    PyObject * inList1 = NULL;
    int32_t dtype = -1;

    if (! PyArg_ParseTuple(args, "O|i", &inList1, &dtype))
    {
        return NULL;
    }

    if (dtype != -1)
    {
        if (dtype < 0 || dtype > NPY_LONGDOUBLE)
        {
            PyErr_Format(PyExc_ValueError,
                         "Dont know how to convert dtype num %d.  Please make sure "
                         "all arrays are ints or floats.",
                         dtype);
            return NULL;
        }
    }

    int64_t totalItemSize = 0;
    int64_t tupleSize = 0;

    // Allow jagged rows
    ArrayInfo * aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, false);

    if (aInfo)
    {
        int64_t totalLength = 0;
        int64_t maxItemSize = 0;
        PyArrayObject * outputArray = NULL;

        if (dtype == -1)
        {
            dtype = GetUpcastDtype(aInfo, tupleSize);

            if (dtype < 0 || dtype > NPY_LONGDOUBLE)
            {
                bool isSameDtype = true;

                // Check for all strings or all unicode which we know how to stack
                for (int t = 0; t < tupleSize; t++)
                {
                    if (dtype != aInfo[t].NumpyDType)
                    {
                        isSameDtype = false;
                        break;
                    }
                    // track max itemsize since for a string we must match it
                    if (aInfo[t].ItemSize > maxItemSize)
                    {
                        maxItemSize = aInfo[t].ItemSize;
                    }
                }

                // Check for strings
                if ((dtype == NPY_STRING || dtype == NPY_UNICODE) && isSameDtype)
                {
                    // they are all strings/unicode
                }
                else
                {
                    PyErr_Format(PyExc_ValueError,
                                 "Dont know how to convert dtype num %d.  Please make "
                                 "sure all arrays are ints or floats.",
                                 dtype);
                    return NULL;
                }
            }
        }

        if (dtype == NPY_STRING || dtype == NPY_UNICODE)
        {
            //
            // Path for strings
            //
            struct stHSTACK_STRING
            {
                int64_t Offset;
                int64_t ItemSize;
                CONVERT_SAFE_STRING ConvertSafeString;
            };

            stHSTACK_STRING * pHStack = (stHSTACK_STRING *)WORKSPACE_ALLOC(sizeof(stHSTACK_STRING) * tupleSize);
            // calculate total size and get conversion function for each row
            for (int t = 0; t < tupleSize; t++)
            {
                int a_dtype = aInfo[t].NumpyDType;

                pHStack[t].ConvertSafeString = ConvertSafeStringCopy;
                pHStack[t].Offset = totalLength;
                pHStack[t].ItemSize = aInfo[t].ItemSize;
                totalLength += aInfo[t].ArrayLength;
            }

            if (dtype == NPY_STRING || dtype == NPY_UNICODE)
            {
                // string allocation
                outputArray = AllocateNumpyArray(1, (npy_intp *)&totalLength, dtype, maxItemSize);
            }
            else
            {
                outputArray = AllocateNumpyArray(1, (npy_intp *)&totalLength, dtype);
            }

            CHECK_MEMORY_ERROR(outputArray);
            if (outputArray)
            {
                int64_t itemSize = PyArray_ITEMSIZE(outputArray);
                char * pOutput = (char *)PyArray_BYTES(outputArray);

                if (tupleSize < 2 || totalLength <= g_cMathWorker->WORK_ITEM_CHUNK)
                {
                    for (int t = 0; t < tupleSize; t++)
                    {
                        pHStack[t].ConvertSafeString(aInfo[t].pData, pOutput + (pHStack[t].Offset * itemSize),
                                                     aInfo[t].ArrayLength, pHStack[t].ItemSize, itemSize);
                    }
                }
                else
                {
                    // Callback routine from multithreaded worker thread (items just count
                    // up from 0,1,2,...)
                    // typedef BOOL(*MTWORK_CALLBACK)(void* callbackArg, int core, int64_t
                    // workIndex);
                    struct stSHSTACK
                    {
                        stHSTACK_STRING * pHStack;
                        ArrayInfo * aInfo;
                        char * pOutput;
                        int64_t ItemSizeOutput;
                    } myhstack;

                    myhstack.pHStack = pHStack;
                    myhstack.aInfo = aInfo;
                    myhstack.pOutput = pOutput;
                    myhstack.ItemSizeOutput = itemSize;

                    LOGGING("MT string hstack work on %lld\n", tupleSize);

                    auto lambdaHSCallback = [](void * callbackArgT, int core, int64_t workIndex) -> bool
                    {
                        stSHSTACK * callbackArg = (stSHSTACK *)callbackArgT;
                        int64_t t = workIndex;
                        callbackArg->pHStack[t].ConvertSafeString(
                            callbackArg->aInfo[t].pData,
                            callbackArg->pOutput + (callbackArg->pHStack[t].Offset * callbackArg->ItemSizeOutput),
                            callbackArg->aInfo[t].ArrayLength, callbackArg->aInfo[t].ItemSize, callbackArg->ItemSizeOutput);

                        return true;
                    };

                    g_cMathWorker->DoMultiThreadedWork((int)tupleSize, lambdaHSCallback, &myhstack);
                }
            }

            WORKSPACE_FREE(pHStack);
        }
        else
        {
            // Path for non-strings
            struct stHSTACK
            {
                int64_t Offset;
                void * pBadInput1;
                CONVERT_SAFE ConvertSafe;
            };

            stHSTACK * pHStack = (stHSTACK *)WORKSPACE_ALLOC(sizeof(stHSTACK) * tupleSize);

            // calculate total size and get conversion function for each row
            for (int t = 0; t < tupleSize; t++)
            {
                int a_dtype = aInfo[t].NumpyDType;

                if (a_dtype > NPY_LONGDOUBLE || aInfo[t].NDim != 1)
                {
                    FreeArrayInfo(aInfo);
                    PyErr_Format(PyExc_ValueError,
                                 "Dont know how to convert dtype num %d or more than 1 dimension. "
                                 " Please make sure all arrays are ints or floats.",
                                 a_dtype);
                    return NULL;
                }

                pHStack[t].ConvertSafe = GetConversionFunctionSafe(aInfo[t].NumpyDType, dtype);
                pHStack[t].Offset = totalLength;
                pHStack[t].pBadInput1 = GetInvalid(aInfo[t].NumpyDType);
                totalLength += aInfo[t].ArrayLength;
            }

            outputArray = AllocateNumpyArray(1, (npy_intp *)&totalLength, dtype);
            CHECK_MEMORY_ERROR(outputArray);

            if (outputArray)
            {
                // if output is boolean, bad means false
                void * pBadOutput1 = GetDefaultForType(dtype);

                int64_t strideOut = PyArray_STRIDE(outputArray, 0);
                char * pOutput = (char *)PyArray_BYTES(outputArray);

                if (tupleSize < 2 || totalLength <= g_cMathWorker->WORK_ITEM_CHUNK)
                {
                    for (int t = 0; t < tupleSize; t++)
                    {
                        pHStack[t].ConvertSafe(aInfo[t].pData, pOutput + (pHStack[t].Offset * strideOut), aInfo[t].ArrayLength,
                                               pHStack[t].pBadInput1, pBadOutput1, PyArray_STRIDE(aInfo[t].pObject, 0), strideOut);
                    }
                }
                else
                {
                    // Callback routine from multithreaded worker thread (items just count
                    // up from 0,1,2,...)
                    // typedef BOOL(*MTWORK_CALLBACK)(void* callbackArg, int core, int64_t
                    // workIndex);
                    struct stSHSTACK
                    {
                        stHSTACK * pHStack;
                        ArrayInfo * aInfo;
                        char * pOutput;
                        int64_t StrideOut;
                        void * pBadOutput1;
                    } myhstack;

                    myhstack.pHStack = pHStack;
                    myhstack.aInfo = aInfo;
                    myhstack.pOutput = pOutput;
                    myhstack.StrideOut = strideOut;
                    myhstack.pBadOutput1 = pBadOutput1;

                    LOGGING("MT hstack work on %lld\n", tupleSize);

                    auto lambdaHSCallback = [](void * callbackArgT, int core, int64_t workIndex) -> bool
                    {
                        stSHSTACK * callbackArg = (stSHSTACK *)callbackArgT;
                        int64_t t = workIndex;
                        callbackArg->pHStack[t].ConvertSafe(
                            callbackArg->aInfo[t].pData,
                            callbackArg->pOutput + (callbackArg->pHStack[t].Offset * callbackArg->StrideOut),
                            callbackArg->aInfo[t].ArrayLength, callbackArg->pHStack[t].pBadInput1, callbackArg->pBadOutput1,
                            PyArray_STRIDE(callbackArg->aInfo[t].pObject, 0), callbackArg->StrideOut);

                        return true;
                    };

                    g_cMathWorker->DoMultiThreadedWork((int)tupleSize, lambdaHSCallback, &myhstack);
                }
            }
            WORKSPACE_FREE(pHStack);
        }

        FreeArrayInfo(aInfo);

        if (! outputArray)
        {
            PyErr_Format(PyExc_ValueError, "hstack out of memory");
            return NULL;
        }

        return (PyObject *)outputArray;
    }

    return NULL;
}

//----------------------------------------------------
// Arg1: Pass in list of arrays
// Arg2: +/- amount to shift
// Returns each array shifted
//
PyObject * ShiftArrays(PyObject * self, PyObject * args)
{
    PyObject * inList1 = NULL;
    int64_t shiftAmount = 0;

    if (! PyArg_ParseTuple(args, "OL", &inList1, &shiftAmount))
    {
        return NULL;
    }

    int64_t totalItemSize = 0;
    int64_t tupleSize = 0;

    // Allow jagged rows
    ArrayInfo * aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, false);

    if (aInfo)
    {
        int64_t totalLength = 0;

        // Callback routine from multithreaded worker thread (items just count up
        // from 0,1,2,...)
        // typedef BOOL(*MTWORK_CALLBACK)(void* callbackArg, int core, int64_t
        // workIndex);
        struct stSHIFT
        {
            ArrayInfo * aInfo;
            int64_t shiftAmount;
        } myshift;

        myshift.aInfo = aInfo;
        myshift.shiftAmount = shiftAmount;

        auto lambdaShiftCallback = [](void * callbackArgT, int core, int64_t workIndex) -> bool
        {
            stSHIFT * pShift = (stSHIFT *)callbackArgT;
            int64_t t = workIndex;

            ArrayInfo * pArrayInfo = &pShift->aInfo[t];

            if (pArrayInfo->pData)
            {
                npy_intp * const pStrides = PyArray_STRIDES(pArrayInfo->pObject);

                // Check for fortran style
                if (pArrayInfo->NDim >= 2 && pStrides[0] < pStrides[1])
                {
                    npy_intp * const pDims = PyArray_DIMS(pArrayInfo->pObject);

                    //-- BEFORE SHIFT rows:5  cols: 3
                    // a d g j m   0  1  2  3  4
                    // b e h k n   5  6  7  8  9
                    // c f i l o  10 11 12 13 14
                    //
                    //-- AFTER SHIFT of 2
                    // g j m x x   0  1  2  3  4
                    // h k n x x   5  6  7  8  9
                    // i l o x x  10 11 12 13 14
                    //
                    int64_t rows = pDims[0];
                    int64_t cols = pDims[1];

                    LOGGING("!! encountered fortran array while shifting! %lld x %lld\n", rows, cols);

                    if (pArrayInfo->NDim >= 3)
                    {
                        printf("!! too many dimensions to shift! %lld x %lld\n", rows, cols);
                    }
                    else
                    {
                        char * pDst = pArrayInfo->pData;
                        char * pSrc = pDst + (pShift->shiftAmount * pArrayInfo->ItemSize);

                        int64_t rowsToMove = rows - pShift->shiftAmount;

                        // Check for negative shift value
                        if (pShift->shiftAmount < 0)
                        {
                            rowsToMove = rows + pShift->shiftAmount;
                            pSrc = pArrayInfo->pData;
                            pDst = pSrc - (pShift->shiftAmount * pArrayInfo->ItemSize);
                        }

                        if (rowsToMove > 0)
                        {
                            int64_t rowsToMoveSize = rowsToMove * pArrayInfo->ItemSize;
                            int64_t rowSize = rows * pArrayInfo->ItemSize;

                            // 2d shift
                            for (int64_t i = 0; i < cols; i++)
                            {
                                memmove(pDst, pSrc, rowsToMoveSize);
                                pDst += rowSize;
                                pSrc += rowSize;
                            }
                        }
                    }
                }
                else
                {
                    // Example:
                    // ArrayLength: 10000
                    // shiftAmount:  1000
                    // deltaShift:   9000 items to move
                    //
                    int64_t deltaShift = pArrayInfo->ArrayLength - pShift->shiftAmount;
                    if (pShift->shiftAmount < 0)
                    {
                        deltaShift = pArrayInfo->ArrayLength + pShift->shiftAmount;
                    }
                    // make sure something to shift
                    if (deltaShift > 0)
                    {
                        char * pTop1 = pArrayInfo->pData;

                        // make sure something to shift
                        if (pShift->shiftAmount < 0)
                        {
                            char * pTop2 = pTop1 - (pShift->shiftAmount * pArrayInfo->ItemSize);
                            LOGGING("[%d] neg shifting %p %p  size: %lld  itemsize: %lld\n", core, pTop2, pTop1, deltaShift,
                                    pArrayInfo->ItemSize);
                            memmove(pTop2, pTop1, deltaShift * pArrayInfo->ItemSize);
                        }
                        else
                        {
                            char * pTop2 = pTop1 + (pShift->shiftAmount * pArrayInfo->ItemSize);
                            LOGGING("[%d] pos shifting %p %p  size: %lld  itemsize: %lld\n", core, pTop1, pTop2, deltaShift,
                                    pArrayInfo->ItemSize);
                            memmove(pTop1, pTop2, deltaShift * pArrayInfo->ItemSize);
                        }
                    }
                }
            }
            return true;
        };

        g_cMathWorker->DoMultiThreadedWork((int)tupleSize, lambdaShiftCallback, &myshift);

        FreeArrayInfo(aInfo);
        Py_IncRef(inList1);
        return inList1;
    }

    PyErr_Format(PyExc_ValueError, "Unable to shift arrays");
    return NULL;
}

//-----------------------
// HomogenizeArrays
// Arg1: List of numpy arrays
// Arg2: Optional final dtype
//
// Returns: list of homogenized arrays
PyObject * HomogenizeArrays(PyObject * self, PyObject * args)
{
    if (! PyTuple_Check(args))
    {
        PyErr_Format(PyExc_ValueError, "HomogenizeArrays arguments needs to be a tuple");
        return NULL;
    }

    Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

    if (argTupleSize < 2)
    {
        PyErr_Format(PyExc_ValueError, "HomogenizeArrays requires two args instead of %llu args", argTupleSize);
        return NULL;
    }

    PyObject * inList1 = PyTuple_GetItem(args, 0);
    PyObject * dtypeObject = PyTuple_GetItem(args, 1);

    int64_t totalItemSize = 0;
    int64_t tupleSize = 0;

    // Do not allow jagged rows
    ArrayInfo * aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, true);

    if (aInfo)
    {
        int32_t dtype = (int32_t)PyLong_AsLong(dtypeObject);

        if (dtype != -1)
        {
            if (dtype < 0 || dtype > NPY_LONGDOUBLE)
            {
                PyErr_Format(PyExc_ValueError,
                             "HomogenizeArrays: Dont know how to convert dtype num %d. "
                             " Please make sure all arrays are ints or floats.",
                             dtype);
                return NULL;
            }
        }
        else
        {
            dtype = GetUpcastDtype(aInfo, tupleSize);
            if (dtype == -1)
            {
                return NULL;
            }
        }

        // Now convert?
        // if output is boolean, bad means false
        void * pBadOutput1 = GetDefaultForType(dtype);

        PyObject * returnList = PyList_New(0);

        // Convert any different types... build a new list...
        for (int t = 0; t < tupleSize; t++)
        {
            if (dtype != aInfo[t].NumpyDType)
            {
                CONVERT_SAFE convertSafe = GetConversionFunctionSafe(aInfo[t].NumpyDType, dtype);
                void * pBadInput1 = GetDefaultForType(aInfo[t].NumpyDType);
                PyArrayObject * pOutput = AllocateLikeNumpyArray(aInfo[t].pObject, dtype);

                // TODO: multithread this
                if (pOutput)
                {
                    // preserve sentinels
                    convertSafe(aInfo[t].pData, PyArray_BYTES(pOutput), aInfo[t].ArrayLength, pBadInput1, pBadOutput1,
                                PyArray_STRIDE(aInfo[t].pObject, 0), PyArray_STRIDE(pOutput, 0));

                    // pylist_append will add a reference count but setitem will not
                    PyList_Append(returnList, (PyObject *)pOutput);
                    Py_DecRef((PyObject *)pOutput);
                }
            }
            else
            {
                // add a refernce
                PyList_Append(returnList, (PyObject *)aInfo[t].pObject);
            }
        }

        // Figure out which arrays will be recast

        FreeArrayInfo(aInfo);
        return returnList;
    }

    return NULL;
}

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
int64_t SumBooleanMask(const int8_t * const pData, const int64_t length)
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

    // Now that we know length is >= 0, it's safe to convert it to unsigned so it
    // agrees with the sizeof() math in the logic below. Make sure to use this
    // instead of 'length' in the code below to avoid signed/unsigned arithmetic
    // warnings.
    const auto ulength = static_cast<size_t>(length);

    // Holds the accumulated result value.
    int64_t result = 0;

    // YMM (32-byte) vector packed with 32 byte values, each set to 1.
    // NOTE: The obvious thing here would be to use _mm256_set1_epi8(1),
    //       but many compilers (e.g. MSVC) store the data for this vector
    //       then load it here, which unnecessarily wastes cache space we could be
    //       using for something else.
    //       Generate the constants using a few intrinsics, it's faster than even
    //       an L1 cache hit anyway.
    const auto zeros_ = _mm256_setzero_si256();
    // compare 0 to 0 returns 0xFF; treated as an int8_t, 0xFF = -1, so abs(-1)
    // = 1.
    const auto ones = _mm256_abs_epi8(_mm256_cmpeq_epi8(zeros_, zeros_));

    //
    // Convert each byte in the input to a 0 or 1 byte according to C-style
    // boolean semantics.
    //

    // This first loop does the bulk of the processing for large vectors -- it
    // doesn't use popcount instructions and instead relies on the fact we can sum
    // 0/1 values to acheive the same result, up to CHAR_MAX. This allows us to
    // use very inexpensive instructions for most of the accumulation so we're
    // primarily limited by memory bandwidth.
    const size_t vector_length = ulength / sizeof(__m256i);
    const auto pVectorData = (__m256i *)pData;
    for (size_t i = 0; i < vector_length;)
    {
        // Determine how much we can process in _this_ iteration of the loop.
        // The maximum number of "inner" iterations here is CHAR_MAX (255),
        // because otherwise our byte-sized counters would overflow.
        const auto inner_loop_iters = std::min(static_cast<size_t>(std::numeric_limits<uint8_t>::max()), vector_length - i);

        // Holds the current per-vector-lane (i.e. per-byte-within-vector) popcount.
        // PERF: If necessary, the loop below can be manually unrolled to ensure we
        // saturate memory bandwidth.
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
        // used as the second operand so that the zeros are 'unpacked' into the high
        // byte(s) of each packed element in the result.
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

        // Sum 4x 8-byte counts -> 1x 32-byte count.
        const auto byte_popcount_256 = _mm256_extract_epi64(byte_popcounts_64, 0) + _mm256_extract_epi64(byte_popcounts_64, 1) +
                                       _mm256_extract_epi64(byte_popcounts_64, 2) + _mm256_extract_epi64(byte_popcounts_64, 3);

        // Add the accumulated popcount from this loop iteration (for 32*255 bytes)
        // to the overall result.
        result += byte_popcount_256;

        // Increment the outer loop counter by the number of inner iterations we
        // performed.
        i += inner_loop_iters;
    }

    // Handle the last few bytes, if any, that couldn't be handled with the
    // vectorized loop.
    const auto vectorized_length = vector_length * sizeof(__m256i);
    for (size_t i = vectorized_length; i < ulength; i++)
    {
        if (pData[i])
        {
            result++;
        }
    }

    return result;
}

//--------------------------------------------------------------------------
// Array copy from one to another using boolean mask
void CopyItemBooleanMask(void * pSrcV, void * pDestV, int8_t * pBoolMask, int64_t arrayLength, int64_t itemsize)
{
    switch (itemsize)
    {
    case 1:
        {
            int8_t * pSrc = (int8_t *)pSrcV;
            int8_t * pDest = (int8_t *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    pDest[i] = *pSrc++;
                }
            }
            break;
        }
    case 2:
        {
            int16_t * pSrc = (int16_t *)pSrcV;
            int16_t * pDest = (int16_t *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    pDest[i] = *pSrc++;
                }
            }
            break;
        }
    case 4:
        {
            int32_t * pSrc = (int32_t *)pSrcV;
            int32_t * pDest = (int32_t *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    pDest[i] = *pSrc++;
                }
            }
            break;
        }
    case 8:
        {
            int64_t * pSrc = (int64_t *)pSrcV;
            int64_t * pDest = (int64_t *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    pDest[i] = *pSrc++;
                }
            }
            break;
        }
    default:
        {
            char * pSrc = (char *)pSrcV;
            char * pDest = (char *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    memcpy(pDest + (itemsize * i), pSrc, itemsize);
                    pSrc += itemsize;
                }
            }
            break;
        }
    }
}

//--------------------------------------------------------------------------
// Copying scalars to array with boolean mask
void CopyItemBooleanMaskScalar(void * pSrcV, void * pDestV, int8_t * pBoolMask, int64_t arrayLength, int64_t itemsize)
{
    switch (itemsize)
    {
    case 1:
        {
            int8_t * pSrc = (int8_t *)pSrcV;
            int8_t * pDest = (int8_t *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    pDest[i] = *pSrc;
                }
            }
            break;
        }
    case 2:
        {
            int16_t * pSrc = (int16_t *)pSrcV;
            int16_t * pDest = (int16_t *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    pDest[i] = *pSrc;
                }
            }
            break;
        }
    case 4:
        {
            int32_t * pSrc = (int32_t *)pSrcV;
            int32_t * pDest = (int32_t *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    pDest[i] = *pSrc;
                }
            }
            break;
        }
    case 8:
        {
            int64_t * pSrc = (int64_t *)pSrcV;
            int64_t * pDest = (int64_t *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    pDest[i] = *pSrc;
                }
            }
            break;
        }
    default:
        {
            char * pSrc = (char *)pSrcV;
            char * pDest = (char *)pDestV;

            for (int64_t i = 0; i < arrayLength; i++)
            {
                if (*pBoolMask++)
                {
                    memcpy(pDest + (itemsize * i), pSrc, itemsize);
                }
            }
            break;
        }
    }
}

//--------------------------------------------------------------------------
//
PyObject * SetItemBooleanMask(PyArrayObject * arr, PyArrayObject * mask, PyArrayObject * inValues, int64_t arrayLength)
{
    // PyObject* boolsum =
    //   ReduceInternal(mask, REDUCE_SUM);
    int8_t * pBoolMask = (int8_t *)PyArray_BYTES(mask);
    int64_t bsum = SumBooleanMask(pBoolMask, ArrayLength(mask));

    // Sum the boolean array
    if (bsum > 0 && bsum == ArrayLength(inValues))
    {
        void * pSrc = PyArray_BYTES(inValues);
        void * pDest = PyArray_BYTES(arr);

        CopyItemBooleanMask(pSrc, pDest, pBoolMask, arrayLength, PyArray_ITEMSIZE(arr));

        Py_IncRef(Py_True);
        return Py_True;
    }

    Py_IncRef(Py_False);
    return Py_False;
}

//--------------------------------------------------------------------------
// Must be a power of 2
// For smaller arrays, change this size
#define SETITEM_PARTITION_SIZE 16384

//--------------------------------------------------------------------------
// TODO: Refactor this method and SetItemBooleanMask
PyObject * SetItemBooleanMaskLarge(PyArrayObject * arr, PyArrayObject * mask, PyArrayObject * inValues, int64_t arrayLength)
{
    struct ST_BOOLCOUNTER
    {
        int8_t * pBoolMask;
        int64_t * pCounts;
        int64_t sections;
        int64_t maskLength;
        char * pSrc;
        char * pDest;
        int64_t itemSize;

    } stBoolCounter;

    stBoolCounter.pSrc = (char *)PyArray_BYTES(inValues);
    stBoolCounter.pDest = (char *)PyArray_BYTES(arr);
    stBoolCounter.itemSize = PyArray_ITEMSIZE(arr);

    stBoolCounter.maskLength = ArrayLength(mask);
    stBoolCounter.sections = (stBoolCounter.maskLength + (SETITEM_PARTITION_SIZE - 1)) / SETITEM_PARTITION_SIZE;

    int64_t allocSize = stBoolCounter.sections * sizeof(int64_t);
    const int64_t maxStackAlloc = 1024 * 1024; // 1 MB

    if (allocSize > maxStackAlloc)
    {
        stBoolCounter.pCounts = (int64_t *)WORKSPACE_ALLOC(allocSize);
    }
    else
    {
        stBoolCounter.pCounts = (int64_t *)alloca(allocSize);
    }
    if (stBoolCounter.pCounts)
    {
        stBoolCounter.pBoolMask = (int8_t *)PyArray_BYTES(mask);

        auto lambdaCallback = [](void * callbackArgT, int core, int64_t workIndex) -> bool
        {
            ST_BOOLCOUNTER * pstBoolCounter = (ST_BOOLCOUNTER *)callbackArgT;
            int64_t t = workIndex;

            int64_t lastCount = SETITEM_PARTITION_SIZE;
            if (t == pstBoolCounter->sections - 1)
            {
                lastCount = pstBoolCounter->maskLength & (SETITEM_PARTITION_SIZE - 1);
                if (lastCount == 0)
                    lastCount = SETITEM_PARTITION_SIZE;
            }
            pstBoolCounter->pCounts[workIndex] =
                SumBooleanMask(pstBoolCounter->pBoolMask + (SETITEM_PARTITION_SIZE * workIndex), lastCount);
            // printf("isum %lld %lld %lld\n", workIndex,
            // pstBoolCounter->pCounts[workIndex], lastCount);

            return true;
        };

        g_cMathWorker->DoMultiThreadedWork((int)stBoolCounter.sections, lambdaCallback, &stBoolCounter);

        // calculate the sum for each section
        int64_t bsum = 0;
        for (int i = 0; i < stBoolCounter.sections; i++)
        {
            int64_t temp = bsum;
            bsum += stBoolCounter.pCounts[i];
            stBoolCounter.pCounts[i] = temp;
        }

        int64_t arrlength = ArrayLength(inValues);

        if (bsum > 0 && bsum == arrlength)
        {
            auto lambda2Callback = [](void * callbackArgT, int core, int64_t workIndex) -> bool
            {
                ST_BOOLCOUNTER * pstBoolCounter = (ST_BOOLCOUNTER *)callbackArgT;
                int64_t t = workIndex;

                int64_t lastCount = SETITEM_PARTITION_SIZE;
                if (t == pstBoolCounter->sections - 1)
                {
                    lastCount = pstBoolCounter->maskLength & (SETITEM_PARTITION_SIZE - 1);
                    if (lastCount == 0)
                        lastCount = SETITEM_PARTITION_SIZE;
                }
                int64_t adjustment = (SETITEM_PARTITION_SIZE * workIndex * pstBoolCounter->itemSize);
                CopyItemBooleanMask(pstBoolCounter->pSrc + (pstBoolCounter->pCounts[workIndex] * pstBoolCounter->itemSize),
                                    pstBoolCounter->pDest + adjustment,
                                    pstBoolCounter->pBoolMask + (SETITEM_PARTITION_SIZE * workIndex), lastCount,
                                    pstBoolCounter->itemSize);

                return true;
            };

            g_cMathWorker->DoMultiThreadedWork((int)stBoolCounter.sections, lambda2Callback, &stBoolCounter);

            if (allocSize > maxStackAlloc)
                WORKSPACE_FREE(stBoolCounter.pCounts);
            Py_IncRef(Py_True);
            return Py_True;
        }

        if (bsum > 0 && arrlength == 1)
        {
            auto lambda2Callback = [](void * callbackArgT, int core, int64_t workIndex) -> bool
            {
                ST_BOOLCOUNTER * pstBoolCounter = (ST_BOOLCOUNTER *)callbackArgT;
                int64_t t = workIndex;

                int64_t lastCount = SETITEM_PARTITION_SIZE;
                if (t == pstBoolCounter->sections - 1)
                {
                    lastCount = pstBoolCounter->maskLength & (SETITEM_PARTITION_SIZE - 1);
                    if (lastCount == 0)
                        lastCount = SETITEM_PARTITION_SIZE;
                }
                int64_t adjustment = (SETITEM_PARTITION_SIZE * workIndex * pstBoolCounter->itemSize);
                CopyItemBooleanMaskScalar(pstBoolCounter->pSrc, pstBoolCounter->pDest + adjustment,
                                          pstBoolCounter->pBoolMask + (SETITEM_PARTITION_SIZE * workIndex), lastCount,
                                          pstBoolCounter->itemSize);

                return true;
            };

            g_cMathWorker->DoMultiThreadedWork((int)stBoolCounter.sections, lambda2Callback, &stBoolCounter);

            if (allocSize > maxStackAlloc)
                WORKSPACE_FREE(stBoolCounter.pCounts);
            Py_IncRef(Py_True);
            return Py_True;
        }
        if (allocSize > maxStackAlloc)
            WORKSPACE_FREE(stBoolCounter.pCounts);
    }
    LOGGING("bsum problem %lld  %lld\n", bsum, arrlength);
    Py_IncRef(Py_False);
    return Py_False;
}

//--------------------------------------------------------------------------
// def __setitem__(self, fld, value) :
//   : param fld : boolean or fancy index mask
//   : param value : scalar, sequence or dataset value as follows
//
// returns true if it worked
// returns false
// NOTE: This routine is not finished yet
PyObject * SetItem(PyObject * self, PyObject * args)
{
    // if (!PyTuple_Check(args)) {
    //   PyErr_Format(PyExc_ValueError, "SetItem arguments needs to be a tuple");
    //   return NULL;
    //}

    Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

    if (argTupleSize < 3)
    {
        PyErr_Format(PyExc_ValueError, "SetItem requires three args instead of %llu args", argTupleSize);
        return NULL;
    }

    PyArrayObject * arr = (PyArrayObject *)PyTuple_GetItem(args, 0);
    PyArrayObject * mask = (PyArrayObject *)PyTuple_GetItem(args, 1);

    // Try to convert value if we have to
    PyObject * value = PyTuple_GetItem(args, 2);
    bool newValue{false};
    if (! PyArray_Check(value))
    {
        value = PyArray_FromAny(value, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
        newValue = true;
    }

    // Make sure from any worked
    if (value)
    {
        if (PyArray_Check(arr) && PyArray_Check(mask) && PyArray_Check(value))
        {
            PyArrayObject * inValues = (PyArrayObject *)value;

            // check for strides, same itemsize, 1 dimensional
            // TODO: improvement when string itemsize is different length -- we can do
            // a custom string copy
            if (PyArray_NDIM(arr) == 1 && PyArray_ITEMSIZE(arr) > 0 && PyArray_NDIM(inValues) == 1 &&
                PyArray_ITEMSIZE(inValues) == PyArray_ITEMSIZE(arr) && PyArray_TYPE(arr) == PyArray_TYPE(inValues))
            {
                // Boolean path...
                int arrType = PyArray_TYPE(mask);

                if (arrType == NPY_BOOL)
                {
                    int64_t arrayLength = ArrayLength(arr);

                    if (arrayLength == ArrayLength(mask))
                    {
                        PyObject * returnValue{nullptr};
                        
                        if (arrayLength <= SETITEM_PARTITION_SIZE)
                        {
                            returnValue =  SetItemBooleanMask(arr, mask, inValues, arrayLength);
                        }
                        else
                        {
                            // special count
                            returnValue = SetItemBooleanMaskLarge(arr, mask, inValues, arrayLength);
                        }
                        
                        if (newValue)
                        {
                            Py_DECREF(value);
                        }

                        return returnValue;
                    }
                }
            }
        }
        LOGGING("SetItem Could not convert value to array %d  %lld  %d  %lld\n", PyArray_NDIM(arr), PyArray_ITEMSIZE(arr),
                PyArray_NDIM((PyArrayObject *)value), PyArray_ITEMSIZE((PyArrayObject *)value));
        
        if (newValue)
        {
            Py_DECREF(value);
        }
    }
    else
    {
        LOGGING("SetItem Could not convert value to array\n");
    }
    // punt to numpy
    Py_IncRef(Py_False);
    return Py_False;
}

//----------------------------------------------------
// rough equivalvent arr[mask] = value[mask]
//
// returns true if it worked
// returns false
PyObject * PutMask(PyObject * self, PyObject * args)
{
    Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

    if (argTupleSize < 3)
    {
        PyErr_Format(PyExc_ValueError, "SetItem requires three args instead of %llu args", argTupleSize);
        return NULL;
    }

    PyArrayObject * arr = (PyArrayObject *)PyTuple_GetItem(args, 0);
    PyArrayObject * mask = (PyArrayObject *)PyTuple_GetItem(args, 1);

    // Try to convert value if we have to
    PyObject * value = PyTuple_GetItem(args, 2);
    if (! PyArray_Check(value))
    {
        value = PyArray_FromAny(value, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    }

    if (PyArray_Check(arr) && PyArray_Check(mask) && PyArray_Check(value))
    {
        PyArrayObject * inValues = (PyArrayObject *)value;

        if (PyArray_TYPE(mask) == NPY_BOOL)
        {
            int64_t itemSizeOut = PyArray_ITEMSIZE(arr);
            int64_t itemSizeIn = PyArray_ITEMSIZE(inValues);

            // check for strides... ?
            int64_t arrayLength = ArrayLength(arr);
            if (arrayLength == ArrayLength(mask) && itemSizeOut == PyArray_STRIDE(arr, 0))
            {
                int64_t valLength = ArrayLength(inValues);

                if (arrayLength == valLength)
                {
                    int outDType = PyArray_TYPE(arr);
                    int inDType = PyArray_TYPE(inValues);
                    MASK_CONVERT_SAFE maskSafe = GetConversionPutMask(inDType, outDType);

                    if (maskSafe)
                    {
                        // MT callback
                        struct MASK_CALLBACK_STRUCT
                        {
                            MASK_CONVERT_SAFE maskSafe;
                            char * pIn;
                            char * pOut;
                            int64_t itemSizeOut;
                            int64_t itemSizeIn;
                            int8_t * pMask;
                            void * pBadInput1;
                            void * pBadOutput1;
                        };

                        MASK_CALLBACK_STRUCT stMask;

                        // This is the routine that will be called back from multiple
                        // threads
                        auto lambdaMaskCallback = [](void * callbackArgT, int core, int64_t start, int64_t length) -> bool
                        {
                            MASK_CALLBACK_STRUCT * callbackArg = (MASK_CALLBACK_STRUCT *)callbackArgT;

                            // printf("[%d] Mask %lld %lld\n", core, start, length);
                            // maskSafe(pIn, pOut, (int8_t*)pMask, length, pBadInput1,
                            // pBadOutput1);
                            // Auto adjust pointers
                            callbackArg->maskSafe(callbackArg->pIn + (start * callbackArg->itemSizeIn),
                                                  callbackArg->pOut + (start * callbackArg->itemSizeOut),
                                                  callbackArg->pMask + start, length, callbackArg->pBadInput1,
                                                  callbackArg->pBadOutput1);

                            return true;
                        };

                        stMask.itemSizeIn = itemSizeIn;
                        stMask.itemSizeOut = itemSizeOut;
                        stMask.pBadInput1 = GetDefaultForType(inDType);
                        stMask.pBadOutput1 = GetDefaultForType(outDType);

                        stMask.pIn = (char *)PyArray_BYTES(inValues);
                        stMask.pOut = (char *)PyArray_BYTES(arr);
                        stMask.pMask = (int8_t *)PyArray_BYTES(mask);
                        stMask.maskSafe = maskSafe;

                        g_cMathWorker->DoMultiThreadedChunkWork(arrayLength, lambdaMaskCallback, &stMask);

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

//-----------------------
// NOTE: Not completed
// Idea is to horizontally retrieve rows (from multiple columns) and to make a
// numpy array
//
// apply rows
// Arg1: List of numpy arrays
// Arg2: Optional final dtype
// Arg3: Function to call
// Arg4 +: args to pass
PyObject * ApplyRows(PyObject * self, PyObject * args, PyObject * kwargs)
{
    Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

    if (argTupleSize < 2)
    {
        PyErr_Format(PyExc_ValueError, "ApplyRows requires two args instead of %llu args", argTupleSize);
        return NULL;
    }

    PyObject * arg1 = PyTuple_GetItem(args, 2);

    // Check if callable
    if (! PyCallable_Check(arg1))
    {
        PyTypeObject * type = (PyTypeObject *)PyObject_Type(arg1);

        PyErr_Format(PyExc_ValueError, "Argument must be a function or a method not %s\n", type->tp_name);
        return NULL;
    }

    PyFunctionObject * function = GetFunctionObject(arg1);

    if (function)
    {
        PyObject * inList1 = PyTuple_GetItem(args, 0);
        PyObject * dtypeObject = PyTuple_GetItem(args, 1);

        int64_t totalItemSize = 0;
        int64_t tupleSize = 0;

        // Do not allow jagged rows
        ArrayInfo * aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, true);

        if (aInfo)
        {
            int32_t dtype = (int32_t)PyLong_AsLong(dtypeObject);

            if (dtype != -1)
            {
                if (dtype < 0 || dtype > NPY_LONGDOUBLE)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "Dont know how to convert dtype num %d.  Please make "
                                 "sure all arrays are ints or floats.",
                                 dtype);
                    return NULL;
                }
            }
            else
            {
                dtype = GetUpcastDtype(aInfo, tupleSize);
            }

            // Now convert?
            // if output is boolean, bad means false
            void * pBadOutput1 = GetDefaultForType(dtype);

            // Convert any different types... build a new list...
            // Figure out which arrays will be recast

            FreeArrayInfo(aInfo);
        }
    }

    return NULL;
}
