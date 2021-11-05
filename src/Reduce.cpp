#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "Reduce.h"
#include "Convert.h"
#include "missing_values.h"
#include "platform_detect.h"

#include <algorithm>

#if defined(__clang__)
    #pragma clang diagnostic ignored "-Wmissing-braces"
    #pragma clang diagnostic ignored "-Wunused-function"
#endif

using namespace riptide;

//#define LOGGING printf
//#define LOGGING OutputDebugStringA
#define LOGGING(...)

#if ! RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED
    // MSVC compiler by default assumed unaligned loads
    #define LOADU(X) *(X)
    #define STOREU(X, Y) *(X) = Y
#else
// gcc compiler assume aligned loads, and usually it is unaligned
static inline __m256d LOADU(__m256d * x)
{
    return _mm256_loadu_pd((double const *)x);
};
static inline __m256 LOADU(__m256 * x)
{
    return _mm256_loadu_ps((float const *)x);
};
static inline __m256i LOADU(__m256i * x)
{
    return _mm256_loadu_si256((__m256i const *)x);
};

static inline void STOREU(__m256d * x, __m256d y)
{
    _mm256_storeu_pd((double *)x, y);
}
static inline void STOREU(__m256 * x, __m256 y)
{
    _mm256_storeu_ps((float *)x, y);
}
static inline void STOREU(__m256i * x, __m256i y)
{
    _mm256_storeu_si256((__m256i *)x, y);
}
#endif

static inline __m256i MIN_OP(int8_t z, __m256i x, __m256i y)
{
    return _mm256_min_epi8(x, y);
}
static inline __m256i MIN_OP(uint8_t z, __m256i x, __m256i y)
{
    return _mm256_min_epu8(x, y);
}
static inline __m256i MIN_OP(uint16_t z, __m256i x, __m256i y)
{
    return _mm256_min_epu16(x, y);
}
static inline __m256i MIN_OP(int16_t z, __m256i x, __m256i y)
{
    return _mm256_min_epi16(x, y);
}
static inline __m256i MIN_OP(int32_t z, __m256i x, __m256i y)
{
    return _mm256_min_epi32(x, y);
}
static inline __m256i MIN_OP(uint32_t z, __m256i x, __m256i y)
{
    return _mm256_min_epu32(x, y);
}
static inline __m256 MIN_OP(float z, __m256 x, __m256 y)
{
    return _mm256_min_ps(x, y);
}
static inline __m256d MIN_OP(double z, __m256d x, __m256d y)
{
    return _mm256_min_pd(x, y);
}

static inline __m256i MAX_OP(int8_t z, __m256i x, __m256i y)
{
    return _mm256_max_epi8(x, y);
}
static inline __m256i MAX_OP(uint8_t z, __m256i x, __m256i y)
{
    return _mm256_max_epu8(x, y);
}
static inline __m256i MAX_OP(uint16_t z, __m256i x, __m256i y)
{
    return _mm256_max_epu16(x, y);
}
static inline __m256i MAX_OP(int16_t z, __m256i x, __m256i y)
{
    return _mm256_max_epi16(x, y);
}
static inline __m256i MAX_OP(int32_t z, __m256i x, __m256i y)
{
    return _mm256_max_epi32(x, y);
}
static inline __m256i MAX_OP(uint32_t z, __m256i x, __m256i y)
{
    return _mm256_max_epu32(x, y);
}
static inline __m256 MAX_OP(float z, __m256 x, __m256 y)
{
    return _mm256_max_ps(x, y);
}
static inline __m256d MAX_OP(double z, __m256d x, __m256d y)
{
    return _mm256_max_pd(x, y);
}

/**
 * @brief AVX psuedo-intrinsic implementing the C 'fmax' function for a vector
 * of packed flats.
 *
 * @param x
 * @param y
 * @return __m256
 */
static inline __m256 FMAX_OP(float z, __m256 x, __m256 y)
{
    const auto max_result = _mm256_max_ps(x, y);
    const auto unord_cmp_mask = _mm256_cmp_ps(max_result, max_result, _CMP_UNORD_Q);
    return _mm256_blendv_ps(x, y, unord_cmp_mask);
}

/**
 * @brief AVX psuedo-intrinsic implementing the C 'fmax' function for a vector
 * of packed doubles.
 *
 * @param x
 * @param y
 * @return __m256d
 */
static inline __m256d FMAX_OP(double z, __m256d x, __m256d y)
{
    const auto max_result = _mm256_max_pd(x, y);
    const auto unord_cmp_mask = _mm256_cmp_pd(max_result, max_result, _CMP_UNORD_Q);
    return _mm256_blendv_pd(x, y, unord_cmp_mask);
}

/**
 * @brief AVX psuedo-intrinsic implementing the C 'fmin' function for a vector
 * of packed flats.
 *
 * @param x
 * @param y
 * @return __m256
 */
static inline __m256 FMIN_OP(float z, __m256 x, __m256 y)
{
    const auto min_result = _mm256_min_ps(x, y);
    const auto unord_cmp_mask = _mm256_cmp_ps(min_result, min_result, _CMP_UNORD_Q);
    return _mm256_blendv_ps(x, y, unord_cmp_mask);
}

/**
 * @brief AVX psuedo-intrinsic implementing the C 'fmin' function for a vector
 * of packed doubles.
 *
 * @param x
 * @param y
 * @return __m256d
 */
static inline __m256d FMIN_OP(double z, __m256d x, __m256d y)
{
    const auto min_result = _mm256_min_pd(x, y);
    const auto unord_cmp_mask = _mm256_cmp_pd(min_result, min_result, _CMP_UNORD_Q);
    return _mm256_blendv_pd(x, y, unord_cmp_mask);
}

// 128 versions---
static inline __m128i MIN_OP128(int8_t z, __m128i x, __m128i y)
{
    return _mm_min_epi8(x, y);
}
static inline __m128i MIN_OP128(uint8_t z, __m128i x, __m128i y)
{
    return _mm_min_epu8(x, y);
}
static inline __m128i MIN_OP128(uint16_t z, __m128i x, __m128i y)
{
    return _mm_min_epu16(x, y);
}
static inline __m128i MIN_OP128(int16_t z, __m128i x, __m128i y)
{
    return _mm_min_epi16(x, y);
}
static inline __m128i MIN_OP128(int32_t z, __m128i x, __m128i y)
{
    return _mm_min_epi32(x, y);
}
static inline __m128i MIN_OP128(uint32_t z, __m128i x, __m128i y)
{
    return _mm_min_epu32(x, y);
}
static inline __m128 MIN_OP128(float z, __m128 x, __m128 y)
{
    return _mm_min_ps(x, y);
}
static inline __m128d MIN_OP128(double z, __m128d x, __m128d y)
{
    return _mm_min_pd(x, y);
}

static inline __m128i MAX_OP128(int8_t z, __m128i x, __m128i y)
{
    return _mm_max_epi8(x, y);
}
static inline __m128i MAX_OP128(uint8_t z, __m128i x, __m128i y)
{
    return _mm_max_epu8(x, y);
}
static inline __m128i MAX_OP128(uint16_t z, __m128i x, __m128i y)
{
    return _mm_max_epu16(x, y);
}
static inline __m128i MAX_OP128(int16_t z, __m128i x, __m128i y)
{
    return _mm_max_epi16(x, y);
}
static inline __m128i MAX_OP128(int32_t z, __m128i x, __m128i y)
{
    return _mm_max_epi32(x, y);
}
static inline __m128i MAX_OP128(uint32_t z, __m128i x, __m128i y)
{
    return _mm_max_epu32(x, y);
}
static inline __m128 MAX_OP128(float z, __m128 x, __m128 y)
{
    return _mm_max_ps(x, y);
}
static inline __m128d MAX_OP128(double z, __m128d x, __m128d y)
{
    return _mm_max_pd(x, y);
}

static inline __m256i CAST_TO256i(__m256d x)
{
    return _mm256_castpd_si256(x);
}
static inline __m256i CAST_TO256i(__m256 x)
{
    return _mm256_castps_si256(x);
}
static inline __m256i CAST_TO256i(__m256i x)
{
    return x;
}

static inline __m128i CAST_TO128i(__m128d x)
{
    return _mm_castpd_si128(x);
}
static inline __m128i CAST_TO128i(__m128 x)
{
    return _mm_castps_si128(x);
}
static inline __m128i CAST_TO128i(__m128i x)
{
    return x;
}

static inline __m128d CAST_TO128d(__m128d x)
{
    return x;
}
static inline __m128d CAST_TO128d(__m128 x)
{
    return _mm_castps_pd(x);
}
static inline __m128d CAST_TO128d(__m128i x)
{
    return _mm_castsi128_pd(x);
}

static inline __m128d CAST_TO128ME(__m128d x, __m128i y)
{
    return _mm_castsi128_pd(y);
}
static inline __m128 CAST_TO128ME(__m128 x, __m128i y)
{
    return _mm_castsi128_ps(y);
}
static inline __m128i CAST_TO128ME(__m128i x, __m128i y)
{
    return y;
}

static inline __m128d CAST_TO128ME(__m128d x, __m128d y)
{
    return y;
}
static inline __m128 CAST_TO128ME(__m128 x, __m128d y)
{
    return _mm_castpd_ps(y);
}
static inline __m128i CAST_TO128ME(__m128i x, __m128d y)
{
    return _mm_castpd_si128(y);
}

static inline __m128d CAST_256_TO_128_LO(__m256d x)
{
    return _mm_castsi128_pd(_mm256_extractf128_si256(_mm256_castpd_si256(x), 0));
}
static inline __m128 CAST_256_TO_128_LO(__m256 x)
{
    return _mm_castsi128_ps(_mm256_extractf128_si256(_mm256_castps_si256(x), 0));
}
static inline __m128i CAST_256_TO_128_LO(__m256i x)
{
    return _mm256_extractf128_si256(x, 0);
}

static inline __m128d CAST_256_TO_128_HI(__m256d x)
{
    return _mm_castsi128_pd(_mm256_extractf128_si256(_mm256_castpd_si256(x), 1));
}
static inline __m128 CAST_256_TO_128_HI(__m256 x)
{
    return _mm_castsi128_ps(_mm256_extractf128_si256(_mm256_castps_si256(x), 1));
}
static inline __m128i CAST_256_TO_128_HI(__m256i x)
{
    return _mm256_extractf128_si256(x, 1);
}

struct stArgScatterGatherFunc
{
    // numpy input type
    NPY_TYPES inputType;

    // the core (if any) making this calculation
    int32_t core;

    // used for nans, how many non nan values
    int64_t lenOut;

    // holds worst case
    long double resultOut;

    // index location output for argmin/argmax
    int64_t resultOutArgInt64;
};

typedef int64_t (*ARG_SCATTER_GATHER_FUNC)(void * pDataIn, int64_t len, int64_t fixup,
                                           stArgScatterGatherFunc * pstScatterGatherFunc);

//============================================================================================
/**
 * @brief Argmax reduction implementation.
 */
class ReduceArgMax final
{
    //--------------------------------------------------------------------------------------------
    template <typename T>
    static int64_t non_vector(void * pDataIn, int64_t len, int64_t fixup, stArgScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * pStart = pIn;
        T * pEnd;

        pEnd = pIn + len;

        // Always set first item
        T result = *pIn++;
        int64_t resultOutArgInt64 = 0;

        while (pIn < pEnd)
        {
            // get the minimum
            T temp = *pIn;
            if (temp > result)
            {
                result = temp;
                resultOutArgInt64 = pIn - pStart;
            }
            pIn++;
        }

        // Check for previous scattering.  If we are the first one
        if (pstScatterGatherFunc->resultOutArgInt64 == -1)
        {
            *(T *)&pstScatterGatherFunc->resultOut = result;
            pstScatterGatherFunc->resultOutArgInt64 = resultOutArgInt64 + fixup;
        }
        else
        {
            // A previous run used this entry
            if (result > *(T *)&pstScatterGatherFunc->resultOut)
            {
                *(T *)&pstScatterGatherFunc->resultOut = result;
                pstScatterGatherFunc->resultOutArgInt64 = resultOutArgInt64 + fixup;
            }
        }
        pstScatterGatherFunc->lenOut += len;
        return pstScatterGatherFunc->resultOutArgInt64;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ARG_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return non_vector<float>;
        case NPY_DOUBLE:
            return non_vector<double>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;
        case NPY_BOOL:
        case NPY_INT8:
            return non_vector<int8_t>;
        case NPY_INT16:
            return non_vector<int16_t>;
        CASE_NPY_INT32:
            return non_vector<int32_t>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;
        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return non_vector<uint32_t>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;
        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Argmin reduction implementation.
 */
class ReduceArgMin final
{
    //--------------------------------------------------------------------------------------------
    // The result is the index location of the minimum value
    template <typename T>
    static int64_t non_vector(void * pDataIn, int64_t len, int64_t fixup, stArgScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * pStart = pIn;
        T * pEnd;

        pEnd = pIn + len;

        // Always set first item
        T result = *pIn++;
        int64_t resultOutArgInt64 = 0;

        while (pIn < pEnd)
        {
            // get the minimum
            T temp = *pIn;
            if (temp < result)
            {
                result = temp;
                resultOutArgInt64 = pIn - pStart;
            }
            pIn++;
        }

        // Check for previous scattering.  If we are the first one
        if (pstScatterGatherFunc->resultOutArgInt64 == -1)
        {
            *(T *)&pstScatterGatherFunc->resultOut = result;
            pstScatterGatherFunc->resultOutArgInt64 = resultOutArgInt64 + fixup;
        }
        else
        {
            // A previous run used this entry
            if (result < *(T *)&pstScatterGatherFunc->resultOut)
            {
                *(T *)&pstScatterGatherFunc->resultOut = result;
                pstScatterGatherFunc->resultOutArgInt64 = resultOutArgInt64 + fixup;
            }
        }
        pstScatterGatherFunc->lenOut += len;
        return pstScatterGatherFunc->resultOutArgInt64;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ARG_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return non_vector<float>;
        case NPY_DOUBLE:
            return non_vector<double>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;
        case NPY_BOOL:
        case NPY_INT8:
            return non_vector<int8_t>;
        case NPY_INT16:
            return non_vector<int16_t>;
        CASE_NPY_INT32:
            return non_vector<int32_t>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;
        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return non_vector<uint32_t>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;
        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Nanargmax reduction implementation.
 */
class ReduceNanargmax final
{
    //--------------------------------------------------------------------------------------------
    // The result is the index location of the maximum value disregarding nans
    // If all values are nans, -1 is the location
    // There are no invalids for bool type (do not call this routine)
    // nan floats require using x==x and failing to detect
    template <typename T>
    static int64_t non_vector(void * pDataIn, int64_t len, int64_t fixup, stArgScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        const T * const pStart = pIn;
        const T * const pEnd = pIn + len;

        // Default to nothing found
        int64_t resultOutArgInt64 = -1;
        T result = 0;

        // TODO: Seems like we should have special support for T=bool below (so that
        // values are clamped to 0/1,
        //       in case we get a bool array that's a view of an int8/uint8 array).
        //       Or will the compiler implement that automatically for us (by
        //       treating bool as having two physical values -- zero and non-zero)?
        // Search for first non Nan
        while (pIn < pEnd)
        {
            const T temp = *pIn;
            // break on first non nan
            if (invalid_for_type<T>::is_valid(temp))
            {
                result = temp;
                resultOutArgInt64 = pIn - pStart;
                ++pIn;
                break;
            }
            ++pIn;
        }

        while (pIn < pEnd)
        {
            const T temp = *pIn;
            // get the maximum
            if (invalid_for_type<T>::is_valid(temp))
            {
                if (temp > result)
                {
                    result = temp;
                    resultOutArgInt64 = pIn - pStart;
                }
            }
            ++pIn;
        }

        // A previous run used this entry
        if (resultOutArgInt64 != -1)
        {
            // Check for previous scattering.  If we are the first one
            if (pstScatterGatherFunc->resultOutArgInt64 == -1)
            {
                // update with our max location
                *(T *)&pstScatterGatherFunc->resultOut = result;
                pstScatterGatherFunc->resultOutArgInt64 = resultOutArgInt64 + fixup;
            }
            else
            {
                // Compare against previous location
                if (result > *(T *)&pstScatterGatherFunc->resultOut)
                {
                    *(T *)&pstScatterGatherFunc->resultOut = result;
                    pstScatterGatherFunc->resultOutArgInt64 = resultOutArgInt64 + fixup;
                }
            }
        }

        pstScatterGatherFunc->lenOut += len;
        return pstScatterGatherFunc->resultOutArgInt64;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ARG_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return non_vector<float>;
        case NPY_DOUBLE:
            return non_vector<double>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;
        case NPY_BOOL:
        case NPY_INT8:
            return non_vector<int8_t>;
        case NPY_INT16:
            return non_vector<int16_t>;
        CASE_NPY_INT32:
            return non_vector<int32_t>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;
        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return non_vector<uint32_t>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;
        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Nanargmin reduction implementation.
 */
class ReduceNanargmin final
{
    //--------------------------------------------------------------------------------------------
    // The result is the index location of the minimum value disregarding nans
    // If all values are nans, -1 is the location
    // There are no invalids for bool type (do not call this routine)
    // nan floats require using x==x and failing to detect
    template <typename T>
    static int64_t non_vector(void * pDataIn, int64_t len, int64_t fixup, stArgScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        const T * const pStart = pIn;
        const T * const pEnd = pIn + len;

        // Default to nothing found
        int64_t resultOutArgInt64 = -1;
        T result = 0;

        // TODO: Seems like we should have special support for T=bool below (so that
        // values are clamped to 0/1,
        //       in case we get a bool array that's a view of an int8/uint8 array).
        // Search for first non Nan
        while (pIn < pEnd)
        {
            const T temp = *pIn;
            // break on first non nan
            if (invalid_for_type<T>::is_valid(temp))
            {
                result = temp;
                resultOutArgInt64 = pIn - pStart;
                ++pIn;
                break;
            }
            ++pIn;
        }

        while (pIn < pEnd)
        {
            const T temp = *pIn;
            // get the minimum
            if (invalid_for_type<T>::is_valid(temp))
            {
                if (temp < result)
                {
                    result = temp;
                    resultOutArgInt64 = pIn - pStart;
                }
            }
            ++pIn;
        }

        // A previous run used this entry
        if (resultOutArgInt64 != -1)
        {
            // Check for previous scattering.  If we are the first one
            if (pstScatterGatherFunc->resultOutArgInt64 == -1)
            {
                // update with our min location
                *(T *)&pstScatterGatherFunc->resultOut = result;
                pstScatterGatherFunc->resultOutArgInt64 = resultOutArgInt64 + fixup;
            }
            else
            {
                // Compare against previous location
                if (result < *(T *)&pstScatterGatherFunc->resultOut)
                {
                    *(T *)&pstScatterGatherFunc->resultOut = result;
                    pstScatterGatherFunc->resultOutArgInt64 = resultOutArgInt64 + fixup;
                }
            }
        }

        pstScatterGatherFunc->lenOut += len;
        return pstScatterGatherFunc->resultOutArgInt64;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ARG_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return non_vector<float>;
        case NPY_DOUBLE:
            return non_vector<double>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;
        case NPY_BOOL:
        case NPY_INT8:
            return non_vector<int8_t>;
        case NPY_INT16:
            return non_vector<int16_t>;
        CASE_NPY_INT32:
            return non_vector<int32_t>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;
        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return non_vector<uint32_t>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;
        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Max (max) reduction implementation.
 */
class ReduceMax final
{
    //--------------------------------------------------------------------------------------------
    template <typename T>
    static double non_vector(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        const T * const pEnd = pIn + len;

        // Always set first item
        T result = *pIn++;

        while (pIn < pEnd)
        {
            // get the maximum
            T temp = *pIn;
            if (temp > result)
            {
                result = temp;
            }
            pIn++;
        }

        // Check for previous scattering.  If we are the first one
        if (pstScatterGatherFunc->lenOut == 0)
        {
            pstScatterGatherFunc->resultOut = (double)result;
            pstScatterGatherFunc->resultOutInt64 = (int64_t)result;
        }
        else
        {
            // in case of nan when calling max (instead of nanmax), preserve nans
            if (pstScatterGatherFunc->resultOut == pstScatterGatherFunc->resultOut)
            {
                pstScatterGatherFunc->resultOut = MAXF(pstScatterGatherFunc->resultOut, (double)result);
            }

            T previous = (T)(pstScatterGatherFunc->resultOutInt64);
            pstScatterGatherFunc->resultOutInt64 = (int64_t)(MAXF(previous, result));
        }
        pstScatterGatherFunc->lenOut += len;
        return (double)pstScatterGatherFunc->resultOutInt64;
    }

    //--------------------------------------------------------------------------------------------
    template <typename T, typename U256, typename U128>
    static double avx2(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * pEnd;

        // Always set first item
        T result = *pIn;

        // For float32 this is 64 /8 since sizeof(float) = 4
        // For float64 this is 32/ 4
        // for int16 this is 128 / 16
        // for int8  this is 256 / 32
        static constexpr int64_t chunkSize = (sizeof(U256) * 8) / sizeof(T);
        static constexpr int64_t perReg = sizeof(U256) / sizeof(T);

        if (len >= chunkSize)
        {
            pEnd = &pIn[chunkSize * (len / chunkSize)];

            U256 * pIn256 = (U256 *)pIn;
            U256 * pEnd256 = (U256 *)pEnd;

            // Use 256 bit registers which hold 8 Ts
            U256 m0 = LOADU(pIn256++);
            U256 m1 = LOADU(pIn256++);
            U256 m2 = LOADU(pIn256++);
            U256 m3 = LOADU(pIn256++);
            U256 m4 = LOADU(pIn256++);
            U256 m5 = LOADU(pIn256++);
            U256 m6 = LOADU(pIn256++);
            U256 m7 = LOADU(pIn256++);

            while (pIn256 < pEnd256)
            {
                m0 = MAX_OP(result, m0, LOADU(pIn256));
                m1 = MAX_OP(result, m1, LOADU(pIn256 + 1));
                m2 = MAX_OP(result, m2, LOADU(pIn256 + 2));
                m3 = MAX_OP(result, m3, LOADU(pIn256 + 3));
                m4 = MAX_OP(result, m4, LOADU(pIn256 + 4));
                m5 = MAX_OP(result, m5, LOADU(pIn256 + 5));
                m6 = MAX_OP(result, m6, LOADU(pIn256 + 6));
                m7 = MAX_OP(result, m7, LOADU(pIn256 + 7));
                pIn256 += 8;
            }

            // Wind this calculation down from 8 256 bit registers to the data T
            // MAX_OP
            m0 = MAX_OP(result, m0, m1);
            m2 = MAX_OP(result, m2, m3);
            m4 = MAX_OP(result, m4, m5);
            m6 = MAX_OP(result, m6, m7);
            m0 = MAX_OP(result, m0, m2);
            m4 = MAX_OP(result, m4, m6);
            m0 = MAX_OP(result, m0, m4);

            // Write 256 bits into memory
            __m256i temp;
            _mm256_storeu_si256(&temp, *(__m256i *)&m0);

            T * tempT = (T *)&temp;
            result = tempT[0];

            for (int i = 1; i < perReg; i++)
            {
                result = MAXF(result, tempT[i]);
            }
            // update pIn to last location we read
            pIn = (T *)pIn256;
        }

        pEnd = &pIn[len & (chunkSize - 1)];
        while (pIn < pEnd)
        {
            // get the minimum
            result = MAXF(result, *pIn);
            pIn++;
        }

        // Check for previous scattering.  If we are the first one
        if (pstScatterGatherFunc->lenOut == 0)
        {
            pstScatterGatherFunc->resultOut = (double)result;
            pstScatterGatherFunc->resultOutInt64 = (int64_t)result;
        }
        else
        {
            pstScatterGatherFunc->resultOut = MAXF(pstScatterGatherFunc->resultOut, (double)result);
            T previous = (T)(pstScatterGatherFunc->resultOutInt64);
            pstScatterGatherFunc->resultOutInt64 = (int64_t)(MAXF(previous, result));
        }
        pstScatterGatherFunc->lenOut += len;
        return pstScatterGatherFunc->resultOut;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return avx2<float, __m256, __m128>;
        case NPY_DOUBLE:
            return avx2<double, __m256d, __m128d>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;
        case NPY_BOOL:
        case NPY_INT8:
            return avx2<int8_t, __m256i, __m128i>;
        case NPY_INT16:
            return avx2<int16_t, __m256i, __m128i>;
        CASE_NPY_INT32:
            return avx2<int32_t, __m256i, __m128i>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;
        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return avx2<uint32_t, __m256i, __m128i>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;
        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Min (min) reduction implementation.
 */
class ReduceMin final
{
    //--------------------------------------------------------------------------------------------
    template <typename T>
    static double non_vector(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        const T * const pEnd = pIn + len;

        // Always set first item
        T result = *pIn++;

        while (pIn < pEnd)
        {
            // get the minimum
            T temp = *pIn;
            if (temp < result)
            {
                result = temp;
            }
            pIn++;
        }

        // Check for previous scattering.  If we are the first one
        if (pstScatterGatherFunc->lenOut == 0)
        {
            pstScatterGatherFunc->resultOut = (double)result;
            pstScatterGatherFunc->resultOutInt64 = (int64_t)result;
        }
        else
        {
            // in case of nan when calling min (instead of nanmin), preserve nans
            if (pstScatterGatherFunc->resultOut == pstScatterGatherFunc->resultOut)
            {
                pstScatterGatherFunc->resultOut = MINF(pstScatterGatherFunc->resultOut, (double)result);
            }
            T previous = (T)(pstScatterGatherFunc->resultOutInt64);
            pstScatterGatherFunc->resultOutInt64 = (int64_t)(MINF(previous, result));
        }
        pstScatterGatherFunc->lenOut += len;
        return (double)pstScatterGatherFunc->resultOutInt64;
    }

    //--------------------------------------------------------------------------------------------
    template <typename T, typename U256, typename U128>
    static double avx2(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * pEnd;

        // Always set first item
        T result = *pIn;

        // For float32 this is 64 /8 since sizeof(float) = 4
        // For float64 this is 32/ 4
        // for int16 this is 128 / 16
        // for int8  this is 256 / 32
        const int64_t chunkSize = (sizeof(U256) * 8) / sizeof(T);
        const int64_t perReg = sizeof(U256) / sizeof(T);

        // printf("hit this %lld %lld\n", len, chunkSize);
        if (len >= chunkSize)
        {
            pEnd = &pIn[chunkSize * (len / chunkSize)];

            U256 * pIn256 = (U256 *)pIn;
            U256 * pEnd256 = (U256 *)pEnd;

            // Use 256 bit registers which hold 8 Ts
            U256 m0 = LOADU(pIn256++);
            U256 m1 = LOADU(pIn256++);
            U256 m2 = LOADU(pIn256++);
            U256 m3 = LOADU(pIn256++);
            U256 m4 = LOADU(pIn256++);
            U256 m5 = LOADU(pIn256++);
            U256 m6 = LOADU(pIn256++);
            U256 m7 = LOADU(pIn256++);

            while (pIn256 < pEnd256)
            {
                m0 = MIN_OP(result, m0, LOADU(pIn256));
                m1 = MIN_OP(result, m1, LOADU(pIn256 + 1));
                m2 = MIN_OP(result, m2, LOADU(pIn256 + 2));
                m3 = MIN_OP(result, m3, LOADU(pIn256 + 3));
                m4 = MIN_OP(result, m4, LOADU(pIn256 + 4));
                m5 = MIN_OP(result, m5, LOADU(pIn256 + 5));
                m6 = MIN_OP(result, m6, LOADU(pIn256 + 6));
                m7 = MIN_OP(result, m7, LOADU(pIn256 + 7));
                pIn256 += 8;
            }

            // Wind this calculation down from 8 256 bit registers to the data T
            // MIN_OP
            m0 = MIN_OP(result, m0, m1);
            m2 = MIN_OP(result, m2, m3);
            m4 = MIN_OP(result, m4, m5);
            m6 = MIN_OP(result, m6, m7);
            m0 = MIN_OP(result, m0, m2);
            m4 = MIN_OP(result, m4, m6);
            m0 = MIN_OP(result, m0, m4);

            if (false)
            {
                // Older path
                // Write 256 bits into memory
                __m256i temp;
                _mm256_storeu_si256(&temp, *(__m256i *)&m0);
                T * tempT = (T *)&temp;
                result = tempT[0];

                // printf("sofar minop 0  chunk: %lld    %lf %lf %lld\n", chunkSize,
                // pstScatterGatherFunc->resultOut, (double)result,
                // pstScatterGatherFunc->resultOutInt64);
                for (int i = 1; i < perReg; i++)
                {
                    result = MINF(result, tempT[i]);
                }
            }
            else
            {
                // go from 256bit to 128bit to 64bit
                // split the single 256bit register into two 128bit registers
                U128 ym0 = CAST_256_TO_128_LO(m0);
                U128 ym1 = CAST_256_TO_128_HI(m0);
                __m128i temp;

                // move the min to ym0
                ym0 = MIN_OP128(result, ym0, ym1);

                // move the min to lower half (64 bits)
                ym1 = CAST_TO128ME(ym1, _mm_shuffle_pd(CAST_TO128d(ym0), CAST_TO128d(ym1), 1));
                ym0 = MIN_OP128(result, ym0, ym1);

                // Write 128 bits into memory (although only need 64bits)
                _mm_storeu_si128(&temp, CAST_TO128i(ym0));
                T * tempT = (T *)&temp;
                result = tempT[0];

                // printf("sofar minop 0  chunk: %lld    %lf %lf %lld\n", chunkSize,
                // pstScatterGatherFunc->resultOut, (double)result,
                // pstScatterGatherFunc->resultOutInt64);
                for (int i = 1; i < (perReg / 4); i++)
                {
                    result = MINF(result, tempT[i]);
                }
            }

            // update pIn to last location we read
            pIn = (T *)pIn256;
        }

        pEnd = &pIn[len & (chunkSize - 1)];
        while (pIn < pEnd)
        {
            // get the minimum
            result = MINF(result, *pIn);
            pIn++;
        }

        // Check for previous scattering.  If we are the first one
        if (pstScatterGatherFunc->lenOut == 0)
        {
            pstScatterGatherFunc->resultOut = (double)result;
            pstScatterGatherFunc->resultOutInt64 = (int64_t)result;
            // printf("minop 0  chunk: %lld    %lf %lf %lld\n", chunkSize,
            // pstScatterGatherFunc->resultOut, (double)result,
            // pstScatterGatherFunc->resultOutInt64);
        }
        else
        {
            // printf("minop !0 %lf %lf\n", pstScatterGatherFunc->resultOut,
            // (double)result);
            pstScatterGatherFunc->resultOut = MINF(pstScatterGatherFunc->resultOut, (double)result);
            T previous = (T)(pstScatterGatherFunc->resultOutInt64);
            pstScatterGatherFunc->resultOutInt64 = (int64_t)(MINF(previous, result));
        }
        pstScatterGatherFunc->lenOut += len;
        return pstScatterGatherFunc->resultOut;
    }

    //--------------------------------------------------------------------------------------------
    // This routine only support floats
    template <typename T, typename U256, typename U128>
    static double avx2_nan_aware(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * pEnd;

        // Always set first item
        T result = *pIn;

        // For float32 this is 64 /8 since sizeof(float) = 4
        // For float64 this is 32/ 4
        const int64_t chunkSize = (sizeof(U256) * 8) / sizeof(T);
        const int64_t perReg = sizeof(U256) / sizeof(T);

        // printf("hit this %lld %lld\n", len, chunkSize);
        if (len >= chunkSize)
        {
            pEnd = &pIn[chunkSize * (len / chunkSize)];

            U256 * pIn256 = (U256 *)pIn;
            U256 * pEnd256 = (U256 *)pEnd;

            // Use 256 bit registers which hold 8 Ts
            U256 m0 = LOADU(pIn256++);
            U256 m1 = LOADU(pIn256++);

            // Stagger the loads, while m1 loads, should be able to calculate mnan
            U256 mnan = _mm256_cmp_ps(m0, m0, _CMP_EQ_OQ);
            U256 m2 = LOADU(pIn256++);
            mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m1, m1, _CMP_EQ_OQ));
            U256 m3 = LOADU(pIn256++);
            mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m2, m2, _CMP_EQ_OQ));
            U256 m4 = LOADU(pIn256++);
            mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m3, m3, _CMP_EQ_OQ));
            U256 m5 = LOADU(pIn256++);
            mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m4, m4, _CMP_EQ_OQ));
            U256 m6 = LOADU(pIn256++);
            mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m5, m5, _CMP_EQ_OQ));
            U256 m7 = LOADU(pIn256++);
            mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m6, m6, _CMP_EQ_OQ));
            mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m7, m7, _CMP_EQ_OQ));

            while (pIn256 < pEnd256)
            {
                U256 m10 = LOADU(pIn256++);
                m0 = MIN_OP(result, m0, m10);
                mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m10, m10, _CMP_EQ_OQ));
                U256 m11 = LOADU(pIn256++);
                m1 = MIN_OP(result, m1, m11);
                mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m11, m11, _CMP_EQ_OQ));

                m10 = LOADU(pIn256++);
                m2 = MIN_OP(result, m2, m10);
                mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m10, m10, _CMP_EQ_OQ));
                m11 = LOADU(pIn256++);
                m3 = MIN_OP(result, m3, m11);
                mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m11, m11, _CMP_EQ_OQ));

                m10 = LOADU(pIn256++);
                m4 = MIN_OP(result, m4, m10);
                mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m10, m10, _CMP_EQ_OQ));
                m11 = LOADU(pIn256++);
                m5 = MIN_OP(result, m5, m11);
                mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m11, m11, _CMP_EQ_OQ));

                m10 = LOADU(pIn256++);
                m6 = MIN_OP(result, m6, m10);
                mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m10, m10, _CMP_EQ_OQ));
                m11 = LOADU(pIn256++);
                m7 = MIN_OP(result, m7, m11);
                mnan = _mm256_and_ps(mnan, _mm256_cmp_ps(m11, m11, _CMP_EQ_OQ));
            }

            // TODO: Check mnan for nans
            // NOTE: a nan check can go here for floating point and short circuit
            if (_mm256_movemask_ps(mnan) == 255)
            {
                // Wind this calculation down from 8 256 bit registers to the data T
                // MIN_OP
                m0 = MIN_OP(result, m0, m1);
                m2 = MIN_OP(result, m2, m3);
                m4 = MIN_OP(result, m4, m5);
                m6 = MIN_OP(result, m6, m7);
                m0 = MIN_OP(result, m0, m2);
                m4 = MIN_OP(result, m4, m6);
                m0 = MIN_OP(result, m0, m4);

                if (false)
                {
                    // Older path
                    // Write 256 bits into memory
                    __m256i temp;
                    //_mm256_movedup_pd
                    //_mm256_shuffle_pd(y0, y0, 0);
                    _mm256_storeu_si256(&temp, *(__m256i *)&m0);
                    T * tempT = (T *)&temp;
                    result = tempT[0];

                    // printf("sofar minop 0  chunk: %lld    %lf %lf %lld\n", chunkSize,
                    // pstScatterGatherFunc->resultOut, (double)result,
                    // pstScatterGatherFunc->resultOutInt64);
                    for (int i = 1; i < perReg; i++)
                    {
                        result = MINF(result, tempT[i]);
                    }
                }
                else
                {
                    // go from 256bit to 128bit to 64bit
                    // split the single 256bit register into two 128bit registers
                    U128 ym0 = CAST_256_TO_128_LO(m0);
                    U128 ym1 = CAST_256_TO_128_HI(m0);
                    __m128i temp;

                    // move the min to ym0
                    ym0 = MIN_OP128(result, ym0, ym1);

                    // move the min to lower half (64 bits)
                    ym1 = CAST_TO128ME(ym1, _mm_shuffle_pd(CAST_TO128d(ym0), CAST_TO128d(ym1), 1));
                    ym0 = MIN_OP128(result, ym0, ym1);

                    // Write 128 bits into memory (although only need 64bits)
                    _mm_storeu_si128(&temp, CAST_TO128i(ym0));
                    T * tempT = (T *)&temp;
                    result = tempT[0];

                    // printf("sofar minop 0  chunk: %lld    %lf %lf %lld\n", chunkSize,
                    // pstScatterGatherFunc->resultOut, (double)result,
                    // pstScatterGatherFunc->resultOutInt64);
                    for (int i = 1; i < (perReg / 4); i++)
                    {
                        result = MINF(result, tempT[i]);
                    }
                }

                // update pIn to last location we read
                pIn = (T *)pIn256;
            }
            else
            {
                // this should be cached somewhere
                result = std::numeric_limits<float>::quiet_NaN();
            }
        }

        pEnd = &pIn[len & (chunkSize - 1)];
        while (pIn < pEnd)
        {
            // get the minimum
            // TODO: Check for nans also
            T value = *pIn++;
            if (value == value)
            {
                result = MINF(result, value);
            }
            else
            {
                // this should be cached somewhere
                result = std::numeric_limits<float>::quiet_NaN();
                break;
            }
        }

        // Check for previous scattering.  If we are the first one
        if (pstScatterGatherFunc->lenOut == 0)
        {
            pstScatterGatherFunc->resultOut = (double)result;
            pstScatterGatherFunc->resultOutInt64 = (int64_t)result;
            // printf("minop 0  chunk: %lld    %lf %lf %lld\n", chunkSize,
            // pstScatterGatherFunc->resultOut, (double)result,
            // pstScatterGatherFunc->resultOutInt64);
        }
        else
        {
            // printf("minop !0 %lf %lf\n", pstScatterGatherFunc->resultOut,
            // (double)result);
            if (result == result)
            {
                pstScatterGatherFunc->resultOut = MINF(pstScatterGatherFunc->resultOut, (double)result);
                T previous = (T)(pstScatterGatherFunc->resultOutInt64);
                pstScatterGatherFunc->resultOutInt64 = (int64_t)(MINF(previous, result));
            }
            else
            {
                // we know this if a float
                pstScatterGatherFunc->resultOut = result;
            }
        }
        pstScatterGatherFunc->lenOut += len;
        return pstScatterGatherFunc->resultOut;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    template <bool is_nan_aware = false>
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            // For testing/development purposes, we have a "nan-aware" (i.e.
            // nan-propagating) vectorized implementation for floats. Use the template
            // parameter to select that version if specified.
            if /*constexpr*/ (is_nan_aware)
            {
                return avx2_nan_aware<float, __m256, __m128>;
            }
            else
            {
                return avx2<float, __m256, __m128>;
            }

        case NPY_DOUBLE:
            return avx2<double, __m256d, __m128d>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;
        case NPY_BOOL:
        case NPY_INT8:
            return avx2<int8_t, __m256i, __m128i>;
        case NPY_INT16:
            return avx2<int16_t, __m256i, __m128i>;
        CASE_NPY_INT32:
            return avx2<int32_t, __m256i, __m128i>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;
        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return avx2<uint32_t, __m256i, __m128i>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;
        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Nanmin (fmin) reduction implementation.
 */
class ReduceNanMin final
{
    // Simple, non-vectorized implementation of the nanmin (fmin) reduction
    // operation.
    template <typename T>
    static double non_vector(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * const pEnd = pIn + len;

        // Initialize the result to the NaN/invalid value for this type,
        // so it'll be overwritten by any non-NaN value encountered below.
        T result = invalid_for_type<T>::value;

        // Search for first non Nan
        while (pIn < pEnd)
        {
            // break on first non nan
            if (invalid_for_type<T>::is_valid(*pIn))
            {
                break;
            }
            pIn++;
        }

        if (pIn < pEnd)
        {
            result = *pIn++;

            while (pIn < pEnd)
            {
                // get the minimum
                // include inf because of range function in cut/qcut
                if (invalid_for_type<T>::is_valid(*pIn))
                {
                    result = (std::min)(result, *pIn);
                }
                pIn++;
            }
        }

        // Only store the result -- or reduce with an existing intermediate result
        // from another chunk in this same array -- if the result was a non-NaN
        // value. This is so that if the array consists of only NaNs, the
        // stScatterGatherFunc value used to accumulate any intermediate results
        // will retain whatever values it was initialized with, which makes it
        // easier to detect and handle the case of an all-NaN array.
        if (invalid_for_type<T>::is_valid(result))
        {
            // Is there a previous intermediate result we need to combine with?
            // If not, this chunk of the array was the first result so we can just
            // store the results.
            if (pstScatterGatherFunc->lenOut == 0)
            {
                pstScatterGatherFunc->resultOut = (double)result;
                pstScatterGatherFunc->resultOutInt64 = (int64_t)result;

                // Set 'lenOut' to 1. This field nominally stores the total number of
                // non-NaN values found in the array while performing the calculation;
                // for the purposes of this reduction, it only matters that we detect
                // the case of an all-NaN array, and for that it's enough to set this to
                // any positive value.
                pstScatterGatherFunc->lenOut = 1;
            }
            else
            {
                pstScatterGatherFunc->resultOut = (std::min)(pstScatterGatherFunc->resultOut, (double)result);

                T previous = (T)(pstScatterGatherFunc->resultOutInt64);
                pstScatterGatherFunc->resultOutInt64 = (int64_t)((std::min)(previous, result));
            }

            return (double)pstScatterGatherFunc->resultOutInt64;
        }
        else
        {
            // This chunk of the array contained all NaNs.
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // AVX2-based implementation of the nanmin (fmin) reduction operation.
    template <typename T>
    static double avx2(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        // TEMP: Call the non-vectorized version of this operation until a
        // vectorized version can be implemented.
        return non_vector<T>(pDataIn, len, pstScatterGatherFunc);
    }

    //--------------------------------------------------------------------------------------------
    template <typename T, typename U256>
    static double avx2(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * pEnd;

        // Initialize the result to the NaN/invalid value for this type,
        // so it'll be overwritten by any non-NaN value encountered below.
        T result = invalid_for_type<T>::value;

        // For float32 this is 64 /8 since sizeof(float) = 4
        // For float64 this is 32/ 4
        // for int16 this is 128 / 16
        // for int8  this is 256 / 32
        static constexpr int64_t chunkSize = (sizeof(U256) * 8) / sizeof(T);
        static constexpr int64_t perReg = sizeof(U256) / sizeof(T);

        // printf("hit this %lld %lld\n", len, chunkSize);
        if (len >= chunkSize)
        {
            pEnd = &pIn[chunkSize * (len / chunkSize)];

            U256 * pIn256 = (U256 *)pIn;
            U256 * pEnd256 = (U256 *)pEnd;

            // Use 256 bit registers which hold 8 Ts
            U256 m0 = LOADU(pIn256++);
            U256 m1 = LOADU(pIn256++);
            U256 m2 = LOADU(pIn256++);
            U256 m3 = LOADU(pIn256++);
            U256 m4 = LOADU(pIn256++);
            U256 m5 = LOADU(pIn256++);
            U256 m6 = LOADU(pIn256++);
            U256 m7 = LOADU(pIn256++);

            while (pIn256 < pEnd256)
            {
                m0 = FMIN_OP(result, m0, LOADU(pIn256));
                m1 = FMIN_OP(result, m1, LOADU(pIn256 + 1));
                m2 = FMIN_OP(result, m2, LOADU(pIn256 + 2));
                m3 = FMIN_OP(result, m3, LOADU(pIn256 + 3));
                m4 = FMIN_OP(result, m4, LOADU(pIn256 + 4));
                m5 = FMIN_OP(result, m5, LOADU(pIn256 + 5));
                m6 = FMIN_OP(result, m6, LOADU(pIn256 + 6));
                m7 = FMIN_OP(result, m7, LOADU(pIn256 + 7));
                pIn256 += 8;
            }

            // Wind this calculation down from 8 256 bit registers to the data T
            // FMIN_OP
            m0 = FMIN_OP(result, m0, m1);
            m2 = FMIN_OP(result, m2, m3);
            m4 = FMIN_OP(result, m4, m5);
            m6 = FMIN_OP(result, m6, m7);
            m0 = FMIN_OP(result, m0, m2);
            m4 = FMIN_OP(result, m4, m6);
            m0 = FMIN_OP(result, m0, m4);

            // Write 256 bits into memory
            // PERF: Replace this memory spill with a vector reduction (e.g. using
            // unpackhi/unpacklo)
            __m256i temp;
            _mm256_storeu_si256(&temp, *(__m256i *)&m0);

            T * tempT = (T *)&temp;
            result = tempT[0];

            // printf("sofar minop 0  chunk: %lld    %lf %lf %lld\n", chunkSize,
            // pstScatterGatherFunc->resultOut, (double)result,
            // pstScatterGatherFunc->resultOutInt64);

            for (int i = 1; i < perReg; i++)
            {
                const T current = tempT[i];
                if (invalid_for_type<T>::is_valid(current))
                {
                    result = (std::min)(result, current);
                }
            }

            // update pIn to last location we read
            pIn = (T *)pIn256;
        }

        pEnd = &pIn[len & (chunkSize - 1)];
        while (pIn < pEnd)
        {
            // get the minimum
            if (invalid_for_type<T>::is_valid(*pIn))
            {
                result = (std::min)(result, *pIn);
            }
            pIn++;
        }

        // Only store the result -- or reduce with an existing intermediate result
        // from another chunk in this same array -- if the result was a non-NaN
        // value. This is so that if the array consists of only NaNs, the
        // stScatterGatherFunc value used to accumulate any intermediate results
        // will retain whatever values it was initialized with, which makes it
        // easier to detect and handle the case of an all-NaN array.
        if (invalid_for_type<T>::is_valid(result))
        {
            // Is there a previous intermediate result we need to combine with?
            // If not, this chunk of the array was the first result so we can just
            // store the results.
            if (pstScatterGatherFunc->lenOut == 0)
            {
                pstScatterGatherFunc->resultOut = (double)result;
                pstScatterGatherFunc->resultOutInt64 = (int64_t)result;

                // Set 'lenOut' to 1. This field nominally stores the total number of
                // non-NaN values found in the array while performing the calculation;
                // for the purposes of this reduction, it only matters that we detect
                // the case of an all-NaN array, and for that it's enough to set this to
                // any positive value.
                pstScatterGatherFunc->lenOut = 1;
            }
            else
            {
                pstScatterGatherFunc->resultOut = (std::min)(pstScatterGatherFunc->resultOut, (double)result);

                T previous = (T)(pstScatterGatherFunc->resultOutInt64);
                pstScatterGatherFunc->resultOutInt64 = (int64_t)((std::min)(previous, result));
            }

            return (double)pstScatterGatherFunc->resultOutInt64;
        }
        else
        {
            // This chunk of the array contained all NaNs.
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        // TODO: Enable avx2 implementation once it's been tested.
        case NPY_FLOAT:
            return non_vector<float>;
        case NPY_DOUBLE:
            return non_vector<double>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;
        case NPY_INT8:
            return non_vector<int8_t>;
        case NPY_INT16:
            return non_vector<int16_t>;
        CASE_NPY_INT32:
            return non_vector<int32_t>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;
        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return non_vector<uint32_t>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;

        // bools are currently handled specially; we don't consider bool to have a
        // nan/invalid value so we utilize the normal reduction operation for it.
        case NPY_BOOL:
            return ReduceMin::GetScatterGatherFuncPtr(inputType);

        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Nanmax (fmax) reduction implementation.
 */
class ReduceNanMax final
{
    // Simple, non-vectorized implementation of the nanmax (fmax) reduction
    // operation.
    template <typename T>
    static double non_vector(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * const pEnd = pIn + len;

        // Initialize the result to the NaN/invalid value for this type,
        // so it'll be overwritten by any non-NaN value encountered below.
        T result = invalid_for_type<T>::value;

        // Search for first non Nan
        while (pIn < pEnd)
        {
            // break on first non nan
            if (invalid_for_type<T>::is_valid(*pIn))
            {
                break;
            }
            pIn++;
        }

        if (pIn < pEnd)
        {
            result = *pIn++;

            while (pIn < pEnd)
            {
                // get the maximum
                // include inf because of range function in cut/qcut
                if (invalid_for_type<T>::is_valid(*pIn))
                {
                    result = (std::max)(result, *pIn);
                }
                pIn++;
            }
        }

        // Only store the result -- or reduce with an existing intermediate result
        // from another chunk in this same array -- if the result was a non-NaN
        // value. This is so that if the array consists of only NaNs, the
        // stScatterGatherFunc value used to accumulate any intermediate results
        // will retain whatever values it was initialized with, which makes it
        // easier to detect and handle the case of an all-NaN array.
        if (invalid_for_type<T>::is_valid(result))
        {
            // Is there a previous intermediate result we need to combine with?
            // If not, this chunk of the array was the first result so we can just
            // store the results.
            if (pstScatterGatherFunc->lenOut == 0)
            {
                pstScatterGatherFunc->resultOut = (double)result;
                pstScatterGatherFunc->resultOutInt64 = (int64_t)result;

                // Set 'lenOut' to 1. This field nominally stores the total number of
                // non-NaN values found in the array while performing the calculation;
                // for the purposes of this reduction, it only matters that we detect
                // the case of an all-NaN array, and for that it's enough to set this to
                // any positive value.
                pstScatterGatherFunc->lenOut = 1;
            }
            else
            {
                pstScatterGatherFunc->resultOut = (std::max)(pstScatterGatherFunc->resultOut, (double)result);

                T previous = (T)(pstScatterGatherFunc->resultOutInt64);
                pstScatterGatherFunc->resultOutInt64 = (int64_t)((std::max)(previous, result));
            }

            return (double)pstScatterGatherFunc->resultOutInt64;
        }
        else
        {
            // This chunk of the array contained all NaNs.
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // AVX2-based implementation of the nanmax (fmax) reduction operation.
    template <typename T>
    static double avx2(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        // TEMP: Call the non-vectorized version of this operation until a
        // vectorized version can be implemented.
        return non_vector<T>(pDataIn, len, pstScatterGatherFunc);
    }

    //--------------------------------------------------------------------------------------------
    template <typename T, typename U256>
    static double avx2(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * pEnd;

        // Initialize the result to the NaN/invalid value for this type,
        // so it'll be overwritten by any non-NaN value encountered below.
        T result = invalid_for_type<T>::value;

        // For float32 this is 64 /8 since sizeof(float) = 4
        // For float64 this is 32/ 4
        // for int16 this is 128 / 16
        // for int8  this is 256 / 32
        static constexpr int64_t chunkSize = (sizeof(U256) * 8) / sizeof(T);
        static constexpr int64_t perReg = sizeof(U256) / sizeof(T);

        // printf("hit this %lld %lld\n", len, chunkSize);
        if (len >= chunkSize)
        {
            pEnd = &pIn[chunkSize * (len / chunkSize)];

            U256 * pIn256 = (U256 *)pIn;
            U256 * pEnd256 = (U256 *)pEnd;

            // Use 256 bit registers which hold 8 Ts
            U256 m0 = LOADU(pIn256++);
            U256 m1 = LOADU(pIn256++);
            U256 m2 = LOADU(pIn256++);
            U256 m3 = LOADU(pIn256++);
            U256 m4 = LOADU(pIn256++);
            U256 m5 = LOADU(pIn256++);
            U256 m6 = LOADU(pIn256++);
            U256 m7 = LOADU(pIn256++);

            while (pIn256 < pEnd256)
            {
                m0 = FMAX_OP(result, m0, LOADU(pIn256));
                m1 = FMAX_OP(result, m1, LOADU(pIn256 + 1));
                m2 = FMAX_OP(result, m2, LOADU(pIn256 + 2));
                m3 = FMAX_OP(result, m3, LOADU(pIn256 + 3));
                m4 = FMAX_OP(result, m4, LOADU(pIn256 + 4));
                m5 = FMAX_OP(result, m5, LOADU(pIn256 + 5));
                m6 = FMAX_OP(result, m6, LOADU(pIn256 + 6));
                m7 = FMAX_OP(result, m7, LOADU(pIn256 + 7));
                pIn256 += 8;
            }

            // Wind this calculation down from 8 256 bit registers to the data T
            // FMAX_OP
            m0 = FMAX_OP(result, m0, m1);
            m2 = FMAX_OP(result, m2, m3);
            m4 = FMAX_OP(result, m4, m5);
            m6 = FMAX_OP(result, m6, m7);
            m0 = FMAX_OP(result, m0, m2);
            m4 = FMAX_OP(result, m4, m6);
            m0 = FMAX_OP(result, m0, m4);

            // Write 256 bits into memory
            // PERF: Replace this memory spill with a vector reduction (e.g. using
            // unpackhi/unpacklo)
            __m256i temp;
            _mm256_storeu_si256(&temp, *(__m256i *)&m0);

            T * tempT = (T *)&temp;
            result = tempT[0];

            // printf("sofar maxop 0  chunk: %lld    %lf %lf %lld\n", chunkSize,
            // pstScatterGatherFunc->resultOut, (double)result,
            // pstScatterGatherFunc->resultOutInt64);

            for (int i = 1; i < perReg; i++)
            {
                const T current = tempT[i];
                if (invalid_for_type<T>::is_valid(current))
                {
                    result = (std::max)(result, current);
                }
            }

            // update pIn to last location we read
            pIn = (T *)pIn256;
        }

        pEnd = &pIn[len & (chunkSize - 1)];
        while (pIn < pEnd)
        {
            // get the maximum
            if (invalid_for_type<T>::is_valid(*pIn))
            {
                result = (std::max)(result, *pIn);
            }
            pIn++;
        }

        // Only store the result -- or reduce with an existing intermediate result
        // from another chunk in this same array -- if the result was a non-NaN
        // value. This is so that if the array consists of only NaNs, the
        // stScatterGatherFunc value used to accumulate any intermediate results
        // will retain whatever values it was initialized with, which makes it
        // easier to detect and handle the case of an all-NaN array.
        if (invalid_for_type<T>::is_valid(result))
        {
            // Is there a previous intermediate result we need to combine with?
            // If not, this chunk of the array was the first result so we can just
            // store the results.
            if (pstScatterGatherFunc->lenOut == 0)
            {
                pstScatterGatherFunc->resultOut = (double)result;
                pstScatterGatherFunc->resultOutInt64 = (int64_t)result;

                // Set 'lenOut' to 1. This field nominally stores the total number of
                // non-NaN values found in the array while performing the calculation;
                // for the purposes of this reduction, it only matters that we detect
                // the case of an all-NaN array, and for that it's enough to set this to
                // any positive value.
                pstScatterGatherFunc->lenOut = 1;
            }
            else
            {
                pstScatterGatherFunc->resultOut = (std::max)(pstScatterGatherFunc->resultOut, (double)result);

                T previous = (T)(pstScatterGatherFunc->resultOutInt64);
                pstScatterGatherFunc->resultOutInt64 = (int64_t)((std::max)(previous, result));
            }

            return (double)pstScatterGatherFunc->resultOutInt64;
        }
        else
        {
            // This chunk of the array contained all NaNs.
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        // TODO: Enable avx2 implementation once it's been tested.
        case NPY_FLOAT:
            return non_vector<float>;
        case NPY_DOUBLE:
            return non_vector<double>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;
        case NPY_INT8:
            return non_vector<int8_t>;
        case NPY_INT16:
            return non_vector<int16_t>;
        CASE_NPY_INT32:
            return non_vector<int32_t>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;
        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return non_vector<uint32_t>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;

        // bools are currently handled specially; we don't consider bool to have a
        // nan/invalid value so we utilize the normal reduction operation for it.
        case NPY_BOOL:
            return ReduceMax::GetScatterGatherFuncPtr(inputType);

        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Sum reduction implementation.
 */
class ReduceSum final
{
    //--------------------------------------------------------------------------------------------
    template <typename T>
    static double ReduceAddSlow(void * pDataIn, int64_t length, stScatterGatherFunc * pstScatterGatherFunc)
    {
        T * pIn = (T *)pDataIn;
        T * pEnd;
        double result = 0;

        if (pstScatterGatherFunc->inputType == NPY_BOOL)
        {
            result = (double)SumBooleanMask((int8_t *)pDataIn, length);
        }
        else
        {
            pEnd = pIn + length;

            // Always set first item
            result = (double)(*pIn++);

            while (pIn < pEnd)
            {
                // get the minimum
                result += (double)(*pIn++);
            }
        }

        pstScatterGatherFunc->lenOut += length;
        // printf("float adding %lf to %lf\n", pstScatterGatherFunc->resultOut,
        // result);
        pstScatterGatherFunc->resultOut += result;
        pstScatterGatherFunc->resultOutInt64 += (int64_t)result;
        return result;
    }

    //=============================================================================================
    static double ReduceAddF32(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double result = 0;
        float * pIn = (float *)pDataIn;
        float * pEnd;
        if (len >= 32)
        {
            pEnd = &pIn[32 * (len / 32)];

            // Use 256 bit registers which hold 8 floats
            __m256d m0 = _mm256_setzero_pd();
            __m256d m1 = _mm256_setzero_pd();
            __m256d m2 = _mm256_setzero_pd();
            __m256d m3 = _mm256_setzero_pd();
            __m256d m4 = _mm256_setzero_pd();
            __m256d m5 = _mm256_setzero_pd();
            __m256d m6 = _mm256_setzero_pd();
            __m256d m7 = _mm256_setzero_pd();

            while (pIn < pEnd)
            {
                m0 = _mm256_add_pd(m0, _mm256_cvtps_pd(_mm_loadu_ps(pIn)));
                m1 = _mm256_add_pd(m1, _mm256_cvtps_pd(_mm_loadu_ps(pIn + 4)));
                m2 = _mm256_add_pd(m2, _mm256_cvtps_pd(_mm_loadu_ps(pIn + 8)));
                m3 = _mm256_add_pd(m3, _mm256_cvtps_pd(_mm_loadu_ps(pIn + 12)));
                m4 = _mm256_add_pd(m4, _mm256_cvtps_pd(_mm_loadu_ps(pIn + 16)));
                m5 = _mm256_add_pd(m5, _mm256_cvtps_pd(_mm_loadu_ps(pIn + 20)));
                m6 = _mm256_add_pd(m6, _mm256_cvtps_pd(_mm_loadu_ps(pIn + 24)));
                m7 = _mm256_add_pd(m7, _mm256_cvtps_pd(_mm_loadu_ps(pIn + 28)));
                pIn += 32;
            }
            m0 = _mm256_add_pd(m0, m1);
            m2 = _mm256_add_pd(m2, m3);
            m4 = _mm256_add_pd(m4, m5);
            m6 = _mm256_add_pd(m6, m7);
            m0 = _mm256_add_pd(m0, m2);
            m4 = _mm256_add_pd(m4, m6);
            m0 = _mm256_add_pd(m0, m4);

            double tempDouble[4];

            _mm256_storeu_pd(tempDouble, m0);
            result = tempDouble[0];
            result += tempDouble[1];
            result += tempDouble[2];
            result += tempDouble[3];
        }

        pEnd = &pIn[len & 31];

        while (pIn < pEnd)
        {
            result += (double)(*pIn++);
        }
        pstScatterGatherFunc->lenOut += len;
        // printf("float adding %lf to %lf\n", pstScatterGatherFunc->resultOut,
        // result);
        pstScatterGatherFunc->resultOut += result;
        pstScatterGatherFunc->resultOutInt64 += (int64_t)result;
        return result;
    }

    //--------------------------------------------------------------------------------------------
    static double ReduceAddI32(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double result = 0;
        int32_t * pIn = (int32_t *)pDataIn;
        int32_t * pEnd;
        if (len >= 32)
        {
            pEnd = &pIn[32 * (len / 32)];

            // Use 256 bit registers which hold 8 floats
            __m256d m0 = _mm256_setzero_pd();
            __m256d m1 = _mm256_setzero_pd();
            __m256d m2 = _mm256_setzero_pd();
            __m256d m3 = _mm256_setzero_pd();
            __m256d m4 = _mm256_setzero_pd();
            __m256d m5 = _mm256_setzero_pd();
            __m256d m6 = _mm256_setzero_pd();
            __m256d m7 = _mm256_setzero_pd();

            while (pIn < pEnd)
            {
                m0 = _mm256_add_pd(m0, _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(pIn))));
                m1 = _mm256_add_pd(m1, _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(pIn + 4))));
                m2 = _mm256_add_pd(m2, _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(pIn + 8))));
                m3 = _mm256_add_pd(m3, _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(pIn + 12))));
                m4 = _mm256_add_pd(m4, _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(pIn + 16))));
                m5 = _mm256_add_pd(m5, _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(pIn + 20))));
                m6 = _mm256_add_pd(m6, _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(pIn + 24))));
                m7 = _mm256_add_pd(m7, _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(pIn + 28))));
                pIn += 32;
            }
            m0 = _mm256_add_pd(m0, m1);
            m2 = _mm256_add_pd(m2, m3);
            m4 = _mm256_add_pd(m4, m5);
            m6 = _mm256_add_pd(m6, m7);
            m0 = _mm256_add_pd(m0, m2);
            m4 = _mm256_add_pd(m4, m6);
            m0 = _mm256_add_pd(m0, m4);

            double tempDouble[4];

            _mm256_storeu_pd(tempDouble, m0);
            result = (double)tempDouble[0];
            result += (double)tempDouble[1];
            result += (double)tempDouble[2];
            result += (double)tempDouble[3];
        }

        pEnd = &pIn[len & 31];

        while (pIn < pEnd)
        {
            result += (double)(*pIn++);
        }
        pstScatterGatherFunc->lenOut += len;
        pstScatterGatherFunc->resultOut += result;
        pstScatterGatherFunc->resultOutInt64 += (int64_t)result;

        return result;
    }

    //--------------------------------------------------------------------------------------------
    static double ReduceAddD64(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double result = 0;
        double * pIn = (double *)pDataIn;
        double * pEnd;
        if (len >= 32)
        {
            pEnd = &pIn[32 * (len / 32)];

            // Use 256 bit registers which hold 8 floats
            __m256d m0 = _mm256_setzero_pd();
            __m256d m1 = _mm256_setzero_pd();
            __m256d m2 = _mm256_setzero_pd();
            __m256d m3 = _mm256_setzero_pd();
            __m256d m4 = _mm256_setzero_pd();
            __m256d m5 = _mm256_setzero_pd();
            __m256d m6 = _mm256_setzero_pd();
            __m256d m7 = _mm256_setzero_pd();

            while (pIn < pEnd)
            {
                m0 = _mm256_add_pd(m0, _mm256_loadu_pd(pIn));
                m1 = _mm256_add_pd(m1, _mm256_loadu_pd(pIn + 4));
                m2 = _mm256_add_pd(m2, _mm256_loadu_pd(pIn + 8));
                m3 = _mm256_add_pd(m3, _mm256_loadu_pd(pIn + 12));
                m4 = _mm256_add_pd(m4, _mm256_loadu_pd(pIn + 16));
                m5 = _mm256_add_pd(m5, _mm256_loadu_pd(pIn + 20));
                m6 = _mm256_add_pd(m6, _mm256_loadu_pd(pIn + 24));
                m7 = _mm256_add_pd(m7, _mm256_loadu_pd(pIn + 28));
                pIn += 32;
            }
            m0 = _mm256_add_pd(m0, m1);
            m2 = _mm256_add_pd(m2, m3);
            m4 = _mm256_add_pd(m4, m5);
            m6 = _mm256_add_pd(m6, m7);
            m0 = _mm256_add_pd(m0, m2);
            m4 = _mm256_add_pd(m4, m6);
            m0 = _mm256_add_pd(m0, m4);

            double tempDouble[4];

            _mm256_storeu_pd(tempDouble, m0);
            result = tempDouble[0];
            result += tempDouble[1];
            result += tempDouble[2];
            result += tempDouble[3];
        }

        pEnd = &pIn[len & 31];

        while (pIn < pEnd)
        {
            result += *pIn++;
        }
        // printf("double adding %lf to %lf\n", pstScatterGatherFunc->resultOut,
        // result);
        pstScatterGatherFunc->lenOut += len;
        pstScatterGatherFunc->resultOut += result;
        pstScatterGatherFunc->resultOutInt64 += (int64_t)result;
        return result;
    }

    //--------------------------------------------------------
    // TODO: Make this a template
    /*
    static double ReduceAddI64(void* pDataIn, int64_t len, stScatterGatherFunc*
    pstScatterGatherFunc) { double result = 0; int64_t* pIn = (int64_t*)pDataIn;
       int64_t* pEnd;
       if (len >= 32) {
          pEnd = &pIn[32 * (len / 32)];

          // Use 256 bit registers which hold 8 floats
          __m256i m0 = _mm256_set1_epi64x(0L);
          __m256i m1 = _mm256_set1_epi64x(0);
          __m256i m2 = _mm256_set1_epi64x(0);
          __m256i m3 = _mm256_set1_epi64x(0);
          __m256i m4 = _mm256_set1_epi64x(0);
          __m256i m5 = _mm256_set1_epi64x(0);
          __m256i m6 = _mm256_set1_epi64x(0);
          __m256i m7 = _mm256_set1_epi64x(0);

          while (pIn < pEnd) {

             m0 = _mm256_add_epi64(m0, *(__m256i*)(pIn));
             m1 = _mm256_add_epi64(m1, *(__m256i*)(pIn + 4));
             m2 = _mm256_add_epi64(m2, *(__m256i*)(pIn + 8));
             m3 = _mm256_add_epi64(m3, *(__m256i*)(pIn + 12));
             m4 = _mm256_add_epi64(m4, *(__m256i*)(pIn + 16));
             m5 = _mm256_add_epi64(m5, *(__m256i*)(pIn + 20));
             m6 = _mm256_add_epi64(m6, *(__m256i*)(pIn + 24));
             m7 = _mm256_add_epi64(m7, *(__m256i*)(pIn + 28));
             pIn += 32;
          }
          m0 = _mm256_add_epi64(m0, m1);
          m2 = _mm256_add_epi64(m2, m3);
          m4 = _mm256_add_epi64(m4, m5);
          m6 = _mm256_add_epi64(m6, m7);
          m0 = _mm256_add_epi64(m0, m2);
          m4 = _mm256_add_epi64(m4, m6);
          m0 = _mm256_add_epi64(m0, m4);

          __m256i temp;

          _mm256_store_si256(&temp, m0);

          int64_t* pTemp = (int64_t*)&temp;
          result = (double)pTemp[0];
          result += (double)pTemp[1];
          result += (double)pTemp[2];
          result += (double)pTemp[3];
       }

       pEnd = &pIn[len & 31];

       while (pIn < pEnd) {
          result += (double)(*pIn++);
       }
       pstScatterGatherFunc->lenOut += len;
       pstScatterGatherFunc->resultOut += result;
       pstScatterGatherFunc->resultOutInt64 += (int64_t)result;
       //printf("reduceadd 1 sum %lld\n", pstScatterGatherFunc->resultOutInt64);
       //printf("reduceadd 1 sum %lf\n", pstScatterGatherFunc->resultOut);
       return result;

    }
    */

    // Always returns a double which is U
    template <typename Input, typename Output>
    static double ReduceAdd(void * pDataIn, int64_t length, stScatterGatherFunc * pstScatterGatherFunc)
    {
        Output result = 0;
        Input * pIn = (Input *)pDataIn;
        Input * pEnd = &pIn[length];

        while (pIn < pEnd)
        {
            result += *pIn++;
        }

        // printf("reduceadd sum %lld\n", (int64_t)result);
        pstScatterGatherFunc->lenOut += length;
        pstScatterGatherFunc->resultOut += result;
        pstScatterGatherFunc->resultOutInt64 += (int64_t)result;
        return result;
    }

    // Always returns a double
    // Double calculation in case of overflow
    static double ReduceAddI64(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double resultd = 0;
        int64_t result = 0;
        int64_t * pIn = (int64_t *)pDataIn;
        int64_t * pEnd = &pIn[len];

        if (len >= 4)
        {
            __m128i m0 = _mm_loadu_si128((__m128i *)pIn);
            __m128d f0 = _mm_cvtsi64_sd(_mm_set1_pd(0.0), pIn[0]);
            __m128d f1 = _mm_cvtsi64_sd(f0, pIn[1]);
            f0 = _mm_unpacklo_pd(f0, f1);
            pIn += 2;
            do
            {
                m0 = _mm_add_epi64(m0, _mm_loadu_si128((__m128i *)pIn));

                __m128d f2 = _mm_cvtsi64_sd(f1, pIn[0]);
                f1 = _mm_cvtsi64_sd(f1, pIn[1]);
                f0 = _mm_add_pd(f0, _mm_unpacklo_pd(f2, f1));

                pIn += 2;
            }
            while (pIn < (pEnd - 1));

            // Add horizontally
            __m128i result128;
            _mm_storeu_si128(&result128, m0);
            int64_t * pResults = (int64_t *)&result128;
            // collect the 2
            result = pResults[0] + pResults[1];

            __m128d result128d;
            _mm_storeu_pd((double *)&result128d, f0);
            double * pResultsd = (double *)&result128d;
            // collect the 2
            resultd = pResultsd[0] + pResultsd[1];
        }

        while (pIn < pEnd)
        {
            // Calculate sum for integer and float
            const auto temp = *pIn;
            result += temp;
            resultd += (double)temp;
            ++pIn;
        }
        pstScatterGatherFunc->lenOut += len;
        pstScatterGatherFunc->resultOut += resultd;
        pstScatterGatherFunc->resultOutInt64 += result;
        return resultd;
    }

    // Always returns a double
    // Double calculation in case of overflow

    static double ReduceAddU64(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double resultd = 0;
        uint64_t result = 0;
        uint64_t * pIn = (uint64_t *)pDataIn;
        uint64_t * pEnd = &pIn[len];

        while (pIn < pEnd)
        {
            // Calculate sum for integer and float
            result += (uint64_t)*pIn;
            resultd += *pIn;
            ++pIn;
        }

        pstScatterGatherFunc->lenOut += len;
        pstScatterGatherFunc->resultOut += resultd;
        *(uint64_t *)&pstScatterGatherFunc->resultOutInt64 += result;
        return resultd;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return ReduceAddF32;
        case NPY_DOUBLE:
            return ReduceAddD64;
        case NPY_LONGDOUBLE:
            return ReduceAddSlow<long double>;

        case NPY_BOOL: // TODO: Call/return our fast SumBooleanMask()
                       // implementation.
        case NPY_INT8:
            return ReduceAddSlow<int8_t>;
        case NPY_INT16:
            return ReduceAddSlow<int16_t>;
        CASE_NPY_INT32:
            return ReduceAddI32;
        CASE_NPY_INT64:

            return ReduceAddI64;

        case NPY_UINT8:
            return ReduceAddSlow<uint8_t>;
        case NPY_UINT16:
            return ReduceAddSlow<uint16_t>;
        CASE_NPY_UINT32:
            return ReduceAddSlow<uint32_t>;
        CASE_NPY_UINT64:

            return ReduceAddU64;

        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Nansum reduction implementation.
 */
class ReduceNanSum final
{
    // Non-vectorized implementation of the nanmax (fmax) reduction.
    template <typename T>
    static double non_vector(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double result = 0;
        int64_t count = 0;
        T * pIn = (T *)pDataIn;

        for (int64_t i = 0; i < len; i++)
        {
            if (invalid_for_type<T>::is_valid(pIn[i]))
            {
                result += pIn[i];
                count += 1;
            }
        }
        pstScatterGatherFunc->lenOut += count;
        pstScatterGatherFunc->resultOut += result;
        pstScatterGatherFunc->resultOutInt64 += (int64_t)result;
        return result;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return non_vector<float>;
        case NPY_DOUBLE:
            return non_vector<double>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;

        // TODO: For booleans, use the optimized BooleanCount() function.
        case NPY_BOOL:
            return non_vector<bool>;
        case NPY_INT8:
            return non_vector<int8_t>;
        case NPY_INT16:
            return non_vector<int16_t>;
        CASE_NPY_INT32:
            return non_vector<int32_t>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;

        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return non_vector<uint32_t>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;

        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Variance (var) reduction implementation.
 */
class ReduceVariance final
{
    //-------------------------------------------------
    // Routine does not work on Linux: to be looked at
    template <typename T>
    static double ReduceVar_TODO(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double mean = pstScatterGatherFunc->meanCalculation;
        double result = 0;

        T * pIn = (T *)pDataIn;

        // Check if we can crunch 8 doubles at a time
        // THE CODE below appears to be buggy
        if (len >= 4)
        {
            __m256d m0;
            __m256d m1;
            __m256d m2 = _mm256_setzero_pd();
            __m256d m3 = _mm256_set1_pd(mean);

            int64_t len2 = len & ~3;

            for (int64_t i = 0; i < len2; i += 4)
            {
                // Load 8 floats or 4 doubles
                // TODO: Use C++ 'if constexpr' here; if T = typeof(double), we can load
                // the first vector with _mm256_loadu_pd();
                //       If T = typeof(float), load+convert the data with
                //       _mm256_cvtps_pd(_mm_loadu_pd(static_cast<double*>(&pIn[i])))
                //       This avoids using _mm256_set_pd() which is somewhat slower.
                m0 = _mm256_sub_pd(_mm256_set_pd((double)pIn[i], (double)pIn[i + 1], (double)pIn[i + 2], (double)pIn[i + 3]), m3);

                // square the diff of mean
                m1 = _mm256_mul_pd(m0, m0);

                // add them up
                m2 = _mm256_add_pd(m2, m1);
            }

            // m2 = _mm256_add_pd(m2, m5);

            double tempDouble[4];

            _mm256_storeu_pd(tempDouble, m2);
            result = tempDouble[0];
            result += tempDouble[1];
            result += tempDouble[2];
            result += tempDouble[3];
        }

        for (int64_t i = 0; i < (len & 3); i++)
        {
            double temp = (double)pIn[i] - mean;
            result += (temp * temp);
        }

        pstScatterGatherFunc->lenOut += len;
        pstScatterGatherFunc->resultOut += result;
        pstScatterGatherFunc->resultOutInt64 += (int64_t)result;
        return pstScatterGatherFunc->resultOut;
    }

    //----------------------------------
    // Multithreaded scatter gather calculation for variance core
    template <typename T>
    static double ReduceVar(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double mean = (double)pstScatterGatherFunc->meanCalculation;
        double result = 0;

        T * pIn = (T *)pDataIn;

        for (int64_t i = 0; i < len; i++)
        {
            double temp = (double)pIn[i] - mean;
            result += (temp * temp);
        }
        pstScatterGatherFunc->lenOut += len;
        pstScatterGatherFunc->resultOut += result;
        pstScatterGatherFunc->resultOutInt64 += (int64_t)result;
        return pstScatterGatherFunc->resultOut;
    }

    //--------------------------------------------------------------------------------------------
    static double ReduceVarF32(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double mean = (double)pstScatterGatherFunc->meanCalculation;
        double result = 0;

        float * pIn = (float *)pDataIn;
        float * pEnd;

        // Check if we can crunch 8 doubles at a time
        if (len >= 4)
        {
            pEnd = &pIn[4 * (len / 4)];

            __m256d m0;
            __m256d m1;
            __m256d m2 = _mm256_setzero_pd();
            __m256d m3 = _mm256_set1_pd(mean);

            while (pIn < pEnd)
            {
                // Load 8 floats or 4 doubles
                m0 = _mm256_sub_pd(_mm256_cvtps_pd(*(__m128 *)pIn), m3);

                // square the diff of mean
                m1 = _mm256_mul_pd(m0, m0);

                // add them up in m2
                m2 = _mm256_add_pd(m2, m1);
                pIn += 4;
            }

            // m2 = _mm256_add_pd(m2, m5);

            double tempDouble[4];

            _mm256_storeu_pd(tempDouble, m2);
            result = (double)tempDouble[0];
            result += (double)tempDouble[1];
            result += (double)tempDouble[2];
            result += (double)tempDouble[3];
        }

        pEnd = &pIn[len & 3];

        while (pIn < pEnd)
        {
            double temp = (double)(*pIn++) - mean;
            result += (temp * temp);
        }

        pstScatterGatherFunc->lenOut += len;
        pstScatterGatherFunc->resultOut += result;
        return pstScatterGatherFunc->resultOut;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return ReduceVarF32;
        case NPY_DOUBLE:
            return ReduceVar<double>;
        case NPY_LONGDOUBLE:
            return ReduceVar<long double>;

        case NPY_BOOL:
        case NPY_INT8:
            return ReduceVar<int8_t>;
        case NPY_INT16:
            return ReduceVar<int16_t>;
        CASE_NPY_INT32:
            return ReduceVar<int32_t>;
        CASE_NPY_INT64:

            return ReduceVar<int64_t>;

        case NPY_UINT8:
            return ReduceVar<uint8_t>;
        case NPY_UINT16:
            return ReduceVar<uint16_t>;
        CASE_NPY_UINT32:
            return ReduceVar<uint32_t>;
        CASE_NPY_UINT64:

            return ReduceVar<uint64_t>;

        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Nan-ignoring variance (nanvar) reduction implementation.
 */
class ReduceNanVariance final
{
    // Multithread-able scatter gather calculation for NAN variance core
    template <typename T>
    static double non_vector(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        double mean = (double)pstScatterGatherFunc->meanCalculation;
        double result = 0;
        int64_t count = 0;

        const T * const pIn = (T *)pDataIn;

        for (int64_t i = 0; i < len; i++)
        {
            if (invalid_for_type<T>::is_valid(pIn[i]))
            {
                double temp = (double)pIn[i] - mean;
                result += (temp * temp);
                count += 1;
            }
        }

        pstScatterGatherFunc->lenOut += count;
        pstScatterGatherFunc->resultOut += result;
        pstScatterGatherFunc->resultOutInt64 += (int64_t)result;
        return pstScatterGatherFunc->resultOut;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return non_vector<float>;
        case NPY_DOUBLE:
            return non_vector<double>;
        case NPY_LONGDOUBLE:
            return non_vector<long double>;

        case NPY_BOOL:
            return non_vector<bool>;
        case NPY_INT8:
            return non_vector<int8_t>;
        case NPY_INT16:
            return non_vector<int16_t>;
        CASE_NPY_INT32:
            return non_vector<int32_t>;
        CASE_NPY_INT64:

            return non_vector<int64_t>;

        case NPY_UINT8:
            return non_vector<uint8_t>;
        case NPY_UINT16:
            return non_vector<uint16_t>;
        CASE_NPY_UINT32:
            return non_vector<uint32_t>;
        CASE_NPY_UINT64:

            return non_vector<uint64_t>;

        default:
            return nullptr;
        }
    }
};

//============================================================================================
/**
 * @brief Arithmetic mean/average reduction implementation.
 */
class ReduceMean final
{
    // Call SUM and then divide
    static double wrapper(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        ReduceSum::GetScatterGatherFuncPtr((NPY_TYPES)pstScatterGatherFunc->inputType)(pDataIn, len, pstScatterGatherFunc);
        // printf("reduce mean dividing %lf by %lf\n",
        // pstScatterGatherFunc->resultOut, (double)pstScatterGatherFunc->lenOut);

        if (pstScatterGatherFunc->lenOut > 1)
        {
            pstScatterGatherFunc->resultOut = pstScatterGatherFunc->resultOut / (double)pstScatterGatherFunc->lenOut;
        }
        else
        {
            pstScatterGatherFunc->resultOut = NAN;
        }
        return pstScatterGatherFunc->resultOut;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        return wrapper;
    }
};

//============================================================================================
/**
 * @brief Nan-ignoring arithmetic mean/average (nanmean) reduction
 * implementation.
 */
class ReduceNanMean final
{
    static double wrapper(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        ReduceNanSum::GetScatterGatherFuncPtr((NPY_TYPES)pstScatterGatherFunc->inputType)(pDataIn, len, pstScatterGatherFunc);
        // printf("Dviding %lf by %lf\n", pstScatterGatherFunc->resultOut,
        // (double)pstScatterGatherFunc->lenOut);
        if (pstScatterGatherFunc->lenOut > 1)
        {
            pstScatterGatherFunc->resultOut = pstScatterGatherFunc->resultOut / (double)pstScatterGatherFunc->lenOut;
        }
        else
        {
            pstScatterGatherFunc->resultOut = NAN;
        }

        return pstScatterGatherFunc->resultOut;
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        return wrapper;
    }
};

//============================================================================================
/**
 * @brief Standard deviation reduction implementation.
 */
class ReduceStdDev final
{
    static double wrapper(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        const double variance = ReduceVariance::GetScatterGatherFuncPtr((NPY_TYPES)pstScatterGatherFunc->inputType)(
            pDataIn, len, pstScatterGatherFunc);
        return sqrt(variance);
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        return wrapper;
    }
};

//============================================================================================
/**
 * @brief Nan-ignoring standard deviation reduction implementation.
 */
class ReduceNanStdDev final
{
    static double wrapper(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc)
    {
        const double variance = ReduceNanVariance::GetScatterGatherFuncPtr((NPY_TYPES)pstScatterGatherFunc->inputType)(
            pDataIn, len, pstScatterGatherFunc);
        return sqrt(variance);
    }

public:
    // Given the numpy dtype number of the input array, returns a pointer to the
    // reduction function specialized for that array type.
    static ANY_SCATTER_GATHER_FUNC GetScatterGatherFuncPtr(const NPY_TYPES inputType)
    {
        return wrapper;
    }
};

//============================================================
// Called by arg min/max which want the array index of where min/max occurs
//
static ARG_SCATTER_GATHER_FUNC GetArgReduceFuncPtr(const NPY_TYPES inputType, const REDUCE_FUNCTIONS func)
{
    switch (func)
    {
    case REDUCE_FUNCTIONS::REDUCE_ARGMIN:
        return ReduceArgMin::GetScatterGatherFuncPtr(inputType);

    case REDUCE_FUNCTIONS::REDUCE_ARGMAX:
        return ReduceArgMax::GetScatterGatherFuncPtr(inputType);

    case REDUCE_FUNCTIONS::REDUCE_NANARGMIN:
        return ReduceNanargmin::GetScatterGatherFuncPtr(inputType);

    case REDUCE_FUNCTIONS::REDUCE_NANARGMAX:
        return ReduceNanargmax::GetScatterGatherFuncPtr(inputType);

    default:
        // Unknown/unsupported function requested; return nullptr.
        return nullptr;
    }
}

//============================================================
//
//
static ANY_SCATTER_GATHER_FUNC GetReduceFuncPtr(const NPY_TYPES inputType, const REDUCE_FUNCTIONS func)
{
    switch (func)
    {
    case REDUCE_FUNCTIONS::REDUCE_SUM:
        return ReduceSum::GetScatterGatherFuncPtr(inputType);
    case REDUCE_FUNCTIONS::REDUCE_NANSUM:
        return ReduceNanSum::GetScatterGatherFuncPtr(inputType);

    case REDUCE_FUNCTIONS::REDUCE_MIN:
        return ReduceMin::GetScatterGatherFuncPtr(inputType);
    case REDUCE_FUNCTIONS::REDUCE_MIN_NANAWARE:
        return ReduceMin::GetScatterGatherFuncPtr<true>(inputType);
    case REDUCE_FUNCTIONS::REDUCE_NANMIN:
        return ReduceNanMin::GetScatterGatherFuncPtr(inputType);

    case REDUCE_FUNCTIONS::REDUCE_MAX:
        return ReduceMax::GetScatterGatherFuncPtr(inputType);
    case REDUCE_FUNCTIONS::REDUCE_NANMAX:
        return ReduceNanMax::GetScatterGatherFuncPtr(inputType);

    case REDUCE_FUNCTIONS::REDUCE_VAR:
        return ReduceVariance::GetScatterGatherFuncPtr(inputType);
    case REDUCE_FUNCTIONS::REDUCE_NANVAR:
        return ReduceNanVariance::GetScatterGatherFuncPtr(inputType);

    case REDUCE_FUNCTIONS::REDUCE_MEAN:
        return ReduceMean::GetScatterGatherFuncPtr(inputType);
    case REDUCE_FUNCTIONS::REDUCE_NANMEAN:
        return ReduceNanMean::GetScatterGatherFuncPtr(inputType);

    case REDUCE_FUNCTIONS::REDUCE_STD:
        return ReduceStdDev::GetScatterGatherFuncPtr(inputType);
    case REDUCE_FUNCTIONS::REDUCE_NANSTD:
        return ReduceNanStdDev::GetScatterGatherFuncPtr(inputType);

    default:
        // Unknown/unsupported reduction function requested.
        return nullptr;
    }
}

//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array
// Arg2: function number
// Arg3: ddof (delta degrees of freedom)
// Returns: None on failure, else float
//
static PyObject * ReduceInternal(PyArrayObject * inArr1, REDUCE_FUNCTIONS func, const int64_t ddof = 1)
{
    const NPY_TYPES numpyInType = (NPY_TYPES)ObjectToDtype(inArr1);

    if (numpyInType < 0)
    {
        LOGGING("Reduce: Dont know how to convert these types %d  for func %llu", numpyInType, func);
        // punt to numpy
        Py_INCREF(Py_None);
        return Py_None;
    }

    LOGGING("Reduce: numpy types %d --> %d   %d %d\n", numpyInType, numpyOutType, gNumpyTypeToSize[numpyInType],
            gNumpyTypeToSize[numpyOutType]);

    void * pDataIn = PyArray_BYTES(inArr1);
    int ndim = PyArray_NDIM(inArr1);
    npy_intp * dims = PyArray_DIMS(inArr1);
    int64_t len = CalcArrayLength(ndim, dims);

    if (len == 0)
    {
        // punt to numpy, often raises a ValueError
        Py_INCREF(Py_None);
        return Py_None;
    }

    //------------------------------------------------
    // Handle Arg style functions here
    if (func >= REDUCE_FUNCTIONS::REDUCE_ARGMIN && func <= REDUCE_FUNCTIONS::REDUCE_NANARGMAX)
    {
        stArgScatterGatherFunc sgFunc = { numpyInType, 0, 0, 0, -1 };
        ARG_SCATTER_GATHER_FUNC pFunction = GetArgReduceFuncPtr(numpyInType, func);

        if (pFunction)
        {
            // TODO: Make multithreaded
            pFunction(pDataIn, len, 0, &sgFunc);
        }
        if (sgFunc.resultOutArgInt64 == -1)
        {
            return PyErr_Format(PyExc_ValueError,
                                "There were no valid values to return the index "
                                "location of for argmin or argmax.");
        }
        return PyLong_FromUnsignedLongLong(sgFunc.resultOutArgInt64);
    }

    const ANY_SCATTER_GATHER_FUNC pFunction = GetReduceFuncPtr(numpyInType, func);

    // If we don't know how to handle this, return None to punt to numpy.
    if (! pFunction)
    {
        LOGGING("Reduce: Dont know how to convert these types %d  for func %llu", numpyInType, func);
        // punt to numpy
        Py_RETURN_NONE;
    }

    stScatterGatherFunc sgFunc = { (int32_t)numpyInType, 0, 0, 0, 0, 0 };

    FUNCTION_LIST fl{};
    fl.AnyScatterGatherCall = pFunction;
    fl.FunctionName = "Reduce";
    fl.NumpyOutputType = 0;
    fl.NumpyType = numpyInType;
    fl.InputItemSize = PyArray_ITEMSIZE(inArr1);
    fl.Input1Strides = len > 1 ? PyArray_STRIDE(inArr1, 0) : 0;
    fl.OutputItemSize = 0;
    fl.TypeOfFunctionCall = TYPE_OF_FUNCTION_CALL::ANY_SCATTER_GATHER;

    if (func >= 100 && func < REDUCE_FUNCTIONS::REDUCE_MIN)
    {
        // Get either normal sum or nansum based on last bit being set in funcnumber
        REDUCE_FUNCTIONS newFunc = (REDUCE_FUNCTIONS)(REDUCE_SUM + (func & 1));
        fl.AnyScatterGatherCall = GetReduceFuncPtr(numpyInType, newFunc);

        // First call add
        g_cMathWorker->WorkScatterGatherCall(&fl, pDataIn, len, newFunc, &sgFunc);

        if (sgFunc.lenOut > 0)
        {
            sgFunc.resultOut = sgFunc.resultOut / (double)sgFunc.lenOut;
            sgFunc.resultOutInt64 = (int64_t)sgFunc.resultOut;
        }
        else
        {
            sgFunc.resultOut = 0;
            sgFunc.resultOutInt64 = 0;
        }

        // in case we move onto VAR this needs to be set
        sgFunc.meanCalculation = sgFunc.resultOut;
        LOGGING("Mean calculation %lf  %llu\n", sgFunc.resultOut, sgFunc.lenOut);

        // See if more work to do... next step is VAR
        if (func > REDUCE_FUNCTIONS::REDUCE_NANMEAN)
        {
            sgFunc.resultOut = 0;
            sgFunc.resultOutInt64 = 0;
            sgFunc.lenOut = 0;

            // Get the proper VAR calculation to call
            newFunc = (REDUCE_FUNCTIONS)(REDUCE_FUNCTIONS::REDUCE_VAR + (func & 1));
            fl.AnyScatterGatherCall = GetReduceFuncPtr(numpyInType, newFunc);

            // Calculate VAR
            g_cMathWorker->WorkScatterGatherCall(&fl, pDataIn, len, newFunc, &sgFunc);

            // Final steps to calc VAR
            if (sgFunc.lenOut > ddof)
            {
                sgFunc.resultOut = sgFunc.resultOut / (double)(sgFunc.lenOut - ddof);
                LOGGING("Var calculation %lf  %llu\n", sgFunc.resultOut, sgFunc.lenOut);

                // Check if std vs var.  For std take the sqrt.
                if ((func & ~1) == REDUCE_FUNCTIONS::REDUCE_STD)
                {
                    sgFunc.resultOut = sqrt(sgFunc.resultOut);
                }
            }
            else
            {
                // if not enough deg of freedom, return nan
                sgFunc.resultOut = std::numeric_limits<double>::quiet_NaN();
            }
        }

        return PyFloat_FromDouble(sgFunc.resultOut);
    }
    else
    {
        // SUM or MINMAX
        // !! Work to do for min max

        g_cMathWorker->WorkScatterGatherCall(&fl, pDataIn, len, func, &sgFunc);

        if (func == REDUCE_FUNCTIONS::REDUCE_SUM || func == REDUCE_FUNCTIONS::REDUCE_NANSUM)
        {
            // Check for overflow
            switch (numpyInType)
            {
            CASE_NPY_UINT64:

                if (sgFunc.resultOut > 18446744073709551615.0)
                {
                    LOGGING("Returning overflow %lf  for func %lld\n", sgFunc.resultOut, func);
                    return PyFloat_FromDouble(sgFunc.resultOut);
                }
                break;
            CASE_NPY_INT64:

                if (sgFunc.resultOut > 9223372036854775807.0 || sgFunc.resultOut < -9223372036854775808.0)
                {
                    LOGGING("Returning overflow %lf  for func %lld\n", sgFunc.resultOut, func);
                    return PyFloat_FromDouble(sgFunc.resultOut);
                }
            default:
                break;
            }
        }

        switch (numpyInType)
        {
        CASE_NPY_UINT64:

            LOGGING("Returning %llu  vs  %lf for func %lld\n", (uint64_t)sgFunc.resultOutInt64, sgFunc.resultOut, func);
            return PyLong_FromUnsignedLongLong(sgFunc.resultOutInt64);

        case NPY_FLOAT:
        case NPY_DOUBLE:
        case NPY_LONGDOUBLE:
            // If the function called was a "nan___" reduction function, we need to
            // check whether it found *any* non-NaN values. Otherwise, we'll return
            // the initial value of the reduction result (usually just initialized to
            // zero when the stScatterGatherFunc is created above) which will be
            // wrong.
            switch (func)
            {
            case REDUCE_FUNCTIONS::REDUCE_NANMIN:
            case REDUCE_FUNCTIONS::REDUCE_NANMAX:
                return PyFloat_FromDouble(sgFunc.lenOut > 0 ? sgFunc.resultOut : Py_NAN);

            default:
                return PyFloat_FromDouble(sgFunc.resultOut);
            }

        default:
            LOGGING("Returning %lld  vs  %lf for func %lld\n", sgFunc.resultOutInt64, sgFunc.resultOut, func);
            return PyLong_FromLongLong(sgFunc.resultOutInt64);
        }
    }
}

//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array
// Arg2: function number
// Arg3: the kwargs dict (serach for ddof, keepdims, axis, dtype) or ddof int
// Returns: None if it cannot handle reduction
// otherwise returns a scalar
PyObject * Reduce(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    int64_t tupleSize = Py_SIZE(args);
    int64_t ddof = 1;

    if (tupleSize == 3)
    {
        PyObject * ddofItem = (PyObject *)PyTuple_GET_ITEM(args, 2);
        if (PyLong_Check(ddofItem))
        {
            // read ddof passed
            ddof = PyLong_AsLongLong(ddofItem);
        }
        tupleSize--;
    }

    // Check we were called with the right number of arguments.
    if (tupleSize != 2)
    {
        return PyErr_Format(PyExc_ValueError, "Reduce only takes two args instead of %lld args", tupleSize);
    }

    inArr1 = (PyArrayObject *)PyTuple_GET_ITEM(args, 0);

    if (IsFastArrayOrNumpy(inArr1))
    {
        // make sure not strided -- we don't currently handle those
        if (PyArray_STRIDE(inArr1, 0) == PyArray_ITEMSIZE(inArr1))
        {
            PyObject * object2 = PyTuple_GET_ITEM(args, 1);
            if (PyLong_Check(object2))
            {
                int64_t func = PyLong_AsLongLong(object2);

                // possible TODO: wrap result back into np scalar type
                return ReduceInternal(inArr1, (REDUCE_FUNCTIONS)func, ddof);
            }
        }
        else
        {
            // TODO: future work -- we can make a copy of the array if strided (which
            // removes the striding)
            //       then call our normal routine.
            //       Or, implement a special modified version of the routine which
            //       handles strided arrays. Could make this decision on a
            //       per-function basis if we modify GetConversionFunction() above to
            //       accept the array shape and a flag indicating whether the array is
            //       strided, then it can return a pointer to the appropriate function
            //       if one is available, or just return nullptr (as normal) in which
            //       case we'll punt to numpy.
        }
    }

    // punt to numpy
    Py_RETURN_NONE;
}
