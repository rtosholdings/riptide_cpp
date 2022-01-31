#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"

#include "CommonInc.h"
#include "Compare.h"
#include "Convert.h"
#include "BasicMath.h"
#include "platform_detect.h"

//#define LOGGING printf
#define LOGGING(...)

// TODO: handle div by zero, etc.
//#pragma STDC FENV_ACCESS ON
// std::feclearexcept(FE_ALL_EXCEPT);
// TODO: add floatstatus
//#include <cfenv>
//#include <cmath>
// int npy_get_floatstatus(void)
//{
//   int fpstatus = fetestexcept(FE_DIVBYZERO | FE_OVERFLOW |
//      FE_UNDERFLOW | FE_INVALID);
//
//   return ((FE_DIVBYZERO  & fpstatus) ? NPY_FPE_DIVIDEBYZERO : 0) |
//      ((FE_OVERFLOW   & fpstatus) ? NPY_FPE_OVERFLOW : 0) |
//      ((FE_UNDERFLOW  & fpstatus) ? NPY_FPE_UNDERFLOW : 0) |
//      ((FE_INVALID    & fpstatus) ? NPY_FPE_INVALID : 0);
//}
//
// int npy_clear_floatstatus(void)
//{
//   /* testing float status is 50-100 times faster than clearing on x86 */
//   int fpstatus = npy_get_floatstatus();
//   if (fpstatus != 0) {
//      feclearexcept(FE_DIVBYZERO | FE_OVERFLOW |
//         FE_UNDERFLOW | FE_INVALID);
//   }
//
//   return fpstatus;
//}

static const inline void STOREU(__m256d * x, __m256d y)
{
    _mm256_storeu_pd((double *)x, y);
}
static const inline void STOREU(__m256 * x, __m256 y)
{
    _mm256_storeu_ps((float *)x, y);
}
static const inline void STOREU(__m256i * x, __m256i y)
{
    _mm256_storeu_si256((__m256i *)x, y);
}

// For aligned loads which must be on 32 byte boundary
static const inline __m256d LOADA(__m256d * x)
{
    return _mm256_load_pd((double const *)x);
};
static const inline __m256 LOADA(__m256 * x)
{
    return _mm256_load_ps((float const *)x);
};
static const inline __m256i LOADA(__m256i * x)
{
    return _mm256_load_si256((__m256i const *)x);
};

static const inline void STOREA(__m256d * x, __m256d y)
{
    _mm256_store_pd((double *)x, y);
}
static const inline void STOREA(__m256 * x, __m256 y)
{
    _mm256_store_ps((float *)x, y);
}
static const inline void STOREA(__m256i * x, __m256i y)
{
    _mm256_store_si256((__m256i *)x, y);
}

static const inline __m256d LOADU(__m256d * x)
{
    return _mm256_loadu_pd((double const *)x);
}
static const inline __m256 LOADU(__m256 * x)
{
    return _mm256_loadu_ps((float const *)x);
}
static const inline __m256i LOADU(__m256i * x)
{
    return _mm256_loadu_si256((__m256i const *)x);
}

static const inline __m256i MIN_OPi8(__m256i x, __m256i y)
{
    return _mm256_min_epi8(x, y);
}
static const inline __m256i MIN_OPu8(__m256i x, __m256i y)
{
    return _mm256_min_epu8(x, y);
}
static const inline __m256i MIN_OPu16(__m256i x, __m256i y)
{
    return _mm256_min_epu16(x, y);
}
static const inline __m256i MIN_OPi16(__m256i x, __m256i y)
{
    return _mm256_min_epi16(x, y);
}
static const inline __m256i MIN_OPi32(__m256i x, __m256i y)
{
    return _mm256_min_epi32(x, y);
}
static const inline __m256i MIN_OPu32(__m256i x, __m256i y)
{
    return _mm256_min_epu32(x, y);
}

// Below unused
// static const inline __m256  MIN_OPf32( __m256  x, __m256 y) { return
// _mm256_min_ps(x, y); } static const inline __m256d MIN_OPf64( __m256d x,
// __m256d y) { return _mm256_min_pd(x, y); }

static const inline __m256i MAX_OPi8(__m256i x, __m256i y)
{
    return _mm256_max_epi8(x, y);
}
static const inline __m256i MAX_OPu8(__m256i x, __m256i y)
{
    return _mm256_max_epu8(x, y);
}
static const inline __m256i MAX_OPu16(__m256i x, __m256i y)
{
    return _mm256_max_epu16(x, y);
}
static const inline __m256i MAX_OPi16(__m256i x, __m256i y)
{
    return _mm256_max_epi16(x, y);
}
static const inline __m256i MAX_OPi32(__m256i x, __m256i y)
{
    return _mm256_max_epi32(x, y);
}
static const inline __m256i MAX_OPu32(__m256i x, __m256i y)
{
    return _mm256_max_epu32(x, y);
}
// NOTE: Cannot use for np.maximum since it handles nans and intel intrinsic
// does not handle the same way NOTE: Could use if we want a maximum and we dont
// care about nan handling Below unused
// static const inline __m256  MAX_OPf32(__m256  x, __m256 y) { return
// _mm256_max_ps(x, y); } static const inline __m256d MAX_OPf64(__m256d x,
// __m256d y) { return _mm256_max_pd(x, y); }

template <typename T>
static const inline T AddOp(T x, T y)
{
    return x + y;
}
template <typename T>
static const inline bool AddOp(bool x, bool y)
{
    return x | y;
}
template <typename T>
static const inline T SubOp(T x, T y)
{
    return x - y;
}
template <typename T>
static const inline bool SubOp(bool x, bool y)
{
    return (x == true) && (y == false);
}
template <typename T>
static const inline T MulOp(T x, T y)
{
    return x * y;
}

// nan != nan --> True
// nan > anynumber --> False
// nan < anynumber --> False
template <typename T>
static const inline float MinOp(float x, float y)
{
    if (x > y || y != y)
        return y;
    else
        return x;
}
template <typename T>
static const inline double MinOp(double x, double y)
{
    if (x > y || y != y)
        return y;
    else
        return x;
}
template <typename T>
static const inline T MinOp(T x, T y)
{
    if (x > y)
        return y;
    else
        return x;
}

template <typename T>
static const inline float MaxOp(float x, float y)
{
    if (x < y || y != y)
        return y;
    else
        return x;
}
template <typename T>
static const inline double MaxOp(double x, double y)
{
    if (x < y || y != y)
        return y;
    else
        return x;
}
template <typename T>
static const inline T MaxOp(T x, T y)
{
    if (x < y)
        return y;
    else
        return x;
}
// template<typename T> static const inline bool MulOp(bool x, bool y) { return
// x & y; }

static const int64_t NAN_FOR_INT64 = (int64_t)0x8000000000000000LL;
static const int64_t NAN_FOR_INT32 = (int32_t)0x80000000;

template <typename T>
static const inline double DivOp(T x, T y)
{
    return (double)x / (double)y;
}
template <typename T>
static const inline float DivOp(float x, T y)
{
    return x / y;
}

// Subtraction only two DateTimeNano subtractions
template <typename T>
static const inline double SubDateTimeOp(int64_t x, int64_t y)
{
    if (x == 0 || y == 0 || x == NAN_FOR_INT64 || y == NAN_FOR_INT64)
    {
        return NAN;
    }
    return (double)(x - y);
}

// Subtraction only both sides checked
template <typename T>
static const inline double SubDateTimeOp(int32_t x, int32_t y)
{
    if (x == 0 || y == 0 || x == NAN_FOR_INT32 || y == NAN_FOR_INT32)
    {
        return NAN;
    }
    return (double)(x - y);
}

// Subtract two dates to produce a DateSpan
template <typename T>
static const inline int32_t SubDatesOp(int32_t x, int32_t y)
{
    if (x == 0 || y == 0 || x == NAN_FOR_INT32 || y == NAN_FOR_INT32)
    {
        return 0;
    }
    return (x - y);
}

template <typename T>
static const inline int64_t SubDatesOp(int64_t x, int64_t y)
{
    if (x == 0 || y == 0 || x == NAN_FOR_INT64 || y == NAN_FOR_INT64)
    {
        return 0;
    }
    return (x - y);
}

template <typename T>
static const inline T FloorDivOp(T x, T y)
{
    if (y != 0)
    {
        if ((x < 0) == (y < 0))
            return x / y;

        T q = x / y;
        T r = x % y;
        if (r != 0)
            --q;
        return q;
    }
    else
        return 0;
}
template <typename T>
static const inline long double FloorDivOp(long double x, long double y)
{
    if (y == y)
        return floorl(x / y);
    else
        return NAN;
}
template <typename T>
static const inline double FloorDivOp(double x, double y)
{
    if (y == y)
        return floor(x / y);
    else
        return NAN;
}
template <typename T>
static const inline float FloorDivOp(float x, float y)
{
    if (y == y)
        return floorf(x / y);
    else
        return NAN;
}

template <typename T>
static const inline T ModOp(T x, T y)
{
    return x % y;
}
template <typename T>
static const inline float ModOp(T x, T y)
{
    return fmodf(x, y);
}
template <typename T>
static const inline double ModOp(T x, T y)
{
    return fmod(x, y);
}
template <typename T>
static const inline bool ModOp(bool x, bool y)
{
    return (x ^ y);
}

template <typename T>
static const inline T PowerOp(T x, T y)
{
    return (T)pow(x, y);
}
template <typename T>
static const inline float PowerOp(float x, float y)
{
    return powf(x, y);
}
template <typename T>
static const inline double PowerOp(double x, double y)
{
    return pow(x, y);
}
template <typename T>
static const inline long double PowerOp(long double x, long double y)
{
    return powl(x, y);
}
template <typename T>
static const inline bool PowerOp(bool x, bool y)
{
    if (! x & y)
        return 0;
    else
        return 1;
}

// numpy np.remainder is not 'C' remainder but there is a new math function
// template<typename T> static const inline long double REMAINDER_OP(long double
// x, long double y) { return remainderl(x, y); } template<typename T> static
// const inline double REMAINDER_OP(double x, double y) { return remainder(x, y);
// } template<typename T> static const inline float REMAINDER_OP(float x, float
// y) { return remainderf(x, y); }

template <typename T>
static const inline long double REMAINDER_OP(long double x, long double y)
{
    return x - (y * floorl(x / y));
}
template <typename T>
static const inline double REMAINDER_OP(double x, double y)
{
    return x - (y * floor(x / y));
}
template <typename T>
static const inline float REMAINDER_OP(float x, float y)
{
    return x - (y * floorf(x / y));
}

template <typename T>
static const inline long double FMOD_OP(long double x, long double y)
{
    return fmodl(x, y);
}
template <typename T>
static const inline double FMOD_OP(double x, double y)
{
    return fmod(x, y);
}
template <typename T>
static const inline float FMOD_OP(float x, float y)
{
    return fmodf(x, y);
}

template <typename T>
static const inline long double LogOp(long double x)
{
    return logl(x);
}
template <typename T>
static const inline double LogOp(double x)
{
    return log(x);
}
template <typename T>
static const inline float LogOp(float x)
{
    return logf(x);
}

// bitwise operations
template <typename T>
static const inline T AndOp(T x, T y)
{
    return x & y;
}
template <typename T>
static const inline T XorOp(T x, T y)
{
    return x ^ y;
}
template <typename T>
static const inline T OrOp(T x, T y)
{
    return x | y;
}
// NOTE: mimics intel intrinsic
template <typename T>
static const inline T AndNotOp(T x, T y)
{
    return ~x & y;
}

//=========================================================================================
// TJD NOTE:inline does not work with some compilers and templates in some loops
static const inline __m256 ADD_OP_256f32(__m256 x, __m256 y)
{
    return _mm256_add_ps(x, y);
}
static const inline __m256d ADD_OP_256f64(__m256d x, __m256d y)
{
    return _mm256_add_pd(x, y);
}
static const inline __m256i ADD_OP_256i32(__m256i x, __m256i y)
{
    return _mm256_add_epi32(x, y);
}
static const inline __m256i ADD_OP_256i64(__m256i x, __m256i y)
{
    return _mm256_add_epi64(x, y);
}
static const inline __m256i ADD_OP_256i16(__m256i x, __m256i y)
{
    return _mm256_add_epi16(x, y);
}
static const inline __m256i ADD_OP_256i8(__m256i x, __m256i y)
{
    return _mm256_add_epi8(x, y);
}

static const inline __m256 SUB_OP_256f32(__m256 x, __m256 y)
{
    return _mm256_sub_ps(x, y);
}
static const inline __m256d SUB_OP_256f64(__m256d x, __m256d y)
{
    return _mm256_sub_pd(x, y);
}
static const inline __m256i SUB_OP_256i32(__m256i x, __m256i y)
{
    return _mm256_sub_epi32(x, y);
}
static const inline __m256i SUB_OP_256i64(__m256i x, __m256i y)
{
    return _mm256_sub_epi64(x, y);
}
static const inline __m256i SUB_OP_256i16(__m256i x, __m256i y)
{
    return _mm256_sub_epi16(x, y);
}
static const inline __m256i SUB_OP_256i8(__m256i x, __m256i y)
{
    return _mm256_sub_epi8(x, y);
}

static const inline __m256 MUL_OP_256f32(__m256 x, __m256 y)
{
    return _mm256_mul_ps(x, y);
}
static const inline __m256d MUL_OP_256f64(__m256d x, __m256d y)
{
    return _mm256_mul_pd(x, y);
}
static const inline __m256i MUL_OP_256i32(__m256i x, __m256i y)
{
    return _mm256_mullo_epi32(x, y);
}
static const inline __m256i MUL_OP_256i16(__m256i x, __m256i y)
{
    return _mm256_mullo_epi16(x, y);
}

// mask off low 32bits
static const __m256i masklo = _mm256_set1_epi64x(0xFFFFFFFFLL);
static const __m128i shifthigh = _mm_set1_epi64x(32);

// This routine only works for positive integers
static const inline __m256i MUL_OP_256u64(__m256i x, __m256i y)
{
    // Algo is lo1*lo2 + (lo1*hi2) << 32 + (lo2*hi1) << 32
    // To get to 128 bit int would have to add (hi1*hi2) << 64
    __m256i lo1 = _mm256_and_si256(x, masklo);
    __m256i lo2 = _mm256_and_si256(y, masklo);
    __m256i hi1 = _mm256_srl_epi64(x, shifthigh); // need to sign extend
    __m256i hi2 = _mm256_srl_epi64(y, shifthigh);
    __m256i add1 = _mm256_mul_epu32(lo1, lo2);
    __m256i add2 = _mm256_sll_epi64(_mm256_mul_epu32(lo1, hi2), shifthigh);
    __m256i add3 = _mm256_sll_epi64(_mm256_mul_epu32(lo2, hi1), shifthigh);
    // add all the results together
    return _mm256_add_epi64(add1, _mm256_add_epi64(add2, add3));
}

// eight 32bit --> produce four 64 bit
// below unused
// static const inline __m256i MULX_OP_256i32(__m256i x, __m256i y) { return
// _mm256_mul_epi32(x, y); } static const inline __m256i MULX_OP_256u32(__m256i
// x, __m256i y) { return _mm256_mul_epu32(x, y); }

static const inline __m256 DIV_OP_256f32(__m256 x, __m256 y)
{
    return _mm256_div_ps(x, y);
}
static const inline __m256d DIV_OP_256f64(__m256d x, __m256d y)
{
    return _mm256_div_pd(x, y);
}
static const inline __m256d CONV_INT32_DOUBLE(__m128i * x)
{
    return _mm256_cvtepi32_pd(*x);
}

// static const inline __m256d CONV_INT16_DOUBLE(__m128i x) { return
// _mm256_cvtepi16_pd(x); } static const inline __m256d DIV_OP_256(const int32_t
// z, __m128i x, __m128i y) { return _mm256_div_pd(_mm256_cvtepi32_pd(x),
// _mm256_cvtepi32_pd(y)); } static const inline __m256d DIV_OP_256(const int64_t
// z, __m128i x, __m128i y) { return _mm256_div_pd(_mm256_cvtepi64_pd(x),
// _mm256_cvtepi64_pd(y)); }

static const inline __m256i AND_OP_256(__m256i x, __m256i y)
{
    return _mm256_and_si256(x, y);
}
static const inline __m256i OR_OP_256(__m256i x, __m256i y)
{
    return _mm256_or_si256(x, y);
}
static const inline __m256i XOR_OP_256(__m256i x, __m256i y)
{
    return _mm256_xor_si256(x, y);
}
static const inline __m256i ANDNOT_OP_256(__m256i x, __m256i y)
{
    return _mm256_andnot_si256(x, y);
}

//_asm cvt2q
// CVTDQ2PD
// inline __m256d DIV_OP_256(const int64_t z, __m256i x, __m256i y) { return
// _mm256_div_pd(_mm256_cvtepi64_pd(x), _mm256_cvtepi64_pd(y)); }

template <typename T, typename U256>
static const inline __m256 AddOp256(const float z, U256 x, U256 y)
{
    return _mm256_add_ps(x, y);
}
template <typename T, typename U256>
static const inline __m256d AddOp256(const double z, U256 x, U256 y)
{
    return _mm256_add_pd(x, y);
}
template <typename T, typename U256>
static const inline __m256i AddOp256(const int32_t z, U256 x, U256 y)
{
    return _mm256_add_epi32(x, y);
}

template <typename T, typename MathFunctionPtr>
FORCEINLINE void SimpleMathOpSlow(MathFunctionPtr MATH_OP, void * pDataIn1X, void * pDataIn2X, void * pDataOutX, int64_t len,
                                  int32_t scalarMode)
{
    T * pDataOut = (T *)pDataOutX;
    T * pDataIn1 = (T *)pDataIn1X;
    T * pDataIn2 = (T *)pDataIn2X;

    LOGGING("slow  In:%p  In2:%p  Out:%p\n", pDataIn1, pDataIn2, pDataOut);

    switch (scalarMode)
    {
    case SCALAR_MODE::NO_SCALARS:
        {
            for (int64_t i = 0; i < len; i++)
            {
                pDataOut[i] = MATH_OP(pDataIn1[i], pDataIn2[i]);
            }
            break;
        }
    case SCALAR_MODE::FIRST_ARG_SCALAR:
        {
            T arg1 = *pDataIn1;
            for (int64_t i = 0; i < len; i++)
            {
                pDataOut[i] = MATH_OP(arg1, pDataIn2[i]);
            }
            break;
        }
    case SCALAR_MODE::SECOND_ARG_SCALAR:
        {
            T arg2 = *pDataIn2;
            // printf("arg2 = %lld  %lld\n", (int64_t)arg2, (int64_t)pDataIn1[0]);
            for (int64_t i = 0; i < len; i++)
            {
                pDataOut[i] = MATH_OP(pDataIn1[i], arg2);
            }
            break;
        }
    default:
        T arg1 = *pDataIn1;
        T arg2 = *pDataIn2;
        for (int64_t i = 0; i < len; i++)
        {
            pDataOut[i] = MATH_OP(arg1, arg2);
        }
        break;
    }
}

//=====================================================================================================
// Used in division where the output is a double
template <typename T, typename MathFunctionPtr>
FORCEINLINE void SimpleMathOpSlowDouble(MathFunctionPtr MATH_OP, void * pDataIn1X, void * pDataIn2X, void * pDataOutX, int64_t len,
                                        int32_t scalarMode)
{
    double * pDataOut = (double *)pDataOutX;
    T * pDataIn1 = (T *)pDataIn1X;
    T * pDataIn2 = (T *)pDataIn2X;

    LOGGING("slow div or dates double %lld  %d  %p  %p  %p\n", len, scalarMode, pDataIn1X, pDataIn2X, pDataOutX);

    switch (scalarMode)
    {
    case SCALAR_MODE::NO_SCALARS:
        {
            for (int64_t i = 0; i < len; i++)
            {
                pDataOut[i] = MATH_OP(pDataIn1[i], pDataIn2[i]);
            }
            break;
        }
    case SCALAR_MODE::FIRST_ARG_SCALAR:
        {
            T arg1 = *pDataIn1;
            for (int64_t i = 0; i < len; i++)
            {
                pDataOut[i] = MATH_OP(arg1, pDataIn2[i]);
            }
            break;
        }
    case SCALAR_MODE::SECOND_ARG_SCALAR:
        {
            T arg2 = *pDataIn2;
            for (int64_t i = 0; i < len; i++)
            {
                pDataOut[i] = MATH_OP(pDataIn1[i], arg2);
            }
            break;
        }
    default:
        T arg1 = *pDataIn1;
        T arg2 = *pDataIn2;
        for (int64_t i = 0; i < len; i++)
        {
            pDataOut[i] = MATH_OP(arg1, arg2);
        }
        break;
        // printf("**error - impossible scalar mode\n");
    }
}

//=====================================================================================================
// Not symmetric -- arg1 must be first, arg2 must be second
template <typename T, typename U256, const T MATH_OP(T, T), const U256 MATH_OP256(U256, U256)>
inline void SimpleMathOpFast(void * pDataIn1X, void * pDataIn2X, void * pDataOutX, int64_t datalen, int32_t scalarMode)
{
    T * pDataOut = (T *)pDataOutX;
    T * pDataIn1 = (T *)pDataIn1X;
    T * pDataIn2 = (T *)pDataIn2X;

    const int64_t NUM_LOOPS_UNROLLED = 1;
    const int64_t chunkSize = NUM_LOOPS_UNROLLED * (sizeof(U256) / sizeof(T));
    int64_t perReg = sizeof(U256) / sizeof(T);

    LOGGING("mathopfast datalen %llu  chunkSize %llu  perReg %llu\n", datalen, chunkSize, perReg);

    switch (scalarMode)
    {
    case SCALAR_MODE::NO_SCALARS:
        {
            if (datalen >= chunkSize)
            {
                T * pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
                U256 * pEnd_256 = (U256 *)pEnd;
                U256 * pIn1_256 = (U256 *)pDataIn1;
                U256 * pIn2_256 = (U256 *)pDataIn2;
                U256 * pOut_256 = (U256 *)pDataOut;

                do
                {
                    // clang requires LOADU on last operand
#ifdef RT_COMPILER_MSVC
                    // Microsoft will create the opcode where the second argument is an
                    // address
                    STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), *pIn2_256));
#else
                    STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), LOADU(pIn2_256)));
#endif
                    pOut_256 += NUM_LOOPS_UNROLLED;
                    pIn1_256 += NUM_LOOPS_UNROLLED;
                    pIn2_256 += NUM_LOOPS_UNROLLED;
                }
                while (pOut_256 < pEnd_256);

                // update pointers to last location of wide pointers
                pDataIn1 = (T *)pIn1_256;
                pDataIn2 = (T *)pIn2_256;
                pDataOut = (T *)pOut_256;
            }

            datalen = datalen & (chunkSize - 1);
            for (int64_t i = 0; i < datalen; i++)
            {
                pDataOut[i] = MATH_OP(pDataIn1[i], pDataIn2[i]);
            }

            break;
        }
    case SCALAR_MODE::FIRST_ARG_SCALAR:
        {
            // NOTE: the unrolled loop is faster
            T arg1 = *pDataIn1;

            if (datalen >= chunkSize)
            {
                T * pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
                U256 * pEnd_256 = (U256 *)pEnd;
                U256 * pIn1_256 = (U256 *)pDataIn1;
                U256 * pIn2_256 = (U256 *)pDataIn2;
                U256 * pOut_256 = (U256 *)pDataOut;

                const U256 m0 = LOADU(pIn1_256);

                do
                {
#ifdef RT_COMPILER_MSVC
                    STOREU(pOut_256, MATH_OP256(m0, *pIn2_256));
#else
                    STOREU(pOut_256, MATH_OP256(m0, LOADU(pIn2_256)));
#endif

                    pOut_256 += NUM_LOOPS_UNROLLED;
                    pIn2_256 += NUM_LOOPS_UNROLLED;
                }
                while (pOut_256 < pEnd_256);

                // update pointers to last location of wide pointers
                pDataIn2 = (T *)pIn2_256;
                pDataOut = (T *)pOut_256;
            }
            datalen = datalen & (chunkSize - 1);
            for (int64_t i = 0; i < datalen; i++)
            {
                pDataOut[i] = MATH_OP(arg1, pDataIn2[i]);
            }
            break;
        }
    case SCALAR_MODE::SECOND_ARG_SCALAR:
        {
            T arg2 = *pDataIn2;

            // Check if the output is the same as the input
            if (pDataOut == pDataIn1)
            {
                // align the load to 32 byte boundary
                int64_t babylen = (int64_t)pDataIn1 & 31;
                if (babylen != 0)
                {
                    // calc how much to align data
                    babylen = (32 - babylen) / sizeof(T);
                    if (babylen <= datalen)
                    {
                        for (int64_t i = 0; i < babylen; i++)
                        {
                            pDataIn1[i] = MATH_OP(pDataIn1[i], arg2);
                        }
                        pDataIn1 += babylen;
                        datalen -= babylen;
                    }
                }

                // inplace operation
                if (datalen >= chunkSize)
                {
                    T * pEnd = &pDataIn1[chunkSize * (datalen / chunkSize)];
                    U256 * pEnd_256 = (U256 *)pEnd;
                    U256 * pIn1_256 = (U256 *)pDataIn1;
                    U256 * pIn2_256 = (U256 *)pDataIn2;

                    const U256 m1 = LOADU(pIn2_256);

                    // apply 256bit aligned operations
                    while (pIn1_256 < pEnd_256)
                    {
                        STOREA(pIn1_256, MATH_OP256(LOADA(pIn1_256), m1));
                        pIn1_256++;
                    }

                    // update pointers to last location of wide pointers
                    pDataIn1 = (T *)pIn1_256;
                }
                datalen = datalen & (chunkSize - 1);
                for (int64_t i = 0; i < datalen; i++)
                {
                    pDataIn1[i] = MATH_OP(pDataIn1[i], arg2);
                }
            }
            else
            {
                if (datalen >= chunkSize)
                {
                    T * pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
                    U256 * pEnd_256 = (U256 *)pEnd;
                    U256 * pIn1_256 = (U256 *)pDataIn1;
                    U256 * pIn2_256 = (U256 *)pDataIn2;
                    U256 * pOut_256 = (U256 *)pDataOut;

                    const U256 m1 = LOADU((U256 *)pIn2_256);

                    // apply 256bit unaligned operations
                    do
                    {
                        STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), m1));

                        pOut_256 += NUM_LOOPS_UNROLLED;
                        pIn1_256 += NUM_LOOPS_UNROLLED;
                    }
                    while (pOut_256 < pEnd_256);

                    // update pointers to last location of wide pointers
                    pDataIn1 = (T *)pIn1_256;
                    pDataOut = (T *)pOut_256;
                }
                datalen = datalen & (chunkSize - 1);
                for (int64_t i = 0; i < datalen; i++)
                {
                    pDataOut[i] = MATH_OP(pDataIn1[i], arg2);
                }
            }
            break;
        }
    default:
        printf("**error - impossible scalar mode\n");
    }
}

//=====================================================================================================
// Flips arg1 and arg2 around for BITWISE_NOTAND
template <typename T, typename U256, const T MATH_OP(T, T), const U256 MATH_OP256(U256, U256)>
inline void SimpleMathOpFastReverse(void * pDataIn1X, void * pDataIn2X, void * pDataOutX, int64_t datalen, int32_t scalarMode)
{
    return SimpleMathOpFast<T, U256, MATH_OP, MATH_OP256>(pDataIn2X, pDataIn1X, pDataOutX, datalen, scalarMode);
}

//=====================================================================================================
// Not symmetric -- arg1 must be first, arg2 must be second
template <typename T, typename U256, const T MATH_OP(T, T), const U256 MATH_OP256(U256, U256)>
inline void SimpleMathOpFastSymmetric(void * pDataIn1X, void * pDataIn2X, void * pDataOutX, int64_t datalen, int32_t scalarMode)
{
    T * pDataOut = (T *)pDataOutX;
    T * pDataIn1 = (T *)pDataIn1X;
    T * pDataIn2 = (T *)pDataIn2X;

    const int64_t NUM_LOOPS_UNROLLED = 1;
    const int64_t chunkSize = NUM_LOOPS_UNROLLED * (sizeof(U256) / sizeof(T));
    int64_t perReg = sizeof(U256) / sizeof(T);

    LOGGING("mathopfast datalen %llu  chunkSize %llu  perReg %llu\n", datalen, chunkSize, perReg);

    switch (scalarMode)
    {
    case SCALAR_MODE::NO_SCALARS:
        {
            if (datalen >= chunkSize)
            {
                T * pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
                U256 * pEnd_256 = (U256 *)pEnd;
                U256 * pIn1_256 = (U256 *)pDataIn1;
                U256 * pIn2_256 = (U256 *)pDataIn2;
                U256 * pOut_256 = (U256 *)pDataOut;

                do
                {
                    // clang requires LOADU on last operand
#ifdef RT_COMPILER_MSVC
                    STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), *pIn2_256));
#else
                    STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), LOADU(pIn2_256)));
#endif
                    pOut_256 += NUM_LOOPS_UNROLLED;
                    pIn1_256 += NUM_LOOPS_UNROLLED;
                    pIn2_256 += NUM_LOOPS_UNROLLED;
                }
                while (pOut_256 < pEnd_256);

                // update pointers to last location of wide pointers
                pDataIn1 = (T *)pIn1_256;
                pDataIn2 = (T *)pIn2_256;
                pDataOut = (T *)pOut_256;
            }

            datalen = datalen & (chunkSize - 1);
            for (int64_t i = 0; i < datalen; i++)
            {
                pDataOut[i] = MATH_OP(pDataIn1[i], pDataIn2[i]);
            }

            break;
        }
    case SCALAR_MODE::FIRST_ARG_SCALAR:
        {
            // NOTE: the unrolled loop is faster
            T arg1 = *pDataIn1;

            if (datalen >= chunkSize)
            {
                T * pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
                U256 * pEnd_256 = (U256 *)pEnd;
                U256 * pIn1_256 = (U256 *)pDataIn1;
                U256 * pIn2_256 = (U256 *)pDataIn2;
                U256 * pOut_256 = (U256 *)pDataOut;

                const U256 m0 = LOADU(pIn1_256);

                do
                {
#ifdef RT_COMPILER_MSVC
                    STOREU(pOut_256, MATH_OP256(m0, *pIn2_256));
#else
                    STOREU(pOut_256, MATH_OP256(m0, LOADU(pIn2_256)));
#endif

                    pOut_256 += NUM_LOOPS_UNROLLED;
                    pIn2_256 += NUM_LOOPS_UNROLLED;
                }
                while (pOut_256 < pEnd_256);

                // update pointers to last location of wide pointers
                pDataIn2 = (T *)pIn2_256;
                pDataOut = (T *)pOut_256;
            }
            datalen = datalen & (chunkSize - 1);
            for (int64_t i = 0; i < datalen; i++)
            {
                pDataOut[i] = MATH_OP(arg1, pDataIn2[i]);
            }
            break;
        }
    case SCALAR_MODE::SECOND_ARG_SCALAR:
        {
            T arg2 = *pDataIn2;

            // Check if the output is the same as the input
            if (pDataOut == pDataIn1)
            {
                // align the load to 32 byte boundary
                int64_t babylen = (int64_t)pDataIn1 & 31;
                if (babylen != 0)
                {
                    // calc how much to align data
                    babylen = (32 - babylen) / sizeof(T);
                    if (babylen <= datalen)
                    {
                        for (int64_t i = 0; i < babylen; i++)
                        {
                            pDataIn1[i] = MATH_OP(pDataIn1[i], arg2);
                        }
                        pDataIn1 += babylen;
                        datalen -= babylen;
                    }
                }

                // inplace operation
                if (datalen >= chunkSize)
                {
                    T * pEnd = &pDataIn1[chunkSize * (datalen / chunkSize)];
                    U256 * pEnd_256 = (U256 *)pEnd;
                    U256 * pIn1_256 = (U256 *)pDataIn1;
                    U256 * pIn2_256 = (U256 *)pDataIn2;

                    const U256 m1 = LOADU(pIn2_256);

                    // apply 256bit aligned operations
                    while (pIn1_256 < pEnd_256)
                    {
                        // pin1_256 is aligned
                        STOREA(pIn1_256, MATH_OP256(m1, *pIn1_256));
                        pIn1_256++;
                    }

                    // update pointers to last location of wide pointers
                    pDataIn1 = (T *)pIn1_256;
                }
                datalen = datalen & (chunkSize - 1);
                for (int64_t i = 0; i < datalen; i++)
                {
                    pDataIn1[i] = MATH_OP(pDataIn1[i], arg2);
                }
            }
            else
            {
                if (datalen >= chunkSize)
                {
                    T * pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
                    U256 * pEnd_256 = (U256 *)pEnd;
                    U256 * pIn1_256 = (U256 *)pDataIn1;
                    U256 * pIn2_256 = (U256 *)pDataIn2;
                    U256 * pOut_256 = (U256 *)pDataOut;

                    const U256 m1 = LOADU((U256 *)pIn2_256);

                    // apply 256bit unaligned operations
                    do
                    {
#ifdef RT_COMPILER_MSVC
                        STOREU(pOut_256, MATH_OP256(m1, *pIn1_256));
#else
                        STOREU(pOut_256, MATH_OP256(m1, LOADU(pIn1_256)));
#endif
                        pOut_256 += NUM_LOOPS_UNROLLED;
                        pIn1_256 += NUM_LOOPS_UNROLLED;
                    }
                    while (pOut_256 < pEnd_256);

                    // update pointers to last location of wide pointers
                    pDataIn1 = (T *)pIn1_256;
                    pDataOut = (T *)pOut_256;
                }
                datalen = datalen & (chunkSize - 1);
                for (int64_t i = 0; i < datalen; i++)
                {
                    pDataOut[i] = MATH_OP(pDataIn1[i], arg2);
                }
            }
            break;
        }
    default:
        printf("**error - impossible scalar mode\n");
    }
}

//=====================================================================================================

template <typename T, typename U128, typename U256, typename MathFunctionPtr, typename MathFunctionConvert,
          typename MathFunctionPtr256>
inline void SimpleMathOpFastDouble(MathFunctionPtr MATH_OP, MathFunctionConvert MATH_CONV, MathFunctionPtr256 MATH_OP256,
                                   void * pDataIn1X, void * pDataIn2X, void * pDataOutX, int64_t len, int32_t scalarMode)
{
    double * pDataOut = (double *)pDataOutX;
    T * pDataIn1 = (T *)pDataIn1X;
    T * pDataIn2 = (T *)pDataIn2X;
    const double dummy = 0;

    int64_t chunkSize = (sizeof(U256)) / sizeof(double);
    int64_t perReg = sizeof(U256) / sizeof(double);

    LOGGING("mathopfastDouble len %llu  chunkSize %llu  perReg %llu\n", len, chunkSize, perReg);

    switch (scalarMode)
    {
    case SCALAR_MODE::NO_SCALARS:
        {
            if (len >= chunkSize)
            {
                double * pEnd = &pDataOut[chunkSize * (len / chunkSize)];
                U256 * pEnd_256 = (U256 *)pEnd;

                U128 * pIn1_256 = (U128 *)pDataIn1;
                U128 * pIn2_256 = (U128 *)pDataIn2;
                U256 * pOut_256 = (U256 *)pDataOut;

                while (pOut_256 < pEnd_256)
                {
                    STOREU(pOut_256, MATH_OP256(MATH_CONV(pIn1_256), MATH_CONV(pIn2_256)));
                    pIn1_256 += 1;
                    pIn2_256 += 1;
                    pOut_256 += 1;
                }

                // update pointers to last location of wide pointers
                pDataIn1 = (T *)pIn1_256;
                pDataIn2 = (T *)pIn2_256;
                pDataOut = (double *)pOut_256;
            }

            len = len & (chunkSize - 1);
            for (int64_t i = 0; i < len; i++)
            {
                pDataOut[i] = MATH_OP(pDataIn1[i], pDataIn2[i]);
            }

            break;
        }
    case SCALAR_MODE::FIRST_ARG_SCALAR:
        {
            T arg1 = *pDataIn1;
            for (int64_t i = 0; i < len; i++)
            {
                pDataOut[i] = MATH_OP(arg1, pDataIn2[i]);
            }
            break;
        }
    case SCALAR_MODE::SECOND_ARG_SCALAR:
        {
            T arg2 = *pDataIn2;
            for (int64_t i = 0; i < len; i++)
            {
                pDataOut[i] = MATH_OP(pDataIn1[i], arg2);
            }
            break;
        }
    default:
        printf("**error - impossible scalar mode\n");
    }
}

template <typename T, typename U128, typename U256>
static void SimpleMathOpFastDivDouble(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpFastDouble<T, U128, U256, const double (*)(T, T), const U256 (*)(U128 *), const U256 (*)(U256, U256)>(
        DivOp<T>, CONV_INT32_DOUBLE, DIV_OP_256f64, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

//--------------------------------------------------------------------------------------

template <typename T>
static void SimpleMathOpSlowAdd(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(AddOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowSub(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(SubOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowMul(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(MulOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowDivFloat(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const float (*)(float, T)>(DivOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowDiv(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlowDouble<T, const double (*)(T, T)>(DivOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSubDateTime(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlowDouble<T, const double (*)(T, T)>(SubDateTimeOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSubDates(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(SubDatesOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowFloorDiv(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(FloorDivOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowMod(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(ModOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowLog(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(LogOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowPower(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(PowerOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowRemainder(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(REMAINDER_OP<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowFmod(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(FMOD_OP<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowMin(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(MinOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

template <typename T>
static void SimpleMathOpSlowMax(void * pDataIn1, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode)
{
    return SimpleMathOpSlow<T, const T (*)(T, T)>(MaxOp<T>, pDataIn1, pDataIn2, pDataOut, len, scalarMode);
}

static ANY_TWO_FUNC GetSimpleMathOpFast(int func, int scalarMode, int numpyInType1, int numpyInType2, int numpyOutType,
                                        int * wantedOutType)
{
    LOGGING("GetSimpleMathOpFastFunc %d %d\n", numpyInType1, func);

    switch (func)
    {
    case MATH_OPERATION::ADD:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_BOOL:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, OrOp<int8_t>, OR_OP_256>;
        case NPY_FLOAT:
            return SimpleMathOpFastSymmetric<float, __m256, AddOp<float>, ADD_OP_256f32>;
        case NPY_DOUBLE:
            return SimpleMathOpFastSymmetric<double, __m256d, AddOp<double>, ADD_OP_256f64>;
            // proof of concept for i32 addition loop
        CASE_NPY_INT32:
            return SimpleMathOpFastSymmetric<int32_t, __m256i, AddOp<int32_t>, ADD_OP_256i32>;
        CASE_NPY_INT64:

            return SimpleMathOpFastSymmetric<int64_t, __m256i, AddOp<int64_t>, ADD_OP_256i64>;
        case NPY_INT16:
            return SimpleMathOpFastSymmetric<int16_t, __m256i, AddOp<int16_t>, ADD_OP_256i16>;
        case NPY_INT8:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, AddOp<int8_t>, ADD_OP_256i8>;
        }
        return nullptr;

    case MATH_OPERATION::MUL:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_BOOL:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, AndOp<int8_t>, AND_OP_256>;
        case NPY_FLOAT:
            return SimpleMathOpFastSymmetric<float, __m256, MulOp<float>, MUL_OP_256f32>;
        case NPY_DOUBLE:
            return SimpleMathOpFastSymmetric<double, __m256d, MulOp<double>, MUL_OP_256f64>;
        CASE_NPY_INT32:
            return SimpleMathOpFastSymmetric<int32_t, __m256i, MulOp<int32_t>, MUL_OP_256i32>;

        // CASE_NPY_INT64:   return SimpleMathOpFast<int64_t, __m256i,
        // MulOp<int64_t>, MUL_OP_256i64>;
        case NPY_INT16:
            return SimpleMathOpFastSymmetric<int16_t, __m256i, MulOp<int16_t>, MUL_OP_256i16>;

        // Below the intrinsic to multiply is slower so we disabled it (really wants
        // 32bit -> 64bit)
        // CASE_NPY_UINT32:  return SimpleMathOpFastMul<uint32_t, __m256i>;
        // TODO: 64bit multiply can be done with algo..
        // lo1 * lo2 + (lo1 * hi2) << 32 + (hi1 *lo2) << 32)
        CASE_NPY_UINT64:

            return SimpleMathOpFastSymmetric<uint64_t, __m256i, MulOp<uint64_t>, MUL_OP_256u64>;
        }
        return nullptr;

    case MATH_OPERATION::SUB:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_FLOAT:
            return SimpleMathOpFast<float, __m256, SubOp<float>, SUB_OP_256f32>;
        case NPY_DOUBLE:
            return SimpleMathOpFast<double, __m256d, SubOp<double>, SUB_OP_256f64>;
        CASE_NPY_INT32:
            return SimpleMathOpFast<int32_t, __m256i, SubOp<int32_t>, SUB_OP_256i32>;
        CASE_NPY_INT64:

            return SimpleMathOpFast<int64_t, __m256i, SubOp<int64_t>, SUB_OP_256i64>;
        case NPY_INT16:
            return SimpleMathOpFast<int16_t, __m256i, SubOp<int16_t>, SUB_OP_256i16>;
        case NPY_INT8:
            return SimpleMathOpFast<int8_t, __m256i, SubOp<int8_t>, SUB_OP_256i8>;
        }
        return nullptr;

    case MATH_OPERATION::MIN:
        // This is min for two arrays
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        // TODO: Vector routine needs to be written
        case NPY_FLOAT:
            return SimpleMathOpSlowMin<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowMin<double>;
        CASE_NPY_INT32:
            return SimpleMathOpFastSymmetric<int32_t, __m256i, MinOp<int32_t>, MIN_OPi32>;
        case NPY_INT16:
            return SimpleMathOpFastSymmetric<int16_t, __m256i, MinOp<int16_t>, MIN_OPi16>;
        case NPY_INT8:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, MinOp<int8_t>, MIN_OPi8>;
        CASE_NPY_UINT32:
            return SimpleMathOpFastSymmetric<uint32_t, __m256i, MinOp<uint32_t>, MIN_OPu32>;
        case NPY_UINT16:
            return SimpleMathOpFastSymmetric<uint16_t, __m256i, MinOp<uint16_t>, MIN_OPu16>;
        case NPY_UINT8:
            return SimpleMathOpFastSymmetric<uint8_t, __m256i, MinOp<uint8_t>, MIN_OPu8>;
        }
        return nullptr;

    case MATH_OPERATION::MAX:
        // This is max for two arrays
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        // TODO: Vector routine needs to be written
        case NPY_FLOAT:
            return SimpleMathOpSlowMax<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowMax<double>;
        CASE_NPY_INT32:
            return SimpleMathOpFastSymmetric<int32_t, __m256i, MaxOp<int32_t>, MAX_OPi32>;
        case NPY_INT16:
            return SimpleMathOpFastSymmetric<int16_t, __m256i, MaxOp<int16_t>, MAX_OPi16>;
        case NPY_INT8:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, MaxOp<int8_t>, MAX_OPi8>;
        CASE_NPY_UINT32:
            return SimpleMathOpFastSymmetric<uint32_t, __m256i, MaxOp<uint32_t>, MAX_OPu32>;
        case NPY_UINT16:
            return SimpleMathOpFastSymmetric<uint16_t, __m256i, MaxOp<uint16_t>, MAX_OPu16>;
        case NPY_UINT8:
            return SimpleMathOpFastSymmetric<uint8_t, __m256i, MaxOp<uint8_t>, MAX_OPu8>;
        }
        return nullptr;

    case MATH_OPERATION::DIV:
        *wantedOutType = NPY_DOUBLE;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
        {
            if (numpyInType2 == NPY_FLOAT)
                *wantedOutType = NPY_FLOAT;
        }
        else
        {
            if (numpyInType1 == NPY_FLOAT)
                *wantedOutType = NPY_FLOAT;
        }
        switch (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR ? numpyInType2 : numpyInType1)
        {
        case NPY_FLOAT:
            return SimpleMathOpFast<float, __m256, DivOp<float>, DIV_OP_256f32>;
        case NPY_DOUBLE:
            return SimpleMathOpFast<double, __m256d, DivOp<double>, DIV_OP_256f64>;
        CASE_NPY_INT32:
            return SimpleMathOpFastDivDouble<int32_t, __m128i, __m256d>;
        }
        return nullptr;

    case MATH_OPERATION::LOGICAL_AND:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_BOOL:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, AndOp<int8_t>, AND_OP_256>;
        }
        return nullptr;

    case MATH_OPERATION::BITWISE_AND:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_INT8:
        case NPY_UINT8:
        case NPY_BOOL:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, AndOp<int8_t>, AND_OP_256>;
        case NPY_UINT16:
        case NPY_INT16:
            return SimpleMathOpFastSymmetric<int16_t, __m256i, AndOp<int16_t>, AND_OP_256>;
        CASE_NPY_UINT32:
        CASE_NPY_INT32:
            return SimpleMathOpFastSymmetric<int32_t, __m256i, AndOp<int32_t>, AND_OP_256>;
        CASE_NPY_UINT64:

        CASE_NPY_INT64:

            return SimpleMathOpFastSymmetric<int64_t, __m256i, AndOp<int64_t>, AND_OP_256>;
        }
        return nullptr;

    case MATH_OPERATION::LOGICAL_OR:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_BOOL:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, OrOp<int8_t>, OR_OP_256>;
        }
        return nullptr;

    case MATH_OPERATION::BITWISE_OR:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_INT8:
        case NPY_UINT8:
        case NPY_BOOL:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, OrOp<int8_t>, OR_OP_256>;
        case NPY_UINT16:
        case NPY_INT16:
            return SimpleMathOpFastSymmetric<int16_t, __m256i, OrOp<int16_t>, OR_OP_256>;
        CASE_NPY_UINT32:
        CASE_NPY_INT32:
            return SimpleMathOpFastSymmetric<int32_t, __m256i, OrOp<int32_t>, OR_OP_256>;
        CASE_NPY_UINT64:

        CASE_NPY_INT64:

            return SimpleMathOpFastSymmetric<int64_t, __m256i, OrOp<int64_t>, OR_OP_256>;
        }
        return nullptr;

    case MATH_OPERATION::BITWISE_XOR:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_INT8:
        case NPY_UINT8:
        case NPY_BOOL:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, XorOp<int8_t>, XOR_OP_256>;
        case NPY_UINT16:
        case NPY_INT16:
            return SimpleMathOpFastSymmetric<int16_t, __m256i, XorOp<int16_t>, XOR_OP_256>;
        CASE_NPY_UINT32:
        CASE_NPY_INT32:
            return SimpleMathOpFastSymmetric<int32_t, __m256i, XorOp<int32_t>, XOR_OP_256>;
        CASE_NPY_UINT64:

        CASE_NPY_INT64:

            return SimpleMathOpFastSymmetric<int64_t, __m256i, XorOp<int64_t>, XOR_OP_256>;
        }
        return nullptr;

    case MATH_OPERATION::BITWISE_XOR_SPECIAL:
        *wantedOutType = numpyInType1;
        // SPECIAL does not change output type
        switch (*wantedOutType)
        {
        case NPY_INT8:
        case NPY_UINT8:
        case NPY_BOOL:
            return SimpleMathOpFastSymmetric<int8_t, __m256i, XorOp<int8_t>, XOR_OP_256>;
        case NPY_UINT16:
        case NPY_INT16:
            return SimpleMathOpFastSymmetric<int16_t, __m256i, XorOp<int16_t>, XOR_OP_256>;
        CASE_NPY_UINT32:
        CASE_NPY_INT32:
            return SimpleMathOpFastSymmetric<int32_t, __m256i, XorOp<int32_t>, XOR_OP_256>;
        CASE_NPY_UINT64:

        CASE_NPY_INT64:

            return SimpleMathOpFastSymmetric<int64_t, __m256i, XorOp<int64_t>, XOR_OP_256>;
        }
        return nullptr;

    case MATH_OPERATION::BITWISE_ANDNOT:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_INT8:
        case NPY_UINT8:
        case NPY_BOOL:
            return SimpleMathOpFastReverse<int8_t, __m256i, AndNotOp<int8_t>, ANDNOT_OP_256>;
        case NPY_UINT16:
        case NPY_INT16:
            return SimpleMathOpFastReverse<int16_t, __m256i, AndNotOp<int16_t>, ANDNOT_OP_256>;
        CASE_NPY_UINT32:
        CASE_NPY_INT32:
            return SimpleMathOpFastReverse<int32_t, __m256i, AndNotOp<int32_t>, ANDNOT_OP_256>;
        CASE_NPY_UINT64:

        CASE_NPY_INT64:

            return SimpleMathOpFastReverse<int64_t, __m256i, AndNotOp<int64_t>, ANDNOT_OP_256>;
        }
        return nullptr;

    case MATH_OPERATION::BITWISE_NOTAND:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_INT8:
        case NPY_UINT8:
        case NPY_BOOL:
            return SimpleMathOpFast<int8_t, __m256i, AndNotOp<int8_t>, ANDNOT_OP_256>;
        case NPY_UINT16:
        case NPY_INT16:
            return SimpleMathOpFast<int16_t, __m256i, AndNotOp<int16_t>, ANDNOT_OP_256>;
        CASE_NPY_UINT32:
        CASE_NPY_INT32:
            return SimpleMathOpFast<int32_t, __m256i, AndNotOp<int32_t>, ANDNOT_OP_256>;
        CASE_NPY_UINT64:

        CASE_NPY_INT64:

            return SimpleMathOpFast<int64_t, __m256i, AndNotOp<int64_t>, ANDNOT_OP_256>;
        }
        return nullptr;
    }

    return nullptr;
}

//==========================================================
// May return nullptr if it cannot handle type or function
static ANY_TWO_FUNC GetSimpleMathOpSlow(int func, int scalarMode, int numpyInType1, int numpyInType2, int numpyOutType,
                                        int * wantedOutType)
{
    switch (func)
    {
    case MATH_OPERATION::ADD:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_BOOL:
            return SimpleMathOpSlowAdd<bool>;
        case NPY_FLOAT:
            return SimpleMathOpSlowAdd<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowAdd<double>;
        case NPY_LONGDOUBLE:
            return SimpleMathOpSlowAdd<long double>;
        CASE_NPY_INT32:
            return SimpleMathOpSlowAdd<int32_t>;
        CASE_NPY_INT64:

            return SimpleMathOpSlowAdd<int64_t>;
        CASE_NPY_UINT32:
            return SimpleMathOpSlowAdd<uint32_t>;
        CASE_NPY_UINT64:

            return SimpleMathOpSlowAdd<uint64_t>;
        case NPY_INT8:
            return SimpleMathOpSlowAdd<int8_t>;
        case NPY_INT16:
            return SimpleMathOpSlowAdd<int16_t>;
        case NPY_UINT8:
            return SimpleMathOpSlowAdd<uint8_t>;
        case NPY_UINT16:
            return SimpleMathOpSlowAdd<uint16_t>;
        case NPY_STRING:
            if (numpyInType1 == numpyInType2 && scalarMode == SCALAR_MODE::NO_SCALARS)
            {
            }
            break;

        case NPY_UNICODE:
            if (numpyInType1 == numpyInType2 && scalarMode == SCALAR_MODE::NO_SCALARS)
            {
            }
            break;
        }
        return nullptr;

    case MATH_OPERATION::SUB:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_BOOL:
            return SimpleMathOpSlowSub<bool>;
        case NPY_FLOAT:
            return SimpleMathOpSlowSub<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowSub<double>;
        case NPY_LONGDOUBLE:
            return SimpleMathOpSlowSub<long double>;
        CASE_NPY_INT32:
            return SimpleMathOpSlowSub<int32_t>;
        CASE_NPY_INT64:

            return SimpleMathOpSlowSub<int64_t>;
        CASE_NPY_UINT32:
            return SimpleMathOpSlowSub<uint32_t>;
        CASE_NPY_UINT64:

            return SimpleMathOpSlowSub<uint64_t>;
        case NPY_INT8:
            return SimpleMathOpSlowSub<int8_t>;
        case NPY_INT16:
            return SimpleMathOpSlowSub<int16_t>;
        case NPY_UINT8:
            return SimpleMathOpSlowSub<uint8_t>;
        case NPY_UINT16:
            return SimpleMathOpSlowSub<uint16_t>;
        }
        return nullptr;

    case MATH_OPERATION::MUL:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        // case NPY_BOOL:   return SimpleMathOpSlowMul<bool>;
        case NPY_FLOAT:
            return SimpleMathOpSlowMul<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowMul<double>;
        case NPY_LONGDOUBLE:
            return SimpleMathOpSlowMul<long double>;
        CASE_NPY_INT32:
            return SimpleMathOpSlowMul<int32_t>;
        CASE_NPY_INT64:

            return SimpleMathOpSlowMul<int64_t>;
        CASE_NPY_UINT32:
            return SimpleMathOpSlowMul<uint32_t>;
        CASE_NPY_UINT64:

            return SimpleMathOpSlowMul<uint64_t>;
        case NPY_INT8:
            return SimpleMathOpSlowMul<int8_t>;
        case NPY_INT16:
            return SimpleMathOpSlowMul<int16_t>;
        case NPY_UINT8:
            return SimpleMathOpSlowMul<uint8_t>;
        case NPY_UINT16:
            return SimpleMathOpSlowMul<uint16_t>;
        }
        return nullptr;

    case MATH_OPERATION::DIV:
        *wantedOutType = NPY_DOUBLE;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
        {
            if (numpyInType2 == NPY_FLOAT)
                *wantedOutType = NPY_FLOAT;
        }
        else
        {
            if (numpyInType1 == NPY_FLOAT)
                *wantedOutType = NPY_FLOAT;
        }

        switch (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR ? numpyInType2 : numpyInType1)
        {
        case NPY_BOOL:
            return SimpleMathOpSlowDiv<bool>;
        case NPY_FLOAT:
            return SimpleMathOpSlowDivFloat<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowDiv<double>;
        case NPY_LONGDOUBLE:
            return SimpleMathOpSlowDiv<long double>;
        CASE_NPY_INT32:
            return SimpleMathOpSlowDiv<int32_t>;
        CASE_NPY_INT64:

            return SimpleMathOpSlowDiv<int64_t>;
        CASE_NPY_UINT32:
            return SimpleMathOpSlowDiv<uint32_t>;
        CASE_NPY_UINT64:

            return SimpleMathOpSlowDiv<uint64_t>;
        case NPY_INT8:
            return SimpleMathOpSlowDiv<int8_t>;
        case NPY_INT16:
            return SimpleMathOpSlowDiv<int16_t>;
        case NPY_UINT8:
            return SimpleMathOpSlowDiv<uint8_t>;
        case NPY_UINT16:
            return SimpleMathOpSlowDiv<uint16_t>;
        }
        return nullptr;

    case MATH_OPERATION::SUBDATETIMES:
        *wantedOutType = NPY_DOUBLE;
        switch (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR ? numpyInType2 : numpyInType1)
        {
        CASE_NPY_INT32:
            return SimpleMathOpSubDateTime<int32_t>;
        CASE_NPY_INT64:

            return SimpleMathOpSubDateTime<int64_t>;
        }
        printf("bad call to subdatetimes %d %d\n", numpyInType2, numpyInType1);
        return nullptr;

    case MATH_OPERATION::SUBDATES:
        *wantedOutType = NPY_INT32;
        switch (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR ? numpyInType2 : numpyInType1)
        {
        CASE_NPY_INT32:
            return SimpleMathOpSubDates<int32_t>;
        CASE_NPY_INT64:

            *wantedOutType = NPY_INT64;
            return SimpleMathOpSubDates<int64_t>;
        }
        printf("bad call to subdates %d %d\n", numpyInType2, numpyInType1);
        return nullptr;

    case MATH_OPERATION::FLOORDIV:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        // printf("checking floordiv type %d\n", *wantedOutType);
        switch (*wantedOutType)
        {
            // case NPY_BOOL:   return SimpleMathOpSlowMul<bool>;
        case NPY_FLOAT:
            return SimpleMathOpSlowFloorDiv<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowFloorDiv<double>;
        case NPY_LONGDOUBLE:
            return SimpleMathOpSlowFloorDiv<long double>;
        CASE_NPY_INT32:
            return SimpleMathOpSlowFloorDiv<int32_t>;
        CASE_NPY_INT64:

            return SimpleMathOpSlowFloorDiv<int64_t>;
        CASE_NPY_UINT32:
            return SimpleMathOpSlowFloorDiv<uint32_t>;
        CASE_NPY_UINT64:

            return SimpleMathOpSlowFloorDiv<uint64_t>;

#ifndef RT_COMPILER_CLANG // possible error with int8 array and
                          // np.floor_divide() with vextractps instruction
        case NPY_INT8:
            return SimpleMathOpSlowFloorDiv<int8_t>;
        case NPY_INT16:
            return SimpleMathOpSlowFloorDiv<int16_t>;
        case NPY_UINT8:
            return SimpleMathOpSlowFloorDiv<uint8_t>;
        case NPY_UINT16:
            return SimpleMathOpSlowFloorDiv<uint16_t>;
#endif
        }
        return nullptr;

    case MATH_OPERATION::MOD:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_BOOL:
            return SimpleMathOpSlowMod<bool>;
        case NPY_FLOAT:
            return SimpleMathOpSlowRemainder<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowRemainder<double>;
        case NPY_LONGDOUBLE:
            return SimpleMathOpSlowRemainder<long double>;
        CASE_NPY_INT32:
            return SimpleMathOpSlowMod<int32_t>;
        CASE_NPY_INT64:

            return SimpleMathOpSlowMod<int64_t>;
        CASE_NPY_UINT32:
            return SimpleMathOpSlowMod<uint32_t>;
        CASE_NPY_UINT64:

            return SimpleMathOpSlowMod<uint64_t>;
        case NPY_INT8:
            return SimpleMathOpSlowMod<int8_t>;
        case NPY_INT16:
            return SimpleMathOpSlowMod<int16_t>;
        case NPY_UINT8:
            return SimpleMathOpSlowMod<uint8_t>;
        case NPY_UINT16:
            return SimpleMathOpSlowMod<uint16_t>;
        }
        return nullptr;

    case MATH_OPERATION::POWER:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        // case NPY_BOOL:   return SimpleMathOpSlowPower<bool>;
        case NPY_FLOAT:
            return SimpleMathOpSlowPower<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowPower<double>;
        case NPY_LONGDOUBLE:
            return SimpleMathOpSlowPower<long double>;
        CASE_NPY_INT32:
            return SimpleMathOpSlowPower<int32_t>;
        CASE_NPY_INT64:

            return SimpleMathOpSlowPower<int64_t>;
        CASE_NPY_UINT32:
            return SimpleMathOpSlowPower<uint32_t>;
        CASE_NPY_UINT64:

            return SimpleMathOpSlowPower<uint64_t>;
        case NPY_INT8:
            return SimpleMathOpSlowPower<int8_t>;
        case NPY_INT16:
            return SimpleMathOpSlowPower<int16_t>;
        case NPY_UINT8:
            return SimpleMathOpSlowPower<uint8_t>;
        case NPY_UINT16:
            return SimpleMathOpSlowPower<uint16_t>;
        }
        return nullptr;

    case MATH_OPERATION::REMAINDER:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_FLOAT:
            return SimpleMathOpSlowRemainder<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowRemainder<double>;
        case NPY_LONGDOUBLE:
            return SimpleMathOpSlowRemainder<long double>;
        }
        return nullptr;

    case MATH_OPERATION::FMOD:
        *wantedOutType = numpyInType1;
        if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            *wantedOutType = numpyInType2;
        switch (*wantedOutType)
        {
        case NPY_FLOAT:
            return SimpleMathOpSlowFmod<float>;
        case NPY_DOUBLE:
            return SimpleMathOpSlowFmod<double>;
        case NPY_LONGDOUBLE:
            return SimpleMathOpSlowFmod<long double>;
        }
        return nullptr;
    }

    return nullptr;
}

//---------------------------------
// Removes confusing dtype nums in linux vs windows
// Converts dtype 7 to either 5 or 9
// Converts dtype 8 to either 6 or 10
// TODO: Replace with call to our FixupDtype() function?
static int GetNonAmbiguousDtype(int dtype)
{
    switch (dtype)
    {
    CASE_NPY_INT32:
        dtype = 5;
        break;
    CASE_NPY_UINT32:
        dtype = 6;
        break;
    CASE_NPY_INT64:

        dtype = 9;
        break;
    CASE_NPY_UINT64:

        dtype = 10;
        break;
    }
    return dtype;
}

//--------------------------------------------------------------------
// Check to see if we can compute this
static ANY_TWO_FUNC CheckMathOpTwoInputs(int func, int scalarMode, int numpyInType1, int numpyInType2, int numpyOutType,
                                         int * wantedOutType)
{
    ANY_TWO_FUNC pTwoFunc = nullptr;

    LOGGING("CheckMathTwo %d  scalar:%d  type1:%d  type2:%d\n", func, scalarMode, numpyInType1, numpyInType2);

    // Not handling complex, 128bit, voids, objects, or strings here
    if (numpyInType1 <= NPY_LONGDOUBLE && numpyInType2 <= NPY_LONGDOUBLE)
    {
        switch (func)
        {
        case MATH_OPERATION::ADD:
        case MATH_OPERATION::SUB:
        case MATH_OPERATION::MUL:
        case MATH_OPERATION::MOD:
        case MATH_OPERATION::MIN:
        case MATH_OPERATION::MAX:
        case MATH_OPERATION::DIV:
        case MATH_OPERATION::POWER:
        case MATH_OPERATION::REMAINDER: // comes in as mod???
        case MATH_OPERATION::FLOORDIV:
        case MATH_OPERATION::SUBDATETIMES:
        case MATH_OPERATION::SUBDATES:
            if (scalarMode == SCALAR_MODE::NO_SCALARS && numpyInType1 != numpyInType2)
            {
                // Additional check to see if dtypes really are different (this path is
                // rare)
                if (GetNonAmbiguousDtype(numpyInType1) != GetNonAmbiguousDtype(numpyInType2))
                {
                    LOGGING("basicmath types do not match %d %d\n", numpyInType1, numpyInType2);
                    break;
                }
            }

            // Try to get fast one first
            pTwoFunc = GetSimpleMathOpFast(func, scalarMode, numpyInType1, numpyInType2, numpyOutType, wantedOutType);
            if (pTwoFunc == nullptr)
            {
                pTwoFunc = GetSimpleMathOpSlow(func, scalarMode, numpyInType1, numpyInType2, numpyOutType, wantedOutType);
            }
            break;

        case MATH_OPERATION::CMP_EQ:
        case MATH_OPERATION::CMP_NE:
        case MATH_OPERATION::CMP_GT:
        case MATH_OPERATION::CMP_GTE:
        case MATH_OPERATION::CMP_LT:
        case MATH_OPERATION::CMP_LTE:
            pTwoFunc = GetComparisonOpFast(func, scalarMode, numpyInType1, numpyInType2, numpyOutType, wantedOutType);
            if (pTwoFunc == nullptr)
            {
                pTwoFunc = GetComparisonOpSlow(func, scalarMode, numpyInType1, numpyInType2, numpyOutType, wantedOutType);
            }
            break;

        case MATH_OPERATION::LOGICAL_AND:
        case MATH_OPERATION::LOGICAL_OR:
        case MATH_OPERATION::BITWISE_AND:
        case MATH_OPERATION::BITWISE_OR:
        case MATH_OPERATION::BITWISE_XOR:
        case MATH_OPERATION::BITWISE_ANDNOT:
        case MATH_OPERATION::BITWISE_NOTAND:
        case MATH_OPERATION::BITWISE_XOR_SPECIAL:
            if (scalarMode == SCALAR_MODE::NO_SCALARS && numpyInType1 != numpyInType2)
            {
                // Additional check to see if dtypes really are different (this path is
                // rare)
                if (GetNonAmbiguousDtype(numpyInType1) != GetNonAmbiguousDtype(numpyInType2))
                {
                    LOGGING("basicmath bitwise types do not match %d %d\n", numpyInType1, numpyInType2);
                    break;
                }
            }

            // Try to get fast one first
            pTwoFunc = GetSimpleMathOpFast(func, scalarMode, numpyInType1, numpyInType2, numpyOutType, wantedOutType);
            if (pTwoFunc == nullptr)
            {
                pTwoFunc = GetSimpleMathOpSlow(func, scalarMode, numpyInType1, numpyInType2, numpyOutType, wantedOutType);
            }
            break;
        }
    }
    else
    {
        LOGGING("input types are out of range %d %d\n", numpyInType1, numpyInType2);
    }

    return pTwoFunc;
}

static int g_signed_table[65] = {
    NPY_INT8,  NPY_INT8,  NPY_INT8,  NPY_INT8,  NPY_INT8,  NPY_INT8,  NPY_INT8,  NPY_INT8,  NPY_INT16, NPY_INT16, NPY_INT16,
    NPY_INT16, NPY_INT16, NPY_INT16, NPY_INT16, NPY_INT16, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32,
    NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT64,
    NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64,
    NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64,
    NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64
};

// the dtype is signed but the scalar has no sign
static int g_mixedsign_table[65] = {
    NPY_INT8,  NPY_INT8,  NPY_INT8,  NPY_INT8,  NPY_INT8,  NPY_INT8,   NPY_INT8,  NPY_INT8,  NPY_UINT8, NPY_INT16, NPY_INT16,
    NPY_INT16, NPY_INT16, NPY_INT16, NPY_INT16, NPY_INT16, NPY_UINT16, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32,
    NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32,  NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32, NPY_UINT32,
    NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64,  NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64,
    NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64,  NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64,
    NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64,  NPY_INT64, NPY_INT64, NPY_INT64, NPY_UINT64
};
static int g_bothunsigned_table[65] = {
    NPY_UINT8,  NPY_UINT8,  NPY_UINT8,  NPY_UINT8,  NPY_UINT8,  NPY_UINT8,  NPY_UINT8,  NPY_UINT8,  NPY_UINT8,  NPY_UINT16,
    NPY_UINT16, NPY_UINT16, NPY_UINT16, NPY_UINT16, NPY_UINT16, NPY_UINT16, NPY_UINT16, NPY_UINT32, NPY_UINT32, NPY_UINT32,
    NPY_UINT32, NPY_UINT32, NPY_UINT32, NPY_UINT32, NPY_UINT32, NPY_UINT32, NPY_UINT32, NPY_UINT32, NPY_UINT32, NPY_UINT32,
    NPY_UINT32, NPY_UINT32, NPY_UINT32, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64,
    NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64,
    NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64,
    NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64, NPY_UINT64
};

//--------------------------------------------------------------------
class CTwoInputs
{
public:
    CTwoInputs(int numpyOutType1, int funcNumber1)
    {
        numpyOutType = numpyOutType1;
        funcNumber = funcNumber1;
    }

    ~CTwoInputs()
    {
        if (toDelete1)
        {
            Py_DecRef(toDelete1);
        }
        if (toDelete2)
        {
            Py_DecRef(toDelete2);
        }
        if (toDelete3)
        {
            Py_DecRef(toDelete3);
        }
        if (toDelete4)
        {
            Py_DecRef(toDelete4);
        }
    }

    //-------------------------------------------------------------------
    // Called when there is scalar which might be too large for the array
    // For instance a large number with an array of int32s
    // returns a recommended dtype which the array can then be upcasted to
    //
    // TOOO: What about string to unicode or numpy scalar string
    int PossiblyUpcast(PyObject * scalarObject, int dtype)
    {
        if (PyLong_Check(scalarObject))
        {
            // If both bools, we are ok
            if (dtype == NPY_BOOL && PyBool_Check(scalarObject))
            {
                return dtype;
            }

            // check if the array is also integer type
            if (dtype <= NPY_ULONGLONG)
            {
                // this path includes bools
                int overflow = 0;
                int64_t val = PyLong_AsLongLongAndOverflow(scalarObject, &overflow);
                int64_t log2 = 64;
                if (! overflow)
                {
                    int issigned = 0;
                    if (val < 0)
                    {
                        issigned = 1;
                        // Handle case of -128 which can exist in an int8_t
                        val = -(val + 1);
                    }
                    // Check for 0 due to builtin_clz issue
                    log2 = (val != 0) ? 64 - lzcnt_64(val) : 0;

                    // Find the smallest dtype that can be used to represent the number
                    int scalar_dtype;
                    if (issigned)
                    {
                        if (dtype > NPY_BOOL && (dtype & 1) == 0)
                        {
                            // dtype is unsigned and scalar is signed
                            scalar_dtype = g_mixedsign_table[log2];
                        }
                        else
                        {
                            // both are signed
                            scalar_dtype = g_signed_table[log2];
                        }
                    }
                    else
                    {
                        if (dtype > NPY_BOOL && (dtype & 1) == 0)
                        {
                            // if both are unsigned
                            scalar_dtype = g_bothunsigned_table[log2];
                        }
                        else
                        {
                            // dtype is signed, scalar is positive
                            scalar_dtype = g_signed_table[log2];
                        }
                    }
                    int convertType1 = dtype;
                    int convertType2 = scalar_dtype;
                    bool result = GetUpcastType(dtype, scalar_dtype, convertType1, convertType2, funcNumber);
                    LOGGING("Getting upcast %d %d, result %d  convert:%d\n", dtype, scalar_dtype, result, convertType1);
                    if (result)
                    {
                        return convertType1;
                    }
                }
                else
                {
                    // if we have overflow then only ulonglong can handle, else float64
                    LOGGING("overflow dtype %d\n", dtype);
                    switch (dtype)
                    {
                        CASE_NPY_UINT64:
                            break;
                        default:
                            return NPY_FLOAT64;
                    }
                }
            }
        }
        else if (PyFloat_Check(scalarObject))
        {
            if (dtype <= NPY_ULONGLONG)
            {
                return NPY_FLOAT64;
            }
        }
        else if (PyUnicode_Check(scalarObject))
        {
            return NPY_UNICODE;
        }
        else if (PyBytes_Check(scalarObject))
        {
            return NPY_STRING;
        }
        // Check if they passed a numpy scalar
        else if (PyArray_IsScalar(scalarObject, Generic))
        {
            PyArray_Descr * pDescr = PyArray_DescrFromScalar(scalarObject);
            int scalarType = pDescr->type_num;
            Py_DECREF(pDescr);

            if (scalarType == NPY_OBJECT)
            {
                // TODO: extract int from object?
                LOGGING("Got object in possibly upcast\n");
            }
            if (scalarType != dtype && scalarType <= NPY_LONGDOUBLE)
            {
                LOGGING("scalars dont match %d %d\n", scalarType, dtype);
                int convertType1 = dtype;
                int convertType2 = scalarType;
                bool result = GetUpcastType(dtype, scalarType, convertType1, convertType2, funcNumber);
                if (result)
                {
                    return convertType1;
                }
            }
        }
        // return same type
        return dtype;
    }

    //-------------------------------------------------------------------
    //
    // returns nullptr on failure
    PyObject * PossiblyConvertString(int newType, int oldType, PyObject * inObject)
    {
        if (newType == NPY_UNICODE && oldType == NPY_STRING)
        {
            if (PyUnicode_Check(inObject))
            {
                inObject = PyUnicode_AsUTF8String(inObject);
                if (inObject)
                {
                    return inObject;
                }
            }
        }
        else if (newType == NPY_STRING && oldType == NPY_UNICODE)
        {
            if (PyBytes_Check(inObject))
            {
                inObject = PyUnicode_FromObject(inObject);
                if (inObject)
                {
                    return inObject;
                }
            }
        }
        return nullptr;
    }

    //-------------------------------------------------------------------
    // TODO: create struct
    void inline FillArray1(PyObject * array1)
    {
        inArr = (PyArrayObject *)array1;
        itemSize1 = PyArray_ITEMSIZE(inArr);
        numpyInType1 = PyArray_TYPE(inArr);
        ndim1 = PyArray_NDIM(inArr);
        dims1 = PyArray_DIMS(inArr);
        pDataIn1 = PyArray_BYTES(inArr);
        len1 = CALC_ARRAY_LENGTH(ndim1, dims1);
        flags1 = PyArray_FLAGS(inArr);
    }

    void inline FillArray2(PyObject * array2)
    {
        inArr2 = (PyArrayObject *)array2;
        itemSize2 = PyArray_ITEMSIZE(inArr2);
        numpyInType2 = PyArray_TYPE(inArr2);
        ndim2 = PyArray_NDIM(inArr2);
        dims2 = PyArray_DIMS(inArr2);
        pDataIn2 = PyArray_BYTES(inArr2);
        len2 = CALC_ARRAY_LENGTH(ndim2, dims2);
        flags2 = PyArray_FLAGS(inArr2);
    }

    //-------------------------------------------------------------------
    // Arg1: defaultScalarType currently not used
    // returns false on failure
    bool CheckInputs(PyObject * args, int defaultScalarType, int64_t funcNumber)
    {
        if (! PyTuple_CheckExact(args))
        {
            PyErr_Format(PyExc_ValueError, "BasicMath arguments needs to be a tuple");
            return false;
        }

        tupleSize = Py_SIZE(args);

        if (tupleSize != (expectedTupleSize - 1) && tupleSize != expectedTupleSize)
        {
            PyErr_Format(PyExc_ValueError, "BasicMath only takes two or three arguments instead of %llu args", tupleSize);
            return false;
        }

        // PyTuple_GetItem will not increment the ref count
        inObject1 = PyTuple_GET_ITEM(args, 0);
        inObject2 = PyTuple_GET_ITEM(args, 1);

        //// For three inputs, the third inputs cannot be a scalar and must be same
        /// array length and type
        // if (expectedTupleSize == 4) {
        //   // Possibly check array type here
        //   pDataIn3 = PyArray_BYTES((PyArrayObject*)PyTuple_GET_ITEM(args, 2), 0);
        //}

        if (tupleSize == expectedTupleSize)
        {
            // TODO: Get the third item, check if numpy array, if so use as output
            outputObject = PyTuple_GetItem(args, (expectedTupleSize - 1));
        }
        return CheckInputsInternal(defaultScalarType, funcNumber);
    }

    // Called from low level __add__ hook
    // If output is set, it came from an inplace operation and the ref count does
    // not need to be incremented
    bool CheckInputs(PyObject * input1, PyObject * input2, PyObject * output, int64_t funcNumber)
    {
        tupleSize = 2;
        inObject1 = input1;
        inObject2 = input2;
        if (output != nullptr)
        {
            outputObject = output;
            // This will force the output type
            numpyOutType = PyArray_TYPE((PyArrayObject *)outputObject);
        }

        return CheckInputsInternal(0, funcNumber);
    }

    //----------------------------------------
    // Input: numpyInType1 and numpyInType2 must be set
    // Returns false on error
    //
    bool CheckUpcast()
    {
        // Check if we need to upcast one or both of the arrays
        // bool, NPY_INT8, NPY_UINT8
        if (numpyInType1 != numpyInType2)
        {
            // Remove the ambiguous type
            if (numpyInType1 == 7 || numpyInType1 == 8)
            {
                if (sizeof(long) == 4)
                {
                    numpyInType1 -= 2;
                }
                else
                {
                    numpyInType1 += 2;
                }
            }
            if (numpyInType2 == 7 || numpyInType2 == 8)
            {
                if (sizeof(long) == 4)
                {
                    numpyInType2 -= 2;
                }
                else
                {
                    numpyInType2 += 2;
                }
            }
            if (numpyInType1 != numpyInType2)
            {
                int convertType1 = numpyInType1;
                int convertType2 = numpyInType2;
                bool result = GetUpcastType(numpyInType1, numpyInType2, convertType1, convertType2, funcNumber);
                if (result)
                {
                    if (numpyInType1 != convertType1 && convertType1 <= NPY_LONGDOUBLE)
                    {
                        // Convert
                        LOGGING("converting array1 to %d  from  %d and %d\n", convertType1, numpyInType1, numpyInType2);
                        PyObject * newarray1 = ConvertSafeInternal(inArr, convertType1);
                        if (! newarray1)
                        {
                            // Failed to convert
                            return false;
                        }
                        toDelete3 = newarray1;
                        FillArray1(newarray1);
                    }
                    if (numpyInType2 != convertType2 && convertType2 <= NPY_LONGDOUBLE)
                    {
                        // Convert
                        LOGGING("converting array2 to %d  from  %d and %d\n", convertType2, numpyInType2, numpyInType1);
                        PyObject * newarray2 = ConvertSafeInternal(inArr2, convertType2);
                        if (! newarray2)
                        {
                            // Failed to convert
                            return false;
                        }
                        toDelete4 = newarray2;
                        FillArray2(newarray2);
                    }
                }
                else
                {
                    LOGGING("BasicMath cannot upcast %d vs %d\n", numpyInType1, numpyInType2);
                    return false;
                }
            }
        }
        return true;
    }

    // Common routine from array_ufunc or lower level slot hook
    // inObject1 is set and will get analyzed
    // inObject2 is set and will get analyzed
    // funcNumber is passed because certain comparisons with int64_t <-> uint64_t
    // dont need to get upcast to float64
    bool CheckInputsInternal(int defaultScalarType, int64_t funcNumber)
    {
        bool isArray1 = IsFastArrayOrNumpy((PyArrayObject *)inObject1);
        bool isArray2 = IsFastArrayOrNumpy((PyArrayObject *)inObject2);
        isScalar1 = ! isArray1;
        isScalar2 = ! isArray2;

        if (! isArray1)
        {
            isScalar1 = PyArray_IsAnyScalar(inObject1);
            if (! isScalar1)
            {
                LOGGING("Converting object1 to array %s\n", inObject1->ob_type->tp_name);
                // Convert to temp array that must be deleted
                toDelete1 = inObject1 = PyArray_FromAny(inObject1, nullptr, 0, 0, NPY_ARRAY_ENSUREARRAY, nullptr);
                if (inObject1 == nullptr)
                {
                    return false;
                }
                isArray1 = true;
                isScalar1 = false;
            }
            else if (isArray2 && ! outputObject)
            {
                //
                int oldType = PyArray_TYPE((PyArrayObject *)inObject2);
                int newType = PossiblyUpcast(inObject1, oldType);

                // Cannot convert string to unicode yet
                if (newType != oldType)
                {
                    if (newType <= NPY_LONGDOUBLE)
                    {
                        // Convert second array - upcast
                        LOGGING("Converting because of scalar1 to %d from %d   in1:%s\n", newType, oldType,
                                inObject1->ob_type->tp_name);
                        PyObject * newarray2 = ConvertSafeInternal((PyArrayObject *)inObject2, newType);
                        if (! newarray2)
                        {
                            // Failed to convert
                            return false;
                        }
                        // remember to delete temp array
                        toDelete1 = inObject2 = newarray2;
                    }
                    else
                    {
                        // Handle conversion from string to unicode here
                        toDelete1 = PossiblyConvertString(newType, oldType, inObject2);
                        if (! toDelete1)
                            return false;
                        inObject2 = toDelete1;
                        return false; // until we handle string compares
                    }
                }
            }
        }

        if (! isArray2)
        {
            isScalar2 = PyArray_IsAnyScalar(inObject2);
            if (! isScalar2)
            {
                LOGGING("Converting object2 to array %s\n", inObject2->ob_type->tp_name);
                toDelete2 = inObject2 = PyArray_FromAny(inObject2, nullptr, 0, 0, NPY_ARRAY_ENSUREARRAY, nullptr);
                if (inObject2 == nullptr)
                {
                    return false;
                }
                isArray2 = true;
                isScalar2 = false;
            }
            else if (isArray1 && ! outputObject)
            {
                //
                int oldType = PyArray_TYPE((PyArrayObject *)inObject1);
                int newType = PossiblyUpcast(inObject2, oldType);
                if (newType != oldType)
                {
                    if (newType <= NPY_LONGDOUBLE)
                    {
                        // Convert first array - upcast
                        LOGGING("Converting because of scalar2 to %d from %d\n", newType, oldType);
                        PyObject * newarray1 = ConvertSafeInternal((PyArrayObject *)inObject1, newType);
                        if (! newarray1)
                        {
                            // Failed to convert
                            return false;
                        }
                        // remember to delete temp array
                        toDelete2 = inObject1 = newarray1;
                    }
                    else
                    {
                        // Handle conversion from string to unicode here
                        toDelete2 = PossiblyConvertString(newType, oldType, inObject1);
                        if (! toDelete2)
                            return false;
                        inObject1 = toDelete2;
                        return false;
                    }
                }
            }
        }

        LOGGING("scalar: %d %d\n", isScalar1, isScalar2);

        if (isScalar1)
        {
            scalarMode = SCALAR_MODE::FIRST_ARG_SCALAR;
            if (! isScalar2)
            {
                // for most basic math, the two inputs must be same dtype, so check
                // other dtype
                inArr2 = (PyArrayObject *)inObject2;
                numpyInType2 = PyArray_TYPE(inArr2);
            }
            else
            {
                // case when both are scalars, so use output type
                // this case should not occur
                numpyInType2 = numpyOutType;
            }

            // Check the scalar object and consider upcasting the other array
            // For example, the scalar is a float and the second array is integer
            // Or the scalar is a larger integer and the second array is a small
            // integer, or unisgned

            // pDataIn1 will be set pointing to 256 bit value
            bool bResult = ConvertScalarObject(inObject1, &m256Scalar1, numpyInType2, &pDataIn1, &itemSize1);

            LOGGING(
                "Converting first scalar type to %d, bresult: %d,  value: %lld  "
                "inObject1: %s\n",
                numpyInType2, bResult, *(int64_t *)&m256Scalar1, inObject1->ob_type->tp_name);

            // TJD added this Dec 11, 2018
            numpyInType1 = numpyInType2;

            if (! bResult)
                return false;
        }
        else
        {
            // We know the first object is an array
            FillArray1(inObject1);
            // Check strides
            if (ndim1 > 0 && itemSize1 != PyArray_STRIDE((PyArrayObject *)inArr, 0))
            {
                return false;
            }

            //// check for a scalar passed as array of 1 item
            // if (len1 == 1) {
            //   if (!isScalar2) {
            //      LOGGING("setting to first arg scalar\n");
            //      scalarMode = SCALAR_MODE::FIRST_ARG_SCALAR;

            //      // NOTE: does not handle single array of bytes/unicode
            //      bool bResult =
            //         ConvertSingleItemArray(pDataIn1, GetArrDType(inArr),
            //         &m256Scalar1, numpyInType2);

            //      if (!bResult) return false;
            //      pDataIn1 = &m256Scalar1;
            //   }
            //}
        }

        if (isScalar2)
        {
            if (scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            {
                // TJD: new Dec 2018, allow both scalar
                scalarMode = SCALAR_MODE::BOTH_SCALAR;
            }
            else
            {
                LOGGING("setting to second arg scalar\n");
                scalarMode = SCALAR_MODE::SECOND_ARG_SCALAR;
            }

            numpyInType2 = numpyInType1;

            // pDataIn2 will be set pointing to 256 bit value
            bool bResult = ConvertScalarObject(inObject2, &m256Scalar2, numpyInType1, &pDataIn2, &itemSize2);

            LOGGING(
                "Converting second scalar type to %d, bresult: %d,  value: %lld  "
                "type: %s\n",
                numpyInType1, bResult, *(int64_t *)&m256Scalar2, inObject2->ob_type->tp_name);

            if (! bResult)
                return false;
        }
        else
        {
            // We know the second object is an array
            FillArray2(inObject2);

            // Check strides
            if (ndim2 > 0 && itemSize2 != PyArray_STRIDE((PyArrayObject *)inArr2, 0))
            {
                return false;
            }

            //// check for a scalar passed as array of 1 item
            // if (len2 == 1) {
            //   if (scalarMode != SCALAR_MODE::FIRST_ARG_SCALAR) {
            //      scalarMode = SCALAR_MODE::SECOND_ARG_SCALAR;

            //      // broadcastable check  -- we always checkd for 1?
            //      bool bResult =
            //         ConvertSingleItemArray(pDataIn1, GetArrDType(inArr),
            //         &m256Scalar1, numpyInType2);

            //      if (!bResult) return false;
            //      pDataIn1 = &m256Scalar1;
            //   }
            //}
        }

        // If both are arrays, need to be same size
        if (scalarMode == SCALAR_MODE::NO_SCALARS)
        {
            // Check for single length array
            // Check for np.where condition, the length is known

            if (len1 != len2)
            {
                // if ndim ==0, then first argument was likely a scalar numpy array
                if (len1 == 1 && len2 >= 1)
                {
                    // First object is a scalar
                    scalarMode = SCALAR_MODE::FIRST_ARG_SCALAR;

                    // Check if we can upcast the scalar or if the scalar causes the array
                    // to upcast
                    numpyInType1 = GetArrDType(inArr);
                    int convertType1 = numpyInType1;
                    int convertType2 = numpyInType2;
                    bool result = GetUpcastType(numpyInType1, numpyInType2, convertType1, convertType2, funcNumber);
                    if (result)
                    {
                        LOGGING("setting BM to first arg scalar from %d to %d.  ndim1: %d\n", numpyInType1, convertType1, ndim1);
                        // Convert
                        bool bResult = ConvertSingleItemArray(pDataIn1, numpyInType1, &m256Scalar1, convertType1);

                        if (! bResult)
                            return false;
                        pDataIn1 = &m256Scalar1;
                        numpyInType1 = convertType1;

                        if (numpyInType2 != convertType2 && convertType2 <= NPY_LONGDOUBLE)
                        {
                            // Convert
                            LOGGING("converting array2 to %d  from  %d and %d\n", convertType2, numpyInType2, numpyInType1);
                            PyObject * newarray2 = ConvertSafeInternal(inArr2, convertType2);
                            if (! newarray2)
                            {
                                // Failed to convert
                                return false;
                            }
                            toDelete4 = newarray2;
                            FillArray2(newarray2);
                        }
                    }
                    else
                    {
                        return false;
                    }
                }
                else if (len2 == 1 && len1 >= 1)
                {
                    // Second object is a scalar
                    scalarMode = SCALAR_MODE::SECOND_ARG_SCALAR;

                    // Check if we can upcast the scalar or if the scalar causes the array
                    // to upcast
                    numpyInType2 = GetArrDType(inArr2);
                    int convertType1 = numpyInType1;
                    int convertType2 = numpyInType2;
                    bool result = GetUpcastType(numpyInType1, numpyInType2, convertType1, convertType2, funcNumber);
                    if (result)
                    {
                        LOGGING("setting BM to second arg scalar from %d to %d.  ndim2: %d\n", numpyInType2, convertType2, ndim2);

                        // Convert
                        bool bResult = ConvertSingleItemArray(pDataIn2, numpyInType2, &m256Scalar2, convertType2);

                        if (! bResult)
                            return false;
                        pDataIn2 = &m256Scalar2;
                        numpyInType2 = convertType2;

                        if (numpyInType1 != convertType1 && convertType1 <= NPY_LONGDOUBLE)
                        {
                            // Convert
                            LOGGING("converting array1 to %d  from  %d and %d\n", convertType1, numpyInType1, numpyInType2);
                            PyObject * newarray1 = ConvertSafeInternal(inArr, convertType1);
                            if (! newarray1)
                            {
                                // Failed to convert
                                return false;
                            }
                            toDelete2 = newarray1;
                            FillArray1(newarray1);
                        }
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    LOGGING(
                        "BasicMath needs matching 1 dim x 1 dim array   %llu x %llu. "
                        " Will punt to numpy.\n",
                        len1, len2);
                    // PyErr_Format(PyExc_ValueError, "GenericAnyTwoFunc needs 1 dim x 1
                    // dim array   %d x %d", len1, len2);
                    return false;
                }
            }
            else
            {
                // Check if we need to upcast one or both of the arrays
                // bool, NPY_INT8, NPY_UINT8
                if (! CheckUpcast())
                    return false;
            }
        }

        return true;
    }

    //-----------------------------------------------------------------------
    // returns nullptr on failure
    // will ALLOCATE or use an existing numpy array
    PyArrayObject * CheckOutputs(int wantOutType)
    {
        LOGGING("Checking output type %d   %lld %lld\n", wantOutType, tupleSize, expectedTupleSize);

        // TJD: Possibly make this dynamic?
        // If the number of args is 3, the third argument can be considered the
        // output argument outputObject will be set if came from inplace opeation
        // like x += 3

        if (outputObject)
        {
            if (! PyArray_Check(outputObject))
            {
                LOGGING("BasicMath needs third argument to be an array\n");
                PyErr_Format(PyExc_ValueError, "BasicMath needs third argument to be an array");
                return nullptr;
            }

            if (PyArray_TYPE((PyArrayObject *)outputObject) != wantOutType)
            {
                LOGGING("BasicMath does not match output type %d vs %d\n", PyArray_TYPE((PyArrayObject *)outputObject),
                        wantOutType);
                // PyErr_Format(PyExc_ValueError, "BasicMath does not match output type
                // %d vs %d", PyArray_TYPE((PyArrayObject*)outputObject), wantOutType);
                // NOTE: Now handled in FastArray where we will copy result back into
                // array
                outputObject = nullptr;
            }
            else
            {
                int64_t len3 = ArrayLength((PyArrayObject *)outputObject);
                LOGGING("oref %llu  len1:%lld  len2:%lld  len3: %lld\n", outputObject->ob_refcnt, len1, len2, len3);
                if (len3 < len2 || len3 < len1)
                {
                    // something wrong with output size (gonna crash)
                    return nullptr;
                }

                if (PyArray_ITEMSIZE((PyArrayObject *)outputObject) != PyArray_STRIDE((PyArrayObject *)outputObject, 0))
                {
                    // output array is strided
                    LOGGING("Punting because output array is strided\n");
                    return nullptr;
                }

                // TJD: since code like this is possible
                // z=rc.addi32(a,b,c)
                // Now z and c point to the same object, thus we must increment the ref
                // count
                Py_INCREF(outputObject);
            }
        }

        // Check if they specified their own output array
        // NOTE: the output array is specified for += operations (in place
        // operations)
        if (outputObject)
        {
            LOGGING("Going to use existing output array %lld %lld\n", len1, len2);
            returnObject = (PyArrayObject *)outputObject;

            // Check to punt on when shifted data (overlapped)
            // a[1:] -= a[:-1]
            char * pOutput = PyArray_BYTES(returnObject);
            char * pOutputEnd = pOutput + ArrayLength(returnObject) * PyArray_ITEMSIZE(returnObject);
            bool bOverlapProblem = false;

            if (len1)
            {
                if (pDataIn1 > pOutput && pDataIn1 < pOutputEnd)
                {
                    printf(
                        "!!!warning: inplace operation has memory overlap in "
                        "beginning for input1\n");
                    bOverlapProblem = true;
                }
                char * pTempEnd = (char *)pDataIn1 + (len1 * itemSize1);
                if (pTempEnd > pOutput && pTempEnd < pOutputEnd)
                {
                    printf(
                        "!!!warning: inplace operation has memory overlap at end for "
                        "input1\n");
                    bOverlapProblem = true;
                }
            }
            if (len2)
            {
                if (pDataIn2 > pOutput && pDataIn2 < pOutputEnd)
                {
                    printf(
                        "!!!warning: inplace operation has memory overlap in "
                        "beginning for input2\n");
                    bOverlapProblem = true;
                }
                char * pTempEnd = (char *)pDataIn2 + (len2 * itemSize2);
                if (pTempEnd > pOutput && pTempEnd < pOutputEnd)
                {
                    printf(
                        "!!!warning: inplace operation has memory overlap at end for "
                        "input2\n");
                    bOverlapProblem = true;
                }
            }

            if (bOverlapProblem)
            {
                return nullptr;
            }
        }
        else
        {
            LOGGING("Going to allocate type %d  %lld  %lld  %d %d  %p  %p\n", wantOutType, len1, len2, ndim1, ndim2, dims1, dims2);

            // TODO: when handling 2 dim arrays that are contiguous and same ORDER,
            // allocate an array with the flags for order 'F' or 'C'
            if (len1 <= 0 && len2 <= 0)
            {
                // Handle empty
                npy_intp dimensions[1] = { 0 };
                returnObject = AllocateNumpyArray(1, dimensions, wantOutType);
            }
            else
                // Check for functions that always output a boolean
                if (len1 >= len2)
            {
                LOGGING("Going to allocate for len1 type %d  %lld  %lld  %d %lld\n", wantOutType, len1, len2, ndim1,
                        dims1 ? (int64_t)dims1[0] : 0);
                if (ndim1 > 1)
                {
                    returnObject = AllocateNumpyArray(ndim1, dims1, wantOutType, 0,
                                                      (flags1 & flags2 & NPY_ARRAY_F_CONTIGUOUS) == NPY_ARRAY_F_CONTIGUOUS);
                }
                else
                {
                    returnObject = AllocateNumpyArray(ndim1, dims1, wantOutType);
                }
            }
            else
            {
                LOGGING("Going to allocate for len2 type %d  %lld  %lld  %d %lld\n", wantOutType, len1, len2, ndim2,
                        dims2 ? (int64_t)dims2[0] : 0);
                if (ndim2 > 1)
                {
                    returnObject = AllocateNumpyArray(ndim2, dims2, wantOutType, 0,
                                                      (flags1 & flags2 & NPY_ARRAY_F_CONTIGUOUS) == NPY_ARRAY_F_CONTIGUOUS);
                }
                else
                {
                    returnObject = AllocateNumpyArray(ndim2, dims2, wantOutType);
                }
            }
            CHECK_MEMORY_ERROR(returnObject);
        }

        if (returnObject == nullptr)
        {
            LOGGING("!!! BasicMath has no output\n");
            PyErr_Format(PyExc_ValueError, "BasicMath has no output (possibly out of memory)");
            return nullptr;
        }

        LOGGING("Len1 %llu   len2  %llu  scalarmode:%d\n", len1, len2, scalarMode);
        return returnObject;
    }

    //---------------------------------------------------------------------
    // Data members
    int numpyOutType; // input as first parameter
    int funcNumber;   // input as second parameter

    _m256all m256Scalar1; // possibly store scalar as 256bit
    _m256all m256Scalar2;

    // Default to 0 dimensions, which is a scalar
    int ndim1 = 0;
    int ndim2 = 0;
    int numpyInType1 = 0;
    int numpyInType2 = 0;
    npy_intp * dims1 = 0;
    npy_intp * dims2 = 0;
    int64_t itemSize1 = 0;
    int64_t itemSize2 = 0;
    // For 2 or 3 dim arrays, should be able to calculate length
    int64_t len1 = 0;
    int64_t len2 = 0;
    // flags are ANDED together to see if still F contiguous for > 1 dim output
    // arrays
    int flags1 = NPY_ARRAY_F_CONTIGUOUS;
    int flags2 = NPY_ARRAY_F_CONTIGUOUS;

    void * pDataIn1 = nullptr;
    void * pDataIn2 = nullptr;

    // PyTuple_GetItem will not increment the ref count
    PyObject * inObject1 = nullptr;
    PyObject * inObject2 = nullptr;

    PyObject * toDelete1 = nullptr;
    PyObject * toDelete2 = nullptr;
    PyObject * toDelete3 = nullptr;
    PyObject * toDelete4 = nullptr;

    // often the same as inObject1/2
    PyArrayObject * inArr = nullptr;
    PyArrayObject * inArr2 = nullptr;

    PyArrayObject * returnObject = nullptr;
    PyObject * outputObject = nullptr;

    int32_t scalarMode = SCALAR_MODE::NO_SCALARS;
    bool isScalar1 = false;
    bool isScalar2 = false;

    Py_ssize_t tupleSize = 0;

    // Change this value to 4 for three inputs
    Py_ssize_t expectedTupleSize = 3;

    // For "where" the array size is already known
    // Set the expected length if known ahead of time
    int64_t expectedLength = 0;
};

OLD_CALLBACK bmOldCallback;

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static bool BasicMathThreadCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    bool didSomeWork = false;
    OLD_CALLBACK * OldCallback = (OLD_CALLBACK *)pstWorkerItem->WorkCallbackArg;

    int64_t typeSizeIn = OldCallback->FunctionList->InputItemSize;
    int64_t typeSizeOut = OldCallback->FunctionList->OutputItemSize;

    char * pDataInX = (char *)OldCallback->pDataInBase1;
    char * pDataInX2 = (char *)OldCallback->pDataInBase2;
    char * pDataOutX = (char *)OldCallback->pDataOutBase1;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
    {
        // Calculate how much to adjust the pointers to get to the data for this
        // work block
        int64_t offsetAdj = pstWorkerItem->BlockSize * workBlock * typeSizeIn;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * typeSizeOut;
        // int64_t outputAdj = offsetAdj;

        THREADLOGGING("workblock %llu   len=%llu  offset=%llu \n", workBlock, lenX, offsetAdj);
        switch (OldCallback->ScalarMode)
        {
        case NO_SCALARS:
            // Process this block of work
            OldCallback->FunctionList->AnyTwoStubCall(pDataInX + offsetAdj, pDataInX2 + offsetAdj, pDataOutX + outputAdj, lenX,
                                                      OldCallback->ScalarMode);
            break;

        case FIRST_ARG_SCALAR:
            // Process this block of work
            OldCallback->FunctionList->AnyTwoStubCall(pDataInX, pDataInX2 + offsetAdj, pDataOutX + outputAdj, lenX,
                                                      OldCallback->ScalarMode);
            break;

        case SECOND_ARG_SCALAR:
            // Process this block of work
            OldCallback->FunctionList->AnyTwoStubCall(pDataInX + offsetAdj, pDataInX2, pDataOutX + outputAdj, lenX,
                                                      OldCallback->ScalarMode);
            break;

        case BOTH_SCALAR:
            printf("** bug both are scalar!\n");
            // Process this block of work
            // FunctionList->AnyTwoStubCall(pDataInX, pDataInX2, pDataOutX +
            // outputAdj, lenX, ScalarMode);
            break;
        }

        // Indicate we completed a block
        didSomeWork = true;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        // printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}

//------------------------------------------------------------------------------
//
void WorkTwoStubCall(FUNCTION_LIST * anyTwoStubCall, void * pDataIn, void * pDataIn2, void * pDataOut, int64_t len,
                     int32_t scalarMode)
{
    // printf("worktwostub %d\n", scalarMode);

    // Check if we can use worker threads
    stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(len);

    if (pWorkItem == nullptr)
    {
        // Threading not allowed for this work item, call it directly from main
        // thread
        anyTwoStubCall->AnyTwoStubCall(pDataIn, pDataIn2, pDataOut, len, scalarMode);
        return;
    }

    // Each thread will call this routine with the callbackArg
    pWorkItem->DoWorkCallback = BasicMathThreadCallback;
    pWorkItem->WorkCallbackArg = &bmOldCallback;

    // Store the base values use in a work item
    bmOldCallback.FunctionList = anyTwoStubCall;
    bmOldCallback.pDataInBase1 = pDataIn;
    bmOldCallback.pDataInBase2 = pDataIn2;
    bmOldCallback.pDataOutBase1 = pDataOut;
    bmOldCallback.ScalarMode = scalarMode;

    // This will notify the worker threads of a new work item
    g_cMathWorker->WorkMain(pWorkItem, len, 0);
}

//-----------------------------------------------------------------------------------
// Called when two strings are added
//
PyObject * ConcatTwoStrings(int32_t scalarMode, char * pStr1, char * pStr2, int64_t itemSize1, int64_t itemSize2, int64_t length)
{
    int64_t itemSizeOut = itemSize1 + itemSize2;

    // printf("concat two strings %lld %lld\n", itemSize1, itemSize2);

    PyArrayObject * pNewString = AllocateNumpyArray(1, (npy_intp *)&length, NPY_STRING, itemSizeOut);
    CHECK_MEMORY_ERROR(pNewString);

    if (pNewString)
    {
        char * pOut = (char *)PyArray_BYTES(pNewString);

        switch (scalarMode)
        {
        case SCALAR_MODE::NO_SCALARS:

            for (int64_t i = 0; i < length; i++)
            {
                char * pOut1 = pOut + (itemSizeOut * i);
                char * pEndOut1 = pOut1 + itemSizeOut;
                char * pIn1 = pStr1 + (itemSize1 * i);
                char * pEnd1 = pIn1 + itemSize1;
                char * pIn2 = pStr2 + (itemSize2 * i);
                char * pEnd2 = pIn2 + itemSize2;

                while (pIn1 < pEnd1 && *pIn1)
                {
                    *pOut1++ = *pIn1++;
                }
                while (pIn2 < pEnd2 && *pIn2)
                {
                    *pOut1++ = *pIn2++;
                }
                // add 0s at end
                while (pOut1 < pEndOut1)
                {
                    *pOut1++ = 0;
                }
            }
            break;

        case SCALAR_MODE::FIRST_ARG_SCALAR:
            for (int64_t i = 0; i < length; i++)
            {
                char * pOut1 = pOut + (itemSizeOut * i);
                char * pEndOut1 = pOut1 + itemSizeOut;
                char * pIn1 = pStr1;
                char * pEnd1 = pIn1 + itemSize1;
                char * pIn2 = pStr2 + (itemSize2 * i);
                char * pEnd2 = pIn2 + itemSize2;

                while (pIn1 < pEnd1 && *pIn1)
                {
                    *pOut1++ = *pIn1++;
                }
                while (pIn2 < pEnd2 && *pIn2)
                {
                    *pOut1++ = *pIn2++;
                }
                // add 0s at end
                while (pOut1 < pEndOut1)
                {
                    *pOut1++ = 0;
                }
            }
            break;

        case SCALAR_MODE::SECOND_ARG_SCALAR:
            for (int64_t i = 0; i < length; i++)
            {
                char * pOut1 = pOut + (itemSizeOut * i);
                char * pEndOut1 = pOut1 + itemSizeOut;
                char * pIn1 = pStr1 + (itemSize1 * i);
                char * pEnd1 = pIn1 + itemSize1;
                char * pIn2 = pStr2;
                char * pEnd2 = pIn2 + itemSize2;

                while (pIn1 < pEnd1 && *pIn1)
                {
                    *pOut1++ = *pIn1++;
                }
                while (pIn2 < pEnd2 && *pIn2)
                {
                    *pOut1++ = *pIn2++;
                }
                // add 0s at end
                while (pOut1 < pEndOut1)
                {
                    *pOut1++ = 0;
                }
            }
            break;
        }
    }
    return (PyObject *)pNewString;
}

//-----------------------------------------------------------------------------------
// Called when two unicode are added
//
PyObject * ConcatTwoUnicodes(int32_t scalarMode, uint32_t * pStr1, uint32_t * pStr2, int64_t itemSize1, int64_t itemSize2,
                             int64_t length)
{
    itemSize1 = itemSize1 / 4;
    itemSize2 = itemSize2 / 4;

    int64_t itemSizeOut = itemSize1 + itemSize2;

    // printf("concat two unicode %lld %lld\n", itemSize1, itemSize2);

    PyArrayObject * pNewString = AllocateNumpyArray(1, (npy_intp *)&length, NPY_UNICODE, itemSizeOut * 4);
    CHECK_MEMORY_ERROR(pNewString);

    if (pNewString)
    {
        uint32_t * pOut = (uint32_t *)PyArray_BYTES(pNewString);
        switch (scalarMode)
        {
        case SCALAR_MODE::NO_SCALARS:
            for (int64_t i = 0; i < length; i++)
            {
                uint32_t * pOut1 = pOut + (itemSizeOut * i);
                uint32_t * pEndOut1 = pOut1 + itemSizeOut;
                uint32_t * pIn1 = pStr1 + (itemSize1 * i);
                uint32_t * pEnd1 = pIn1 + itemSize1;
                uint32_t * pIn2 = pStr2 + (itemSize2 * i);
                uint32_t * pEnd2 = pIn2 + itemSize2;

                // printf("%lld %lld ", i, pEndOut1 - pOut1);

                while (pIn1 < pEnd1 && *pIn1)
                {
                    *pOut1++ = *pIn1++;
                }
                while (pIn2 < pEnd2 && *pIn2)
                {
                    *pOut1++ = *pIn2++;
                }

                // printf(" > %lld %lld < ", i, pEndOut1 - pOut1);

                // add 0s at end
                while (pOut1 < pEndOut1)
                {
                    *pOut1++ = 0;
                }
            }
            break;
        case SCALAR_MODE::FIRST_ARG_SCALAR:
            for (int64_t i = 0; i < length; i++)
            {
                uint32_t * pOut1 = pOut + (itemSizeOut * i);
                uint32_t * pEndOut1 = pOut1 + itemSizeOut;
                uint32_t * pIn1 = pStr1;
                uint32_t * pEnd1 = pIn1 + itemSize1;
                uint32_t * pIn2 = pStr2 + (itemSize2 * i);
                uint32_t * pEnd2 = pIn2 + itemSize2;

                // printf("%lld %lld ", i, pEndOut1 - pOut1);

                while (pIn1 < pEnd1 && *pIn1)
                {
                    *pOut1++ = *pIn1++;
                }
                while (pIn2 < pEnd2 && *pIn2)
                {
                    *pOut1++ = *pIn2++;
                }

                // printf(" > %lld %lld < ", i, pEndOut1 - pOut1);

                // add 0s at end
                while (pOut1 < pEndOut1)
                {
                    *pOut1++ = 0;
                }
            }
            break;
        case SCALAR_MODE::SECOND_ARG_SCALAR:
            for (int64_t i = 0; i < length; i++)
            {
                uint32_t * pOut1 = pOut + (itemSizeOut * i);
                uint32_t * pEndOut1 = pOut1 + itemSizeOut;
                uint32_t * pIn1 = pStr1 + (itemSize1 * i);
                uint32_t * pEnd1 = pIn1 + itemSize1;
                uint32_t * pIn2 = pStr2;
                uint32_t * pEnd2 = pIn2 + itemSize2;

                // printf("%lld %lld ", i, pEndOut1 - pOut1);

                while (pIn1 < pEnd1 && *pIn1)
                {
                    *pOut1++ = *pIn1++;
                }
                while (pIn2 < pEnd2 && *pIn2)
                {
                    *pOut1++ = *pIn2++;
                }

                // printf(" > %lld %lld < ", i, pEndOut1 - pOut1);

                // add 0s at end
                while (pOut1 < pEndOut1)
                {
                    *pOut1++ = 0;
                }
            }
            break;
        }
    }

    return (PyObject *)pNewString;
}

//-------------------------------------------------------------
// Called when adding two strings
// May return nullptr on failure, else new string array
static PyObject * PossiblyAddUnicode(CTwoInputs & twoInputs)
{
    int32_t inType = -1;

    if (twoInputs.scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR &&
        (twoInputs.numpyInType2 == NPY_STRING || twoInputs.numpyInType2 == NPY_UNICODE))
    {
        inType = twoInputs.numpyInType2;
        LOGGING("First arg scalar %d %lld %lld\n", inType, twoInputs.itemSize1, twoInputs.itemSize2);
    }
    if (twoInputs.scalarMode == SCALAR_MODE::SECOND_ARG_SCALAR &&
        (twoInputs.numpyInType1 == NPY_STRING || twoInputs.numpyInType1 == NPY_UNICODE))
    {
        inType = twoInputs.numpyInType1;
        LOGGING("Second arg scalar %d %lld %lld\n", inType, twoInputs.itemSize1, twoInputs.itemSize2);
    }

    if (twoInputs.scalarMode == SCALAR_MODE::NO_SCALARS && twoInputs.numpyInType1 == twoInputs.numpyInType2)
    {
        inType = twoInputs.numpyInType1;
        LOGGING("Both arg scalar %d %lld %lld\n", inType, twoInputs.itemSize1, twoInputs.itemSize2);
    }
    if (inType != -1)
    {
        if (inType == NPY_STRING)
        {
            // printf("before call\n");
            // NOTE: handle out of memory condition
            return ConcatTwoStrings(twoInputs.scalarMode, (char *)twoInputs.pDataIn1, (char *)twoInputs.pDataIn2,
                                    twoInputs.itemSize1, twoInputs.itemSize2,
                                    twoInputs.len1 >= twoInputs.len2 ? twoInputs.len1 : twoInputs.len2);
        }
        if (inType == NPY_UNICODE)
        {
            // printf("before call\n");
            return ConcatTwoUnicodes(twoInputs.scalarMode, (uint32_t *)twoInputs.pDataIn1, (uint32_t *)twoInputs.pDataIn2,
                                     twoInputs.itemSize1, twoInputs.itemSize2,
                                     twoInputs.len1 >= twoInputs.len2 ? twoInputs.len1 : twoInputs.len2);
        }
    }
    return nullptr;
}

//-----------------------------------------------------------------------------------
// Called when two arrays are used as input params
// Such as min/max functions
// GreaterThan/LessThan
// Call with (tuple(arr1,arr2), funcNumber, numpyOutputType)  or
// Call with (tuple(arr1,arr2,out), funcNumber, numpyOutputType)
//
// New: Bitwise many arr1 |= arr2 | arr3
//
// If NONE is returned, punt to numpy
PyObject * TwoInputsInternal(CTwoInputs & twoInputs, int64_t funcNumber)
{
    // SPECIAL HOOK FOR ADDING TWO STRINGS -------------------------
    if (funcNumber == MATH_OPERATION::ADD)
    {
        PyObject * result = PossiblyAddUnicode(twoInputs);
        if (result)
            return result;
    }

    // Now check if we can perform the function requested
    ANY_TWO_FUNC pTwoFunc = nullptr;
    int wantedOutputType = -1;

    pTwoFunc = CheckMathOpTwoInputs((int)funcNumber, twoInputs.scalarMode, twoInputs.numpyInType1, twoInputs.numpyInType2,
                                    twoInputs.numpyOutType, &wantedOutputType);

    // Place holder for output array
    PyArrayObject * outputArray = nullptr;

    // if (pTwoFunc && wantedOutputType != -1 && twoInputs.numpyOutType ==
    // wantedOutputType) {
    //   printf("Wanted output type %d does not match %d\n", wantedOutputType,
    //   numpyOutputType);
    //} else
    // Aborting - punt this to numpy since we cannot do it/understand it
    if (pTwoFunc && wantedOutputType != -1)
    {
        // This will allocate the output type
        outputArray = twoInputs.CheckOutputs(wantedOutputType);
    }
    LOGGING(" inner %p %p %d  lens: %lld %lld\n", outputArray, pTwoFunc, wantedOutputType, twoInputs.len1, twoInputs.len2);

    // If everything worked so far, perform the math operation
    if (outputArray && pTwoFunc && wantedOutputType != -1)
    {
        void * pDataOut = PyArray_BYTES(outputArray);

        FUNCTION_LIST fl;
        fl.AnyTwoStubCall = pTwoFunc;
        fl.FunctionName = "BMath";

        if (twoInputs.len1 > 0 || twoInputs.len2 > 0)
        {
            // if there is a scalar
            fl.Input1Strides = 0;
            fl.Input2Strides = 0;

            // TODO: handle C vs F two dim arrays (if not C contiguous punt)
            if (twoInputs.len1 > 1)
                fl.Input1Strides = PyArray_STRIDE(twoInputs.inArr, 0);
            if (twoInputs.len2 > 1)
                fl.Input2Strides = PyArray_STRIDE(twoInputs.inArr2, 0);

            fl.InputItemSize =
                twoInputs.len1 >= twoInputs.len2 ? PyArray_ITEMSIZE(twoInputs.inArr) : PyArray_ITEMSIZE(twoInputs.inArr2);
            fl.OutputItemSize = PyArray_ITEMSIZE(outputArray);

            fl.NumpyOutputType = wantedOutputType;
            fl.NumpyType = twoInputs.len1 >= twoInputs.len2 ? twoInputs.numpyInType1 : twoInputs.numpyInType2;
            fl.TypeOfFunctionCall = TYPE_OF_FUNCTION_CALL::ANY_TWO;

            LOGGING("Calling func %p %p %p %llu %llu %d\n", twoInputs.pDataIn1, twoInputs.pDataIn2, pDataOut, twoInputs.len1,
                    twoInputs.len2, twoInputs.scalarMode);
            // pTwoFunc(twoInputs.pDataIn1, twoInputs.pDataIn2, pDataOut,
            // twoInputs.len1 >= twoInputs.len2 ? twoInputs.len1 : twoInputs.len2,
            // twoInputs.scalarMode);
            WorkTwoStubCall(&fl, twoInputs.pDataIn1, twoInputs.pDataIn2, pDataOut,
                            twoInputs.len1 >= twoInputs.len2 ? twoInputs.len1 : twoInputs.len2, twoInputs.scalarMode);

            // Not sure why this final step is required to call fastarrayview
            // otherwise, the view is not set properly from a new array (recycled
            // arrays do work though)
            LOGGING("Setting view\n");
            return SetFastArrayView(outputArray);
        }

        // RETURN EMPTY ARRAY?
        LOGGING("Setting empty array view\n");
        return SetFastArrayView(outputArray);
    }
    LOGGING("BasicMath punting inner loop\n");
    // punt to numpy
    Py_INCREF(Py_None);
    return Py_None;
}

//----------------------------------------------------------------------------------
// Called when two arrays are used as input params
// Such as min/max functions
// GreaterThan/LessThan
// Call with (tuple(arr1,arr2), funcNumber, numpyOutputType)  or
// Call with (tuple(arr1,arr2,out), funcNumber, numpyOutputType)
//
// New: Bitwise many arr1 |= arr2 | arr3
//
// If NONE is returned, punt to numpy
PyObject * BasicMathTwoInputs(PyObject * self, PyObject * args)
{
    PyObject * tuple = nullptr;
    int64_t funcNumber;
    int64_t numpyOutputType;

    if (Py_SIZE(args) != 3)
    {
        PyErr_Format(PyExc_ValueError, "BasicMathTwoInputs requires three inputs: tuple, long, long");
        return nullptr;
    }

    tuple = PyTuple_GET_ITEM(args, 0);
    funcNumber = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 1));
    numpyOutputType = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 2));

    if (! PyTuple_CheckExact(tuple))
    {
        PyErr_Format(PyExc_ValueError, "BasicMathTwoInputs arguments needs to be a tuple");
        return nullptr;
    }

    // Examine data
    CTwoInputs twoInputs((int)numpyOutputType, (int)funcNumber);

    // Check the inputs to see if the user request makes sense
    bool result = twoInputs.CheckInputs(tuple, 0, funcNumber);

    if (result)
    {
        return TwoInputsInternal(twoInputs, funcNumber);
    }

    // punt to numpy
    Py_INCREF(Py_None);
    return Py_None;
}

//-----------------------------------------------------------------------------------
// Called internally when type class numbermethod called
// inputOut can be nullptr if there is no output array
PyObject * BasicMathTwoInputsFromNumber(PyObject * input1, PyObject * input2, PyObject * output, int64_t funcNumber)
{
    CTwoInputs twoInputs((int)0, (int)funcNumber);

    // Check the inputs to see if the user request makes sense
    bool result = false;
    result = twoInputs.CheckInputs(input1, input2, output, funcNumber);

    if (result)
    {
        return TwoInputsInternal(twoInputs, funcNumber);
    }

    // punt to numpy
    Py_INCREF(Py_None);
    return Py_None;
}

// MT callback
struct WhereCallbackStruct
{
    int8_t * pBooleanMask; // condition
    void * pValuesOut;
    void * pChoice1;
    void * pChoice2;
    int64_t itemSize; // of output array, used for strings
    int64_t strideSize1;
    int64_t strideSize2;
    int64_t strideSize3;
    int32_t scalarMode;
};

template <typename T>
bool WhereCallback(void * callbackArgT, int core, int64_t start, int64_t length)
{
    WhereCallbackStruct * pCallback = (WhereCallbackStruct *)callbackArgT;
    bool * pCondition = (bool *)pCallback->pBooleanMask;
    T * pOutput = (T *)pCallback->pValuesOut;
    T * pInput1 = (T *)pCallback->pChoice1;
    T * pInput2 = (T *)pCallback->pChoice2;
    int64_t strides1 = pCallback->strideSize1;
    int64_t strides2 = pCallback->strideSize2;
    int64_t strides3 = pCallback->strideSize3;

    int64_t itemSize = pCallback->itemSize;
    pCondition = &pCondition[start];
    pOutput = &pOutput[start];

    if (pCallback->scalarMode == SCALAR_MODE::NO_SCALARS || pCallback->scalarMode == SCALAR_MODE::SECOND_ARG_SCALAR)
    {
        pInput1 = &pInput1[start * strides1];
    }
    if (pCallback->scalarMode == SCALAR_MODE::NO_SCALARS || pCallback->scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
    {
        pInput2 = &pInput2[start * strides2];
    }

    LOGGING(
        "[%d] Where start: %lld  length: %lld   scalar: %d  strides1: %lld  "
        "strides2: %lld\n",
        core, start, length, pCallback->scalarMode, strides1, strides2);
    // printf("Where strides:%lld %lld\n", strides1, strides2);

    // BUG BUG what if... itemsize is small?
    // What if condition is strided?
    if (strides1 == 1 && strides2 == 1)
    {
        switch (pCallback->scalarMode)
        {
        case SCALAR_MODE::NO_SCALARS:
            {
                // TODO: future optimization to load 16 bools at once
                // Create 16 ones in a row
                //__m128i allones = _mm_set1_epi8(1);
                //__m128i* pCondition128 = (__m128i*)pCondition;
                //__m128i condition = _mm_min_epu8(*pCondition128, allones);

                // calculate the address delta since we know input1 and input2 are the
                // same itemsize assumes 64bit addressing
                uint64_t delta = pInput2 - pInput1;
                uint64_t ulength = (uint64_t)length;

                for (uint64_t i = 0; i < ulength; i++)
                {
                    // The condition is either 1 or 0.  if 1 we use the address delta to get
                    // from input2 to input1
                    uint64_t test = pCondition[i] == 0;
                    pOutput[i] = pInput1[i + delta * test];
                }

                // This is slower on msft windows compiler
                // does not produce cmove (conditional move instruction)
                // for (uint64_t i = 0; i < ulength; i++) {
                //   // The condition is either 1 or 0.  if 1 we use the address delta to
                //   get from input2 to input1 T* address = pCondition[i] ? pInput1 :
                //   pInput2; pOutput[i] = address[i];
                //}
            }
            break;
        case SCALAR_MODE::FIRST_ARG_SCALAR:
            {
                T input1 = pInput1[0];
                for (int64_t i = 0; i < length; i++)
                {
                    pOutput[i] = pCondition[i] ? input1 : pInput2[i];
                }
            }
            break;
        case SCALAR_MODE::SECOND_ARG_SCALAR:
            {
                T input2 = pInput2[0];
                for (int64_t i = 0; i < length; i++)
                {
                    pOutput[i] = pCondition[i] ? pInput1[i] : input2;
                }
            }
            break;
        case SCALAR_MODE::BOTH_SCALAR:
            {
                T input1 = pInput1[0];
                T input2 = pInput2[0];
                for (int64_t i = 0; i < length; i++)
                {
                    pOutput[i] = pCondition[i] ? input1 : input2;
                }
            }
            break;
        }
    }
    else
    {
        switch (pCallback->scalarMode)
        {
        case SCALAR_MODE::NO_SCALARS:
            {
                for (int64_t i = 0; i < length; i++)
                {
                    if (pCondition[i])
                    {
                        pOutput[i] = pInput1[i * strides1];
                    }
                    else
                    {
                        pOutput[i] = pInput2[i * strides2];
                    }
                }
            }
            break;
        case SCALAR_MODE::FIRST_ARG_SCALAR:
            {
                for (int64_t i = 0; i < length; i++)
                {
                    if (pCondition[i])
                    {
                        pOutput[i] = pInput1[0];
                    }
                    else
                    {
                        pOutput[i] = pInput2[i];
                    }
                }
            }
            break;
        case SCALAR_MODE::SECOND_ARG_SCALAR:
            {
                for (int64_t i = 0; i < length; i++)
                {
                    if (pCondition[i])
                    {
                        pOutput[i] = pInput1[i * strides1];
                    }
                    else
                    {
                        pOutput[i] = pInput2[0];
                    }
                }
            }
            break;
        case SCALAR_MODE::BOTH_SCALAR:
            {
                for (int64_t i = 0; i < length; i++)
                {
                    if (pCondition[i])
                    {
                        pOutput[i] = pInput1[0];
                    }
                    else
                    {
                        pOutput[i] = pInput2[0];
                    }
                }
            }
            break;
        }
    }

    return true;
}

//========================================================================
// callback return for strings or odd itemsizes
bool WhereCallbackString(void * callbackArgT, int core, int64_t start, int64_t length)
{
    WhereCallbackStruct * pCallback = (WhereCallbackStruct *)callbackArgT;
    int8_t * pCondition = pCallback->pBooleanMask;
    char * pOutput = (char *)pCallback->pValuesOut;
    char * pInput1 = (char *)pCallback->pChoice1;
    char * pInput2 = (char *)pCallback->pChoice2;
    int64_t itemSize = pCallback->itemSize;

    // BUG BUG what if... itemsize is small?
    switch (pCallback->scalarMode)
    {
    case SCALAR_MODE::NO_SCALARS:
        {
            for (int64_t i = start; i < (start + length); i++)
            {
                if (pCondition[i])
                {
                    memcpy(&pOutput[i * itemSize], &pInput1[i * itemSize], itemSize);
                }
                else
                {
                    memcpy(&pOutput[i * itemSize], &pInput2[i * itemSize], itemSize);
                }
            }
        }
        break;
    case SCALAR_MODE::FIRST_ARG_SCALAR:
        {
            for (int64_t i = start; i < (start + length); i++)
            {
                if (pCondition[i])
                {
                    memcpy(&pOutput[i * itemSize], pInput1, itemSize);
                }
                else
                {
                    memcpy(&pOutput[i * itemSize], &pInput2[i * itemSize], itemSize);
                }
            }
        }
        break;
    case SCALAR_MODE::SECOND_ARG_SCALAR:
        {
            for (int64_t i = start; i < (start + length); i++)
            {
                if (pCondition[i])
                {
                    memcpy(&pOutput[i * itemSize], &pInput1[i * itemSize], itemSize);
                }
                else
                {
                    memcpy(&pOutput[i * itemSize], pInput2, itemSize);
                }
            }
        }
        break;
    case SCALAR_MODE::BOTH_SCALAR:
        {
            for (int64_t i = start; i < (start + length); i++)
            {
                if (pCondition[i])
                {
                    memcpy(&pOutput[i * itemSize], pInput1, itemSize);
                }
                else
                {
                    memcpy(&pOutput[i * itemSize], pInput2, itemSize);
                }
            }
        }
        break;
    }
    return true;
}

//-----------------------------------------------------------------------------------
// Called when three arrays are used as input params, and one is boolean array
//
// see: np.where
// where(condition, x, y)   condition must be boolean array
//
// rc.where(condition, (x,y), 0, dtype.num, dtype.itemsize)
//
PyObject * Where(PyObject * self, PyObject * args)
{
    PyArrayObject * inBool1 = nullptr;
    PyObject * tuple = nullptr;
    int64_t numpyOutputType;
    int64_t numpyItemSize;

    if (! PyArg_ParseTuple(args, "O!OLL", &PyArray_Type, &inBool1, &tuple, &numpyOutputType, &numpyItemSize))
    {
        return nullptr;
    }

    if (! PyTuple_CheckExact(tuple))
    {
        PyErr_Format(PyExc_ValueError, "Where: second argument needs to be a tuple");
        return nullptr;
    }

    if (PyArray_TYPE(inBool1) != 0)
    {
        PyErr_Format(PyExc_ValueError, "Where: first argument must be a boolean array");
        return nullptr;
    }

    // Examine data
    CTwoInputs twoInputs((int)numpyOutputType, 0);

    // get length of boolean array
    int64_t length = ArrayLength(inBool1);
    twoInputs.expectedLength = length;

    LOGGING("Where: Examining data for func types: %d %d\n", twoInputs.numpyInType1, twoInputs.numpyInType2);

    // Check the inputs to see if the user request makes sense
    bool result = false;

    result = twoInputs.CheckInputs(tuple, 0, MATH_OPERATION::WHERE);

    if (result)
    {
        LOGGING(
            "Where input: scalarmode: %d  in1:%d in2:%d  out:%d  length: %lld  "
            "itemsize: %lld\n",
            twoInputs.scalarMode, twoInputs.numpyInType1, twoInputs.numpyInType2, twoInputs.numpyOutType, length, numpyItemSize);

        // Place holder for output array
        PyArrayObject * outputArray = nullptr;

        outputArray = AllocateNumpyArray(1, (npy_intp *)&length, (int)numpyOutputType, numpyItemSize);

        CHECK_MEMORY_ERROR(outputArray);

        if (outputArray)
        {
            int8_t * pCondition = (int8_t *)PyArray_BYTES(inBool1);
            void * pOutput = PyArray_BYTES(outputArray);

            // for scalar strings, the item size is twoInputs.itemSize1
            int64_t itemSize1 = twoInputs.itemSize1;
            int64_t itemSize2 = twoInputs.itemSize2;
            char * pInput1 = (char *)twoInputs.pDataIn1;
            char * pInput2 = (char *)twoInputs.pDataIn2;

            char * extraAlloc1 = nullptr;
            char * extraAlloc2 = nullptr;

            // If we have scalar strings, we must make sure the length is the same
            if (numpyOutputType == NPY_STRING || numpyOutputType == NPY_UNICODE)
            {
                if (twoInputs.scalarMode == SCALAR_MODE::BOTH_SCALAR || twoInputs.scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
                {
                    // These are small allocs that should never fail
                    extraAlloc1 = (char *)malloc(numpyItemSize);
                    memset(extraAlloc1, 0, numpyItemSize);
                    memcpy(extraAlloc1, pInput1, itemSize1);

                    // swap out for the new one
                    pInput1 = extraAlloc1;
                }
                if (twoInputs.scalarMode == SCALAR_MODE::BOTH_SCALAR || twoInputs.scalarMode == SCALAR_MODE::SECOND_ARG_SCALAR)
                {
                    extraAlloc2 = (char *)malloc(numpyItemSize);
                    memset(extraAlloc2, 0, numpyItemSize);
                    memcpy(extraAlloc2, pInput2, itemSize2);
                    pInput2 = extraAlloc2;
                }
            }

            WhereCallbackStruct WCBS;
            WCBS.itemSize = numpyItemSize;
            WCBS.pBooleanMask = pCondition;
            WCBS.pValuesOut = pOutput;
            WCBS.scalarMode = twoInputs.scalarMode;
            WCBS.pChoice1 = pInput1;
            WCBS.pChoice2 = pInput2;
            WCBS.strideSize1 = 1;
            WCBS.strideSize2 = 1;
            WCBS.strideSize3 = 1;

            // handle strided arrays only when one dimension
            if (twoInputs.scalarMode == SCALAR_MODE::NO_SCALARS || twoInputs.scalarMode == SCALAR_MODE::SECOND_ARG_SCALAR)
            {
                if (twoInputs.ndim1 == 1)
                    WCBS.strideSize1 = PyArray_STRIDE(twoInputs.inArr, 0) / twoInputs.itemSize1;
            }
            if (twoInputs.scalarMode == SCALAR_MODE::NO_SCALARS || twoInputs.scalarMode == SCALAR_MODE::FIRST_ARG_SCALAR)
            {
                if (twoInputs.ndim2 == 1)
                    WCBS.strideSize2 = PyArray_STRIDE(twoInputs.inArr2, 0) / twoInputs.itemSize2;
            }

            switch (numpyItemSize)
            {
            case 1:
                g_cMathWorker->DoMultiThreadedChunkWork(length, WhereCallback<int8_t>, &WCBS);
                break;
            case 2:
                g_cMathWorker->DoMultiThreadedChunkWork(length, WhereCallback<int16_t>, &WCBS);
                break;
            case 4:
                g_cMathWorker->DoMultiThreadedChunkWork(length, WhereCallback<int32_t>, &WCBS);
                break;
            case 8:
                g_cMathWorker->DoMultiThreadedChunkWork(length, WhereCallback<int64_t>, &WCBS);
                break;
            default:
                g_cMathWorker->DoMultiThreadedChunkWork(length, WhereCallbackString, &WCBS);
                break;
            }

            //==========================================================================================
            if (extraAlloc1)
            {
                free(extraAlloc1);
            }
            if (extraAlloc2 == nullptr)
            {
                free(extraAlloc2);
            }
            return (PyObject *)outputArray;
        }

        PyErr_Format(PyExc_ValueError, "Where could not allocate a numpy array");
        return nullptr;
    }

    // punt to numpy
    Py_INCREF(Py_None);
    return Py_None;
}
