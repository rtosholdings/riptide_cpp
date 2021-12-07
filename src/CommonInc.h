#pragma once

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

// TODO: Remove these includes, they don't seem to be used anymore (at least directly within this file).
#include <cstdio>
#include <cmath>
#include <cstring>

#if defined(_WIN32) && ! defined(__GNUC__)
    #define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
    #define NOMINMAX
    // Windows Header Files:
    #include <Windows.h>
    #include <winnt.h>
#else

#endif

using HANDLE = void *;

#ifdef RtlEqualMemory
    #undef RtlEqualMemory
#endif
#define RtlEqualMemory(Destination, Source, Length) (! memcmp((Destination), (Source), (Length)))
#ifdef RtlMoveMemory
    #undef RtlMoveMemory
#endif
#define RtlMoveMemory(Destination, Source, Length) memmove((Destination), (Source), (Length))
#ifdef RtlCopyMemory
    #undef RtlCopyMemory
#endif
#define RtlCopyMemory(Destination, Source, Length) memcpy((Destination), (Source), (Length))
#ifdef RtlFillMemory
    #undef RtlFillMemory
#endif
#define RtlFillMemory(Destination, Length, Fill) memset((Destination), (Fill), (Length))
#ifdef RtlZeroMemory
    #undef RtlZeroMemory
#endif
#define RtlZeroMemory(Destination, Length) memset((Destination), 0, (Length))

#if defined(_WIN32) && ! defined(__GNUC__)
    #define WINAPI __stdcall
    #define InterlockedCompareExchange128 _InterlockedCompareExchange128
    #ifndef InterlockedAdd64
        #define InterlockedAdd64 _InterlockedAdd64
    #endif
    #define InterlockedDecrement64 _InterlockedDecrement64
    #define InterlockedIncrement64 _InterlockedIncrement64
    #define YieldProcessor _mm_pause
    #define InterlockedIncrement _InterlockedIncrement

    #define FMInterlockedOr(X, Y) _InterlockedOr64((int64_t *)X, Y)

    #include <intrin.h>
    #ifndef SFW_ALIGN
        #define SFW_ALIGN(x) __declspec(align(x))
        #define ALIGN(x) __declspec(align(64))

        #define FORCEINLINE __forceinline
        #define FORCE_INLINE __forceinline

        #define ALIGNED_ALLOC(Size, Alignment) _aligned_malloc(Size, Alignment)
        #define ALIGNED_FREE(block) _aligned_free(block)

        #define lzcnt_64 _lzcnt_u64

        #define CASE_NPY_INT32 \
        case NPY_INT32: \
        case NPY_INT
        #define CASE_NPY_UINT32 \
        case NPY_UINT32: \
        case NPY_UINT
        #define CASE_NPY_INT64 case NPY_INT64
        #define CASE_NPY_UINT64 case NPY_UINT64
        #define CASE_NPY_FLOAT64 \
        case NPY_DOUBLE: \
        case NPY_LONGDOUBLE

    #endif

#else

    #define CASE_NPY_INT32 case NPY_INT32
    #define CASE_NPY_UINT32 case NPY_UINT32
    #define CASE_NPY_INT64 \
    case NPY_INT64: \
    case NPY_LONGLONG
    #define CASE_NPY_UINT64 \
    case NPY_UINT64: \
    case NPY_ULONGLONG
    #define CASE_NPY_FLOAT64 case NPY_DOUBLE

using INT_PTR = ptrdiff_t;
using DWORD = uint32_t;
using LPVOID = void *;

    #define WINAPI
    #include <pthread.h>

    // consider sync_add_and_fetch
    #define InterlockedAdd64(val, len) (__sync_fetch_and_add(val, len) + len)
    #define InterlockedIncrement64(val) (__sync_fetch_and_add(val, 1) + 1)
    #define YieldProcessor _mm_pause
    #define InterlockedIncrement(val) (__sync_fetch_and_add(val, 1) + 1)
    #define FMInterlockedOr(val, bitpos) (__sync_fetch_and_or(val, bitpos))

    #ifndef __GNUC_PREREQ
        #define __GNUC_PREREQ(major, minor) ((((__GNUC__) << 16) + (__GNUC_MINOR__)) >= (((major) << 16) + (minor)))
    #endif
    #if __GNUC_PREREQ(4, 4) || (__clang__ > 0 && __clang_major__ >= 3) || ! defined(__GNUC__)
        /* GCC >= 4.4 or clang or non-GCC compilers */
        #include <x86intrin.h>
    #elif __GNUC_PREREQ(4, 1)
        /* GCC 4.1, 4.2, and 4.3 do not have x86intrin.h, directly include SSE2 header */
        #include <emmintrin.h>
    #endif
    #ifndef SFW_ALIGN
        #define SFW_ALIGN(x) __attribute__((aligned(x)))
    #endif

    #define FORCEINLINE inline __attribute__((always_inline))
    #define FORCE_INLINE inline __attribute__((always_inline))
    //#define FORCE_INLINE __attribute__((always_inline)) inline
    //#define __forceinline __attribute__((always_inline))
    #define ALIGN(x) x __attribute__((aligned(64)))

    // Workaround for platforms/compilers which don't support C11 aligned_alloc
    // but which do have posix_memalign().
    #ifndef aligned_alloc

        #ifdef posix_memalign
FORCEINLINE void * aligned_alloc(size_t alignment, size_t size)
{
    void * buffer = NULL;
    posix_memalign(&buffer, alignment, size);
    return buffer;
}

        #else
            // clang compiler does not support so we default to malloc
            //#warning Unable to determine how to perform aligned allocations on this platform.
            #define aligned_alloc(alignment, size) malloc(size)
        #endif // defined(posix_memalign)

    #endif // !defined(aligned_alloc)

    #define ALIGNED_ALLOC(Size, Alignment) aligned_alloc(Alignment, Size)
    #define ALIGNED_FREE(block) free(block)

    #define lzcnt_64 __builtin_clzll

#endif

// add this after memory allocation to help debug
#define CHECK_MEMORY_ERROR(_X_) \
    if (! _X_) \
        printf("!!!Out of MEMORY: File: %s  Line: %d  Function: %s\n", __FILE__, (int)__LINE__, __FUNCTION__);

#ifndef ASSERT
    #include <cassert>
    #define ASSERT assert
#endif

#define LogInform printf
#define LogError printf

// Uncomment to allow verbose logging
//#define VERBOSE LogInform
#define VERBOSE(...)

void * FmAlloc(size_t _Size);
void FmFree(void * _Block);

namespace internal
{
    struct fm_mem_deleter
    {
        void operator()(void * const block)
        {
            FmFree(block);
        }
    };
}

// Smart pointer managing FmAlloc'ed memory.
using fm_mem_ptr = std::unique_ptr<void, internal::fm_mem_deleter>;

#define ARRAY_ALLOC malloc
#define ARRAY_FREE free

#define WORKSPACE_ALLOC FmAlloc
#define WORKSPACE_FREE FmFree
using workspace_mem_ptr = fm_mem_ptr;

#define COMPRESS_ALLOC FmAlloc
#define COMPRESS_FREE FmFree

#define PYTHON_ALLOC FmAlloc
#define PYTHON_FREE FmFree

// NAN are ODD NUMBERED! follow this rule
enum REDUCE_FUNCTIONS
{
    REDUCE_SUM = 0,
    REDUCE_NANSUM = 1,

    // These output a float/double
    REDUCE_MEAN = 102,
    REDUCE_NANMEAN = 103,

    // ddof =1 for pandas, matlab   =0 for numpy
    REDUCE_VAR = 106,
    REDUCE_NANVAR = 107,
    REDUCE_STD = 108,
    REDUCE_NANSTD = 109,

    REDUCE_MIN = 200,
    REDUCE_NANMIN = 201,
    REDUCE_MAX = 202,
    REDUCE_NANMAX = 203,

    REDUCE_ARGMIN = 204,
    REDUCE_NANARGMIN = 205,
    REDUCE_ARGMAX = 206,
    REDUCE_NANARGMAX = 207,

    // For Jack TODO
    REDUCE_ANY = 208,
    REDUCE_ALL = 209,

    REDUCE_MIN_NANAWARE = 210,
};

enum MATH_OPERATION
{
    // Two ops, returns same type
    ADD = 1,
    SUB = 2,
    MUL = 3,
    MOD = 5,
    MIN = 6,
    MAX = 7,
    NANMIN = 8,
    NANMAX = 9,
    FLOORDIV = 10,
    POWER = 11,
    REMAINDER = 12,
    FMOD = 13,

    // where is special
    WHERE = 50,

    // Two ops, always return a double
    DIV = 101,
    SUBDATETIMES = 102, // returns double
    SUBDATES = 103,     // returns int

    // One input, returns same data type
    ABS = 201,
    NEG = 202,
    FABS = 203,
    INVERT = 204,
    FLOOR = 205,
    CEIL = 206,
    TRUNC = 207,
    ROUND = 208,
    REMOVED_BAD_VALUE_CAN_REUSE = 211,
    NEGATIVE = 212,
    POSITIVE = 213,
    SIGN = 214,
    RINT = 215,
    EXP = 216,
    EXP2 = 217,
    NAN_TO_NUM = 218,
    NAN_TO_ZERO = 219,

    // One input, always return a float one input
    SQRT = 301,
    LOG = 302,
    LOG2 = 303,
    LOG10 = 304,
    EXPM1 = 305,
    LOG1P = 306,
    SQUARE = 307,
    CBRT = 308,
    RECIPROCAL = 309,

    // Two inputs, Always return a bool
    CMP_EQ = 401,
    CMP_NE = 402,
    CMP_LT = 403,
    CMP_GT = 404,
    CMP_LTE = 405,
    CMP_GTE = 406,
    LOGICAL_AND = 407,
    LOGICAL_XOR = 408,
    LOGICAL_OR = 409,

    // Two inputs, second input must be int based
    BITWISE_LSHIFT = 501,
    BITWISE_RSHIFT = 502,
    BITWISE_AND = 503,
    BITWISE_XOR = 504,
    BITWISE_OR = 505,
    BITWISE_ANDNOT = 506,

    BITWISE_NOTAND = 507,

    BITWISE_XOR_SPECIAL = 550,

    // one input, output bool
    LOGICAL_NOT = 601,
    ISINF = 603,
    ISNAN = 604,
    ISFINITE = 605,
    ISNORMAL = 606,

    ISNOTINF = 607,
    ISNOTNAN = 608,
    ISNOTFINITE = 609,
    ISNOTNORMAL = 610,
    ISNANORZERO = 611,
    SIGNBIT = 612,

    // One input, does not allow floats
    BITWISE_NOT = 701,

    LAST = 999,
};

struct stScatterGatherFunc
{
    // numpy intput ttype
    int32_t inputType;

    // the core (if any) making this calculation
    int32_t core;

    // used for nans, how many non nan values
    int64_t lenOut;

    // !!must be set when used by var and std
    double meanCalculation;

    double resultOut;

    // Separate output for min/max
    int64_t resultOutInt64;
};

// Generic function declarations
// The first one passes in a vector and returns a vector
// Used for operations like B = MIN(A)
typedef double (*ANY_SCATTER_GATHER_FUNC)(void * pDataIn, int64_t len, stScatterGatherFunc * pstScatterGatherFunc);

// typedef void(*UNARY_FUNC)(void* pDataIn, void* pDataOut, int64_t len);
typedef void (*UNARY_FUNC)(void * pDataIn, void * pDataOut, int64_t len, int64_t strideIn, int64_t strideOut);
typedef void (*UNARY_FUNC_STRIDED)(void * pDataIn, void * pDataOut, int64_t len, int64_t strideIn, int64_t strideOut);

// Pass in two vectors and return one vector
// Used for operations like C = A + B
typedef void (*ANY_TWO_FUNC)(void * pDataIn, void * pDataIn2, void * pDataOut, int64_t len, int32_t scalarMode);

// typedef void(*MERGE_TWO_FUNC)(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t valSize, int64_t start, int64_t len, void*
// pDefault);

// Fast path sum
// Arg1 data such as TradeSize
// Arg2 which bin (which index)
// Arg3 option boolean filter
// Arg4 optional - count out

// Used for Groupby Sum/Mean/Min/Max/etc
typedef void (*GROUPBY_TWO_FUNC)(void * pDataIn, void * pIndex, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                                 int64_t binHigh, int64_t pass, void * pDataTmp);

// Used for Groupby Mode/Median/etc
typedef void (*GROUPBY_X_FUNC32)(void * pColumn, void * pGroup, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                 int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam);
typedef void (*GROUPBY_X_FUNC64)(void * pColumn, void * pGroup, int64_t * pFirst, int64_t * pCount, void * pAccumBin,
                                 int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam);

// Pass in three vectors and return one vector
// Used for operations like D = A*B + C
typedef void (*ANY_THREE_FUNC)(void * pDataIn, void * pDataIn2, void * pDataIn3, void * pDataOut, int64_t len, int32_t scalarMode);

typedef void (*GROUPBY_FUNC)(void * pstGroupBy, int64_t index);

// On TSEBAL650
// To add TWO floating points (25 million rows) takes about 0.05 seconds
// 25 million rows * sizeof(float) ==> 100MB of data
// Adding two numbers and then writing the results is 2xREAD and 1xWRITE ==> 300MB total in 0.05 seconds
// 300MB * 20 = 6GB/sec bandwidth
// Bit count stuff
// Output is int8_t
typedef void (*I16_I8_FUNC)(int16_t * pDataIn, int8_t * pDataOut, int64_t len);
typedef void (*I32_I8_FUNC)(int32_t * pDataIn, int8_t * pDataOut, int64_t len);
typedef void (*I64_I8_FUNC)(int64_t * pDataIn, int8_t * pDataOut, int64_t len);

//----------------------------------------------------
// returns pointer to a data type (of same size in memory) that holds the invalid value for the type
// does not yet handle strings
void * GetDefaultForType(int numpyInType);

// Overloads to handle invalids
static inline bool GET_INVALID(bool x)
{
    return false;
}
static inline int8_t GET_INVALID(int8_t X)
{
    return -128;
}
static inline uint8_t GET_INVALID(uint8_t X)
{
    return 0xFF;
}
static inline int16_t GET_INVALID(int16_t X)
{
    return -32768;
}
static inline uint16_t GET_INVALID(uint16_t X)
{
    return 0xFFFF;
}
static inline int32_t GET_INVALID(int32_t X)
{
    return 0x80000000;
}
static inline uint32_t GET_INVALID(uint32_t X)
{
    return 0xFFFFFFFF;
}
static inline int64_t GET_INVALID(int64_t X)
{
    return 0x8000000000000000;
}
static inline uint64_t GET_INVALID(uint64_t X)
{
    return 0xFFFFFFFFFFFFFFFF;
}
static inline float GET_INVALID(float X)
{
    return std::numeric_limits<float>::quiet_NaN();
}
static inline double GET_INVALID(double X)
{
    return std::numeric_limits<double>::quiet_NaN();
}
static inline long double GET_INVALID(long double X)
{
    return std::numeric_limits<long double>::quiet_NaN();
}

//-----------------------------------------------------------
// Build a list of callable vector functions
enum TYPE_OF_FUNCTION_CALL
{
    ANY_ONE = 1,
    ANY_TWO = 2,
    ANY_THREEE = 3,
    ANY_GROUPBY_FUNC = 4,
    ANY_GROUPBY_XFUNC32 = 5,
    ANY_GROUPBY_XFUNC64 = 6,
    ANY_SCATTER_GATHER = 7,
    ANY_MERGE_TWO_FUNC = 8,
    ANY_MERGE_STEP_ONE = 9
};

enum SCALAR_MODE
{
    NO_SCALARS = 0,
    FIRST_ARG_SCALAR = 1,
    SECOND_ARG_SCALAR = 2,
    BOTH_SCALAR = 3 // not used
};

//-----------------------------------------------------------
// List of function calls
struct FUNCTION_LIST
{
    int16_t TypeOfFunctionCall; // See enum
    int16_t NumpyType;          // For the array and constants
    int16_t NumpyOutputType;

    // The item size for two input arrays assumed to be the same
    int64_t InputItemSize;
    int64_t OutputItemSize;

    // Strides may be 0 if it is a scalar or length 1
    int64_t Input1Strides;
    int64_t Input2Strides;

    // TODO: Why not make this void and recast?
    // Only one of these can be set
    union
    {
        void * FunctionPtr;
        ANY_SCATTER_GATHER_FUNC AnyScatterGatherCall;
        UNARY_FUNC AnyOneStubCall;
        ANY_TWO_FUNC AnyTwoStubCall;
        ANY_THREE_FUNC AnyThreeStubCall;
        GROUPBY_FUNC GroupByCall;
    };

    const char * FunctionName;
};

//-----------------------------------------------------
// Determines the CAP on threads
#define MAX_THREADS_WHEN_CANNOT_DETECT 5

// set this value lower to help windows wake up threads
#define MAX_THREADS_ALLOWED 31

#define FUTEX_WAKE_DEFAULT 11
#define FUTEX_WAKE_MAX 31

// Macro stub for returning None
#define RETURN_NONE \
    Py_INCREF(Py_None); \
    return Py_None;
#define STRIDE_NEXT(_TYPE_, _MEM_, _STRIDE_) (_TYPE_ *)((char *)_MEM_ + _STRIDE_)

#define LOGGING(...)
