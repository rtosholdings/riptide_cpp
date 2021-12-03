#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "MultiKey.h"
#include "Ema.h"

#include <cmath>
#include <pymem.h>
#include <stdio.h>

#if defined(_WIN32) && ! defined(__GNUC__)
    #include <../Lib/site-packages/numpy/core/include/numpy/arrayobject.h>
    #include <../Lib/site-packages/numpy/core/include/numpy/ndarraytypes.h>
    #include <../Lib/site-packages/numpy/core/include/numpy/npy_common.h>
#else
    #include <numpy/arrayobject.h>
    #include <numpy/ndarraytypes.h>
    #include <numpy/npy_common.h>
#endif

#define LOGGING(...)
//#define LOGGING printf

typedef void (*ROLLING_FUNC)(void * pDataIn, void * pDataOut, int64_t len, int64_t windowSize);

// These are non-groupby routine -- straight array
enum ROLLING_FUNCTIONS
{
    ROLLING_SUM = 0,
    ROLLING_NANSUM = 1,

    // These output a float/double
    ROLLING_MEAN = 102,
    ROLLING_NANMEAN = 103,

    ROLLING_VAR = 106,
    ROLLING_NANVAR = 107,
    ROLLING_STD = 108,
    ROLLING_NANSTD = 109,
};

// these are functions that output same size
enum EMA_FUNCTIONS
{
    EMA_CUMSUM = 300,
    EMA_DECAY = 301,
    EMA_CUMPROD = 302,
    EMA_FINDNTH = 303,
    EMA_NORMAL = 304,
    EMA_WEIGHTED = 305
};

//=========================================================================================================================
typedef void (*EMA_BY_TWO_FUNC)(void * pKey, void * pAccumBin, void * pColumn, int64_t numUnique, int64_t totalInputRows,
                                void * pTime, int8_t * pIncludeMask, int8_t * pResetMask, double decayRate);

struct stEmaReturn
{
    PyArrayObject * outArray;
    int32_t numpyOutType;
    EMA_BY_TWO_FUNC pFunction;
    PyObject * returnObject;
};

//---------------------------------
// 32bit indexes
struct stEma32
{
    ArrayInfo * aInfo;
    int64_t tupleSize;
    int32_t funcNum;
    int64_t uniqueRows;
    int64_t totalInputRows;

    TYPE_OF_FUNCTION_CALL typeOfFunctionCall;
    void * pKey;

    // from params
    void * pTime;
    int8_t * inIncludeMask;
    int8_t * inResetMask;
    double doubleParam;

    stEmaReturn returnObjects[1];
};

//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// thus <float, int32> converts a float to an int32
template <typename T, typename U>
class EmaBase
{
public:
    EmaBase(){};
    ~EmaBase(){};

    // Pass in two vectors and return one vector
    // Used for operations like C = A + B
    // typedef void(*ANY_TWO_FUNC)(void* pDataIn, void* pDataIn2, void* pDataOut,
    // int64_t len, int32_t scalarMode); typedef void(*ANY_ONE_FUNC)(void* pDataIn,
    // void* pDataOut, int64_t len);

    static void RollingSum(void * pDataIn, void * pDataOut, int64_t len, int64_t windowSize)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        U currentSum = 0;

        // Priming of the summation
        for (int64_t i = 0; i < len && i < windowSize; i++)
        {
            currentSum += pIn[i];
            pOut[i] = currentSum;
        }

        for (int64_t i = windowSize; i < len; i++)
        {
            currentSum += pIn[i];

            // subtract the item leaving the window
            currentSum -= pIn[i - windowSize];

            pOut[i] = currentSum;
        }
    }

    static void RollingNanSum(void * pDataIn, void * pDataOut, int64_t len, int64_t windowSize)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        U currentSum = 0;

        T invalid = GET_INVALID((T)0);

        if (invalid == invalid)
        {
            // NON_FLOAT
            // Priming of the summation
            for (int64_t i = 0; i < len && i < windowSize; i++)
            {
                T temp = pIn[i];

                if (temp != invalid)
                {
                    currentSum += temp;
                }
                pOut[i] = currentSum;
            }

            for (int64_t i = windowSize; i < len; i++)
            {
                T temp = pIn[i];

                if (temp != invalid)
                    currentSum += pIn[i];

                // subtract the item leaving the window
                temp = pIn[i - windowSize];
                if (temp != invalid)
                    currentSum -= pIn[i - windowSize];

                pOut[i] = currentSum;
            }
        }
        else
        {
            // FLOAT
            // Priming of the summation
            for (int64_t i = 0; i < len && i < windowSize; i++)
            {
                T temp = pIn[i];

                if (temp == temp)
                {
                    currentSum += temp;
                }
                pOut[i] = currentSum;
            }

            for (int64_t i = windowSize; i < len; i++)
            {
                T temp = pIn[i];

                if (temp == temp)
                    currentSum += pIn[i];

                // subtract the item leaving the window
                temp = pIn[i - windowSize];
                if (temp == temp)
                    currentSum -= pIn[i - windowSize];

                pOut[i] = currentSum;
            }
        }
    }

    static void RollingMean(void * pDataIn, void * pDataOut, int64_t len, int64_t windowSize)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        U currentSum = 0;

        // Priming of the summation
        for (int64_t i = 0; i < len && i < windowSize; i++)
        {
            currentSum += pIn[i];
            pOut[i] = currentSum / (i + 1);
        }

        for (int64_t i = windowSize; i < len; i++)
        {
            currentSum += pIn[i];

            // subtract the item leaving the window
            currentSum -= pIn[i - windowSize];

            pOut[i] = currentSum / windowSize;
        }
    }

    static void RollingNanMean(void * pDataIn, void * pDataOut, int64_t len, int64_t windowSize)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        U currentSum = 0;
        U count = 0;

        // Priming of the summation
        for (int64_t i = 0; i < len && i < windowSize; i++)
        {
            T temp = pIn[i];

            if (temp == temp)
            {
                currentSum += temp;
                count++;
            }
            pOut[i] = currentSum / count;
        }

        for (int64_t i = windowSize; i < len; i++)
        {
            T temp = pIn[i];

            if (temp == temp)
            {
                currentSum += pIn[i];
                count++;
            }

            // subtract the item leaving the window
            temp = pIn[i - windowSize];

            if (temp == temp)
            {
                currentSum -= pIn[i - windowSize];
                count--;
            }

            pOut[i] = currentSum / count;
        }
    }

    static void RollingVar(void * pDataIn, void * pDataOut, int64_t len, int64_t windowSize)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        U amean = 0;
        U asqr = 0;
        U delta;

        // Priming of the summation
        for (int64_t i = 0; i < len && i < windowSize; i++)
        {
            T item = pIn[i];

            delta = item - amean;
            amean += delta / (i + 1);
            asqr += delta * (item - amean);
            pOut[i] = asqr / i;
        }

        U count_inv = (U)1.0 / windowSize;

        for (int64_t i = windowSize; i < len; i++)
        {
            U item = (U)pIn[i];
            U old = (U)pIn[i - windowSize];

            delta = item - old;
            old -= amean;
            amean += delta * count_inv;
            item -= amean;
            asqr += (item + old) * delta;

            pOut[i] = asqr * count_inv;
        }
    }

    static void RollingStd(void * pDataIn, void * pDataOut, int64_t len, int64_t windowSize)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        U amean = 0;
        U asqr = 0;
        U delta;

        // Priming of the summation
        for (int64_t i = 0; i < len && i < windowSize; i++)
        {
            T item = pIn[i];

            delta = item - amean;
            amean += delta / (i + 1);
            asqr += delta * (item - amean);
            pOut[i] = sqrt(asqr / i);
        }

        U count_inv = (U)1.0 / windowSize;

        for (int64_t i = windowSize; i < len; i++)
        {
            U item = (U)pIn[i];
            U old = (U)pIn[i - windowSize];

            delta = item - old;
            old -= amean;
            amean += delta * count_inv;
            item -= amean;
            asqr += (item + old) * delta;

            pOut[i] = sqrt(asqr * count_inv);
        }
    }

    static void RollingNanVar(void * pDataIn, void * pDataOut, int64_t len, int64_t windowSize)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        U amean = 0;
        U asqr = 0;
        U delta;
        U count = 0;

        // Priming of the summation
        for (int64_t i = 0; i < len && i < windowSize; i++)
        {
            U item = (U)pIn[i];

            if (item == item)
            {
                count += 1;
                delta = item - amean;
                amean += delta / count;
                asqr += delta * (item - amean);
                pOut[i] = asqr / count;
            }
            else
            {
                pOut[i] = NAN;
            }
        }

        U count_inv = (U)1.0 / windowSize;

        for (int64_t i = windowSize; i < len; i++)
        {
            U item = (U)pIn[i];
            U old = (U)pIn[i - windowSize];

            if (item == item)
            {
                if (old == old)
                {
                    delta = item - old;
                    old -= amean;
                    amean += delta * count_inv;
                    item -= amean;
                    asqr += (item + old) * delta;
                }
                else
                {
                    count += 1;
                    count_inv = (U)1 / count;
                    // ddof
                    delta = item - amean;
                    amean += delta * count_inv;
                    asqr += delta * (item - amean);
                }
            }
            else
            {
                if (old == old)
                {
                    count -= 1;
                    count_inv = (U)1 / count;
                    // dd
                    if (count > 0)
                    {
                        delta = old = amean;
                        amean -= delta * count_inv;
                        asqr -= delta * (old - amean);
                    }
                    else
                    {
                        amean = 0;
                        asqr = 0;
                    }
                }
            }
            if (! (asqr >= 0))
            {
                asqr = 0;
            }

            pOut[i] = asqr * count_inv;

            // SQR pOut[i] = sqrt(asqr*count_inv);
        }
    }

    static void RollingNanStd(void * pDataIn, void * pDataOut, int64_t len, int64_t windowSize)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;

        U amean = 0;
        U asqr = 0;
        U delta;
        U count = 0;

        // Priming of the summation
        for (int64_t i = 0; i < len && i < windowSize; i++)
        {
            U item = (U)pIn[i];

            if (item == item)
            {
                count += 1;
                delta = item - amean;
                amean += delta / count;
                asqr += delta * (item - amean);
                pOut[i] = sqrt(asqr / count);
            }
            else
            {
                pOut[i] = NAN;
            }
        }

        U count_inv = (U)1.0 / windowSize;

        for (int64_t i = windowSize; i < len; i++)
        {
            U item = (U)pIn[i];
            U old = (U)pIn[i - windowSize];

            if (item == item)
            {
                if (old == old)
                {
                    delta = item - old;
                    old -= amean;
                    amean += delta * count_inv;
                    item -= amean;
                    asqr += (item + old) * delta;
                }
                else
                {
                    count += 1;
                    count_inv = (U)1 / count;
                    // ddof
                    delta = item - amean;
                    amean += delta * count_inv;
                    asqr += delta * (item - amean);
                }
            }
            else
            {
                if (old == old)
                {
                    count -= 1;
                    count_inv = (U)1 / count;
                    // dd
                    if (count > 0)
                    {
                        delta = old = amean;
                        amean -= delta * count_inv;
                        asqr -= delta * (old - amean);
                    }
                    else
                    {
                        amean = 0;
                        asqr = 0;
                    }
                }
            }
            if (! (asqr >= 0))
            {
                asqr = 0;
            }

            pOut[i] = sqrt(asqr * count_inv);

            // SQR pOut[i] = sqrt(asqr*count_inv);
        }
    }

    static ROLLING_FUNC GetRollingFunction(int64_t func)
    {
        switch (func)
        {
        case ROLLING_SUM:
            return RollingSum;
        case ROLLING_NANSUM:
            return RollingNanSum;
        }
        return NULL;
    }

    static ROLLING_FUNC GetRollingFunction2(int64_t func)
    {
        switch (func)
        {
        case ROLLING_MEAN:
            return RollingMean;
        case ROLLING_NANMEAN:
            return RollingNanMean;
        case ROLLING_VAR:
            return RollingVar;
        case ROLLING_NANVAR:
            return RollingNanVar;
        case ROLLING_STD:
            return RollingStd;
        case ROLLING_NANSTD:
            return RollingNanStd;
        }
        return NULL;
    }
};

ROLLING_FUNC GetRollingFunction(int64_t func, int32_t inputType)
{
    switch (inputType)
    {
    case NPY_BOOL:
        return EmaBase<int8_t, int64_t>::GetRollingFunction(func);
    case NPY_FLOAT:
        return EmaBase<float, float>::GetRollingFunction(func);
    case NPY_DOUBLE:
        return EmaBase<double, double>::GetRollingFunction(func);
    case NPY_LONGDOUBLE:
        return EmaBase<long double, long double>::GetRollingFunction(func);
    case NPY_INT8:
        return EmaBase<int8_t, int64_t>::GetRollingFunction(func);
    case NPY_INT16:
        return EmaBase<int16_t, int64_t>::GetRollingFunction(func);
    CASE_NPY_INT32:
        return EmaBase<int32_t, int64_t>::GetRollingFunction(func);
    CASE_NPY_UINT32:
        return EmaBase<uint32_t, int64_t>::GetRollingFunction(func);
    CASE_NPY_INT64:

        return EmaBase<int64_t, int64_t>::GetRollingFunction(func);
    case NPY_UINT8:
        return EmaBase<uint8_t, int64_t>::GetRollingFunction(func);
    case NPY_UINT16:
        return EmaBase<uint16_t, int64_t>::GetRollingFunction(func);
    CASE_NPY_UINT64:

        return EmaBase<uint64_t, int64_t>::GetRollingFunction(func);
    }

    return NULL;
}

ROLLING_FUNC GetRollingFunction2(int64_t func, int32_t inputType)
{
    switch (inputType)
    {
    case NPY_BOOL:
        return EmaBase<int8_t, double>::GetRollingFunction2(func);
    case NPY_FLOAT:
        return EmaBase<float, float>::GetRollingFunction2(func);
    case NPY_DOUBLE:
        return EmaBase<double, double>::GetRollingFunction2(func);
    case NPY_LONGDOUBLE:
        return EmaBase<long double, long double>::GetRollingFunction2(func);
    case NPY_INT8:
        return EmaBase<int8_t, double>::GetRollingFunction2(func);
    case NPY_INT16:
        return EmaBase<int16_t, double>::GetRollingFunction2(func);
    CASE_NPY_INT32:
        return EmaBase<int32_t, double>::GetRollingFunction2(func);
    CASE_NPY_UINT32:
        return EmaBase<uint32_t, double>::GetRollingFunction2(func);
    CASE_NPY_INT64:

        return EmaBase<int64_t, double>::GetRollingFunction2(func);
    case NPY_UINT8:
        return EmaBase<uint8_t, double>::GetRollingFunction2(func);
    case NPY_UINT16:
        return EmaBase<uint16_t, double>::GetRollingFunction2(func);
    CASE_NPY_UINT64:

        return EmaBase<uint64_t, double>::GetRollingFunction2(func);
    }

    return NULL;
}

// Basic call for rolling
// Arg1: input numpy array
// Arg2: rolling function
// Arg2: window size
//
// Output: numpy array with rolling calculation
//
PyObject * Rolling(PyObject * self, PyObject * args)
{
    PyArrayObject * inArrObject = NULL;
    int64_t func = 0;
    int64_t param1 = 0;

    if (! PyArg_ParseTuple(args, "O!LL", &PyArray_Type, &inArrObject, &func, &param1))
    {
        return NULL;
    }

    // TODO: determine based on function
    int32_t numpyOutType = NPY_FLOAT64;

    // In case user passes in sliced array or reversed array
    PyArrayObject * inArr = EnsureContiguousArray(inArrObject);
    if (! inArr)
        return NULL;

    int32_t dType = PyArray_TYPE(inArr);

    PyArrayObject * outArray = NULL;
    int64_t size = ArrayLength(inArr);
    ROLLING_FUNC pRollingFunc;

    numpyOutType = NPY_INT64;

    if (func >= 100)
    {
        pRollingFunc = GetRollingFunction2(func, dType);

        // Always want some sort of float
        numpyOutType = NPY_DOUBLE;
        if (dType == NPY_FLOAT)
        {
            numpyOutType = NPY_FLOAT;
        }
        if (dType == NPY_LONGDOUBLE)
        {
            numpyOutType = NPY_LONGDOUBLE;
        }
    }
    else
    {
        pRollingFunc = GetRollingFunction(func, dType);

        // Always want some sort of int64 or float
        numpyOutType = NPY_INT64;
        if (dType == NPY_FLOAT)
        {
            numpyOutType = NPY_FLOAT;
        }
        if (dType == NPY_DOUBLE)
        {
            numpyOutType = NPY_DOUBLE;
        }
        if (dType == NPY_LONGDOUBLE)
        {
            numpyOutType = NPY_LONGDOUBLE;
        }
    }

    if (pRollingFunc)
    {
        // Dont bother allocating if we cannot call the function
        outArray = AllocateNumpyArray(1, (npy_intp *)&size, numpyOutType);

        if (outArray)
        {
            pRollingFunc(PyArray_BYTES(inArr), PyArray_BYTES(outArray), size, param1);
        }
    }
    else
    {
        Py_INCREF(Py_None);
        outArray = (PyArrayObject *)Py_None;
    }

    // cleanup if we made a copy
    if (inArr != inArrObject)
        Py_DecRef((PyObject *)inArr);
    return (PyObject *)outArray;
}

//===================================================================
//
//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// K = data type for indexing (often int32_t* or int8_t*)
template <typename T, typename U, typename K>
static void CumSum(void * pKeyT, void * pAccumBin, void * pColumn, int64_t numUnique, int64_t totalInputRows,
                   void * pTime1, // not used
                   int8_t * pIncludeMask, int8_t * pResetMask, double windowSize1)
{
    T * pSrc = (T *)pColumn;
    U * pDest = (U *)pAccumBin;
    K * pKey = (K *)pKeyT;

    U Invalid = GET_INVALID(pDest[0]);

    int64_t windowSize = static_cast<int64_t>(windowSize1);

    LOGGING("cumsum %lld  %lld  %lld  %p  %p\n", numUnique, totalInputRows, (int64_t)Invalid, pIncludeMask, pResetMask);

    // Alloc a workspace
    int64_t size = (numUnique + GB_BASE_INDEX) * sizeof(U);
    U * pWorkSpace = (U *)WORKSPACE_ALLOC(size);

    // Default every bin to 0, including floats
    memset(pWorkSpace, 0, size);

    if (pIncludeMask != NULL)
    {
        if (pResetMask != NULL)
        {
            // filter + reset loop
            for (ptrdiff_t i = 0; i < totalInputRows; i++)
            {
                K location = pKey[i];
                // Bin 0 is bad
                if (location >= GB_BASE_INDEX)
                {
                    if (pIncludeMask[i] != 0)
                    {
                        if (pResetMask[i])
                            pWorkSpace[location] = 0;
                        pWorkSpace[location] += (U)pSrc[i];
                    }
                    pDest[i] = pWorkSpace[location];
                }
                else
                {
                    pDest[i] = Invalid;
                }
            }
        }
        else
        {
            // filter loop
            for (ptrdiff_t i = 0; i < totalInputRows; i++)
            {
                K location = pKey[i];
                // Bin 0 is bad
                if (location >= GB_BASE_INDEX)
                {
                    if (pIncludeMask[i] != 0)
                    {
                        // printf("adding %lld to %lld,", (int64_t)pSrc[location],
                        // (int64_t)pWorkSpace[location]);
                        pWorkSpace[location] += (U)pSrc[i];
                    }
                    pDest[i] = pWorkSpace[location];
                }
                else
                {
                    pDest[i] = Invalid;
                }
            }
        }
    }
    else
    {
        if (pResetMask != NULL)
        {
            // reset loop
            for (ptrdiff_t i = 0; i < totalInputRows; i++)
            {
                K location = pKey[i];

                // Bin 0 is bad
                if (location >= GB_BASE_INDEX)
                {
                    if (pResetMask[i])
                        pWorkSpace[location] = 0;
                    pWorkSpace[location] += (U)pSrc[i];
                    pDest[i] = pWorkSpace[location];
                }
                else
                {
                    pDest[i] = Invalid;
                }
            }
        }

        else
        {
            for (ptrdiff_t i = 0; i < totalInputRows; i++)
            {
                K location = pKey[i];

                // Bin 0 is bad
                if (location >= GB_BASE_INDEX)
                {
                    pWorkSpace[location] += (U)pSrc[i];
                    pDest[i] = pWorkSpace[location];
                }
                else
                {
                    // out of range bin printf("!!!%d --- %lld\n", i, (int64_t)Invalid);
                    pDest[i] = Invalid;
                }
            }
        }
    }

    WORKSPACE_FREE(pWorkSpace);
}

//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// K = key index pointer type (int32_t* or int8_t*)
template <typename T, typename U, typename K>
static void CumProd(void * pKeyT, void * pAccumBin, void * pColumn, int64_t numUnique, int64_t totalInputRows,
                    void * pTime1, // not used
                    int8_t * pIncludeMask, int8_t * pResetMask, double windowSize1)
{
    T * pSrc = (T *)pColumn;
    U * pDest = (U *)pAccumBin;
    K * pKey = (K *)pKeyT;

    U Invalid = GET_INVALID(pDest[0]);

    int64_t windowSize = static_cast<int64_t>(windowSize1);

    LOGGING("cumprod %lld  %lld  %p  %p\n", numUnique, totalInputRows, pIncludeMask, pResetMask);

    // Alloc a workspace
    int64_t size = (numUnique + GB_BASE_INDEX) * sizeof(U);
    U * pWorkSpace = (U *)WORKSPACE_ALLOC(size);

    // Default every bin to 1, including floats
    for (ptrdiff_t i = 0; i < (numUnique + GB_BASE_INDEX); i++)
    {
        pWorkSpace[i] = 1;
    }

    if (pIncludeMask != NULL)
    {
        if (pResetMask != NULL)
        {
            // filter + reset loop
            for (ptrdiff_t i = 0; i < totalInputRows; i++)
            {
                K location = pKey[i];
                // Bin 0 is bad
                if (location >= GB_BASE_INDEX)
                {
                    if (pIncludeMask[i])
                    {
                        if (pResetMask[i])
                            pWorkSpace[location] = 1;
                        pWorkSpace[location] *= pSrc[i];
                    }
                    pDest[i] = pWorkSpace[location];
                }
                else
                {
                    pDest[i] = Invalid;
                }
            }
        }
        else
        {
            // filter loop
            for (ptrdiff_t i = 0; i < totalInputRows; i++)
            {
                K location = pKey[i];
                // Bin 0 is bad
                if (location >= GB_BASE_INDEX)
                {
                    if (pIncludeMask[i])
                    {
                        pWorkSpace[location] *= pSrc[i];
                    }
                    pDest[i] = pWorkSpace[location];
                }
                else
                {
                    pDest[i] = Invalid;
                }
            }
        }
    }
    else
    {
        if (pResetMask != NULL)
        {
            // reset loop
            for (ptrdiff_t i = 0; i < totalInputRows; i++)
            {
                K location = pKey[i];

                // Bin 0 is bad
                if (location >= GB_BASE_INDEX)
                {
                    if (pResetMask[i])
                        pWorkSpace[location] = 1;
                    pWorkSpace[location] *= pSrc[i];
                    pDest[i] = pWorkSpace[location];
                }
                else
                {
                    pDest[i] = Invalid;
                }
            }
        }
        else
        {
            // plain
            for (ptrdiff_t i = 0; i < totalInputRows; i++)
            {
                K location = pKey[i];

                // if (location < 0 || location >= numUnique) {
                //   printf("!!! invalid location %d\n", location);
                //}

                // Bin 0 is bad
                if (location >= GB_BASE_INDEX)
                {
                    pWorkSpace[location] *= pSrc[i];
                    pDest[i] = pWorkSpace[location];
                }
                else
                {
                    pDest[i] = Invalid;
                }
            }
        }
    }

    WORKSPACE_FREE(pWorkSpace);
}

//-------------------------------------------------------------------
// T = data type as input (NOT USED)
// U = data type as output
// K = key index pointer type (int32_t* or int8_t*)
template <typename U, typename K>
static void FindNth(void * pKeyT, void * pAccumBin, void * pColumn, int64_t numUnique, int64_t totalInputRows,
                    void * pTime1, // not used
                    int8_t * pIncludeMask, int8_t * pResetMask, double windowSize1)
{
    U * pDest = (U *)pAccumBin;
    K * pKey = (K *)pKeyT;

    LOGGING("FindNth %lld  %lld  %p  %p\n", numUnique, totalInputRows, pIncludeMask, pResetMask);

    // Alloc a workspace
    int64_t size = (numUnique + GB_BASE_INDEX) * sizeof(U);
    U * pWorkSpace = (U *)WORKSPACE_ALLOC(size);

    memset(pWorkSpace, 0, size);

    if (pIncludeMask != NULL)
    {
        // filter loop
        for (ptrdiff_t i = 0; i < totalInputRows; i++)
        {
            K location = pKey[i];
            // Bin 0 is bad
            if (location >= GB_BASE_INDEX && pIncludeMask[i])
            {
                pWorkSpace[location]++;
                pDest[i] = pWorkSpace[location];
            }
            else
            {
                pDest[i] = 0;
            }
        }
    }
    else
    {
        // plain
        for (ptrdiff_t i = 0; i < totalInputRows; i++)
        {
            K location = pKey[i];

            // Bin 0 is bad
            if (location >= GB_BASE_INDEX)
            {
                pWorkSpace[location]++;
                pDest[i] = pWorkSpace[location];
            }
            else
            {
                pDest[i] = 0;
            }
        }
    }

    WORKSPACE_FREE(pWorkSpace);
}

//-------------------------------------------------------------------
// T = data type as input
// U = data type as output (always double?)
// V = time datatype
// K = key index data type (int32_t* or int8_t*)
// thus <float, double, int64>
template <typename T, typename U, typename V, typename K>
class EmaByBase
{
public:
    EmaByBase(){};
    ~EmaByBase(){};

    //------------------------------
    // EmaDecay uses entire size: totalInputRows
    // AccumBin is output array
    // pColumn is the user's data
    // pIncludeMask might be NULL
    //    boolean mask
    // pResetMask might be NULL
    //    mask when to reset
    static void EmaDecay(void * pKeyT, void * pAccumBin, void * pColumn, int64_t numUnique, int64_t totalInputRows, void * pTime1,
                         int8_t * pIncludeMask, int8_t * pResetMask, double decayRate)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        V * pTime = (V *)pTime1;
        K * pKey = (K *)pKeyT;

        LOGGING("emadecay %lld  %lld %p\n", numUnique, totalInputRows, pTime1);

        // Alloc a workspace to store
        // lastEma -- type U
        // lastTime -- type V

        int64_t size = (numUnique + GB_BASE_INDEX) * sizeof(U);
        U * pLastEma = (U *)WORKSPACE_ALLOC(size);

        // Default every bin to 0, including floats
        memset(pLastEma, 0, size);

        size = (numUnique + GB_BASE_INDEX) * sizeof(V);
        V * pLastTime = (V *)WORKSPACE_ALLOC(size);

        // Default every LastTime bin to 0, including floats
        memset(pLastTime, 0, size);

        size = (numUnique + GB_BASE_INDEX) * sizeof(T);
        T * pLastValue = (T *)WORKSPACE_ALLOC(size);

        // Default every LastValue bin to 0, including floats
        memset(pLastValue, 0, size);

        U Invalid = GET_INVALID(pDest[0]);

        // Neiman's matlab loop below
        // if (p >= low && p < high) {
        //   p -= low;
        //   ema[i] = v[i] + lastEma[j][p] * exp(-decay * (t[i] - lastTime[j][p]));
        //   lastEma[j][p] = ema[i];
        //   lastTime[j][p] = t[i];
        //}

        if (pIncludeMask != NULL)
        {
            // filter loop
            if (pResetMask != NULL)
            {
                // filter + reset
                for (ptrdiff_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];
                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        T value = 0;

                        // NOTE: fill in last value
                        if (pIncludeMask[i] != 0)
                        {
                            value = pSrc[i];

                            if (pResetMask[i])
                            {
                                pLastEma[location] = 0;
                                pLastTime[location] = 0;
                            }
                            pLastEma[location] = value + pLastEma[location] * exp(-decayRate * (pTime[i] - pLastTime[location]));
                            pLastTime[location] = pTime[i];
                        }

                        pDest[i] = pLastEma[location];
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
            else
            {
                // filter only
                for (ptrdiff_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];

                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        T value = 0;

                        // NOTE: fill in last value
                        if (pIncludeMask[i] != 0)
                        {
                            value = pSrc[i];
                        }
                        else
                        {
                            // Acts like fill forward
                            LOGGING("fill forward location: %lld\n", (long long)location);
                            value = pLastValue[location];
                        }
                        pLastEma[location] = value + pLastEma[location] * exp(-decayRate * (pTime[i] - pLastTime[location]));
                        pLastTime[location] = pTime[i];
                        pLastValue[location] = value;
                        pDest[i] = pLastEma[location];
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
        }
        else
        {
            if (pResetMask != NULL)
            {
                // reset loop
                for (ptrdiff_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];

                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        if (pResetMask[i])
                        {
                            pLastEma[location] = 0;
                            pLastTime[location] = 0;
                        }
                        pLastEma[location] = pSrc[i] + pLastEma[location] * exp(-decayRate * (pTime[i] - pLastTime[location]));
                        pLastTime[location] = pTime[i];
                        pDest[i] = pLastEma[location];
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
            else
            {
                // plain loop (no reset / no filter)
                for (ptrdiff_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];

                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        // printf("inputs: %lf  %lf  %lf  %lf  %lf\n", (double)pSrc[i],
                        // (double)pLastEma[location], (double)-decayRate, (double)pTime[i],
                        // (double)pLastTime[location] );
                        pLastEma[location] = pSrc[i] + pLastEma[location] * exp(-decayRate * (pTime[i] - pLastTime[location]));
                        // printf("[%d][%d] %lf\n", i, (int32_t)location,
                        // (double)pLastEma[location]);
                        pLastTime[location] = pTime[i];
                        pDest[i] = pLastEma[location];
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
        }

        WORKSPACE_FREE(pLastTime);
        WORKSPACE_FREE(pLastEma);
        WORKSPACE_FREE(pLastValue);
    }

// Handle negative timeDelta
// We are now setting the lasttime to a very low number
// This can cause integer overflow
#define EMA_NORMAL_FUNC \
    double timeDelta = double(pTime[i] - pLastTime[location]); \
    double decayedWeight = exp(-decayRate * timeDelta); \
    if (timeDelta < 0) \
        decayedWeight = 0; \
    pLastEma[location] = value * (1 - decayedWeight) + pLastEma[location] * decayedWeight; \
    pLastTime[location] = pTime[i]; \
    pDest[i] = pLastEma[location];

    //-------------------------------------------------------------------------------
    //------------------------------
    // EmaNormal uses entire size: totalInputRows
    // AccumBin is output array
    // pColumn is the user's data
    // pIncludeMask might be NULL
    //    boolean mask
    // pResetMask might be NULL
    //    mask when to reset
    static void EmaNormal(void * pKeyT, void * pAccumBin, void * pColumn, int64_t numUnique, int64_t totalInputRows, void * pTime1,
                          int8_t * pIncludeMask, int8_t * pResetMask, double decayRate)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        V * pTime = (V *)pTime1;
        K * pKey = (K *)pKeyT;

        LOGGING("emanormal %lld  %lld %p\n", numUnique, totalInputRows, pTime1);

        // Alloc a workspace to store
        // lastEma -- type U
        // lastTime -- type V

        int64_t size = (numUnique + GB_BASE_INDEX) * sizeof(U);
        U * pLastEma = (U *)WORKSPACE_ALLOC(size);

        // Default every bin to 0, including floats
        // memset(pLastEma, 0, size);
        // the first value should be valid
        // go backwards so that first value is in there
        for (int64_t i = totalInputRows - 1; i >= 0; i--)
        {
            K location = pKey[i];
            T value = pSrc[i];
            pLastEma[location] = (U)value;
        }

        //-----------------------------
        size = (numUnique + GB_BASE_INDEX) * sizeof(V);
        V * pLastTime = (V *)WORKSPACE_ALLOC(size);

        size = (numUnique + GB_BASE_INDEX) * sizeof(T);
        T * pLastValue = (T *)WORKSPACE_ALLOC(size);

        // Default every LastValue bin to 0, including floats
        memset(pLastValue, 0, size);

        // Default every LastTime bin to 0, including floats
        // Set first time to very low value
        V largeNegative = 0;
        if (sizeof(V) == 4)
        {
            // largeNegative = -INFINITY;
            largeNegative = (V)0x80000000;
        }
        else
        {
            largeNegative = (V)0x8000000000000000LL;
        }
        for (int64_t i = 0; i < (numUnique + GB_BASE_INDEX); i++)
        {
            pLastTime[i] = largeNegative;
        }

        U Invalid = GET_INVALID(pDest[0]);

        if (pIncludeMask != NULL)
        {
            // filter loop
            if (pResetMask != NULL)
            {
                // filter + reset
                for (int64_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];
                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        T value = 0;

                        // NOTE: fill in last value
                        if (pIncludeMask[i] != 0)
                        {
                            value = pSrc[i];

                            if (pResetMask[i])
                            {
                                pLastEma[location] = 0;
                                pLastTime[location] = 0;
                            }
                            EMA_NORMAL_FUNC
                        }
                        else
                        {
                            pDest[i] = pLastEma[location];
                        }
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
            else
            {
                // filter only
                for (int64_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];

                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        T value = 0;

                        // NOTE: fill in last value
                        if (pIncludeMask[i] != 0)
                        {
                            value = pSrc[i];
                        }
                        else
                        {
                            // Acts like fill forward
                            LOGGING("fill forward location: %lld\n", (long long)location);
                            value = pLastValue[location];
                        }
                        EMA_NORMAL_FUNC
                        pLastValue[location] = value;
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
        }
        else
        {
            if (pResetMask != NULL)
            {
                // reset loop
                for (int64_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];

                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        if (pResetMask[i])
                        {
                            pLastEma[location] = 0;
                            pLastTime[location] = 0;
                        }
                        T value = pSrc[i];
                        EMA_NORMAL_FUNC
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
            else
            {
                // plain loop (no reset / no filter)
                for (int64_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];

                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        T value = pSrc[i];
                        // double DW = exp(-decayRate * (pTime[i] - pLastTime[location]));
                        // printf("**dw %lf  %lld  %lld\n", DW, (int64_t)pTime[i],
                        // (int64_t)pLastTime[location]);
                        EMA_NORMAL_FUNC
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
        }

        WORKSPACE_FREE(pLastTime);
        WORKSPACE_FREE(pLastEma);
        WORKSPACE_FREE(pLastValue);
    }

// NOTE: This routine not in use yet
#define EMA_WEIGHTED_FUNC \
    pLastEma[location] = value * (1 - decayedWeight) + pLastEma[location] * decayedWeight; \
    pDest[i] = pLastEma[location];

    //-------------------------------------------------------------------------------
    //------------------------------
    // EmaWeighted uses entire size: totalInputRows
    // AccumBin is output array
    // pColumn is the user's data
    // pIncludeMask might be NULL
    //    boolean mask
    // pResetMask might be NULL
    //    mask when to reset
    static void EmaWeighted(void * pKeyT, void * pAccumBin, void * pColumn, int64_t numUnique, int64_t totalInputRows,
                            void * pTime1, int8_t * pIncludeMask, int8_t * pResetMask, double decayedWeight)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        K * pKey = (K *)pKeyT;

        LOGGING("emaweighted %lld  %lld %p\n", numUnique, totalInputRows, pTime1);

        // Alloc a workspace to store
        // lastEma -- type U
        // lastTime -- type V

        int64_t size = (numUnique + GB_BASE_INDEX) * sizeof(U);
        U * pLastEma = (U *)WORKSPACE_ALLOC(size);

        // Default every bin to 0, including floats
        // memset(pLastEma, 0, size);

        // the first value should be valid
        // go backwards so that first value is in there
        for (int64_t i = totalInputRows - 1; i >= 0; i--)
        {
            K location = pKey[i];
            T value = pSrc[i];
            pLastEma[location] = (U)value;
        }

        U Invalid = GET_INVALID(pDest[0]);

        if (pIncludeMask != NULL)
        {
            // filter loop
            if (pResetMask != NULL)
            {
                // filter + reset
                for (int64_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];
                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        T value = 0;

                        // NOTE: fill in last value
                        if (pIncludeMask[i] != 0)
                        {
                            value = pSrc[i];

                            if (pResetMask[i])
                            {
                                pLastEma[location] = 0;
                            }
                        }

                        EMA_WEIGHTED_FUNC
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
            else
            {
                // filter only
                for (int64_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];

                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        T value = 0;

                        // NOTE: fill in last value
                        if (pIncludeMask[i] != 0)
                        {
                            value = pSrc[i];
                        }
                        EMA_WEIGHTED_FUNC
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
        }
        else
        {
            if (pResetMask != NULL)
            {
                // reset loop
                for (int64_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];

                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        if (pResetMask[i])
                        {
                            pLastEma[location] = 0;
                        }
                        T value = pSrc[i];
                        EMA_WEIGHTED_FUNC
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
            else
            {
                // plain loop (no reset / no filter)
                for (int64_t i = 0; i < totalInputRows; i++)
                {
                    K location = pKey[i];

                    // Bin 0 is bad
                    if (location >= GB_BASE_INDEX)
                    {
                        T value = pSrc[i];
                        // printf("**dw %d  %d   %lld %lld\n", i, (int)location,
                        // (int64_t)value, (int64_t)pLastEma[location]);
                        EMA_WEIGHTED_FUNC
                    }
                    else
                    {
                        pDest[i] = Invalid;
                    }
                }
            }
        }

        WORKSPACE_FREE(pLastEma);
    }

    //-------------------------------------------------------------------------------
    static EMA_BY_TWO_FUNC GetFunc(EMA_FUNCTIONS func)
    {
        switch (func)
        {
        case EMA_DECAY:
            return EmaDecay;
        case EMA_NORMAL:
            return EmaNormal;
        case EMA_WEIGHTED:
            return EmaWeighted;
        default:
            break;
        }
        return NULL;
    }
};

template <typename T, typename K>
static EMA_BY_TWO_FUNC GetEmaByStep2(int timeType, EMA_FUNCTIONS func)
{
    switch (timeType)
    {
    case NPY_FLOAT:
        return EmaByBase<T, double, float, K>::GetFunc(func);
    case NPY_DOUBLE:
        return EmaByBase<T, double, double, K>::GetFunc(func);
    case NPY_LONGDOUBLE:
        return EmaByBase<T, long double, long double, K>::GetFunc(func);
    CASE_NPY_INT32:
        return EmaByBase<T, double, int32_t, K>::GetFunc(func);
    CASE_NPY_INT64:

        return EmaByBase<T, double, int64_t, K>::GetFunc(func);
    CASE_NPY_UINT32:
        return EmaByBase<T, double, uint32_t, K>::GetFunc(func);
    CASE_NPY_UINT64:

        return EmaByBase<T, double, uint64_t, K>::GetFunc(func);
    }
    return NULL;
}

//------------------------------------------------------
// timeType is -1 for cumsum
// K is the iKey type (int8.int16,int32,int64)
template <typename K>
static EMA_BY_TWO_FUNC GetEmaByFunction(int inputType, int * outputType, int timeType, EMA_FUNCTIONS func)
{
    // only support EMADecay

    switch (func)
    {
    case EMA_CUMSUM:
        switch (inputType)
        {
        case NPY_FLOAT:
            *outputType = NPY_FLOAT32;
            return CumSum<float, float, K>;
        case NPY_DOUBLE:
            *outputType = NPY_FLOAT64;
            return CumSum<double, double, K>;
        case NPY_LONGDOUBLE:
            *outputType = NPY_FLOAT64;
            return CumSum<long double, long double, K>;
        case NPY_BOOL:
            *outputType = NPY_INT64;
            return CumSum<int8_t, int64_t, K>;
        case NPY_INT8:
            *outputType = NPY_INT64;
            return CumSum<int8_t, int64_t, K>;
        case NPY_INT16:
            *outputType = NPY_INT64;
            return CumSum<int16_t, int64_t, K>;
        CASE_NPY_INT32:
            *outputType = NPY_INT64;
            return CumSum<int32_t, int64_t, K>;
        CASE_NPY_INT64:

            *outputType = NPY_INT64;
            return CumSum<int64_t, int64_t, K>;

        case NPY_UINT8:
            *outputType = NPY_UINT64;
            return CumSum<uint8_t, uint64_t, K>;
        case NPY_UINT16:
            *outputType = NPY_UINT64;
            return CumSum<uint16_t, uint64_t, K>;
        CASE_NPY_UINT32:
            *outputType = NPY_UINT64;
            return CumSum<uint32_t, uint64_t, K>;
        CASE_NPY_UINT64:

            *outputType = NPY_UINT64;
            return CumSum<uint64_t, uint64_t, K>;
        }
        break;

    case EMA_CUMPROD:
        switch (inputType)
        {
        case NPY_FLOAT:
            *outputType = NPY_FLOAT32;
            return CumProd<float, float, K>;
        case NPY_DOUBLE:
            *outputType = NPY_FLOAT64;
            return CumProd<double, double, K>;
        case NPY_LONGDOUBLE:
            *outputType = NPY_FLOAT64;
            return CumProd<long double, long double, K>;
        case NPY_BOOL:
            *outputType = NPY_INT64;
            return CumProd<int8_t, int64_t, K>;
        case NPY_INT8:
            *outputType = NPY_INT64;
            return CumProd<int8_t, int64_t, K>;
        case NPY_INT16:
            *outputType = NPY_INT64;
            return CumProd<int16_t, int64_t, K>;
        CASE_NPY_INT32:
            *outputType = NPY_INT64;
            return CumProd<int32_t, int64_t, K>;
        CASE_NPY_INT64:

            *outputType = NPY_INT64;
            return CumProd<int64_t, int64_t, K>;

        case NPY_UINT8:
            *outputType = NPY_UINT64;
            return CumProd<uint8_t, uint64_t, K>;
        case NPY_UINT16:
            *outputType = NPY_UINT64;
            return CumProd<uint16_t, uint64_t, K>;
        CASE_NPY_UINT32:
            *outputType = NPY_UINT64;
            return CumProd<uint32_t, uint64_t, K>;
        CASE_NPY_UINT64:

            *outputType = NPY_UINT64;
            return CumProd<uint64_t, uint64_t, K>;
        }
        break;

    case EMA_FINDNTH:
        *outputType = NPY_INT32;
        return FindNth<int32_t, K>;
        break;

    case EMA_NORMAL:
    case EMA_WEIGHTED:
    case EMA_DECAY:
        *outputType = NPY_FLOAT64;
        switch (inputType)
        {
        case NPY_BOOL:
            return GetEmaByStep2<int8_t, K>(timeType, func);
        case NPY_FLOAT:
            return GetEmaByStep2<float, K>(timeType, func);
        case NPY_DOUBLE:
            return GetEmaByStep2<double, K>(timeType, func);
        case NPY_LONGDOUBLE:
            return GetEmaByStep2<long double, K>(timeType, func);
        case NPY_INT8:
            return GetEmaByStep2<int8_t, K>(timeType, func);
        case NPY_INT16:
            return GetEmaByStep2<int16_t, K>(timeType, func);
        CASE_NPY_INT32:
            return GetEmaByStep2<int32_t, K>(timeType, func);
        CASE_NPY_INT64:

            return GetEmaByStep2<int64_t, K>(timeType, func);
        case NPY_UINT8:
            return GetEmaByStep2<uint8_t, K>(timeType, func);
        case NPY_UINT16:
            return GetEmaByStep2<uint16_t, K>(timeType, func);
        CASE_NPY_UINT32:
            return GetEmaByStep2<uint32_t, K>(timeType, func);
        CASE_NPY_UINT64:

            return GetEmaByStep2<uint64_t, K>(timeType, func);
        }
        break;
    }

    return NULL;
}

//------------------------------------------------------
// Calculate the groupby
// BOTH groupby versions call this routine
// ** THIS ROUTINE IS CALLED FROM MULTIPLE CONCURRENT THREADS!
// i is the column number
void EmaByCall(void * pEmaBy, int64_t i)
{
    stEma32 * pstEma32 = (stEma32 *)pEmaBy;

    ArrayInfo * aInfo = pstEma32->aInfo;
    int64_t uniqueRows = pstEma32->uniqueRows;

    EMA_FUNCTIONS func = (EMA_FUNCTIONS)pstEma32->funcNum;

    // Data in was passed
    void * pDataIn = aInfo[i].pData;
    int64_t len = aInfo[i].ArrayLength;

    PyArrayObject * outArray = pstEma32->returnObjects[i].outArray;
    EMA_BY_TWO_FUNC pFunction = pstEma32->returnObjects[i].pFunction;
    int32_t numpyOutType = pstEma32->returnObjects[i].numpyOutType;
    TYPE_OF_FUNCTION_CALL typeCall = pstEma32->typeOfFunctionCall;

    if (outArray && pFunction)
    {
        void * pDataOut = PyArray_BYTES(outArray);
        LOGGING(
            "col %llu  ==> outsize %llu   len: %llu   numpy types %d --> %d   "
            "%d %d  ptr: %p\n",
            i, uniqueRows, len, aInfo[i].NumpyDType, numpyOutType, gNumpyTypeToSize[aInfo[i].NumpyDType],
            gNumpyTypeToSize[numpyOutType], pDataOut);

        // Accum the calculation
        EMA_BY_TWO_FUNC pFunctionX = pstEma32->returnObjects[i].pFunction;

        if (pFunctionX)
        {
            pFunctionX(pstEma32->pKey, (char *)pDataOut, (char *)pDataIn, uniqueRows, pstEma32->totalInputRows,

                       // params
                       pstEma32->pTime, pstEma32->inIncludeMask, pstEma32->inResetMask, pstEma32->doubleParam);

            pstEma32->returnObjects[i].returnObject = (PyObject *)outArray;
        }
        else
        {
            printf("!!!internal error EmaByCall");
        }
    }
    else
    {
        // TJD: memory leak?
        if (outArray)
        {
            printf("!!! deleting extra object\n");
            Py_DecRef((PyObject *)outArray);
        }

        LOGGING(
            "**skipping col %llu  ==> outsize %llu   len: %llu   numpy types "
            "%d --> %d   %d %d\n",
            i, uniqueRows, len, aInfo[i].NumpyDType, numpyOutType, gNumpyTypeToSize[aInfo[i].NumpyDType],
            gNumpyTypeToSize[numpyOutType]);
        pstEma32->returnObjects[i].returnObject = Py_None;
    }
}

//---------------------------------------------------------------
// Arg1 = LIST of numpy arrays which has the values to accumulate (often all the
// columns in a dataset) Arg2 = iKey = numpy array (int32_t) which has the index
// to the unique keys (ikey from MultiKeyGroupBy32) Arg3 = integer unique rows
// Arg4 = integer (function number to execute for cumsum, ema)
// Arg5 = params for function must be (decay/window, time, includemask,
// resetmask) Example: EmaAll32(array, ikey, 3, EMA_DECAY, (5.6, timeArray))
// Returns entire dataset per column
//
PyObject * EmaAll32(PyObject * self, PyObject * args)
{
    PyObject * inList1 = NULL;
    PyArrayObject * iKey = NULL;
    PyTupleObject * params = NULL;
    PyArrayObject * inTime = NULL;
    PyArrayObject * inIncludeMask = NULL;
    PyArrayObject * inResetMask = NULL;

    double doubleParam = 0.0;
    int64_t unique_rows = 0;
    int64_t funcNum = 0;

    if (! PyArg_ParseTuple(args, "OO!LLO", &inList1, &PyArray_Type, &iKey, &unique_rows, &funcNum, &params))
    {
        return NULL;
    }

    if (! PyTuple_Check(params))
    {
        PyErr_Format(PyExc_ValueError, "EmaAll32 params argument needs to be a tuple");
        return NULL;
    }

    int32_t iKeyType = PyArray_TYPE(iKey);

    switch (iKeyType)
    {
    case NPY_INT8:
    case NPY_INT16:
    CASE_NPY_INT32:
    CASE_NPY_INT64:

        break;
    default:
        PyErr_Format(PyExc_ValueError, "EmaAll32 key param must int8, int16, int32, int64");
        return NULL;
    }

    Py_ssize_t tupleSize = PyTuple_GET_SIZE(params);

    switch (tupleSize)
    {
    case 4:
        if (! PyArg_ParseTuple((PyObject *)params, "dOOO", &doubleParam, &inTime,
                               &inIncludeMask, // must be boolean for now or empty
                               &inResetMask))
        {
            return NULL;
        }

        // If they pass in NONE make it NULL
        if (inTime == (PyArrayObject *)Py_None)
        {
            inTime = NULL;
        }
        else if (! PyArray_Check(inTime))
        {
            PyErr_Format(PyExc_ValueError, "EmaAll32 inTime must be an array");
        }

        if (inIncludeMask == (PyArrayObject *)Py_None)
        {
            inIncludeMask = NULL;
        }
        else if (! PyArray_Check(inIncludeMask))
        {
            PyErr_Format(PyExc_ValueError, "EmaAll32 inIncludeMask must be an array");
        }

        if (inResetMask == (PyArrayObject *)Py_None)
        {
            inResetMask = NULL;
        }
        else if (! PyArray_Check(inResetMask))
        {
            PyErr_Format(PyExc_ValueError, "EmaAll32 inResetMask must be an array");
        }
        break;

    default:
        PyErr_Format(PyExc_ValueError, "EmaAll32 cannot parse arguments.  tuple size %lld\n", tupleSize);
        return NULL;
    }

    int64_t totalArrayLength = ArrayLength(iKey);

    if (inResetMask != NULL && (PyArray_TYPE(inResetMask) != 0 || ArrayLength(inResetMask) != totalArrayLength))
    {
        PyErr_Format(PyExc_ValueError, "EmaAll32 inResetMask must be a bool mask of same size");
        return NULL;
    }

    if (inIncludeMask != NULL && (PyArray_TYPE(inIncludeMask) != 0 || ArrayLength(inIncludeMask) != totalArrayLength))
    {
        PyErr_Format(PyExc_ValueError, "EmaAll32 inIncludeMask must be a bool mask of same size");
        return NULL;
    }

    if (inTime != NULL &&
        (PyArray_TYPE(inTime) < NPY_INT || PyArray_TYPE(inTime) > NPY_LONGDOUBLE || ArrayLength(inTime) != totalArrayLength))
    {
        PyErr_Format(PyExc_ValueError, "EmaAll32 inTime must be a 32 or 64 bit value of same size");
        return NULL;
    }

    int32_t numpyInType2 = ObjectToDtype(iKey);

    int64_t totalItemSize = 0;
    ArrayInfo * aInfo = BuildArrayInfo(inList1, (int64_t *)&tupleSize, &totalItemSize);

    if (! aInfo)
    {
        PyErr_Format(PyExc_ValueError, "EmaAll32 failed to produce aInfo");
        return NULL;
    }

    LOGGING("Ema started %llu  param:%lf\n", tupleSize, doubleParam);

    // Allocate the struct + ROOM at the end of struct for all the tuple objects
    // being produced
    stEma32 * pstEma32 = (stEma32 *)WORKSPACE_ALLOC((sizeof(stEma32) + 8 + sizeof(stEmaReturn)) * tupleSize);

    pstEma32->aInfo = aInfo;
    pstEma32->funcNum = (int32_t)funcNum;
    pstEma32->pKey = (int32_t *)PyArray_BYTES(iKey);
    pstEma32->tupleSize = tupleSize;
    pstEma32->uniqueRows = unique_rows;
    pstEma32->totalInputRows = totalArrayLength;
    pstEma32->doubleParam = doubleParam;
    pstEma32->pTime = inTime == NULL ? NULL : PyArray_BYTES(inTime);
    pstEma32->inIncludeMask = inIncludeMask == NULL ? NULL : (int8_t *)PyArray_BYTES(inIncludeMask);
    pstEma32->inResetMask = inResetMask == NULL ? NULL : (int8_t *)PyArray_BYTES(inResetMask);
    pstEma32->typeOfFunctionCall = TYPE_OF_FUNCTION_CALL::ANY_GROUPBY_FUNC;

    LOGGING("Ema unique %lld  total: %lld  arrays: %p %p %p\n", unique_rows, totalArrayLength, pstEma32->pTime,
            pstEma32->inIncludeMask, pstEma32->inResetMask);

    // Allocate all the memory and output arrays up front since Python is single
    // threaded
    for (int i = 0; i < tupleSize; i++)
    {
        // TODO: determine based on function
        int32_t numpyOutType = -1;

        EMA_BY_TWO_FUNC pFunction = NULL;
        switch (iKeyType)
        {
        case NPY_INT8:
            pFunction = GetEmaByFunction<int8_t>(aInfo[i].NumpyDType, &numpyOutType, inTime == NULL ? -1 : PyArray_TYPE(inTime),
                                                 (EMA_FUNCTIONS)funcNum);
            break;
        case NPY_INT16:
            pFunction = GetEmaByFunction<int16_t>(aInfo[i].NumpyDType, &numpyOutType, inTime == NULL ? -1 : PyArray_TYPE(inTime),
                                                  (EMA_FUNCTIONS)funcNum);
            break;
        CASE_NPY_INT32:
            pFunction = GetEmaByFunction<int32_t>(aInfo[i].NumpyDType, &numpyOutType, inTime == NULL ? -1 : PyArray_TYPE(inTime),
                                                  (EMA_FUNCTIONS)funcNum);
            break;
        CASE_NPY_INT64:

            pFunction = GetEmaByFunction<int64_t>(aInfo[i].NumpyDType, &numpyOutType, inTime == NULL ? -1 : PyArray_TYPE(inTime),
                                                  (EMA_FUNCTIONS)funcNum);
            break;
        }

        PyArrayObject * outArray = NULL;

        // Dont bother allocating if we cannot call the function
        if (pFunction)
        {
            // Allocate the output size for each column
            outArray = AllocateNumpyArray(1, (npy_intp *)&totalArrayLength, numpyOutType);
            LOGGING("[%d] Allocated output array size %lld for type %d  ptr:%p\n", i, totalArrayLength, numpyOutType,
                    PyArray_BYTES(outArray, 0));
        }
        else
        {
            LOGGING("Failed to find function %llu for type %d\n", funcNum, numpyOutType);
            printf("Failed to find function %llu for type %d\n", funcNum, numpyOutType);
        }

        pstEma32->returnObjects[i].outArray = outArray;
        pstEma32->returnObjects[i].pFunction = pFunction;
        pstEma32->returnObjects[i].returnObject = Py_None;
        pstEma32->returnObjects[i].numpyOutType = numpyOutType;
    }

    // Do the work (multithreaded)
    g_cMathWorker->WorkGroupByCall(EmaByCall, pstEma32, tupleSize);

    LOGGING("!!ema done %llu\n", tupleSize);

    // New reference
    PyObject * returnTuple = PyTuple_New(tupleSize);

    // Fill in results
    for (int i = 0; i < tupleSize; i++)
    {
        PyObject * item = pstEma32->returnObjects[i].returnObject;

        if (item == Py_None)
            Py_INCREF(Py_None);

        // Set item will not change reference
        PyTuple_SET_ITEM(returnTuple, i, item);
        // printf("after ref %d  %llu\n", i, item->ob_refcnt);
    }

    // LOGGING("Return tuple ref %llu\n", returnTuple->ob_refcnt);
    WORKSPACE_FREE(pstEma32);
    FreeArrayInfo(aInfo);

    LOGGING("!!ema returning\n");

    return returnTuple;
}

//--------------------------------------------
// T is float or double
// x and out are 1 dimensional
// xp and yp are 2 dimensional
// N: first dimension length
// M: second dimension length (must be > 1)
template <typename T>
void mat_interp_extrap(void * xT, void * xpT, void * ypT, void * outT, int64_t N, int64_t M, int32_t clip)
{
    T * x = (T *)xT;
    T * xp = (T *)xpT;
    T * yp = (T *)ypT;
    T * out = (T *)outT;

    T mynan = std::numeric_limits<T>::quiet_NaN();

    if (! clip)
    {
        // auto increment xp and yp
        for (int64_t i = 0; i < N; ++i, xp += M, yp += M)
        {
            T xi = x[i];
            T result = mynan;

            if (xi == xi)
            {
                if (xi > xp[0])
                {
                    int64_t j = 1;
                    while (xi > xp[j] && j < M)
                        j++;
                    if (j == M)
                    {
                        T right_slope = (yp[M - 1] - yp[M - 2]) / (xp[M - 1] - xp[M - 2]);
                        result = yp[M - 1] + right_slope * (xi - xp[M - 1]);
                    }
                    else
                    {
                        // middle slope
                        result = (yp[j] - yp[j - 1]) * (xi - xp[j - 1]) / (xp[j] - xp[j - 1]) + yp[j - 1];
                    }
                }
                else
                {
                    T left_slope = (yp[1] - yp[0]) / (xp[1] - xp[0]);
                    result = yp[0] + left_slope * (xi - xp[0]);
                }
            }
            out[i] = result;
        }
    }
    else
    {
        // clipping
        for (int64_t i = 0; i < N; ++i, xp += M, yp += M)
        {
            T xi = x[i];
            T result = mynan;

            if (xi == xi)
            {
                if (xi > xp[0])
                {
                    int64_t j = 1;
                    while (xi > xp[j] && j < M)
                        j++;
                    if (j == M)
                    {
                        result = yp[M - 1];
                    }
                    else
                    {
                        // middle slope
                        result = (yp[j] - yp[j - 1]) * (xi - xp[j - 1]) / (xp[j] - xp[j - 1]) + yp[j - 1];
                    }
                }
                else
                {
                    result = yp[0];
                }
            }
            out[i] = result;
        }
    }
}

template <typename T>
void mat_interp(void * xT, void * xpT, void * ypT, void * outT, int64_t N, int64_t M, int32_t clip)
{
    T * x = (T *)xT;
    T * xp = (T *)xpT;
    T * yp = (T *)ypT;
    T * out = (T *)outT;

    T xp0 = xp[0];
    T yp0 = yp[0];
    T mynan = std::numeric_limits<T>::quiet_NaN();

    if (! clip)
    {
        for (int64_t i = 0; i < N; ++i)
        {
            T xi = x[i];
            T result = mynan;

            if (xi == xi)
            {
                if (xi > xp0)
                {
                    int64_t j = 1;
                    while (xi > xp[j] && j < M)
                        j++;
                    if (j == M)
                    {
                        T right_slope = (yp[M - 1] - yp[M - 2]) / (xp[M - 1] - xp[M - 2]);
                        result = yp[M - 1] + right_slope * (xi - xp[M - 1]);
                    }
                    else
                    {
                        // middle slope
                        result = (yp[j] - yp[j - 1]) * (xi - xp[j - 1]) / (xp[j] - xp[j - 1]) + yp[j - 1];
                    }
                }
                else
                {
                    T left_slope = (yp[1] - yp[0]) / (xp[1] - xp0);
                    result = yp[0] + left_slope * (xi - xp0);
                }
            }
            out[i] = result;
        }
    }
    else
    {
        for (int64_t i = 0; i < N; ++i)
        {
            T xi = x[i];
            T result = mynan;

            if (xi == xi)
            {
                if (xi > xp0)
                {
                    int64_t j = 1;
                    while (xi > xp[j] && j < M)
                        j++;
                    if (j == M)
                    {
                        // clipped
                        result = yp[M - 1];
                    }
                    else
                    {
                        // middle slope
                        result = (yp[j] - yp[j - 1]) * (xi - xp[j - 1]) / (xp[j] - xp[j - 1]) + yp[j - 1];
                    }
                }
                else
                {
                    // clipped
                    result = yp0;
                }
            }
            out[i] = result;
        }
    }
}

// struct for multithreading
struct stInterp
{
    char * x;
    char * xp;
    char * yp;
    char * out;
    int64_t N;
    int64_t M;
    int32_t mode;
    int32_t clip;
    int itemsize;
};

//-----------------------------
// multithreaded callback
// move pointers to start offset
// shrink N to what length is
bool InterpolateExtrap(void * callbackArgT, int core, int64_t start, int64_t length)
{
    stInterp * pInterp = (stInterp *)callbackArgT;
    int64_t M = pInterp->M;
    int64_t N = pInterp->N;
    int32_t clip = pInterp->clip;

    int64_t fixup = start * pInterp->itemsize;
    if (pInterp->mode == 2)
    {
        int64_t fixup2d = start * pInterp->itemsize * M;
        if (pInterp->itemsize == 8)
        {
            mat_interp_extrap<double>(pInterp->x + fixup, pInterp->xp + fixup2d, pInterp->yp + fixup2d, pInterp->out + fixup,
                                      length, M, clip);
        }
        else
        {
            mat_interp_extrap<float>(pInterp->x + fixup, pInterp->xp + fixup2d, pInterp->yp + fixup2d, pInterp->out + fixup,
                                     length, M, clip);
        }
    }
    else
    {
        if (pInterp->itemsize == 8)
        {
            mat_interp<double>(pInterp->x + fixup, pInterp->xp, pInterp->yp, pInterp->out + fixup, length, M, clip);
        }
        else
        {
            mat_interp<float>(pInterp->x + fixup, pInterp->xp, pInterp->yp, pInterp->out + fixup, length, M, clip);
        }
    }
    return true;
}

//--------------------------------------------
// arg1: arr: 1 dimensional double or float
// arg2: xp:  2 dimensional double or float
// arg3: yp:  2 dimensional double or float
// arg4: clip  set to 1 to clip (optional defaults to no clip)
// Returns 1 dimensional array of interpolated values
PyObject * InterpExtrap2d(PyObject * self, PyObject * args)
{
    PyArrayObject * arr;
    PyArrayObject * xp;
    PyArrayObject * yp;
    PyArrayObject * returnArray; // we allocate this
    int32_t clip = 0;
    int32_t mode = 0;

    if (! PyArg_ParseTuple(args, "O!O!O!|i", &PyArray_Type, &arr, &PyArray_Type, &xp, &PyArray_Type, &yp, &clip))
    {
        // If pyargparsetuple fails, it will set the error for us
        return NULL;
    }

    if (PyArray_NDIM(arr) > 1)
    {
        PyErr_Format(PyExc_ValueError, "The 1st argument must be 1 dimensional arrays");
        return NULL;
    }

    if ((PyArray_NDIM(xp) != PyArray_NDIM(yp)) || PyArray_NDIM(yp) > 2)
    {
        PyErr_Format(PyExc_ValueError, "The 2nd and 3rd argument must be the same dimensions");
        return NULL;
    }

    if (PyArray_NDIM(xp) == 2)
    {
        mode = 2;
    }
    else
    {
        mode = 1;
    }

    if (! (PyArray_FLAGS(xp) & PyArray_FLAGS(yp) & NPY_ARRAY_C_CONTIGUOUS))
    {
        PyErr_Format(PyExc_ValueError,
                     "The 2nd and 3rd argument must be row "
                     "major, contiguous 2 dimensional arrays");
        return NULL;
    }
    // NOTE: could check for strides also here

    int64_t N = PyArray_DIM(arr, 0);
    int64_t M = 0;

    if (mode == 2)
    {
        if ((N != PyArray_DIM(xp, 0)) || (N != PyArray_DIM(yp, 0)))
        {
            PyErr_Format(PyExc_ValueError, "The arrays must be the same size on the first dimension: %lld", N);
            return NULL;
        }
        M = PyArray_DIM(xp, 1);
        if (M != PyArray_DIM(yp, 1) || M < 2)
        {
            PyErr_Format(PyExc_ValueError,
                         "The 2nd and 3rd arrays must be the same size on the second "
                         "dimension: %lld",
                         M);
            return NULL;
        }
    }
    else
    {
        M = PyArray_DIM(xp, 0);
        if (M != PyArray_DIM(yp, 0) || M < 2)
        {
            PyErr_Format(PyExc_ValueError,
                         "The 2nd and 3rd arrays must be the same size on the first "
                         "dimension: %lld",
                         M);
            return NULL;
        }
    }

    // Accept all double or all floats
    int dtype = PyArray_TYPE(arr);
    if (dtype != PyArray_TYPE(xp) || dtype != PyArray_TYPE(yp))
    {
        PyErr_Format(PyExc_ValueError, "The arrays must all be the same type: %d", dtype);
        return NULL;
    }

    if (dtype != NPY_FLOAT64 && dtype != NPY_FLOAT32)
    {
        PyErr_Format(PyExc_ValueError, "The arrays must all be float32 or float64 not type: %d", dtype);
        return NULL;
    }

    // allocate a float or a double
    returnArray = AllocateLikeNumpyArray(arr, dtype);

    if (returnArray)
    {
        // copy params we will use into a struct on the stack
        stInterp interp;
        interp.itemsize = (int)PyArray_ITEMSIZE(xp);
        interp.x = PyArray_BYTES(arr);
        interp.xp = PyArray_BYTES(xp);
        interp.yp = PyArray_BYTES(yp);
        interp.out = PyArray_BYTES(returnArray);
        interp.N = N;
        interp.M = M;
        interp.mode = mode;
        interp.clip = clip;

        // release the threads
        g_cMathWorker->DoMultiThreadedChunkWork(N, InterpolateExtrap, &interp);
    }

    // return the output array by default
    return (PyObject *)returnArray;
}

//-----------------------------------------------------
//
//
// PyObject* EmaSimple(PyObject* self, PyObject* args) {
//  PyArrayObject* arrTime;
//  PyArrayObject* arrTime; xp;
//  PyArrayObject* yp;
//  PyArrayObject* returnArray; // we allocate this
//  int32_t clip = 0;
//  int32_t mode = 0;

//  if (!PyArg_ParseTuple(args, "O!O!O!|i",
//      &PyArray_Type, &arr,
//      &PyArray_Type, &xp,
//      &PyArray_Type, &yp,
//      &clip)) {

//     double timeDelta = double(pTime[i] - pLastTime[location]);
//     double decayedWeight = exp(-decayRate * timeDelta);
//     if (timeDelta < 0) decayedWeight = 0;
//     pLastEma[location] = value * (1 - decayedWeight) + pLastEma[location] *
//     decayedWeight; pLastTime[location] = pTime[i]; pDest[i] =
//     pLastEma[location];

//  }
//}
