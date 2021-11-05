#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "TimeWindow.h"
#define LOGGING(...)

typedef void (*TIMEWINDOW_FUNC)(void * pDataIn, void * pTimeIn, void * pDataOut, int64_t start, int64_t length,
                                int64_t windowSize);

enum TIMEWINDOW_FUNCTIONS
{
    TIMEWINDOW_SUM = 0,
    TIMEWINDOW_PROD = 1,
};

//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// V = data type for time
template <typename T, typename U, typename V>
class TimeWindowBase
{
public:
    TimeWindowBase(){};
    ~TimeWindowBase(){};

    static void TimeWindowSum(void * pDataIn, void * pTimeIn, void * pDataOut, int64_t start, int64_t length, int64_t timeDelta)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pTime = (V *)pTimeIn;

        U currentSum = 0;

        for (int64_t i = start; i < (start + length); i++)
        {
            currentSum = pIn[i];
            int64_t timeIndex = i - 1;
            while (timeIndex >= 0)
            {
                // see if in time difference within range
                int64_t deltaTime = pTime[i] - pTime[timeIndex];
                if (deltaTime <= timeDelta)
                {
                    // keep tallying
                    currentSum += pIn[timeIndex];
                }
                else
                {
                    break;
                }
                timeIndex--;
            }
            pOut[i] = currentSum;
        }
    }

    static void TimeWindowProd(void * pDataIn, void * pTimeIn, void * pDataOut, int64_t start, int64_t length, int64_t timeDelta)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pTime = (V *)pTimeIn;

        U currentProd = 1;

        for (int64_t i = start; i < (start + length); i++)
        {
            currentProd = pIn[i];
            int64_t timeIndex = i - 1;
            while (timeIndex >= 0)
            {
                // see if in time difference within range
                int64_t deltaTime = pTime[i] - pTime[timeIndex];
                if (deltaTime <= timeDelta)
                {
                    // keep tallying
                    currentProd *= pIn[timeIndex];
                }
                else
                {
                    break;
                }
                timeIndex--;
            }
            pOut[i] = currentProd;
        }
    }

    static TIMEWINDOW_FUNC GeTimeWindowFunction(int64_t func)
    {
        switch (func)
        {
        case TIMEWINDOW_SUM:
            return TimeWindowSum;
        case TIMEWINDOW_PROD:
            return TimeWindowProd;
        }
        return NULL;
    }
};

TIMEWINDOW_FUNC GeTimeWindowFunction(int64_t func, int32_t inputType, int32_t * outputType)
{
    switch (inputType)
    {
    case NPY_BOOL:
        *outputType = NPY_INT64;
        return TimeWindowBase<int8_t, int64_t, int64_t>::GeTimeWindowFunction(func);
    case NPY_FLOAT:
        *outputType = NPY_FLOAT;
        return TimeWindowBase<float, float, int64_t>::GeTimeWindowFunction(func);
    case NPY_DOUBLE:
        *outputType = NPY_DOUBLE;
        return TimeWindowBase<double, double, int64_t>::GeTimeWindowFunction(func);
    case NPY_LONGDOUBLE:
        *outputType = NPY_LONGDOUBLE;
        return TimeWindowBase<long double, long double, int64_t>::GeTimeWindowFunction(func);
    case NPY_INT8:
        *outputType = NPY_INT64;
        return TimeWindowBase<int8_t, int64_t, int64_t>::GeTimeWindowFunction(func);
    case NPY_INT16:
        *outputType = NPY_INT64;
        return TimeWindowBase<int16_t, int64_t, int64_t>::GeTimeWindowFunction(func);
    CASE_NPY_INT32:
        *outputType = NPY_INT64;
        return TimeWindowBase<int32_t, int64_t, int64_t>::GeTimeWindowFunction(func);
    CASE_NPY_INT64:

        *outputType = NPY_INT64;
        return TimeWindowBase<int64_t, int64_t, int64_t>::GeTimeWindowFunction(func);
    case NPY_UINT8:
        *outputType = NPY_UINT64;
        return TimeWindowBase<uint8_t, uint64_t, int64_t>::GeTimeWindowFunction(func);
    case NPY_UINT16:
        *outputType = NPY_UINT64;
        return TimeWindowBase<uint16_t, uint64_t, int64_t>::GeTimeWindowFunction(func);
    CASE_NPY_UINT32:
        *outputType = NPY_UINT64;
        return TimeWindowBase<uint32_t, uint64_t, int64_t>::GeTimeWindowFunction(func);
    CASE_NPY_UINT64:

        *outputType = NPY_UINT64;
        return TimeWindowBase<uint64_t, uint64_t, int64_t>::GeTimeWindowFunction(func);
    }

    return NULL;
}

// Basic call for rolling
// Arg1: input numpy array
// Arg2: input numpy array time == assumes INT64 for now
// Arg3: timewindow function (must be int) see enum
// Arg4: time delta (must be int)
//
// Output: numpy array with rolling calculation
//
PyObject * TimeWindow(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr = NULL;
    PyArrayObject * timeArr = NULL;
    int64_t func = 0;
    int64_t param1 = 0;

    if (! PyArg_ParseTuple(args, "O!O!LL", &PyArray_Type, &inArr, &PyArray_Type, &timeArr, &func, &param1))
    {
        return NULL;
    }

    int32_t dType = PyArray_TYPE(inArr);

    PyArrayObject * outArray = NULL;
    int64_t arrayLength = ArrayLength(inArr);
    int64_t arrayLengthTime = ArrayLength(timeArr);

    // TODO: Check to make sure inArr and timeArr sizes are the same
    if (arrayLength != arrayLengthTime)
    {
        PyErr_Format(PyExc_ValueError, "TimeWindow array and time array must have same length: %lld  %lld", arrayLength,
                     arrayLengthTime);
        return NULL;
    }

    switch (PyArray_TYPE(timeArr))
    {
    CASE_NPY_INT64:

        break;
    default:
        PyErr_Format(PyExc_ValueError, "TimeWindow time array must be int64");
        return NULL;
    }

    TIMEWINDOW_FUNC pTimeWindowFunc;

    // determine based on function
    int32_t numpyOutType = NPY_FLOAT64;

    pTimeWindowFunc = GeTimeWindowFunction(func, dType, &numpyOutType);

    if (pTimeWindowFunc)
    {
        // Dont bother allocating if we cannot call the function
        outArray = AllocateNumpyArray(1, (npy_intp *)&arrayLength, numpyOutType);

        if (outArray)
        {
            // MT callback
            struct TWCallbackStruct
            {
                TIMEWINDOW_FUNC func;
                void * pDataIn;
                void * pTimeIn;
                void * pDataOut;
                int64_t timeDelta;
            };

            TWCallbackStruct stTWCallback;

            // This is the routine that will be called back from multiple threads
            auto lambdaTWCallback = [](void * callbackArgT, int core, int64_t start, int64_t length) -> bool
            {
                TWCallbackStruct * callbackArg = (TWCallbackStruct *)callbackArgT;

                // printf("[%d] TW %lld %lld\n", core, start, length);

                callbackArg->func(callbackArg->pDataIn, callbackArg->pTimeIn, callbackArg->pDataOut, start, length,
                                  callbackArg->timeDelta);
                return true;
            };

            stTWCallback.func = pTimeWindowFunc;
            stTWCallback.pDataIn = PyArray_BYTES(inArr);
            stTWCallback.pTimeIn = PyArray_BYTES(timeArr);
            stTWCallback.pDataOut = PyArray_BYTES(outArray);
            stTWCallback.timeDelta = param1;

            g_cMathWorker->DoMultiThreadedChunkWork(arrayLength, lambdaTWCallback, &stTWCallback);

            return (PyObject *)outArray;
        }

        // out of memory
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}
