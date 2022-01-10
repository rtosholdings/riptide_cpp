#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "MultiKey.h"
#include "GroupBy.h"
#include "Sort.h"

#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <vector>

#define LOGGING(...)
//#define LOGGING printf

enum GB_FUNCTIONS
{
    GB_SUM = 0,
    GB_MEAN = 1,
    GB_MIN = 2,
    GB_MAX = 3,

    // STD uses VAR with the param set to 1
    GB_VAR = 4,
    GB_STD = 5,

    GB_NANSUM = 50,
    GB_NANMEAN = 51,
    GB_NANMIN = 52,
    GB_NANMAX = 53,
    GB_NANVAR = 54,
    GB_NANSTD = 55,

    GB_FIRST = 100,
    GB_NTH = 101,
    GB_LAST = 102,

    GB_MEDIAN = 103, // auto handles nan
    GB_MODE = 104,   // auto handles nan
    GB_TRIMBR = 105, // auto handles nan

    // All int/uints output upgraded to int64_t
    // Output is all elements (not just grouped)
    // Input must be same length
    GB_ROLLING_SUM = 200,
    GB_ROLLING_NANSUM = 201,

    GB_ROLLING_DIFF = 202,
    GB_ROLLING_SHIFT = 203,
    GB_ROLLING_COUNT = 204,
    GB_ROLLING_MEAN = 205,
    GB_ROLLING_NANMEAN = 206,
};

// Overloads to handle case of bool
inline bool MEDIAN_SPLIT(bool X, bool Y)
{
    return (X | Y);
}
inline int8_t MEDIAN_SPLIT(int8_t X, int8_t Y)
{
    return (X + Y) / 2;
}
inline uint8_t MEDIAN_SPLIT(uint8_t X, uint8_t Y)
{
    return (X + Y) / 2;
}
inline int16_t MEDIAN_SPLIT(int16_t X, int16_t Y)
{
    return (X + Y) / 2;
}
inline uint16_t MEDIAN_SPLIT(uint16_t X, uint16_t Y)
{
    return (X + Y) / 2;
}
inline int32_t MEDIAN_SPLIT(int32_t X, int32_t Y)
{
    return (X + Y) / 2;
}
inline uint32_t MEDIAN_SPLIT(uint32_t X, uint32_t Y)
{
    return (X + Y) / 2;
}
inline int64_t MEDIAN_SPLIT(int64_t X, int64_t Y)
{
    return (X + Y) / 2;
}
inline uint64_t MEDIAN_SPLIT(uint64_t X, uint64_t Y)
{
    return (X + Y) / 2;
}
inline float MEDIAN_SPLIT(float X, float Y)
{
    return (X + Y) / 2;
}
inline double MEDIAN_SPLIT(double X, double Y)
{
    return (X + Y) / 2;
}
inline long double MEDIAN_SPLIT(long double X, long double Y)
{
    return (X + Y) / 2;
}

// taken from multiarray/scalartypesc.src
//// need this routine
//// the numpy routine can also search for registered types
// int
//_typenum_fromtypeobj(PyObject *type, int user)
//{
//   int typenum, i;
//
//   typenum = NPY_NOTYPE;
//   i = get_typeobj_idx((PyTypeObject*)type);
//   if (i >= 0) {
//      typenum = typeobjects[i].typenum;
//   }
//
//   if (!user) {
//      return typenum;
//   }
//   /* Search any registered types */
//   i = 0;
//   while (i < NPY_NUMUSERTYPES) {
//      if (type == (PyObject *)(userdescrs[i]->typeobj)) {
//         typenum = i + NPY_USERDEF;
//         break;
//      }
//      i++;
//   }
//   return typenum;
//}
//

static size_t ntrimbad(size_t nsamp)
{
    double BADRATE = 0.0001;
    double CONFLEVEL = 0.996;
    double NSTDEV = 2.6521; // confidence level 0.996
    double nbase = BADRATE * nsamp;
    size_t nsafe = (size_t)ceil(nbase);

    if (nbase > 50)
        nsafe = (size_t)ceil(nbase + NSTDEV * sqrt(nbase));
    else
    {
        double ptot = exp(-nbase);
        double addon = ptot;
        size_t i = 0;
        while (ptot < CONFLEVEL && i < 100)
        {
            i = i + 1;
            addon *= nbase / i;
            ptot += addon;
        }
        nsafe = nsafe < i ? i : nsafe;
    }
    if (nsafe == 0)
        nsafe = 1;
    if (nsafe > nsamp)
        nsafe = nsamp;
    return nsafe;
}

template <typename T>
static T get_nth_element(T * first, T * last, size_t n)
{
    // return nth element in sorted version of [first,last); array is changed(!)
    // by function
    std::nth_element<T *>(first, first + n, last);
    return *(first + n);
}

template <typename T>
static size_t strip_nans(T * x, size_t n)
{
    // move NaNs to end of array; return value is number of non-NaN
    T tmp;
    size_t i = 0, j = n;
    while (i < j)
    {
        if (x[i] == x[i])
            ++i;
        else
        {
            --j;
            if (x[j] == x[j])
            {
                tmp = x[j];
                x[j] = x[i];
                x[i++] = tmp;
            }
        }
    }
    return i;
}

//-------------------------------------------------------------------
// Defined as a macro so we can change this in one place
// The algos look for a range to operate on.  This range can be used when
// multi-threading.
#define ACCUM_INNER_LOOP(_index, _binLow, _binHigh) if (_index >= _binLow && _index < _binHigh)

//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// V - index type (int8_t, int16_t, int32_t, int64_t)
// thus <float, int32> converts a float to an int32
template <typename T, typename U, typename V>
class GroupByBase
{
public:
    GroupByBase(){};
    ~GroupByBase(){};

    // Pass in two vectors and return one vector
    // Used for operations like C = A + B
    // typedef void(*ANY_TWO_FUNC)(void* pDataIn, void* pDataIn2, void* pDataOut,
    // int64_t len, int32_t scalarMode); typedef void(*ANY_ONE_FUNC)(void* pDataIn,
    // void* pDataOut, int64_t len);

    static void AccumSum(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                         int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                pOut[index] += (U)pIn[i];
            }
        }
    }

    // This routine is only for float32.  It will upcast to float64, add up all
    // the numbers, then convert back to float32
    static void AccumSumFloat(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                              int64_t binHigh, int64_t pass, void * pDataTmp)
    {
        float * pIn = (float *)pDataIn;
        V * pIndex = (V *)pIndexT;
        double * pOutAccum = static_cast<double *>(pDataTmp);

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOutAccum, 0, sizeof(double) * (binHigh - binLow));
        }

        // Main loop
        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                pOutAccum[index - binLow] += (double)pIn[i];
            }
        }

        // Downcast from double to single
        float * pOut = (float *)pDataOut;
        for (int64_t i = binLow; i < binHigh; i++)
        {
            pOut[i] = (float)pOutAccum[i - binLow];
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanSum(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                            int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        // get invalid
        T invalid = GET_INVALID(pIn[0]);

        if (invalid == invalid)
        {
            // non-float path
            for (int64_t i = 0; i < len; i++)
            {
                V index = pIndex[i];

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    T temp = pIn[i];
                    if (temp != invalid)
                    {
                        pOut[index] += (U)temp;
                    }
                }
            }
        }
        else
        {
            // float path
            for (int64_t i = 0; i < len; i++)
            {
                V index = pIndex[i];

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    T temp = pIn[i];
                    if (temp == temp)
                    {
                        pOut[index] += (U)temp;
                    }
                }
            }
        }
    }

    // This is for float only
    // $TODO: Adapted from AccumSumFloat: should refactor this common behavior and
    // potentially cache the working buffer in between invocations.
    static void AccumNanSumFloat(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                                 int64_t binHigh, int64_t pass, void * pDataTmp)
    {
        float * pIn = (float *)pDataIn;
        V * pIndex = (V *)pIndexT;
        double * pOutAccum = static_cast<double *>(pDataTmp);

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOutAccum, 0, sizeof(double) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                float temp = pIn[i];
                if (temp == temp)
                {
                    pOutAccum[index - binLow] += (double)temp;
                }
            }
        }

        // Downcast from double to single
        float * pOut = (float *)pDataOut;
        for (int64_t i = binLow; i < binHigh; i++)
        {
            pOut[i] = (float)pOutAccum[i - binLow];
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumMin(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                         int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        // Fill with invalid?
        U invalid = GET_INVALID(pOut[0]);

        if (pass <= 0)
        {
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i] = invalid;
            }
        }

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T temp = pIn[i];
                if (pCountOut[index] == 0)
                {
                    // first time
                    pOut[index] = (U)temp;
                    pCountOut[index] = 1;
                }
                else if ((U)temp < pOut[index])
                {
                    pOut[index] = (U)temp;
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanMin(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                            int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        // Fill with NaNs
        U invalid = GET_INVALID(pOut[0]);
        if (pass <= 0)
        {
            // printf("NanMin clearing at %p  %lld  %lld\n", pOut, binLow, binHigh);
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i] = invalid;
            }
        }

        if (invalid == invalid)
        {
            // non-float path
            for (int64_t i = 0; i < len; i++)
            {
                V index = pIndex[i];

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    T temp = pIn[i];
                    // Note filled with nans, so comparing with nans
                    if (pOut[index] != invalid)
                    {
                        if (pOut[index] > temp)
                        {
                            pOut[index] = temp;
                        }
                    }
                    else
                    {
                        pOut[index] = temp;
                    }
                }
            }
        }
        else
        {
            // float path
            for (int64_t i = 0; i < len; i++)
            {
                V index = pIndex[i];

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    T temp = pIn[i];
                    // Note filled with nans, so comparing with nans
                    if (pOut[index] == pOut[index])
                    {
                        if (pOut[index] > temp)
                        {
                            pOut[index] = temp;
                        }
                    }
                    else
                    {
                        pOut[index] = temp;
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumMax(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                         int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        // Fill with invalid?
        U invalid = GET_INVALID(pOut[0]);
        if (pass <= 0)
        {
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i] = invalid;
            }
        }

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T temp = pIn[i];
                if (pCountOut[index] == 0)
                {
                    // first time
                    pOut[index] = (U)temp;
                    pCountOut[index] = 1;
                }
                else if ((U)temp > pOut[index])
                {
                    pOut[index] = (U)temp;
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanMax(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                            int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        // Fill with invalid?
        U invalid = GET_INVALID(pOut[0]);
        if (pass <= 0)
        {
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i] = invalid;
            }
        }

        if (invalid == invalid)
        {
            // non-float path
            for (int64_t i = 0; i < len; i++)
            {
                V index = pIndex[i];

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    T temp = pIn[i];
                    // Note filled with nans, so comparing with nans
                    if (pOut[index] != invalid)
                    {
                        if (pOut[index] < temp)
                        {
                            pOut[index] = temp;
                        }
                    }
                    else
                    {
                        pOut[index] = temp;
                    }
                }
            }
        }
        else
        {
            // float path
            for (int64_t i = 0; i < len; i++)
            {
                V index = pIndex[i];

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    T temp = pIn[i];
                    // Note filled with nans, so comparing with nans
                    if (pOut[index] == pOut[index])
                    {
                        if (pOut[index] < temp)
                        {
                            pOut[index] = temp;
                        }
                    }
                    else
                    {
                        pOut[index] = temp;
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumMean(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                          int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T temp = pIn[i];
                pOut[index] += (U)temp;
                pCountOut[index]++;
            }
        }

        if (pass < 0)
        {
            for (int64_t i = binLow; i < binHigh; i++)
            {
                if (pCountOut[i] > 0)
                {
                    pOut[i] /= (U)(pCountOut[i]);
                }
                else
                {
                    pOut[i] = GET_INVALID(pOut[i]);
                }
            }
        }
    }

    // Just for float32 since we can upcast
    //-------------------------------------------------------------------------------
    static void AccumMeanFloat(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                               int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOriginalOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        // Allocate pOut
        double * pOut = (double *)WORKSPACE_ALLOC(sizeof(double) * (binHigh - binLow));
        if (pOut)
        {
            if (pass <= 0)
            {
                // Clear out memory for our range
                memset(pOriginalOut + binLow, 0, sizeof(U) * (binHigh - binLow));
            }
            // copy over original values
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i - binLow] = (double)pOriginalOut[i];
            }

            for (int64_t i = 0; i < len; i++)
            {
                V index = pIndex[i];

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    double temp = pIn[i];
                    pOut[index - binLow] += (double)temp;
                    pCountOut[index]++;
                }
            }

            if (pass < 0)
            {
                for (int64_t i = binLow; i < binHigh; i++)
                {
                    if (pCountOut[i] > 0)
                    {
                        pOriginalOut[i] = (U)(pOut[i - binLow] / (double)(pCountOut[i]));
                    }
                    else
                    {
                        pOriginalOut[i] = GET_INVALID(pOriginalOut[i]);
                    }
                }
            }
            else
            {
                // copy over original values
                for (int64_t i = binLow; i < binHigh; i++)
                {
                    pOriginalOut[i] = (U)pOut[i - binLow];
                }
            }

            WORKSPACE_FREE(pOut);
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanMean(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                             int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOriginalOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        // Allocate pOut
        double * pOut = (double *)WORKSPACE_ALLOC(sizeof(double) * (binHigh - binLow));
        if (pOut)
        {
            if (pass <= 0)
            {
                // Clear out memory for our range
                memset(pOriginalOut + binLow, 0, sizeof(U) * (binHigh - binLow));
            }
            // copy over original values
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i - binLow] = (double)pOriginalOut[i];
            }

            for (int64_t i = 0; i < len; i++)
            {
                V index = pIndex[i];

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    T temp = pIn[i];
                    if (temp == temp)
                    {
                        pOut[index - binLow] += (double)temp;
                        pCountOut[index]++;
                    }
                }
            }
            if (pass < 0)
            {
                for (int64_t i = binLow; i < binHigh; i++)
                {
                    if (pCountOut[i] > 0)
                    {
                        pOriginalOut[i] = (U)(pOut[i - binLow] / (double)(pCountOut[i]));
                    }
                    else
                    {
                        pOriginalOut[i] = GET_INVALID(pOriginalOut[i]);
                    }
                }
            }
            else
            {
                // copy over original values
                for (int64_t i = binLow; i < binHigh; i++)
                {
                    pOriginalOut[i] = (U)pOut[i - binLow];
                }
            }
            WORKSPACE_FREE(pOut);
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanMeanFloat(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len,
                                  int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T temp = pIn[i];
                if (temp == temp)
                {
                    pOut[index] += (U)temp;
                    pCountOut[index]++;
                }
            }
        }

        if (pass < 0)
        {
            for (int64_t i = binLow; i < binHigh; i++)
            {
                if (pCountOut[i] > 0)
                {
                    pOut[i] /= (U)(pCountOut[i]);
                }
                else
                {
                    pOut[i] = GET_INVALID(pOut[i]);
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumVar(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                         int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        // TODO: optimize this for our range
        U * sumsquares = (U *)WORKSPACE_ALLOC(sizeof(U) * binHigh);
        memset(sumsquares, 0, sizeof(U) * binHigh);

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T temp = pIn[i];
                pOut[index] += (U)temp;
                pCountOut[index]++;
            }
        }

        for (int64_t i = binLow; i < binHigh; i++)
        {
            pOut[i] /= (U)(pCountOut[i]);
        }

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T temp = pIn[i];
                U diff = (U)temp - pOut[index];
                sumsquares[index] += (diff * diff);
            }
        }

        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCountOut[i] > 1)
            {
                pOut[i] = sumsquares[i] / (U)(pCountOut[i] - 1);
            }
            else
            {
                pOut[i] = GET_INVALID(pOut[i]);
            }
        }
        WORKSPACE_FREE(sumsquares);
    }

    //-------------------------------------------------------------------------------
    static void AccumNanVar(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                            int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T * pIn = (T *)pDataIn;
        U * pOut = (U *)pDataOut;
        V * pIndex = (V *)pIndexT;

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        U * sumsquares = (U *)WORKSPACE_ALLOC(sizeof(U) * binHigh);
        memset(sumsquares, 0, sizeof(U) * binHigh);

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T temp = pIn[i];
                if (temp == temp)
                {
                    pOut[index] += (U)temp;
                    pCountOut[index]++;
                }
            }
        }

        for (int64_t i = binLow; i < binHigh; i++)
        {
            pOut[i] /= (U)(pCountOut[i]);
        }

        for (int64_t i = 0; i < len; i++)
        {
            V index = pIndex[i];

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T temp = pIn[i];
                if (temp == temp)
                {
                    U diff = (U)temp - pOut[index];
                    sumsquares[index] += (diff * diff);
                }
            }
        }

        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCountOut[i] > 1)
            {
                pOut[i] = sumsquares[i] / (U)(pCountOut[i] - 1);
            }
            else
            {
                pOut[i] = GET_INVALID(pOut[i]);
            }
        }
        WORKSPACE_FREE(sumsquares);
    }

    //-------------------------------------------------------------------------------
    static void AccumStd(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                         int64_t binHigh, int64_t pass, void * pDataTmp)
    {
        U * pOut = (U *)pDataOut;
        AccumVar(pDataIn, pIndexT, pCountOut, pDataOut, len, binLow, binHigh, pass, pDataTmp);
        for (int64_t i = binLow; i < binHigh; i++)
        {
            pOut[i] = sqrt(pOut[i]);
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanStd(void * pDataIn, void * pIndexT, int32_t * pCountOut, void * pDataOut, int64_t len, int64_t binLow,
                            int64_t binHigh, int64_t pass, void * pDataTmp)
    {
        U * pOut = (U *)pDataOut;

        AccumNanVar(pDataIn, pIndexT, pCountOut, pDataOut, len, binLow, binHigh, pass, pDataTmp);
        for (int64_t i = binLow; i < binHigh; i++)
        {
            pOut[i] = sqrt(pOut[i]);
        }
    }

    //-------------------------------------------------------------------------------
    // V is is the INDEX Type
    static void AccumNth(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin, int64_t binLow,
                         int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        int32_t nth = (int32_t)funcParam;

        U invalid = GET_INVALID(pDest[0]);

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCount[i] > 0 && nth < pCount[i])
            {
                int32_t grpIndex = pFirst[i] + nth;
                int32_t bin = pGroup[grpIndex];
                pDest[i] = pSrc[bin];
            }
            else
            {
                pDest[i] = invalid;
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNthString(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                               int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        int32_t nth = (int32_t)funcParam;

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCount[i] > 0 && nth < pCount[i])
            {
                int32_t grpIndex = pFirst[i] + nth;
                int32_t bin = pGroup[grpIndex];
                memcpy(&pDest[i * itemSize], &pSrc[bin * itemSize], itemSize);
            }
            else
            {
                memset(&pDest[i * itemSize], 0, itemSize);
            }
        }
    }

    //-------------------------------------------------------------------------------
    // V is is the INDEX Type
    static void AccumFirst(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin, int64_t binLow,
                           int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        LOGGING("in accum first low: %lld  high: %lld   group:%p  first:%p  count:%p\n", binLow, binHigh, pGroup, pFirst, pCount);

        U invalid = GET_INVALID(pDest[0]);

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            // printf("[%lld]", i);
            if (pCount[i] > 0)
            {
                int32_t grpIndex = pFirst[i];
                // printf("(%d)", grpIndex);
                int32_t bin = pGroup[grpIndex];
                // printf("{%lld}", (int64_t)bin);
                pDest[i] = pSrc[bin];
            }
            else
            {
                pDest[i] = invalid;
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumFirstString(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                 int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCount[i] > 0)
            {
                int32_t grpIndex = pFirst[i];
                int32_t bin = pGroup[grpIndex];
                memcpy(&pDest[i * itemSize], &pSrc[bin * itemSize], itemSize);
            }
            else
            {
                memset(&pDest[i * itemSize], 0, itemSize);
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumLast(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin, int64_t binLow,
                          int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        U invalid = GET_INVALID(pDest[0]);
        // printf("last called %lld -- %llu %llu %llu\n", numUnique, sizeof(T),
        // sizeof(U), sizeof(V));

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            // printf("Last:  %d %d\n", (int)pFirst[i], (int)pCount[i]);
            // printf("Last2:  %d\n", (int)(pGroup[pFirst[i] + pCount[i] - 1]));
            if (pCount[i] > 0)
            {
                int32_t grpIndex = pFirst[i] + pCount[i] - 1;
                int32_t bin = pGroup[grpIndex];
                pDest[i] = pSrc[bin];
            }
            else
            {
                pDest[i] = invalid;
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumLastString(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCount[i] > 0)
            {
                int32_t grpIndex = pFirst[i] + pCount[i] - 1;
                int32_t bin = pGroup[grpIndex];
                memcpy(&pDest[i * itemSize], &pSrc[bin * itemSize], itemSize);
            }
            else
            {
                memset(&pDest[i * itemSize], 0, itemSize);
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    // TODO: Check if group is always int32_t
    static void AccumRollingSum(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        int32_t windowSize = (int32_t)funcParam;
        U invalid = GET_INVALID(pDest[0]);

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            int32_t start = pFirst[0];
            int32_t last = start + pCount[0];
            for (int j = start; j < last; j++)
            {
                int32_t index = pGroup[j];
                pDest[index] = invalid;
            }
            binLow++;
        }

        // negative window sizes not accepted yet
        if (windowSize < 0)
            return;

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            int32_t start = pFirst[i];
            int32_t last = start + pCount[i];

            U currentSum = 0;

            // Priming of the summation
            for (int j = start; j < last && j < (start + windowSize); j++)
            {
                int32_t index = pGroup[j];

                currentSum += (U)pSrc[index];
                pDest[index] = currentSum;
            }

            for (int j = start + windowSize; j < last; j++)
            {
                int32_t index = pGroup[j];

                currentSum += (U)pSrc[index];

                // subtract the item leaving the window
                currentSum -= (U)pSrc[pGroup[j - windowSize]];

                pDest[index] = currentSum;
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    // TODO: Check if group is always int32_t
    static void AccumRollingNanSum(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                   int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;

        // TODO allow int64_t in the future
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        int32_t windowSize = (int32_t)funcParam;
        U invalid = GET_INVALID(pDest[0]);

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            int32_t start = pFirst[0];
            int32_t last = start + pCount[0];
            for (int j = start; j < last; j++)
            {
                int32_t index = pGroup[j];
                pDest[index] = invalid;
            }
            binLow++;
        }

        // negative window sizes not accepted yet
        if (windowSize < 0)
            return;

        if (invalid == invalid)
        {
            // NOT FLOAT (integer based)
            // For all the bins we have to fill
            for (int64_t i = binLow; i < binHigh; i++)
            {
                int32_t start = pFirst[i];
                int32_t last = start + pCount[i];

                U currentSum = 0;

                // Priming of the summation
                for (int j = start; j < last && j < (start + windowSize); j++)
                {
                    int32_t index = pGroup[j];
                    U value = (U)pSrc[index];
                    if (value != invalid)
                    {
                        currentSum += value;
                    }
                    pDest[index] = currentSum;
                }

                for (int j = start + windowSize; j < last; j++)
                {
                    int32_t index = pGroup[j];

                    U value = (U)pSrc[index];
                    if (value != invalid)
                    {
                        currentSum += value;
                    }

                    // subtract the item leaving the window
                    value = (U)pSrc[pGroup[j - windowSize]];
                    if (value != invalid)
                    {
                        currentSum -= value;
                    }

                    pDest[index] = currentSum;
                }
            }
        }
        else
        {
            // FLOAT BASED
            for (int64_t i = binLow; i < binHigh; i++)
            {
                int32_t start = pFirst[i];
                int32_t last = start + pCount[i];

                U currentSum = 0;

                // Priming of the summation
                for (int j = start; j < last && j < (start + windowSize); j++)
                {
                    int32_t index = pGroup[j];
                    U value = (U)pSrc[index];
                    if (value == value)
                    {
                        currentSum += value;
                    }
                    pDest[index] = currentSum;
                }

                for (int j = start + windowSize; j < last; j++)
                {
                    int32_t index = pGroup[j];

                    U value = (U)pSrc[index];
                    if (value == value)
                    {
                        currentSum += value;
                    }

                    // subtract the item leaving the window
                    value = (U)pSrc[pGroup[j - windowSize]];
                    if (value == value)
                    {
                        currentSum -= value;
                    }

                    pDest[index] = currentSum;
                }
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    // TODO: Check if group is always int32_t
    static void AccumRollingMean(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                 int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        double * pDest = (double *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        int32_t windowSize = (int32_t)funcParam;
        U invalid = GET_INVALID((U)0);
        double invalid_out = GET_INVALID(pDest[0]);

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            int32_t start = pFirst[0];
            int32_t last = start + pCount[0];
            for (int j = start; j < last; j++)
            {
                int32_t index = pGroup[j];
                pDest[index] = invalid_out;
            }
            binLow++;
        }

        // negative window sizes not accepted yet
        if (windowSize < 0)
            return;

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            int32_t start = pFirst[i];
            int32_t last = start + pCount[i];

            double currentSum = 0;

            // Priming of the summation
            for (int j = start; j < last && j < (start + windowSize); j++)
            {
                int32_t index = pGroup[j];

                currentSum += pSrc[index];
                pDest[index] = currentSum / (j - start + 1);
            }

            for (int j = start + windowSize; j < last; j++)
            {
                int32_t index = pGroup[j];

                currentSum += pSrc[index];

                // subtract the item leaving the window
                currentSum -= pSrc[pGroup[j - windowSize]];

                pDest[index] = currentSum / windowSize;
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    // TODO: Check if group is always int32_t
    static void AccumRollingNanMean(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                    int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        double * pDest = (double *)pAccumBin;

        // TODO allow int64_t in the future
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        int32_t windowSize = (int32_t)funcParam;
        U invalid = GET_INVALID((U)0);
        double invalid_out = GET_INVALID(pDest[0]);

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            int32_t start = pFirst[0];
            int32_t last = start + pCount[0];
            for (int j = start; j < last; j++)
            {
                int32_t index = pGroup[j];
                pDest[index] = invalid_out;
            }
            binLow++;
        }

        // negative window sizes not accepted yet
        if (windowSize < 0)
            return;

        if (invalid == invalid)
        {
            // NOT FLOAT (integer based)
            // For all the bins we have to fill
            for (int64_t i = binLow; i < binHigh; i++)
            {
                int32_t start = pFirst[i];
                int32_t last = start + pCount[i];

                double currentSum = 0;
                double count = 0;

                // Priming of the summation
                for (int j = start; j < last && j < (start + windowSize); j++)
                {
                    int32_t index = pGroup[j];
                    U value = (U)pSrc[index];
                    if (value != invalid)
                    {
                        currentSum += value;
                        count++;
                    }
                    pDest[index] = count > 0 ? currentSum / count : invalid_out;
                }

                for (int j = start + windowSize; j < last; j++)
                {
                    int32_t index = pGroup[j];

                    U value = (U)pSrc[index];
                    if (value != invalid)
                    {
                        currentSum += value;
                        count++;
                    }

                    // subtract the item leaving the window
                    value = (U)pSrc[pGroup[j - windowSize]];
                    if (value != invalid)
                    {
                        currentSum -= value;
                        count--;
                    }

                    pDest[index] = count > 0 ? currentSum / count : invalid_out;
                }
            }
        }
        else
        {
            // FLOAT BASED
            for (int64_t i = binLow; i < binHigh; i++)
            {
                int32_t start = pFirst[i];
                int32_t last = start + pCount[i];

                double currentSum = 0;
                double count = 0;

                // Priming of the summation
                for (int j = start; j < last && j < (start + windowSize); j++)
                {
                    int32_t index = pGroup[j];
                    U value = (U)pSrc[index];
                    if (value == value)
                    {
                        currentSum += value;
                        count++;
                    }
                    pDest[index] = count > 0 ? currentSum / count : invalid_out;
                }

                for (int j = start + windowSize; j < last; j++)
                {
                    int32_t index = pGroup[j];

                    U value = (U)pSrc[index];
                    if (value == value)
                    {
                        currentSum += value;
                        count++;
                    }

                    // subtract the item leaving the window
                    value = (U)pSrc[pGroup[j - windowSize]];
                    if (value == value)
                    {
                        currentSum -= value;
                        count--;
                    }

                    pDest[index] = count > 0 ? currentSum / count : invalid_out;
                }
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    // TODO: Check if group is always int32_t
    static void AccumRollingCount(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                  int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        int32_t windowSize = (int32_t)funcParam;
        U invalid = GET_INVALID(pDest[0]);

        LOGGING("in rolling count %lld %lld  sizeofdest %lld\n", binLow, binHigh, sizeof(U));

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            int32_t start = pFirst[0];
            int32_t last = start + pCount[0];
            for (int j = start; j < last; j++)
            {
                int32_t index = pGroup[j];
                pDest[index] = invalid;
            }
            binLow++;
        }

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            int32_t start = pFirst[i];
            int32_t last = start + pCount[i];

            U currentSum = 0;

            // printf("in rolling count [%lld] %d %d\n", i, start, last);

            if (windowSize < 0)
            {
                for (int j = last - 1; j >= start; j--)
                {
                    int32_t index = pGroup[j];
                    pDest[index] = currentSum;
                    currentSum += 1;
                }
            }
            else
            {
                for (int j = start; j < last; j++)
                {
                    int32_t index = pGroup[j];
                    pDest[index] = currentSum;
                    currentSum += 1;
                }
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    // TODO: Check if group is always int32_t
    // NOTE: pDest/pAccumBin must be the size
    static void AccumRollingShift(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                  int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        int32_t windowSize = (int32_t)funcParam;
        U invalid = GET_INVALID(pDest[0]);

        // printf("binlow %lld,  binhigh %lld,  windowSize: %d\n", binLow, binHigh,
        // windowSize);

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            int32_t start = pFirst[0];
            int32_t last = start + pCount[0];
            for (int j = start; j < last; j++)
            {
                int32_t index = pGroup[j];
                pDest[index] = invalid;
            }
            binLow++;
        }

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            int32_t start = pFirst[i];
            int32_t last = start + pCount[i];

            if (windowSize >= 0)
            {
                // invalid for window
                for (int32_t j = start; j < last && j < (start + windowSize); j++)
                {
                    int32_t index = pGroup[j];
                    pDest[index] = invalid;
                }

                for (int32_t j = start + windowSize; j < last; j++)
                {
                    int32_t index = pGroup[j];
                    pDest[index] = (U)pSrc[pGroup[j - windowSize]];
                }
            }
            else
            {
                // invalid for window
                windowSize = -windowSize;
                last--;
                start--;
                // printf("bin[%lld]  start:%d  last:%d  windowSize:%d\n", i, start,
                // last, windowSize);

                for (int32_t j = last; j > start && j > (last - windowSize); j--)
                {
                    int32_t index = pGroup[j];
                    pDest[index] = invalid;
                }

                for (int32_t j = last - windowSize; j > start; j--)
                {
                    int32_t index = pGroup[j];
                    pDest[index] = (U)pSrc[pGroup[j + windowSize]];
                }
                // put it back to what it was
                windowSize = -windowSize;
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    // TODO: Check if group is always int32_t
    static void AccumRollingDiff(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                 int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        int32_t windowSize = (int32_t)funcParam;
        U invalid = GET_INVALID(pDest[0]);

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            int32_t start = pFirst[0];
            int32_t last = start + pCount[0];
            for (int j = start; j < last; j++)
            {
                int32_t index = pGroup[j];
                pDest[index] = invalid;
            }
            binLow++;
        }

        if (windowSize == 1)
        {
            // For all the bins we have to fill
            for (int64_t i = binLow; i < binHigh; i++)
            {
                int32_t start = pFirst[i];
                int32_t last = start + pCount[i];

                if (last > start)
                {
                    // Very first is invalid
                    int32_t index = pGroup[start];
                    U previous = (U)pSrc[index];
                    pDest[index] = invalid;

                    // Priming of the summation
                    for (int j = start + 1; j < last; j++)
                    {
                        index = pGroup[j];
                        U temp = (U)pSrc[index];
                        pDest[index] = temp - previous;
                        previous = temp;
                    }
                }
            }
        }
        else
        {
            // For all the bins we have to fill
            for (int64_t i = binLow; i < binHigh; i++)
            {
                int32_t start = pFirst[i];
                int32_t last = start + pCount[i];
                if (windowSize >= 0)
                {
                    // invalid for window
                    U previous = 0;

                    for (int j = start; j < last && j < (start + windowSize); j++)
                    {
                        int32_t index = pGroup[j];
                        pDest[index] = invalid;
                    }

                    for (int j = start + windowSize; j < last; j++)
                    {
                        int32_t index = pGroup[j];
                        U temp = (U)pSrc[index];
                        U previous = (U)pSrc[pGroup[j - windowSize]];
                        pDest[index] = temp - previous;
                    }
                }
                else
                {
                    // negative window size
                    windowSize = -windowSize;
                    last--;
                    start--;

                    for (int j = last; j > start && j > (last - windowSize); j--)
                    {
                        int32_t index = pGroup[j];
                        pDest[index] = invalid;
                    }

                    for (int j = last - windowSize; j > start; j--)
                    {
                        int32_t index = pGroup[j];
                        U temp = (U)pSrc[index];
                        U previous = (U)pSrc[pGroup[j + windowSize]];
                        pDest[index] = temp - previous;
                    }
                    // put it back to what it was
                    windowSize = -windowSize;
                }
            }
        }
    }

    //------------------------------
    // median does a sort for now -- but could use nth
    //
    static void AccumTrimMeanBR(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;

        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        // Alloc worst case
        T * pSort = (T *)WORKSPACE_ALLOC(totalInputRows * sizeof(T));

        U invalid = GET_INVALID(pDest[0]);

        LOGGING("TrimMean rows: %lld\n", totalInputRows);

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            int32_t index = pFirst[i];
            int32_t nCount = pCount[i];

            if (nCount == 0)
            {
                pDest[i] = invalid;
                continue;
            }

            // Copy over the items for this group
            for (int j = 0; j < nCount; j++)
            {
                pSort[j] = pSrc[pGroup[index + j]];
            }

            size_t n = strip_nans<T>(pSort, nCount);

            if (n == 0)
            {
                pDest[i] = invalid;
                continue;
            }

            size_t ntrim = ntrimbad(n);

            if (n <= 2 * ntrim)
            {
                pDest[i] = invalid;
                continue;
            }

            double sum = 0;
            size_t cnt = 0;
            T lb = get_nth_element(pSort, pSort + n, ntrim - 1);
            T ub = get_nth_element(pSort, pSort + n, n - ntrim);

            if (lb <= ub)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    if (pSort[i] >= lb && pSort[i] <= ub)
                    {
                        sum += pSort[i];
                        ++cnt;
                    }
                }
                pDest[i] = cnt ? (T)(sum / cnt) : invalid;
            }
            else
            {
                pDest[i] = invalid;
            }
        }

        WORKSPACE_FREE(pSort);
    }

    //------------------------------
    // mode does a sort (auto handles nans?)
    // pGroup -> int8_t/16/32/64  (V typename)
    static void AccumMode(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin, int64_t binLow,
                          int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        // Alloc worst case
        T * pSort = (T *)WORKSPACE_ALLOC(totalInputRows * sizeof(T));
        U invalid = GET_INVALID(pDest[0]);

        // printf("Mode %llu\n", totalInputRows);

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            int32_t index = pFirst[i];
            int32_t nCount = pCount[i];

            if (nCount == 0)
            {
                pDest[i] = GET_INVALID(pDest[i]);
                continue;
            }

            // Copy over the items for this group
            for (int j = 0; j < nCount; j++)
            {
                pSort[j] = pSrc[pGroup[index + j]];
            }

            // BUGBUG: consider using rank
            // BUGBUG consider counting nans from back
            quicksort_<T>(pSort, nCount);

            // remove nans
            T * pEnd = pSort + nCount - 1;
            while (pEnd >= pSort)
            {
                if (*pEnd == *pEnd)
                    break;
                pEnd--;
            }

            nCount = (int32_t)((pEnd + 1) - pSort);

            if (nCount == 0)
            {
                // nothing valid
                pDest[i] = GET_INVALID(pDest[i]);
                continue;
            }

            U currValue = *pSort, bestValue = *pSort;
            int32_t currCount = 1, bestCount = 1;
            for (int32_t i = 1; i < nCount; ++i)
            {
                if (pSort[i] == currValue)
                    ++currCount;
                else
                {
                    currValue = pSort[i];
                    currCount = 1;
                }
                if (currCount > bestCount)
                {
                    bestValue = currValue;
                    bestCount = currCount;
                }
            }

            // copy the data over from pCount[i]
            pDest[i] = bestValue;
        }

        WORKSPACE_FREE(pSort);
    }

    //------------------------------
    // median does a sort
    // auto-nan
    // pGroup -> int8_t/16/32/64  (V typename)
    static void AccumMedian(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin, int64_t binLow,
                            int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        // pGroup -> int8_t/16/32/64  (V typename)

        // Alloc
        T * pSort = (T *)WORKSPACE_ALLOC(totalInputRows * sizeof(T));

        LOGGING("Median %llu  %lld  %lld  sizeof: %lld %lld %lld\n", totalInputRows, binLow, binHigh, sizeof(T), sizeof(U),
                sizeof(V));

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            int32_t index = pFirst[i];
            int32_t nCount = pCount[i];

            if (nCount == 0)
            {
                pDest[i] = GET_INVALID(pDest[i]);
                continue;
            }

            // Copy over the items for this group
            for (int j = 0; j < nCount; j++)
            {
                // printf("**[%lld][%d]  %d\n", i, index + j, (int32_t)pGroup[index +
                // j]);
                pSort[j] = pSrc[pGroup[index + j]];
            }

            // BUGBUG: consider using rank
            quicksort_<T>(pSort, nCount);

            // remove nans
            // walk backwards until we find a non-nan
            T * pEnd = pSort + nCount - 1;
            while (pEnd >= pSort)
            {
                // printf("checking %lf\n", (double)*pEnd);
                if (*pEnd == *pEnd)
                    break;
                pEnd--;
            }

            nCount = (int32_t)((pEnd + 1) - pSort);

            if (nCount == 0)
            {
                // nothing valid
                pDest[i] = GET_INVALID(pDest[i]);
                continue;
            }

            T middle = 0;

            // find the median...
            // what about nans?  nans should sort at the end
            if (nCount & 1)
            {
                middle = pSort[nCount / 2];
            }
            else
            {
                middle = MEDIAN_SPLIT(pSort[nCount / 2], pSort[(nCount / 2) - 1]);
            }

            // printf("step3 %lf, %lf ==> %lf\n", (double)pSort[nCount / 2],
            // (double)pSort[(nCount / 2) - 1], (double)middle);
            // copy the data over from pCount[i]
            pDest[i] = middle;
        }

        WORKSPACE_FREE(pSort);
    }

    //-------------------------------------------------------------------------------
    static void AccumMedianString(void * pColumn, void * pGroupT, int32_t * pFirst, int32_t * pCount, void * pAccumBin,
                                  int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T * pSrc = (T *)pColumn;
        U * pDest = (U *)pAccumBin;
        // only allow int32_t since comes from group and not ikey
        int32_t * pGroup = (int32_t *)pGroupT;

        // printf("Median string %llu\n", totalInputRows);
        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            for (int j = 0; j < itemSize; j++)
            {
                pDest[i * itemSize + j] = 0;
            }
        }
    }

    //-------------------------------------------------------------------------------
    static GROUPBY_X_FUNC32 GetXFunc2(GB_FUNCTIONS func)
    {
        switch (func)
        {
        case GB_ROLLING_SUM:
            return AccumRollingSum;
        case GB_ROLLING_NANSUM:
            return AccumRollingNanSum;
        case GB_ROLLING_DIFF:
            return AccumRollingDiff;
        case GB_ROLLING_SHIFT:
            return AccumRollingShift;
        case GB_ROLLING_COUNT:
            return AccumRollingCount;
        case GB_ROLLING_MEAN:
            return AccumRollingMean;
        case GB_ROLLING_NANMEAN:
            return AccumRollingNanMean;
        default:
            break;
        }
        return NULL;
    }

    //-------------------------------------------------------------------------------
    static GROUPBY_X_FUNC32 GetXFunc(GB_FUNCTIONS func)
    {
        switch (func)
        {
        case GB_FIRST:
            return AccumFirst;
        case GB_NTH:
            return AccumNth;
        case GB_LAST:
            return AccumLast;
        case GB_MEDIAN:
            return AccumMedian;
        case GB_MODE:
            return AccumMode;
        default:
            break;
        }
        return NULL;
    }

    static GROUPBY_X_FUNC32 GetXFuncString(GB_FUNCTIONS func)
    {
        // void AccumBinFirst(int32_t* pGroup, int32_t* pFirst, int32_t* pCount,
        // char* pAccumBin, char* pColumn, int64_t numUnique, int64_t itemSize,
        // int64_t funcParam) {

        // Disable all of this for now...
        switch (func)
        {
        // case GB_MIN:
        //   return AccumMinString;
        // case GB_MAX:
        //   return AccumMaxString;
        case GB_FIRST:
            return AccumFirstString;
        case GB_LAST:
            return AccumLastString;
        case GB_NTH:
            return AccumNthString;
        case GB_MEDIAN:
            return AccumMedianString;
        default:
            break;
        }
        return NULL;
    }
};

//-------------------------------------------------------------------
typedef void (*GROUPBY_GATHER_FUNC)(stGroupBy32 * pstGroupBy32, void * pDataIn, void * pDataOut, int32_t * pCountOut,
                                    int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh);

template <typename U>
static void GatherSum(stGroupBy32 * pstGroupBy32, void * pDataInT, void * pDataOutT, int32_t * pCountOut, int64_t numUnique,
                      int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U * pDataInBase = (U *)pDataInT;
    U * pDataOut = (U *)pDataOutT;

    memset(pDataOut, 0, sizeof(U) * numUnique);

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U * pDataIn = &pDataInBase[j * numUnique];

            for (int64_t i = binLow; i < binHigh; i++)
            {
                pDataOut[i] += pDataIn[i];
            }
        }
    }
}

template <typename U>
static void GatherMean(stGroupBy32 * pstGroupBy32, void * pDataInT, void * pDataOutT, int32_t * pCountOutBase, int64_t numUnique,
                       int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U * pDataInBase = (U *)pDataInT;
    U * pDataOut = (U *)pDataOutT;

    int64_t allocSize = sizeof(int32_t) * numUnique;
    int32_t * pCountOut = (int32_t *)WORKSPACE_ALLOC(allocSize);
    memset(pCountOut, 0, allocSize);

    memset(pDataOut, 0, sizeof(U) * numUnique);

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U * pDataIn = &pDataInBase[j * numUnique];
            int32_t * pCountOutCore = &pCountOutBase[j * numUnique];

            for (int64_t i = binLow; i < binHigh; i++)
            {
                pDataOut[i] += pDataIn[i];
                pCountOut[i] += pCountOutCore[i];
            }
        }
    }

    // calculate the mean
    for (int64_t i = binLow; i < binHigh; i++)
    {
        pDataOut[i] = pDataOut[i] / pCountOut[i];
    }

    WORKSPACE_FREE(pCountOut);
}

template <typename U>
static void GatherMinFloat(stGroupBy32 * pstGroupBy32, void * pDataInT, void * pDataOutT, int32_t * pCountOutBase,
                           int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U * pDataInBase = (U *)pDataInT;
    U * pDataOut = (U *)pDataOutT;

    // Fill with invalid
    U invalid = GET_INVALID(pDataOut[0]);
    for (int64_t i = binLow; i < binHigh; i++)
    {
        pDataOut[i] = invalid;
    }

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U * pDataIn = &pDataInBase[j * numUnique];

            for (int64_t i = binLow; i < binHigh; i++)
            {
                U curValue = pDataOut[i];
                U compareValue = pDataIn[i];

                // nan != nan --> true
                // nan == nan --> false
                // == invalid
                if (compareValue == compareValue)
                {
                    if (! (curValue <= compareValue))
                    {
                        pDataOut[i] = compareValue;
                    }
                }
            }
        }
    }
}

template <typename U>
static void GatherMin(stGroupBy32 * pstGroupBy32, void * pDataInT, void * pDataOutT, int32_t * pCountOutBase, int64_t numUnique,
                      int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U * pDataInBase = (U *)pDataInT;
    U * pDataOut = (U *)pDataOutT;

    // Fill with invalid
    U invalid = GET_INVALID(pDataOut[0]);

    for (int64_t i = binLow; i < binHigh; i++)
    {
        pDataOut[i] = invalid;
    }

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U * pDataIn = &pDataInBase[j * numUnique];

            for (int64_t i = binLow; i < binHigh; i++)
            {
                U curValue = pDataOut[i];
                U compareValue = pDataIn[i];

                if (compareValue != invalid)
                {
                    if (compareValue < curValue || curValue == invalid)
                    {
                        pDataOut[i] = compareValue;
                    }
                }
            }
        }
    }
}

template <typename U>
static void GatherMaxFloat(stGroupBy32 * pstGroupBy32, void * pDataInT, void * pDataOutT, int32_t * pCountOutBase,
                           int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U * pDataInBase = (U *)pDataInT;
    U * pDataOut = (U *)pDataOutT;

    // Fill with invalid
    U invalid = GET_INVALID(pDataOut[0]);

    for (int64_t i = binLow; i < binHigh; i++)
    {
        pDataOut[i] = invalid;
    }

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U * pDataIn = &pDataInBase[j * numUnique];

            for (int64_t i = binLow; i < binHigh; i++)
            {
                U curValue = pDataOut[i];
                U compareValue = pDataIn[i];

                // nan != nan --> true
                // nan == nan --> false
                // == invalid
                if (compareValue == compareValue)
                {
                    if (! (curValue >= compareValue))
                    {
                        pDataOut[i] = compareValue;
                    }
                }
            }
        }
    }
}

template <typename U>
static void GatherMax(stGroupBy32 * pstGroupBy32, void * pDataInT, void * pDataOutT, int32_t * pCountOutBase, int64_t numUnique,
                      int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U * pDataInBase = (U *)pDataInT;
    U * pDataOut = (U *)pDataOutT;

    // Fill with invalid?
    U invalid = GET_INVALID(pDataOut[0]);

    for (int64_t i = binLow; i < binHigh; i++)
    {
        pDataOut[i] = invalid;
    }

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U * pDataIn = &pDataInBase[j * numUnique];

            for (int64_t i = binLow; i < binHigh; i++)
            {
                U curValue = pDataOut[i];
                U compareValue = pDataIn[i];

                if (compareValue != invalid)
                {
                    if (compareValue > curValue || curValue == invalid)
                    {
                        pDataOut[i] = compareValue;
                    }
                }
            }
        }
    }
}

static GROUPBY_GATHER_FUNC GetGroupByGatherFunction(int outputType, GB_FUNCTIONS func)
{
    switch (func)
    {
    case GB_SUM:
    case GB_NANSUM:
        switch (outputType)
        {
        case NPY_BOOL:
            return GatherSum<int64_t>;
        case NPY_FLOAT:
            return GatherSum<float>;
        case NPY_DOUBLE:
            return GatherSum<double>;
        case NPY_LONGDOUBLE:
            return GatherSum<long double>;
        case NPY_INT8:
            return GatherSum<int64_t>;
        case NPY_INT16:
            return GatherSum<int64_t>;
        CASE_NPY_INT32:
            return GatherSum<int64_t>;
        CASE_NPY_INT64:
            return GatherSum<int64_t>;
        case NPY_UINT8:
            return GatherSum<uint64_t>;
        case NPY_UINT16:
            return GatherSum<uint64_t>;
        CASE_NPY_UINT32:
            return GatherSum<uint64_t>;
        CASE_NPY_UINT64:

            return GatherSum<uint64_t>;
        }
        break;

    case GB_MEAN:
    case GB_NANMEAN:
        switch (outputType)
        {
        case NPY_FLOAT:
            return GatherMean<float>;
        case NPY_BOOL:
        case NPY_DOUBLE:
        case NPY_LONGDOUBLE:
        case NPY_INT8:
        case NPY_INT16:
        CASE_NPY_INT32:
        CASE_NPY_INT64:

        case NPY_UINT8:
        case NPY_UINT16:
        CASE_NPY_UINT32:
        CASE_NPY_UINT64:

            return GatherMean<double>;
        }
        break;

    case GB_MAX:
    case GB_NANMAX:
        switch (outputType)
        {
        case NPY_BOOL:
            return GatherMax<int8_t>;
        case NPY_FLOAT:
            return GatherMaxFloat<float>;
        case NPY_DOUBLE:
            return GatherMaxFloat<double>;
        case NPY_LONGDOUBLE:
            return GatherMaxFloat<long double>;
        case NPY_INT8:
            return GatherMax<int8_t>;
        case NPY_INT16:
            return GatherMax<int16_t>;
        CASE_NPY_INT32:
            return GatherMax<int32_t>;
        CASE_NPY_INT64:

            return GatherMax<int64_t>;
        case NPY_UINT8:
            return GatherMax<uint8_t>;
        case NPY_UINT16:
            return GatherMax<uint16_t>;
        CASE_NPY_UINT32:
            return GatherMax<uint32_t>;
        CASE_NPY_UINT64:

            return GatherMax<uint64_t>;
        }
        break;

    case GB_MIN:
    case GB_NANMIN:
        switch (outputType)
        {
        case NPY_BOOL:
            return GatherMin<int8_t>;
        case NPY_FLOAT:
            return GatherMinFloat<float>;
        case NPY_DOUBLE:
            return GatherMinFloat<double>;
        case NPY_LONGDOUBLE:
            return GatherMinFloat<long double>;
        case NPY_INT8:
            return GatherMin<int8_t>;
        case NPY_INT16:
            return GatherMin<int16_t>;
        CASE_NPY_INT32:
            return GatherMin<int32_t>;
        CASE_NPY_INT64:

            return GatherMin<int64_t>;
        case NPY_UINT8:
            return GatherMin<uint8_t>;
        case NPY_UINT16:
            return GatherMin<uint16_t>;
        CASE_NPY_UINT32:
            return GatherMin<uint32_t>;
        CASE_NPY_UINT64:

            return GatherMin<uint64_t>;
        }
        break;
    default:
        break;
    }
    return NULL;
}

//-------------------------------------------------------------------
// T is the input type
// U is the output type
// V is the index type, like int32_t or int8_t
template <typename V>
static GROUPBY_TWO_FUNC GetGroupByFunction(bool * hasCounts, int32_t * wantedOutputType, int32_t * wantedTempType, int inputType, GB_FUNCTIONS func)
{
    *hasCounts = false;
    *wantedTempType = -1;
    switch (func)
    {
    case GB_SUM:
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int8_t, int64_t, V>::AccumSum;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            *wantedTempType = NPY_DOUBLE;
            return GroupByBase<float, float, V>::AccumSumFloat;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumSum;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V>::AccumSum;
        case NPY_INT8:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int8_t, int64_t, V>::AccumSum;
        case NPY_INT16:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int16_t, int64_t, V>::AccumSum;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int32_t, int64_t, V>::AccumSum;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V>::AccumSum;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint8_t, uint64_t, V>::AccumSum;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint16_t, uint64_t, V>::AccumSum;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint32_t, uint64_t, V>::AccumSum;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V>::AccumSum;
        default:
            break;
        }

    case GB_NANSUM:
        switch (inputType)
        {
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            *wantedTempType = NPY_DOUBLE;
            return GroupByBase<float, float, V>::AccumNanSumFloat;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumNanSum;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V>::AccumNanSum;
        // bool has no invalid
        case NPY_BOOL:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int8_t, int64_t, V>::AccumSum;
        case NPY_INT8:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int8_t, int64_t, V>::AccumNanSum;
        case NPY_INT16:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int16_t, int64_t, V>::AccumNanSum;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int32_t, int64_t, V>::AccumNanSum;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V>::AccumNanSum;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint8_t, uint64_t, V>::AccumNanSum;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint16_t, uint64_t, V>::AccumNanSum;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint32_t, uint64_t, V>::AccumNanSum;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V>::AccumNanSum;
        default:
            break;
        }

    case GB_MIN:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_BOOL;
            return GroupByBase<int8_t, int8_t, V>::AccumMin;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumMin;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumMin;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V>::AccumMin;
        case NPY_INT8:
            *wantedOutputType = NPY_INT8;
            return GroupByBase<int8_t, int8_t, V>::AccumMin;
        case NPY_INT16:
            *wantedOutputType = NPY_INT16;
            return GroupByBase<int16_t, int16_t, V>::AccumMin;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT32;
            return GroupByBase<int32_t, int32_t, V>::AccumMin;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V>::AccumMin;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT8;
            return GroupByBase<uint8_t, uint8_t, V>::AccumMin;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT16;
            return GroupByBase<uint16_t, uint16_t, V>::AccumMin;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT32;
            return GroupByBase<uint32_t, uint32_t, V>::AccumMin;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V>::AccumMin;
        default:
            break;
        }

    case GB_NANMIN:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_FLOAT:
            *hasCounts = false;
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumNanMin;
        case NPY_DOUBLE:
            *hasCounts = false;
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumNanMin;
        case NPY_LONGDOUBLE:
            *hasCounts = false;
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V>::AccumNanMin;
        case NPY_BOOL:
            *wantedOutputType = NPY_BOOL;
            return GroupByBase<int8_t, int8_t, V>::AccumMin;
        case NPY_INT8:
            *wantedOutputType = NPY_INT8;
            return GroupByBase<int8_t, int8_t, V>::AccumNanMin;
        case NPY_INT16:
            *wantedOutputType = NPY_INT16;
            return GroupByBase<int16_t, int16_t, V>::AccumNanMin;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT32;
            return GroupByBase<int32_t, int32_t, V>::AccumNanMin;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V>::AccumNanMin;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT8;
            return GroupByBase<uint8_t, uint8_t, V>::AccumNanMin;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT16;
            return GroupByBase<uint16_t, uint16_t, V>::AccumNanMin;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT32;
            return GroupByBase<uint32_t, uint32_t, V>::AccumNanMin;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V>::AccumNanMin;
        default:
            break;
        }

    case GB_MAX:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_BOOL;
            return GroupByBase<int8_t, int8_t, V>::AccumMax;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumMax;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumMax;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V>::AccumMax;
        case NPY_INT8:
            *wantedOutputType = NPY_INT8;
            return GroupByBase<int8_t, int8_t, V>::AccumMax;
        case NPY_INT16:
            *wantedOutputType = NPY_INT16;
            return GroupByBase<int16_t, int16_t, V>::AccumMax;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT32;
            return GroupByBase<int32_t, int32_t, V>::AccumMax;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V>::AccumMax;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT8;
            return GroupByBase<uint8_t, uint8_t, V>::AccumMax;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT16;
            return GroupByBase<uint16_t, uint16_t, V>::AccumMax;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT32;
            return GroupByBase<uint32_t, uint32_t, V>::AccumMax;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V>::AccumMax;
        default:
            break;
        }

    case GB_NANMAX:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumNanMax;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumNanMax;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V>::AccumNanMax;
        case NPY_BOOL:
            *wantedOutputType = NPY_BOOL;
            return GroupByBase<int8_t, int8_t, V>::AccumMax;
        case NPY_INT8:
            *wantedOutputType = NPY_INT8;
            return GroupByBase<int8_t, int8_t, V>::AccumNanMax;
        case NPY_INT16:
            *wantedOutputType = NPY_INT16;
            return GroupByBase<int16_t, int16_t, V>::AccumNanMax;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT32;
            return GroupByBase<int32_t, int32_t, V>::AccumNanMax;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V>::AccumNanMax;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT8;
            return GroupByBase<uint8_t, uint8_t, V>::AccumNanMax;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT16;
            return GroupByBase<uint16_t, uint16_t, V>::AccumNanMax;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT32;
            return GroupByBase<uint32_t, uint32_t, V>::AccumNanMax;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V>::AccumNanMax;
        default:
            break;
        }

    case GB_MEAN:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumMean;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumMeanFloat;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumMean;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V>::AccumMean;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumMean;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V>::AccumMean;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V>::AccumMean;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V>::AccumMean;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V>::AccumMean;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V>::AccumMean;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V>::AccumMean;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V>::AccumMean;
        default:
            break;
        }

    case GB_NANMEAN:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumNanMean;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumNanMeanFloat;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumNanMean;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V>::AccumNanMean;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumNanMean;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V>::AccumNanMean;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V>::AccumNanMean;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V>::AccumNanMean;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V>::AccumNanMean;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V>::AccumNanMean;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V>::AccumNanMean;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V>::AccumNanMean;
        default:
            break;
        }

    case GB_VAR:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumVar;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumVar;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumVar;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V>::AccumVar;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumVar;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V>::AccumVar;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V>::AccumVar;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V>::AccumVar;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V>::AccumVar;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V>::AccumVar;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V>::AccumVar;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V>::AccumVar;
        default:
            break;
        }

    case GB_NANVAR:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumNanVar;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumNanVar;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumNanVar;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V>::AccumNanVar;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumNanVar;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V>::AccumNanVar;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V>::AccumNanVar;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V>::AccumNanVar;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V>::AccumNanVar;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V>::AccumNanVar;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V>::AccumNanVar;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V>::AccumNanVar;
        default:
            break;
        }

    case GB_STD:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumStd;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumStd;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumStd;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V>::AccumStd;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumStd;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V>::AccumStd;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V>::AccumStd;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V>::AccumStd;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V>::AccumStd;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V>::AccumStd;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V>::AccumStd;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V>::AccumStd;
        default:
            break;
        }

    case GB_NANSTD:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumNanStd;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V>::AccumNanStd;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V>::AccumNanStd;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V>::AccumNanStd;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V>::AccumNanStd;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V>::AccumNanStd;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V>::AccumNanStd;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V>::AccumNanStd;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V>::AccumNanStd;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V>::AccumNanStd;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V>::AccumNanStd;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V>::AccumNanStd;
        default:
            break;
        }

    default:
        break;
    }
    return NULL;
}

// template<typename T>
// static GROUPBY_X_FUNC GetGroupByXStep2(int outputType, GB_FUNCTIONS func) {
//   switch (outputType) {
//      //   case NPY_BOOL:   return GroupByBase<T, bool>::GetFunc(func);
//   case NPY_FLOAT:  return GroupByBase<T, float>::GetXFunc(func);
//   case NPY_DOUBLE: return GroupByBase<T, double>::GetXFunc(func);
//      //   case NPY_BYTE:   return GroupByBase<T, int8_t>::GetFunc(func);
//      //   case NPY_INT16:  return GroupByBase<T, int16_t>::GetFunc(func);
//   case NPY_INT:    return GroupByBase<T, int32_t>::GetXFunc(func);
//   CASE_NPY_INT32:  return GroupByBase<T, int32_t>::GetXFunc(func);
//   CASE_NPY_INT64:  return GroupByBase<T, int64_t>::GetXFunc(func);
//      //   case NPY_UBYTE:  return GroupByBase<T, uint8_t>::GetFunc(func);
//      //   case NPY_UINT16: return GroupByBase<T, uint16_t>::GetFunc(func);
//   case NPY_UINT:   return GroupByBase<T, uint32_t>::GetXFunc(func);
//   CASE_NPY_UINT32: return GroupByBase<T, uint32_t>::GetXFunc(func);
//   CASE_NPY_UINT64: return GroupByBase<T, uint64_t>::GetXFunc(func);
//   }
//   return NULL;
//
//}

template <typename V>
static GROUPBY_X_FUNC32 GetGroupByXFunction32(int inputType, int outputType, GB_FUNCTIONS func)
{
    LOGGING("GBX32 Func is %d  inputtype: %d  outputtype: %d\n", func, inputType, outputType);

    if (func == GB_TRIMBR)
    {
        switch (inputType)
        {
        case NPY_FLOAT:
            return GroupByBase<float, float, V>::AccumTrimMeanBR;
        case NPY_DOUBLE:
            return GroupByBase<double, double, V>::AccumTrimMeanBR;
        case NPY_LONGDOUBLE:
            return GroupByBase<long double, double, V>::AccumTrimMeanBR;
        case NPY_INT8:
            return GroupByBase<int8_t, double, V>::AccumTrimMeanBR;
        case NPY_INT16:
            return GroupByBase<int16_t, double, V>::AccumTrimMeanBR;
        CASE_NPY_INT32:
            return GroupByBase<int32_t, double, V>::AccumTrimMeanBR;
        CASE_NPY_INT64:
            return GroupByBase<int64_t, double, V>::AccumTrimMeanBR;
        case NPY_UINT8:
            return GroupByBase<uint8_t, double, V>::AccumTrimMeanBR;
        case NPY_UINT16:
            return GroupByBase<uint16_t, double, V>::AccumTrimMeanBR;
        CASE_NPY_UINT32:
            return GroupByBase<uint32_t, double, V>::AccumTrimMeanBR;
        CASE_NPY_UINT64:
            return GroupByBase<uint64_t, double, V>::AccumTrimMeanBR;
        }
        return NULL;
    }
    else if (func == GB_ROLLING_COUNT)
    {
        switch (inputType)
        {
        case NPY_INT8:
            return GroupByBase<int8_t, int32_t, V>::GetXFunc2(func);
        case NPY_INT16:
            return GroupByBase<int16_t, int32_t, V>::GetXFunc2(func);
        CASE_NPY_INT32:
            return GroupByBase<int32_t, int32_t, V>::GetXFunc2(func);
        CASE_NPY_INT64:
            return GroupByBase<int64_t, int32_t, V>::GetXFunc2(func);
        }
        return NULL;
    }
    else if (func >= GB_ROLLING_DIFF && func < GB_ROLLING_MEAN)
    {
        LOGGING("Rolling+diff called with type %d\n", inputType);
        switch (inputType)
        {
            // really need to change output type for accumsum/rolling
            // case NPY_BOOL:   return GroupByBase<bool, int64_t, V>::GetXFunc2(func);
        case NPY_FLOAT:
            return GroupByBase<float, float, V>::GetXFunc2(func);
        case NPY_DOUBLE:
            return GroupByBase<double, double, V>::GetXFunc2(func);
        case NPY_LONGDOUBLE:
            return GroupByBase<long double, long double, V>::GetXFunc2(func);
        case NPY_INT8:
            return GroupByBase<int8_t, int8_t, V>::GetXFunc2(func);
        case NPY_INT16:
            return GroupByBase<int16_t, int16_t, V>::GetXFunc2(func);
        CASE_NPY_INT32:
            return GroupByBase<int32_t, int32_t, V>::GetXFunc2(func);
        CASE_NPY_INT64:
            return GroupByBase<int64_t, int64_t, V>::GetXFunc2(func);
        case NPY_UINT8:
            return GroupByBase<uint8_t, uint8_t, V>::GetXFunc2(func);
        case NPY_UINT16:
            return GroupByBase<uint16_t, uint16_t, V>::GetXFunc2(func);
        CASE_NPY_UINT32:
            return GroupByBase<uint32_t, uint32_t, V>::GetXFunc2(func);
        CASE_NPY_UINT64:
            return GroupByBase<uint64_t, uint64_t, V>::GetXFunc2(func);
        }
        return NULL;
    }
    else if (func >= GB_ROLLING_SUM)
    {
        if (func == GB_ROLLING_MEAN || func == GB_ROLLING_NANMEAN)
        {
            LOGGING("Rolling+mean called with type %d\n", inputType);
            // default to a double for output
            switch (inputType)
            {
            case NPY_FLOAT:
                return GroupByBase<float, double, V>::GetXFunc2(func);
            case NPY_DOUBLE:
                return GroupByBase<double, double, V>::GetXFunc2(func);
            case NPY_LONGDOUBLE:
                return GroupByBase<long double, double, V>::GetXFunc2(func);
            case NPY_INT8:
                return GroupByBase<int8_t, double, V>::GetXFunc2(func);
            case NPY_INT16:
                return GroupByBase<int16_t, double, V>::GetXFunc2(func);
            CASE_NPY_INT32:
                return GroupByBase<int32_t, double, V>::GetXFunc2(func);
            CASE_NPY_INT64:
                return GroupByBase<int64_t, double, V>::GetXFunc2(func);
            case NPY_UINT8:
                return GroupByBase<uint8_t, double, V>::GetXFunc2(func);
            case NPY_UINT16:
                return GroupByBase<uint16_t, double, V>::GetXFunc2(func);
            CASE_NPY_UINT32:
                return GroupByBase<uint32_t, double, V>::GetXFunc2(func);
            CASE_NPY_UINT64:
                return GroupByBase<uint64_t, double, V>::GetXFunc2(func);
            }
            return NULL;
        }
        else
        {
            // due to overflow, all ints become int64_t
            LOGGING("Rolling+sum called with type %d\n", inputType);
            switch (inputType)
            {
                // really need to change output type for accumsum/rolling
            case NPY_BOOL:
                return GroupByBase<int8_t, int64_t, V>::GetXFunc2(func);
            case NPY_FLOAT:
                return GroupByBase<float, float, V>::GetXFunc2(func);
            case NPY_DOUBLE:
                return GroupByBase<double, double, V>::GetXFunc2(func);
            case NPY_LONGDOUBLE:
                return GroupByBase<long double, long double, V>::GetXFunc2(func);
            case NPY_INT8:
                return GroupByBase<int8_t, int64_t, V>::GetXFunc2(func);
            case NPY_INT16:
                return GroupByBase<int16_t, int64_t, V>::GetXFunc2(func);
            CASE_NPY_INT32:
                return GroupByBase<int32_t, int64_t, V>::GetXFunc2(func);
            CASE_NPY_INT64:
                return GroupByBase<int64_t, int64_t, V>::GetXFunc2(func);
            case NPY_UINT8:
                return GroupByBase<uint8_t, int64_t, V>::GetXFunc2(func);
            case NPY_UINT16:
                return GroupByBase<uint16_t, int64_t, V>::GetXFunc2(func);
            CASE_NPY_UINT32:
                return GroupByBase<uint32_t, int64_t, V>::GetXFunc2(func);
            CASE_NPY_UINT64:
                return GroupByBase<uint64_t, int64_t, V>::GetXFunc2(func);
            }
        }
    }
    else
    {
        switch (inputType)
        {
        // first,last,median,nth
        case NPY_BOOL:
            return GroupByBase<bool, bool, V>::GetXFunc(func);
        case NPY_FLOAT:
            return GroupByBase<float, float, V>::GetXFunc(func);
        case NPY_DOUBLE:
            return GroupByBase<double, double, V>::GetXFunc(func);
        case NPY_LONGDOUBLE:
            return GroupByBase<long double, long double, V>::GetXFunc(func);
        case NPY_INT8:
            return GroupByBase<int8_t, int8_t, V>::GetXFunc(func);
        case NPY_INT16:
            return GroupByBase<int16_t, int16_t, V>::GetXFunc(func);
        CASE_NPY_INT32:
            return GroupByBase<int32_t, int32_t, V>::GetXFunc(func);
        CASE_NPY_INT64:
            return GroupByBase<int64_t, int64_t, V>::GetXFunc(func);
        case NPY_UINT8:
            return GroupByBase<uint8_t, uint8_t, V>::GetXFunc(func);
        case NPY_UINT16:
            return GroupByBase<uint16_t, uint16_t, V>::GetXFunc(func);
        CASE_NPY_UINT32:
            return GroupByBase<uint32_t, uint32_t, V>::GetXFunc(func);
        CASE_NPY_UINT64:
            return GroupByBase<uint64_t, uint64_t, V>::GetXFunc(func);
        case NPY_STRING:
            return GroupByBase<char, char, V>::GetXFuncString(func);
        case NPY_UNICODE:
        case NPY_VOID:
            return GroupByBase<char, char, V>::GetXFuncString(func);
        }
    }
    return NULL;
}

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static bool BandedGroupByCall(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    // -1 is the first core
    core = core + 1;
    bool didSomeWork = false;
    stGroupBy32 * pGroupBy32 = (stGroupBy32 *)pstWorkerItem->WorkCallbackArg;

    int64_t index;
    int64_t workBlock;

    // As long as there is work to do
    while ((index = pstWorkerItem->GetNextWorkIndex(&workBlock)) > 0)
    {
        // aInfo only valid if we are the worker (otherwise this pointer is invalid)
        ArrayInfo * aInfo = pGroupBy32->aInfo;

        // Data in was passed
        char * pDataIn = (char *)(aInfo[0].pData);
        int64_t len = aInfo[0].ArrayLength;
        int64_t totalRows = pGroupBy32->totalInputRows;

        int32_t * pCountOut = pGroupBy32->returnObjects[0].pCountOut;
        GROUPBY_X_FUNC32 pFunctionX = pGroupBy32->returnObjects[0].pFunctionX32;
        void * pDataOut = pGroupBy32->returnObjects[0].pOutArray;

        // First index is 1 so we subtract
        index--;

        LOGGING("|%d %d %lld %p %p %p", core, (int)workBlock, index, pDataIn, pCountOut, pDataOut);
        int64_t binLow = pGroupBy32->returnObjects[index].binLow;
        int64_t binHigh = pGroupBy32->returnObjects[index].binHigh;

        pFunctionX((void *)pDataIn, (void *)pGroupBy32->pGroup, (int32_t *)pGroupBy32->pFirst, (int32_t *)pGroupBy32->pCount,
                   (char *)pDataOut, binLow, binHigh, pGroupBy32->totalInputRows, aInfo[0].ItemSize, pGroupBy32->funcParam);

        pGroupBy32->returnObjects[index].didWork = 1;

        // Indicate we completed a block
        didSomeWork = true;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
    }
    return didSomeWork;
}

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static bool ScatterGroupByCall(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    // -1 is the first core
    core = core + 1;

    bool didSomeWork = false;
    stGroupBy32 * pGroupBy32 = (stGroupBy32 *)pstWorkerItem->WorkCallbackArg;
    ArrayInfo * aInfo = pGroupBy32->aInfo;

    // Data in was passed
    char * pDataIn = (char *)(aInfo[0].pData);
    int64_t len = aInfo[0].ArrayLength;

    // iKey
    char * pDataIn2 = (char *)(pGroupBy32->pDataIn2);

    int64_t binLow = pGroupBy32->returnObjects[core].binLow;
    int64_t binHigh = pGroupBy32->returnObjects[core].binHigh;
    int32_t * pCountOut = pGroupBy32->returnObjects[core].pCountOut;
    GROUPBY_TWO_FUNC pFunction = pGroupBy32->returnObjects[core].pFunction;
    void * pDataOut = pGroupBy32->returnObjects[core].pOutArray;
    void * pDataTmp = pGroupBy32->returnObjects[core].pTmpArray;

    int64_t lenX;
    int64_t workBlock;
    int64_t pass = 0;

    int64_t itemSize1 = aInfo[0].ItemSize;
    int64_t itemSize2 = pGroupBy32->itemSize2;

    // printf("Scatter working core %d  %lld\n", core, len);
    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
    {
        // printf("|%d %d %lld %p %p %p %p", core, (int)workBlock, lenX, pDataIn,
        // pDataIn2, pCountOut, pDataOut);

        int64_t inputAdj1 = pstWorkerItem->BlockSize * workBlock * itemSize1;
        int64_t inputAdj2 = pstWorkerItem->BlockSize * workBlock * itemSize2;

        // shift pDataIn by T
        // shift pDataIn2 by U
        pFunction(pDataIn + inputAdj1, pDataIn2 + inputAdj2, pCountOut, pDataOut, lenX, binLow, binHigh, pass++, pDataTmp);
        pGroupBy32->returnObjects[core].didWork = 1;

        // Indicate we completed a block
        didSomeWork = true;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
    }
    return didSomeWork;
}

//------------------------------------------------------
// Calculate the groupby
// BOTH groupby versions call this routine
// ** THIS ROUTINE IS CALLED FROM MULTIPLE CONCURRENT THREADS!
// i is the column number
void GroupByCall(void * pGroupBy, int64_t i)
{
    stGroupBy32 * pGroupBy32 = (stGroupBy32 *)pGroupBy;
    ArrayInfo * aInfo = pGroupBy32->aInfo;

    // iKey
    void * pDataIn2 = pGroupBy32->pDataIn2;

    int64_t uniqueRows = pGroupBy32->uniqueRows;
    int64_t binLow = pGroupBy32->returnObjects[i].binLow;
    int64_t binHigh = pGroupBy32->returnObjects[i].binHigh;

    // Data in was passed
    void * pDataIn = aInfo[i].pData;
    int64_t len = aInfo[i].ArrayLength;

    PyArrayObject * outArray = pGroupBy32->returnObjects[i].outArray;
    int32_t * pCountOut = pGroupBy32->returnObjects[i].pCountOut;
    GROUPBY_TWO_FUNC pFunction = pGroupBy32->returnObjects[i].pFunction;
    int32_t numpyOutType = pGroupBy32->returnObjects[i].numpyOutType;
    TYPE_OF_FUNCTION_CALL typeCall = pGroupBy32->typeOfFunctionCall;

    if (outArray && pFunction)
    {
        LOGGING(
            "col %llu  ==> outsize %llu   len: %llu   numpy types %d --> %d   "
            "%d %d\n",
            i, uniqueRows, len, aInfo[i].NumpyDType, numpyOutType, gNumpyTypeToSize[aInfo[i].NumpyDType],
            gNumpyTypeToSize[numpyOutType]);

        void * pDataOut = PyArray_BYTES(outArray);
        void * pDataTmp = pGroupBy32->returnObjects[i].pTmpArray;

        LOGGING("%llu  typeCall %d  numpyOutType %d\n", i, (int)typeCall, numpyOutType);

        if (typeCall == ANY_GROUPBY_FUNC)
        {
            // Accum the calculation
            // Sum/NanSum
            // Make the range from 1 to uniqueRows to skip over bin 0
            pFunction(pDataIn, pDataIn2 /* USE IKEY which can be int8/16/32/64*/, pCountOut, pDataOut, len, binLow, binHigh, -1, pDataTmp);
        }
        else

            if (typeCall == ANY_GROUPBY_XFUNC32)
        {
            // Accum the calculation
            GROUPBY_X_FUNC32 pFunctionX = pGroupBy32->returnObjects[i].pFunctionX32;

            int32_t funcNum = pGroupBy32->returnObjects[i].funcNum;

            // static void AccumLast(void* pColumn, void* pGroupT, int32_t* pFirst,
            // int32_t* pCount, void* pAccumBin, int64_t numUnique, int64_t
            // totalInputRows, int64_t itemSize, int64_t funcParam) {
            if (funcNum < GB_FIRST)
                printf("!!! internal bug in GroupByCall -- %d\n", funcNum);

            if (pFunctionX)
            {
                pFunctionX((void *)pDataIn, (int32_t *)pGroupBy32->pGroup /*USE GROUP wihch must be int32*/,
                           (int32_t *)pGroupBy32->pFirst, (int32_t *)pGroupBy32->pCount, (char *)pDataOut, binLow, binHigh,
                           pGroupBy32->totalInputRows, aInfo[i].ItemSize, pGroupBy32->funcParam);
            }
            else
            {
                printf("!!!internal error no pfunctionx\n");
            }
        }

        pGroupBy32->returnObjects[i].returnObject = (PyObject *)outArray;
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
        pGroupBy32->returnObjects[i].returnObject = Py_None;
    }
}

//---------------------------------------------------------------
// Arg1 = LIST of numpy arrays which has the values to accumulate (often all the
// columns in a dataset) Arg2 = numpy array (int32_t) which has the index to the
// unique keys (ikey from MultiKeyGroupBy32) Arg3 = integer unique rows Arg4 =
// integer (function number to execute for sum,mean,min, max) Example:
// GroupByOp2(array, ikey, 3, np.float32) Returns cells
PyObject * GroupByAll64(PyObject * self, PyObject * args)
{
    PyObject * inList1 = NULL;
    PyArrayObject * inArr2 = NULL;

    int64_t unique_rows = 0;
    int64_t funcNum = 0;

    if (! PyArg_ParseTuple(args, "OO!LL", &inList1, &PyArray_Type, &inArr2, &unique_rows, &funcNum))
    {
        return NULL;
    }

    // STUB NOT COMPLETED
    return NULL;
}

//---------------------------------------------------------------
GROUPBY_TWO_FUNC GetGroupByFunctionStep1(int32_t iKeyType, bool * hasCounts, int32_t * numpyOutType, int32_t * numpyTmpType, int32_t numpyInType,
                                         GB_FUNCTIONS funcNum)
{
    GROUPBY_TWO_FUNC pFunction = NULL;

    switch (iKeyType)
    {
    case NPY_INT8:
        pFunction = GetGroupByFunction<int8_t>(hasCounts, numpyOutType, numpyTmpType, numpyInType, funcNum);
        break;
    case NPY_INT16:
        pFunction = GetGroupByFunction<int16_t>(hasCounts, numpyOutType, numpyTmpType, numpyInType, funcNum);
        break;
    CASE_NPY_INT32:
        pFunction = GetGroupByFunction<int32_t>(hasCounts, numpyOutType, numpyTmpType, numpyInType, funcNum);
        break;
    CASE_NPY_INT64:
        pFunction = GetGroupByFunction<int64_t>(hasCounts, numpyOutType, numpyTmpType, numpyInType, funcNum);
        break;
    }

    return pFunction;
}

//---------------------------------------------------------------
// When just a single array needs to be calculated
// For some operations we can multithread by scattering the work
// and then gathering the work from threads.
PyObject * GroupBySingleOpMultiBands(ArrayInfo * aInfo, PyArrayObject * iKey, PyArrayObject * iFirst, PyArrayObject * iGroup,
                                     PyArrayObject * nCount, GB_FUNCTIONS firstFuncNum, int64_t unique_rows, int64_t tupleSize,
                                     int64_t binLow, int64_t binHigh)
{
    PyObject * returnTuple = NULL;
    int32_t iKeyType = PyArray_TYPE(iKey);

    int32_t numpyOutType = aInfo[0].NumpyDType;
    bool hasCounts = false;

    LOGGING("In banded groupby %d\n", (int)firstFuncNum);

    GROUPBY_X_FUNC32 pFunction = NULL;

    switch (iKeyType)
    {
    case NPY_INT8:
        pFunction = GetGroupByXFunction32<int8_t>(numpyOutType, numpyOutType, (GB_FUNCTIONS)firstFuncNum);
        break;
    case NPY_INT16:
        pFunction = GetGroupByXFunction32<int16_t>(numpyOutType, numpyOutType, (GB_FUNCTIONS)firstFuncNum);
        break;
    CASE_NPY_INT32:
        pFunction = GetGroupByXFunction32<int32_t>(numpyOutType, numpyOutType, (GB_FUNCTIONS)firstFuncNum);
        break;
    CASE_NPY_INT64:
        pFunction = GetGroupByXFunction32<int64_t>(numpyOutType, numpyOutType, (GB_FUNCTIONS)firstFuncNum);
        break;
    }

    if (pFunction)
    {
        void * pDataIn2 = PyArray_BYTES(iKey);
        int64_t arraySizeKey = ArrayLength(iKey);

        int32_t numCores = g_cMathWorker->WorkerThreadCount + 1;
        int64_t bins = binHigh - binLow;
        int64_t cores = numCores;
        if (bins < cores)
            cores = bins;

        LOGGING("Banded cores %lld\n", cores);

        // See if we get a work item (threading might be off)
        stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItemCount(cores);

        // cores will be zero when there are no bins, all filtered out
        if (pWorkItem && cores > 0)
        {
            PyArrayObject * outArray = NULL;
            outArray = AllocateNumpyArray(1, (npy_intp *)&unique_rows, numpyOutType);
            CHECK_MEMORY_ERROR(outArray);

            if (outArray == NULL)
            {
                return NULL;
            }

            void * pOutArray = PyArray_BYTES(outArray);

            // Allocate the struct + ROOM at the end of struct for all the tuple
            // objects being produced
            stGroupBy32 * pstGroupBy32 = (stGroupBy32 *)WORKSPACE_ALLOC(sizeof(stGroupBy32) + (cores * sizeof(stGroupByReturn)));

            if (pstGroupBy32 == NULL)
            {
                // out of memory
                return NULL;
            }

            int64_t itemSize = PyArray_ITEMSIZE(outArray);
            int32_t * pCountOut = NULL;

            //// Allocate room for all the threads to participate, this will be
            /// gathered later
            // char* pWorkspace = (char*)WORKSPACE_ALLOC(unique_rows * itemSize *
            // numCores); LOGGING("***workspace %p   unique:%lld   itemsize:%lld
            // cores:%d\n", pWorkspace, unique_rows, itemSize, cores);

            // if (pWorkspace == NULL) {
            //   return NULL;
            //}

            if (hasCounts)
            {
                // Zero out for them
                int64_t allocSize = sizeof(int32_t) * unique_rows;
                pCountOut = (int32_t *)WORKSPACE_ALLOC(allocSize);
                if (pCountOut == NULL)
                {
                    return NULL;
                }
                memset(pCountOut, 0, allocSize);
                LOGGING("***pCountOut %p   unique:%lld  allocsize:%lld   cores:%lld\n", pCountOut, unique_rows, allocSize, cores);
            }

            // build in data
            pstGroupBy32->aInfo = aInfo;
            pstGroupBy32->pDataIn2 = pDataIn2;
            pstGroupBy32->itemSize2 = PyArray_ITEMSIZE(iKey);
            pstGroupBy32->tupleSize = tupleSize;
            pstGroupBy32->uniqueRows = unique_rows;
            pstGroupBy32->pKey = PyArray_BYTES(iKey);
            pstGroupBy32->pFirst = PyArray_BYTES(iFirst);
            pstGroupBy32->pGroup = PyArray_BYTES(iGroup);
            pstGroupBy32->pCount = PyArray_BYTES(nCount);
            pstGroupBy32->typeOfFunctionCall = TYPE_OF_FUNCTION_CALL::ANY_GROUPBY_XFUNC32;

            pstGroupBy32->totalInputRows = arraySizeKey;

            LOGGING("groupby dtypes:  key:%d  ifirst:%d  igroup:%d  count:%d\n", PyArray_TYPE(iKey), PyArray_TYPE(iFirst),
                    PyArray_TYPE(iGroup), PyArray_TYPE(nCount));

            int64_t dividend = unique_rows / cores;
            int64_t remainder = unique_rows % cores;

            int64_t low = 0;
            int64_t high = 0;

            for (int64_t i = 0; i < cores; i++)
            {
                // Calculate band range
                high = low + dividend;

                // add in any remainder until nothing left
                if (remainder > 0)
                {
                    high++;
                    remainder--;
                }

                pstGroupBy32->returnObjects[i].binLow = low;
                pstGroupBy32->returnObjects[i].binHigh = high;

                // next low bin is the previous high bin
                low = high;

                pstGroupBy32->returnObjects[i].funcNum = (int32_t)firstFuncNum;
                pstGroupBy32->returnObjects[i].didWork = 0;

                // Assign working memory per call
                pstGroupBy32->returnObjects[i].pOutArray = pOutArray;
                pstGroupBy32->returnObjects[i].pCountOut = pCountOut;
                pstGroupBy32->returnObjects[i].pFunctionX32 = pFunction;
                pstGroupBy32->returnObjects[i].returnObject = Py_None;
                pstGroupBy32->returnObjects[i].numpyOutType = numpyOutType;
            }

            // Each thread will call this routine with the callbackArg
            pWorkItem->DoWorkCallback = BandedGroupByCall;
            pWorkItem->WorkCallbackArg = pstGroupBy32;

            LOGGING("before threaded\n");

            // This will notify the worker threads of a new work item
            g_cMathWorker->WorkMain(pWorkItem, cores, 0, 1, false);
            LOGGING("after threaded\n");

            WORKSPACE_FREE(pstGroupBy32);

            if (hasCounts)
            {
                WORKSPACE_FREE(pCountOut);
            }

            // New reference
            returnTuple = PyTuple_New(tupleSize);
            PyTuple_SET_ITEM(returnTuple, 0, (PyObject *)outArray);
        }
    }

    return returnTuple;
}

//---------------------------------------------------------------
// When just a single array needs to be calculated
// For some operations we can multithread by scattering the work
// and then gathering the work from threads.
PyObject * GroupBySingleOpMultithreaded(ArrayInfo * aInfo, PyArrayObject * iKey, GB_FUNCTIONS firstFuncNum, int64_t unique_rows,
                                        int64_t tupleSize, int64_t binLow, int64_t binHigh)
{
    // Parallel one way
    // Divide up by memory
    PyObject * returnTuple = NULL;
    int32_t iKeyType = PyArray_TYPE(iKey);

    int32_t numpyOutType = -1;
    int32_t numpyTmpType = -1;
    bool hasCounts = false;

    GROUPBY_TWO_FUNC pFunction =
        GetGroupByFunctionStep1(iKeyType, &hasCounts, &numpyOutType, &numpyTmpType, aInfo[0].NumpyDType, (GB_FUNCTIONS)firstFuncNum);

    // printf("Taking the divide path  %lld \n", unique_rows);

    if (pFunction && numpyOutType != -1)
    {
        void * pDataIn2 = PyArray_BYTES(iKey);
        int64_t arraySizeKey = ArrayLength(iKey);

        stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(arraySizeKey);

        if (pWorkItem != NULL)
        {
            std::vector<workspace_mem_ptr> workspaceMemList;

            int32_t numCores = g_cMathWorker->WorkerThreadCount + 1;

            PyArrayObject * outArray = NULL;

            // Dont bother allocating if we cannot call the function
            outArray = AllocateNumpyArray(1, (npy_intp *)&unique_rows, numpyOutType);
            CHECK_MEMORY_ERROR(outArray);

            if (outArray == NULL)
            {
                return NULL;
            }

            int64_t itemSize = PyArray_ITEMSIZE(outArray);
            int32_t * pCountOut = NULL;

            // Allocate room for all the threads to participate, this will be gathered
            // later
            workspaceMemList.emplace_back(WORKSPACE_ALLOC(unique_rows * itemSize * numCores));
            char * pWorkspace = (char *)workspaceMemList.back().get();

            LOGGING("***workspace %p   unique:%lld   itemsize:%lld   cores:%d\n", pWorkspace, unique_rows, itemSize, numCores);

            if (pWorkspace == NULL)
            {
                return NULL;
            }

            if (hasCounts)
            {
                // Zero out for them
                int64_t allocSize = sizeof(int32_t) * unique_rows * numCores;
                workspaceMemList.emplace_back(WORKSPACE_ALLOC(allocSize));
                pCountOut = (int32_t *)workspaceMemList.back().get();
                if (pCountOut == NULL)
                {
                    return NULL;
                }
                memset(pCountOut, 0, allocSize);
                LOGGING("***pCountOut %p   unique:%lld  allocsize:%lld   cores:%d\n", pCountOut, unique_rows, allocSize, numCores);
            }

            void * pTmpWorkspace{nullptr};
            int32_t tempItemSize{0};

            if (numpyTmpType >= 0)
            {
                tempItemSize = NpyToSize(numpyTmpType);
                int64_t tempSize = tempItemSize * (binHigh - binLow) * numCores;
                workspaceMemList.emplace_back(WORKSPACE_ALLOC(tempSize));
                pTmpWorkspace = workspaceMemList.back().get();
                if (!pTmpWorkspace)
                {
                    return NULL;
                }
            }

            // Allocate the struct + ROOM at the end of struct for all the tuple
            // objects being produced
            workspaceMemList.emplace_back(WORKSPACE_ALLOC(sizeof(stGroupBy32) + (numCores * sizeof(stGroupByReturn))));
            stGroupBy32 * pstGroupBy32 = (stGroupBy32 *)workspaceMemList.back().get();

            if (pstGroupBy32 == NULL)
            {
                // out of memory
                return NULL;
            }

            // build in data
            pstGroupBy32->aInfo = aInfo;
            pstGroupBy32->pDataIn2 = pDataIn2;
            pstGroupBy32->itemSize2 = PyArray_ITEMSIZE(iKey);
            pstGroupBy32->tupleSize = tupleSize;
            pstGroupBy32->uniqueRows = unique_rows;

            pstGroupBy32->totalInputRows = arraySizeKey;
            pstGroupBy32->typeOfFunctionCall = TYPE_OF_FUNCTION_CALL::ANY_GROUPBY_FUNC;

            for (int i = 0; i < numCores; i++)
            {
                pstGroupBy32->returnObjects[i].funcNum = (int32_t)firstFuncNum;
                pstGroupBy32->returnObjects[i].binLow = binLow;
                pstGroupBy32->returnObjects[i].binHigh = binHigh;

                pstGroupBy32->returnObjects[i].didWork = 0;

                // Assign working memory per call
                pstGroupBy32->returnObjects[i].pOutArray = &pWorkspace[unique_rows * i * itemSize];
                pstGroupBy32->returnObjects[i].pTmpArray = pTmpWorkspace
                    ? &(static_cast<char *>(pTmpWorkspace)[(binHigh - binLow) * i * tempItemSize])
                    : nullptr;
                pstGroupBy32->returnObjects[i].pCountOut = &pCountOut[unique_rows * i];
                pstGroupBy32->returnObjects[i].pFunction = pFunction;
                pstGroupBy32->returnObjects[i].returnObject = Py_None;
                pstGroupBy32->returnObjects[i].numpyOutType = numpyOutType;
            }

            // Each thread will call this routine with the callbackArg
            pWorkItem->DoWorkCallback = ScatterGroupByCall;
            pWorkItem->WorkCallbackArg = pstGroupBy32;

            LOGGING("before threaded\n");

            // This will notify the worker threads of a new work item
            g_cMathWorker->WorkMain(pWorkItem, arraySizeKey, 0);

            LOGGING("after threaded\n");

            // Gather resullts
            GROUPBY_GATHER_FUNC pGather = GetGroupByGatherFunction(numpyOutType, (GB_FUNCTIONS)firstFuncNum);
            if (pGather)
            {
                void * pDataOut = PyArray_BYTES(outArray);
                pGather(pstGroupBy32, pWorkspace, pDataOut, pCountOut, unique_rows, numCores, binLow, binHigh);
            }
            else
            {
                printf("!!!Internal error in GetGroupByGatherFunction\n");
            }

            // New reference
            returnTuple = PyTuple_New(tupleSize);
            PyTuple_SET_ITEM(returnTuple, 0, (PyObject *)outArray);
        }
    }
    return returnTuple;
}

//---------------------------------------------------------------
// Arg1 = LIST of numpy arrays which has the values to accumulate (often all the
// columns in a dataset) Arg2 = iKey = numpy array (int32_t) which has the index
// to the unique keys (ikey from MultiKeyGroupBy32) Arg3 = integer unique rows
// Arg4 = LIST of integer (function number to execute for sum,mean,min, max)
// Arg5 = LIST of integers (binLow -- invalid bin)
// Arg6 = LIST of integers (binHigh -- invalid bin)
// Arg7 = optional param
// Example: GroupByOp2(array, ikey, 3, np.float32)
// Returns cells
PyObject * GroupByAll32(PyObject * self, PyObject * args)
{
    PyObject * inList1 = NULL;
    PyArrayObject * iKey = NULL;
    PyObject * param = NULL;

    int64_t unique_rows = 0;
    PyListObject * listFuncNum = NULL;
    PyListObject * listBinLow = NULL;
    PyListObject * listBinHigh = NULL;

    if (! PyArg_ParseTuple(args, "OO!LO!O!O!O", &inList1, &PyArray_Type, &iKey, &unique_rows, &PyList_Type, &listFuncNum,
                           &PyList_Type, &listBinLow, &PyList_Type, &listBinHigh, &param))
    {
        return NULL;
    }

    int32_t ndim = PyArray_NDIM(iKey);

    if (ndim != 1)
    {
        PyErr_Format(PyExc_ValueError, "GroupByAll32 ndim must be 1 not %d", ndim);
        return NULL;
    }

    int32_t iKeyType = PyArray_TYPE(iKey);

    // Valid types we can index by
    switch (iKeyType)
    {
    case NPY_INT8:
    case NPY_INT16:
    CASE_NPY_INT32:
    CASE_NPY_INT64:

        break;
    default:
        PyErr_Format(PyExc_ValueError, "GroupByAll32 key param must be int8, int16, int32, int64 not type %d", iKeyType);
        return NULL;
    }

    // Add 1 for zero bin
    unique_rows += GB_BASE_INDEX;

    int32_t numpyInType2 = ObjectToDtype(iKey);

    int64_t totalItemSize = 0;
    int64_t tupleSize = 0;
    ArrayInfo * aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize);

    if (! aInfo)
    {
        return NULL;
    }

    int64_t funcTupleSize = PyList_GET_SIZE(listFuncNum);

    if (tupleSize != funcTupleSize)
    {
        PyErr_Format(PyExc_ValueError, "GroupByAll32 func numbers do not match array columns %lld %lld", tupleSize, funcTupleSize);
        return NULL;
    }

    int64_t binTupleSize = PyList_GET_SIZE(listBinLow);

    if (tupleSize != binTupleSize)
    {
        PyErr_Format(PyExc_ValueError, "GroupByAll32 bin numbers do not match array columns %lld %lld", tupleSize, binTupleSize);
        return NULL;
    }

    // Since this is the 32 bit function, the array indexes are 32 bit
    void * pDataIn2 = PyArray_BYTES(iKey);
    int64_t arraySizeKey = ArrayLength(iKey);

    if (aInfo->ArrayLength != arraySizeKey)
    {
        PyErr_Format(PyExc_ValueError, "GroupByAll32 iKey length does not match value length %lld %lld", aInfo->ArrayLength,
                     arraySizeKey);
        return NULL;
    }

    int overflow = 0;
    int64_t firstFuncNum = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listFuncNum, 0), &overflow);
    PyObject * returnTuple = NULL;

    // NOTE: what if bin size 10x larger?
    if (true && ((unique_rows * 10) < arraySizeKey) && tupleSize == 1)
    {
        int64_t binLow = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinLow, 0), &overflow);
        int64_t binHigh = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinHigh, 0), &overflow);

        if ((firstFuncNum >= GB_SUM && firstFuncNum <= GB_MAX) || (firstFuncNum >= GB_NANSUM && firstFuncNum <= GB_NANMAX))

        {
            // multithread by data segments (NOT bin ranges)
            // scatter/gather technique -- no memory is read twice
            returnTuple =
                GroupBySingleOpMultithreaded(aInfo, iKey, (GB_FUNCTIONS)firstFuncNum, unique_rows, tupleSize, binLow, binHigh);
        }
    }

    //-----------------------------------------------------------
    //
    if (returnTuple == NULL)
    {
        std::vector<workspace_mem_ptr> workspaceMemList;

        // Allocate the struct + ROOM at the end of struct for all the tuple objects
        // being produced
        workspaceMemList.emplace_back(WORKSPACE_ALLOC(sizeof(stGroupBy32) + (tupleSize * sizeof(stGroupByReturn))));
        stGroupBy32 * pstGroupBy32 = (stGroupBy32 *)workspaceMemList.back().get();

        if (pstGroupBy32 == NULL)
        {
            // out of memory
            return NULL;
        }

        pstGroupBy32->aInfo = aInfo;
        pstGroupBy32->pDataIn2 = pDataIn2;
        pstGroupBy32->itemSize2 = PyArray_ITEMSIZE(iKey);
        pstGroupBy32->tupleSize = tupleSize;
        pstGroupBy32->uniqueRows = unique_rows;

        pstGroupBy32->totalInputRows = arraySizeKey;
        pstGroupBy32->typeOfFunctionCall = TYPE_OF_FUNCTION_CALL::ANY_GROUPBY_FUNC;

        // Allocate all the memory and output arrays up front since Python is single
        // threaded
        for (int i = 0; i < tupleSize; i++)
        {
            // TODO: determine based on function
            int32_t numpyOutType = -1;
            int32_t numpyTmpType = -1;
            bool hasCounts = false;

            int overflow = 0;
            int64_t funcNum = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listFuncNum, i), &overflow);
            pstGroupBy32->returnObjects[i].funcNum = (int32_t)funcNum;

            int64_t binLow = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinLow, i), &overflow);
            pstGroupBy32->returnObjects[i].binLow = binLow;

            int64_t binHigh = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinHigh, i), &overflow);
            pstGroupBy32->returnObjects[i].binHigh = binHigh;

            GROUPBY_TWO_FUNC pFunction =
                GetGroupByFunctionStep1(iKeyType, &hasCounts, &numpyOutType, &numpyTmpType, aInfo[i].NumpyDType, (GB_FUNCTIONS)funcNum);

            PyArrayObject * outArray = NULL;
            int32_t * pCountOut = NULL;
            void * pOutArray = NULL;
            void * pTmpArray = nullptr;

            if (pFunction && numpyOutType != -1)
            {
                // Dont bother allocating if we cannot call the function
                outArray = AllocateNumpyArray(1, (npy_intp *)&unique_rows, numpyOutType);
                CHECK_MEMORY_ERROR(outArray);

                if (outArray == NULL)
                {
                    return NULL;
                }

                pOutArray = PyArray_BYTES(outArray);
                int64_t itemSize = PyArray_ITEMSIZE(outArray);

                if (hasCounts)
                {
                    // Zero out for them
                    workspaceMemList.emplace_back(WORKSPACE_ALLOC(sizeof(int32_t) * unique_rows));
                    pCountOut = (int32_t *)workspaceMemList.back().get();
                    if (pCountOut == NULL)
                    {
                        return NULL;
                    }
                    memset(pCountOut, 0, sizeof(int32_t) * unique_rows);
                }

                if (numpyTmpType >= 0)
                {
                    int32_t tempItemSize = NpyToSize(numpyTmpType);
                    workspaceMemList.emplace_back(WORKSPACE_ALLOC(tempItemSize * (binHigh - binLow)));
                    pTmpArray = workspaceMemList.back().get();
                    if (!pTmpArray)
                    {
                        return NULL;
                    }
                }
            }
            else
            {
                LOGGING("Failed to find function %llu for type %d\n", funcNum, numpyOutType);
            }

            pstGroupBy32->returnObjects[i].outArray = outArray;
            pstGroupBy32->returnObjects[i].pOutArray = pOutArray;
            pstGroupBy32->returnObjects[i].pTmpArray = pTmpArray;
            pstGroupBy32->returnObjects[i].pCountOut = pCountOut;
            pstGroupBy32->returnObjects[i].pFunction = pFunction;
            pstGroupBy32->returnObjects[i].returnObject = Py_None;
            pstGroupBy32->returnObjects[i].numpyOutType = numpyOutType;
        }

        g_cMathWorker->WorkGroupByCall(GroupByCall, pstGroupBy32, tupleSize);

        LOGGING("!!groupby done %llu\n", tupleSize);

        // New reference
        returnTuple = PyTuple_New(tupleSize);
        PyObject * returnCount = NULL;

        // Fill in results
        for (int i = 0; i < tupleSize; i++)
        {
            PyObject * item = pstGroupBy32->returnObjects[i].returnObject;

            if (item == Py_None)
                Py_INCREF(Py_None);

            // printf("ref %d  %llu\n", i, item->ob_refcnt);
            PyTuple_SET_ITEM(returnTuple, i, item);

            int32_t * pCountOut = pstGroupBy32->returnObjects[i].pCountOut;
        }
    }

    // LOGGING("Return tuple ref %llu\n", returnTuple->ob_refcnt);
    FreeArrayInfo(aInfo);

    LOGGING("!!groupby returning\n");

    return returnTuple;
}

//---------------------------------------------------------------
// Arg1 = LIST of numpy arrays which has the values to accumulate (often all the
// columns in a dataset) Arg2 =iKey = numpy array (int32_t) which has the index
// to the unique keys (ikey from MultiKeyGroupBy32) Arg3 =iGroup: (int32_t)
// array size is same as multikey, unique keys are grouped together Arg4
// =iFirst: (int32_t) array size is number of unique keys, indexes into iGroup
// Arg5 =nCount: (int32_t) array size is number of unique keys for the group, is
// how many member of the group (paired with iFirst) Arg6 = integer unique rows
// Arg7 = LIST of integers (function number to execute for sum,mean,min, max)
// Arg8 = LIST of integers (binLow -- invalid bin)
// Arg9 = funcParam (?? should be a list?)
// Example: GroupByOp2(array, ikey, 3, np.float32)
// Returns AccumBins
//
// Formula for first
// AccumBin[iKey[iFirst[i]]] = Column[iFirst[i]]

PyObject * GroupByAllPack32(PyObject * self, PyObject * args)
{
    PyObject * inList1 = NULL;
    PyArrayObject * iKey = NULL;
    PyArrayObject * iGroup;
    PyArrayObject * iFirst;
    PyArrayObject * nCount;

    int64_t unique_rows = 0;
    PyListObject * listFuncNum = 0;
    PyListObject * listBinLow = 0;
    PyListObject * listBinHigh = 0;
    int64_t funcParam = 0;

    if (! PyArg_ParseTuple(args, "OO!O!O!O!LO!O!O!L", &inList1, &PyArray_Type, &iKey, &PyArray_Type, &iGroup, &PyArray_Type,
                           &iFirst, &PyArray_Type, &nCount,

                           &unique_rows, &PyList_Type, &listFuncNum, &PyList_Type, &listBinLow, &PyList_Type, &listBinHigh,
                           &funcParam))
    {
        return NULL;
    }

    LOGGING("GroupByAllPack32 types: key:%d  group:%d  first:%d  count:%d\n", PyArray_TYPE(iKey), PyArray_TYPE(iGroup),
            PyArray_TYPE(iFirst), PyArray_TYPE(nCount));

    int32_t iKeyType = PyArray_TYPE(iKey);

    // Valid types we can index by
    switch (iKeyType)
    {
    case NPY_INT8:
    case NPY_INT16:
    CASE_NPY_INT32:
    CASE_NPY_INT64:

        break;
    default:
        PyErr_Format(PyExc_ValueError, "GroupByAllPack32 key param must int8, int16, int32, int64");
        return NULL;
    }

    // Add 1 for zero bin
    unique_rows += GB_BASE_INDEX;

    int64_t totalItemSize = 0;
    int64_t tupleSize = 0;
    ArrayInfo * aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize);

    if (! aInfo)
    {
        return NULL;
    }

    // New reference
    PyObject * returnTuple = NULL;
    int64_t arraySizeKey = ArrayLength(iKey);

    if (tupleSize == 1 && arraySizeKey > 65536)
    {
        int overflow = 0;
        int64_t binLow = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinLow, 0), &overflow);
        int64_t binHigh = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinHigh, 0), &overflow);

        int64_t firstFuncNum = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listFuncNum, 0), &overflow);

        LOGGING("Checking banded %lld\n", firstFuncNum);

        if ((firstFuncNum >= GB_MEDIAN && firstFuncNum <= GB_TRIMBR))

        {
            returnTuple = GroupBySingleOpMultiBands(aInfo, iKey, iFirst, iGroup, nCount, (GB_FUNCTIONS)firstFuncNum, unique_rows,
                                                    tupleSize, binLow, binHigh);
        }
    }

    if (returnTuple == NULL)
    {
        int64_t funcTupleSize = PyList_GET_SIZE(listFuncNum);

        if (tupleSize != funcTupleSize)
        {
            PyErr_Format(PyExc_ValueError, "GroupByAll32 func numbers do not match array columns %lld %lld", tupleSize,
                         funcTupleSize);
        }

        int64_t binTupleSize = PyList_GET_SIZE(listBinLow);

        if (tupleSize != binTupleSize)
        {
            PyErr_Format(PyExc_ValueError, "GroupByAll32 bin numbers do not match array columns %lld %lld", tupleSize,
                         binTupleSize);
            return NULL;
        }

        // TODO: determine based on function
        int32_t numpyOutType = NPY_FLOAT64;
        int32_t numpyInType2 = ObjectToDtype(iKey);

        // Since this is the 32 bit function, the array indexes are 32 bit
        void * pDataIn2 = PyArray_BYTES(iKey);

        // Allocate the struct + ROOM at the end of struct for all the tuple objects
        // being produced
        int64_t allocSize = (sizeof(stGroupBy32) + 8 + sizeof(stGroupByReturn)) * tupleSize;
        LOGGING("in groupby32 allocating %lld\n", allocSize);

        stGroupBy32 * pstGroupBy32 = (stGroupBy32 *)WORKSPACE_ALLOC(allocSize);

        pstGroupBy32->aInfo = aInfo;
        pstGroupBy32->pDataIn2 = pDataIn2;
        pstGroupBy32->itemSize2 = PyArray_ITEMSIZE(iKey);
        pstGroupBy32->tupleSize = tupleSize;
        pstGroupBy32->uniqueRows = unique_rows;

        pstGroupBy32->totalInputRows = ArrayLength(iKey);

        // printf("funcParam %lld\n", funcParam);
        pstGroupBy32->funcParam = funcParam;

        pstGroupBy32->pKey = PyArray_BYTES(iKey);
        pstGroupBy32->pFirst = PyArray_BYTES(iFirst);
        pstGroupBy32->pGroup = PyArray_BYTES(iGroup);
        pstGroupBy32->pCount = PyArray_BYTES(nCount);
        pstGroupBy32->typeOfFunctionCall = TYPE_OF_FUNCTION_CALL::ANY_GROUPBY_XFUNC32;

        // Allocate all the memory and output arrays up front since Python is single
        // threaded
        for (int i = 0; i < tupleSize; i++)
        {
            int overflow = 0;
            int64_t funcNum = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listFuncNum, i), &overflow);
            pstGroupBy32->returnObjects[i].funcNum = (int32_t)funcNum;

            int64_t binLow = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinLow, i), &overflow);
            pstGroupBy32->returnObjects[i].binLow = binLow;

            int64_t binHigh = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinHigh, i), &overflow);
            pstGroupBy32->returnObjects[i].binHigh = binHigh;

            numpyOutType = aInfo[i].NumpyDType;

            GROUPBY_X_FUNC32 pFunction = NULL;

            switch (iKeyType)
            {
            case NPY_INT8:
                pFunction = GetGroupByXFunction32<int8_t>(numpyOutType, numpyOutType, (GB_FUNCTIONS)funcNum);
                break;
            case NPY_INT16:
                pFunction = GetGroupByXFunction32<int16_t>(numpyOutType, numpyOutType, (GB_FUNCTIONS)funcNum);
                break;
            CASE_NPY_INT32:
                pFunction = GetGroupByXFunction32<int32_t>(numpyOutType, numpyOutType, (GB_FUNCTIONS)funcNum);
                break;
            CASE_NPY_INT64:

                pFunction = GetGroupByXFunction32<int64_t>(numpyOutType, numpyOutType, (GB_FUNCTIONS)funcNum);
                break;
            }

            PyArrayObject * outArray = NULL;

            if (pFunction)
            {
                // dont allocate if no function to call
                // pull in strings also
                if (funcNum == GB_TRIMBR)
                {
                    // Variance must be in float form
                    numpyOutType = NPY_FLOAT64;

                    // Everything is a float64 unless it is already a float32, then we
                    // keep it as float32
                    if (aInfo[i].NumpyDType == NPY_FLOAT32)
                    {
                        numpyOutType = NPY_FLOAT32;
                    }
                    outArray = AllocateNumpyArray(1, (npy_intp *)&unique_rows, numpyOutType);
                    CHECK_MEMORY_ERROR(outArray);
                }
                else
                {
                    // For functions in the 200+ range like rolling we use all the items
                    if (funcNum >= GB_ROLLING_SUM)
                    {
                        // shift and diff keep the same dtype
                        if (funcNum == GB_ROLLING_SUM || funcNum == GB_ROLLING_NANSUM || funcNum == GB_ROLLING_COUNT)
                        {
                            numpyOutType = NPY_INT64;

                            if (funcNum == GB_ROLLING_COUNT)
                            {
                                numpyOutType = NPY_INT32;
                            }
                            else
                            {
                                switch (aInfo[i].NumpyDType)
                                {
                                case NPY_FLOAT32:
                                    numpyOutType = NPY_FLOAT32;
                                    break;
                                case NPY_DOUBLE:
                                case NPY_LONGDOUBLE:
                                    numpyOutType = NPY_FLOAT64;
                                    break;
                                CASE_NPY_UINT64:

                                    numpyOutType = NPY_UINT64;
                                    break;
                                }
                            }
                        }
                        else if (funcNum == GB_ROLLING_MEAN || funcNum == GB_ROLLING_NANMEAN)
                        {
                            numpyOutType = NPY_FLOAT64;
                        }

                        if (aInfo[i].ArrayLength != pstGroupBy32->totalInputRows)
                        {
                            PyErr_Format(PyExc_ValueError,
                                         "GroupByAll32 for rolling functions, input size "
                                         "must be same size as group size: %lld vs %lld",
                                         aInfo[i].ArrayLength, pstGroupBy32->totalInputRows);
                            goto ERROR_EXIT;
                        }

                        outArray = AllocateLikeNumpyArray(aInfo[i].pObject, numpyOutType);
                    }
                    else
                    {
                        LOGGING("GBALLPACK32:  Allocating for output type: %d\n", aInfo[i].NumpyDType);
                        outArray = AllocateLikeResize(aInfo[i].pObject, unique_rows);
                    }
                }

                // Bail if out of memory (possible memory leak)
                if (outArray == NULL)
                {
                    goto ERROR_EXIT;
                }
            }
            else
            {
                LOGGING("Failed to find function %llu for type %d\n", funcNum, numpyOutType);
            }

            pstGroupBy32->returnObjects[i].outArray = outArray;
            pstGroupBy32->returnObjects[i].pFunctionX32 = pFunction;
            pstGroupBy32->returnObjects[i].returnObject = Py_None;
            pstGroupBy32->returnObjects[i].numpyOutType = numpyOutType;
        }

        g_cMathWorker->WorkGroupByCall(GroupByCall, pstGroupBy32, tupleSize);

        LOGGING("!!groupby done %llu\n", tupleSize);

        // New reference
        returnTuple = PyTuple_New(tupleSize);

        // Fill in results
        for (int i = 0; i < tupleSize; i++)
        {
            PyObject * item = pstGroupBy32->returnObjects[i].returnObject;

            if (item == Py_None)
                Py_INCREF(Py_None);

            // printf("ref %d  %llu\n", i, item->ob_refcnt);
            PyTuple_SET_ITEM(returnTuple, i, item);
        }

        // LOGGING("!!groupby done %llu\n", tupleSize);
        //// New reference
        // PyObject* returnTuple = PyTuple_New(tupleSize);

        //// Fill in results
        // for (int i = 0; i < tupleSize; i++) {
        //   PyArrayObject* pAccumObject = pstGroupBy32->returnObjects[i].outArray;

        //   void* pAccumBin = PyArray_BYTES(pAccumObject);
        //   GROUPBY_X_FUNC32  pFunction =
        //   GetGroupByXFunction32(aInfo[i].NumpyDType, numpyOutType,
        //   (GB_FUNCTIONS)funcNum);

        //   if (pFunction) {
        //      // Perform op
        //      pFunction((int32_t*)pGroup, (int32_t*)pFirst, (int32_t*)pCount,
        //      (char*)pAccumBin, (char*)aInfo[i].pData, unique_rows,
        //      aInfo[i].ItemSize, funcParam); PyTuple_SET_ITEM(returnTuple, i,
        //      (PyObject*)pstGroupBy32->returnObjects[i].outArray);
        //   }
        //   else {
        //      Py_INCREF(Py_None);
        //      PyTuple_SET_ITEM(returnTuple, i, (PyObject*)Py_None);

        //   }
        //}

        LOGGING("Return tuple ref %llu\n", returnTuple->ob_refcnt);

    ERROR_EXIT:
        WORKSPACE_FREE(pstGroupBy32);
        FreeArrayInfo(aInfo);

        LOGGING("!!groupby returning\n");
    }

    return returnTuple;
}
