#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "MultiKey.h"
#include "GroupBy.h"
#include "Sort.h"
#include "Heap.h"
#include "missing_values.h"
#include "numpy_traits.h"

#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <vector>

#define LOGGING(...)
// #define LOGGING printf

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
    return (X + Y) / 2.0f;
}
inline double MEDIAN_SPLIT(double X, double Y)
{
    return (X + Y) / 2.0;
}
inline long double MEDIAN_SPLIT(long double X, long double Y)
{
    return (X + Y) / 2.0L;
}

template <typename T, typename U>
inline U QUANTILE_SPLIT(T X, T Y)
{
    return (X + Y) / 2.0;
}

// Overloads to handle cases of bool, float, long double
template <>
inline bool QUANTILE_SPLIT<bool, bool>(bool X, bool Y)
{
    return (X | Y);
}

template <>
inline float QUANTILE_SPLIT<float, float>(float X, float Y)
{
    return (X + Y) / 2.0f;
}

template <>
inline long double QUANTILE_SPLIT<long double, long double>(long double X, long double Y)
{
    return (X + Y) / 2.0L;
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
        if (riptide::invalid_for_type<T>::is_valid(x[i]))
            ++i;
        else
        {
            --j;
            if (riptide::invalid_for_type<T>::is_valid(x[j]))
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
// thus <float, int32> converts a float to an int32
// V - index type (int8_t, int16_t, int32_t, int64_t)
// W - pCountOut type int32_t or int64_t (not used for many functions)
template <typename T, typename U, typename V, typename W = void>
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

    static void AccumSum(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                         int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };
        W * const pCountOut{ (W *)pCountOutT };

        U const invalid{ riptide::invalid_for_type<U>::value };

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                // pCountOut = -1 means the answer for this bin is already invalid
                // (encountered invalid value before)
                if (pCountOut[index] >= 0)
                {
                    T const temp{ pIn[i] };

                    // check if this is a nan
                    if (riptide::invalid_for_type<T>::is_valid(temp))
                    {
                        pOut[index] += (U)temp;
                    }
                    else
                    {
                        pOut[index] = invalid;
                        pCountOut[index] = -1;
                    }
                }
            }
        }
    }

    // This routine is only for float32.  It will upcast to float64, add up all
    // the numbers, then convert back to float32
    static void AccumSumFloat(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                              int64_t binLow, int64_t binHigh, int64_t pass, void * pDataTmp)
    {
        float const * const pIn{ (float *)pDataIn };
        V const * const pIndex{ (V *)pIndexT };
        double * const pOutAccum{ static_cast<double *>(pDataTmp) };

        W * const pCountOut{ (W *)pCountOutT };

        double const invalid_double{ riptide::invalid_for_type<double>::value };
        float const invalid_float{ riptide::invalid_for_type<float>::value };

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOutAccum, 0, sizeof(double) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                // pCountOut = -1 means the answer for this bin is already invalid
                // (encountered invalid value before)
                if (pCountOut[index] >= 0)
                {
                    float const temp{ pIn[i] };
                    // check if this is a nan
                    if (riptide::invalid_for_type<float>::is_valid(temp))
                    {
                        pOutAccum[index - binLow] += (double)temp;
                    }
                    else
                    {
                        pOutAccum[index - binLow] = invalid_double;
                        pCountOut[index] = -1;
                    }
                }
            }
        }

        // Downcast from double to single
        float * const pOut{ (float *)pDataOut };
        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCountOut[i] >= 0)
            {
                pOut[i] = (float)pOutAccum[i - binLow];
            }
            else
            {
                pOut[i] = invalid_float;
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanSum(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                            int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T const temp{ pIn[i] };
                if (riptide::invalid_for_type<T>::is_valid(temp))
                {
                    pOut[index] += (U)temp;
                }
            }
        }
    }

    // This is for float only
    // $TODO: Adapted from AccumSumFloat: should refactor this common behavior and
    // potentially cache the working buffer in between invocations.
    static void AccumNanSumFloat(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                                 int64_t binLow, int64_t binHigh, int64_t pass, void * pDataTmp)
    {
        float const * const pIn{ (float *)pDataIn };
        V const * const pIndex{ (V *)pIndexT };
        double * const pOutAccum{ static_cast<double *>(pDataTmp) };

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOutAccum, 0, sizeof(double) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                float const temp{ pIn[i] };
                if (riptide::invalid_for_type<float>::is_valid(temp))
                {
                    pOutAccum[index - binLow] += (double)temp;
                }
            }
        }

        // Downcast from double to single
        float * const pOut{ (float *)pDataOut };
        for (int64_t i = binLow; i < binHigh; i++)
        {
            pOut[i] = (float)pOutAccum[i - binLow];
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumMin(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                         int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };
        W * const pCountOut{ (W *)pCountOutT };

        // Fill with invalid?
        U const invalid{ riptide::invalid_for_type<U>::value };

        if (pass <= 0)
        {
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i] = invalid;
            }
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                // pCountOut = -1 means the answer for this bin is already invalid
                // (encountered invalid value before)
                if (pCountOut[index] >= 0)
                {
                    T const temp{ pIn[i] };
                    // check if this is a nan
                    if (not riptide::invalid_for_type<T>::is_valid(temp))
                    {
                        // output invalid value and set pCountOut to -1
                        pOut[index] = invalid;
                        pCountOut[index] = -1;
                    }
                    else
                    {
                        U const tempOut{ temp };
                        if (pCountOut[index] == 0)
                        {
                            // first time
                            pOut[index] = tempOut;
                            pCountOut[index] = 1;
                        }
                        else if (tempOut < pOut[index])
                        {
                            pOut[index] = tempOut;
                        }
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanMin(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                            int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };

        // Fill with NaNs
        U const invalid{ riptide::invalid_for_type<U>::value };
        if (pass <= 0)
        {
            // printf("NanMin clearing at %p  %lld  %lld\n", pOut, binLow, binHigh);
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i] = invalid;
            }
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T const temp{ pIn[i] };
                // if temp is invalid, just ignore and continue
                if (not riptide::invalid_for_type<T>::is_valid(temp))
                {
                    continue;
                }
                // if the current answer is invalid (first valid data), or otherwise if comparison holds
                else if ((not riptide::invalid_for_type<T>::is_valid(pOut[index])) || (pOut[index] > temp))
                {
                    pOut[index] = temp;
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumMax(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                         int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };
        W * const pCountOut{ (W *)pCountOutT };

        // Fill with invalid?
        U const invalid{ riptide::invalid_for_type<U>::value };
        if (pass <= 0)
        {
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i] = invalid;
            }
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                // pCountOut = -1 means the answer for this bin is already invalid
                // (encountered invalid value before)
                if (pCountOut[index] >= 0)
                {
                    T const temp{ pIn[i] };
                    // check if this is a nan
                    if (not riptide::invalid_for_type<T>::is_valid(temp))
                    {
                        // output invalid value and set pCountOut to -1
                        pOut[index] = invalid;
                        pCountOut[index] = -1;
                    }
                    else
                    {
                        U const tempOut{ temp };
                        if (pCountOut[index] == 0)
                        {
                            // first time
                            pOut[index] = tempOut;
                            pCountOut[index] = 1;
                        }
                        else if (tempOut > pOut[index])
                        {
                            pOut[index] = tempOut;
                        }
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanMax(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                            int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };

        // Fill with invalid?
        U const invalid{ riptide::invalid_for_type<U>::value };
        if (pass <= 0)
        {
            for (int64_t i = binLow; i < binHigh; i++)
            {
                pOut[i] = invalid;
            }
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T const temp{ pIn[i] };
                // if temp is invalid, just ignore and continue
                if (not riptide::invalid_for_type<T>::is_valid(temp))
                {
                    continue;
                }
                // if the current answer is invalid (first valid data), or otherwise if comparison holds
                else if ((not riptide::invalid_for_type<T>::is_valid(pOut[index])) || (pOut[index] < temp))
                {
                    pOut[index] = temp;
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumMean(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                          int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };
        W * const pCountOut{ (W *)pCountOutT };

        U const invalid{ riptide::invalid_for_type<U>::value };

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                // pCountOut = -1 means the answer for this bin is already invalid
                // (encountered invalid value before)
                if (pCountOut[index] >= 0)
                {
                    T const temp{ pIn[i] };
                    if (not riptide::invalid_for_type<T>::is_valid(temp))
                    {
                        // output invalid value and set pCountOut to -1
                        pOut[index] = invalid;
                        pCountOut[index] = -1;
                    }
                    else
                    {
                        pOut[index] += (U)temp;
                        pCountOut[index]++;
                    }
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
                    pOut[i] = invalid;
                }
            }
        }
    }

    // Just for float32 since we can upcast
    //-------------------------------------------------------------------------------
    static void AccumMeanFloat(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                               int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOriginalOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };
        W * const pCountOut{ (W *)pCountOutT };

        U const invalid{ riptide::invalid_for_type<U>::value };
        double const invalid_double{ riptide::invalid_for_type<double>::value };

        // Allocate pOut
        double * const pOut = (double *)WORKSPACE_ALLOC(sizeof(double) * (binHigh - binLow));
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
                V const index{ pIndex[i] };

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    // pCountOut = -1 means the answer for this bin is already invalid
                    // (encountered invalid value before)
                    if (pCountOut[index] >= 0)
                    {
                        double const temp{ pIn[i] };
                        if (not riptide::invalid_for_type<double>::is_valid(temp))
                        {
                            // output invalid value and set pCountOut to -1
                            pOut[index - binLow] = invalid_double;
                            pCountOut[index] = -1;
                        }
                        else
                        {
                            pOut[index - binLow] += (double)temp;
                            pCountOut[index]++;
                        }
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
                        pOriginalOut[i] = invalid;
                    }
                }
            }
            else
            {
                // copy over original values
                for (int64_t i = binLow; i < binHigh; i++)
                {
                    if (pCountOut[i] >= 0)
                    {
                        pOriginalOut[i] = (U)pOut[i - binLow];
                    }
                    else
                    {
                        pOriginalOut[i] = invalid;
                    }
                }
            }

            WORKSPACE_FREE(pOut);
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanMean(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                             int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };
        W * const pCountOut{ (W *)pCountOutT };

        U const invalid{ riptide::invalid_for_type<U>::value };

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T const temp{ pIn[i] };
                if (riptide::invalid_for_type<T>::is_valid(temp))
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
                    pOut[i] = invalid;
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanMeanFloat(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                                  int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOriginalOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };
        W * const pCountOut{ (W *)pCountOutT };

        U const invalid{ riptide::invalid_for_type<U>::value };
        // Allocate pOut
        double * const pOut = (double *)WORKSPACE_ALLOC(sizeof(double) * (binHigh - binLow));
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
                V const index{ pIndex[i] };

                //--------------------------------------
                ACCUM_INNER_LOOP(index, binLow, binHigh)
                {
                    T const temp{ pIn[i] };
                    if (riptide::invalid_for_type<T>::is_valid(temp))
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
                        pOriginalOut[i] = invalid;
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
    static void AccumVar(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                         int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };
        W * const pCountOut{ (W *)pCountOutT };

        U const invalid{ riptide::invalid_for_type<U>::value };

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        // TODO: optimize this for our range
        U * const sumsquares = (U *)WORKSPACE_ALLOC(sizeof(U) * binHigh);
        memset(sumsquares, 0, sizeof(U) * binHigh);

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                // pCountOut = -1 means the answer for this bin is already invalid
                // (encountered invalid value before)
                if (pCountOut[index] >= 0)
                {
                    T const temp{ pIn[i] };
                    if (not riptide::invalid_for_type<T>::is_valid(temp))
                    {
                        // output invalid value and set pCountOut to -1
                        pOut[index] = invalid;
                        pCountOut[index] = -1;
                    }
                    else
                    {
                        pOut[index] += (U)temp;
                        pCountOut[index]++;
                    }
                }
            }
        }

        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCountOut[i] >= 0)
            {
                pOut[i] /= (U)(pCountOut[i]);
            }
        }

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                if (pCountOut[index] >= 0)
                {
                    // since this is a second pass, already nothing is invalid in `index` bin
                    T const temp{ pIn[i] };
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
                pOut[i] = invalid;
            }
        }
        WORKSPACE_FREE(sumsquares);
    }

    //-------------------------------------------------------------------------------
    static void AccumNanVar(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                            int64_t binLow, int64_t binHigh, int64_t pass, void * /*pDataTmp*/)
    {
        T const * const pIn{ (T *)pDataIn };
        U * const pOut{ (U *)pDataOut };
        V const * const pIndex{ (V *)pIndexT };
        W * const pCountOut{ (W *)pCountOutT };

        U const invalid{ riptide::invalid_for_type<U>::value };

        if (pass <= 0)
        {
            // Clear out memory for our range
            memset(pOut + binLow, 0, sizeof(U) * (binHigh - binLow));
        }

        U * sumsquares = (U *)WORKSPACE_ALLOC(sizeof(U) * binHigh);
        memset(sumsquares, 0, sizeof(U) * binHigh);

        for (int64_t i = 0; i < len; i++)
        {
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T const temp{ pIn[i] };
                if (riptide::invalid_for_type<T>::is_valid(temp))
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
            V const index{ pIndex[i] };

            //--------------------------------------
            ACCUM_INNER_LOOP(index, binLow, binHigh)
            {
                T const temp{ pIn[i] };
                if (riptide::invalid_for_type<T>::is_valid(temp))
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
                pOut[i] = invalid;
            }
        }
        WORKSPACE_FREE(sumsquares);
    }

    //-------------------------------------------------------------------------------
    static void AccumStd(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                         int64_t binLow, int64_t binHigh, int64_t pass, void * pDataTmp)
    {
        U * pOut = (U *)pDataOut;

        AccumVar(pDataIn, pIndexT, pCountOutT, pDataOut, len, binLow, binHigh, pass, pDataTmp);
        for (int64_t i = binLow; i < binHigh; i++)
        {
            pOut[i] = sqrt(pOut[i]);
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNanStd(void const * pDataIn, void const * pIndexT, void * pCountOutT, void * pDataOut, int64_t len,
                            int64_t binLow, int64_t binHigh, int64_t pass, void * pDataTmp)
    {
        U * pOut = (U *)pDataOut;

        AccumNanVar(pDataIn, pIndexT, pCountOutT, pDataOut, len, binLow, binHigh, pass, pDataTmp);
        for (int64_t i = binLow; i < binHigh; i++)
        {
            pOut[i] = sqrt(pOut[i]);
        }
    }

    //-------------------------------------------------------------------------------
    // V is is the INDEX Type
    static void AccumNth(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT, void * pAccumBin,
                         int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        int64_t nth = funcParam;

        U const invalid{ riptide::invalid_for_type<U>::value };

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            auto const nthIndex{ nth >= 0 ? nth : pCount[i] + nth }; // handle wraparound
            if (pCount[i] > 0 && nthIndex >= 0 && nthIndex < pCount[i])
            {
                V grpIndex = pFirst[i] + nthIndex;
                V bin = pGroup[grpIndex];
                pDest[i] = pSrc[bin];
            }
            else
            {
                pDest[i] = invalid;
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumNthString(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                               void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                               int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        int64_t nth = funcParam;

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            auto const nthIndex{ nth >= 0 ? nth : pCount[i] + nth }; // handle wraparound
            if (pCount[i] > 0 && nthIndex >= 0 && nthIndex < pCount[i])
            {
                V grpIndex = pFirst[i] + nthIndex;
                V bin = pGroup[grpIndex];
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
    static void AccumFirst(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                           void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                           int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        LOGGING("in accum first low: %lld  high: %lld   group:%p  first:%p  count:%p\n", binLow, binHigh, pGroup, pFirst, pCount);

        U const invalid{ riptide::invalid_for_type<U>::value };

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            // printf("[%lld]", i);
            if (pCount[i] > 0)
            {
                V grpIndex = pFirst[i];
                // printf("(%d)", grpIndex);
                V bin = pGroup[grpIndex];
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
    static void AccumFirstString(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                 void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                 int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCount[i] > 0)
            {
                V grpIndex = pFirst[i];
                V bin = pGroup[grpIndex];
                memcpy(&pDest[i * itemSize], &pSrc[bin * itemSize], itemSize);
            }
            else
            {
                memset(&pDest[i * itemSize], 0, itemSize);
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumLast(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT, void * pAccumBin,
                          int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        U const invalid{ riptide::invalid_for_type<U>::value };
        // printf("last called %lld -- %llu %llu %llu\n", numUnique, sizeof(T),
        // sizeof(U), sizeof(V));

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            // printf("Last:  %d %d\n", (int)pFirst[i], (int)pCount[i]);
            // printf("Last2:  %d\n", (int)(pGroup[pFirst[i] + pCount[i] - 1]));
            if (pCount[i] > 0)
            {
                V grpIndex = pFirst[i] + pCount[i] - 1;
                V bin = pGroup[grpIndex];
                pDest[i] = pSrc[bin];
            }
            else
            {
                pDest[i] = invalid;
            }
        }
    }

    //-------------------------------------------------------------------------------
    static void AccumLastString(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            if (pCount[i] > 0)
            {
                V grpIndex = pFirst[i] + pCount[i] - 1;
                V bin = pGroup[grpIndex];
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
    static void AccumRollingSum(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        int64_t windowSize = funcParam;
        U const invalid{ riptide::invalid_for_type<U>::value };

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            V start = pFirst[0];
            V last = start + pCount[0];
            for (V j = start; j < last; j++)
            {
                V const index{ pGroup[j] };
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
            V start = pFirst[i];
            V last = start + pCount[i];

            U currentSum = 0;

            // Priming of the summation
            for (V j = start; j < last && j < (start + windowSize); j++)
            {
                V const index{ pGroup[j] };

                currentSum += (U)pSrc[index];
                pDest[index] = currentSum;
            }

            for (V j = start + windowSize; j < last; j++)
            {
                V const index{ pGroup[j] };

                currentSum += (U)pSrc[index];

                // subtract the item leaving the window
                currentSum -= (U)pSrc[pGroup[j - windowSize]];

                pDest[index] = currentSum;
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    static void AccumRollingNanSum(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                   void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                   int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        int64_t windowSize = funcParam;
        U const invalid{ riptide::invalid_for_type<U>::value };

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            V start = pFirst[0];
            V last = start + pCount[0];
            for (V j = start; j < last; j++)
            {
                V const index{ pGroup[j] };
                pDest[index] = invalid;
            }
            binLow++;
        }

        // negative window sizes not accepted yet
        if (windowSize < 0)
            return;

        for (int64_t i = binLow; i < binHigh; i++)
        {
            V start = pFirst[i];
            V last = start + pCount[i];

            U currentSum = 0;

            // Priming of the summation
            for (V j = start; j < last && j < (start + windowSize); j++)
            {
                V const index{ pGroup[j] };
                U value = (U)pSrc[index];
                if (riptide::invalid_for_type<T>::is_valid(value))
                {
                    currentSum += value;
                }
                pDest[index] = currentSum;
            }

            for (V j = start + windowSize; j < last; j++)
            {
                V const index{ pGroup[j] };

                U value = (U)pSrc[index];
                if (riptide::invalid_for_type<T>::is_valid(value))
                {
                    currentSum += value;
                }

                // subtract the item leaving the window
                value = (U)pSrc[pGroup[j - windowSize]];
                if (riptide::invalid_for_type<T>::is_valid(value))
                {
                    currentSum -= value;
                }

                pDest[index] = currentSum;
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    static void AccumRollingMean(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                 void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                 int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        int64_t const windowSize = funcParam;
        U const invalid_out{ riptide::invalid_for_type<U>::value };

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            V start = pFirst[0];
            V last = start + pCount[0];
            for (V j = start; j < last; j++)
            {
                V const index{ pGroup[j] };
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
            V start = pFirst[i];
            V last = start + pCount[i];

            double currentSum = 0;

            // Priming of the summation
            for (V j = start; j < last && j < (start + windowSize); j++)
            {
                V const index{ pGroup[j] };

                currentSum += pSrc[index];
                pDest[index] = currentSum / (j - start + 1);
            }

            for (V j = start + windowSize; j < last; j++)
            {
                V const index{ pGroup[j] };

                currentSum += pSrc[index];

                // subtract the item leaving the window
                currentSum -= pSrc[pGroup[j - windowSize]];

                pDest[index] = currentSum / windowSize;
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    static void AccumRollingNanMean(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                    void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                    int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        int64_t const windowSize = funcParam;
        U const invalid_out{ riptide::invalid_for_type<U>::value };

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            V start = pFirst[0];
            V last = start + pCount[0];
            for (V j = start; j < last; j++)
            {
                V const index{ pGroup[j] };
                pDest[index] = invalid_out;
            }
            binLow++;
        }

        // negative window sizes not accepted yet
        if (windowSize < 0)
            return;

        for (int64_t i = binLow; i < binHigh; i++)
        {
            V start = pFirst[i];
            V last = start + pCount[i];

            double currentSum = 0;
            double count = 0;

            // Priming of the summation
            for (V j = start; j < last && j < (start + windowSize); j++)
            {
                V const index{ pGroup[j] };
                U value = (U)pSrc[index];
                if (riptide::invalid_for_type<T>::is_valid(value))
                {
                    currentSum += value;
                    count++;
                }
                pDest[index] = count > 0 ? currentSum / count : invalid_out;
            }

            for (V j = start + windowSize; j < last; j++)
            {
                V const index{ pGroup[j] };

                U value = (U)pSrc[index];
                if (riptide::invalid_for_type<T>::is_valid(value))
                {
                    currentSum += value;
                    count++;
                }

                // subtract the item leaving the window
                value = (U)pSrc[pGroup[j - windowSize]];
                if (riptide::invalid_for_type<T>::is_valid(value))
                {
                    currentSum -= value;
                    count--;
                }

                pDest[index] = count > 0 ? currentSum / count : invalid_out;
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    static void AccumRollingQuantile1e9Mult(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                            void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows,
                                            int64_t itemSize, int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        static constexpr int64_t multiplier{ static_cast<int64_t>(1e9) };
        // funcParam = q * multiplier + (window_size) * (multiplier + 1),
        // where q is the quantile to take, window_size is the size of the rolling window.

        int64_t quantile_with_1e9_mult{ funcParam % (multiplier + 1) };
        long double const quantile{ static_cast<long double>(quantile_with_1e9_mult) / static_cast<long double>(multiplier) };
        int64_t const windowSize = funcParam / (multiplier + 1);

        U const invalid{ riptide::invalid_for_type<U>::value };

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            V start{ pFirst[0] };
            V last{ start + pCount[0] };
            for (V j = start; j < last; j++)
            {
                V index{ pGroup[j] };
                pDest[index] = invalid;
            }
            binLow++;
        }

        if (windowSize <= 0)
        {
            // negative or zero window sizes not accepted yet
            PyErr_Format(PyExc_ValueError, "Negative or zero window sizes are not supported for rolling_quantile.");
            return;
        }
        else if (windowSize == 1)
        {
            // return data itself (probably can do better than copying everything, but no one should call this with window=1
            // anyway) (things in RollingQuantile:: will break (for now) with windowSize = 1)
            for (int64_t i = binLow; i < binHigh; i++)
            {
                V start{ pFirst[i] };
                V last{ start + pCount[i] };

                for (V j = start; j < last; j++)
                {
                    V index{ pGroup[j] };
                    if (riptide::invalid_for_type<T>::is_valid(pSrc[index]))
                    {
                        pDest[index] = pSrc[index];
                    }
                    else
                    {
                        pDest[index] = invalid;
                    }
                }
            }
        }
        else
        {
            using QElement = RollingQuantile::StructureElement<T>;

            // Space will be reused for each bin. Only need O(window) extra space
            QElement * all_elements = (QElement *)WORKSPACE_ALLOC(windowSize * sizeof(QElement));
            QElement ** all_element_pointers = (QElement **)WORKSPACE_ALLOC(windowSize * sizeof(QElement *));

            // max_heap_size is (floor((windowSize - 1) * quantile) + 1), but need to handle some precision issues
            size_t max_heap_size = RollingQuantile::IntegralIndex(windowSize, quantile) + 1;
            size_t min_heap_size = windowSize - max_heap_size;

            // Can't pass another parameter here at this point, fix to 1
            size_t min_count = 1;

            RollingQuantile::RollingQuantile<T, U> rolling_quantile(windowSize, quantile, all_elements, all_element_pointers,
                                                                    max_heap_size, min_heap_size, min_count, QUANTILE_SPLIT<T, U>);

            for (int64_t i = binLow; i < binHigh; i++)
            {
                V start{ pFirst[i] };
                V last{ start + pCount[i] };

                for (V j = start; j < last; j++)
                {
                    V index{ pGroup[j] };
                    pDest[index] = rolling_quantile.Update(&pSrc[index]);
                }

                rolling_quantile.Clean();
            }
            WORKSPACE_FREE(all_elements);
            WORKSPACE_FREE(all_element_pointers);
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    static void AccumRollingCount(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                  void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                  int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        int64_t const windowSize = funcParam;
        U const invalid{ riptide::invalid_for_type<U>::value };

        LOGGING("in rolling count %lld %lld  sizeofdest %lld\n", binLow, binHigh, sizeof(U));

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            V start = pFirst[0];
            V last = start + pCount[0];
            for (V j = start; j < last; j++)
            {
                V const index{ pGroup[j] };
                pDest[index] = invalid;
            }
            binLow++;
        }

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            V start = pFirst[i];
            V last = start + pCount[i];

            U currentSum = 0;

            // printf("in rolling count [%lld] %d %d\n", i, start, last);

            if (windowSize < 0)
            {
                for (V j = last - 1; j >= start; j--)
                {
                    V const index{ pGroup[j] };
                    pDest[index] = currentSum;
                    currentSum += 1;
                }
            }
            else
            {
                for (V j = start; j < last; j++)
                {
                    V const index{ pGroup[j] };
                    pDest[index] = currentSum;
                    currentSum += 1;
                }
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    // NOTE: pDest/pAccumBin must be the size
    static void AccumRollingShift(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                  void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                  int64_t funcParam)
    {
        constexpr bool is_scalar{ ! (is_flexible_v<T> && is_flexible_v<U>)};

        using TData = std::conditional_t<is_scalar, T, char>;
        using UData = std::conditional_t<is_scalar, U, char>;

        TData const * const pSrc = (TData *)pColumn;
        UData * const pDest = (UData *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        int64_t windowSize = (int64_t)funcParam;

        // printf("binlow %lld,  binhigh %lld,  windowSize: %d\n", binLow, binHigh,
        // windowSize);

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            V start = pFirst[0];
            V last = start + pCount[0];
            for (V j = start; j < last; j++)
            {
                V const index{ pGroup[j] };
                if constexpr (is_scalar)
                {
                    pDest[index] = riptide::invalid_for_type<U>::value;
                }
                else
                {
                    memset(&pDest[index * itemSize], 0, itemSize);
                }
            }
            binLow++;
        }

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            V start = pFirst[i];
            V last = start + pCount[i];

            if (windowSize >= 0)
            {
                // invalid for window
                for (V j = start; j < last && j < (start + windowSize); j++)
                {
                    V const index{ pGroup[j] };
                    if constexpr (is_scalar)
                    {
                        pDest[index] = riptide::invalid_for_type<U>::value;
                    }
                    else
                    {
                        memset(&pDest[index * itemSize], 0, itemSize);
                    }
                }

                for (V j = start + windowSize; j < last; j++)
                {
                    V const index{ pGroup[j] };
                    if constexpr (is_scalar)
                    {
                        pDest[index] = (U)pSrc[pGroup[j - windowSize]];
                    }
                    else
                    {
                        memcpy(&pDest[index * itemSize], &pSrc[pGroup[j - windowSize] * itemSize], itemSize);
                    }
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

                for (V j = last; j > start && j > (last - windowSize); j--)
                {
                    V const index{ pGroup[j] };
                    if constexpr (is_scalar)
                    {
                        pDest[index] = riptide::invalid_for_type<U>::value;
                    }
                    else
                    {
                        memset(&pDest[index * itemSize], 0, itemSize);
                    }
                }

                for (V j = last - windowSize; j > start; j--)
                {
                    V const index{ pGroup[j] };
                    if constexpr (is_scalar)
                    {
                        pDest[index] = (U)pSrc[pGroup[j + windowSize]];
                    }
                    else
                    {
                        memcpy(&pDest[index * itemSize], &pSrc[pGroup[j + windowSize] * itemSize], itemSize);
                    }
                }
                // put it back to what it was
                windowSize = -windowSize;
            }
        }
    }

    //------------------------------
    // Rolling uses entire size: totalInputRows
    static void AccumRollingDiff(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                 void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                 int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        int64_t windowSize = funcParam;
        U const invalid{ riptide::invalid_for_type<U>::value };

        if (binLow == 0)
        {
            // Mark all invalid if invalid bin
            V start = pFirst[0];
            V last = start + pCount[0];
            for (V j = start; j < last; j++)
            {
                V const index{ pGroup[j] };
                pDest[index] = invalid;
            }
            binLow++;
        }

        if (windowSize == 1)
        {
            // For all the bins we have to fill
            for (int64_t i = binLow; i < binHigh; i++)
            {
                V start = pFirst[i];
                V last = start + pCount[i];

                if (last > start)
                {
                    // Very first is invalid
                    V index = pGroup[start];
                    U previous = (U)pSrc[index];
                    pDest[index] = invalid;

                    // Priming of the summation
                    for (V j = start + 1; j < last; j++)
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
                V start = pFirst[i];
                V last = start + pCount[i];
                if (windowSize >= 0)
                {
                    // invalid for window
                    U previous = 0;

                    for (V j = start; j < last && j < (start + windowSize); j++)
                    {
                        V const index{ pGroup[j] };
                        pDest[index] = invalid;
                    }

                    for (V j = start + windowSize; j < last; j++)
                    {
                        V const index{ pGroup[j] };
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

                    for (V j = last; j > start && j > (last - windowSize); j--)
                    {
                        V const index{ pGroup[j] };
                        pDest[index] = invalid;
                    }

                    for (V j = last - windowSize; j > start; j--)
                    {
                        V const index{ pGroup[j] };
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
    static void AccumTrimMeanBR(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        // Alloc worst case
        T * pSort = (T *)WORKSPACE_ALLOC(totalInputRows * sizeof(T));

        U const invalid{ riptide::invalid_for_type<U>::value };

        LOGGING("TrimMean rows: %lld\n", totalInputRows);

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            V index = pFirst[i];
            V nCount = pCount[i];

            if (nCount == 0)
            {
                pDest[i] = invalid;
                continue;
            }

            // Copy over the items for this group
            for (V j = 0; j < nCount; j++)
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
    // pGroup -> int32/64  (V typename)
    static void AccumMode(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT, void * pAccumBin,
                          int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize, int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        // Alloc worst case
        T * pSort = (T *)WORKSPACE_ALLOC(totalInputRows * sizeof(T));
        U const invalid{ riptide::invalid_for_type<U>::value };

        // printf("Mode %llu\n", totalInputRows);

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            V index = pFirst[i];
            V nCount = pCount[i];

            if (nCount == 0)
            {
                pDest[i] = GET_INVALID(pDest[i]);
                continue;
            }

            // Copy over the items for this group
            for (V j = 0; j < nCount; j++)
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

            nCount = (int64_t)((pEnd + 1) - pSort);

            if (nCount == 0)
            {
                // nothing valid
                pDest[i] = GET_INVALID(pDest[i]);
                continue;
            }

            U currValue = *pSort, bestValue = *pSort;
            int64_t currCount = 1, bestCount = 1;
            for (int64_t i = 1; i < nCount; ++i)
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
    // Quantiles and medians are all here (nan) and non-nan versions
    // pGroup, pFirst, pCount -> int32/64  (V typename)
    static void AccumQuantile1e9Mult(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                     void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                     int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        // pGroup -> int8_t/16/32/64  (V typename)
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        U const invalid{ riptide::invalid_for_type<U>::value };

        static constexpr double multiplier{ 1e9 };
        // funcParam = q * multiplier + (isNanQuantile) * (multiplier + 1),
        // where q is the quantile to take.
        // so funcParam = 5e8 is median, funcParam = (15e8 + 1) is nanmedian

        double quantile_with_1e9_mult{ static_cast<double>(funcParam) };

        bool const is_nan_quantile{ quantile_with_1e9_mult > multiplier };

        if (is_nan_quantile)
        {
            quantile_with_1e9_mult -= multiplier + 1;
        }

        double const quantile{ quantile_with_1e9_mult / multiplier };

        // Alloc
        T * const pSort = (T *)WORKSPACE_ALLOC(totalInputRows * sizeof(T));

        LOGGING("Quantile %llu  %lld  %lld  sizeof: %lld %lld %lld\n", totalInputRows, binLow, binHigh, sizeof(T), sizeof(U),
                sizeof(V));

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            V const index{ pFirst[i] };
            V nCount{ pCount[i] };

            if (nCount == 0)
            {
                pDest[i] = invalid;
                continue;
            }

            // Copy over the items for this group
            // check for nans while copying elements to pSort
            bool no_nans{ true };
            V pSort_index{ 0 };
            V pSrc_index{};
            for (V j = 0; j < nCount; j++)
            {
                pSrc_index = pGroup[index + j];
                if (not riptide::invalid_for_type<T>::is_valid(pSrc[pSrc_index]))
                {
                    // if see a nan and this is a non-nan function, break and return nan
                    if (not is_nan_quantile)
                    {
                        no_nans = false;
                        break;
                    }
                    // if see a nan and this is a nan-function, just ignore and don't copy
                    else
                    {
                        continue;
                    }
                }
                // if don't see a nan, copy over the value
                else
                {
                    pSort[pSort_index++] = pSrc[pSrc_index];
                }
            }

            // if there are nans and is non-nan quantile, set result to nan and go to next group
            if (not (is_nan_quantile || no_nans))
            {
                pDest[i] = invalid;
                continue;
            }

            // since handled nans already, know the correct number of non-nan elements
            nCount = pSort_index;

            if (nCount == 0)
            {
                // nothing valid
                pDest[i] = invalid;
                continue;
            }

            U answer{};

            // find the quantile. It will be at index (N - 1) * quantile
            // there are no nans between pSort and pSort + nCount,
            // so apply nth_element

            double const frac_index{ (nCount - 1) * quantile };

            int64_t const idx_round_up{ static_cast<int64_t>(ceil(frac_index)) };
            int64_t const idx_round_down{ static_cast<int64_t>(floor(frac_index)) };

            if (idx_round_up == idx_round_down)
            {
                answer = (U)get_nth_element<T>(pSort, pSort + nCount, idx_round_up);
            }
            else
            {
                // for now only "midpoint" interpolation
                // first `get_nth_element` modifies pSort (partial sorting)
                // only need max element after partial sorting
                T const upper{ get_nth_element<T>(pSort, pSort + nCount, idx_round_up) };
                T const lower{ *(std::max_element<T *>(pSort, pSort + idx_round_up)) };
                answer = QUANTILE_SPLIT<T, U>(upper, lower);
            }

            // copy the data over from pSort
            pDest[i] = answer;
        }

        WORKSPACE_FREE(pSort);
    }

    //------------------------------
    // median does a sort
    // auto-nan
    // pGroup -> int32_t/64  (V typename)
    static void AccumMedian(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                            void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                            int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        // Alloc
        T * pSort = (T *)WORKSPACE_ALLOC(totalInputRows * sizeof(T));

        LOGGING("Median %llu  %lld  %lld  sizeof: %lld %lld %lld\n", totalInputRows, binLow, binHigh, sizeof(T), sizeof(U),
                sizeof(V));

        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            V index = pFirst[i];
            V nCount = pCount[i];

            if (nCount == 0)
            {
                pDest[i] = GET_INVALID(pDest[i]);
                continue;
            }

            // Copy over the items for this group
            for (V j = 0; j < nCount; j++)
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

            nCount = (V)((pEnd + 1) - pSort);

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
    static void AccumMedianString(void const * pColumn, void const * pGroupT, void const * pFirstT, void const * pCountT,
                                  void * pAccumBin, int64_t binLow, int64_t binHigh, int64_t totalInputRows, int64_t itemSize,
                                  int64_t funcParam)
    {
        T const * const pSrc = (T *)pColumn;
        U * const pDest = (U *)pAccumBin;
        // iGroup, iFirst, and nCount can be int32 or int64
        V const * const pGroup = (V *)pGroupT;
        V const * const pFirst = (V *)pFirstT;
        V const * const pCount = (V *)pCountT;

        // printf("Median string %llu\n", totalInputRows);
        // For all the bins we have to fill
        for (int64_t i = binLow; i < binHigh; i++)
        {
            for (V j = 0; j < itemSize; j++)
            {
                pDest[i * itemSize + j] = 0;
            }
        }
    }

    //-------------------------------------------------------------------------------
    static GROUPBY_X_FUNC GetXFunc2(GB_FUNCTIONS func)
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
        case GB_ROLLING_QUANTILE:
            return AccumRollingQuantile1e9Mult;
        default:
            break;
        }
        return NULL;
    }

    //-------------------------------------------------------------------------------
    static GROUPBY_X_FUNC GetXFunc(GB_FUNCTIONS func)
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
        case GB_QUANTILE_MULT:
            return AccumQuantile1e9Mult;
        default:
            break;
        }
        return NULL;
    }

    static GROUPBY_X_FUNC GetXFuncString(GB_FUNCTIONS func)
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
typedef void (*GROUPBY_GATHER_FUNC)(stGroupBy32 * pstGroupBy32, void const * pDataIn, void * pDataOut, void * pCountOutT,
                                    int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh);

template <typename U, typename W>
static void GatherSum(stGroupBy32 * pstGroupBy32, void const * pDataInT, void * pDataOutT, void * pCountOutBaseT,
                      int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U const * const pDataInBase{ (U *)pDataInT };
    U * const pDataOut{ (U *)pDataOutT };
    W * const pCountOutBase = (W *)pCountOutBaseT;

    // Array indicating if the final answer for a bin will be invalid
    // if one thread saw data and returned invalid, answer is fixed to be invalid.
    // Let's just reuse pCountOut of worker 0 to avoid allocating/freeing more memory.
    W * const pInvFinal{ pCountOutBase };
    // pInvFinal[i] == -1 means the naswer must remain invalid till the end

    U const invalid{ riptide::invalid_for_type<U>::value };

    // TODO: if no data was seen for some bin, answer will be 0 instead of NaN
    memset(pDataOut, 0, sizeof(U) * numUnique);

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U const * const pDataIn{ &pDataInBase[j * numUnique] };
            W const * const pCountOut{ &pCountOutBase[j * numUnique] };

            for (int64_t i = binLow; i < binHigh; i++)
            {
                if (pInvFinal[i] == -1)
                {
                    // since pInvFinal reuses space from worker 0, must always update the output for that worker
                    if (j == 0)
                    {
                        pDataOut[i] = invalid;
                    }
                    continue;
                }
                else if (pCountOut[i] == -1)
                {
                    // a worker saw negiative for this bin
                    pInvFinal[i] = -1;
                    pDataOut[i] = invalid;
                }
                else
                {
                    pDataOut[i] += pDataIn[i];
                }
            }
        }
    }
}

template <typename U, typename W>
static void GatherNanSum(stGroupBy32 * pstGroupBy32, void const * pDataInT, void * pDataOutT, void * pCountOutT, int64_t numUnique,
                         int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U const * const pDataInBase{ (U *)pDataInT };
    U * const pDataOut{ (U *)pDataOutT };

    memset(pDataOut, 0, sizeof(U) * numUnique);

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U const * const pDataIn{ &pDataInBase[j * numUnique] };

            for (int64_t i = binLow; i < binHigh; i++)
            {
                pDataOut[i] += pDataIn[i];
            }
        }
    }
}

template <typename U, typename W>
static void GatherMean(stGroupBy32 * pstGroupBy32, void const * pDataInT, void * pDataOutT, void * pCountOutBaseT,
                       int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U const * const pDataInBase{ (U *)pDataInT };
    U * const pDataOut{ (U *)pDataOutT };
    W * const pCountOutBase{ (W *)pCountOutBaseT };

    W * const pInvFinal{ pCountOutBase };

    U const invalid{ riptide::invalid_for_type<U>::value };

    int64_t allocSize = sizeof(W) * numUnique;
    W * const pTotalCountOut = (W *)WORKSPACE_ALLOC(allocSize);
    memset(pTotalCountOut, 0, allocSize);

    memset(pDataOut, 0, sizeof(U) * numUnique);

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U const * const pDataIn{ &pDataInBase[j * numUnique] };
            W const * const pCountOut{ &pCountOutBase[j * numUnique] };

            for (int64_t i = binLow; i < binHigh; i++)
            {
                if (pInvFinal[i] == -1)
                {
                    // since pInvFinal reuses space from worker 0, set answer for it manually
                    if (j == 0)
                    {
                        pDataOut[i] = invalid;
                    }
                }
                else if (pCountOut[i] == -1)
                {
                    // a worker saw invalid entry for this bin
                    pInvFinal[i] = -1;
                    pDataOut[i] = invalid;
                }
                else
                {
                    pDataOut[i] += pDataIn[i];
                    pTotalCountOut[i] += pCountOut[i];
                }
            }
        }
    }

    // calculate the mean
    for (int64_t i = binLow; i < binHigh; i++)
    {
        if (pTotalCountOut[i] > 0 && riptide::invalid_for_type<U>::is_valid(pDataOut[i]))
        {
            pDataOut[i] = pDataOut[i] / pTotalCountOut[i];
        }
        else
        {
            pDataOut[i] = invalid;
        }
    }

    WORKSPACE_FREE(pTotalCountOut);
}

template <typename U, typename W>
static void GatherNanMean(stGroupBy32 * pstGroupBy32, void const * pDataInT, void * pDataOutT, void * pCountOutBaseT,
                          int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U const * const pDataInBase{ (U *)pDataInT };
    U * const pDataOut{ (U *)pDataOutT };
    W const * const pCountOutBase{ (W *)pCountOutBaseT };

    int64_t allocSize = sizeof(W) * numUnique;
    W * const pCountOut = (W *)WORKSPACE_ALLOC(allocSize);
    memset(pCountOut, 0, allocSize);

    memset(pDataOut, 0, sizeof(U) * numUnique);

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U const * const pDataIn = &pDataInBase[j * numUnique];
            W const * const pCountOutCore = &pCountOutBase[j * numUnique];

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

template <typename U, typename W>
static void GatherNanMin(stGroupBy32 * pstGroupBy32, void const * pDataInT, void * pDataOutT, void * pCountOutBaseT,
                         int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U const * const pDataInBase{ (U *)pDataInT };
    U * const pDataOut{ (U *)pDataOutT };

    // Fill with invalid
    U const invalid{ riptide::invalid_for_type<U>::value };
    for (int64_t i = binLow; i < binHigh; i++)
    {
        pDataOut[i] = invalid;
    }

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U const * const pDataIn{ &pDataInBase[j * numUnique] };

            for (int64_t i = binLow; i < binHigh; i++)
            {
                U const curValue{ pDataOut[i] };
                U const compareValue{ pDataIn[i] };

                if (riptide::invalid_for_type<U>::is_valid(compareValue))
                {
                    if ((not riptide::invalid_for_type<U>::is_valid(curValue)) || (compareValue < curValue))
                    {
                        pDataOut[i] = compareValue;
                    }
                }
            }
        }
    }
}

template <typename U, typename W>
static void GatherMin(stGroupBy32 * pstGroupBy32, void const * pDataInT, void * pDataOutT, void * pCountOutBaseT,
                      int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U const * const pDataInBase{ (U *)pDataInT };
    U * const pDataOut{ (U *)pDataOutT };
    W * const pCountOutBase = (W *)pCountOutBaseT;

    // Array indicating if the final answer for each bin will be invalid
    // if one thread saw data and returned invalid, answer is fixed to be invalid.
    // Let's just reuse pCountOut of worker 0 to avoid allocating/freeing more memory.
    W * const pInvFinal{ pCountOutBase };
    // pInvFinal[i] == -1 means the naswer must remain invalid till the end

    // Fill with invalid
    U const invalid{ riptide::invalid_for_type<U>::value };

    for (int64_t i = binLow; i < binHigh; i++)
    {
        pDataOut[i] = invalid;
    }

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U const * const pDataIn{ &pDataInBase[j * numUnique] };
            W const * const pCountOut{ &pCountOutBase[j * numUnique] };
            // pCountOut[i] = 0 means worker j did not see any value for bin i
            // pCountOut[i] = 1 means worker j saw a value, and did not see any invalid for bin i
            // pCountOut[i] = -1 means worker j saw an invalid value for bin i
            // these are set in AccumMin

            for (int64_t i = binLow; i < binHigh; i++)
            {
                // if pCountOut[i] = 0, this thread never saw any data for this entry
                // or if final answer is already invalid:
                // don't update anything
                if (pCountOut[i] == 0 || pInvFinal[i] == -1)
                {
                    continue;
                }
                // if pCountOut[i] = -1, this worker saw an invalid. Fix answer to invalid
                else if (pCountOut[i] == -1)
                {
                    pInvFinal[i] = -1;
                    pDataOut[i] = invalid;
                }
                else
                {
                    // answer for i is not fixed to invalid, and worker did not see invalid
                    U const curValue{ pDataOut[i] };
                    U const compareValue{ pDataIn[i] };

                    // if curValue is invalid, update it without comparison (first valid data for i)
                    // otherwise none of the two values are invalid, just compare
                    // (compareValue is valid since pCountOut[i] != -1 here)
                    if ((not riptide::invalid_for_type<U>::is_valid(curValue)) || (compareValue < curValue))
                    {
                        pDataOut[i] = compareValue;
                    }
                }
            }
        }
    }
}

template <typename U, typename W>
static void GatherNanMax(stGroupBy32 * pstGroupBy32, void const * pDataInT, void * pDataOutT, void * pCountOutBase,
                         int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U const * const pDataInBase{ (U *)pDataInT };
    U * const pDataOut{ (U *)pDataOutT };

    // Fill with invalid
    U const invalid{ riptide::invalid_for_type<U>::value };
    for (int64_t i = binLow; i < binHigh; i++)
    {
        pDataOut[i] = invalid;
    }

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U const * const pDataIn{ &pDataInBase[j * numUnique] };

            for (int64_t i = binLow; i < binHigh; i++)
            {
                U const curValue{ pDataOut[i] };
                U const compareValue{ pDataIn[i] };

                if (riptide::invalid_for_type<U>::is_valid(compareValue))
                {
                    if ((not riptide::invalid_for_type<U>::is_valid(curValue)) || (compareValue > curValue))
                    {
                        pDataOut[i] = compareValue;
                    }
                }
            }
        }
    }
}

template <typename U, typename W>
static void GatherMax(stGroupBy32 * pstGroupBy32, void const * pDataInT, void * pDataOutT, void * pCountOutBaseT,
                      int64_t numUnique, int64_t numCores, int64_t binLow, int64_t binHigh)
{
    U const * const pDataInBase{ (U *)pDataInT };
    U * const pDataOut{ (U *)pDataOutT };
    W * const pCountOutBase = (W *)pCountOutBaseT;

    // Array indicating if the final answer for each bin will be invalid
    // if one thread saw data and returned invalid, answer is fixed to be invalid.
    // Let's just reuse pCountOut of worker 0 to avoid allocating/freeing more memory.
    W * const pInvFinal{ pCountOutBase };
    // pInvFinal[i] == -1 means the naswer must remain invalid till the end

    // Fill with invalid
    U const invalid{ riptide::invalid_for_type<U>::value };

    for (int64_t i = binLow; i < binHigh; i++)
    {
        pDataOut[i] = invalid;
    }

    // Collect the results from the core
    for (int64_t j = 0; j < numCores; j++)
    {
        if (pstGroupBy32->returnObjects[j].didWork)
        {
            U const * const pDataIn{ &pDataInBase[j * numUnique] };
            W const * const pCountOut{ &pCountOutBase[j * numUnique] };
            // pCountOut[i] = 0 means worker j did not see any value for bin i
            // pCountOut[i] = 1 means worker j saw a value, and did not see any invalid for bin i
            // pCountOut[i] = -1 means worker j saw an invalid value for bin i
            // these are set in AccumMin

            for (int64_t i = binLow; i < binHigh; i++)
            {
                // if pCountOut[i] = 0, this thread never saw any data for this entry
                // or if final answer is already invalid:
                // don't update anything
                if (pCountOut[i] == 0 || pInvFinal[i] == -1)
                {
                    continue;
                }
                // if pCountOut[i] = -1, this worker saw an invalid. Fix answer to invalid
                else if (pCountOut[i] == -1)
                {
                    pInvFinal[i] = -1;
                    pDataOut[i] = invalid;
                }
                else
                {
                    // answer for i is not fixed to invalid, and worker did not see invalid
                    U const curValue{ pDataOut[i] };
                    U const compareValue{ pDataIn[i] };

                    // if curValue is invalid, update it without comparison (first valid data for i)
                    // otherwise none of the two values are invalid, just compare
                    // (compareValue is valid since pCountOut[i] != -1 here)
                    if ((not riptide::invalid_for_type<U>::is_valid(curValue)) || (compareValue > curValue))
                    {
                        pDataOut[i] = compareValue;
                    }
                }
            }
        }
    }
}

template <typename W>
static GROUPBY_GATHER_FUNC GetGroupByGatherFunction(int outputType, GB_FUNCTIONS func)
{
    switch (func)
    {
    case GB_SUM:
        switch (outputType)
        {
        case NPY_BOOL:
            return GatherSum<int64_t, W>;
        case NPY_FLOAT:
            return GatherSum<float, W>;
        case NPY_DOUBLE:
            return GatherSum<double, W>;
        case NPY_LONGDOUBLE:
            return GatherSum<long double, W>;
        case NPY_INT8:
            return GatherSum<int64_t, W>;
        case NPY_INT16:
            return GatherSum<int64_t, W>;
        CASE_NPY_INT32:
            return GatherSum<int64_t, W>;
        CASE_NPY_INT64:
            return GatherSum<int64_t, W>;
        case NPY_UINT8:
            return GatherSum<uint64_t, W>;
        case NPY_UINT16:
            return GatherSum<uint64_t, W>;
        CASE_NPY_UINT32:
            return GatherSum<uint64_t, W>;
        CASE_NPY_UINT64:
            return GatherSum<uint64_t, W>;
        }
        break;
    case GB_NANSUM:
        switch (outputType)
        {
        case NPY_BOOL:
            return GatherNanSum<int64_t, W>;
        case NPY_FLOAT:
            return GatherNanSum<float, W>;
        case NPY_DOUBLE:
            return GatherNanSum<double, W>;
        case NPY_LONGDOUBLE:
            return GatherNanSum<long double, W>;
        case NPY_INT8:
            return GatherNanSum<int64_t, W>;
        case NPY_INT16:
            return GatherNanSum<int64_t, W>;
        CASE_NPY_INT32:
            return GatherNanSum<int64_t, W>;
        CASE_NPY_INT64:
            return GatherNanSum<int64_t, W>;
        case NPY_UINT8:
            return GatherNanSum<uint64_t, W>;
        case NPY_UINT16:
            return GatherNanSum<uint64_t, W>;
        CASE_NPY_UINT32:
            return GatherNanSum<uint64_t, W>;
        CASE_NPY_UINT64:
            return GatherNanSum<uint64_t, W>;
        }
        break;

    case GB_MEAN:
        switch (outputType)
        {
        case NPY_FLOAT:
            return GatherMean<float, W>;
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

            return GatherMean<double, W>;
        }
        break;
    case GB_NANMEAN:
        switch (outputType)
        {
        case NPY_FLOAT:
            return GatherNanMean<float, W>;
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

            return GatherNanMean<double, W>;
        }
        break;

    case GB_MAX:
        switch (outputType)
        {
        case NPY_BOOL:
            return GatherMax<int8_t, W>;
        case NPY_FLOAT:
            return GatherMax<float, W>;
        case NPY_DOUBLE:
            return GatherMax<double, W>;
        case NPY_LONGDOUBLE:
            return GatherMax<long double, W>;
        case NPY_INT8:
            return GatherMax<int8_t, W>;
        case NPY_INT16:
            return GatherMax<int16_t, W>;
        CASE_NPY_INT32:
            return GatherMax<int32_t, W>;
        CASE_NPY_INT64:

            return GatherMax<int64_t, W>;
        case NPY_UINT8:
            return GatherMax<uint8_t, W>;
        case NPY_UINT16:
            return GatherMax<uint16_t, W>;
        CASE_NPY_UINT32:
            return GatherMax<uint32_t, W>;
        CASE_NPY_UINT64:

            return GatherMax<uint64_t, W>;
        }
        break;

    case GB_NANMAX:
        switch (outputType)
        {
        case NPY_BOOL:
            return GatherNanMax<int8_t, W>;
        case NPY_FLOAT:
            return GatherNanMax<float, W>;
        case NPY_DOUBLE:
            return GatherNanMax<double, W>;
        case NPY_LONGDOUBLE:
            return GatherNanMax<long double, W>;
        case NPY_INT8:
            return GatherNanMax<int8_t, W>;
        case NPY_INT16:
            return GatherNanMax<int16_t, W>;
        CASE_NPY_INT32:
            return GatherNanMax<int32_t, W>;
        CASE_NPY_INT64:

            return GatherNanMax<int64_t, W>;
        case NPY_UINT8:
            return GatherNanMax<uint8_t, W>;
        case NPY_UINT16:
            return GatherNanMax<uint16_t, W>;
        CASE_NPY_UINT32:
            return GatherNanMax<uint32_t, W>;
        CASE_NPY_UINT64:

            return GatherNanMax<uint64_t, W>;
        }
        break;

    case GB_MIN:
        switch (outputType)
        {
        case NPY_BOOL:
            return GatherMin<int8_t, W>;
        case NPY_FLOAT:
            return GatherMin<float, W>;
        case NPY_DOUBLE:
            return GatherMin<double, W>;
        case NPY_LONGDOUBLE:
            return GatherMin<long double, W>;
        case NPY_INT8:
            return GatherMin<int8_t, W>;
        case NPY_INT16:
            return GatherMin<int16_t, W>;
        CASE_NPY_INT32:
            return GatherMin<int32_t, W>;
        CASE_NPY_INT64:

            return GatherMin<int64_t, W>;
        case NPY_UINT8:
            return GatherMin<uint8_t, W>;
        case NPY_UINT16:
            return GatherMin<uint16_t, W>;
        CASE_NPY_UINT32:
            return GatherMin<uint32_t, W>;
        CASE_NPY_UINT64:

            return GatherMin<uint64_t, W>;
        }
        break;

    case GB_NANMIN:
        switch (outputType)
        {
        case NPY_BOOL:
            return GatherNanMin<int8_t, W>;
        case NPY_FLOAT:
            return GatherNanMin<float, W>;
        case NPY_DOUBLE:
            return GatherNanMin<double, W>;
        case NPY_LONGDOUBLE:
            return GatherNanMin<long double, W>;
        case NPY_INT8:
            return GatherNanMin<int8_t, W>;
        case NPY_INT16:
            return GatherNanMin<int16_t, W>;
        CASE_NPY_INT32:
            return GatherNanMin<int32_t, W>;
        CASE_NPY_INT64:

            return GatherNanMin<int64_t, W>;
        case NPY_UINT8:
            return GatherNanMin<uint8_t, W>;
        case NPY_UINT16:
            return GatherNanMin<uint16_t, W>;
        CASE_NPY_UINT32:
            return GatherNanMin<uint32_t, W>;
        CASE_NPY_UINT64:

            return GatherNanMin<uint64_t, W>;
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
// W is the pCountOut type, either int32_t or int64_t
template <typename V, typename W>
static GROUPBY_TWO_FUNC GetGroupByFunction(bool * hasCounts, int32_t * wantedOutputType, int32_t * wantedTempType, int inputType,
                                           GB_FUNCTIONS func)
{
    *hasCounts = false;
    *wantedTempType = -1;
    static_assert(std::is_signed<V>::value, "Array types must be signed");
    switch (func)
    {
    case GB_SUM:
        // used for checking if NaN was encountered
        // TODO: for sum, min, max, nanmin, nanmax, can use int8_t? Since only store 0 and +-1 there
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int8_t, int64_t, V, W>::AccumSum;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            *wantedTempType = NPY_DOUBLE;
            return GroupByBase<float, float, V, W>::AccumSumFloat;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumSum;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V, W>::AccumSum;
        case NPY_INT8:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int8_t, int64_t, V, W>::AccumSum;
        case NPY_INT16:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int16_t, int64_t, V, W>::AccumSum;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int32_t, int64_t, V, W>::AccumSum;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V, W>::AccumSum;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint8_t, uint64_t, V, W>::AccumSum;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint16_t, uint64_t, V, W>::AccumSum;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint32_t, uint64_t, V, W>::AccumSum;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V, W>::AccumSum;
        default:
            break;
        }

    case GB_NANSUM:
        switch (inputType)
        {
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            *wantedTempType = NPY_DOUBLE;
            return GroupByBase<float, float, V, W>::AccumNanSumFloat;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumNanSum;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V, W>::AccumNanSum;
        // bool has no invalid
        case NPY_BOOL:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int8_t, int64_t, V, W>::AccumNanSum;
        case NPY_INT8:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int8_t, int64_t, V, W>::AccumNanSum;
        case NPY_INT16:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int16_t, int64_t, V, W>::AccumNanSum;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int32_t, int64_t, V, W>::AccumNanSum;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V, W>::AccumNanSum;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint8_t, uint64_t, V, W>::AccumNanSum;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint16_t, uint64_t, V, W>::AccumNanSum;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint32_t, uint64_t, V, W>::AccumNanSum;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V, W>::AccumNanSum;
        default:
            break;
        }

    case GB_MIN:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_BOOL;
            return GroupByBase<int8_t, int8_t, V, W>::AccumMin;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V, W>::AccumMin;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumMin;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V, W>::AccumMin;
        case NPY_INT8:
            *wantedOutputType = NPY_INT8;
            return GroupByBase<int8_t, int8_t, V, W>::AccumMin;
        case NPY_INT16:
            *wantedOutputType = NPY_INT16;
            return GroupByBase<int16_t, int16_t, V, W>::AccumMin;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT32;
            return GroupByBase<int32_t, int32_t, V, W>::AccumMin;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V, W>::AccumMin;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT8;
            return GroupByBase<uint8_t, uint8_t, V, W>::AccumMin;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT16;
            return GroupByBase<uint16_t, uint16_t, V, W>::AccumMin;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT32;
            return GroupByBase<uint32_t, uint32_t, V, W>::AccumMin;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V, W>::AccumMin;
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
            return GroupByBase<float, float, V, W>::AccumNanMin;
        case NPY_DOUBLE:
            *hasCounts = false;
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumNanMin;
        case NPY_LONGDOUBLE:
            *hasCounts = false;
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V, W>::AccumNanMin;
        case NPY_BOOL:
            *wantedOutputType = NPY_BOOL;
            return GroupByBase<int8_t, int8_t, V, W>::AccumMin;
        case NPY_INT8:
            *wantedOutputType = NPY_INT8;
            return GroupByBase<int8_t, int8_t, V, W>::AccumNanMin;
        case NPY_INT16:
            *wantedOutputType = NPY_INT16;
            return GroupByBase<int16_t, int16_t, V, W>::AccumNanMin;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT32;
            return GroupByBase<int32_t, int32_t, V, W>::AccumNanMin;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V, W>::AccumNanMin;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT8;
            return GroupByBase<uint8_t, uint8_t, V, W>::AccumNanMin;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT16;
            return GroupByBase<uint16_t, uint16_t, V, W>::AccumNanMin;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT32;
            return GroupByBase<uint32_t, uint32_t, V, W>::AccumNanMin;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V, W>::AccumNanMin;
        default:
            break;
        }

    case GB_MAX:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_BOOL;
            return GroupByBase<int8_t, int8_t, V, W>::AccumMax;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V, W>::AccumMax;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumMax;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V, W>::AccumMax;
        case NPY_INT8:
            *wantedOutputType = NPY_INT8;
            return GroupByBase<int8_t, int8_t, V, W>::AccumMax;
        case NPY_INT16:
            *wantedOutputType = NPY_INT16;
            return GroupByBase<int16_t, int16_t, V, W>::AccumMax;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT32;
            return GroupByBase<int32_t, int32_t, V, W>::AccumMax;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V, W>::AccumMax;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT8;
            return GroupByBase<uint8_t, uint8_t, V, W>::AccumMax;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT16;
            return GroupByBase<uint16_t, uint16_t, V, W>::AccumMax;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT32;
            return GroupByBase<uint32_t, uint32_t, V, W>::AccumMax;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V, W>::AccumMax;
        default:
            break;
        }

    case GB_NANMAX:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V, W>::AccumNanMax;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumNanMax;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_LONGDOUBLE;
            return GroupByBase<long double, long double, V, W>::AccumNanMax;
        case NPY_BOOL:
            *wantedOutputType = NPY_BOOL;
            return GroupByBase<int8_t, int8_t, V, W>::AccumMax;
        case NPY_INT8:
            *wantedOutputType = NPY_INT8;
            return GroupByBase<int8_t, int8_t, V, W>::AccumNanMax;
        case NPY_INT16:
            *wantedOutputType = NPY_INT16;
            return GroupByBase<int16_t, int16_t, V, W>::AccumNanMax;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_INT32;
            return GroupByBase<int32_t, int32_t, V, W>::AccumNanMax;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_INT64;
            return GroupByBase<int64_t, int64_t, V, W>::AccumNanMax;
        case NPY_UINT8:
            *wantedOutputType = NPY_UINT8;
            return GroupByBase<uint8_t, uint8_t, V, W>::AccumNanMax;
        case NPY_UINT16:
            *wantedOutputType = NPY_UINT16;
            return GroupByBase<uint16_t, uint16_t, V, W>::AccumNanMax;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_UINT32;
            return GroupByBase<uint32_t, uint32_t, V, W>::AccumNanMax;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_UINT64;
            return GroupByBase<uint64_t, uint64_t, V, W>::AccumNanMax;
        default:
            break;
        }

    case GB_MEAN:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumMean;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V, W>::AccumMeanFloat;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumMean;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V, W>::AccumMean;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumMean;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V, W>::AccumMean;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V, W>::AccumMean;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V, W>::AccumMean;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V, W>::AccumMean;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V, W>::AccumMean;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V, W>::AccumMean;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V, W>::AccumMean;
        default:
            break;
        }

    case GB_NANMEAN:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumNanMean;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V, W>::AccumNanMeanFloat;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumNanMean;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V, W>::AccumNanMean;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumNanMean;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V, W>::AccumNanMean;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V, W>::AccumNanMean;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V, W>::AccumNanMean;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V, W>::AccumNanMean;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V, W>::AccumNanMean;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V, W>::AccumNanMean;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V, W>::AccumNanMean;
        default:
            break;
        }

    case GB_VAR:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumVar;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V, W>::AccumVar;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumVar;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V, W>::AccumVar;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumVar;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V, W>::AccumVar;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V, W>::AccumVar;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V, W>::AccumVar;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V, W>::AccumVar;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V, W>::AccumVar;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V, W>::AccumVar;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V, W>::AccumVar;
        default:
            break;
        }

    case GB_NANVAR:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumNanVar;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V, W>::AccumNanVar;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumNanVar;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V, W>::AccumNanVar;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumNanVar;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V, W>::AccumNanVar;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V, W>::AccumNanVar;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V, W>::AccumNanVar;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V, W>::AccumNanVar;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V, W>::AccumNanVar;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V, W>::AccumNanVar;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V, W>::AccumNanVar;
        default:
            break;
        }

    case GB_STD:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumStd;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V, W>::AccumStd;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumStd;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V, W>::AccumStd;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumStd;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V, W>::AccumStd;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V, W>::AccumStd;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V, W>::AccumStd;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V, W>::AccumStd;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V, W>::AccumStd;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V, W>::AccumStd;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V, W>::AccumStd;
        default:
            break;
        }

    case GB_NANSTD:
        *hasCounts = true;
        switch (inputType)
        {
        case NPY_BOOL:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumNanStd;
        case NPY_FLOAT:
            *wantedOutputType = NPY_FLOAT;
            return GroupByBase<float, float, V, W>::AccumNanStd;
        case NPY_DOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<double, double, V, W>::AccumNanStd;
        case NPY_LONGDOUBLE:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<long double, double, V, W>::AccumNanStd;
        case NPY_INT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int8_t, double, V, W>::AccumNanStd;
        case NPY_INT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int16_t, double, V, W>::AccumNanStd;
        CASE_NPY_INT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int32_t, double, V, W>::AccumNanStd;
        CASE_NPY_INT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<int64_t, double, V, W>::AccumNanStd;
        case NPY_UINT8:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint8_t, double, V, W>::AccumNanStd;
        case NPY_UINT16:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint16_t, double, V, W>::AccumNanStd;
        CASE_NPY_UINT32:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint32_t, double, V, W>::AccumNanStd;
        CASE_NPY_UINT64:
            *wantedOutputType = NPY_DOUBLE;
            return GroupByBase<uint64_t, double, V, W>::AccumNanStd;
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
//      //   case NPY_BOOL:   return GroupByBase<T, bool, W>::GetFunc(func);
//   case NPY_FLOAT:  return GroupByBase<T, float, W>::GetXFunc(func);
//   case NPY_DOUBLE: return GroupByBase<T, double, W>::GetXFunc(func);
//      //   case NPY_BYTE:   return GroupByBase<T, int8_t, W>::GetFunc(func);
//      //   case NPY_INT16:  return GroupByBase<T, int16_t, W>::GetFunc(func);
//   case NPY_INT:    return GroupByBase<T, int32_t, W>::GetXFunc(func);
//   CASE_NPY_INT32:  return GroupByBase<T, int32_t, W>::GetXFunc(func);
//   CASE_NPY_INT64:  return GroupByBase<T, int64_t, W>::GetXFunc(func);
//      //   case NPY_UBYTE:  return GroupByBase<T, uint8_t, W>::GetFunc(func);
//      //   case NPY_UINT16: return GroupByBase<T, uint16_t, W>::GetFunc(func);
//   case NPY_UINT:   return GroupByBase<T, uint32_t, W>::GetXFunc(func);
//   CASE_NPY_UINT32: return GroupByBase<T, uint32_t, W>::GetXFunc(func);
//   CASE_NPY_UINT64: return GroupByBase<T, uint64_t, W>::GetXFunc(func);
//   }
//   return NULL;
//
//}

template <typename V>
static GROUPBY_X_FUNC GetGroupByXFunction(int inputType, int outputType, GB_FUNCTIONS func)
{
    LOGGING("GBX Func is %d  inputtype: %d  outputtype: %d\n", func, inputType, outputType);
    static_assert(std::is_signed<V>::value, "Array types must be signed");

    if (func == GB_TRIMBR)
    {
        switch (inputType)
        {
        case NPY_BOOL:
            return GroupByBase<bool, float, V>::AccumTrimMeanBR;
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
    else if (func == GB_QUANTILE_MULT)
    {
        switch (inputType)
        {
        case NPY_BOOL:
            return GroupByBase<bool, bool, V>::GetXFunc(func);
        case NPY_FLOAT:
            return GroupByBase<float, float, V>::GetXFunc(func);
        case NPY_DOUBLE:
            return GroupByBase<double, double, V>::GetXFunc(func);
        case NPY_LONGDOUBLE:
            return GroupByBase<long double, long double, V>::GetXFunc(func);
        case NPY_INT8:
            return GroupByBase<int8_t, double, V>::GetXFunc(func);
        case NPY_INT16:
            return GroupByBase<int16_t, double, V>::GetXFunc(func);
        CASE_NPY_INT32:
            return GroupByBase<int32_t, double, V>::GetXFunc(func);
        CASE_NPY_INT64:
            return GroupByBase<int64_t, double, V>::GetXFunc(func);
        case NPY_UINT8:
            return GroupByBase<uint8_t, double, V>::GetXFunc(func);
        case NPY_UINT16:
            return GroupByBase<uint16_t, double, V>::GetXFunc(func);
        CASE_NPY_UINT32:
            return GroupByBase<uint32_t, double, V>::GetXFunc(func);
        CASE_NPY_UINT64:
            return GroupByBase<uint64_t, double, V>::GetXFunc(func);
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
    else if (func == GB_ROLLING_DIFF)
    {
        LOGGING("Rolling+diff called with type %d\n", inputType);
        switch (inputType)
        {
        // case NPY_BOOL:
        //         return GroupByBase<bool, int64_t, V>::GetXFunc2(func);
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
    else if (func == GB_ROLLING_SHIFT)
    {
        LOGGING("Rolling shift called with type %d\n", inputType);
        switch (inputType)
        {
        case NPY_BOOL:
            return GroupByBase<bool, bool, V>::AccumRollingShift;
        case NPY_FLOAT:
            return GroupByBase<float, float, V>::AccumRollingShift;
        case NPY_DOUBLE:
            return GroupByBase<double, double, V>::AccumRollingShift;
        case NPY_LONGDOUBLE:
            return GroupByBase<long double, long double, V>::AccumRollingShift;
        case NPY_INT8:
            return GroupByBase<int8_t, int8_t, V>::AccumRollingShift;
        case NPY_INT16:
            return GroupByBase<int16_t, int16_t, V>::AccumRollingShift;
        CASE_NPY_INT32:
            return GroupByBase<int32_t, int32_t, V>::AccumRollingShift;
        CASE_NPY_INT64:
            return GroupByBase<int64_t, int64_t, V>::AccumRollingShift;
        case NPY_UINT8:
            return GroupByBase<uint8_t, uint8_t, V>::AccumRollingShift;
        case NPY_UINT16:
            return GroupByBase<uint16_t, uint16_t, V>::AccumRollingShift;
        CASE_NPY_UINT32:
            return GroupByBase<uint32_t, uint32_t, V>::AccumRollingShift;
        CASE_NPY_UINT64:
            return GroupByBase<uint64_t, uint64_t, V>::AccumRollingShift;
        case NPY_STRING:
        case NPY_UNICODE:
        case NPY_VOID:
            return GroupByBase<flexible_t, flexible_t, V>::AccumRollingShift;
        }
        return NULL;
    }
    else if (func >= GB_ROLLING_SUM)
    {
        if (func == GB_ROLLING_MEAN || func == GB_ROLLING_NANMEAN || func == GB_ROLLING_QUANTILE)
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

        GROUPBY_X_FUNC pFunctionX = pGroupBy32->returnObjects[0].pFunctionX;
        void * pDataOut = pGroupBy32->returnObjects[0].pOutArray;

        // First index is 1 so we subtract
        index--;

        LOGGING("|%d %d %lld %p %p", core, (int)workBlock, index, pDataIn, pDataOut);
        int64_t binLow = pGroupBy32->returnObjects[index].binLow;
        int64_t binHigh = pGroupBy32->returnObjects[index].binHigh;

        pFunctionX((void *)pDataIn, (void *)pGroupBy32->pGroup, (void *)pGroupBy32->pFirst, (void *)pGroupBy32->pCount,
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
    void * pCountOut = pGroupBy32->returnObjects[core].pCountOut;
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
    void * pCountOut = pGroupBy32->returnObjects[i].pCountOut;
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
            pFunction(pDataIn, pDataIn2 /* USE IKEY which can be int8/16/32/64*/, pCountOut, pDataOut, len, binLow, binHigh, -1,
                      pDataTmp);
        }
        else

            if (typeCall == ANY_GROUPBY_XFUNC32)
        {
            // Accum the calculation
            GROUPBY_X_FUNC pFunctionX = pGroupBy32->returnObjects[i].pFunctionX;

            int32_t funcNum = pGroupBy32->returnObjects[i].funcNum;

            // static void AccumLast(void* pColumn, void* pGroupT, int32_t* pFirst,
            // int32_t* pCount, void* pAccumBin, int64_t numUnique, int64_t
            // totalInputRows, int64_t itemSize, int64_t funcParam) {
            if (funcNum < GB_FIRST)
                printf("!!! internal bug in GroupByCall -- %d\n", funcNum);

            if (pFunctionX)
            {
                pFunctionX((void *)pDataIn, (void *)pGroupBy32->pGroup /* GROUP might be int32 or int64*/,
                           (void *)pGroupBy32->pFirst, (void *)pGroupBy32->pCount, (char *)pDataOut, binLow, binHigh,
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
template <typename W>
GROUPBY_TWO_FUNC GetGroupByFunctionStep1(int32_t iKeyType, bool * hasCounts, int32_t * numpyOutType, int32_t * numpyTmpType,
                                         int32_t numpyInType, GB_FUNCTIONS funcNum)
{
    GROUPBY_TWO_FUNC pFunction = NULL;

    switch (iKeyType)
    {
    case NPY_INT8:
        pFunction = GetGroupByFunction<int8_t, W>(hasCounts, numpyOutType, numpyTmpType, numpyInType, funcNum);
        break;
    case NPY_INT16:
        pFunction = GetGroupByFunction<int16_t, W>(hasCounts, numpyOutType, numpyTmpType, numpyInType, funcNum);
        break;
    CASE_NPY_INT32:
        pFunction = GetGroupByFunction<int32_t, W>(hasCounts, numpyOutType, numpyTmpType, numpyInType, funcNum);
        break;
    CASE_NPY_INT64:
        pFunction = GetGroupByFunction<int64_t, W>(hasCounts, numpyOutType, numpyTmpType, numpyInType, funcNum);
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
                                     int64_t binLow, int64_t binHigh, int64_t funcParam)
{
    PyObject * returnTuple = NULL;
    int32_t iGroupType = PyArray_TYPE(iGroup);

    int32_t numpyOutType = aInfo[0].NumpyDType;
    bool hasCounts = false;

    LOGGING("In banded groupby %d\n", (int)firstFuncNum);

    GROUPBY_X_FUNC pFunction = NULL;

    switch (iGroupType)
    {
    CASE_NPY_INT32:
        pFunction = GetGroupByXFunction<int32_t>(numpyOutType, numpyOutType, firstFuncNum);
        break;
    CASE_NPY_INT64:
        pFunction = GetGroupByXFunction<int64_t>(numpyOutType, numpyOutType, firstFuncNum);
        break;
    }

    if ((firstFuncNum == GB_TRIMBR) || (firstFuncNum == GB_QUANTILE_MULT))
    {
        numpyOutType = NPY_FLOAT64;
        if (aInfo[0].NumpyDType == NPY_FLOAT32)
        {
            numpyOutType = NPY_FLOAT32;
        }
        else if (aInfo[0].NumpyDType == NPY_BOOL && firstFuncNum != GB_TRIMBR)
        {
            numpyOutType = NPY_BOOL;
        }
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

            //// Allocate room for all the threads to participate, this will be
            /// gathered later
            // char* pWorkspace = (char*)WORKSPACE_ALLOC(unique_rows * itemSize *
            // numCores); LOGGING("***workspace %p   unique:%lld   itemsize:%lld
            // cores:%d\n", pWorkspace, unique_rows, itemSize, cores);

            // if (pWorkspace == NULL) {
            //   return NULL;
            //}

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
            pstGroupBy32->funcParam = funcParam;

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

                pstGroupBy32->returnObjects[i].funcNum = firstFuncNum;
                pstGroupBy32->returnObjects[i].didWork = 0;

                // Assign working memory per call
                pstGroupBy32->returnObjects[i].pOutArray = pOutArray;
                pstGroupBy32->returnObjects[i].pFunctionX = pFunction;
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
    int64_t arraySizeKey = ArrayLength(iKey);

    int32_t numpyOutType = -1;
    int32_t numpyTmpType = -1;
    bool hasCounts = false;

    int64_t pCountOutTypeSize;
    GROUPBY_TWO_FUNC pFunction;

    if (arraySizeKey < riptide::int32_index_cutoff)
    {
        pCountOutTypeSize = sizeof(int32_t);
        pFunction = GetGroupByFunctionStep1<int32_t>(iKeyType, &hasCounts, &numpyOutType, &numpyTmpType, aInfo[0].NumpyDType,
                                                     firstFuncNum);
    }
    else
    {
        pCountOutTypeSize = sizeof(int64_t);
        pFunction = GetGroupByFunctionStep1<int64_t>(iKeyType, &hasCounts, &numpyOutType, &numpyTmpType, aInfo[0].NumpyDType,
                                                     firstFuncNum);
    }

    // printf("Taking the divide path  %lld \n", unique_rows);

    if (pFunction && numpyOutType != -1)
    {
        void * pDataIn2 = PyArray_BYTES(iKey);

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
            void * pCountOut = NULL;

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
                int64_t allocSize = pCountOutTypeSize * unique_rows * numCores;

                workspaceMemList.emplace_back(WORKSPACE_ALLOC(allocSize));
                pCountOut = (void *)workspaceMemList.back().get();
                if (pCountOut == NULL)
                {
                    return NULL;
                }
                memset(pCountOut, 0, allocSize);
                LOGGING("***pCountOut %p   unique:%lld  allocsize:%lld   cores:%d\n", pCountOut, unique_rows, allocSize, numCores);
            }

            void * pTmpWorkspace{ nullptr };
            int32_t tempItemSize{ 0 };

            if (numpyTmpType >= 0)
            {
                tempItemSize = NpyToSize(numpyTmpType);
                int64_t tempSize = tempItemSize * (binHigh - binLow) * numCores;
                workspaceMemList.emplace_back(WORKSPACE_ALLOC(tempSize));
                pTmpWorkspace = workspaceMemList.back().get();
                if (! pTmpWorkspace)
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
                pstGroupBy32->returnObjects[i].funcNum = firstFuncNum;
                pstGroupBy32->returnObjects[i].binLow = binLow;
                pstGroupBy32->returnObjects[i].binHigh = binHigh;

                pstGroupBy32->returnObjects[i].didWork = 0;

                // Assign working memory per call
                pstGroupBy32->returnObjects[i].pOutArray = &pWorkspace[unique_rows * i * itemSize];
                pstGroupBy32->returnObjects[i].pTmpArray =
                    pTmpWorkspace ? &(static_cast<char *>(pTmpWorkspace)[(binHigh - binLow) * i * tempItemSize]) : nullptr;

                pstGroupBy32->returnObjects[i].pCountOut =
                    pCountOut ? (void *)&(static_cast<char *>(pCountOut)[(unique_rows * i) * pCountOutTypeSize]) : nullptr;

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
            GROUPBY_GATHER_FUNC pGather;
            if (pCountOutTypeSize == sizeof(int32_t))
            {
                pGather = GetGroupByGatherFunction<int32_t>(numpyOutType, firstFuncNum);
            }
            else
            {
                pGather = GetGroupByGatherFunction<int64_t>(numpyOutType, firstFuncNum);
            }

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
    GB_FUNCTIONS const firstFuncNum{ static_cast<GB_FUNCTIONS>(
        PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listFuncNum, 0), &overflow)) };
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
            returnTuple = GroupBySingleOpMultithreaded(aInfo, iKey, firstFuncNum, unique_rows, tupleSize, binLow, binHigh);
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
            GB_FUNCTIONS const funcNum{ static_cast<GB_FUNCTIONS>(
                PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listFuncNum, i), &overflow)) };
            pstGroupBy32->returnObjects[i].funcNum = funcNum;

            int64_t binLow = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinLow, i), &overflow);
            pstGroupBy32->returnObjects[i].binLow = binLow;

            int64_t binHigh = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinHigh, i), &overflow);
            pstGroupBy32->returnObjects[i].binHigh = binHigh;

            int64_t pCountOutTypeSize;
            GROUPBY_TWO_FUNC pFunction;

            if (arraySizeKey < riptide::int32_index_cutoff)
            {
                pCountOutTypeSize = sizeof(int32_t);
                pFunction = GetGroupByFunctionStep1<int32_t>(iKeyType, &hasCounts, &numpyOutType, &numpyTmpType,
                                                             aInfo[i].NumpyDType, funcNum);
            }
            else
            {
                pCountOutTypeSize = sizeof(int64_t);
                pFunction = GetGroupByFunctionStep1<int64_t>(iKeyType, &hasCounts, &numpyOutType, &numpyTmpType,
                                                             aInfo[i].NumpyDType, funcNum);
            }

            PyArrayObject * outArray = NULL;
            void * pCountOut = NULL;
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
                    int64_t allocSize = pCountOutTypeSize * unique_rows;
                    workspaceMemList.emplace_back(WORKSPACE_ALLOC(allocSize));
                    pCountOut = (void *)workspaceMemList.back().get();
                    if (pCountOut == NULL)
                    {
                        return NULL;
                    }
                    memset(pCountOut, 0, allocSize);
                }

                if (numpyTmpType >= 0)
                {
                    int32_t tempItemSize = NpyToSize(numpyTmpType);
                    workspaceMemList.emplace_back(WORKSPACE_ALLOC(tempItemSize * (binHigh - binLow)));
                    pTmpArray = workspaceMemList.back().get();
                    if (! pTmpArray)
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

            void * pCountOut = pstGroupBy32->returnObjects[i].pCountOut;
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

    int32_t iGroupType = PyArray_TYPE(iGroup);
    int32_t iFirstType = PyArray_TYPE(iFirst);
    int32_t nCountType = PyArray_TYPE(nCount);

    if ((iGroupType != iFirstType) || (iGroupType != nCountType))
    {
        PyErr_Format(PyExc_ValueError, "GroupByAllPack32 iGroup, iFirstGroup, and nCountGroup params must be of same type.");
        return NULL;
    }

    // Valid types for iGroup, iFirst, and nCount
    switch (iGroupType)
    {
    CASE_NPY_INT32:
    CASE_NPY_INT64:

        break;
    default:
        PyErr_Format(PyExc_ValueError, "GroupByAllPack32 iGroup, iFirstGroup, and nCountGroup params must be int32 or int64");
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

    int64_t funcTupleSize = PyList_GET_SIZE(listFuncNum);

    if (tupleSize != funcTupleSize)
    {
        PyErr_Format(PyExc_ValueError, "GroupByAllPack32 func numbers do not match array columns %lld %lld", tupleSize,
                     funcTupleSize);
        return NULL;
    }

    int64_t binTupleSize = PyList_GET_SIZE(listBinLow);

    if (tupleSize != binTupleSize)
    {
        PyErr_Format(PyExc_ValueError, "GroupByAllPack32 bin numbers do not match array columns %lld %lld", tupleSize,
                     binTupleSize);
        return NULL;
    }

    void * pDataIn2 = PyArray_BYTES(iKey);
    // New reference
    PyObject * returnTuple = NULL;
    int64_t arraySizeKey = ArrayLength(iKey);

    if (aInfo->ArrayLength != arraySizeKey)
    {
        PyErr_Format(PyExc_ValueError, "GroupByAllPack32 iKey length does not match value length %lld %lld", aInfo->ArrayLength,
                     arraySizeKey);
        return NULL;
    }

    if (tupleSize == 1 && arraySizeKey > 65536)
    {
        int overflow = 0;
        int64_t binLow = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinLow, 0), &overflow);
        int64_t binHigh = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinHigh, 0), &overflow);

        GB_FUNCTIONS const firstFuncNum{ static_cast<GB_FUNCTIONS>(
            PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listFuncNum, 0), &overflow)) };

        LOGGING("Checking banded %lld\n", firstFuncNum);

        if ((firstFuncNum >= GB_MEDIAN && firstFuncNum <= GB_TRIMBR) || (firstFuncNum == GB_QUANTILE_MULT))
        {
            returnTuple = GroupBySingleOpMultiBands(aInfo, iKey, iFirst, iGroup, nCount, firstFuncNum, unique_rows, tupleSize,
                                                    binLow, binHigh, funcParam);
        }
    }

    if (returnTuple == NULL)
    {
        int64_t funcTupleSize = PyList_GET_SIZE(listFuncNum);

        if (tupleSize != funcTupleSize)
        {
            PyErr_Format(PyExc_ValueError, "GroupByAllPack32 func numbers do not match array columns %lld %lld", tupleSize,
                         funcTupleSize);
        }

        int64_t binTupleSize = PyList_GET_SIZE(listBinLow);

        if (tupleSize != binTupleSize)
        {
            PyErr_Format(PyExc_ValueError, "GroupByAllPack32 bin numbers do not match array columns %lld %lld", tupleSize,
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
        LOGGING("in GroupByAllPack32 allocating %lld\n", allocSize);

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
            GB_FUNCTIONS const funcNum{ static_cast<GB_FUNCTIONS>(
                PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listFuncNum, i), &overflow)) };
            pstGroupBy32->returnObjects[i].funcNum = funcNum;

            int64_t binLow = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinLow, i), &overflow);
            pstGroupBy32->returnObjects[i].binLow = binLow;

            int64_t binHigh = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(listBinHigh, i), &overflow);
            pstGroupBy32->returnObjects[i].binHigh = binHigh;

            numpyOutType = aInfo[i].NumpyDType;

            GROUPBY_X_FUNC pFunction = NULL;

            switch (iGroupType)
            {
            CASE_NPY_INT32:
                pFunction = GetGroupByXFunction<int32_t>(numpyOutType, numpyOutType, funcNum);
                break;

            CASE_NPY_INT64:
                pFunction = GetGroupByXFunction<int64_t>(numpyOutType, numpyOutType, funcNum);
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
                    // Everything is a float64 unless it is already a float32 or bool, then we
                    // keep it as float32
                    switch (aInfo[i].NumpyDType)
                    {
                    case NPY_BOOL:
                    case NPY_FLOAT32:
                        numpyOutType = NPY_FLOAT32;
                        break;

                    default:
                        numpyOutType = NPY_FLOAT64;
                        break;
                    }
                    outArray = AllocateNumpyArray(1, (npy_intp *)&unique_rows, numpyOutType);
                    CHECK_MEMORY_ERROR(outArray);
                }
                // almost same for quantiles, but check bool case
                // string will need to be different when they are supported
                else if (funcNum == GB_QUANTILE_MULT)
                {
                    // Variance must be in float form
                    numpyOutType = NPY_FLOAT64;

                    // Everything is a float64 unless it is already a float32, then we
                    // keep it as float32
                    if (aInfo[i].NumpyDType == NPY_FLOAT32)
                    {
                        numpyOutType = NPY_FLOAT32;
                    }
                    else if (aInfo[i].NumpyDType == NPY_BOOL)
                    {
                        numpyOutType = NPY_BOOL;
                    }
                    else if (aInfo[i].NumpyDType == NPY_LONGDOUBLE)
                    {
                        numpyOutType = NPY_LONGDOUBLE;
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
                        else if (funcNum == GB_ROLLING_MEAN || funcNum == GB_ROLLING_NANMEAN || funcNum == GB_ROLLING_QUANTILE)
                        {
                            numpyOutType = NPY_FLOAT64;
                        }

                        if (aInfo[i].ArrayLength != pstGroupBy32->totalInputRows)
                        {
                            PyErr_Format(PyExc_ValueError,
                                         "GroupByAllPack32 for rolling functions, input size "
                                         "must be same size as group size: %lld vs %lld",
                                         aInfo[i].ArrayLength, pstGroupBy32->totalInputRows);
                            goto ERROR_EXIT;
                        }

                        outArray = AllocateLikeNumpyArray(aInfo[i].pObject, numpyOutType);
                    }
                    else
                    {
                        LOGGING("GroupByAllPack32:  Allocating for output type: %d\n", aInfo[i].NumpyDType);
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
                PyErr_Format(PyExc_NotImplementedError, "GroupByAllPack32 doesn't implement function %llu for type %d", funcNum,
                             numpyOutType);
                goto ERROR_EXIT;
            }

            pstGroupBy32->returnObjects[i].outArray = outArray;
            pstGroupBy32->returnObjects[i].pFunctionX = pFunction;
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
        //   GROUPBY_X_FUNC  pFunction =
        //   GetGroupByXFunction(aInfo[i].NumpyDType, numpyOutType,
        //   funcNum);

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
