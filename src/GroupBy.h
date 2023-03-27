#pragma once

#include "CommonInc.h"
#include "MultiKey.h"

enum GB_FUNCTIONS : int32_t
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

    GB_MEDIAN = 103,        // auto handles nan
    GB_MODE = 104,          // auto handles nan
    GB_TRIMBR = 105,        // auto handles nan
    GB_QUANTILE_MULT = 106, // handles all (nan)median/quantile versions

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
    GB_ROLLING_QUANTILE = 207,
};

// PyObject *
// GroupByOp32(PyObject *self, PyObject *args);

// Basic call for sum, average
PyObject * GroupByAll32(PyObject * self, PyObject * args);

PyObject * GroupByAll64(PyObject * self, PyObject * args);

// More advanced call to calculate median, nth, last
PyObject * GroupByAllPack32(PyObject * self, PyObject * args);

struct stGroupByReturn
{
    union
    {
        PyArrayObject * outArray;
        int64_t didWork;
    };

    // for multithreaded sum this is set
    void * pOutArray;
    void * pTmpArray;

    void * pCountOut;

    int32_t numpyOutType;
    GB_FUNCTIONS funcNum;

    int64_t binLow;
    int64_t binHigh;

    union
    {
        GROUPBY_TWO_FUNC pFunction;
        GROUPBY_X_FUNC pFunctionX;
    };

    PyObject * returnObject;
};

struct stGroupBy32
{
    ArrayInfo * aInfo;
    int64_t tupleSize;
    /*  int32_t    numpyOutType;
      int32_t    reserved;
    */
    void * pDataIn2;
    int64_t itemSize2;

    int64_t uniqueRows;
    int64_t totalInputRows;

    TYPE_OF_FUNCTION_CALL typeOfFunctionCall;
    int64_t funcParam;
    void * pKey;
    void * pGroup;
    void * pFirst;
    void * pCount;

    stGroupByReturn returnObjects[1];
};
