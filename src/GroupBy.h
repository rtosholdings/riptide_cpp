#pragma once

#include "CommonInc.h"
#include "MultiKey.h"

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

    int32_t * pCountOut;

    int32_t numpyOutType;
    int32_t funcNum;

    int64_t binLow;
    int64_t binHigh;

    union
    {
        GROUPBY_TWO_FUNC pFunction;
        GROUPBY_X_FUNC32 pFunctionX32;
        GROUPBY_X_FUNC64 pFunctionX64;
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
