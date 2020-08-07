#pragma once

#include "CommonInc.h"

//PyObject *
//GroupByOp32(PyObject *self, PyObject *args);

// Basic call for sum, average
PyObject *
GroupByAll32(PyObject *self, PyObject *args);

PyObject *
GroupByAll64(PyObject *self, PyObject *args);

// More advanced call to calculate median, nth, last
PyObject *
GroupByAllPack32(PyObject *self, PyObject *args);

struct stGroupByReturn {

   union {
      PyArrayObject*    outArray;
      INT64             didWork;
   };

   // for multithreaded sum this is set
   void*             pOutArray;

   INT32*            pCountOut;

   INT32             numpyOutType;
   INT32             funcNum;

   INT64             binLow;
   INT64             binHigh;

   union {
      GROUPBY_TWO_FUNC  pFunction;
      GROUPBY_X_FUNC32  pFunctionX32;
      GROUPBY_X_FUNC64  pFunctionX64;
   };

   PyObject*         returnObject;

};

struct stGroupBy32 {

   ArrayInfo* aInfo;
   INT64    tupleSize;
 /*  INT32    numpyOutType;
   INT32    reserved;
 */  
   void*    pDataIn2;
   INT64    itemSize2;

   INT64    uniqueRows;
   INT64    totalInputRows;

   TYPE_OF_FUNCTION_CALL   typeOfFunctionCall;
   INT64    funcParam;
   void*    pKey;
   void*    pGroup;
   void*    pFirst;
   void*    pCount;

   stGroupByReturn returnObjects[1];
};

