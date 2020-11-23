#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "MultiKey.h"
#include "Ema.h"
#include <stdio.h>
#include <cmath>
#include <pymem.h>
//#include <numpy/ndarraytypes.h>
//#include "numpy/arrayobject.h"
//#include <numpy/npy_common.h>
//#include "npy_config.h"
#if defined(_WIN32) && !defined(__GNUC__)
#include <../Lib/site-packages/numpy/core/include/numpy/ndarraytypes.h>
#include <../Lib/site-packages/numpy/core/include/numpy/arrayobject.h>
#include <../Lib/site-packages/numpy/core/include/numpy/npy_common.h>
#else
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_common.h>
#endif
//#include <../Lib/site-packages/numpy/npy_config.h>

#define LOGGING(...)
//#define LOGGING printf

typedef void(*ROLLING_FUNC)(void* pDataIn, void* pDataOut, INT64 len, INT64 windowSize);

// These are non-groupby routine -- straight array
enum ROLLING_FUNCTIONS {
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
enum EMA_FUNCTIONS {
   EMA_CUMSUM = 300,
   EMA_DECAY = 301,
   EMA_CUMPROD = 302,
   EMA_FINDNTH = 303,
   EMA_NORMAL= 304,
   EMA_WEIGHTED = 305

};

//=========================================================================================================================
typedef void(*EMA_BY_TWO_FUNC)(void* pKey, void* pAccumBin, void* pColumn, INT64 numUnique, INT64 totalInputRows, void* pTime, INT8* pIncludeMask, INT8* pResetMask, double decayRate);

struct stEmaReturn {

   PyArrayObject*    outArray;
   INT32             numpyOutType;
   EMA_BY_TWO_FUNC   pFunction;
   PyObject*         returnObject;
};

//---------------------------------
// 32bit indexes
struct stEma32 {

   ArrayInfo* aInfo;
   INT64    tupleSize;
   INT32    funcNum;
   INT64    uniqueRows;
   INT64    totalInputRows;

   TYPE_OF_FUNCTION_CALL   typeOfFunctionCall;
   void*   pKey;

   // from params
   void*    pTime;
   INT8*    inIncludeMask;
   INT8*    inResetMask;
   double   doubleParam;

   stEmaReturn returnObjects[1];
};



//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// thus <float, int32> converts a float to an int32
template<typename T, typename U>
class EmaBase {
public:
   EmaBase() {};
   ~EmaBase() {};

   // Pass in two vectors and return one vector
   // Used for operations like C = A + B
   //typedef void(*ANY_TWO_FUNC)(void* pDataIn, void* pDataIn2, void* pDataOut, INT64 len, INT32 scalarMode);
   //typedef void(*ANY_ONE_FUNC)(void* pDataIn, void* pDataOut, INT64 len);

   static void RollingSum(void* pDataIn, void* pDataOut, INT64 len, INT64 windowSize) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      U currentSum = 0;

      // Priming of the summation
      for (INT64 i = 0; i < len && i < windowSize; i++) {
         currentSum += pIn[i];
         pOut[i] = currentSum;
      }

      for (INT64 i = windowSize; i < len; i++) {
         currentSum += pIn[i];

         // subtract the item leaving the window
         currentSum -= pIn[i - windowSize];

         pOut[i] = currentSum;
      }
   }


   static void RollingNanSum(void* pDataIn, void* pDataOut, INT64 len, INT64 windowSize) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      U currentSum = 0;

      T invalid = GET_INVALID((T)0);

      if (invalid == invalid) {
         // NON_FLOAT
         // Priming of the summation
         for (INT64 i = 0; i < len && i < windowSize; i++) {
            T temp = pIn[i];

            if (temp != invalid) {
               currentSum += temp;
            }
            pOut[i] = currentSum;
         }

         for (INT64 i = windowSize; i < len; i++) {
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
      else {
         // FLOAT
         // Priming of the summation
         for (INT64 i = 0; i < len && i < windowSize; i++) {
            T temp = pIn[i];

            if (temp == temp) {
               currentSum += temp;
            }
            pOut[i] = currentSum;
         }

         for (INT64 i = windowSize; i < len; i++) {
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


   static void RollingMean(void* pDataIn, void* pDataOut, INT64 len, INT64 windowSize) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      U currentSum = 0;

      // Priming of the summation
      for (INT64 i = 0; i < len && i < windowSize; i++) {
         currentSum += pIn[i];
         pOut[i] = currentSum / (i+1);
      }

      for (INT64 i = windowSize; i < len; i++) {
         currentSum += pIn[i];

         // subtract the item leaving the window
         currentSum -= pIn[i - windowSize];

         pOut[i] = currentSum / windowSize;
      }
   }


   static void RollingNanMean(void* pDataIn, void* pDataOut, INT64 len, INT64 windowSize) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      U currentSum = 0;
      U count = 0;

      // Priming of the summation
      for (INT64 i = 0; i < len && i < windowSize; i++) {
         T temp = pIn[i];

         if (temp == temp) {
            currentSum += temp;
            count++;
         }
         pOut[i] = currentSum/count;
      }

      for (INT64 i = windowSize; i < len; i++) {
         T temp = pIn[i];

         if (temp == temp) {
            currentSum += pIn[i];
            count++;
         }

         // subtract the item leaving the window
         temp = pIn[i - windowSize];

         if (temp == temp) {
            currentSum -= pIn[i - windowSize];
            count--;
         }

         pOut[i] = currentSum/count;
      }
   }


   static void RollingVar(void* pDataIn, void* pDataOut, INT64 len, INT64 windowSize) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      U amean = 0;
      U asqr = 0;
      U delta;

      // Priming of the summation
      for (INT64 i = 0; i < len && i < windowSize; i++) {
         T item = pIn[i];

         delta = item - amean;
         amean += delta / (i+1);
         asqr += delta*(item - amean);
         pOut[i] = asqr/i;
      }

      U count_inv = (U)1.0 / windowSize;

      for (INT64 i = windowSize; i < len; i++) {
         U item = (U)pIn[i];
         U old = (U)pIn[i - windowSize];

         delta = item - old;
         old -= amean;
         amean += delta * count_inv;
         item -= amean;
         asqr += (item + old) * delta;

         pOut[i] = asqr*count_inv;
      }
   }



   static void RollingStd(void* pDataIn, void* pDataOut, INT64 len, INT64 windowSize) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      U amean = 0;
      U asqr = 0;
      U delta;

      // Priming of the summation
      for (INT64 i = 0; i < len && i < windowSize; i++) {
         T item = pIn[i];

         delta = item - amean;
         amean += delta / (i+1);
         asqr += delta*(item - amean);
         pOut[i] = sqrt(asqr/i);
      }

      U count_inv = (U)1.0 / windowSize;

      for (INT64 i = windowSize; i < len; i++) {
         U item = (U)pIn[i];
         U old = (U)pIn[i - windowSize];

         delta = item - old;
         old -= amean;
         amean += delta * count_inv;
         item -= amean;
         asqr += (item + old) * delta;

         pOut[i] = sqrt(asqr*count_inv);
      }
   }


   static void RollingNanVar(void* pDataIn, void* pDataOut, INT64 len, INT64 windowSize) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      U amean = 0;
      U asqr = 0;
      U delta;
      U count = 0;

      // Priming of the summation
      for (INT64 i = 0; i < len && i < windowSize; i++) {
         U item = (U)pIn[i];

         if (item == item) {
            count += 1;
            delta = item - amean;
            amean += delta / count;
            asqr += delta*(item - amean);
            pOut[i] = asqr/count;
         }
         else {
            pOut[i] = NAN;
         }
      }

      U count_inv = (U)1.0 / windowSize;

      for (INT64 i = windowSize; i < len; i++) {
         U item = (U)pIn[i];
         U old = (U)pIn[i - windowSize];

         if (item == item) {
            if (old == old) {
               delta = item - old;
               old -= amean;
               amean += delta * count_inv;
               item -= amean;
               asqr += (item + old)*delta;
            }
            else {
               count += 1;
               count_inv = (U)1 / count;
               //ddof
               delta = item - amean;
               amean += delta * count_inv;
               asqr += delta * (item - amean);
            }
         }
         else {
            if (old == old) {
               count -= 1;
               count_inv = (U)1 / count;
               //dd
               if (count > 0) {
                  delta = old = amean;
                  amean -= delta * count_inv;
                  asqr -= delta * (old - amean);
               }
               else {
                  amean = 0;
                  asqr = 0;
               }

            }
         }
         if (!(asqr >= 0)) {
            asqr = 0;
         }

         pOut[i] = asqr*count_inv;

         // SQR pOut[i] = sqrt(asqr*count_inv);
      }
   }





   static void RollingNanStd(void* pDataIn, void* pDataOut, INT64 len, INT64 windowSize) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      U amean = 0;
      U asqr = 0;
      U delta;
      U count = 0;

      // Priming of the summation
      for (INT64 i = 0; i < len && i < windowSize; i++) {
         U item = (U)pIn[i];

         if (item == item) {
            count += 1;
            delta = item - amean;
            amean += delta / count;
            asqr += delta*(item - amean);
            pOut[i] = sqrt(asqr/count);
         }
         else {
            pOut[i] = NAN;
         }
      }

      U count_inv = (U)1.0 / windowSize;

      for (INT64 i = windowSize; i < len; i++) {
         U item = (U)pIn[i];
         U old = (U)pIn[i - windowSize];

         if (item == item) {
            if (old == old) {
               delta = item - old;
               old -= amean;
               amean += delta * count_inv;
               item -= amean;
               asqr += (item + old)*delta;
            }
            else {
               count += 1;
               count_inv = (U)1 / count;
               //ddof
               delta = item - amean;
               amean += delta * count_inv;
               asqr += delta * (item - amean);
            }
         }
         else {
            if (old == old) {
               count -= 1;
               count_inv = (U)1 / count;
               //dd
               if (count > 0) {
                  delta = old = amean;
                  amean -= delta * count_inv;
                  asqr -= delta * (old - amean);
               }
               else {
                  amean = 0;
                  asqr = 0;
               }

            }
         }
         if (!(asqr >= 0)) {
            asqr = 0;
         }

         pOut[i] = sqrt(asqr * count_inv);

         // SQR pOut[i] = sqrt(asqr*count_inv);
      }
   }




   static ROLLING_FUNC GetRollingFunction(INT64 func) {
      switch (func) {
      case ROLLING_SUM: return RollingSum;
      case ROLLING_NANSUM: return RollingNanSum;
      }
      return NULL;
   }

   static ROLLING_FUNC GetRollingFunction2(INT64 func) {
      switch (func) {
      case ROLLING_MEAN: return RollingMean;
      case ROLLING_NANMEAN: return RollingNanMean;
      case ROLLING_VAR: return RollingVar;
      case ROLLING_NANVAR: return RollingNanVar;
      case ROLLING_STD: return RollingStd;
      case ROLLING_NANSTD: return RollingNanStd;
      }
      return NULL;
   }

};



ROLLING_FUNC GetRollingFunction(INT64 func, INT32 inputType) {
   switch (inputType) {
   case NPY_BOOL:   return EmaBase<INT8, INT64>::GetRollingFunction(func);
   case NPY_FLOAT:  return EmaBase<float, float>::GetRollingFunction(func);
   case NPY_DOUBLE: return EmaBase<double, double>::GetRollingFunction(func);
   case NPY_LONGDOUBLE: return EmaBase<long double, long double>::GetRollingFunction(func);
   case NPY_INT8:   return EmaBase<INT8, INT64>::GetRollingFunction(func);
   case NPY_INT16:  return EmaBase<INT16, INT64>::GetRollingFunction(func);
   CASE_NPY_INT32:  return EmaBase<INT32, INT64>::GetRollingFunction(func);
   CASE_NPY_UINT32: return EmaBase<UINT32, INT64>::GetRollingFunction(func);
   CASE_NPY_INT64:  return EmaBase<INT64, INT64>::GetRollingFunction(func);
   case NPY_UINT8:  return EmaBase<UINT8, INT64>::GetRollingFunction(func);
   case NPY_UINT16: return EmaBase<UINT16, INT64>::GetRollingFunction(func);
   CASE_NPY_UINT64: return EmaBase<UINT64, INT64>::GetRollingFunction(func);
   }

   return NULL;
}

ROLLING_FUNC GetRollingFunction2(INT64 func, INT32 inputType) {
   switch (inputType) {
   case NPY_BOOL:   return EmaBase<INT8, double>::GetRollingFunction2(func);
   case NPY_FLOAT:  return EmaBase<float, float>::GetRollingFunction2(func);
   case NPY_DOUBLE: return EmaBase<double, double>::GetRollingFunction2(func);
   case NPY_LONGDOUBLE: return EmaBase<long double, long double>::GetRollingFunction2(func);
   case NPY_INT8:   return EmaBase<INT8, double>::GetRollingFunction2(func);
   case NPY_INT16:  return EmaBase<INT16, double>::GetRollingFunction2(func);
   CASE_NPY_INT32:  return EmaBase<INT32, double>::GetRollingFunction2(func);
   CASE_NPY_UINT32:  return EmaBase<UINT32, double>::GetRollingFunction2(func);
   CASE_NPY_INT64:  return EmaBase<INT64, double>::GetRollingFunction2(func);
   case NPY_UINT8:  return EmaBase<UINT8, double>::GetRollingFunction2(func);
   case NPY_UINT16: return EmaBase<UINT16, double>::GetRollingFunction2(func);
   CASE_NPY_UINT64: return EmaBase<UINT64, double>::GetRollingFunction2(func);
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
PyObject *
Rolling(PyObject *self, PyObject *args)
{
   PyArrayObject* inArrObject = NULL;
   INT64 func = 0;
   INT64 param1 = 0;

   if (!PyArg_ParseTuple(
      args, "O!LL",
      &PyArray_Type, &inArrObject,
      &func,
      &param1)) {

      return NULL;
   }

   // TODO: determine based on function
   INT32 numpyOutType = NPY_FLOAT64;

   // In case user passes in sliced array or reversed array
   PyArrayObject* inArr = EnsureContiguousArray(inArrObject);
   if (!inArr) return NULL;

   INT32 dType = PyArray_TYPE(inArr);

   PyArrayObject* outArray = NULL;
   INT64 size = ArrayLength(inArr);
   ROLLING_FUNC pRollingFunc;

   numpyOutType = NPY_INT64;

   if (func >= 100) {
      pRollingFunc = GetRollingFunction2(func, dType);

      // Always want some sort of float
      numpyOutType = NPY_DOUBLE;
      if (dType == NPY_FLOAT) {
         numpyOutType = NPY_FLOAT;
      }
      if (dType == NPY_LONGDOUBLE) {
         numpyOutType = NPY_LONGDOUBLE;
      }

   }
   else {
      pRollingFunc = GetRollingFunction(func, dType);

      // Always want some sort of int64 or float
      numpyOutType = NPY_INT64;
      if (dType == NPY_FLOAT) {
         numpyOutType = NPY_FLOAT;
      }
      if (dType == NPY_DOUBLE) {
         numpyOutType = NPY_DOUBLE;
      }
      if (dType == NPY_LONGDOUBLE) {
         numpyOutType = NPY_LONGDOUBLE;
      }
   }


   if (pRollingFunc) {
      // Dont bother allocating if we cannot call the function
      outArray = AllocateNumpyArray(1, (npy_intp*)&size, numpyOutType);

      if (outArray) {
         pRollingFunc(PyArray_BYTES(inArr), PyArray_BYTES(outArray), size, param1);
      }
   }
   else {
      Py_INCREF(Py_None);
      outArray = (PyArrayObject*)Py_None;
   }

   // cleanup if we made a copy
   if (inArr != inArrObject) Py_DecRef((PyObject*)inArr);
   return (PyObject*)outArray;
}


//===================================================================
//
//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// K = data type for indexing (often INT32* or INT8*)
template<typename T, typename U, typename K>
static void CumSum(
   void*    pKeyT,
   void*    pAccumBin,
   void*    pColumn,
   INT64    numUnique,
   INT64    totalInputRows,
   void*    pTime1, // not used
   INT8*    pIncludeMask,
   INT8*    pResetMask,
   double   windowSize1) {

   T* pSrc = (T*)pColumn;
   U* pDest = (U*)pAccumBin;
   K* pKey = (K*)pKeyT;

   U Invalid = GET_INVALID(pDest[0]);

   INT32 windowSize = (INT32)windowSize1;

   LOGGING("cumsum %lld  %lld  %lld  %p  %p\n", numUnique, totalInputRows, (INT64)Invalid, pIncludeMask, pResetMask);

   // Alloc a workspace
   INT64 size = (numUnique + GB_BASE_INDEX) * sizeof(U);
   U* pWorkSpace = (U*)WORKSPACE_ALLOC(size);

   // Default every bin to 0, including floats
   memset(pWorkSpace, 0, size);

   if (pIncludeMask != NULL) {
      if (pResetMask != NULL) {
         // filter + reset loop
         for (int i = 0; i < totalInputRows; i++) {
            K location = pKey[i];
            // Bin 0 is bad
            if (location >= GB_BASE_INDEX) {
               if (pIncludeMask[i] != 0) {
                  if (pResetMask[i]) pWorkSpace[location] = 0;
                  pWorkSpace[location] += (U)pSrc[i];
               }
               pDest[i] = pWorkSpace[location];
            }
            else {
               pDest[i] = Invalid;
            }
         }
      }
      else {
         // filter loop
         for (int i = 0; i < totalInputRows; i++) {
            K location = pKey[i];
            // Bin 0 is bad
            if (location >= GB_BASE_INDEX) {
               if (pIncludeMask[i] != 0) {
                  //printf("adding %lld to %lld,", (INT64)pSrc[location], (INT64)pWorkSpace[location]);
                  pWorkSpace[location] += (U)pSrc[i];
               }
               pDest[i] = pWorkSpace[location];
            }
            else {
               pDest[i] = Invalid;
            }
         }
      }
   }
   else {
      if (pResetMask != NULL) {
         // reset loop
         for (int i = 0; i < totalInputRows; i++) {
            K location = pKey[i];

            // Bin 0 is bad
            if (location >= GB_BASE_INDEX) {
               if (pResetMask[i]) pWorkSpace[location] = 0;
               pWorkSpace[location] += (U)pSrc[i];
               pDest[i] = pWorkSpace[location];
            }
            else {
               pDest[i] = Invalid;
            }
         }
      }

      else {
         for (int i = 0; i < totalInputRows; i++) {
            K location = pKey[i];

            // Bin 0 is bad
            if (location >= GB_BASE_INDEX) {
               pWorkSpace[location] += (U)pSrc[i];
               pDest[i] = pWorkSpace[location];
            }
            else {
               // out of range bin printf("!!!%d --- %lld\n", i, (INT64)Invalid);
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
// K = key index pointer type (INT32* or INT8*)
template<typename T, typename U, typename K>
static void CumProd(
   void*    pKeyT,
   void*    pAccumBin,
   void*    pColumn,
   INT64    numUnique,
   INT64    totalInputRows,
   void*    pTime1, // not used
   INT8*    pIncludeMask,
   INT8*    pResetMask,
   double   windowSize1) {

   T* pSrc = (T*)pColumn;
   U* pDest = (U*)pAccumBin;
   K* pKey = (K*)pKeyT;

   U Invalid = GET_INVALID(pDest[0]);

   INT32 windowSize = (INT32)windowSize1;

   LOGGING("cumprod %lld  %lld  %p  %p\n", numUnique, totalInputRows, pIncludeMask, pResetMask);

   // Alloc a workspace
   INT64 size = (numUnique + GB_BASE_INDEX) * sizeof(U);
   U* pWorkSpace = (U*)WORKSPACE_ALLOC(size);

   // Default every bin to 1, including floats
   for (int i = 0; i < (numUnique + GB_BASE_INDEX); i++) {
      pWorkSpace[i] = 1;
   }

   if (pIncludeMask != NULL) {
      if (pResetMask != NULL) {
         // filter + reset loop
         for (int i = 0; i < totalInputRows; i++) {
            K location = pKey[i];
            // Bin 0 is bad
            if (location >= GB_BASE_INDEX) {
               if (pIncludeMask[i]) {
                  if (pResetMask[i]) pWorkSpace[location] = 1;
                  pWorkSpace[location] *= pSrc[i];
               }
               pDest[i] = pWorkSpace[location];
            }
            else {
               pDest[i] = Invalid;

            }
         }
      }
      else {
         // filter loop
         for (int i = 0; i < totalInputRows; i++) {
            K location = pKey[i];
            // Bin 0 is bad
            if (location >= GB_BASE_INDEX) {
               if (pIncludeMask[i]) {
                  pWorkSpace[location] *= pSrc[i];
               }
               pDest[i] = pWorkSpace[location];
            }
            else {
               pDest[i] = Invalid;
            }
         }

      }
   }
   else {
      if (pResetMask != NULL) {
         // reset loop
         for (int i = 0; i < totalInputRows; i++) {
            K location = pKey[i];

            // Bin 0 is bad
            if (location >= GB_BASE_INDEX) {
               if (pResetMask[i]) pWorkSpace[location] = 1;
               pWorkSpace[location] *= pSrc[i];
               pDest[i] = pWorkSpace[location];
            }
            else {
               pDest[i] = Invalid;
            }
         }

      }
      else {
         // plain
         for (int i = 0; i < totalInputRows; i++) {
            K location = pKey[i];

            //if (location < 0 || location >= numUnique) {
            //   printf("!!! invalid location %d\n", location);
            //}

            // Bin 0 is bad
            if (location >= GB_BASE_INDEX) {
               pWorkSpace[location] *= pSrc[i];
               pDest[i] = pWorkSpace[location];
            }
            else {
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
// K = key index pointer type (INT32* or INT8*)
template<typename U, typename K>
static void FindNth(
   void*    pKeyT,
   void*    pAccumBin,
   void*    pColumn,
   INT64    numUnique,
   INT64    totalInputRows,
   void*    pTime1, // not used
   INT8*    pIncludeMask,
   INT8*    pResetMask,
   double   windowSize1) {

   U* pDest = (U*)pAccumBin;
   K* pKey = (K*)pKeyT;

   LOGGING("FindNth %lld  %lld  %p  %p\n", numUnique, totalInputRows, pIncludeMask, pResetMask);

   // Alloc a workspace
   INT64 size = (numUnique + GB_BASE_INDEX) * sizeof(U);
   U* pWorkSpace = (U*)WORKSPACE_ALLOC(size);

   memset(pWorkSpace, 0, size);

   if (pIncludeMask != NULL) {
      // filter loop
      for (int i = 0; i < totalInputRows; i++) {
         K location = pKey[i];
         // Bin 0 is bad
         if (location >= GB_BASE_INDEX && pIncludeMask[i]) {
            pWorkSpace[location]++;
            pDest[i] = pWorkSpace[location];
         }
         else {
            pDest[i] = 0;
         }
      }
   }
   else {
      // plain
      for (int i = 0; i < totalInputRows; i++) {
         K location = pKey[i];

         // Bin 0 is bad
         if (location >= GB_BASE_INDEX) {
            pWorkSpace[location]++;
            pDest[i] = pWorkSpace[location];
         }
         else {
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
// K = key index data type (INT32* or INT8*)
// thus <float, double, int64> 
template<typename T, typename U, typename V, typename K>
class EmaByBase {
public:
   EmaByBase() {};
   ~EmaByBase() {};


   //------------------------------
   // EmaDecay uses entire size: totalInputRows
   // AccumBin is output array
   // pColumn is the user's data
   // pIncludeMask might be NULL
   //    boolean mask
   // pResetMask might be NULL
   //    mask when to reset
   static void EmaDecay(
      void*    pKeyT,
      void*    pAccumBin, 
      void*    pColumn, 
      INT64    numUnique, 
      INT64    totalInputRows, 
      void*    pTime1,
      INT8*    pIncludeMask,
      INT8*    pResetMask, 
      double   decayRate) {

      T* pSrc = (T*)pColumn;
      U* pDest = (U*)pAccumBin;
      V* pTime = (V*)pTime1;
      K* pKey = (K*)pKeyT;

      LOGGING("emadecay %lld  %lld %p\n", numUnique, totalInputRows, pTime1);

      // Alloc a workspace to store
      // lastEma -- type U
      // lastTime -- type V

      INT64 size = (numUnique + GB_BASE_INDEX) * sizeof(U);
      U* pLastEma = (U*)WORKSPACE_ALLOC(size);

      // Default every bin to 0, including floats
      memset(pLastEma, 0, size);

      size = (numUnique + GB_BASE_INDEX) * sizeof(V);
      V* pLastTime = (V*)WORKSPACE_ALLOC(size);

      // Default every LastTime bin to 0, including floats
      memset(pLastTime, 0, size);

      size = (numUnique + GB_BASE_INDEX) * sizeof(T);
      T* pLastValue = (T*)WORKSPACE_ALLOC(size);

      // Default every LastValue bin to 0, including floats
      memset(pLastValue, 0, size);

      U Invalid = GET_INVALID(pDest[0]);

      // Neiman's matlab loop below
      //if (p >= low && p < high) {
      //   p -= low;
      //   ema[i] = v[i] + lastEma[j][p] * exp(-decay * (t[i] - lastTime[j][p]));
      //   lastEma[j][p] = ema[i];
      //   lastTime[j][p] = t[i];
      //}

      if (pIncludeMask != NULL) {
         // filter loop
         if (pResetMask != NULL) {
            // filter + reset
            for (int i = 0; i < totalInputRows; i++) {
               K location = pKey[i];
               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  T value = 0;

                  // NOTE: fill in last value
                  if (pIncludeMask[i] != 0) {
                     value = pSrc[i];

                     if (pResetMask[i]) {
                        pLastEma[location] = 0;
                        pLastTime[location] = 0;
                     }
                     pLastEma[location] = value + pLastEma[location] * exp(-decayRate * (pTime[i] - pLastTime[location]));
                     pLastTime[location] = pTime[i];
                  }

                  pDest[i] = pLastEma[location];
               }
               else {
                  pDest[i] = Invalid;
               }
            }

         }
         else {
            // filter only
            for (int i = 0; i < totalInputRows; i++) {
               K location = pKey[i];
 
               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  T value = 0;

                  // NOTE: fill in last value
                  if (pIncludeMask[i] != 0) {
                     value = pSrc[i];
                  }
                  else {
                     // Acts like fill forward
                     LOGGING("fill forward location: %lld\n", (long long)location);
                     value = pLastValue[location];
                  }
                  pLastEma[location] = value + pLastEma[location] * exp(-decayRate * (pTime[i] - pLastTime[location]));
                  pLastTime[location] = pTime[i];
                  pLastValue[location] = value;
                  pDest[i] = pLastEma[location];
               }
               else {
                  pDest[i] = Invalid;
               }
            }
         }
      }
      else {
         if (pResetMask != NULL) {
            // reset loop
            for (int i = 0; i < totalInputRows; i++) {
               K location = pKey[i];

               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  if (pResetMask[i]) {
                     pLastEma[location] = 0;
                     pLastTime[location] = 0;
                  }
                  pLastEma[location] = pSrc[i] + pLastEma[location] * exp(-decayRate * (pTime[i] - pLastTime[location]));
                  pLastTime[location] = pTime[i];
                  pDest[i] = pLastEma[location];
               }
               else {
                  pDest[i] = Invalid;
               }
            }
         }
         else {
            // plain loop (no reset / no filter)
            for (int i = 0; i < totalInputRows; i++) {
               K location = pKey[i];

               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  //printf("inputs: %lf  %lf  %lf  %lf  %lf\n", (double)pSrc[i], (double)pLastEma[location], (double)-decayRate, (double)pTime[i], (double)pLastTime[location] );
                  pLastEma[location] = pSrc[i] + pLastEma[location] * exp(-decayRate * (pTime[i] - pLastTime[location]));
                  //printf("[%d][%d] %lf\n", i, (INT32)location, (double)pLastEma[location]);
                  pLastTime[location] = pTime[i];
                  pDest[i] = pLastEma[location];
               }
               else {
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
   if (timeDelta < 0) decayedWeight = 0; \
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
   static void EmaNormal(
      void*    pKeyT,
      void*    pAccumBin,
      void*    pColumn,
      INT64    numUnique,
      INT64    totalInputRows,
      void*    pTime1,
      INT8*    pIncludeMask,
      INT8*    pResetMask,
      double   decayRate) {

      T* pSrc = (T*)pColumn;
      U* pDest = (U*)pAccumBin;
      V* pTime = (V*)pTime1;
      K* pKey = (K*)pKeyT;

      LOGGING("emanormal %lld  %lld %p\n", numUnique, totalInputRows, pTime1);

      // Alloc a workspace to store
      // lastEma -- type U
      // lastTime -- type V

      INT64 size = (numUnique + GB_BASE_INDEX) * sizeof(U);
      U* pLastEma = (U*)WORKSPACE_ALLOC(size);

      // Default every bin to 0, including floats
      //memset(pLastEma, 0, size);
      // the first value should be valid
      // go backwards so that first value is in there
      for (INT64 i = totalInputRows - 1; i >= 0; i--)
      {
         K location = pKey[i];
         T value = pSrc[i];
         pLastEma[location] = (U)value;
      }

      //-----------------------------
      size = (numUnique + GB_BASE_INDEX) * sizeof(V);
      V* pLastTime = (V*)WORKSPACE_ALLOC(size);

      size = (numUnique + GB_BASE_INDEX) * sizeof(T);
      T* pLastValue = (T*)WORKSPACE_ALLOC(size);

      // Default every LastValue bin to 0, including floats
      memset(pLastValue, 0, size);

      // Default every LastTime bin to 0, including floats
      // Set first time to very low value
      V largeNegative = 0;
      if (sizeof(V) == 4) {         
         //largeNegative = -INFINITY;
         largeNegative = (V)0x80000000;
      }
      else {
         largeNegative = (V)0x8000000000000000LL;
      }
      for (INT64 i = 0; i < (numUnique + GB_BASE_INDEX); i++) {
         pLastTime[i] = largeNegative;
      }

      U Invalid = GET_INVALID(pDest[0]);

      if (pIncludeMask != NULL) {
         // filter loop
         if (pResetMask != NULL) {
            // filter + reset
            for (INT64 i = 0; i < totalInputRows; i++) {
               K location = pKey[i];
               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  T value = 0;

                  // NOTE: fill in last value
                  if (pIncludeMask[i] != 0) {
                     value = pSrc[i];

                     if (pResetMask[i]) {
                        pLastEma[location] = 0;
                        pLastTime[location] = 0;
                     }
                     EMA_NORMAL_FUNC
                  }
                  else {
                     pDest[i] = pLastEma[location];

                  }

               }
               else {
                  pDest[i] = Invalid;
               }
            }

         }
         else {
            // filter only
            for (INT64 i = 0; i < totalInputRows; i++) {
               K location = pKey[i];

               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  T value = 0;

                  // NOTE: fill in last value
                  if (pIncludeMask[i] != 0) {
                     value = pSrc[i];
                  }
                  else {
                     // Acts like fill forward
                     LOGGING("fill forward location: %lld\n", (long long)location);
                     value = pLastValue[location];
                  }
                  EMA_NORMAL_FUNC
                  pLastValue[location] = value;
               }
               else {
                  pDest[i] = Invalid;
               }
            }
         }
      }
      else {
         if (pResetMask != NULL) {
            // reset loop
            for (INT64 i = 0; i < totalInputRows; i++) {
               K location = pKey[i];

               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  if (pResetMask[i]) {
                     pLastEma[location] = 0;
                     pLastTime[location] = 0;
                  }
                  T value = pSrc[i];
                  EMA_NORMAL_FUNC
               }
               else {
                  pDest[i] = Invalid;
               }
            }
         }
         else {
            // plain loop (no reset / no filter)
            for (INT64 i = 0; i < totalInputRows; i++) {
               K location = pKey[i];

               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  T value = pSrc[i];
                  //double DW = exp(-decayRate * (pTime[i] - pLastTime[location])); 
                  //printf("**dw %lf  %lld  %lld\n", DW, (INT64)pTime[i], (INT64)pLastTime[location]);
                  EMA_NORMAL_FUNC
               }
               else {
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
   static void EmaWeighted(
      void*    pKeyT,
      void*    pAccumBin,
      void*    pColumn,
      INT64    numUnique,
      INT64    totalInputRows,
      void*    pTime1,
      INT8*    pIncludeMask,
      INT8*    pResetMask,
      double   decayedWeight) {

      T* pSrc = (T*)pColumn;
      U* pDest = (U*)pAccumBin;
      K* pKey = (K*)pKeyT;

      LOGGING("emaweighted %lld  %lld %p\n", numUnique, totalInputRows, pTime1);

      // Alloc a workspace to store
      // lastEma -- type U
      // lastTime -- type V

      INT64 size = (numUnique + GB_BASE_INDEX) * sizeof(U);
      U* pLastEma = (U*)WORKSPACE_ALLOC(size);

      // Default every bin to 0, including floats
      //memset(pLastEma, 0, size);

      // the first value should be valid
      // go backwards so that first value is in there
      for (INT64 i = totalInputRows - 1; i >= 0; i--)
      {
         K location = pKey[i];
         T value = pSrc[i];
         pLastEma[location] = (U)value;
      }

      U Invalid = GET_INVALID(pDest[0]);

      if (pIncludeMask != NULL) {
         // filter loop
         if (pResetMask != NULL) {
            // filter + reset
            for (INT64 i = 0; i < totalInputRows; i++) {
               K location = pKey[i];
               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  T value = 0;

                  // NOTE: fill in last value
                  if (pIncludeMask[i] != 0) {
                     value = pSrc[i];

                     if (pResetMask[i]) {
                        pLastEma[location] = 0;
                     }
                  }

                  EMA_WEIGHTED_FUNC
               }
               else {
                  pDest[i] = Invalid;
               }
            }

         }
         else {
            // filter only
            for (INT64 i = 0; i < totalInputRows; i++) {
               K location = pKey[i];

               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  T value = 0;

                  // NOTE: fill in last value
                  if (pIncludeMask[i] != 0) {
                     value = pSrc[i];
                  }
                  EMA_WEIGHTED_FUNC
               }
               else {
                  pDest[i] = Invalid;
               }
            }
         }
      }
      else {
         if (pResetMask != NULL) {
            // reset loop
            for (INT64 i = 0; i < totalInputRows; i++) {
               K location = pKey[i];

               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  if (pResetMask[i]) {
                     pLastEma[location] = 0;
                  }
                  T value = pSrc[i];
                  EMA_WEIGHTED_FUNC
               }
               else {
                  pDest[i] = Invalid;
               }
            }
         }
         else {
            // plain loop (no reset / no filter)
            for (INT64 i = 0; i < totalInputRows; i++) {
               K location = pKey[i];

               // Bin 0 is bad
               if (location >= GB_BASE_INDEX) {
                  T value = pSrc[i];
                  //printf("**dw %d  %d   %lld %lld\n", i, (int)location, (INT64)value, (INT64)pLastEma[location]);
                  EMA_WEIGHTED_FUNC
               }
               else {
                  pDest[i] = Invalid;
               }
            }
         }
      }

      WORKSPACE_FREE(pLastEma);
   }



   //-------------------------------------------------------------------------------
   static EMA_BY_TWO_FUNC GetFunc(EMA_FUNCTIONS func) {

      switch (func) {
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





template<typename T, typename K>
static EMA_BY_TWO_FUNC GetEmaByStep2(int timeType, EMA_FUNCTIONS func) {
   switch (timeType) {
   case NPY_FLOAT:  return EmaByBase<T, double, float, K>::GetFunc(func);
   case NPY_DOUBLE: return EmaByBase<T, double, double, K>::GetFunc(func);
   case NPY_LONGDOUBLE: return EmaByBase<T, long double, long double, K>::GetFunc(func);
   CASE_NPY_INT32:  return EmaByBase<T, double, INT32, K>::GetFunc(func);
   CASE_NPY_INT64:  return EmaByBase<T, double, INT64, K>::GetFunc(func);
   CASE_NPY_UINT32: return EmaByBase<T, double, UINT32, K>::GetFunc(func);
   CASE_NPY_UINT64: return EmaByBase<T, double, UINT64, K>::GetFunc(func);
   }
   return NULL;

}

//------------------------------------------------------
// timeType is -1 for cumsum
// K is the iKey type (int8.int16,int32,int64)
template<typename K>
static EMA_BY_TWO_FUNC GetEmaByFunction(int inputType, int *outputType, int timeType, EMA_FUNCTIONS func) {

   // only support EMADecay

   switch (func) {
   case EMA_CUMSUM:
      switch (inputType) {
      case NPY_FLOAT:  *outputType = NPY_FLOAT32; return CumSum<float, float, K>;
      case NPY_DOUBLE: *outputType = NPY_FLOAT64; return CumSum<double, double, K> ;
      case NPY_LONGDOUBLE: *outputType = NPY_FLOAT64; return CumSum<long double, long double, K>;
      case NPY_BOOL:   *outputType = NPY_INT64; return CumSum<INT8, INT64, K>;
      case NPY_INT8:   *outputType = NPY_INT64; return CumSum<INT8, INT64, K>;
      case NPY_INT16:  *outputType = NPY_INT64; return CumSum<INT16, INT64, K>;
      CASE_NPY_INT32:  *outputType = NPY_INT64; return CumSum<INT32, INT64, K>;
      CASE_NPY_INT64:  *outputType = NPY_INT64; return CumSum<INT64, INT64, K>;

      case NPY_UINT8:  *outputType = NPY_UINT64; return CumSum<UINT8,  UINT64, K>;
      case NPY_UINT16: *outputType = NPY_UINT64; return CumSum<UINT16, UINT64, K>;
      CASE_NPY_UINT32: *outputType = NPY_UINT64; return CumSum<UINT32, UINT64, K>;
      CASE_NPY_UINT64: *outputType = NPY_UINT64; return CumSum<UINT64, UINT64, K>;

      }
      break;

   case EMA_CUMPROD:
      switch (inputType) {
      case NPY_FLOAT:  *outputType = NPY_FLOAT32; return CumProd<float, float, K>;
      case NPY_DOUBLE: *outputType = NPY_FLOAT64; return CumProd<double, double, K>;
      case NPY_LONGDOUBLE: *outputType = NPY_FLOAT64; return CumProd<long double, long double, K>;
      case NPY_BOOL:   *outputType = NPY_INT64; return CumProd<INT8, INT64, K>;
      case NPY_INT8:   *outputType = NPY_INT64; return CumProd<INT8, INT64, K>;
      case NPY_INT16:  *outputType = NPY_INT64; return CumProd<INT16, INT64, K>;
      CASE_NPY_INT32:  *outputType = NPY_INT64; return CumProd<INT32, INT64, K>;
      CASE_NPY_INT64:  *outputType = NPY_INT64; return CumProd<INT64, INT64, K>;

      case NPY_UINT8:  *outputType = NPY_UINT64; return CumProd<UINT8, UINT64, K>;
      case NPY_UINT16: *outputType = NPY_UINT64; return CumProd<UINT16, UINT64, K>;
      CASE_NPY_UINT32: *outputType = NPY_UINT64; return CumProd<UINT32, UINT64, K>;
      CASE_NPY_UINT64: *outputType = NPY_UINT64; return CumProd<UINT64, UINT64, K>;

      }
      break;


   case EMA_FINDNTH:
      *outputType = NPY_INT32; return FindNth< INT32, K>;
      break;

   case EMA_NORMAL:
   case EMA_WEIGHTED:
   case EMA_DECAY:
      *outputType = NPY_FLOAT64;
      switch (inputType) {
      case NPY_BOOL:   return GetEmaByStep2<INT8, K>(timeType, func);
      case NPY_FLOAT:  return GetEmaByStep2<float, K>(timeType, func);
      case NPY_DOUBLE: return GetEmaByStep2<double, K>(timeType, func);
      case NPY_LONGDOUBLE: return GetEmaByStep2<long double, K>(timeType, func);
      case NPY_INT8:   return GetEmaByStep2<INT8, K>(timeType, func);
      case NPY_INT16:  return GetEmaByStep2<INT16, K>(timeType, func);
      CASE_NPY_INT32:  return GetEmaByStep2<INT32, K>(timeType, func);
      CASE_NPY_INT64:  return GetEmaByStep2<INT64, K>(timeType, func);
      case NPY_UINT8:  return GetEmaByStep2<UINT8, K>(timeType, func);
      case NPY_UINT16: return GetEmaByStep2<UINT16, K>(timeType, func);
      CASE_NPY_UINT32: return GetEmaByStep2<UINT32, K>(timeType, func);
      CASE_NPY_UINT64: return GetEmaByStep2<UINT64, K>(timeType, func);

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
void EmaByCall(void* pEmaBy, INT64 i) {

   stEma32* pstEma32 = (stEma32*)pEmaBy;

   ArrayInfo* aInfo = pstEma32->aInfo;
   INT64 uniqueRows = pstEma32->uniqueRows;

   EMA_FUNCTIONS func = (EMA_FUNCTIONS)pstEma32->funcNum;

   // Data in was passed
   void* pDataIn = aInfo[i].pData;
   INT64 len = aInfo[i].ArrayLength;

   PyArrayObject* outArray = pstEma32->returnObjects[i].outArray;
   EMA_BY_TWO_FUNC  pFunction = pstEma32->returnObjects[i].pFunction;
   INT32 numpyOutType = pstEma32->returnObjects[i].numpyOutType;
   TYPE_OF_FUNCTION_CALL typeCall = pstEma32->typeOfFunctionCall;

   if (outArray && pFunction) {
      void* pDataOut = PyArray_BYTES(outArray);
      LOGGING("col %llu  ==> outsize %llu   len: %llu   numpy types %d --> %d   %d %d  ptr: %p\n", i, uniqueRows, len, aInfo[i].NumpyDType, numpyOutType, gNumpyTypeToSize[aInfo[i].NumpyDType], gNumpyTypeToSize[numpyOutType], pDataOut);

      // Accum the calculation
      EMA_BY_TWO_FUNC  pFunctionX = pstEma32->returnObjects[i].pFunction;

      if (pFunctionX) {
         pFunctionX(
            pstEma32->pKey,
            (char*)pDataOut,
            (char*)pDataIn,
            uniqueRows,
            pstEma32->totalInputRows,

            // params
            pstEma32->pTime, 
            pstEma32->inIncludeMask,
            pstEma32->inResetMask,
            pstEma32->doubleParam);

         pstEma32->returnObjects[i].returnObject = (PyObject*)outArray;
      }
      else {
         printf("!!!internal error EmaByCall");
      }
   }
   else {

      // TJD: memory leak?
      if (outArray) {
         printf("!!! deleting extra object\n");
         Py_DecRef((PyObject*)outArray);
      }

      LOGGING("**skipping col %llu  ==> outsize %llu   len: %llu   numpy types %d --> %d   %d %d\n", i, uniqueRows, len, aInfo[i].NumpyDType, numpyOutType, gNumpyTypeToSize[aInfo[i].NumpyDType], gNumpyTypeToSize[numpyOutType]);
      pstEma32->returnObjects[i].returnObject = Py_None;
   }

}





//---------------------------------------------------------------
// Arg1 = LIST of numpy arrays which has the values to accumulate (often all the columns in a dataset)
// Arg2 = iKey = numpy array (INT32) which has the index to the unique keys (ikey from MultiKeyGroupBy32)
// Arg3 = integer unique rows
// Arg4 = integer (function number to execute for cumsum, ema)
// Arg5 = params for function must be (decay/window, time, includemask, resetmask)
// Example: EmaAll32(array, ikey, 3, EMA_DECAY, (5.6, timeArray))
// Returns entire dataset per column
//
PyObject *
EmaAll32(PyObject *self, PyObject *args)
{
   PyObject *inList1 = NULL;
   PyArrayObject *iKey = NULL;
   PyTupleObject *params = NULL;
   PyArrayObject *inTime = NULL;
   PyArrayObject *inIncludeMask = NULL;
   PyArrayObject *inResetMask = NULL;

   double doubleParam = 0.0;
   INT64 unique_rows = 0;
   INT64 funcNum = 0;

   if (!PyArg_ParseTuple(
      args, "OO!LLO",
      &inList1,
      &PyArray_Type, &iKey,
      &unique_rows,
      &funcNum,
      &params)) {

      return NULL;
   }

   if (!PyTuple_Check(params)) {
      PyErr_Format(PyExc_ValueError, "EmaAll32 params argument needs to be a tuple");
      return NULL;
   }

   INT32 iKeyType = PyArray_TYPE(iKey);

   switch (iKeyType) {
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

   switch (tupleSize) {
   case 4:
      if (!PyArg_ParseTuple(
         (PyObject*)params, "dOOO",
         &doubleParam,
         &inTime,
         &inIncludeMask,  // must be boolean for now or empty
         &inResetMask
         )) {

         return NULL;
      }

      // If they pass in NONE make it NULL
      if (inTime == (PyArrayObject*)Py_None) { inTime = NULL; }
      else if (!PyArray_Check(inTime)) {
         PyErr_Format(PyExc_ValueError, "EmaAll32 inTime must be an array");
      }

      if (inIncludeMask == (PyArrayObject*)Py_None) { inIncludeMask = NULL; }
      else if (!PyArray_Check(inIncludeMask)) {
         PyErr_Format(PyExc_ValueError, "EmaAll32 inIncludeMask must be an array");
      }

      if (inResetMask == (PyArrayObject*)Py_None) { inResetMask = NULL; }
      else if (!PyArray_Check(inResetMask)) {
         PyErr_Format(PyExc_ValueError, "EmaAll32 inResetMask must be an array");
      }
      break;

   default:
      PyErr_Format(PyExc_ValueError, "EmaAll32 cannot parse arguments.  tuple size %lld\n", tupleSize);
      return NULL;
   }

   INT64 totalArrayLength = ArrayLength(iKey);

   if (inResetMask != NULL && (PyArray_TYPE(inResetMask) != 0 || ArrayLength(inResetMask) != totalArrayLength)) {
      PyErr_Format(PyExc_ValueError, "EmaAll32 inResetMask must be a bool mask of same size");
      return NULL;
   }

   if (inIncludeMask != NULL && (PyArray_TYPE(inIncludeMask) != 0 || ArrayLength(inIncludeMask) != totalArrayLength)) {
      PyErr_Format(PyExc_ValueError, "EmaAll32 inIncludeMask must be a bool mask of same size");
      return NULL;
   }

   if (inTime != NULL && (PyArray_TYPE(inTime) < NPY_INT || PyArray_TYPE(inTime) > NPY_LONGDOUBLE || ArrayLength(inTime) != totalArrayLength)) {
      PyErr_Format(PyExc_ValueError, "EmaAll32 inTime must be a 32 or 64 bit value of same size");
      return NULL;
   }

   INT32 numpyInType2 = ObjectToDtype(iKey);

   INT64 totalItemSize = 0;
   ArrayInfo* aInfo = BuildArrayInfo(inList1, (INT64*)&tupleSize, &totalItemSize);

   if (!aInfo) {
      PyErr_Format(PyExc_ValueError, "EmaAll32 failed to produce aInfo");
      return NULL;
   }

   LOGGING("Ema started %llu  param:%lf\n", tupleSize, doubleParam);

   // Allocate the struct + ROOM at the end of struct for all the tuple objects being produced
   stEma32* pstEma32 = (stEma32*)WORKSPACE_ALLOC((sizeof(stEma32) + 8 + sizeof(stEmaReturn))*tupleSize);

   pstEma32->aInfo = aInfo;
   pstEma32->funcNum = (INT32)funcNum;
   pstEma32->pKey = (INT32*)PyArray_BYTES(iKey);
   pstEma32->tupleSize = tupleSize;
   pstEma32->uniqueRows = unique_rows;
   pstEma32->totalInputRows = totalArrayLength;
   pstEma32->doubleParam = doubleParam;
   pstEma32->pTime = inTime == NULL ? NULL : PyArray_BYTES(inTime);
   pstEma32->inIncludeMask = inIncludeMask == NULL ? NULL : (INT8*)PyArray_BYTES(inIncludeMask);
   pstEma32->inResetMask = inResetMask == NULL ? NULL : (INT8*)PyArray_BYTES(inResetMask);
   pstEma32->typeOfFunctionCall = TYPE_OF_FUNCTION_CALL::ANY_GROUPBY_FUNC;

   LOGGING("Ema unique %lld  total: %lld  arrays: %p %p %p\n", unique_rows, totalArrayLength, pstEma32->pTime, pstEma32->inIncludeMask, pstEma32->inResetMask);

   // Allocate all the memory and output arrays up front since Python is single threaded
   for (int i = 0; i < tupleSize; i++) {
      // TODO: determine based on function
      INT32 numpyOutType = -1;

      EMA_BY_TWO_FUNC  pFunction = NULL;
      switch (iKeyType) {
      case NPY_INT8:
         pFunction = GetEmaByFunction<INT8>(aInfo[i].NumpyDType, &numpyOutType, inTime == NULL ? -1 : PyArray_TYPE(inTime), (EMA_FUNCTIONS)funcNum);
         break;
      case NPY_INT16:
         pFunction = GetEmaByFunction<INT16>(aInfo[i].NumpyDType, &numpyOutType, inTime == NULL ? -1 : PyArray_TYPE(inTime), (EMA_FUNCTIONS)funcNum);
         break;
      CASE_NPY_INT32:
         pFunction = GetEmaByFunction<INT32>(aInfo[i].NumpyDType, &numpyOutType, inTime == NULL ? -1 : PyArray_TYPE(inTime), (EMA_FUNCTIONS)funcNum);
         break;
      CASE_NPY_INT64:
         pFunction = GetEmaByFunction<INT64>(aInfo[i].NumpyDType, &numpyOutType, inTime == NULL ? -1 : PyArray_TYPE(inTime), (EMA_FUNCTIONS)funcNum);
         break;
      }

      PyArrayObject* outArray = NULL;

      // Dont bother allocating if we cannot call the function
      if (pFunction) {
         // Allocate the output size for each column
         outArray = AllocateNumpyArray(1, (npy_intp*)&totalArrayLength, numpyOutType);
         LOGGING("[%d] Allocated output array size %lld for type %d  ptr:%p\n", i, totalArrayLength, numpyOutType, PyArray_BYTES(outArray, 0));
      }
      else {
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
   PyObject* returnTuple = PyTuple_New(tupleSize);

   // Fill in results
   for (int i = 0; i < tupleSize; i++) {
      PyObject* item = pstEma32->returnObjects[i].returnObject;

      if (item == Py_None)
         Py_INCREF(Py_None);

      // Set item will not change reference
      PyTuple_SET_ITEM(returnTuple, i, item);
      //printf("after ref %d  %llu\n", i, item->ob_refcnt);
   }

   //LOGGING("Return tuple ref %llu\n", returnTuple->ob_refcnt);
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
template<typename T>
void mat_interp_extrap(void* xT, void* xpT, void* ypT, void* outT, INT64 N, INT64 M, INT32 clip) {

   T* x = (T*)xT;
   T* xp = (T*)xpT;
   T* yp = (T*)ypT;
   T* out = (T*)outT;

   T mynan = std::numeric_limits<T>::quiet_NaN();

   if (!clip) {
      // auto increment xp and yp
      for (INT64 i = 0; i < N; ++i, xp += M, yp += M) {
         T xi = x[i];
         T result = mynan;

         if (xi == xi) {
            if (xi > xp[0]) {
               INT64 j = 1;
               while (xi > xp[j] && j < M) j++;
               if (j == M) {
                  T right_slope = (yp[M - 1] - yp[M - 2]) / (xp[M - 1] - xp[M - 2]);
                  result = yp[M - 1] + right_slope * (xi - xp[M - 1]);
               }
               else {
                  // middle slope
                  result = (yp[j] - yp[j - 1])*(xi - xp[j - 1]) / (xp[j] - xp[j - 1]) + yp[j - 1];
               }
            }
            else {
               T left_slope = (yp[1] - yp[0]) / (xp[1] - xp[0]);
               result = yp[0] + left_slope * (xi - xp[0]);
            }
         }
         out[i] = result;
      }
   }
   else {
      // clipping
      for (INT64 i = 0; i < N; ++i, xp += M, yp += M) {
         T xi = x[i];
         T result=mynan;

         if (xi == xi) {
            if (xi > xp[0]) {
               INT64 j = 1;
               while (xi > xp[j] && j < M) j++;
               if (j == M) {
                  result = yp[M - 1];
               }
               else {
                  // middle slope
                  result = (yp[j] - yp[j - 1])*(xi - xp[j - 1]) / (xp[j] - xp[j - 1]) + yp[j - 1];
               }
            }
            else {
               result = yp[0];
            }
         }
         out[i] = result;
      }
   }
}


template<typename T>
void mat_interp(void* xT, void* xpT, void* ypT, void* outT, INT64 N, INT64 M, INT32 clip) {

   T* x = (T*)xT;
   T* xp = (T*)xpT;
   T* yp = (T*)ypT;
   T* out = (T*)outT;

   T xp0 = xp[0];
   T yp0 = yp[0];
   T mynan = std::numeric_limits<T>::quiet_NaN();

   if (!clip) {
      for (INT64 i = 0; i < N; ++i) {
         T xi = x[i];
         T result = mynan;

         if (xi == xi) {
            if (xi > xp0) {
               INT64 j = 1;
               while (xi > xp[j] && j < M) j++;
               if (j == M) {
                  T right_slope = (yp[M - 1] - yp[M - 2]) / (xp[M - 1] - xp[M - 2]);
                  result = yp[M - 1] + right_slope * (xi - xp[M - 1]);
               }
               else {
                  // middle slope
                  result = (yp[j] - yp[j - 1])*(xi - xp[j - 1]) / (xp[j] - xp[j - 1]) + yp[j - 1];
               }
            }
            else {
               T left_slope = (yp[1] - yp[0]) / (xp[1] - xp0);
               result = yp[0] + left_slope * (xi - xp0);
            }
         }
         out[i] = result;
      }
   }
   else {
      for (INT64 i = 0; i < N; ++i) {
         T xi = x[i];
         T result = mynan;

         if (xi == xi) {
            if (xi > xp0) {
               INT64 j = 1;
               while (xi > xp[j] && j < M) j++;
               if (j == M) {
                  // clipped
                  result = yp[M - 1];
               }
               else {
                  // middle slope
                  result = (yp[j] - yp[j - 1])*(xi - xp[j - 1]) / (xp[j] - xp[j - 1]) + yp[j - 1];
               }
            }
            else {
               // clipped
               result = yp0;
            }
         }
         out[i] = result;
      }
   }
}

// struct for multithreading
struct stInterp {
   char* x;
   char* xp;
   char* yp;
   char* out;
   INT64 N;
   INT64 M;
   INT32 mode;
   INT32 clip;
   int itemsize;
};

//-----------------------------
// multithreaded callback
// move pointers to start offset
// shrink N to what length is
BOOL InterpolateExtrap(void* callbackArgT, int core, INT64 start, INT64 length) {
   stInterp* pInterp = (stInterp*)callbackArgT;
   INT64 M = pInterp->M;
   INT64 N = pInterp->N;
   INT32 clip = pInterp->clip;

   INT64 fixup = start * pInterp->itemsize;
   if (pInterp->mode == 2) {
      INT64 fixup2d = start * pInterp->itemsize* M;
      if (pInterp->itemsize == 8) {
         mat_interp_extrap<double>(
            pInterp->x + fixup,
            pInterp->xp + fixup2d,
            pInterp->yp + fixup2d,
            pInterp->out + fixup,
            length,
            M,
            clip);
      }
      else {
         mat_interp_extrap<float>(
            pInterp->x + fixup,
            pInterp->xp + fixup2d,
            pInterp->yp + fixup2d,
            pInterp->out + fixup,
            length,
            M,
            clip);
      }
   }
   else {
      if (pInterp->itemsize == 8) {
         mat_interp<double>(
            pInterp->x + fixup,
            pInterp->xp ,
            pInterp->yp ,
            pInterp->out + fixup,
            length,
            M,
            clip);
      }
      else {
         mat_interp<float>(
            pInterp->x + fixup,
            pInterp->xp ,
            pInterp->yp ,
            pInterp->out + fixup,
            length,
            M,
            clip);
      }

   }
   return TRUE;
}



//--------------------------------------------
// arg1: arr: 1 dimensional double or float
// arg2: xp:  2 dimensional double or float
// arg3: yp:  2 dimensional double or float
// arg4: clip  set to 1 to clip (optional defaults to no clip)
// Returns 1 dimensional array of interpolated values
 PyObject* InterpExtrap2d(PyObject* self, PyObject* args) {
   PyArrayObject* arr;
   PyArrayObject* xp;
   PyArrayObject* yp;
   PyArrayObject* returnArray; // we allocate this
   INT32 clip = 0;
   INT32 mode = 0;

   if (!PyArg_ParseTuple(args, "O!O!O!|i",
      &PyArray_Type, &arr,
      &PyArray_Type, &xp,
      &PyArray_Type, &yp,
      &clip)) {

      // If pyargparsetuple fails, it will set the error for us
      return NULL;
   }

   if (PyArray_NDIM(arr) > 1) {
      PyErr_Format(PyExc_ValueError, "The 1st argument must be 1 dimensional arrays");
      return NULL;
   }

   if ((PyArray_NDIM(xp) != PyArray_NDIM(yp)) || PyArray_NDIM(yp) > 2) {
      PyErr_Format(PyExc_ValueError, "The 2nd and 3rd argument must be the same dimensions");
      return NULL;
   }

   if (PyArray_NDIM(xp) == 2) {
      mode = 2;
   }
   else {
      mode = 1;
   }

   if (!(PyArray_FLAGS(xp) & PyArray_FLAGS(yp) & NPY_ARRAY_C_CONTIGUOUS)) {
      PyErr_Format(PyExc_ValueError, "The 2nd and 3rd argument must be row major, contiguous 2 dimensional arrays");
      return NULL;
   }
   // NOTE: could check for strides also here

   INT64 N = PyArray_DIM(arr, 0);
   INT64 M = 0;

   if (mode == 2) {
      if ((N != PyArray_DIM(xp, 0)) || (N != PyArray_DIM(yp, 0))) {
         PyErr_Format(PyExc_ValueError, "The arrays must be the same size on the first dimension: %lld", N);
         return NULL;
      }
      M = PyArray_DIM(xp, 1);
      if (M != PyArray_DIM(yp, 1) || M < 2) {
         PyErr_Format(PyExc_ValueError, "The 2nd and 3rd arrays must be the same size on the second dimension: %lld", M);
         return NULL;
      }
   }
   else {
      M = PyArray_DIM(xp, 0);
      if (M != PyArray_DIM(yp, 0) || M < 2) {
         PyErr_Format(PyExc_ValueError, "The 2nd and 3rd arrays must be the same size on the first dimension: %lld", M);
         return NULL;
      }
   }

   // Accept all double or all floats
   int dtype = PyArray_TYPE(arr);
   if (dtype != PyArray_TYPE(xp) || dtype != PyArray_TYPE(yp)) {
      PyErr_Format(PyExc_ValueError, "The arrays must all be the same type: %d", dtype);
      return NULL;
   }

   if (dtype != NPY_FLOAT64 && dtype != NPY_FLOAT32) {
      PyErr_Format(PyExc_ValueError, "The arrays must all be float32 or float64 not type: %d", dtype);
      return NULL;
   }

   // allocate a float or a double
   returnArray = AllocateLikeNumpyArray(arr, dtype);

   if (returnArray) {

      // copy params we will use into a struct on the stack
      stInterp  interp;
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
   return (PyObject*)returnArray;
}


//-----------------------------------------------------
//
//
 //PyObject* EmaSimple(PyObject* self, PyObject* args) {
 //  PyArrayObject* arrTime;
 //  PyArrayObject* arrTime; xp;
 //  PyArrayObject* yp;
 //  PyArrayObject* returnArray; // we allocate this
 //  INT32 clip = 0;
 //  INT32 mode = 0;

 //  if (!PyArg_ParseTuple(args, "O!O!O!|i",
 //      &PyArray_Type, &arr,
 //      &PyArray_Type, &xp,
 //      &PyArray_Type, &yp,
 //      &clip)) {

 //     double timeDelta = double(pTime[i] - pLastTime[location]); 
 //     double decayedWeight = exp(-decayRate * timeDelta); 
 //     if (timeDelta < 0) decayedWeight = 0; 
 //     pLastEma[location] = value * (1 - decayedWeight) + pLastEma[location] * decayedWeight; 
 //     pLastTime[location] = pTime[i]; 
 //     pDest[i] = pLastEma[location];

 //  }
//}