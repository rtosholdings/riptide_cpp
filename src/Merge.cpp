#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "Merge.h"
#include "Convert.h"

#include <algorithm>

// for _pext_u64
#if defined(__GNUC__) || defined(__clang__)
//#include <bmi2intrin.h>
#include <x86intrin.h>
#endif

//#define LOGGING printf
#define LOGGING(...)

typedef void(*MBGET_FUNC)(void* pDataIn, void* pDataIn2, void* pDataOut, INT64 valSize, INT64 itemSize, INT64 start, INT64 len, void* pDefault);


//-------------------------------------------------------------------
// T = data type as input type for values
// U = data type as index type
// thus <float, int32> converts a float to an int32
template<typename T, typename U>
class MergeBase {
public:
   MergeBase() {};
   ~MergeBase() {};

   //----------------------------------------------------
   // In parallel mode aValues DOES NOT change
   // aValues  : remains constant
   // aIndex   : incremented each call
   // aDataOut : incremented each call
   // start    : incremented each call
   static void MBGetInt(void* aValues, void* aIndex, void* aDataOut, INT64 valSize, INT64 itemSize, INT64 start, INT64 len, void* pDefault) {
      const T* pValues = (T*)aValues;
      const U* pIndex = (U*)aIndex;
      T* pDataOut = (T*)aDataOut;
      T  defaultVal = *(T*)pDefault;

      LOGGING("mbget sizes %lld  start:%lld  len: %lld   def: %lld  or  %lf\n", valSize, start, len, (INT64)defaultVal, (double)defaultVal);
      LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

      for (INT64 i = 0; i < len; i++)
      {
         const auto index = pIndex[i];
         pDataOut[i] =
            // Make sure the item is in range; if the index is negative -- but otherwise
            // still in range -- mimic Python's negative-indexing support.
            index >= -valSize && index < valSize
            ? pValues[index >= 0 ? index : index + valSize]

            // Index is out of range -- assign the invalid value.
            : defaultVal;
      }
   }

   //----------------------------------------------------------
   //
   static void MBGetIntU(void* aValues, void* aIndex, void* aDataOut, INT64 valSizeX, INT64 itemSize, INT64 start, INT64 len, void* pDefault) {
      const T* pValues = (T*)aValues;
      const U* pIndex = (U*)aIndex;
      T* pDataOut = (T*)aDataOut;
      T  defaultVal = *(T*)pDefault;

      UINT64 valSize = (UINT64)valSizeX;

      LOGGING("mbgetu sizes %lld  start:%lld  len: %lld   def: %lld  or  %lf\n", valSize, start, len, (INT64)defaultVal, (double)defaultVal);
      LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

      for (INT64 i = 0; i < len; i++) {
         const auto index = pIndex[i];
         pDataOut[i] =
            // Make sure the item is in range
            index >= 0 && index < valSize
            ? pValues[index]
            : defaultVal;
      }
   }


   //----------------------------------------------------------
   //
   static void MBGetIntF(void* aValues, void* aIndex, void* aDataOut, INT64 valSizeX, INT64 itemSize, INT64 start, INT64 len, void* pDefault) {
      const T* pValues = (T*)aValues;
      const U* pIndex = (U*)aIndex;
      T* pDataOut = (T*)aDataOut;
      T  defaultVal = *(T*)pDefault;

      UINT64 valSize = (UINT64)valSizeX;

      LOGGING("mbgetf sizes %lld  start:%lld  len: %lld   def: %lld  or  %lf\n", valSize, start, len, (INT64)defaultVal, (double)defaultVal);
      LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

      for (INT64 i = 0; i < len; i++) {
         // Make sure float is not fractional value
         const auto index = (INT64)pIndex[i];

         pDataOut[i] =
            // Make sure the item is in range
            (U)index == pIndex[i] && index >= 0 && index < (INT64)valSize
            ? pValues[index]
            : defaultVal;
      }
   }


   //----------------------------------------------------
   // In parallel mode aValues DOES NOT change
   // aValues  : remains constant
   // aIndex   : incremented each call
   // aDataOut : incremented each call
   // start    : incremented each call
   static void MBGetString(void* aValues, void* aIndex, void* aDataOut, INT64 valSize, INT64 itemSize, INT64 start, INT64 len, void* pDefault) {
      const char* pValues = (char*)aValues;
      const U* pIndex = (U*)aIndex;
      char* pDataOut = (char*)aDataOut;
      char* defaultVal = (char*)pDefault;

      LOGGING("mbget string sizes %lld  %lld %lld   \n", valSize, len, itemSize);
      //printf("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

      for (INT64 i = 0; i < len; i++) {
         const U index = pIndex[i];
         char* const dest = pDataOut + (i * itemSize);

         // Make sure the item is in range
         if (index >= -valSize && index < valSize) {
            // Handle Python-style negative indexing.
            const INT64 newIndex = index >= 0 ? index : index + valSize;
            const char* const src = pValues + (newIndex * itemSize);

            for (int j = 0; j < itemSize; j++) {
               dest[j] = src[j];
            }
         }
         else {
            // This is an out-of-bounds index -- set the result to the default value,
            // which for string types is all NUL (0x00) characters.
            // TODO: Consider using memset here instead -- need to benchmark before changing.
            for (int j = 0; j < itemSize; j++) {
               dest[j] = 0;
            }
         }
      }
   }


   //----------------------------------------------------
   // In parallel mode aValues DOES NOT change
   // aValues  : remains constant
   // aIndex   : incremented each call
   // aDataOut : incremented each call
   // start    : incremented each call
   static void MBGetStringU(void* aValues, void* aIndex, void* aDataOut, INT64 valSizeX, INT64 itemSize, INT64 start, INT64 len, void* pDefault) {
      const char* pValues = (char*)aValues;
      const U* pIndex = (U*)aIndex;
      char* pDataOut = (char*)aDataOut;
      char* defaultVal = (char*)pDefault;

      UINT64 valSize = (UINT64)valSizeX;

      LOGGING("mbgetu string sizes %lld  %lld %lld   \n", valSize, len, itemSize);
      //printf("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

      for (INT64 i = 0; i < len; i++) {
         U index = pIndex[i];
         char* const dest = pDataOut + (i*itemSize);

         // Make sure the item is in range
         if (index >= 0 && index < valSize) {
            const char* const src = pValues + (index * itemSize);

            for (int j = 0; j < itemSize; j++) {
               dest[j] = src[j];
            }
         }
         else {
            // This is an out-of-bounds index -- set the result to the default value,
            // which for string types is all NUL (0x00) characters.
            // TODO: Consider using memset here instead -- need to benchmark before changing.
            for (int j = 0; j < itemSize; j++) {
               dest[j] = 0;
            }
         }

      }
   }


   static void MBGetStringF(void* aValues, void* aIndex, void* aDataOut, INT64 valSizeX, INT64 itemSize, INT64 start, INT64 len, void* pDefault) {
      const char* pValues = (char*)aValues;
      const U* pIndex = (U*)aIndex;
      char* pDataOut = (char*)aDataOut;
      char* defaultVal = (char*)pDefault;

      UINT64 valSize = (UINT64)valSizeX;

      LOGGING("mbgetf string sizes %lld  %lld %lld   \n", valSize, len, itemSize);
      //printf("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

      for (INT64 i = 0; i < len; i++) {
         const INT64 index = (INT64)pIndex[i];
         char* const dest = pDataOut + (i*itemSize);

         // Make sure the item is in range
         if ((U)index == pIndex[i] && index >= 0 && index < (INT64)valSize) {
            const char* const src = pValues + (index * itemSize);

            for (int j = 0; j < itemSize; j++) {
               dest[j] = src[j];
            }
         }
         else {
            // This is an out-of-bounds index -- set the result to the default value,
            // which for string types is all NUL (0x00) characters.
            // TODO: Consider using memset here instead -- need to benchmark before changing.
            for (int j = 0; j < itemSize; j++) {
               dest[j] = 0;
            }
         }

      }
   }

   //static MBGET_FUNC  GetConversionFunction(int inputType, int func) {

   //   if (inputType == NPY_STRING) {
   //      return MBGetString;
   //   }

   //   return MBGetInt;
   //}


};


//------------------------------------------------------------
// inputType is Values type
// inputType2 is Index type
static MBGET_FUNC GetConversionFunction(int inputType, int inputType2, int func) {

   switch (inputType2) {
   case NPY_INT8:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::GetConversionFunction(inputType, func);
      case NPY_FLOAT:  return MergeBase<float, INT8>::MBGetInt;
      case NPY_DOUBLE: return MergeBase<double, INT8>::MBGetInt;
      case NPY_LONGDOUBLE: return MergeBase<long double, INT8>::MBGetInt;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, INT8>::MBGetInt;
      case NPY_INT16:  return MergeBase<INT16, INT8>::MBGetInt;
      CASE_NPY_INT32:  return MergeBase<INT32, INT8>::MBGetInt;
      CASE_NPY_INT64:  return MergeBase<INT64, INT8>::MBGetInt;
      case NPY_UINT8:  return MergeBase<UINT8, INT8>::MBGetInt;
      case NPY_UINT16: return MergeBase<UINT16, INT8>::MBGetInt;
      CASE_NPY_UINT32: return MergeBase<UINT32, INT8>::MBGetInt;
      CASE_NPY_UINT64: return MergeBase<UINT64, INT8>::MBGetInt;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, INT8>::MBGetString;

      }
      break;

   case NPY_INT16:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::MBGetInt;
      case NPY_FLOAT:  return MergeBase<float, INT16>::MBGetInt;
      case NPY_DOUBLE: return MergeBase<double, INT16>::MBGetInt;
      case NPY_LONGDOUBLE: return MergeBase<long double, INT16>::MBGetInt;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, INT16>::MBGetInt;
      case NPY_INT16:  return MergeBase<INT16, INT16>::MBGetInt;
      CASE_NPY_INT32:  return MergeBase<INT32, INT16>::MBGetInt;
      CASE_NPY_INT64:  return MergeBase<INT64, INT16>::MBGetInt;
      case NPY_UINT8:  return MergeBase<UINT8, INT16>::MBGetInt;
      case NPY_UINT16: return MergeBase<UINT16, INT16>::MBGetInt;
      CASE_NPY_UINT32: return MergeBase<UINT32, INT16>::MBGetInt;
      CASE_NPY_UINT64: return MergeBase<UINT64, INT16>::MBGetInt;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, INT16>::MBGetString;

      }
      break;

   CASE_NPY_INT32:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::MBGetInt;
      case NPY_FLOAT:  return MergeBase<float, INT32>::MBGetInt;
      case NPY_DOUBLE: return MergeBase<double, INT32>::MBGetInt;
      case NPY_LONGDOUBLE: return MergeBase<long double, INT32>::MBGetInt;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, INT32>::MBGetInt;
      case NPY_INT16:  return MergeBase<INT16, INT32>::MBGetInt;
      CASE_NPY_INT32:  return MergeBase<INT32, INT32>::MBGetInt;
      CASE_NPY_INT64:  return MergeBase<INT64, INT32>::MBGetInt;
      case NPY_UINT8:  return MergeBase<UINT8, INT32>::MBGetInt;
      case NPY_UINT16: return MergeBase<UINT16, INT32>::MBGetInt;
      CASE_NPY_UINT32: return MergeBase<UINT32, INT32>::MBGetInt;
      CASE_NPY_UINT64: return MergeBase<UINT64, INT32>::MBGetInt;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, INT32>::MBGetString;

      }
      break;

   CASE_NPY_INT64:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::MBGetInt;
      case NPY_FLOAT:  return MergeBase<float, INT64>::MBGetInt;
      case NPY_DOUBLE: return MergeBase<double, INT64>::MBGetInt;
      case NPY_LONGDOUBLE: return MergeBase<long double, INT64>::MBGetInt;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, INT64>::MBGetInt;
      case NPY_INT16:  return MergeBase<INT16, INT64>::MBGetInt;
      CASE_NPY_INT32:  return MergeBase<INT32, INT64>::MBGetInt;
      CASE_NPY_INT64:  return MergeBase<INT64, INT64>::MBGetInt;
      case NPY_UINT8:  return MergeBase<UINT8, INT64>::MBGetInt;
      case NPY_UINT16: return MergeBase<UINT16, INT64>::MBGetInt;
      CASE_NPY_UINT32: return MergeBase<UINT32, INT64>::MBGetInt;
      CASE_NPY_UINT64: return MergeBase<UINT64, INT64>::MBGetInt;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, INT64>::MBGetString;
      }
      break;


   case NPY_UINT8:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::GetConversionFunction(inputType, func);
      case NPY_FLOAT:  return MergeBase<float, UINT8>::MBGetIntU;
      case NPY_DOUBLE: return MergeBase<double, UINT8>::MBGetIntU;
      case NPY_LONGDOUBLE: return MergeBase<long double, UINT8>::MBGetIntU;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, UINT8>::MBGetIntU;
      case NPY_INT16:  return MergeBase<INT16, UINT8>::MBGetIntU;
      CASE_NPY_INT32:  return MergeBase<INT32, UINT8>::MBGetIntU;
      CASE_NPY_INT64:  return MergeBase<INT64, UINT8>::MBGetIntU;
      case NPY_UINT8:  return MergeBase<UINT8, UINT8>::MBGetIntU;
      case NPY_UINT16: return MergeBase<UINT16, UINT8>::MBGetIntU;
      CASE_NPY_UINT32: return MergeBase<UINT32, UINT8>::MBGetIntU;
      CASE_NPY_UINT64: return MergeBase<UINT64, UINT8>::MBGetIntU;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, UINT8>::MBGetStringU;

      }
      break;

   case NPY_UINT16:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::MBGetInt;
      case NPY_FLOAT:  return MergeBase<float, UINT16>::MBGetIntU;
      case NPY_DOUBLE: return MergeBase<double, UINT16>::MBGetIntU;
      case NPY_LONGDOUBLE: return MergeBase<long double, UINT16>::MBGetIntU;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, UINT16>::MBGetIntU;
      case NPY_INT16:  return MergeBase<INT16, UINT16>::MBGetIntU;
      CASE_NPY_INT32:  return MergeBase<INT32, UINT16>::MBGetIntU;
      CASE_NPY_INT64:  return MergeBase<INT64, UINT16>::MBGetIntU;
      case NPY_UINT8:  return MergeBase<UINT8, UINT16>::MBGetIntU;
      case NPY_UINT16: return MergeBase<UINT16, UINT16>::MBGetIntU;
      CASE_NPY_UINT32: return MergeBase<UINT32, UINT16>::MBGetIntU;
      CASE_NPY_UINT64: return MergeBase<UINT64, UINT16>::MBGetIntU;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, UINT16>::MBGetStringU;

      }
      break;

   CASE_NPY_UINT32:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::MBGetInt;
      case NPY_FLOAT:  return MergeBase<float, UINT32>::MBGetIntU;
      case NPY_DOUBLE: return MergeBase<double, UINT32>::MBGetIntU;
      case NPY_LONGDOUBLE: return MergeBase<long double, UINT32>::MBGetIntU;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, UINT32>::MBGetIntU;
      case NPY_INT16:  return MergeBase<INT16, UINT32>::MBGetIntU;
      CASE_NPY_INT32:  return MergeBase<INT32, UINT32>::MBGetIntU;
      CASE_NPY_INT64:  return MergeBase<INT64, UINT32>::MBGetIntU;
      case NPY_UINT8:  return MergeBase<UINT8, UINT32>::MBGetIntU;
      case NPY_UINT16: return MergeBase<UINT16, UINT32>::MBGetIntU;
      CASE_NPY_UINT32: return MergeBase<UINT32, UINT32>::MBGetIntU;
      CASE_NPY_UINT64: return MergeBase<UINT64, UINT32>::MBGetIntU;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, UINT32>::MBGetStringU;

      }
      break;

   CASE_NPY_UINT64:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::MBGetInt;
      case NPY_FLOAT:  return MergeBase<float, UINT64>::MBGetIntU;
      case NPY_DOUBLE: return MergeBase<double, UINT64>::MBGetIntU;
      case NPY_LONGDOUBLE: return MergeBase<long double, UINT64>::MBGetIntU;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, UINT64>::MBGetIntU;
      case NPY_INT16:  return MergeBase<INT16, UINT64>::MBGetIntU;
      CASE_NPY_INT32:  return MergeBase<INT32, UINT64>::MBGetIntU;
      CASE_NPY_INT64:  return MergeBase<INT64, UINT64>::MBGetIntU;
      case NPY_UINT8:  return MergeBase<UINT8, UINT64>::MBGetIntU;
      case NPY_UINT16: return MergeBase<UINT16, UINT64>::MBGetIntU;
      CASE_NPY_UINT32: return MergeBase<UINT32, UINT64>::MBGetIntU;
      CASE_NPY_UINT64: return MergeBase<UINT64, UINT64>::MBGetIntU;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, UINT64>::MBGetStringU;
      }
      break;


   case NPY_FLOAT32:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::MBGetInt;
      case NPY_FLOAT:  return MergeBase<float, float>::MBGetIntF;
      case NPY_DOUBLE: return MergeBase<double, float>::MBGetIntF;
      case NPY_LONGDOUBLE: return MergeBase<long double, float>::MBGetIntF;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, float>::MBGetIntF;
      case NPY_INT16:  return MergeBase<INT16, float>::MBGetIntF;
      CASE_NPY_INT32:  return MergeBase<INT32, float>::MBGetIntF;
      CASE_NPY_INT64:  return MergeBase<INT64, float>::MBGetIntF;
      case NPY_UINT8:  return MergeBase<UINT8, float>::MBGetIntF;
      case NPY_UINT16: return MergeBase<UINT16, float>::MBGetIntF;
      CASE_NPY_UINT32: return MergeBase<UINT32, float>::MBGetIntF;
      CASE_NPY_UINT64: return MergeBase<UINT64, float>::MBGetIntF;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, float>::MBGetStringF;

      }
      break;

   case NPY_FLOAT64:
      switch (inputType) {
         //case NPY_BOOL:   return MergeBase<bool, bool>::MBGetInt;
      case NPY_FLOAT:  return MergeBase<float, UINT64>::MBGetIntF;
      case NPY_DOUBLE: return MergeBase<double, UINT64>::MBGetIntF;
      case NPY_LONGDOUBLE: return MergeBase<long double, UINT64>::MBGetIntF;
      case NPY_BOOL:
      case NPY_BYTE:   return MergeBase<INT8, UINT64>::MBGetIntF;
      case NPY_INT16:  return MergeBase<INT16, UINT64>::MBGetIntF;
      CASE_NPY_INT32:  return MergeBase<INT32, UINT64>::MBGetIntF;
      CASE_NPY_INT64:  return MergeBase<INT64, UINT64>::MBGetIntF;
      case NPY_UINT8:  return MergeBase<UINT8, UINT64>::MBGetIntF;
      case NPY_UINT16: return MergeBase<UINT16, UINT64>::MBGetIntF;
      CASE_NPY_UINT32: return MergeBase<UINT32, UINT64>::MBGetIntF;
      CASE_NPY_UINT64: return MergeBase<UINT64, UINT64>::MBGetIntF;
      case NPY_VOID:
      case NPY_UNICODE:
      case NPY_STRING: return MergeBase<char*, UINT64>::MBGetStringF;
      }
      break;

   }
   printf("mbget cannot find type for %d\n", inputType);
   return NULL;
}



struct MBGET_CALLBACK {
   MBGET_FUNC MBGetCallback;

   void*    pValues;
   void*    pIndex;
   void*    pDataOut;
   INT64    valSize1;
   INT64    aIndexSize;
   void*    pDefault;
   INT64    TypeSizeValues;
   INT64    TypeSizeIndex;

} stMBGCallback;


//---------------------------------------------------------
// Used by MBGet
//  Concurrent callback from multiple threads
static BOOL AnyMBGet(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex) {

   BOOL didSomeWork = FALSE;
   MBGET_CALLBACK* Callback = &stMBGCallback; // (MBGET_CALLBACK*)&pstWorkerItem->WorkCallbackArg;

   char* aValues = (char *)Callback->pValues;
   char* aIndex = (char *)Callback->pIndex;

   INT64 typeSizeValues = Callback->TypeSizeValues;
   INT64 typeSizeIndex = Callback->TypeSizeIndex;

   LOGGING("check2 ** %lld %lld\n", typeSizeValues, typeSizeIndex);

   INT64 lenX;
   INT64 workBlock;

   // As long as there is work to do
   while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

      // Do NOT move aValues
      // Move aIndex
      // Move pDataOut (same type as Values)
      // move starting position

      // Calculate how much to adjust the pointers to get to the data for this work block
      INT64 blockStart = workBlock * pstWorkerItem->BlockSize;

      INT64 valueAdj = blockStart * typeSizeValues;
      INT64 indexAdj = blockStart * typeSizeIndex;

      LOGGING("%d : workBlock %lld   blocksize: %lld    lenx: %lld  %lld  %lld  %lld %lld\n", core, workBlock, pstWorkerItem->BlockSize, lenX, typeSizeValues, typeSizeIndex, valueAdj, indexAdj);

      Callback->MBGetCallback(aValues, aIndex + indexAdj, (char*)Callback->pDataOut + valueAdj, Callback->valSize1, typeSizeValues, blockStart, lenX, Callback->pDefault);

      // Indicate we completed a block
      didSomeWork = TRUE;

      // tell others we completed this work block
      pstWorkerItem->CompleteWorkBlock();
   }

   return didSomeWork;
}





//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aValues (can be anything)
// Arg2: numpy array aIndex (must be int8/int16/int32 or int64)
// Arg3: default value
//
//def fixMbget(aValues, aIndex, result, default) :
//   """
//   A numba routine to speed up mbget for numerical values.
//   """
//   N = aIndex.shape[0]
//   valSize = aValues.shape[0]
//   for i in range(N) :
//      if (aIndex[i] >= 0 and aIndex[i] < valSize) :
//         result[i] = aValues[aIndex[i]]
//      else :
//   result[i] = default
PyObject *
MBGet(PyObject *self, PyObject *args)
{
   PyArrayObject *aValues = NULL;
   PyArrayObject *aIndex = NULL;
   PyObject* defaultValue = NULL;

   if (PyTuple_GET_SIZE(args) == 2) {
      if (!PyArg_ParseTuple(
         args, "O!O!:MBGet",
         &PyArray_Type, &aValues,
         &PyArray_Type, &aIndex
         )) {

         return NULL;
      }
      defaultValue = Py_None;

   } else
   if (!PyArg_ParseTuple(
      args, "O!O!O:MBGet",
      &PyArray_Type, &aValues,
      &PyArray_Type, &aIndex,

      &defaultValue)) {

      return NULL;
   }

   INT32 numpyValuesType = ObjectToDtype(aValues);
   INT32 numpyIndexType = ObjectToDtype(aIndex);

   if (numpyValuesType < 0 || numpyIndexType < 0) {
      PyErr_Format(PyExc_ValueError, "Dont know how to convert these types %d", numpyValuesType);
      return NULL;
   }

   //printf("numpy types %d %d\n", numpyValuesType, numpyIndexType);

   void* pValues = PyArray_BYTES(aValues);
   void* pIndex = PyArray_BYTES(aIndex);

   int ndim = PyArray_NDIM(aValues);
   npy_intp* dims = PyArray_DIMS(aValues);
   INT64 valSize1 = CalcArrayLength(ndim, dims);
   INT64 len = valSize1;

   MBGET_FUNC  pFunction = GetConversionFunction(numpyValuesType, numpyIndexType, 0);

   if (pFunction != NULL) {

      PyArrayObject* outArray = (PyArrayObject*)Py_None;
      INT64 aIndexSize = ArrayLength(aIndex);

      // Allocate the size of aIndex but the type is the value
      outArray = AllocateLikeResize(aValues, aIndexSize);

      if (outArray) {
         void* pDataOut = PyArray_BYTES(outArray);
         void* pDefault = GetDefaultForType(numpyValuesType);

         // reserve a full 16 bytes for default in case we have oneS
         _m256all tempDefault;

         // Check if a default value was passed in as third parameter
         if (defaultValue != Py_None) {
            BOOL result;
            INT64 itemSize;
            void* pTempData = NULL;

            // Try to convert the scalar
            result = ConvertScalarObject(defaultValue, &tempDefault, numpyValuesType, &pTempData, &itemSize);

            if (result) {
               // Assign the new default for out of range indexes
               pDefault = &tempDefault;
            }
         }

         stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(aIndexSize);

         if (pWorkItem == NULL) {

            // Threading not allowed for this work item, call it directly from main thread
            pFunction(pValues, pIndex, pDataOut, valSize1, PyArray_ITEMSIZE(aValues), 0, aIndexSize, pDefault);

         }
         else {
            // Each thread will call this routine with the callbackArg
            pWorkItem->DoWorkCallback = AnyMBGet;

            pWorkItem->WorkCallbackArg = &stMBGCallback;

            stMBGCallback.MBGetCallback = pFunction;
            stMBGCallback.pValues = pValues;
            stMBGCallback.pIndex = pIndex;
            stMBGCallback.pDataOut = pDataOut;

            // arraylength of values input array -- used to check array bounds
            stMBGCallback.valSize1 = valSize1;
            stMBGCallback.aIndexSize = aIndexSize;
            stMBGCallback.pDefault = pDefault;

            //
            stMBGCallback.TypeSizeValues = PyArray_ITEMSIZE(aValues);
            stMBGCallback.TypeSizeIndex = PyArray_ITEMSIZE(aIndex);

            //printf("**check %p %p %p %lld %lld\n", pValues, pIndex, pDataOut, stMBGCallback.TypeSizeValues, stMBGCallback.TypeSizeIndex);

            // This will notify the worker threads of a new work item
            g_cMathWorker->WorkMain(pWorkItem, aIndexSize, 0);
            //g_cMathWorker->WorkMain(pWorkItem, aIndexSize);
         }

         return (PyObject*)outArray;
      }
      PyErr_Format(PyExc_ValueError, "MBGet ran out of memory %d %d", numpyValuesType, numpyIndexType);
      return NULL;

   }

   PyErr_Format(PyExc_ValueError, "Dont know how to convert these types %d %d", numpyValuesType, numpyIndexType);
   return NULL;
}

//===================================================
// Input: boolean array
// Output: chunk count and ppChunkCount
// NOTE: CALLER MUST FREE pChunkCount
//
INT64 BooleanCount(PyArrayObject* aIndex, INT64** ppChunkCount) {

   // Pass one, count the values
   // Eight at a time
   const INT64 lengthBool = ArrayLength(aIndex);
   const INT8* const pBooleanMask = (INT8*)PyArray_BYTES(aIndex);

   // Count the number of chunks (of boolean elements).
   // It's important we handle the case of an empty array (zero length) when determining the number
   // of per-chunk counts to return; the behavior of malloc'ing zero bytes is undefined, and the code
   // below assumes there's always at least one entry in the count-per-chunk array. If we don't handle
   // the empty array case we'll allocate an empty count-per-chunk array and end up doing an
   // out-of-bounds write.
   const INT64 chunkSize = g_cMathWorker->WORK_ITEM_CHUNK;
   const INT64 chunks = (std::max(lengthBool, 1LL) + (chunkSize - 1)) / chunkSize;

   // TOOD: divide up per core instead
   INT64* const pChunkCount = (INT64*)WORKSPACE_ALLOC(chunks * sizeof(INT64));


   // MT callback
   struct BSCallbackStruct {
      INT64* pChunkCount;
      const INT8*  pBooleanMask;
   };

   // This is the routine that will be called back from multiple threads
   auto lambdaBSCallback = [](void* callbackArgT, int core, INT64 start, INT64 length) -> BOOL {
      BSCallbackStruct* callbackArg = (BSCallbackStruct*)callbackArgT;

      const INT8*  pBooleanMask = callbackArg->pBooleanMask;
      INT64* pChunkCount = callbackArg->pChunkCount;

      // Use the single-threaded implementation to sum the number of
      // 1-byte boolean TRUE values in the current chunk.
      // This means the current function is just responsible for parallelizing over the chunks
      // but doesn't do any real "math" itself.
      INT64 total = SumBooleanMask(&pBooleanMask[start], length);

      pChunkCount[start / g_cMathWorker->WORK_ITEM_CHUNK] = total;
      return TRUE;
   };

   BSCallbackStruct stBSCallback;
   stBSCallback.pChunkCount = pChunkCount;
   stBSCallback.pBooleanMask = pBooleanMask;

   BOOL didMtWork = g_cMathWorker->DoMultiThreadedChunkWork(lengthBool, lambdaBSCallback, &stBSCallback);


   *ppChunkCount = pChunkCount;
   // if multithreading turned off...
   return didMtWork ? chunks : 1;
}



//===============================================================================
// checks for kwargs 'both'
// if exists, and is True return True
BOOL GetKwargBoth(PyObject *kwargs) {
   // Check for cutoffs kwarg to see if going into parallel mode
   if (kwargs && PyDict_Check(kwargs)) {
      PyObject* pBoth = NULL;
      // Borrowed reference
      // Returns NULL if key not present
      pBoth = PyDict_GetItemString(kwargs, "both");

      if (pBoth != NULL && pBoth == Py_True) {
         return TRUE;
      }
   }
   return FALSE;
}

//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aIndex (must be BOOL)
// Kwarg: "both"
//
// Returns: fancy index array where the true values are
// if 'both' is set to True it returns an index array which has both True and False
// if 'both' is set, the number of True values is also returned
PyObject *
BooleanToFancy(PyObject *self, PyObject *args, PyObject *kwargs)
{
   PyArrayObject *aIndex = NULL;

   if (!PyArg_ParseTuple(
      args, "O!",
      &PyArray_Type, &aIndex
   )) {

      return NULL;
   }

   if (PyArray_TYPE(aIndex) != NPY_BOOL) {
      PyErr_Format(PyExc_ValueError, "First argument must be boolean array");
      return NULL;
   }

   // if bothMode is set, will return fancy index for both True and False
   BOOL     bothMode = GetKwargBoth(kwargs);

   INT64*   pChunkCount = NULL;
   INT64*   pChunkCountFalse = NULL;
   INT64    chunks = BooleanCount(aIndex, &pChunkCount);
   INT64    indexLength = ArrayLength(aIndex);

   INT64    totalTrue = 0;

   if (bothMode) {
      // now count up the chunks
      // TJD: April 2019 note -- when the chunk size is between 65536 and 128000
      // it is really one chunk, but the code still works
      INT64 chunkSize = g_cMathWorker->WORK_ITEM_CHUNK;
      INT64 chunks = (indexLength + (chunkSize - 1)) / chunkSize;

      // Also need false count
      pChunkCountFalse = (INT64*)WORKSPACE_ALLOC(chunks * sizeof(INT64));
      INT64 totalFalse = 0;

      // Store the offset
      for (INT64 i = 0; i < chunks; i++) {
         INT64 temp = totalFalse;

         // check for last chunk
         totalFalse += (chunkSize - pChunkCount[i]);

         // reassign to the cumulative sum so we know the offset
         pChunkCountFalse[i] = temp;
      }

      //printf("both mode - chunks: %lld  totalTrue: %lld   toatlFalse: %lld\n", chunks, pChunkCount[0], pChunkCountFalse[0]);
   }

   // Store the offset
   for (INT64 i = 0; i < chunks; i++) {
      INT64 temp = totalTrue;
      totalTrue += pChunkCount[i];

      // reassign to the cumulative sum so we know the offset
      pChunkCount[i] = temp;
   }


   PyArrayObject* returnArray = NULL;
   int dtype = NPY_INT64;
   // INT32 or INT64
   if (indexLength < 2000000000) {
      dtype = NPY_INT32;
   }

   if (bothMode) {
      // Allocate for both True and False
      returnArray = AllocateNumpyArray(1, (npy_intp*)&indexLength, dtype);
   }
   else {
      // INT32 or INT64
      returnArray = AllocateNumpyArray(1, (npy_intp*)&totalTrue, dtype);
   }

   CHECK_MEMORY_ERROR(returnArray);

   if (returnArray) {
      // MT callback
      struct BTFCallbackStruct {
         INT64*   pChunkCount;
         INT64*   pChunkCountFalse;
         INT8*    pBooleanMask;
         void*    pValuesOut;
         INT64    totalTrue;
         int      dtype;
         BOOL     bothMode;
      };

      // This is the routine that will be called back from multiple threads
      auto lambdaCallback = [](void* callbackArgT, int core, INT64 start, INT64 length) -> BOOL {
         BTFCallbackStruct* callbackArg = (BTFCallbackStruct*)callbackArgT;

         INT64  chunkCount = callbackArg->pChunkCount[start / g_cMathWorker->WORK_ITEM_CHUNK];
         INT8*  pBooleanMask = callbackArg->pBooleanMask;
         BOOL   bothMode = callbackArg->bothMode;

         if (bothMode) {
            INT64  chunkCountFalse = callbackArg->pChunkCountFalse[start / g_cMathWorker->WORK_ITEM_CHUNK];
            //printf("[%lld] ccf %lld  length %lld\n", start, chunkCountFalse, length);

            if (callbackArg->dtype == NPY_INT64) {
               INT64* pOut = (INT64*)callbackArg->pValuesOut;
               pOut = pOut + chunkCount;

               INT64* pOutFalse = (INT64*)callbackArg->pValuesOut;
               pOutFalse = pOutFalse + callbackArg->totalTrue + chunkCountFalse;

               for (INT64 i = start; i < (start + length); i++) {
                  if (pBooleanMask[i]) {
                     *pOut++ = i;
                  }
                  else {
                     *pOutFalse++ = i;
                  }
               }
            }
            else {
               INT32* pOut = (INT32*)callbackArg->pValuesOut;
               pOut = pOut + chunkCount;

               INT32* pOutFalse = (INT32*)callbackArg->pValuesOut;
               pOutFalse = pOutFalse + callbackArg->totalTrue + chunkCountFalse;

               for (INT64 i = start; i < (start + length); i++) {
                  if (pBooleanMask[i]) {
                     *pOut++ = (INT32)i;
                  }
                  else {
                     *pOutFalse++ = (INT32)i;
                  }
               }

            }
         }
         else {

            if (callbackArg->dtype == NPY_INT64) {
               INT64* pOut = (INT64*)callbackArg->pValuesOut;
               pOut = pOut + chunkCount;

               for (INT64 i = start; i < (start + length); i++) {
                  if (pBooleanMask[i]) {
                     *pOut++ = i;
                  }
               }
            }
            else {
               INT32* pOut = (INT32*)callbackArg->pValuesOut;
               pOut = pOut + chunkCount;

               for (INT64 i = start; i < (start + length); i++) {
                  if (pBooleanMask[i]) {
                     *pOut++ = (INT32)i;
                  }
               }

            }
         }

         return TRUE;
      };

      BTFCallbackStruct stBTFCallback;
      stBTFCallback.pChunkCount = pChunkCount;
      stBTFCallback.pChunkCountFalse = pChunkCountFalse;
      stBTFCallback.pBooleanMask = (INT8*)PyArray_BYTES(aIndex);
      stBTFCallback.pValuesOut = (INT64*)PyArray_BYTES(returnArray);
      stBTFCallback.dtype = dtype;
      stBTFCallback.totalTrue = totalTrue;
      stBTFCallback.bothMode = bothMode;

      g_cMathWorker->DoMultiThreadedChunkWork(indexLength, lambdaCallback, &stBTFCallback);

   }

   //_mm_i32gather_epi32

   WORKSPACE_FREE(pChunkCount);
   if (pChunkCountFalse) {
      WORKSPACE_FREE(pChunkCountFalse);
   }
   if (bothMode) {
      // also return the true count so user knows cutoff
      PyObject* returnTuple = PyTuple_New(2);
      PyTuple_SET_ITEM(returnTuple, 0, (PyObject*)returnArray);
      PyTuple_SET_ITEM(returnTuple, 1, PyLong_FromSize_t(totalTrue));
      return returnTuple;
   }
   return (PyObject*)returnArray;
}


//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aIndex (must be BOOL)
// Returns: how many true values there are
// NOTE: faster than calling sum
PyObject *
BooleanSum(PyObject *self, PyObject *args)
{
   PyArrayObject *aIndex = NULL;

   if (!PyArg_ParseTuple(
      args, "O!",
      &PyArray_Type, &aIndex
   )) {

      return NULL;
   }

   if (PyArray_TYPE(aIndex) != NPY_BOOL) {
      PyErr_Format(PyExc_ValueError, "First argument must be boolean array");
      return NULL;
   }

   INT64*   pChunkCount = NULL;
   INT64 chunks = BooleanCount(aIndex, &pChunkCount);

   INT64 totalTrue = 0;
   for (INT64 i = 0; i < chunks; i++) {
      totalTrue += pChunkCount[i];
   }

   WORKSPACE_FREE(pChunkCount);
   return PyLong_FromSize_t(totalTrue);

}



//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aValues (can be anything)
// Arg2: numpy array aIndex (must be BOOL)
//
PyObject *
BooleanIndex(PyObject *self, PyObject *args)
{
   PyArrayObject *aValues = NULL;
   PyArrayObject *aIndex = NULL;

   if (!PyArg_ParseTuple(
      args, "O!O!",
      &PyArray_Type, &aValues,
      &PyArray_Type, &aIndex
   )) {

      return NULL;
   }

   if (PyArray_TYPE(aIndex) != NPY_BOOL) {
      PyErr_Format(PyExc_ValueError, "Second argument must be boolean array");
      return NULL;
   }

   // Pass one, count the values
   // Eight at a time
   INT64 lengthBool = ArrayLength(aIndex);
   INT64 lengthValue = ArrayLength(aValues);

   if (lengthBool != lengthValue) {
      PyErr_Format(PyExc_ValueError, "Array lengths must match %lld vs %lld", lengthBool, lengthValue);
      return NULL;
   }

   INT64*   pChunkCount = NULL;
   INT64    chunks = BooleanCount(aIndex, &pChunkCount);

   INT64 totalTrue = 0;

   // Store the offset
   for (INT64 i = 0; i < chunks; i++) {
      INT64 temp = totalTrue;
      totalTrue += pChunkCount[i];

      // reassign to the cumulative sum so we know the offset
      pChunkCount[i] = temp;
   }

   LOGGING("boolindex total: %I64d  length: %I64d  type:%d\n", totalTrue, lengthBool, PyArray_TYPE(aValues));

   INT8* pBooleanMask = (INT8*)PyArray_BYTES(aIndex);


   // Now we know per chunk how many true there are... we can allocate the new array
   PyArrayObject* pReturnArray = AllocateLikeResize(aValues, totalTrue);

   if (pReturnArray) {

      // MT callback
      struct BICallbackStruct {
         INT64*   pChunkCount;
         INT8*    pBooleanMask;
         char*    pValuesIn;
         char*    pValuesOut;
         INT64    itemSize;
      };


      //-----------------------------------------------
      //-----------------------------------------------
      // This is the routine that will be called back from multiple threads
      auto lambdaBICallback2 = [](void* callbackArgT, int core, INT64 start, INT64 length) -> BOOL {
         BICallbackStruct* callbackArg = (BICallbackStruct*)callbackArgT;

         INT8*  pBooleanMask = callbackArg->pBooleanMask;
         INT64* pData = (INT64*)&pBooleanMask[start];
         INT64  chunkCount = callbackArg->pChunkCount[start / g_cMathWorker->WORK_ITEM_CHUNK];
         INT64  itemSize = callbackArg->itemSize;
         char*  pValuesIn = &callbackArg->pValuesIn[start * itemSize];
         char*  pValuesOut = &callbackArg->pValuesOut[chunkCount * itemSize];

         INT64  blength = length / 8;

         switch (itemSize) {
         case 1:
         {
            INT8* pVOut = (INT8*)pValuesOut;
            INT8* pVIn = (INT8*)pValuesIn;

            for (INT64 i = 0; i < blength; i++) {

               // little endian, so the first value is low bit (not high bit)
               UINT32 bitmask = (UINT32)(_pext_u64(*pData, 0x0101010101010101));
               if (bitmask != 0) {
                  if (bitmask & 1) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 2) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 4) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 8) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 16) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 32) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 64) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 128) { *pVOut++ = *pVIn; } pVIn++;
               }
               else {
                  pVIn += 8;
               }
               pData++;
            }

            // Get last
            pBooleanMask = (INT8*)pData;

            blength = length & 7;
            for (INT64 i = 0; i < blength; i++) {
               if (*pBooleanMask++) {
                  *pVOut++ = *pVIn;
               }
               pVIn++;
            }
         }
         break;
         case 2:
         {
            INT16* pVOut = (INT16*)pValuesOut;
            INT16* pVIn = (INT16*)pValuesIn;

            for (INT64 i = 0; i < blength; i++) {

               // little endian, so the first value is low bit (not high bit)
               UINT32 bitmask = (UINT32)(_pext_u64(*pData, 0x0101010101010101));
               if (bitmask != 0) {
                  if (bitmask & 1) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 2) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 4) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 8) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 16) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 32) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 64) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 128) { *pVOut++ = *pVIn; } pVIn++;
               }
               else {
                  pVIn += 8;
               }
               pData++;
            }

            // Get last
            pBooleanMask = (INT8*)pData;

            blength = length & 7;
            for (INT64 i = 0; i < blength; i++) {
               if (*pBooleanMask++) {
                  *pVOut++ = *pVIn;
               }
               pVIn++;
            }
         }
         break;
         case 4:
         {
            INT32* pVOut = (INT32*)pValuesOut;
            INT32* pVIn = (INT32*)pValuesIn;

            for (INT64 i = 0; i < blength; i++) {

               // little endian, so the first value is low bit (not high bit)
               UINT32 bitmask = (UINT32)(_pext_u64(*pData, 0x0101010101010101));
               if (bitmask != 0) {
                  if (bitmask & 1) { *pVOut++ = *pVIn;} pVIn++;
                  if (bitmask & 2) { *pVOut++ = *pVIn;} pVIn++;
                  if (bitmask & 4) { *pVOut++ = *pVIn;} pVIn++;
                  if (bitmask & 8) { *pVOut++ = *pVIn;} pVIn++;
                  if (bitmask & 16){ *pVOut++ = *pVIn;} pVIn++;
                  if (bitmask & 32){ *pVOut++ = *pVIn;} pVIn++;
                  if (bitmask & 64){ *pVOut++ = *pVIn;} pVIn++;
                  if (bitmask & 128){*pVOut++ = *pVIn;} pVIn++;
               }
               else {
                  pVIn += 8;
               }
               pData++;
            }

            // Get last
            pBooleanMask = (INT8*)pData;

            blength = length & 7;
            for (INT64 i = 0; i < blength; i++) {
               if (*pBooleanMask++) {
                  *pVOut++ = *pVIn;
               }
               pVIn++;
            }
         }
         break;
         case 8:
         {
            INT64* pVOut = (INT64*)pValuesOut;
            INT64* pVIn = (INT64*)pValuesIn;

            for (INT64 i = 0; i < blength; i++) {

               // little endian, so the first value is low bit (not high bit)
               UINT32 bitmask = (UINT32)(_pext_u64(*pData, 0x0101010101010101));
               if (bitmask != 0) {
                  if (bitmask & 1) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 2) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 4) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 8) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 16) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 32) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 64) { *pVOut++ = *pVIn; } pVIn++;
                  if (bitmask & 128) { *pVOut++ = *pVIn; } pVIn++;
               }
               else {
                  pVIn += 8;
               }
               pData++;
            }

            // Get last
            pBooleanMask = (INT8*)pData;

            blength = length & 7;
            for (INT64 i = 0; i < blength; i++) {
               if (*pBooleanMask++) {
                  *pVOut++ = *pVIn;
               }
               pVIn++;
            }
         }
         break;

         default:
         {
            for (INT64 i = 0; i < blength; i++) {

               // little endian, so the first value is low bit (not high bit)
               UINT32 bitmask = (UINT32)(_pext_u64(*pData, 0x0101010101010101));
               if (bitmask != 0) {
                  int counter = 8;
                  while (counter--) {
                     if (bitmask & 1) {
                        memcpy(pValuesOut, pValuesIn, itemSize);
                        pValuesOut += itemSize;
                     }

                     pValuesIn += itemSize;
                     bitmask >>= 1;
                  }
               }
               else {
                  pValuesIn += (itemSize * 8);
               }
               pData++;
            }

            // Get last
            pBooleanMask = (INT8*)pData;

            blength = length & 7;
            for (INT64 i = 0; i < blength; i++) {
               if (*pBooleanMask++) {
                  memcpy(pValuesOut, pValuesIn, itemSize);
                  pValuesOut += itemSize;
               }
               pValuesIn += itemSize;
            }
         }
         break;
         }

         return TRUE;
      };

      BICallbackStruct stBICallback;
      stBICallback.pChunkCount = pChunkCount;
      stBICallback.pBooleanMask = pBooleanMask;
      stBICallback.pValuesIn = (char*)PyArray_BYTES(aValues);
      stBICallback.pValuesOut = (char*)PyArray_BYTES(pReturnArray);
      stBICallback.itemSize = PyArray_ITEMSIZE(aValues);

      g_cMathWorker->DoMultiThreadedChunkWork(lengthBool, lambdaBICallback2, &stBICallback);
   }

   WORKSPACE_FREE(pChunkCount);
   return (PyObject*)pReturnArray;
}

//
//
//#--------- START OF C++ ROUTINE -------------
//#based on how many uniques we have, allocate the new ikey
//# do we have a routine for this ?
//uikey_length = len(uikey)
//if uikey_length < 100 :
//   dtype = np.int8
//   elif uikey_length < 30_000 :
//   dtype = np.int16
//   elif uikey_length < 2_000_000_000 :
//   dtype = np.int32
//else:
//dtype = np.int64
//
//newikey = empty((len(ikey), ), dtype = dtype)
//
//start = 0
//starti = 0
//for i in range(len(u_cutoffs)) :
//   stop = u_cutoffs[i]
//   stopi = i_cutoffs[i]
//   uikey_slice = uikey[start:stop]
//   oldikey_slice = ikey[starti:stopi]
//
//   if verbose:
//      print("fixing ", starti, stopi)
//      print("newikey ", newikey)
//      print("oldikey_slice ", oldikey_slice)
//
//   if base_index == 1 :
//      # write a routine for this in C++
//      # if 0 and base_index=1, then keep the 0
//      filtermask = oldikey_slice == 0
//      newikey[starti:stopi] = uikey_slice[oldikey_slice - 1]
//      if filtermask.sum() > 0:
//         newikey[starti:stopi][filtermask] = 0
//   else:
//         newikey[starti:stopi] = uikey_slice[oldikey_slice]
//
//start = stop
//starti = stopi
//#END C++ ROUTINE-------------------------------- -
//

struct stReIndex {
   INT64*      pUCutOffs;
   INT64*      pICutOffs;
   INT32*      pUKey;
   void*       pIKey;

   INT64       ikey_length;
   INT64       uikey_length;
   INT64       u_cutoffs_length;

} ;

//
// t is the partition/cutoff index
template<typename KEYTYPE>
BOOL ReIndexGroupsMT(void* preindexV, int core, INT64 t) {
   stReIndex* preindex = (stReIndex*)preindexV;

   INT64* pUCutOffs = preindex->pUCutOffs;
   INT64* pICutOffs = preindex->pICutOffs;
   INT32* pUKey = preindex->pUKey;
   KEYTYPE* pIKey = (KEYTYPE*)preindex->pIKey;

   // Base 1 loop
   INT64    starti = 0;
   INT64    start = 0;
   if (t > 0) {
      starti = pICutOffs[t - 1];
      start = pUCutOffs[t - 1];
   }

   INT64   stopi = pICutOffs[t];
   INT32*  pUniques = &pUKey[start];

   // Check for out of bounds when indexing uniques
   INT64   uKeyLength = preindex->uikey_length - start;
   if (uKeyLength < 0) uKeyLength = 0;

   LOGGING("Start %lld  Stop %lld  Len:%lld\n", starti, stopi, preindex->ikey_length);

   for (INT64 j = starti; j < stopi; j++) {
      KEYTYPE index = pIKey[j];
      if (index <= 0 || index > uKeyLength) {
         // preserve filtered out or mark as filtered if out of range
         pIKey[j] = 0;
      }
      else {
         // reindex ikey inplace
         pIKey[j] = (KEYTYPE)pUniques[index - 1];
      }
   }

   return TRUE;
}

//---------------------------------------------------------------------------
// Input:
// Arg1: ikey numpy array of old ikey after hstack or multistack load
// Arg2: uikey numpy array unique index
// Arg3: u_cutoffs
// Arg3: i_cutoffs
//
PyObject *
ReIndexGroups(PyObject *self, PyObject *args)
{

   PyArrayObject *ikey = NULL;
   PyArrayObject *uikey = NULL;
   PyArrayObject *u_cutoffs = NULL;
   PyArrayObject *i_cutoffs = NULL;

   if (!PyArg_ParseTuple(
      args, "O!O!O!O!",
      &PyArray_Type, &ikey,
      &PyArray_Type, &uikey,
      &PyArray_Type, &u_cutoffs,
      &PyArray_Type, &i_cutoffs
      )) {

      return NULL;
   }
   if (PyArray_ITEMSIZE(u_cutoffs) != 8) {
      PyErr_Format(PyExc_ValueError, "u-cutoffs must be int64");
      return NULL;
   }
   if (PyArray_ITEMSIZE(i_cutoffs) != 8) {
      PyErr_Format(PyExc_ValueError, "i-cutoffs must be int64");
      return NULL;
   }

   if (PyArray_ITEMSIZE(uikey) != 4) {
      PyErr_Format(PyExc_ValueError, "uikey must be int32");
      return NULL;
   }

   INT64 u_cutoffs_length = ArrayLength(u_cutoffs);

   stReIndex preindex;

   preindex.pUCutOffs = (INT64*)PyArray_BYTES(u_cutoffs);
   preindex.pICutOffs = (INT64*)PyArray_BYTES(i_cutoffs);
   preindex.pUKey = (INT32*)PyArray_BYTES(uikey);
   preindex.pIKey = PyArray_BYTES(ikey);

   preindex.ikey_length = ArrayLength(ikey);
   preindex.u_cutoffs_length = u_cutoffs_length;
   preindex.uikey_length = ArrayLength(uikey);

   switch (PyArray_ITEMSIZE(ikey)) {
   case 1:
      g_cMathWorker->DoMultiThreadedWork((int)u_cutoffs_length, ReIndexGroupsMT<INT8>, &preindex);
      break;
   case 2:
      g_cMathWorker->DoMultiThreadedWork((int)u_cutoffs_length, ReIndexGroupsMT<INT16>, &preindex);
      break;
   case 4:
      g_cMathWorker->DoMultiThreadedWork((int)u_cutoffs_length, ReIndexGroupsMT<INT32>, &preindex);
      break;
   case 8:
      g_cMathWorker->DoMultiThreadedWork((int)u_cutoffs_length, ReIndexGroupsMT<INT64>, &preindex);
      break;
   default:
      PyErr_Format(PyExc_ValueError, "ikey must be int8/16/32/64");
      return NULL;
   }

   Py_IncRef((PyObject*)ikey);
   return (PyObject*)ikey;

}


struct stReverseIndex {
   void*       pIKey;
   void*       pOutKey;
   INT64       ikey_length;
};


// This routine is parallelized
// Algo: out[in[i]] = i
template<typename KEYTYPE>
BOOL ReverseShuffleMT(void* preindexV, int core, INT64 start, INT64 length) {
   stReverseIndex* preindex = (stReverseIndex*)preindexV;

   KEYTYPE* pIn = (KEYTYPE*)preindex->pIKey;
   KEYTYPE* pOut = (KEYTYPE*)preindex->pOutKey;
   INT64 maxindex = preindex->ikey_length;

   for (INT64 i = start; i < (start + length); i++) {
      KEYTYPE index = pIn[i];
      if (index >= 0 && index < maxindex) {
         pOut[index] = (KEYTYPE)i;
      }
   }
   return TRUE;
}

//---------------------------------------------------------------------------
// Input:
// Arg1: ikey numpy array from lexsort or grouping.iGroup
//       array must be integers
//       array must have integers only from 0 to len(arr)-1
//       all values must be unique, then it can be reversed quickly
//
//       if "in" is the input array and "out" is the output array
//       out[in[i]] = i
// Output:
//      Returns index array with indexes reversed back prior to lexsort
PyObject *
ReverseShuffle(PyObject *self, PyObject *args)
{
   PyArrayObject *ikey = NULL;

   if (!PyArg_ParseTuple(
      args, "O!",
      &PyArray_Type, &ikey
   )) {
      return NULL;
   }

   stReverseIndex preindex;

   int dtype=PyArray_TYPE(ikey);

   // check for only signed ints
   if (dtype >= 10 || (dtype &1) == 0) {
      PyErr_Format(PyExc_ValueError, "ReverseShuffle: ikey must be int8/16/32/64");
      return NULL;
   }

   PyArrayObject *pReturnArray = AllocateLikeNumpyArray(ikey, dtype);

   if (pReturnArray) {

      preindex.pIKey = PyArray_BYTES(ikey);
      preindex.pOutKey = PyArray_BYTES(pReturnArray);

      INT64 arrlength = ArrayLength(ikey);
      preindex.ikey_length = arrlength;

      switch (PyArray_ITEMSIZE(ikey)) {
      case 1:
         g_cMathWorker->DoMultiThreadedChunkWork(arrlength, ReverseShuffleMT<INT8>, &preindex);
         break;
      case 2:
         g_cMathWorker->DoMultiThreadedChunkWork(arrlength, ReverseShuffleMT<INT16>, &preindex);
         break;
      case 4:
         g_cMathWorker->DoMultiThreadedChunkWork(arrlength, ReverseShuffleMT<INT32>, &preindex);
         break;
      case 8:
         g_cMathWorker->DoMultiThreadedChunkWork(arrlength, ReverseShuffleMT<INT64>, &preindex);
         break;
      default:
         PyErr_Format(PyExc_ValueError, "ReverseShuffle: ikey must be int8/16/32/64");
         return NULL;
      }

      return (PyObject*)pReturnArray;
   }

   PyErr_Format(PyExc_ValueError, "ReverseShuffle: ran out of memory");
   return NULL;
}

/*
//-----------------------------------------------------
PyObject *
MergeBinnedCutoffs(PyObject *self, PyObject *args) {

   //#--------- START OF C++ ROUTINE -------------
   //   #based on how many uniques we have, allocate the new ikey
   //      # do we have a routine for this ?
   //      uikey_length = len(uikey)
   //      if uikey_length < 100 :
   //         dtype = np.int8
   //         elif uikey_length < 30_000 :
   //         dtype = np.int16
   //         elif uikey_length < 2_000_000_000 :
   //         dtype = np.int32
   //      else:
   //   dtype = np.int64
   //
   //      newikey = empty((len(ikey), ), dtype = dtype)
   //
   //      start = 0
   //      starti = 0
   //      for i in range(len(u_cutoffs)) :
   //         stop = u_cutoffs[i]
   //         stopi = i_cutoffs[i]
   //         uikey_slice = uikey[start:stop]
   //         oldikey_slice = ikey[starti:stopi]
   //
   //         if verbose:
   //   print("fixing ", starti, stopi)
   //      print("newikey ", newikey)
   //      print("oldikey_slice ", oldikey_slice)
   //
   //      # write a routine for this in C++
   //# if 0 and base_index=1, then keep the 0
   //      newikey[starti:stopi] = uikey_slice[oldikey_slice - 1]
   //      start = stop
   //      starti = stopi
   //      #END C++ ROUTINE-------------------------------- -

   Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

   if (argTupleSize < 3) {
      PyErr_Format(PyExc_ValueError, "SetItem requires three args instead of %llu args", argTupleSize);
      return NULL;
   }

   PyArrayObject* arr = (PyArrayObject*)PyTuple_GetItem(args, 0);
   PyArrayObject* mask = (PyArrayObject*)PyTuple_GetItem(args, 1);

   // Try to convert value if we have to
   PyObject* value = PyTuple_GetItem(args, 2);
   if (!PyArray_Check(value)) {
      value = PyArray_FromAny(value, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
   }

   if (PyArray_Check(arr) && PyArray_Check(mask) && PyArray_Check(value)) {
      PyArrayObject* inValues = (PyArrayObject*)value;

      if (PyArray_TYPE(mask) == NPY_BOOL) {

         INT64 itemSizeOut = PyArray_ITEMSIZE(arr);
         INT64 itemSizeIn = PyArray_ITEMSIZE(inValues);

         // check for strides... ?
         INT64 arrayLength = ArrayLength(arr);
         if (arrayLength == ArrayLength(mask) && itemSizeOut == PyArray_STRIDE(arr, 0)) {
            INT64 valLength = ArrayLength(inValues);

            if (arrayLength == valLength) {
               int outDType = PyArray_TYPE(arr);
               int inDType = PyArray_TYPE(inValues);
               MASK_CONVERT_SAFE maskSafe = GetConversionPutMask(inDType, outDType);

               if (maskSafe) {

                  // MT callback
                  struct MASK_CALLBACK_STRUCT {
                     MASK_CONVERT_SAFE maskSafe;
                     char* pIn;
                     char* pOut;
                     INT64 itemSizeOut;
                     INT64 itemSizeIn;
                     INT8* pMask;
                     void* pBadInput1;
                     void* pBadOutput1;

                  };

                  MASK_CALLBACK_STRUCT stMask;

                  // This is the routine that will be called back from multiple threads
                  auto lambdaMaskCallback = [](void* callbackArgT, int core, INT64 start, INT64 length) -> BOOL {
                     MASK_CALLBACK_STRUCT* callbackArg = (MASK_CALLBACK_STRUCT*)callbackArgT;

                     //printf("[%d] Mask %lld %lld\n", core, start, length);
                     //maskSafe(pIn, pOut, (INT8*)pMask, length, pBadInput1, pBadOutput1);
                     // Auto adjust pointers
                     callbackArg->maskSafe(
                        callbackArg->pIn + (start * callbackArg->itemSizeIn),
                        callbackArg->pOut + (start * callbackArg->itemSizeOut),
                        callbackArg->pMask + start,
                        length,
                        callbackArg->pBadInput1,
                        callbackArg->pBadOutput1);

                     return TRUE;
                  };

                  stMask.itemSizeIn = itemSizeIn;
                  stMask.itemSizeOut = itemSizeOut;
                  stMask.pBadInput1 = GetDefaultForType(inDType);
                  stMask.pBadOutput1 = GetDefaultForType(outDType);

                  stMask.pIn = (char*)PyArray_BYTES(inValues, 0);
                  stMask.pOut = (char*)PyArray_BYTES(arr, 0);
                  stMask.pMask = (INT8*)PyArray_BYTES(mask, 0);
                  stMask.maskSafe = maskSafe;

                  g_cMathWorker->DoMultiThreadedChunkWork(arrayLength, lambdaMaskCallback, &stMask);

                  Py_IncRef(Py_True);
                  return Py_True;
               }
            }
         }
      }
   }

   // punt to numpy
   Py_IncRef(Py_False);
   return Py_False;



}

*/
