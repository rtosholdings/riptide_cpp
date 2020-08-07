#include "RipTide.h"
#include <Python.h>
#include "ndarray.h"
#include "Bins.h"
#include "Sort.h"
//#include "MultiKey.h"
#include "MathWorker.h"

#define LOGGING(...)
//#define LOGGING printf

//--------------------------------------------------------------------------
typedef  void(*REINDEXCALLBACK)(void* pDataOut, void* pDataIn, void* pIndex1, INT64 indexLen, INT64 itemSize);

//===============================================================================================================
//--------------------------------------------------------
// T is the type of the index -- usually INT32 or INT64
// U is a datatype = to the itemsize
// The INDEXER is auto incremented by its type size
// The Out is auto incremented by itemsize
// The pDataIn MUST be passed as the base pointer
// size is the block size
template<typename T, typename U>
void ReIndexData(void* pDataOut, void* pDataIn, void* pIndex1, INT64 size, INT64 itemSize) {
   U* pOut = (U*)pDataOut;
   U* pIn = (U*)pDataIn;
   T* pIndex = (T*)pIndex1;

   for (INT64 i = 0; i < size; i++) {
      T index = pIndex[i];
      pOut[i] = pIn[index];
   }

}

//--------------------------------------------------------
// T is the type of the index -- usually INT32 or INT64
// U is a datatype = to the itemsize
template<typename T>
void ReIndexData(void* pDataOut, void* pDataIn, void* pIndex1, INT64 size, INT64 itemSize) {
   char* pOut = (char*)pDataOut;
   char* pIn = (char*)pDataIn;
   T* pIndex = (T*)pIndex1;

   for (INT64 i = 0; i < size; i++) {
      T index = pIndex[i];

      for (INT64 j = 0; j < itemSize; j++) {
         pOut[i*itemSize + j] = pIn[index*itemSize + j];

      }
   }

}


//--------------------------------------------------------
// T is the type of the index -- usually INT32 or INT64
template<typename T>
REINDEXCALLBACK ReIndexDataStep1(INT64 itemSize) {
   switch (itemSize) {
   case 1:
      return ReIndexData<T, BYTE>;
   case 2:
      return ReIndexData<T, INT16>;
   case 4:
      return ReIndexData<T, float>;
   case 8:
      return ReIndexData<T, double>;
   case 16:
      return ReIndexData<T, __m128>;
   default:
      return ReIndexData<T>;
   }
   PyErr_Format(PyExc_ValueError, "ReIndexing failed on unknown index size");
   return NULL;
}

//=========================================================
REINDEXCALLBACK ReIndexer(INT64 itemSize, int dtypeIndex) {

   // Get the dtype for the INDEXER (should be INT32 or INT64)
   switch (dtypeIndex) {

   CASE_NPY_INT32:
   CASE_NPY_UINT32:
      return ReIndexDataStep1<INT32>(itemSize);
   CASE_NPY_UINT64:
   CASE_NPY_INT64:
      return ReIndexDataStep1<INT64>(itemSize);
   case NPY_FLOAT:
      // allow matlab style float indexing?
      //return ReIndexDataStep1<float>(pDataOut, pDataIn, pIndex, size, itemSize);
   case NPY_DOUBLE:
      // allow matlab style float indexing?
      //return ReIndexDataStep1<double>(pDataOut, pDataIn, pIndex, size, itemSize);
   default:
      break;
   }
   PyErr_Format(PyExc_ValueError, "ReIndexing failed on unknown indexer %d", dtypeIndex);
   return NULL;
}



struct RIDX_CALLBACK {
   REINDEXCALLBACK reIndexCallback;
   char* pDataOut;
   char* pDataIn;
   char* pIndex1;
   INT64 indexLen;
   INT64 itemSize;

   INT64 indexTypeSize;

} stRICallback;



//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static BOOL ReIndexThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex) {
   BOOL didSomeWork = FALSE;
   RIDX_CALLBACK* OldCallback = (RIDX_CALLBACK*)pstWorkerItem->WorkCallbackArg;

   char* pDataIn = (char *)OldCallback->pDataIn;
   char* pDataIndex = (char *)OldCallback->pIndex1;
   char* pDataOut = (char*)OldCallback->pDataOut;
   INT64 lenX;
   INT64 workBlock;

   // As long as there is work to do
   while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

      INT64 offsetAdj = pstWorkerItem->BlockSize * workBlock * OldCallback->itemSize;
      INT64 indexAdj = pstWorkerItem->BlockSize * workBlock * OldCallback->indexTypeSize;

      OldCallback->reIndexCallback(pDataOut + offsetAdj, pDataIn, pDataIndex + indexAdj, lenX, stRICallback.itemSize);

      // Indicate we completed a block
      didSomeWork = TRUE;

      // tell others we completed this work block
      pstWorkerItem->CompleteWorkBlock();
      //printf("|%d %d", core, (int)workBlock);
   }

   return didSomeWork;
}


//===============================================================================
// The first param is the INDEXER (int or float based)
// second, third, fourth... etc are those to be
//
// TJD: this is a fast version of mget
// since it does not check for negative numbers or out of range numbers
PyObject* ReIndex(PyObject *self, PyObject *args) {
   CMultiListPrepare mlp(args);

   if (mlp.aInfo && mlp.tupleSize > 1) {

      // Bug bug -- arraysize should come from first arg and must <= second arg
      INT64 arraySize1 = mlp.totalRows;

      PyArrayObject* result = AllocateLikeResize(mlp.aInfo[1].pObject, arraySize1);

      if (result) {
         void* pDataOut = PyArray_BYTES(result);
         void* pDataIn = PyArray_BYTES(mlp.aInfo[1].pObject);
         void* pIndex = PyArray_BYTES(mlp.aInfo[0].pObject);

         // Get the util function to reindex the data
         REINDEXCALLBACK reIndexCallback =
            ReIndexer(mlp.aInfo[1].ItemSize, mlp.aInfo[0].NumpyDType);

         if (reIndexCallback != NULL) {
            // Check if we can use worker threads
            stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(arraySize1);

            if (pWorkItem == NULL) {
               // Threading not allowed for this work item, call it directly from main thread
               reIndexCallback(pDataOut, pDataIn, pIndex, arraySize1, mlp.aInfo[1].ItemSize);

            }
            else {
               // Each thread will call this routine with the callbackArg
               pWorkItem->DoWorkCallback = ReIndexThreadCallback;
               pWorkItem->WorkCallbackArg = &stRICallback;

               stRICallback.reIndexCallback = reIndexCallback;
               stRICallback.pDataOut = (char*)pDataOut;
               stRICallback.pDataIn = (char*)pDataIn;
               stRICallback.pIndex1 = (char*)pIndex;
               stRICallback.indexLen = arraySize1;
               stRICallback.itemSize = mlp.aInfo[1].ItemSize;
               stRICallback.indexTypeSize = mlp.aInfo[0].ItemSize;

               // This will notify the worker threads of a new work item
               g_cMathWorker->WorkMain(pWorkItem, arraySize1, 0);

            }

            return SetFastArrayView(result);
         }
      }
   }
   else {
      PyErr_Format(PyExc_ValueError, "ReIndex: Must pass in at least two params, first param is the indexer");

   }
   Py_INCREF(Py_None);
   return Py_None;
}




//=================================================================================================================
typedef void(*REMAP_INDEX)(void* pInput1, void* pOutput1, INT64 length, INT32* pMapper, INT64 maplength);

//--------------------------------------------------------------------------------
template<typename T, typename U>
void ReMapIndex(void* pInput1, void* pOutput1, INT64 length, INT32* pMapper, INT64 maplength) {

   T* pInput = (T*)pInput1;
   U* pOutput = (U*)pOutput1;

   for (INT64 i = 0; i < length; i++) {
      pOutput[i] = (U)(pMapper[pInput[i]]);
   }

}


//--------------------------------------------------------
// T is the type of the index -- usually INT32 or INT64
template<typename T>
REMAP_INDEX ReMapIndexStep1(int numpyOutputType) {

   switch (numpyOutputType) {
   case NPY_INT8:
      return ReMapIndex<T, INT8>;
      break;
   case NPY_INT16:
      return ReMapIndex<T, INT16>;
      break;
   CASE_NPY_INT32:
      return ReMapIndex<T, INT32>;
      break;
   CASE_NPY_INT64:
      return ReMapIndex<T, INT64>;
      break;
   default:
      printf("ReMapIndexStep1 does not understand type %d\n", numpyOutputType);
      break;
   }
   return NULL;

}


//===============================================================================
// Arg1: input array (int8,int16,int32, int64)
// Arg2: output array (int8, int16, int32, int64)
// Arg3: new mapping (always int32 for now)
//
// Used by categoricals to merge
// [0 1 3 1 0 1 2]
// [1 2 5 6]
//-----------------------------
// [1 2 6 2 1 2 5]

// !!!! THIS ROUTINE IS NOT USED -- CAN USE FOR SOMETHING ELSE
PyObject* ReMapIndex(PyObject *self, PyObject *args) {
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *outArr1 = NULL;
   PyArrayObject *remapArr1 = NULL;

   if (!PyArg_ParseTuple(args, "O!O!O",
      &PyArray_Type, &inArr1,
      &PyArray_Type, &outArr1,
      &PyArray_Type, &remapArr1)) return NULL;


   if (PyArray_TYPE(remapArr1) != NPY_INT32) {

      PyErr_Format(PyExc_ValueError, "third arg array must be NPY_INT32 -- not %d", PyArray_TYPE(remapArr1));
   }
   else {
      int numpyInType = PyArray_TYPE(inArr1);
      int numpyOutType = PyArray_TYPE(outArr1);

      REMAP_INDEX func = NULL;

      switch (numpyInType) {
      case NPY_INT8:
         func = ReMapIndexStep1<INT8>(numpyOutType);
         break;
      case NPY_INT16:
         func = ReMapIndexStep1<INT16>(numpyOutType);
         break;
      CASE_NPY_INT32:
         func = ReMapIndexStep1<INT32>(numpyOutType);
         break;
      CASE_NPY_INT64:
         func = ReMapIndexStep1<INT64>(numpyOutType);
         break;
      default:
         printf("ReMapIndexStep1 does not understand type %d\n", numpyOutType);
         break;
      }

      if (func) {
         void *pInput = PyArray_BYTES(inArr1);
         void *pOutput = PyArray_BYTES(outArr1);
         void* pRemap = PyArray_BYTES(remapArr1);
         INT64 length = ArrayLength(inArr1);
         INT64 mapLength = ArrayLength(remapArr1);

         func(pInput, pOutput, length, (INT32*)pRemap, mapLength);
      }
   }

   Py_INCREF(Py_None);
   return Py_None;

}




//=================================================================================================================
typedef PyObject*(*NAN_INF_COUNT)(void* pDataIn1, void* pIndex, INT64 arraySize1, int numpyInType);

//--------------------------------------------------------
// T is the value type (int,float, bool) and matches numpyInType
// U is the type of the index -- usually INT32 or INT64
//-----------------------------------------------------------------------------------------------
template <typename T, typename U> PyObject*
NanInfCount(void* pDataIn1, void* pIndex1, INT64 arraySize1, int numpyInType) {

   void* pDefault = GetDefaultForType(numpyInType);
   T defaultNan = *(T*)pDefault;

   T* pData = (T*)pDataIn1;
   U* pIndex = (U*)pIndex1;

   INT64 nancount = 0;
   INT64 posInfCount = 0;
   INT64 negInfCount = 0;


   if (numpyInType == NPY_FLOAT || numpyInType == NPY_DOUBLE || numpyInType == NPY_LONGDOUBLE) {
      INT64 i = arraySize1 - 1;

      // Scan from back
      while ((i >= 0) && pData[pIndex[i]] != pData[pIndex[i]]) {
         i--;
         nancount++;
      }

      while ((i >= 0) && pData[pIndex[i]] == INFINITY) {
         i--;
         posInfCount++;
      }

      int j = 0;

      while ((j <= i) && pData[pIndex[j]] == -INFINITY) {
         j++;
         negInfCount++;
      }

   }

   // BOOL cannot have invalid
   else if (numpyInType > 0) {

      // check for integer which can have invalid in front     
      if (numpyInType & 1) {
         // SCAN FORWARD
         //printf("***scan forward\n");
         INT64 i = arraySize1 - 1;
         int j = 0;

         while ((j <= i) && pData[pIndex[j]] == defaultNan) {
            j++;
            negInfCount++;
         }
      }
      else {
         //printf("***scan backward\n");
         INT64 i = arraySize1 - 1;

         // check for default value?
         while ((i >= 0) && pData[pIndex[i]] == defaultNan) {
            i--;
            posInfCount++;
         }

      }

   }

   return Py_BuildValue("(LLL)", nancount, posInfCount, negInfCount);
}

//------------------------------------------------------------------------------
// U is the index type: INT32 or INT64
// Returns NULL if no routine
template<typename U> NAN_INF_COUNT
GetNanInfCount(int numpyInType) {
   NAN_INF_COUNT result = NULL;

   switch (numpyInType) {
   case NPY_BOOL:
   case NPY_INT8:
      result = NanInfCount<INT8, U>;
      break;
   case NPY_INT16:
      result = NanInfCount<INT16, U>;
      break;
   CASE_NPY_INT32:
      result = NanInfCount<INT32, U>;
      break;
   CASE_NPY_INT64:
      result = NanInfCount<INT64, U>;
      break;
   case NPY_UINT8:
      result = NanInfCount<UINT8, U>;
      break;
   case NPY_UINT16:
      result = NanInfCount<UINT16, U>;
      break;
   CASE_NPY_UINT32:
      result = NanInfCount<UINT32, U>;
      break;
   CASE_NPY_UINT64:
      result = NanInfCount<UINT64, U>;
      break;
   case NPY_FLOAT:
      result = NanInfCount<float, U>;
      break;
   case NPY_DOUBLE:
      result = NanInfCount<double, U>;
      break;
   case NPY_LONGDOUBLE:
      result = NanInfCount<long double, U>;
      break;

   default:
      printf("NanInfCountFromSort does not understand type %d\n", numpyInType);
   }
   return result;

}

//===============================================================================
// First argument: values from unsorted array (e.g. array of floats)
// Second argument: result from lexsort or argsort (integer index) 
// Returns nancount, infcount,-infcount
PyObject* NanInfCountFromSort(PyObject *self, PyObject *args) {
   CMultiListPrepare mlp(args);

   if (mlp.aInfo && mlp.tupleSize == 2) {

      if (mlp.aInfo[0].ArrayLength == mlp.aInfo[1].ArrayLength) {
         void* pDataIn1 = mlp.aInfo[0].pData;
         void* pIndex1 = mlp.aInfo[1].pData;
         INT64 arraySize1 = mlp.aInfo[0].ArrayLength;
         int numpyInType = mlp.aInfo[0].NumpyDType;
         int numpyIndexType = mlp.aInfo[1].NumpyDType;

         NAN_INF_COUNT func = NULL;

         switch (numpyIndexType) {
         CASE_NPY_INT32:
            func = GetNanInfCount<INT32>(numpyInType);
            break;
         CASE_NPY_INT64:
            func = GetNanInfCount<INT64>(numpyInType);
            break;
         }

         if (func != NULL) {
            return func(pDataIn1, pIndex1, arraySize1, numpyInType);
         }

      }
      else {
         PyErr_Format(PyExc_ValueError, "NanInfCountFromSort: Array sizes must match");
      }


   }
   else {
      PyErr_Format(PyExc_ValueError, "NanInfCountFromSort: Must pass in at least two params, first param is the value array");

   }

   Py_INCREF(Py_None);
   return Py_None;

}


//=================================================================================================
//=================================================================================================

typedef void(*MAKE_BINS_SORTED)(void* pDataIn1, void* pIndex1, void* pOut1, INT64 length, double* pBin1, INT64 maxbin, INT64 nancount, INT64 infcount, INT64 neginfcount);
typedef void(*MAKE_BINS_BSEARCH)(void* pDataIn1, void* pOut1, INT64 start, INT64 length, void* pBin1, INT64 maxbin, int numpyInType);


//=================================================================================================
// T is the value type
// U is the sort index type (often INT32)
// Returns pOut1 wihch can be INT8,INT16,INT32
//
// NOTE: Uses double to separate bins but this does not have resolution for INT64 nanos
template<typename T, typename U, typename V>
void MakeBinsSorted(void* pDataIn1, void* pSortIndex1, void* pOut1, INT64 length, double* pBin1, INT64 maxbin, INT64 nancount, INT64 infcount, INT64 neginfcount) {
   T* pData = (T*)pDataIn1;
   U* pIndex = (U*)pSortIndex1;
   V* pOut = (V*)pOut1;

   INT64 i = 0;

   LOGGING("Array length %lld   bin length %lld   nancount %lld  neginfcount%lld\n", length, maxbin, nancount, neginfcount);

   // FIRST MARK all the bad bins in the beginning
   for (i = 0; i < neginfcount; i++) {
      U index = pIndex[i];
      LOGGING("setting index at %lld as invalid due to neginf\n", (long long)index);
      pOut[index] = 0;
   }

   // Now MARK all the bad bins at the end
   for (i = 0; i < (nancount + infcount); i++) {
      U index = pIndex[length - i - 1];
      LOGGING("setting index at %lld as invalid due to naninf\n", (long long)index);
      pOut[index] = 0;
   }

   // TODO: multithread optimize this section
   INT64 newlength = length - (nancount + infcount);
   V bin = 0;
   double compare = pBin1[bin];
   //printf("comparing0 to %lld  %lf\n", (INT64)bin, compare);
   //printf("comparing1 to %lld  %lf\n", (INT64)bin, pBin1[1]);

   // Start looking at beginning of real data
   // anything less, is bin 0, or the qcut invalid
   i = neginfcount;
   while (i < newlength) {
      U index = pIndex[i];
      if ((double)pData[index] < compare) {
         pOut[index] = 1;
      }
      else {
         break;
      }
      i++;
   }

   // Edge case test
   if (i < newlength) {
      U index = pIndex[i];
      if ((double)pData[index] == compare) {
         if (1 < maxbin) {
            bin = 1;
            compare = pBin1[bin];
         }
      }
   }

   // Now assign a valid bin to the good data in the middle
   while (i < newlength) {
      U index = pIndex[i];
      if ((double)pData[index] <= compare) {
         pOut[index] = bin + 1;
      }
      else {
         // find next bin
         while (bin < maxbin && (double)pData[index] > compare) {

            // move to next bin
            ++bin;
            compare = pBin1[bin];
            LOGGING("comparing to %lld %lld  %lf  %lf\n", i, (INT64)bin, (double)pData[index], compare);
         }
         if (bin >= maxbin) {
            break;
         }
         pOut[index] = bin + 1;
      }
      i++;
   }

   // Anything on end that is out of range
   while (i < newlength) {
      U index = pIndex[i];
      pOut[index] = 1;
      i++;
   }
}

//------------------------------------------------------------------------------
// U is the value index type: INT32 or INT64
// V is the bin index type: INT8, INT16, INT32, INT64
// Returns NULL if no routine
template<typename U, typename V> MAKE_BINS_SORTED
GetMakeBinsSorted(int numpyInType) {
   MAKE_BINS_SORTED result = NULL;

   switch (numpyInType) {
   case NPY_BOOL:
   case NPY_INT8:
      result = MakeBinsSorted<INT8, U, V>;
      break;
   case NPY_INT16:
      result = MakeBinsSorted<INT16, U, V>;
      break;
   CASE_NPY_INT32:
      result = MakeBinsSorted<INT32, U, V>;
      break;
   CASE_NPY_INT64:
      result = MakeBinsSorted<INT64, U, V>;
      break;
   case NPY_UINT8:
      result = MakeBinsSorted<UINT8, U, V>;
      break;
   case NPY_UINT16:
      result = MakeBinsSorted<UINT16, U, V>;
      break;
   CASE_NPY_UINT32:
      result = MakeBinsSorted<UINT32, U, V>;
      break;
   CASE_NPY_UINT64:
      result = MakeBinsSorted<UINT64, U, V>;
      break;
   case NPY_FLOAT:
      result = MakeBinsSorted<float, U, V>;
      break;
   case NPY_DOUBLE:
      result = MakeBinsSorted<double, U, V>;
      break;
   case NPY_LONGDOUBLE:
      result = MakeBinsSorted<long double, U, V>;
      break;

   default:
      printf("MakeBinsSorted does not understand type %d\n", numpyInType);
   }
   return result;

}



//===============================================================================
// First argument:  values from unsorted array (e.g. array of floats)
// 2nd arg: bin array of float64
// 3rd argument: result from lexsort or argsort (integer index) 
// 4th argument: counts (tuple of 3 values nan. inf, neginf)
// 5th arg: mode
// Returns BINS -
PyObject* BinsToCutsSorted(PyObject *self, PyObject *args) {
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *binArr1 = NULL;
   PyArrayObject *indexArr1 = NULL;
   PyTupleObject *counts = NULL;

   int binMode = 0;

   if (!PyArg_ParseTuple(args, "O!O!O!O!i",
      &PyArray_Type, &inArr1,
      &PyArray_Type, &binArr1,
      &PyArray_Type, &indexArr1,
      &PyTuple_Type, &counts,
      &binMode)) return NULL;

   INT64 nancount = 0;
   INT64 infcount = 0;
   INT64 neginfcount = 0;

   if (!PyArg_ParseTuple((PyObject*)counts, "LLL", &nancount, &infcount, &neginfcount)) return NULL;

   LOGGING("counts %lld %lld %lld\n", nancount, infcount, neginfcount);

   if (PyArray_TYPE(binArr1) != NPY_DOUBLE) {

      PyErr_Format(PyExc_ValueError, "bin array must be float64 -- not %d", PyArray_TYPE(binArr1));
   }
   else {

      int numpyInType = PyArray_TYPE(inArr1);
      int numpyIndexType = PyArray_TYPE(indexArr1);
      INT64 binSize = ArrayLength(binArr1);

      // Choose INT8,INT16,INT32 for bin mode
      int binmode = 0;
      if (binSize > 100) {
         binmode = 1;
         if (binSize > 30000) {
            binmode = 2;
            if (binSize > 2000000000LL) {
               binmode = 3;
            }
         }
      }
      MAKE_BINS_SORTED func = NULL;

      switch (numpyIndexType) {
      CASE_NPY_INT32:
         switch (binmode) {
         case 0:
            func = GetMakeBinsSorted<INT32, INT8>(numpyInType);
            break;
         case 1:
            func = GetMakeBinsSorted<INT32, INT16>(numpyInType);
            break;
         case 2:
            func = GetMakeBinsSorted<INT32, INT32>(numpyInType);
            break;
         case 3:
            func = GetMakeBinsSorted<INT32, INT64>(numpyInType);
            break;
         }
         break;

      CASE_NPY_INT64:
         switch (binmode) {
         case 0:
            func = GetMakeBinsSorted<INT64, INT8>(numpyInType);
            break;
         case 1:
            func = GetMakeBinsSorted<INT64, INT16>(numpyInType);
            break;
         case 2:
            func = GetMakeBinsSorted<INT64, INT32>(numpyInType);
            break;
         case 3:
            func = GetMakeBinsSorted<INT64, INT64>(numpyInType);
            break;
         }
         break;
      }

      if (func != NULL) {
         PyArrayObject* result = NULL;
         switch (binmode) {
         case 0:
            result = AllocateLikeNumpyArray(inArr1, NPY_INT8);
            break;
         case 1:
            result = AllocateLikeNumpyArray(inArr1, NPY_INT16);
            break;
         case 2:
            result = AllocateLikeNumpyArray(inArr1, NPY_INT32);
            break;
         case 3:
            result = AllocateLikeNumpyArray(inArr1, NPY_INT64);
            break;

         }

         if (result) {
            void* pDataOut = PyArray_BYTES(result);
            void* pDataIn1 = PyArray_BYTES(inArr1);
            void* pIndex1 = PyArray_BYTES(indexArr1);
            void* pBin1 = PyArray_BYTES(binArr1);

            func(pDataIn1, pIndex1, pDataOut, ArrayLength(inArr1), (double*)pBin1, binSize, nancount, infcount, neginfcount);
            return (PyObject*)result;
         }
      }
   }

   Py_INCREF(Py_None);
   return Py_None;
}



//=================================================================================================
// T is the value type
// U is the index type (often INT32)
// Returns pOut1 wihch can be INT8,INT16,INT32
//
// NOTE: Uses double to separate bins but this does not have resolution for INT64 nanos
template<typename T, typename V, typename BINTYPE>
void SearchSortedRight(void* pDataIn1, void* pOut1, const INT64 start, const INT64 length, void* pBin1T, INT64 maxbin1, int numpyInType) {
   const T* pData = (const T*)pDataIn1;
   pData = &pData[start];

   V* pOut = (V*)pOut1;
   pOut = &pOut[start];

   const BINTYPE* pBin1 = (const BINTYPE*)pBin1T;

   V maxbin = (V)maxbin1;

   const V lastbin = maxbin - 1;
   const T verylast = (const T)pBin1[lastbin];
   const T veryfirst = (const T)pBin1[0];

   LOGGING("SearchSortedLeft Array length %lld   bin length %lld\n", length, (long long)maxbin);

   // Now assign a valid bin to the good data in the middle
   for (INT64 i = 0; i < length; i++) {

      const T value = pData[i];

      if (value >= veryfirst && value < verylast) {

         // bin search
         V middle;
         V first = 0;
         V last = lastbin;
         BINTYPE val = (BINTYPE)value;

         // NOTE: this loop can be sped up
         do {
            middle = (first + last) >> 1;    //this finds the mid point
            if (pBin1[middle] > val)         // if it's in the lower half
            {
               last = middle - 1;
               if (first >= last) break;
            }
            else if (pBin1[middle] < val) {
               first = middle + 1;           //if it's in the upper half
               if (first >= last) break;
            }
            else {
               first = middle;
               break;
            }
         } while (1);

         //printf("bin %d  %d  %d  %lf %lf\n", (int)first, (int)middle, (int)last, (double)value, (double)pBin1[first]);
         //if (first > last) first = last;

         if (val < pBin1[first]) {
            pOut[i] = first;
         }
         else {
            // edge test
            pOut[i] = first + 1;
         }

      }
      else {
         // First bin for invalid values
         if (value < veryfirst) {
            pOut[i] = 0;
         }
         else {
            // numpy puts nans in last bin
            pOut[i] = lastbin + 1;
         }
      }

   }
}

//=================================================================================================
// T is the value type
// U is the index type (often INT32)
// Returns pOut1 wihch can be INT8,INT16,INT32
//
// NOTE: Uses double to separate bins but this does not have resolution for INT64 nanos
template<typename T, typename V, typename BINTYPE>
void SearchSortedLeft(void* pDataIn1, void* pOut1, const INT64 start, const INT64 length, void* pBin1T, INT64 maxbin1, int numpyInType) {
   const T* pData = (const T*)pDataIn1;
   pData = &pData[start];

   V* pOut = (V*)pOut1;
   pOut = &pOut[start];

   const BINTYPE* pBin1 = (const BINTYPE*)pBin1T;

   V maxbin = (V)maxbin1;

   const V lastbin = maxbin - 1;
   const T verylast = (const T)pBin1[lastbin];
   const T veryfirst = (const T)pBin1[0];

   LOGGING("SearchSortedLeft Array length %lld   bin length %lld\n", length, (long long)maxbin);

   // Now assign a valid bin to the good data in the middle
   for (INT64 i = 0; i < length; i++) {

      const T value = pData[i];

      if (value > veryfirst && value <= verylast) {

         // bin search
         V middle;
         V first = 0;
         V last = lastbin;
         BINTYPE val = (BINTYPE)value;

         // NOTE: this loop can be sped up
         do {
            middle = (first + last) >> 1;    //this finds the mid point
            if (pBin1[middle] > val)         // if it's in the lower half
            {
               last = middle - 1;
               if (first >= last) break;
            }
            else if (pBin1[middle] < val) {
               first = middle + 1;           //if it's in the upper half
               if (first >= last) break;
            }
            else {
               first = middle;
               break;
            }
         } while (1);

         //printf("bin %d  %d  %d  %lf %lf\n", (int)first, (int)middle, (int)last, (double)value, (double)pBin1[first]);
         //if (first > last) first = last;

         if (val <= pBin1[first]) {
            pOut[i] = first;
         }
         else {
            // edge test
            pOut[i] = first + 1;
         }

      }
      else {
         // First bin for invalid values
         if (value <= veryfirst) {
            pOut[i] = 0;
         }
         else {
            // numpy puts nans in last bin
            pOut[i] = lastbin + 1;
         }
      }

   }
}


NPY_INLINE static int
STRING_LTEQ(const char *s1, const char *s2, INT64 len1, INT64 len2)
{
   const unsigned char *c1 = (unsigned char *)s1;
   const unsigned char *c2 = (unsigned char *)s2;
   INT64 i;
   if (len1 == len2) {
      for (i = 0; i < len1; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
         }
      }
      // match
      return 1;
   }
   else if (len1 < len2) {
      for (i = 0; i < len1; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
         }
      }
      if (c2[i]==0) return 1;  // equal
      // c2 is longer and therefore c1 < c2
      return 1;

   } else {
      for (i = 0; i < len2; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
         }
      }
      if (c1[i] == 0) return 1;  // equal
      // c1 is longer and therefore c1 > c2
      return 0;
   }
}


NPY_INLINE static int
STRING_LT(const char *s1, const char *s2, INT64 len1, INT64 len2)
{
   const unsigned char *c1 = (unsigned char *)s1;
   const unsigned char *c2 = (unsigned char *)s2;
   INT64 i;
   if (len1 == len2) {
      for (i = 0; i < len1; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
         }
      }
      // match
      return 0;
   }
   else if (len1 < len2) {
      for (i = 0; i < len1; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
         }
      }
      if (c2[i] == 0) return 0;  // equal
      return 1;                  // c2 is longer and therefore c1 < c2      

   }
   else {
      for (i = 0; i < len2; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
         }
      }
      if (c1[i] == 0) return 0;  // equal
      return 0;                  // c1 is longer and therefore c1 > c2
   }
}


NPY_INLINE static int
STRING_GTEQ(const char *s1, const char *s2, INT64 len1, INT64 len2)
{
   const unsigned char *c1 = (unsigned char *)s1;
   const unsigned char *c2 = (unsigned char *)s2;
   INT64 i;
   if (len1 == len2) {
      for (i = 0; i < len1; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] > c2[i];
         }
      }
      // match
      return 1;
   }
   else if (len1 < len2) {
      for (i = 0; i < len1; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] > c2[i];
         }
      }
      if (c2[i] == 0) return 1;  // equal
      return 0;                  // c2 is longer and therefore c1 < c2

   }
   else {
      for (i = 0; i < len2; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] > c2[i];
         }
      }
      if (c1[i] == 0) return 1;  // equal
      return 1;                  // c1 is longer and therefore c1 > c2
      
   }
}


NPY_INLINE static int
STRING_GT(const char *s1, const char *s2, INT64 len1, INT64 len2)
{
   const unsigned char *c1 = (unsigned char *)s1;
   const unsigned char *c2 = (unsigned char *)s2;
   INT64 i;
   if (len1 == len2) {
      for (i = 0; i < len1; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] > c2[i];
         }
      }
      // match
      return 0;
   }
   else if (len1 < len2) {
      for (i = 0; i < len1; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] > c2[i];
         }
      }
      if (c2[i] == 0) return 0;  // equal
      return 0;                  // c2 is longer and therefore c1 < c2

   }
   else {
      for (i = 0; i < len2; ++i) {
         if (c1[i] != c2[i]) {
            return c1[i] > c2[i];
         }
      }
      if (c1[i] == 0) return 0;  // equal
      return 1;                  // c1 is longer and therefore c1 > c2

   }
}


//=================================================================================================
// T is the value type
// OUT_TYPE is the output index type
// Returns pOut1 wihch can be INT8,INT16,INT32,INT64
//
// NOTE: Uses double to separate bins but this does not have resolution for INT64 nanos
template<typename OUT_TYPE>
void SearchSortedLeftString(void* pDataIn1, void* pOut1, const INT64 start, const INT64 length, char* pBin1T, INT64 maxbin1, INT64 itemSizeValue, INT64 itemSizeBin) {
   const char* pData = (const char*)pDataIn1;
   pData = &pData[start];

   OUT_TYPE* pOut = (OUT_TYPE*)pOut1;
   pOut = &pOut[start];

   const char* pBin1 = (const char*)pBin1T;

   OUT_TYPE maxbin = (OUT_TYPE)maxbin1;

   const OUT_TYPE lastbin = maxbin - 1;
   const char* verylast = (const char*)&pBin1[lastbin*itemSizeBin];
   const char* veryfirst = (const char*)&pBin1[0];

   LOGGING("SearchSortedLeftString Array length %lld   bin length %lld\n", length, (long long)maxbin);

   // Now assign a valid bin to the good data in the middle
   for (INT64 i = 0; i < length; i++) {

      const char* value = &pData[i*itemSizeValue];

      if (STRING_GT(value, veryfirst, itemSizeValue, itemSizeBin) && STRING_LTEQ(value, verylast, itemSizeValue, itemSizeBin)) {

         // bin search
         OUT_TYPE middle;
         OUT_TYPE first = 0;
         OUT_TYPE last = lastbin;

         // NOTE: this loop can be sped up
         do {
            middle = (first + last) >> 1;    //this finds the mid point
            const char* pBinMiddle = &pBin1[middle*itemSizeBin];

            if (STRING_GT(pBinMiddle, value, itemSizeBin, itemSizeValue))         // if it's in the lower half
            {
               last = middle - 1;
               if (first >= last) break;
            }
            else if (STRING_LT(pBinMiddle, value, itemSizeBin, itemSizeValue)) {
               first = middle + 1;           //if it's in the upper half
               if (first >= last) break;
            }
            else {
               first = middle;
               break;
            }
         } while (1);

         //printf("bin %d  %d  %d  %lf %lf\n", (int)first, (int)middle, (int)last, (double)value, (double)pBin1[first]);
         //if (first > last) first = last;

         if (STRING_LTEQ(value, &pBin1[first*itemSizeBin], itemSizeValue, itemSizeBin)) {
            pOut[i] = first;
         }
         else {
            // edge test
            pOut[i] = first + 1;
         }

      }
      else {
         // First bin for invalid values
         if (STRING_LTEQ(value, veryfirst, itemSizeValue, itemSizeBin)) {
            pOut[i] = 0;
         }
         else {
            // numpy puts nans in last bin
            pOut[i] = lastbin + 1;
         }
      }

   }
}

//=================================================================================================
// T is the value type
// OUT_TYPE is the output index type
// Returns pOut1 wihch can be INT8,INT16,INT32,INT64
//
// NOTE: Uses double to separate bins but this does not have resolution for INT64 nanos
template<typename OUT_TYPE>
void MakeBinsBSearchString(void* pDataIn1, void* pOut1, const INT64 start, const INT64 length, char* pBin1T, INT64 maxbin1, INT64 itemSizeValue, INT64 itemSizeBin) {
   const char* pData = (const char*)pDataIn1;
   pData = &pData[start];

   OUT_TYPE* pOut = (OUT_TYPE*)pOut1;
   pOut = &pOut[start];

   const char* pBin1 = (const char*)pBin1T;

   OUT_TYPE maxbin = (OUT_TYPE)maxbin1;

   const OUT_TYPE lastbin = maxbin - 1;
   const char* verylast = (const char*)&pBin1[lastbin*itemSizeBin];
   const char* veryfirst = (const char*)&pBin1[0];

   LOGGING("MakeBinsBSearchString Array length %lld   bin length %lld    itemsizebin:%lld  itemsizevalue:%lld\n", length, maxbin1, itemSizeBin, itemSizeValue);

   // Now assign a valid bin to the good data in the middle
   for (INT64 i = 0; i < length; i++) {

      const char* value = &pData[i * itemSizeValue];

      if (*value != 0 && STRING_GTEQ(value, veryfirst, itemSizeValue, itemSizeBin) && STRING_LTEQ(value,verylast, itemSizeValue, itemSizeBin)) {

         // bin search
         OUT_TYPE middle;
         OUT_TYPE first = 0;
         OUT_TYPE last = lastbin;

         do {
            middle = (first + last) >> 1;    //this finds the mid point
            const char* pBinMiddle = &pBin1[middle*itemSizeBin];

            // OPTIMIZATION -- in one comparision can get > 0 ==0 <0
            if (STRING_GT(pBinMiddle, value, itemSizeBin, itemSizeValue))         // if it's in the lower half
            {
               last = middle - 1;
               if (first >= last) break;
            }
            else if (STRING_LT(pBinMiddle, value, itemSizeBin, itemSizeValue)) {
               first = middle + 1;           //if it's in the upper half
               if (first >= last) break;
            }
            else {
               first = middle;
               break;
            }
         } while (1);

         //printf("bin %d  %d  %d  %lf %lf\n", (int)first, (int)middle, (int)last, (double)value, (double)pBin1[first]);
         //if (first > last) first = last;

         if (first > 0) {
            if (STRING_LTEQ(value, &pBin1[first*itemSizeBin], itemSizeValue, itemSizeBin)) {
               pOut[i] = first;
            }
            else {
               // edge test
               pOut[i] = first + 1;
            }
         }
         else {
            pOut[i] = 1;
         }

      }
      else {
         // First bin for invalid values
         pOut[i] = 0;
      }
      //printf("for value %lf -- %d\n", (double)value, (int)pOut[i]);
   }

}




//=================================================================================================
// T is the value type
// OUT_TYPE is the output index type
// Returns pOut1 wihch can be INT8,INT16,INT32,INT64
//
// NOTE: Uses double to separate bins but this does not have resolution for INT64 nanos
template<typename T, typename OUT_TYPE, typename BINTYPE>
void MakeBinsBSearchFloat(void* pDataIn1, void* pOut1, const INT64 start, const INT64 length, void* pBin1T, INT64 maxbin1, int numpyInType) {
   const T* pData = (const T*)pDataIn1;
   pData = &pData[start];

   OUT_TYPE* pOut = (OUT_TYPE*)pOut1;
   pOut = &pOut[start];

   const BINTYPE* pBin1 = (const BINTYPE*)pBin1T;

   OUT_TYPE maxbin = (OUT_TYPE)maxbin1;

   const OUT_TYPE lastbin = maxbin - 1;
   const T verylast = (const T)pBin1[lastbin];
   const T veryfirst = (const T)pBin1[0];

   LOGGING("MakeBinsBSearchFloat Array length %lld   bin length %lld    bintype %d\n", length, maxbin1, numpyInType);

   // Now assign a valid bin to the good data in the middle
   for (INT64 i = 0; i < length; i++) {

      const T value = pData[i];
      if (std::isfinite(value) && value >= veryfirst && value <= verylast) {

         // bin search
         OUT_TYPE middle;
         OUT_TYPE first = 0;
         OUT_TYPE last = lastbin;
         BINTYPE val = (BINTYPE)value;

         do {
            middle = (first + last) >> 1;    //this finds the mid point
            if (pBin1[middle] > val)         // if it's in the lower half
            {
               last = middle - 1;
               if (first >= last) break;
            }
            else if (pBin1[middle] < val) {
               first = middle + 1;           //if it's in the upper half
               if (first >= last) break;
            }
            else {
               first = middle;
               break;
            }
         } while (1);

         //printf("bin %d  %d  %d  %lf %lf\n", (int)first, (int)middle, (int)last, (double)value, (double)pBin1[first]);
         //if (first > last) first = last;

         if (first > 0) {
            if (val <= pBin1[first]) {
               pOut[i] = first;
            }
            else {
               // edge test
               pOut[i] = first + 1;
            }
         }
         else {
            pOut[i] = 1;
         }

      }
      else {
         // First bin for invalid values
         pOut[i] = 0;
      }
      //printf("for value %lf -- %d\n", (double)value, (int)pOut[i]);
   }

}




//=================================================================================================
// T is the value type
// OUT_TYPE is the bin index type: INT8, INT16, INT32, INT64
// U is the index type (often INT32)
// Returns pOut1 wihch can be INT8,INT16,INT32,INT64 (the OUT_TYPE)
//
// NOTE: Uses double to separate bins but this does not have resolution for INT64 nanos
template<typename T, typename OUT_TYPE, typename BINTYPE>
void MakeBinsBSearch(void* pDataIn1, void* pOut1, const INT64 start, const INT64 length, void* pBin1T, INT64 maxbin1, int numpyInType) {
   const T* pData = (const T*)pDataIn1;
   pData = &pData[start];

   OUT_TYPE* pOut = (OUT_TYPE*)pOut1;
   pOut = &pOut[start];

   const BINTYPE* pBin1 = (const BINTYPE*)pBin1T;

   OUT_TYPE maxbin = (OUT_TYPE)maxbin1;

   void* pDefault = GetDefaultForType(numpyInType);
   const T defaultNan = *(T*)pDefault;
   const OUT_TYPE lastbin = maxbin - 1;
   const T verylast = (const T)pBin1[lastbin];
   const T veryfirst = (const T)pBin1[0];

   LOGGING("MakeBinsBSearch Array length %lld   bin length %lld\n", length, (long long)maxbin);

   // Now assign a valid bin to the good data in the middle
   for (INT64 i = 0; i < length; i++) {

      const T value = pData[i];

      if (value != defaultNan && value >= veryfirst && value <= verylast) {

         // bin search
         OUT_TYPE middle;
         OUT_TYPE first = 0;
         OUT_TYPE last = lastbin;
         BINTYPE val = (BINTYPE)value;

         // NOTE: this loop can be sped up
         do {
            middle = (first + last) >> 1;    //this finds the mid point
            if (pBin1[middle] > val)         // if it's in the lower half
            {
               last = middle - 1;
               if (first >= last) break;
            }
            else if (pBin1[middle] < val) {
               first = middle + 1;           //if it's in the upper half
               if (first >= last) break;
            }
            else {
               first = middle;
               break;
            }
         } while (1);

         //printf("bin %d  %d  %d  %lf %lf\n", (int)first, (int)middle, (int)last, (double)value, (double)pBin1[first]);
         //if (first > last) first = last;

         if (first > 0) {
            if (val <= pBin1[first]) {
               pOut[i] = first;
            }
            else {
               // edge test
               pOut[i] = first + 1;
            }
         }
         else {
            pOut[i] = 1;
         }

      }
      else {
         // First bin for invalid values
         pOut[i] = 0;
      }

   }
}

//------------------------------------------------------------------------------
// BINTYPE is the value index type: INT32 or INT64
// OUT_TYPE is the bin index type: INT8, INT16, INT32, INT64
// Returns NULL if no routine
template<typename OUT_TYPE, typename BINTYPE> MAKE_BINS_BSEARCH
GetMakeBinsBSearchPart2(int numpyInType, int searchMode) {
   MAKE_BINS_BSEARCH result = NULL;

   if (searchMode == 0) {
      switch (numpyInType) {
      case NPY_BOOL:
      case NPY_INT8:
         result = MakeBinsBSearch<INT8, OUT_TYPE, BINTYPE>;
         break;
      case NPY_INT16:
         result = MakeBinsBSearch<INT16, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_INT32:
         result = MakeBinsBSearch<INT32, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_INT64:
         result = MakeBinsBSearch<INT64, OUT_TYPE, BINTYPE>;
         break;
      case NPY_UINT8:
         result = MakeBinsBSearch<UINT8, OUT_TYPE, BINTYPE>;
         break;
      case NPY_UINT16:
         result = MakeBinsBSearch<UINT16, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_UINT32:
         result = MakeBinsBSearch<UINT32, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_UINT64:
         result = MakeBinsBSearch<UINT64, OUT_TYPE, BINTYPE>;
         break;
      case NPY_FLOAT:
         result = MakeBinsBSearchFloat<float, OUT_TYPE, BINTYPE>;
         break;
      case NPY_DOUBLE:
         result = MakeBinsBSearchFloat<double, OUT_TYPE, BINTYPE>;
         break;
      case NPY_LONGDOUBLE:
         result = MakeBinsBSearchFloat<long double, OUT_TYPE, BINTYPE>;
         break;

      default:
         printf("MakeBinsBSearch does not understand type %d\n", numpyInType);
      }
      return result;
   } 
   if (searchMode == 1) {
      switch (numpyInType) {
      case NPY_BOOL:
      case NPY_INT8:
         result = SearchSortedLeft<INT8, OUT_TYPE, BINTYPE>;
         break;
      case NPY_INT16:
         result = SearchSortedLeft<INT16, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_INT32:
         result = SearchSortedLeft<INT32, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_INT64:
         result = SearchSortedLeft<INT64, OUT_TYPE, BINTYPE>;
         break;
      case NPY_UINT8:
         result = SearchSortedLeft<UINT8, OUT_TYPE, BINTYPE>;
         break;
      case NPY_UINT16:
         result = SearchSortedLeft<UINT16, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_UINT32:
         result = SearchSortedLeft<UINT32, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_UINT64:
         result = SearchSortedLeft<UINT64, OUT_TYPE, BINTYPE>;
         break;
      case NPY_FLOAT:
         result = SearchSortedLeft<float, OUT_TYPE, BINTYPE>;
         break;
      case NPY_DOUBLE:
         result = SearchSortedLeft<double, OUT_TYPE, BINTYPE>;
         break;
      case NPY_LONGDOUBLE:
         result = SearchSortedLeft<long double, OUT_TYPE, BINTYPE>;
         break;

      default:
         printf("MakeBinsBSearch does not understand type %d\n", numpyInType);
      }
      return result;
   }
   if (searchMode == 2) {
      switch (numpyInType) {
      case NPY_BOOL:
      case NPY_INT8:
         result = SearchSortedRight<INT8, OUT_TYPE, BINTYPE>;
         break;
      case NPY_INT16:
         result = SearchSortedRight<INT16, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_INT32:
         result = SearchSortedRight<INT32, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_INT64:
         result = SearchSortedRight<INT64, OUT_TYPE, BINTYPE>;
         break;
      case NPY_UINT8:
         result = SearchSortedRight<UINT8, OUT_TYPE, BINTYPE>;
         break;
      case NPY_UINT16:
         result = SearchSortedRight<UINT16, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_UINT32:
         result = SearchSortedRight<UINT32, OUT_TYPE, BINTYPE>;
         break;
      CASE_NPY_UINT64:
         result = SearchSortedRight<UINT64, OUT_TYPE, BINTYPE>;
         break;
      case NPY_FLOAT:
         result = SearchSortedRight<float, OUT_TYPE, BINTYPE>;
         break;
      case NPY_DOUBLE:
         result = SearchSortedRight<double, OUT_TYPE, BINTYPE>;
         break;
      case NPY_LONGDOUBLE:
         result = SearchSortedRight<long double, OUT_TYPE, BINTYPE>;
         break;

      default:
         printf("MakeBinsBSearch does not understand type %d\n", numpyInType);
      }
      return result;
   }
   return result;

}

template<typename OUT_TYPE> MAKE_BINS_BSEARCH
GetMakeBinsBSearch(int numpyInType, int binType, int searchMode) {
   MAKE_BINS_BSEARCH result = NULL;
      
   switch (binType) {
   case NPY_INT8:
      result = GetMakeBinsBSearchPart2<OUT_TYPE, INT8>(numpyInType, searchMode);
      break;
   case NPY_INT16:
      result = GetMakeBinsBSearchPart2<OUT_TYPE, INT16>(numpyInType, searchMode);
      break;
   CASE_NPY_INT32:
      result = GetMakeBinsBSearchPart2<OUT_TYPE, INT32>(numpyInType, searchMode);
      break;
   CASE_NPY_INT64:
      result = GetMakeBinsBSearchPart2<OUT_TYPE, INT64>(numpyInType, searchMode);
      break;
   CASE_NPY_UINT64:
      result = GetMakeBinsBSearchPart2<OUT_TYPE, UINT64>(numpyInType, searchMode);
      break;
   case NPY_FLOAT:
      result = GetMakeBinsBSearchPart2<OUT_TYPE, float>(numpyInType, searchMode);
      break;
   case NPY_DOUBLE:
      result = GetMakeBinsBSearchPart2<OUT_TYPE, double>(numpyInType, searchMode);
      break;

   default:
      //printf("MakeBinsBSearch does not support type %d\n", numpyInType);
      break;
   }
   return result;

}



//===============================================================================
// 1st arg: values (e.g. array of ints or floats) <-- this array needs to be binned
// 2nd arg: bin array of float64 (values to split on) <-- this array must be sorted
// 3rd arg: mode (0= for bin cuts, left =1, right =2)
// Returns BINS -
// NOTE: When out of bounds, it will insert 0
// if < first bin -- you will get 0
// if == first bin -- you will get 1
// if > last bin -- you will get 0
// NOTE: can handle sentinels and nans
// TODO: multithread this
PyObject* BinsToCutsBSearch(PyObject *self, PyObject *args) {
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *binArr1 = NULL;
   
   int searchMode = 0;

   if (!PyArg_ParseTuple(args, "O!O!i",
      &PyArray_Type, &inArr1,
      &PyArray_Type, &binArr1,
      &searchMode)) return NULL;

   int binType = PyArray_TYPE(binArr1);

   int numpyInType = PyArray_TYPE(inArr1);
   INT64 binSize = ArrayLength(binArr1);

   // Choose INT8,INT16,INT32 for bin mode
   int binmode = 0;
   if (binSize > 100) {
      binmode = 1;
      if (binSize > 30000) {
         binmode = 2;
         if (binSize > 2000000000) {
            binmode = 3;
         }
      }
   }

   BOOL isString = FALSE;

   if (numpyInType == binType) {
      // string or unicode comparison
      if (numpyInType == NPY_STRING || numpyInType == NPY_UNICODE) {
         isString = TRUE;
      }
   }

   MAKE_BINS_BSEARCH func = NULL;

   switch (binmode) {
   case 0:
      func = GetMakeBinsBSearch<INT8>(numpyInType, binType, searchMode);
      break;
   case 1:
      func = GetMakeBinsBSearch<INT16>(numpyInType, binType, searchMode);
      break;
   case 2:
      func = GetMakeBinsBSearch<INT32>(numpyInType, binType, searchMode);
      break;
   case 3:
      func = GetMakeBinsBSearch<INT64>(numpyInType, binType, searchMode);
      break;
   }

   if (func != NULL || isString) {
      PyArrayObject* result = NULL;
      switch (binmode) {
      case 0:
         result = AllocateLikeNumpyArray(inArr1, NPY_INT8);
         break;
      case 1:
         result = AllocateLikeNumpyArray(inArr1, NPY_INT16);
         break;
      case 2:
         result = AllocateLikeNumpyArray(inArr1, NPY_INT32);
         break;
      case 3:
         result = AllocateLikeNumpyArray(inArr1, NPY_INT64);
         break;
      }

      if (result) {

         if (isString) {
            // handles string or unicode
            struct BSearchCallbackStruct {
               void* pDataOut;
               void* pDataIn1;
               void* pBin1;
               INT64 binSize;
               int   numpyInType;
               int   binmode;
               INT64 inputItemSize;
               INT64 searchItemSize;
            };

            BSearchCallbackStruct stBSearchCallback;

            // This is the routine that will be called back from multiple threads
            auto lambdaBSearchCallback = [](void* callbackArgT, int core, INT64 start, INT64 length) -> BOOL {
               BSearchCallbackStruct* callbackArg = (BSearchCallbackStruct*)callbackArgT;

               //printf("[%d] Bsearch string %lld %lld\n", core, start, length);
               if (callbackArg->numpyInType == NPY_STRING) {

                  switch (callbackArg->binmode) {
                  case 0:
                     MakeBinsBSearchString<INT8>(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, (char*)callbackArg->pBin1, callbackArg->binSize, callbackArg->inputItemSize, callbackArg->searchItemSize);
                     break;
                  case 1:
                     MakeBinsBSearchString<INT16>(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, (char*)callbackArg->pBin1, callbackArg->binSize, callbackArg->inputItemSize, callbackArg->searchItemSize);
                     break;
                  case 2:
                     MakeBinsBSearchString<INT32>(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, (char*)callbackArg->pBin1, callbackArg->binSize, callbackArg->inputItemSize, callbackArg->searchItemSize);
                     break;
                  case 3:
                     MakeBinsBSearchString<INT64>(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, (char*)callbackArg->pBin1, callbackArg->binSize, callbackArg->inputItemSize, callbackArg->searchItemSize);
                     break;
                  }
               }
               else {
                  //switch (callbackArg->binmode) {
                  //case 0:
                  //   MakeBinsBSearchUnicode<INT8>(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, (UINT32*)callbackArg->pBin1, callbackArg->binSize, callbackArg->inputItemSize, callbackArg->searchItemSize);
                  //   break;
                  //case 1:
                  //   MakeBinsBSearchUnicode<INT16>(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, (UINT32*)callbackArg->pBin1, callbackArg->binSize, callbackArg->inputItemSize, callbackArg->searchItemSize);
                  //   break;
                  //case 2:
                  //   MakeBinsBSearchUnicode<INT32>(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, (UINT32*)callbackArg->pBin1, callbackArg->binSize, callbackArg->inputItemSize, callbackArg->searchItemSize);
                  //   break;
                  //case 3:
                  //   MakeBinsBSearchUnicode<INT64>(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, (UINT32*)callbackArg->pBin1, callbackArg->binSize, callbackArg->inputItemSize, callbackArg->searchItemSize);
                  //   break;
                  //}

               }
               //callbackArg->func(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, callbackArg->pBin1, callbackArg->binSize, callbackArg->numpyInType);
               return TRUE;
            };

            stBSearchCallback.pDataOut = PyArray_BYTES(result);
            stBSearchCallback.pDataIn1 = PyArray_BYTES(inArr1);
            stBSearchCallback.pBin1 = PyArray_BYTES(binArr1);
            stBSearchCallback.binSize = binSize;  // length of the bin
            stBSearchCallback.numpyInType = numpyInType;
            stBSearchCallback.binmode = binmode;
            stBSearchCallback.inputItemSize = PyArray_ITEMSIZE(inArr1);
            stBSearchCallback.searchItemSize = PyArray_ITEMSIZE(binArr1);

            INT64 lengthData = ArrayLength(inArr1);
            g_cMathWorker->DoMultiThreadedChunkWork(lengthData, lambdaBSearchCallback, &stBSearchCallback);

         }
         else {
            // MT callback
            struct BSearchCallbackStruct {
               MAKE_BINS_BSEARCH func;
               void* pDataOut;
               void* pDataIn1;
               void* pBin1;
               INT64 binSize;
               int   numpyInType;
            };

            BSearchCallbackStruct stBSearchCallback;

            // This is the routine that will be called back from multiple threads
            auto lambdaBSearchCallback = [](void* callbackArgT, int core, INT64 start, INT64 length) -> BOOL {
               BSearchCallbackStruct* callbackArg = (BSearchCallbackStruct*)callbackArgT;

               //printf("[%d] Bsearch %lld %lld\n", core, start, length);

               callbackArg->func(callbackArg->pDataIn1, callbackArg->pDataOut, start, length, callbackArg->pBin1, callbackArg->binSize, callbackArg->numpyInType);
               return TRUE;
            };

            stBSearchCallback.pDataOut = PyArray_BYTES(result);
            stBSearchCallback.pDataIn1 = PyArray_BYTES(inArr1);
            stBSearchCallback.pBin1 = PyArray_BYTES(binArr1);
            stBSearchCallback.binSize = binSize;
            stBSearchCallback.numpyInType = numpyInType;
            stBSearchCallback.func = func;

            INT64 lengthData = ArrayLength(inArr1);
            g_cMathWorker->DoMultiThreadedChunkWork(lengthData, lambdaBSearchCallback, &stBSearchCallback);
         }

         return (PyObject*)result;
      }
   }
   PyErr_Format(PyExc_ValueError, "bin array must be int32/int64/float32/float64 -- not %d", PyArray_TYPE(binArr1));

   Py_INCREF(Py_None);
   return Py_None;
}

