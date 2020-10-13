#include "RipTide.h"
#include "ndarray.h"
#include "MultiKey.h"
#include "HashLinear.h"
#include "MathWorker.h"
#include "Sort.h"

#define LOGGING(...)
//#define LOGGING printf

//---------------------------------------------------------------------
//
char* RotateArrays(INT64 tupleSize, ArrayInfo* aInfo) {

   // TODO: Is it fast to calculate the hash here?
   //UINT32* pHashArray = (UINT32*)malloc(sizeof(UINT32) * totalRows);

   INT64 totalItemSize = 0;
   INT64 totalRows = 0;

   totalRows = aInfo[0].ArrayLength;

   for (INT64 i = 0; i < tupleSize; i++) {
      if (aInfo[i].ArrayLength != totalRows) {
         PyErr_Format(PyExc_ValueError, "MultiKeyHash all arrays must be same number of rows %llu", totalRows);
         return NULL;
      }
      totalItemSize += aInfo[i].ItemSize;
   }

   LOGGING("Rotate start at %lld\n", _PyTime_GetSystemClock());

   // Allocate what we need
   char* pSuperArray = (char*)WORKSPACE_ALLOC(totalItemSize * totalRows);

   if (pSuperArray) {

      // THIS CAN BE PARALLELIZED

      INT64 currentRow = 0;
      char* pDest = pSuperArray;

      // We need to build this
      for (INT64 i = 0; i < totalRows; i++) {

         for (INT64 j = 0; j < tupleSize; j++) {
            INT64 itemSize = aInfo[j].ItemSize;
            char* pSrc = aInfo[j].pData;

            // Get to current row
            pSrc += (i*itemSize);

            switch (itemSize) {
            case 8:
               *(UINT64*)pDest = *(UINT64*)pSrc;
               pDest += 8;
               break;
            case 4:
               *(UINT32*)pDest = *(UINT32*)pSrc;
               pDest += 4;
               break;
            case 2:
               *(UINT16*)pDest = *(UINT16*)pSrc;
               pDest += 2;
               break;
            case 1:
               *pDest++ = *pSrc;
               break;
            default:
               memcpy(pDest, pSrc, itemSize);
               pDest += itemSize;
            }

         }

         // While items are fresh, hash it
         //pHashArray[i] = mHash(pDest - totalItemSize, totalItemSize);
      }

   } else {
      PyErr_Format(PyExc_ValueError, "MultiKeyHash out of memory    %llu", totalRows);
   }

   LOGGING("Rotate end at %lld\n", _PyTime_GetSystemClock());
   return pSuperArray;
}

//-------------------------------------------------------------------
// TODO: Make this a class
// Free what was allocated with AllocArrayInfo
void
FreeArrayInfo(ArrayInfo* pAlloc) {
   if (pAlloc) {
      INT64* pRawAlloc = (INT64*)pAlloc;

      // go back one to find where we stuffed the array size
      --pRawAlloc;

      INT64 tupleSize = *pRawAlloc;
      // The next entry is the arrayInfo
      ArrayInfo* aInfo = (ArrayInfo*)&pRawAlloc[1];
      for (INT64 i = 0; i < tupleSize; i++) {
         if (aInfo[i].pOriginalObject) {
            Py_DecRef((PyObject*)aInfo[i].pObject);
         }
      }
      WORKSPACE_FREE(pRawAlloc);
   }
}

//---------------------------------------------------------
// Allocate array info object we can free laater
ArrayInfo*
AllocArrayInfo(INT64 tupleSize) {
   INT64* pRawAlloc = (INT64*)WORKSPACE_ALLOC((sizeof(ArrayInfo)*tupleSize) + sizeof(INT64));
   if (pRawAlloc) {
      // store in first 8 bytes the count
      *pRawAlloc = tupleSize;

      // The next entry is the arrayInfo
      ArrayInfo* aInfo = (ArrayInfo*)&pRawAlloc[1];

      // make sure we clear out pOriginalObject
      for (INT64 i = 0; i < tupleSize; i++) {
         aInfo[i].pOriginalObject = NULL;
      }
      return aInfo;
   }
   return NULL;
}

// Pass in a list or tuple of arrays of the same size
// Returns an array of info (which must be freed later)
// checkrows:
// convert: whether or not to convert non-contiguous arrays
ArrayInfo* BuildArrayInfo(
   PyObject* listObject,
   INT64* pTupleSize,
   INT64* pTotalItemSize,
   BOOL checkrows,
   BOOL convert) {

   bool isTuple = false;
   bool isArray = false;
   bool isList = false;
   INT64 tupleSize = 0;


   if (PyArray_Check(listObject)) {
      isArray = true;
      tupleSize = 1;
   } else

   if (PyTuple_Check(listObject)) {
       isTuple = true;
      tupleSize = PyTuple_GET_SIZE(listObject);
   }
   else 
   if (PyList_Check(listObject))
   {
      isList = true;
      tupleSize = PyList_GET_SIZE(listObject);
   }
   else {

      PyErr_Format(PyExc_ValueError, "BuildArrayInfo must pass in a list or tuple");
      return NULL;
   }

   // NOTE: If the list is empty, this will allocate 0 memory (which C99 says can return NULL
   ArrayInfo* aInfo = AllocArrayInfo(tupleSize);

   INT64 totalItemSize = 0;

   // Build a list of array information so we can rotate it
   for (INT64 i = 0; i < tupleSize; i++) {
      PyObject* inObject=NULL;

      if (isTuple) {
         inObject = PyTuple_GET_ITEM(listObject, i);
      }

      if (isList) {
         inObject = PyList_GetItem(listObject, i);
      }

      if (isArray) {
         inObject = listObject;
      }

      if (inObject == Py_None) {
         // NEW Code to handle none
         aInfo[i].pObject = NULL;
         aInfo[i].ItemSize = 0;
         aInfo[i].NDim = 0;
         aInfo[i].NumpyDType = 0;
         aInfo[i].ArrayLength = 0;
         aInfo[i].pData = NULL;
         aInfo[i].NumBytes = 0;

      } else
      if (PyArray_Check(inObject)) {

         aInfo[i].pObject = (PyArrayObject*)inObject;

         // Check if we need to convert non-contiguous
         if (convert) {
            // If we copy, we have an extra ref count
            inObject = (PyObject*)EnsureContiguousArray((PyArrayObject*)inObject);
            if (!inObject) {
               goto EXIT_ROUTINE;
            }

            if ((PyArrayObject*)inObject != aInfo[i].pObject) {
               // the pObject was copied and needs to be deleted
               // pOriginalObject is the original object
               aInfo[i].pOriginalObject = aInfo[i].pObject;
               aInfo[i].pObject = (PyArrayObject*)inObject;
            }
         }

         aInfo[i].ItemSize = PyArray_ITEMSIZE((PyArrayObject*)inObject);
         aInfo[i].NDim = PyArray_NDIM((PyArrayObject*)inObject);
         aInfo[i].NumpyDType = ObjectToDtype((PyArrayObject*)inObject);
         aInfo[i].ArrayLength = ArrayLength(aInfo[i].pObject);

         if (aInfo[i].NumpyDType == -1) {
            PyErr_Format(PyExc_ValueError, "BuildArrayInfo array has bad dtype of %d", PyArray_TYPE((PyArrayObject*)inObject));
            goto EXIT_ROUTINE;
         }

         INT64 stride0 = PyArray_STRIDE((PyArrayObject*)inObject, 0);
         INT64 itemSize = aInfo[i].ItemSize;

         if (checkrows) {
            if (aInfo[i].NDim != 1) {
               PyErr_Format(PyExc_ValueError, "BuildArrayInfo array must have ndim ==1 instead of %d", aInfo[i].NDim);
               goto EXIT_ROUTINE;
            }
            if (itemSize != stride0) {
               PyErr_Format(PyExc_ValueError, "BuildArrayInfo array strides must match itemsize -- %lld %lld", itemSize, stride0 );
               goto EXIT_ROUTINE;
            }
         }
         else {

            if (itemSize != stride0) {
               // If 2 dims and Fortran, then strides will not match
               // TODO: better check
               if (aInfo[i].NDim == 1) {
                  PyErr_Format(PyExc_ValueError, "BuildArrayInfo without checkows, array strides must match itemsize for 1 dim -- %lld %lld", itemSize, stride0);
                  goto EXIT_ROUTINE;
               }
            }
         }

         if (aInfo[i].ItemSize == 0 || aInfo[i].ArrayLength == 0) {
            LOGGING("**zero size warning BuildArrayInfo: %lld %lld\n", aInfo[i].ItemSize, aInfo[i].ArrayLength);
            //PyErr_Format(PyExc_ValueError, "BuildArrayInfo array must have size");
            //goto EXIT_ROUTINE;
         }
         aInfo[i].pData = (char*)PyArray_BYTES(aInfo[i].pObject);
         aInfo[i].NumBytes = aInfo[i].ArrayLength * aInfo[i].ItemSize;

         LOGGING("Array %llu has %llu bytes  %llu size\n", i, aInfo[i].NumBytes, aInfo[i].ItemSize);
         totalItemSize += aInfo[i].ItemSize;
      }
      else {
         PyErr_Format(PyExc_ValueError, "BuildArrayInfo only accepts numpy arrays");
         goto EXIT_ROUTINE;
      }

   }

   // Don't perform checks for an empty list of arrays;
   // otherwise we'll dereference an empty 'aInfo'.
   if (checkrows && tupleSize > 0) {
      const INT64 totalRows = aInfo[0].ArrayLength;

      for (INT64 i = 0; i < tupleSize; i++) {
         if (aInfo[i].ArrayLength != totalRows) {
            PyErr_Format(PyExc_ValueError, "BuildArrayInfo all arrays must be same number of rows %llu", totalRows);
            goto EXIT_ROUTINE;
         }
      }
   }

   *pTupleSize = tupleSize;
   *pTotalItemSize = totalItemSize;
   return aInfo;

EXIT_ROUTINE:
   *pTupleSize = 0;
   *pTotalItemSize = 0;
   FreeArrayInfo(aInfo);
   return NULL;

}

//----------------------------------------------------------------------
// First arg is list of keys to hash
//
class CMultiKeyPrepare {

public:

   Py_ssize_t tupleSize;
   ArrayInfo* aInfo;
   INT64 totalItemSize;
   INT64 totalRows;
   INT64 hintSize;
   INT64 listSize; // 1 when only one array, > 1 otherwise
   PyArrayObject* pBoolFilterObject;
   bool* pBoolFilter;

   char* pSuperArray ;
   bool bAllocated;

   CMultiKeyPrepare(PyObject* args) {

      aInfo = NULL;
      totalItemSize = 0;
      totalRows = 0;
      hintSize = 0;
      listSize = 0;
      pSuperArray = NULL;
      pBoolFilterObject = NULL;
      pBoolFilter = NULL;
      bAllocated = FALSE;

      tupleSize = PyTuple_GET_SIZE(args);

      LOGGING("MKP Tuple size %llu\n", tupleSize);

      if (tupleSize >= 1) {
         if (tupleSize >= 2) {
            // check for hintSize
            PyObject* longObject = PyTuple_GetItem(args, 1);
            if (PyLong_Check(longObject)) {
               hintSize = PyLong_AsSize_t(longObject);
               LOGGING("Hint size is %llu\n", hintSize);
            }
         }

         if (tupleSize >= 3) {
            // check for filter
            PyObject* filterArray = PyTuple_GetItem(args, 2);
            if (PyArray_Check(filterArray)) {
               pBoolFilterObject = (PyArrayObject*)filterArray;
               pBoolFilter = (bool*)PyArray_BYTES(pBoolFilterObject);
               LOGGING("Bool array is at %p\n", pBoolFilter);
            }
         }

         // Check if they passed in a list
         PyObject* listObject = PyTuple_GetItem(args, 0);
         if (PyList_Check(listObject)) {
            args = listObject;
            tupleSize = PyList_GET_SIZE(args);
            LOGGING("Found list inside tuple size %llu\n", tupleSize);
         }

      }

      aInfo = BuildArrayInfo(args, &listSize, &totalItemSize);

      if (aInfo) {

         totalRows = aInfo[0].ArrayLength;

         for (INT64 i = 0; i < listSize; i++) {
            if (aInfo[i].ArrayLength != totalRows) {
               PyErr_Format(PyExc_ValueError, "MultiKeyHash all arrays must be same number of rows %llu", totalRows);
               totalRows = 0;
            }
         }

         if (pBoolFilterObject) {
            if (PyArray_TYPE(pBoolFilterObject) != NPY_BOOL || ArrayLength(pBoolFilterObject) != totalRows) {
               PyErr_Format(PyExc_ValueError, "MultiKeyHash filter passed must be boolean array of same size %llu", totalRows);
               totalRows = 0;
            }
         }

         if (totalRows != 0) {
            //printf("row width %llu   rows %llu\n", totalItemSize, totalRows);

            if (listSize > 1) {
               bAllocated = TRUE;
               // Make rows
               pSuperArray = RotateArrays(listSize, aInfo);
            }
            else {
               // No need to rotate, just 1 item
               pSuperArray = (char*)PyArray_BYTES(aInfo[0].pObject);
            }

            if (!pSuperArray) {
               printf("MultiKeyHash out of memory    %llu", totalRows);
               PyErr_Format(PyExc_ValueError, "MultiKeyHash out of memory    %llu", totalRows);
            }
         } 
      }
   }

   ~CMultiKeyPrepare() {
      if (aInfo != NULL) {
         FreeArrayInfo(aInfo);
         aInfo = NULL;
      }
      if (bAllocated) {
         WORKSPACE_FREE(pSuperArray);
         pSuperArray = NULL;
      }
   }

};


//-------------------------------------------------------
// Calculate inext from ikey, keep same dtype
// if there is no next it will be the invalid
template<typename KEYTYPE>
void MakeNextKey(
   INT64       mode,
   INT64       numUnique,
   INT64       totalRows,
   void*       pIndexArrayK,
   void*       pNextArrayK)
{

   KEYTYPE* pIndexArray = (KEYTYPE*)pIndexArrayK;
   KEYTYPE* pNextArray  = (KEYTYPE*)pNextArrayK;

   // TODO: option to pass this in
   numUnique += GB_BASE_INDEX;

   // get the invalid for int8/16/32/64
   KEYTYPE invalid = *(KEYTYPE*)GetInvalid<KEYTYPE>();

   INT64 size = sizeof(KEYTYPE) * numUnique;
   KEYTYPE* pGroupArray = (KEYTYPE*)WORKSPACE_ALLOC(size);

   if (pGroupArray) {
      // mark all invalid
      for (INT64 i = 0; i < numUnique; i++) {
         pGroupArray[i] = invalid;
      }

      if (mode == 0) {
         // Go backwards to calc next
         for (INT64 i = totalRows - 1; i >= 0; i--) {
            KEYTYPE group = pIndexArray[i];
            if (group >= 0 && group < numUnique) {
               pNextArray[i] = pGroupArray[group];
               pGroupArray[group] = (KEYTYPE)i;
            }
         }
      }
      else {
         // Go forward to calc previous
         for (INT64 i = 0; i < totalRows; i++) {
            KEYTYPE group = pIndexArray[i];
            if (group >= 0 && group < numUnique) {
               pNextArray[i] = pGroupArray[group];
               pGroupArray[group] = (KEYTYPE)i;
            }
         }
      }
      WORKSPACE_FREE(pGroupArray);
   }
}


//-------------------------------------------------------
// Input:
// Arg1: ikey numpy array from lexsort or grouping.iGroup
//       array must be integers
//       array must have integers only from 0 to len(arr)-1
//       all values must be unique, then it can be reversed quickly
//
// Arg2: unique_rows
// Arg3: mode  0= next, 1= prev
//
// Output:
//      Returns index array with next or previous
PyObject *
MakeiNext(PyObject *self, PyObject *args)
{
   PyArrayObject *ikey = NULL;
   INT64 unique_rows = 0;
   INT64 mode = 0;

   if (!PyArg_ParseTuple(
      args, "O!LL",
      &PyArray_Type, &ikey,
      &unique_rows,
      &mode
      )) {
      return NULL;
   }

   int dtype = PyArray_TYPE(ikey);

   // check for only signed ints
   if (dtype >= 10 || (dtype & 1) == 0) {
      PyErr_Format(PyExc_ValueError, "MakeINext: ikey must be int8/16/32/64");
      return NULL;
   }

   PyArrayObject *pReturnArray = AllocateLikeNumpyArray(ikey, dtype);

   if (pReturnArray) {
      INT64 arrlength = ArrayLength(ikey);
      void* pIKey = PyArray_BYTES(ikey);
      void* pOutKey = PyArray_BYTES(pReturnArray);

      switch (PyArray_ITEMSIZE(ikey)) {
      case 1:
         MakeNextKey<INT8>(mode, unique_rows, arrlength, pIKey, pOutKey);
         break;
      case 2:
         MakeNextKey<INT16>(mode, unique_rows, arrlength, pIKey, pOutKey);
         break;
      case 4:
         MakeNextKey<INT32>(mode, unique_rows, arrlength, pIKey, pOutKey);
         break;
      case 8:
         MakeNextKey<INT64>(mode, unique_rows, arrlength, pIKey, pOutKey);
         break;

      default:
         PyErr_Format(PyExc_ValueError, "MakeINext: ikey must be int8/16/32/64");
         return NULL;
      }

      return (PyObject*)pReturnArray;
   }

   PyErr_Format(PyExc_ValueError, "MakeINext: ran out of memory");
   return NULL;
}


//-------------------------------------------------------
// When next was not calculated
template<typename K>
INT32* GroupByPackFixup32(
   INT64 numUnique,
   INT64 totalRows,
   void* pIndexArrayK,
   INT32* pGroupArray)
{

   K* pIndexArray = (K*)pIndexArrayK;

   //printf("%lld %lld\n", numUnique, totalRows);
   //for (INT64 i = 0; i < totalRows; i++) {
   //   printf("%d ", (int)pIndexArray[i]);
   //}
   //printf("\n");

   // reserve for invalid bin
   numUnique += GB_BASE_INDEX;

   INT64 size = sizeof(INT32) * totalRows;
   INT32* pNextArray = (INT32*)WORKSPACE_ALLOC(size);

   // mark all invalid
   for (INT32 i = 0; i < numUnique; i++) {
      pGroupArray[i] = GB_INVALID_INDEX;
   }

   // Go backwards
   for (INT32 i = (INT32)totalRows - 1; i >= 0; i--) {
      K group = pIndexArray[i];

      pNextArray[i] = pGroupArray[group];

      //printf("%d  - group %d next %d\n", i, (int)group, pNextArray[i]);
      pGroupArray[group] = i;
   }

   return pNextArray;

}



//-----------------------------------------------
// Used for custom function
// Input from return values of MultiKeyGroupBy32
//
// Input: pIndexArrayK (templatized allowed INT8/16/32/64)
//
// Returns:
//   pSortArray  [totalRows]- array where the groupings are next to eachother
//                thus, mbget can be called to pull all grouping values together
//   pFirstArray [numUnique]- an index into pSortArray where the grouping starts for the unique key
//   pCountArray [numUnique]- paired with pFirstArray -- the count for that unique key
template<typename K>
BOOL GroupByPackFinal32(
   INT64 numUnique,
   INT64 totalRows,
   void* pIndexArrayK,
   INT32* pNextArray,
   INT32* pGroupArray,
   PyObject** ppSortArray,
   PyObject** ppFirstArray,
   PyObject** ppCountArray)

{

   K* pIndexArray = (K*)pIndexArrayK;

   // reserve for invalid bin
   numUnique += GB_BASE_INDEX;

   //-----------------------------------------------------
   // sortArray is iGroup in python land
   PyArrayObject* sortArray =  AllocateNumpyArray(1, (npy_intp*)&totalRows, NPY_INT32);
   PyArrayObject* firstArray = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT32);
   PyArrayObject* countArray = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT32);

   if (sortArray && firstArray && countArray) {

      // Build the first and count array
      INT32* pSortArray = (INT32*)PyArray_BYTES(sortArray);
      INT32* pFirstArray = (INT32*)PyArray_BYTES(firstArray);
      INT32* pCountArray = (INT32*)PyArray_BYTES(countArray);

      INT32 i = (INT32)totalRows;

      // TODO -- how to handle empty?
      pSortArray[0] = GB_INVALID_INDEX;
      pFirstArray[0] = GB_INVALID_INDEX;
      pCountArray[0] = 0;

      // fills up sort array
      INT32 runningCount = 0;

      // Check if caller knows the first
      if (pGroupArray) {
         // Next unique id to look for
         INT32 lookfor = GB_BASE_INDEX;

         // Second pass thru array
         for (INT32 lookfor = 0; lookfor < numUnique; lookfor++) {

            // Get head of the unique id (the key)
            INT32 j = pGroupArray[lookfor];

            //printf("%d head is at %d\n", lookfor, j);

            INT32 count = runningCount;

            // Keep track of sorting
            pFirstArray[lookfor] = runningCount;

            // We are at the head of the list, now count up how many
            while (j != GB_INVALID_INDEX) {
               pSortArray[runningCount++] = j;
               j = pNextArray[j];
            }

            // store the count
            pCountArray[lookfor] = runningCount - count;
         }
         LOGGING("running count at end %d\n", runningCount);

      }
      else {
         // Check if first item is filtered out
         // If so, go ahead and chain this now
         if (pIndexArray[0] == 0) {
            INT32 count = runningCount;

            // Keep track of sorting
            pFirstArray[0] = runningCount;
            pSortArray[runningCount++] = 0;

            INT32 j = pNextArray[0];

            // We are at the head of the list, now count up how many
            while (j != GB_INVALID_INDEX) {
               pSortArray[runningCount++] = j;
               j = pNextArray[j];
            }

            // store the count
            pCountArray[0] = runningCount - count;

         }

         // Next unique id to look for
         INT32 lookfor = GB_BASE_INDEX;

         // Second pass thru array
         for (INT32 i = 0; i < totalRows; i++) {

            // Check if we found the head of the unique id (the key)
            if (pIndexArray[i] == lookfor) {
               INT32 count = runningCount;

               // Keep track of sorting
               pFirstArray[lookfor] = runningCount;
               pSortArray[runningCount++] = i;

               INT32 j = pNextArray[i];

               // We are at the head of the list, now count up how many
               while (j != GB_INVALID_INDEX) {
                  pSortArray[runningCount++] = j;
                  j = pNextArray[j];
               }

               // store the count
               pCountArray[lookfor] = runningCount - count;

               ++lookfor;
            }
         }

         LOGGING("running count %d\n", runningCount);

         // Check if first item was not filtered out
         if (pIndexArray[0] != 0) {

            // Have to search for invalid bins
            for (INT32 i = 0; i < totalRows; i++) {

               // Check if we found the head of the invalid unique id (the key)
               if (pIndexArray[i] == 0) {
                  INT32 count = runningCount;

                  LOGGING("first invalid found at %d, current count %d\n", i, runningCount);

                  // Keep track of sorting
                  pFirstArray[0] = runningCount;
                  pSortArray[runningCount++] = i;

                  INT32 j = pNextArray[i];

                  // We are at the head of the list, now count up how many
                  while (j != GB_INVALID_INDEX) {
                     pSortArray[runningCount++] = j;
                     j = pNextArray[j];
                  }

                  // store the count
                  pCountArray[0] = runningCount - count;
                  break;
               }
            }

         }
      }

      *ppSortArray = (PyObject*)sortArray;
      *ppFirstArray = (PyObject*)firstArray;
      *ppCountArray = (PyObject*)countArray;
      return TRUE;
   }

   CHECK_MEMORY_ERROR(0);

   // known possible memory leak here
   return FALSE;
}


//-------------------------------
// reorgranize the iKey array so that all the groupings (aka bins) are together
// then mbget can be called and a custom function can be called
//
// Input:
//   iKey array (OR index array)
//   iNext array (optional) (may pass None)
//   unique count (number of unique)
// Returns:
//   iGroup / or sort  (fancy index)
//   iFirst
//   nCount
//
PyObject*
GroupByPack32(PyObject* self, PyObject* args) {

   PyArrayObject *indexArray = NULL;
   PyObject *nextArray = NULL;

   INT64 numUnique = 0;

   if (!PyArg_ParseTuple(
      args, "O!OL",
      &PyArray_Type, &indexArray,
      &nextArray,
      &numUnique)) {

      return NULL;
   }

   try {
      INT64 totalRows = ArrayLength(indexArray);
      INT32 numpyIndexType = PyArray_TYPE(indexArray);
      void* pIndexArray = (void*)PyArray_BYTES(indexArray);

      INT32* pNextArray = NULL;

      BOOL bMustFree = FALSE;

      INT32* pGroupArray = NULL;

      if (!PyArray_Check(nextArray)) {
         // Next was not supplied and must be calculated
         bMustFree = TRUE;

         INT64 allocsize = sizeof(INT32) * (numUnique + GB_BASE_INDEX);
         pGroupArray = (INT32*)WORKSPACE_ALLOC(allocsize);

         switch (numpyIndexType) {
         case NPY_INT8:
            pNextArray = GroupByPackFixup32<INT8>(numUnique, totalRows, pIndexArray, pGroupArray);
            break;
         case NPY_INT16:
            pNextArray = GroupByPackFixup32<INT16>(numUnique, totalRows, pIndexArray, pGroupArray);
            break;
         CASE_NPY_INT32:
            pNextArray = GroupByPackFixup32<INT32>(numUnique, totalRows, pIndexArray, pGroupArray);
            break;
         CASE_NPY_INT64:
            pNextArray = GroupByPackFixup32<INT64>(numUnique, totalRows, pIndexArray, pGroupArray);
            break;
         default:
            PyErr_Format(PyExc_ValueError, "GroupByPack32 index must be int8 int16, int32, int64");
            return NULL;
         }
      }
      else {
         if (totalRows != ArrayLength((PyArrayObject*)nextArray)) {
            PyErr_Format(PyExc_ValueError, "GroupByPack32 array length does not match %llu", totalRows);
            return NULL;
         }
         pNextArray = (INT32*)PyArray_BYTES((PyArrayObject*)nextArray);
      }

      PyObject* sortGroupArray = NULL;
      PyObject* firstArray = NULL;
      PyObject* countArray = NULL;

      BOOL bResult = FALSE;

      switch (numpyIndexType) {
      case NPY_INT8:
         bResult = GroupByPackFinal32<INT8>(numUnique, totalRows, pIndexArray, pNextArray, pGroupArray, &sortGroupArray, &firstArray, &countArray);
         break;
      case NPY_INT16:
         bResult = GroupByPackFinal32<INT16>(numUnique, totalRows, pIndexArray, pNextArray, pGroupArray, &sortGroupArray, &firstArray, &countArray);
         break;
      CASE_NPY_INT32:
         bResult = GroupByPackFinal32<INT32>(numUnique, totalRows, pIndexArray, pNextArray, pGroupArray, &sortGroupArray, &firstArray, &countArray);
         break;
      CASE_NPY_INT64:
         bResult = GroupByPackFinal32<INT64>(numUnique, totalRows, pIndexArray, pNextArray, pGroupArray, &sortGroupArray, &firstArray, &countArray);
         break;
      default:
         PyErr_Format(PyExc_ValueError, "GroupByPack32 index must be int8 int16, int32, int64");
         return NULL;
      }

      if (bMustFree) {
         WORKSPACE_FREE(pNextArray);
         WORKSPACE_FREE(pGroupArray);
      }

      if (bResult) {
         // Build tuple to return
         PyObject* retObject = Py_BuildValue("(OOO)", sortGroupArray, firstArray, countArray);
         Py_DECREF((PyObject*)sortGroupArray);
         Py_DECREF((PyObject*)firstArray);
         Py_DECREF((PyObject*)countArray);
         return (PyObject*)retObject;
      }
   }
   catch (...) {

   }

   PyErr_Format(PyExc_ValueError, "GroupByPack32 failed internally");
   return NULL;

}



// ---------------------------------------------------------------------------------- -
// Multikey hash for groupby
// Input:
//     Arg1: a list of numpy arrays to hash on (multikey) - all arrays must be the same size
//     Arg2: <optional> set to 0 for default, an integer hint if the number of unique keys is known in advance
//     Arg3: <optional> set to Py_None for default, a BOOLEAN array of what to filter out
//     Arg4: <optional> hash mode (defaults to 2 which is hash mask)
//
// kwarg: cutoffs  INT64 cutoff array for parallel mode
// Return 1 numpy array + 1 long
// iKey -- unique group that each row belongs to
// nCount -- number of unique groups
//
PyObject *
MultiKeyGroupBy32(PyObject *self, PyObject *args, PyObject *kwargs) {

   long hashMode = HASH_MODE::HASH_MODE_MASK;
   BOOL parallelMode = FALSE;

   if (!PyTuple_Check(args)) {
      PyErr_Format(PyExc_ValueError, "MultiKeyGroupBy32 arguments needs to be a tuple");
      return NULL;
   }

   Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

   if (tupleSize < 3) {
      PyErr_Format(PyExc_ValueError, "MultiKeyGroupBy32 only has %llu args but requires 3", tupleSize);
      return NULL;
   }

   if (tupleSize == 4) {
      PyObject* hashmodeobject = PyTuple_GetItem(args, 3);

      if (PyLong_Check(hashmodeobject)) {
         hashMode = PyLong_AsLong(hashmodeobject);
         LOGGING("hashmode is %ld\n", hashMode);
      }
   }

   try {

      INT32 numpyIndexType = NPY_INT32;

      // Rotate the arrays
      CMultiKeyPrepare mkp(args);

      if (mkp.totalRows > 2100000000) {
         numpyIndexType = NPY_INT64;
         LOGGING("gb 64bit mode  hintsize:%lld\n", mkp.hintSize);
      }

      INT64    cutOffLength = 0;
      INT64*   pCutOffs = GetCutOffs(kwargs, cutOffLength);

      if (pCutOffs && pCutOffs[cutOffLength - 1] != mkp.totalRows) {
         PyErr_Format(PyExc_ValueError, "MultiKeyGroupBy32 last cutoff length does not match array length %lld", mkp.totalRows);
         return NULL;
      }
      if (cutOffLength == -1) {
         PyErr_Format(PyExc_ValueError, "MultiKeyGroupBy32 'cutoffs' must be an array of type INT64");
         return NULL;
      }

      if (mkp.pSuperArray && mkp.listSize > 0) {

         PyArrayObject* indexArray = AllocateLikeNumpyArray(mkp.aInfo[0].pObject, numpyIndexType);

         if (indexArray == NULL) {
            PyErr_Format(PyExc_ValueError, "MultiKeyGroupBy32 out of memory    %llu", mkp.totalRows);
            return NULL;
         }
         else {

            //printf("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);
            // now one based, so zero reserved
            // TODO: use recycled memory?
            PyArrayObject* firstArray = NULL;

            INT64 numUnique = 0;

            // default to unknown core type
            int coreType = -1;
            if (mkp.listSize == 1) {
               // if just one element in list, its type is the core type
               coreType = mkp.aInfo[0].NumpyDType;
            }

            LOGGING("Starting mkgp32  indxtype: %d  rows: %lld   itemsize: %lld  hintsize: %lld  filter: %p  coreType:%d\n", numpyIndexType, mkp.totalRows, mkp.totalItemSize, mkp.hintSize, mkp.pBoolFilter, coreType);
            LOGGING("mkgp32 parallelmode: %p\n", pCutOffs);

            GROUPBYCALL pGroupByCall = NULL;
            void* pIndexArray = PyArray_BYTES(indexArray);

            if (numpyIndexType == NPY_INT32) {
               pGroupByCall = GroupBy32;
            }
            else {
               pGroupByCall = GroupBy64;
            }

            numUnique = (INT64)
               pGroupByCall(
                  cutOffLength,
                  pCutOffs,
                  mkp.totalRows, mkp.totalItemSize, (const char*)mkp.pSuperArray,
                  coreType,          // set to -1 for unknown
                  pIndexArray,
                  &firstArray,
                  HASH_MODE(hashMode),
                  mkp.hintSize,
                  mkp.pBoolFilter);

            PyObject* retObject = Py_BuildValue("(OOL)", indexArray, firstArray, numUnique);
            Py_DECREF((PyObject*)indexArray);
            Py_DECREF((PyObject*)firstArray);

            //PyObject* retObject = Py_BuildValue("(OOL)", indexArray, Py_None, numUnique);
            //Py_DECREF((PyObject*)indexArray);
            LOGGING("mkgp32 returning %lld\n", numUnique);
            return (PyObject*)retObject;
         }
      }
      else {
         // error should already be set
         return NULL;
      }
   }
   catch (...) {

   }

   PyErr_Format(PyExc_ValueError, "MultiKeyGroupBy32 failed in multikey prepare:  %llu", tupleSize);
   return NULL;

}


//-----------------------------------------------------------------------------------
// Multikey SUPER hash for groupby
// Input:
//     Arg1: a list of numpy arrays to hash on (multikey) - all arrays must be the same size
//     Arg2: <optional> set to 0 for default, an integer hint if the number of unique keys is known in advance
//     Arg3: <optional> st to Py_None for defauly, a BOOLEAN array of what to filter out
//     Arg4: <optional> hash mode (defaults to 2 which is hash mask)
//
// Return 4 numpy arrays
// iKey -- unique group that each row belongs to
// iNext -- the next index in the group (size is same as original array)
// iFirst -- index into first location into the group
// nCount -- count of how many in the group
//
PyObject *
MultiKeyGroupBy32Super(PyObject *self, PyObject *args) {

   long hashMode = HASH_MODE::HASH_MODE_MASK;

   if (!PyTuple_Check(args)) {
      PyErr_Format(PyExc_ValueError, "MultiKeyHash arguments needs to be a tuple");
      return NULL;
   }

   Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

   if (tupleSize < 3) {
      PyErr_Format(PyExc_ValueError, "MultiKeyHash only has %llu args but requires 3", tupleSize);
      return NULL;
   }

   if (tupleSize == 4) {
      PyObject* hashmodeobject = PyTuple_GetItem(args,3);

      if (PyLong_Check(hashmodeobject)) {
         hashMode = PyLong_AsLong(hashmodeobject);
         LOGGING("hashmode is %ld\n", hashMode);
      }

   }

   try {

      INT32 numpyIndexType = NPY_INT32;

      CMultiKeyPrepare mkp(args);

      if (mkp.totalRows > 2100000000) {
         PyErr_Format(PyExc_ValueError, "MultiKeyHash exceeding 32bit limits %llu", mkp.totalRows);
      }

      // Rotate the arrays

      if (mkp.pSuperArray && mkp.listSize > 0) {

         PyArrayObject* indexArray = AllocateLikeNumpyArray(mkp.aInfo[0].pObject, numpyIndexType);
         PyArrayObject* nextArray = AllocateLikeNumpyArray(mkp.aInfo[0].pObject, numpyIndexType);

         if (nextArray == NULL) {
            PyErr_Format(PyExc_ValueError, "MultiKeyHash out of memory    %llu", mkp.totalRows);
         }
         else {

            //printf("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

            INT32* pIndexArray = (INT32*)PyArray_BYTES(indexArray);
            INT32* pNextArray = (INT32*)PyArray_BYTES(nextArray);

            // now one based, so zero reserved
            INT32* pUniqueArray = (INT32*)WORKSPACE_ALLOC((mkp.aInfo[0].ArrayLength + 1) * sizeof(INT32));
            INT32* pUniqueCountArray = (INT32*)WORKSPACE_ALLOC((mkp.aInfo[0].ArrayLength + 1) * sizeof(INT32));

            INT64 numUnique = 0;

            // default to unknown core type
            int coreType = -1;
            if (mkp.listSize == 1) {
               // if just one element in list, its type is the core type
               coreType = mkp.aInfo[0].NumpyDType;
            }
            LOGGING("Starting hash  indxtype: %d  rows: %lld   itemsize: %lld  hintsize: %lld  filter: %p  coreType:%d\n", numpyIndexType, mkp.totalRows, mkp.totalItemSize, mkp.hintSize, mkp.pBoolFilter, coreType);

            Py_BEGIN_ALLOW_THREADS

               numUnique = (INT64)
               GroupBy32Super(
                  mkp.totalRows, mkp.totalItemSize, (const char*)mkp.pSuperArray,
                  coreType,
                  pIndexArray,
                  pNextArray,
                  pUniqueArray,
                  pUniqueCountArray,
                  HASH_MODE(hashMode),
                  mkp.hintSize,
                  mkp.pBoolFilter);

            Py_END_ALLOW_THREADS

               // now one based, so zero reserved
               //++numUnique;

               // Once we know the number of unique, we can allocate the smaller array
               PyArrayObject* uniqueArray = AllocateNumpyArray(1, (npy_intp*)&numUnique, numpyIndexType);
            PyArrayObject* uniqueCountArray = AllocateNumpyArray(1, (npy_intp*)&numUnique, numpyIndexType);

            CHECK_MEMORY_ERROR(uniqueArray);
            CHECK_MEMORY_ERROR(uniqueCountArray);

            if (uniqueArray != NULL && uniqueCountArray != NULL) {
               INT32* pUniqueArrayDest = (INT32*)PyArray_BYTES(uniqueArray);
               memcpy(pUniqueArrayDest, pUniqueArray, numUnique * sizeof(INT32));
               INT32* pUniqueCountArrayDest = (INT32*)PyArray_BYTES(uniqueCountArray);
               memcpy(pUniqueCountArrayDest, pUniqueCountArray, numUnique * sizeof(INT32));
            }
            WORKSPACE_FREE(pUniqueArray);
            WORKSPACE_FREE(pUniqueCountArray);

            //printf("--MakeSlices...");
            //// OPTIONAL CALL
            //PyObject* sortArray = NULL;
            //PyObject* firstArray = NULL;
            //PyObject* countArray = NULL;
            //GroupByPackFinal32(numUnique, totalRows, pIndexArray, pNextArray, &sortArray, &firstArray, &countArray);

            //printf("--done...\n");

            // Build tuple to return
            PyObject* retObject = Py_BuildValue("(OOOO)", indexArray, nextArray, uniqueArray, uniqueCountArray);
            Py_DECREF((PyObject*)indexArray);
            Py_DECREF((PyObject*)nextArray);
            Py_DECREF((PyObject*)uniqueArray);
            Py_DECREF((PyObject*)uniqueCountArray);

            return (PyObject*)retObject;
         }
      }
   }
   catch (...) {

   }

   PyErr_Format(PyExc_ValueError, "MultiKeySuper failed in multikey prepare:  %llu", tupleSize);
   return NULL;

}




//-----------------------------------------------------------------------------------
// Multikey hash for unique items
//     Arg1: a list of numpy arrays to hash on (multikey) - all arrays must be the same size
//     Arg2: <optional> set to 0 for default, an integer hint if the number of unique keys is known in advance
//     Arg3: <optional> set to Py_None for default, a BOOLEAN array of what to filter out
// Returns 1 numpy arrays
//     index location of first found unique
//
PyObject *
MultiKeyUnique32(PyObject *self, PyObject *args) {

   if (!PyTuple_Check(args)) {
      PyErr_Format(PyExc_ValueError, "MultiKeyUnique32 arguments needs to be a tuple");
      return NULL;
   }

   Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

   if (tupleSize < 1) {
      PyErr_Format(PyExc_ValueError, "MultiKeyUnique32 only has %llu args", tupleSize);
      return NULL;
   }

   try {

      CMultiKeyPrepare mkp(args);

      if (mkp.pSuperArray) {

         if (mkp.totalRows < 2100000000) {
            // worst case alloc
            INT32* pIndexArray = (INT32*)WORKSPACE_ALLOC(mkp.totalRows * sizeof(INT32));

            // worst case alloc
            INT32* pCountArray = (INT32*)WORKSPACE_ALLOC(mkp.totalRows * sizeof(INT32));

            if (pIndexArray == NULL || pCountArray == NULL) {
               PyErr_Format(PyExc_ValueError, "MultiKeyUnique32 out of memory    %llu", mkp.totalRows);
               return NULL;
            }
            else {
               //printf("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

               LOGGING("Starting hash unique\n");

               INT64 numUnique = (INT64)
                  Unique32(
                     mkp.totalRows, mkp.totalItemSize, (const char*)mkp.pSuperArray,
                     pIndexArray,
                     pCountArray,
                     HASH_MODE::HASH_MODE_MASK,
                     mkp.hintSize,
                     mkp.pBoolFilter);

               // We allocated for worst case, now copy over only the unique indexes
               PyArrayObject* indexArray2 = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT32);
               CHECK_MEMORY_ERROR(indexArray2);

               if (indexArray2 != NULL) {
                  INT32* pIndexArray2 = (INT32*)PyArray_BYTES(indexArray2);
                  memcpy(pIndexArray2, pIndexArray, numUnique * sizeof(INT32));
               }

               PyArrayObject* countArray2 = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT32);
               CHECK_MEMORY_ERROR(countArray2);

               if (countArray2 != NULL) {
                  INT32* pCountArray2 = (INT32*)PyArray_BYTES(countArray2);
                  memcpy(pCountArray2, pCountArray, numUnique * sizeof(INT32));
               }

               // free the side array
               WORKSPACE_FREE(pIndexArray);
               WORKSPACE_FREE(pCountArray);
               PyObject* retObject = Py_BuildValue("(OO)", indexArray2, countArray2);
               Py_DECREF((PyObject*)indexArray2);
               Py_DECREF((PyObject*)countArray2);

               return (PyObject*)retObject;
            }
         }
         else {
            // worst case alloc
            INT64* pIndexArray = (INT64*)WORKSPACE_ALLOC(mkp.totalRows * sizeof(INT64));

            // worst case alloc
            INT64* pCountArray = (INT64*)WORKSPACE_ALLOC(mkp.totalRows * sizeof(INT64));

            if (pIndexArray == NULL || pCountArray == NULL) {
               PyErr_Format(PyExc_ValueError, "MultiKeyUnique64 out of memory    %llu", mkp.totalRows);
               return NULL;
            }
            else {
               //printf("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

               LOGGING("Starting hash64 unique\n");

               INT64 numUnique = (INT64)
                  Unique64(
                     mkp.totalRows, mkp.totalItemSize, (const char*)mkp.pSuperArray,
                     pIndexArray,
                     pCountArray,
                     HASH_MODE::HASH_MODE_MASK,
                     mkp.hintSize,
                     mkp.pBoolFilter);

               // We allocated for worst case, now copy over only the unique indexes
               PyArrayObject* indexArray2 = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT64);
               CHECK_MEMORY_ERROR(indexArray2);

               if (indexArray2 != NULL) {
                  INT64* pIndexArray2 = (INT64*)PyArray_BYTES(indexArray2);
                  memcpy(pIndexArray2, pIndexArray, numUnique * sizeof(INT64));
               }

               PyArrayObject* countArray2 = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT64);
               CHECK_MEMORY_ERROR(countArray2);

               if (countArray2 != NULL) {
                  INT64* pCountArray2 = (INT64*)PyArray_BYTES(countArray2);
                  memcpy(pCountArray2, pCountArray, numUnique * sizeof(INT64));
               }

               // free the side array
               WORKSPACE_FREE(pIndexArray);
               WORKSPACE_FREE(pCountArray);
               PyObject* retObject = Py_BuildValue("(OO)", indexArray2, countArray2);
               Py_DECREF((PyObject*)indexArray2);
               Py_DECREF((PyObject*)countArray2);

               return (PyObject*)retObject;
            }

         }
      }
   }
   catch (...) {

   }

   Py_INCREF(Py_None);
   return Py_None;

}


//-----------------------------------------------------------------------------------
// Input:
//   First arg is tuple that can hold one or more numpy arrays
//   Second arg is eiher hintsize or if -1, indicates to delete the hash object
//   Third arg: optional and is the hash object to resused
// Example:
//   # first roll
//   roll=rc.MultiKeyRolling((z.Header_InstrumentId, z.PerExp_Expiration), 1_000_000)
//   # keep rolling
//   roll2=rc.MultiKeyRolling((z.Header_InstrumentId, z.PerExp_Expiration), 0, roll[3])
//   # delete
//   roll2=rc.MultiKeyRolling((z.Header_InstrumentId, z.PerExp_Expiration), -1, roll[3])
// Returns:
//   First array is the bin #
//   Second array is the number of uniques so far for that bin
//   Third how many uniques so far
//   Fourth ptr to hash object which can be passed as 3rd input arg
PyObject *
MultiKeyRolling(PyObject *self, PyObject *args)
{
   INT64 hintSize = 1000000;
   INT64 memPointer = 0;
   UINT64 numUnique = 0;

   Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

   if (tupleSize != 2 && tupleSize != 3) {
      PyErr_Format(PyExc_ValueError, "MultiKeyRolling only %llu args, but requires exactly 2 or 3 args.", tupleSize);
      return NULL;
   }

   PyObject* firstArg = PyTuple_GET_ITEM(args, 0);
   PyObject* secondArg = PyTuple_GET_ITEM(args, 1);
   if (PyLong_Check(secondArg)) {
      hintSize = PyLong_AsLongLong(secondArg);
   }
   else {
      PyErr_Format(PyExc_ValueError, "MultiKeyRolling second arg must be an integer.");
      return NULL;
   }
   if (tupleSize == 3) {
      PyObject* thirdArg = PyTuple_GET_ITEM(args, 2);
      if (PyLong_Check(thirdArg)) {
         memPointer = PyLong_AsLongLong(thirdArg);
         if (hintSize == -1) {
            MultiKeyRollingStep2Delete((void*)memPointer);
            RETURN_NONE;
         }
      }
   }

   CMultiKeyPrepare mkp(firstArg);

   if (mkp.pSuperArray) {
      PyArrayObject* firstObject = mkp.aInfo[0].pObject;

      PyArrayObject* indexArray = AllocateLikeNumpyArray(firstObject, NPY_INT64);
      PyArrayObject* runningCountArray = AllocateLikeNumpyArray(firstObject, NPY_INT64);

      INT64* pIndexArray = (INT64*)PyArray_BYTES(indexArray);
      INT64* pRunningCountArray = (INT64*)PyArray_BYTES(runningCountArray);

      // Turn off caching because we want this to remain
      BOOL prevValue = g_cMathWorker->NoCaching;
      g_cMathWorker->NoCaching = TRUE;

      void* pKeepRolling = 
      MultiKeyRollingStep2(
         mkp.totalRows, mkp.totalItemSize, (const char*)mkp.pSuperArray,
         pIndexArray,
         pRunningCountArray,
         HASH_MODE::HASH_MODE_MASK,
         hintSize,
         &numUnique,
         (void*)memPointer);

      g_cMathWorker->NoCaching = prevValue;

      // Build tuple to return
      PyObject* retObject = Py_BuildValue("(OOLL)", indexArray, runningCountArray, (INT64)numUnique, (INT64)pKeepRolling);

      Py_DECREF((PyObject*)indexArray);
      Py_DECREF((PyObject*)runningCountArray);
      return (PyObject*)retObject;
   }

   RETURN_NONE;
}


//-----------------------------------------------------------------------------------
// Called when two arrays are used as input params
// GreaterThan/LessThan
PyObject *
MultiKeyHash(PyObject *self, PyObject *args)
{
 
   Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

   if (tupleSize < 1) {
      PyErr_Format(PyExc_ValueError, "MultiKeyHash only %llu args", tupleSize);
      return NULL;
   }

   CMultiKeyPrepare mkp(args);
  
   if (mkp.pSuperArray) {

      PyArrayObject* firstObject = mkp.aInfo[0].pObject;

      PyArrayObject* indexArray = AllocateLikeNumpyArray(firstObject, NPY_INT32);
      PyArrayObject* runningCountArray = AllocateLikeNumpyArray(firstObject, NPY_INT32);
      PyArrayObject* prevArray = AllocateLikeNumpyArray(firstObject, NPY_INT32);
      PyArrayObject* nextArray = AllocateLikeNumpyArray(firstObject, NPY_INT32);

      // Second pass
      PyArrayObject* firstArray = AllocateLikeNumpyArray(firstObject, NPY_INT32);
      PyArrayObject* bktSizeArray = AllocateLikeNumpyArray(firstObject, NPY_INT32);
      PyArrayObject* lastArray = AllocateLikeNumpyArray(firstObject, NPY_INT32);

      if (lastArray == NULL) {
         PyErr_Format(PyExc_ValueError, "MultiKeyHash out of memory    %llu", mkp.totalRows);
      }
      else {

         //printf("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

         INT32* pIndexArray = (INT32*)PyArray_BYTES(indexArray);
         INT32* pRunningCountArray = (INT32*)PyArray_BYTES(runningCountArray);
         INT32* pPrevArray = (INT32*)PyArray_BYTES(prevArray);
         INT32* pNextArray = (INT32*)PyArray_BYTES(nextArray);


         INT32* pFirstArray = (INT32*)PyArray_BYTES(firstArray);
         INT32* pBktSize = (INT32*)PyArray_BYTES(bktSizeArray);
         INT32* pLastArray = (INT32*)PyArray_BYTES(lastArray);

         MultiKeyHash32(
            mkp.totalRows, mkp.totalItemSize, (const char*)mkp.pSuperArray,
            pIndexArray,
            pRunningCountArray,
            pPrevArray,
            pNextArray,
            pFirstArray,
            HASH_MODE::HASH_MODE_MASK,
            mkp.hintSize,
            mkp.pBoolFilter);
         
         assert(mkp.totalRows < 2100000000);

         INT32 i = (INT32)mkp.totalRows;

         while (i > 0) {
            --i;
            // Search for last item, we know we are the end when points to invalid index
            if (pNextArray[i] == GB_INVALID_INDEX) {

               // now we now last and total
               INT32 count = pRunningCountArray[i];
               pBktSize[i] = count;
               pLastArray[i] = i;

               INT64 j = i;

               while (pPrevArray[j] != GB_INVALID_INDEX) {
                  // walk the previous link until we reach the end
                  j = pPrevArray[j];
                  pBktSize[j] = count;
                  pLastArray[j] = i;
               }
            }

         }

      }

      // Build tuple to return
      PyObject* retObject = Py_BuildValue("(OOOOOOO)", indexArray, runningCountArray, bktSizeArray, prevArray, nextArray, firstArray, lastArray);

      Py_DECREF((PyObject*)indexArray);
      Py_DECREF((PyObject*)runningCountArray);
      Py_DECREF((PyObject*)bktSizeArray);
      Py_DECREF((PyObject*)prevArray);
      Py_DECREF((PyObject*)nextArray);
      Py_DECREF((PyObject*)firstArray);
      Py_DECREF((PyObject*)lastArray);

      return (PyObject*)retObject;

   }
   Py_INCREF(Py_None);
   return Py_None;

}




//-----------------------------------------------------------------------------------
// Called when two arrays are used as input params
// First arg: a tuple of numpy objects
// Second arg: another tuple of numpy objects
// Third arg: optional hintSize
// Returns two arrays bool and index
PyObject *
MultiKeyIsMember32(PyObject *self, PyObject *args)
{
   //printf("2arrays!");
   Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

   if (tupleSize < 2) {
      PyErr_Format(PyExc_ValueError, "MultiKeyIsMember32 only %llu args", tupleSize);
      return NULL;
   }

   // Check if they passed in a list
   PyObject* tupleObject1 = PyTuple_GetItem(args, 0);
   PyObject* tupleObject2 = PyTuple_GetItem(args, 1);
   INT64 hintSize = 0;

   if (!PyTuple_CheckExact(tupleObject1) || !PyTuple_CheckExact(tupleObject2)) {
      PyErr_Format(PyExc_ValueError, "MultiKeyIsMember32 first two args must be tuple");
      return NULL;
   }

   if (tupleSize >= 3) {
      // check for hintSize
      PyObject* longObject = PyTuple_GetItem(args, 2);
      if (PyLong_Check(longObject)) {
         hintSize = PyLong_AsSize_t(longObject);
         LOGGING("Hint size is %llu\n", hintSize);
      }
   }

   try {
      CMultiKeyPrepare mkp1(tupleObject1);
      CMultiKeyPrepare mkp2(tupleObject2);

      if (mkp1.totalRows == 0) {
         PyErr_Format(PyExc_ValueError, "MultiKeyIsMember32 first argument --  array lengths do not match");
         return NULL;
      }

      if (mkp2.totalRows == 0) {
         PyErr_Format(PyExc_ValueError, "MultiKeyIsMember32 second argument --  array lengths do not match");
         return NULL;
      }

      if (mkp1.totalItemSize != mkp2.totalItemSize) {
         PyErr_Format(PyExc_ValueError, "MultiKeyIsMember32 total itemsize is not equal %lld vs %lld", mkp1.totalItemSize, mkp2.totalItemSize);
         return NULL;
      }

      if (mkp1.pSuperArray && mkp2.pSuperArray) {

         PyArrayObject* boolArray = AllocateLikeNumpyArray(mkp1.aInfo[0].pObject, NPY_BOOL);

         if (boolArray) {
            INT8* pDataOut1 = (INT8*)PyArray_BYTES(boolArray);

            PyArrayObject* indexArray = NULL;

            IsMemberHashMKPre(&indexArray, mkp1.totalRows, mkp1.pSuperArray, mkp2.totalRows, mkp2.pSuperArray, pDataOut1, mkp1.totalItemSize, hintSize, HASH_MODE_MASK);

            if (indexArray) {
               PyObject* retObject = Py_BuildValue("(OO)", boolArray, indexArray);
               Py_DECREF((PyObject*)boolArray);
               Py_DECREF((PyObject*)indexArray);

               return (PyObject*)retObject;
            }
         }
         PyErr_Format(PyExc_ValueError, "MultiKeyIsMember32 ran out of memory");
         return NULL;
      }
   } catch (...) {

   }
   // error path
   Py_INCREF(Py_None);
   return Py_None;

}




//-----------------------------------------------------------------------------------
// Called when 4 arrays are used as input params
// First arg: a tuple of numpy arrays representing the keys of the target ((key1,key2), hashmode, filter)
// Second arg: another tuple of numpy arrays representing keys of object to align
// Third arg: a numpy object that are target alignment values
// Fourth arg: another numpy object the values to align
// Returns two arrays bool and index
PyObject *MultiKeyAlign32(PyObject *self, PyObject *args)
{
   if (!PyTuple_Check(args)) {
      PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 arguments needs to be a tuple");
      return NULL;
   }

   Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

   if (tupleSize < 6) {
      PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 only %llu args", tupleSize);
      return NULL;
   }

   // Check if they passed in a list
   PyObject* tupleObject1 = PyTuple_GetItem(args, 0);
   PyObject* tupleObject2 = PyTuple_GetItem(args, 1);
   PyObject* valObject1 = PyTuple_GetItem(args, 2);
   PyObject* valObject2 = PyTuple_GetItem(args, 3);
   PyObject* isForwardObj = PyTuple_GetItem(args, 4);
   PyObject* allowExactObj = PyTuple_GetItem(args, 5);

   if (!PyTuple_Check(tupleObject1) || !PyTuple_Check(tupleObject2)) {
      PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 first two args must be tuples");
      return NULL;
   }

   bool isForward = PyObject_IsTrue(isForwardObj) > 0;
   bool allowExact = PyObject_IsTrue(allowExactObj) > 0;

   PyArrayObject* pvalArray1;
   PyArrayObject* pvalArray2;
   if (PyArray_Check(valObject1) && PyArray_Check(valObject2)) {
      pvalArray1 = (PyArrayObject*)valObject1;
      pvalArray2 = (PyArrayObject*)valObject2;
   }
   else {
      PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 second two args must be arrays");
      return NULL;
   }

   CMultiKeyPrepare mkp1(tupleObject1);
   CMultiKeyPrepare mkp2(tupleObject2);

   if (mkp1.totalItemSize != mkp2.totalItemSize) {
      PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 keys are not the same itemsize");
      return NULL;
   }

   // NOTE: this check seems unnec as i think it can handle different row sizes
   //if (mkp1.totalRows != mkp2.totalRows) {
   //   PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 rows are not the same %llu vs %llu", mkp1.totalRows, mkp2.totalRows);
   //   return NULL;
   //}

   //TODO: better handling of input types
   INT32 dtype1 = ObjectToDtype((PyArrayObject*)pvalArray1);
   INT32 dtype2 = ObjectToDtype((PyArrayObject*)pvalArray2);

   if (dtype1 < 0) {
      PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 data types are not understood dtype.num: %d vs %d", dtype1, dtype2);
      return NULL;
   }

   if (dtype1 != dtype2) {

      // Check for when numpy has 7==9 or 8==10 on Linux 5==7, 6==8 on Windows
      if (!((dtype1 <= NPY_ULONGLONG && dtype2 <= NPY_ULONGLONG) &&
            ((dtype1 & 1) == (dtype2 & 1)) &&
            PyArray_ITEMSIZE((PyArrayObject*)pvalArray1) == PyArray_ITEMSIZE((PyArrayObject*)pvalArray2))) {

         PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 data types are not the same dtype.num: %d vs %d", dtype1, dtype2);
         return NULL;
      }
   }

   // TJD --- check correct length passed in
   if (ArrayLength(pvalArray1) != mkp1.totalRows) {
      PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 val1 length does not match key input length of %lld", mkp1.totalRows);
      return NULL;
   }
   if (ArrayLength(pvalArray2) != mkp2.totalRows) {
      PyErr_Format(PyExc_ValueError, "MultiKeyAlign32 val2 length does not match key input length of %lld", mkp2.totalRows);
      return NULL;
   }

   if (mkp1.pSuperArray && mkp2.pSuperArray) {

      void* pVal1 = PyArray_BYTES(pvalArray1);
      void* pVal2 = PyArray_BYTES(pvalArray2);
      PyArrayObject* indexArray = (PyArrayObject*)Py_None;
      bool isIndex32 = TRUE;
      BOOL success = FALSE;
      if (mkp1.totalRows > 2000000000 || mkp2.totalRows > 2000000000) {
         isIndex32 = FALSE;
      }
      LOGGING("MultiKeyAlign32 total rows %lld %lld\n", mkp1.totalRows, mkp2.totalRows);
      try {
         if (isIndex32) {
            indexArray = AllocateLikeNumpyArray(mkp1.aInfo[0].pObject, NPY_INT32);
            if (!indexArray) return PyErr_Format(PyExc_BufferError, "MultiKeyAlign32");
            INT32* pDataOut2 = (INT32*)PyArray_BYTES(indexArray);
            success = AlignHashMK32(mkp1.totalRows, mkp1.pSuperArray, pVal1, mkp2.totalRows, mkp2.pSuperArray, pVal2, pDataOut2, mkp1.totalItemSize, HASH_MODE_MASK, dtype1, isForward, allowExact);
         }
         else {
            indexArray = AllocateLikeNumpyArray(mkp1.aInfo[0].pObject, NPY_INT64);
            if (!indexArray) return PyErr_Format(PyExc_BufferError, "MultiKeyAlign32");
            INT64* pDataOut2 = (INT64*)PyArray_BYTES(indexArray);
            success = AlignHashMK64(mkp1.totalRows, mkp1.pSuperArray, pVal1, mkp2.totalRows, mkp2.pSuperArray, pVal2, pDataOut2, mkp1.totalItemSize, HASH_MODE_MASK, dtype1, isForward, allowExact);
         }

         if (!success) {
            PyErr_Format(PyExc_ValueError, "MultiKeyAlign failed.  Only accepts INT32,INT64,FLOAT32,FLOAT64");
         }

      }
      catch (const std::exception &e) {
         LogError("Exception thrown %s\n", e.what());
      }
      return (PyObject*)indexArray;
   }

   // error path
   Py_INCREF(Py_None);
   return Py_None;
}


//==================================================================
//------------------------------------------------------------------
typedef void(*MAKE_I_GROUP2)(
   void* piKeyT,
   void* piFirstGroupT,
   void* piGroupT,
   INT64 totalRows,
   INT64 offset,
   INT64 unique_count);

//-------------------------------------------------------------------
// Parameters
// ----------
// Arg1 iKey from grouping object
// piFirstGroupT a cumsum of pnCountGroup (must be int32 or int64)
//
// This routine is used when ikey data is partitioned for low unique count
//
// Returns
// -------------
// piGroup
// the piFirstGroup is throw away
template<typename KEYTYPE, typename OUTDTYPE>
void MakeiGroup2(
   void* piKeyT,
   void* piFirstGroupT,
   void* piGroupT,
   INT64 totalRows,
   INT64 offset,
   INT64 unique_count) {

   KEYTYPE* piKey = (KEYTYPE*)piKeyT;
   OUTDTYPE* piFirstGroup = (OUTDTYPE*)piFirstGroupT;
   OUTDTYPE* piGroup = (OUTDTYPE*)piGroupT;

   // shift the data to work on (based on which thread we are on)
   piKey = piKey + offset;

   LOGGING("makeigroup2 - totalrows:%lld  offset:%lld   unique:%lld \n", totalRows, offset, unique_count);
   // in this routine we own all bins
   // keep taking the scattered groups and putting them together in a fancy index for piGroup
   // as a result, piFirstGroup keeps creeping 
   for (INT64 i = 0; i < totalRows; i++) {
      INT64 key = (INT64)piKey[i];
      if (key >= 0 && key < unique_count) {
         OUTDTYPE nextloc = piFirstGroup[key];
         //printf("[%lld] key: %lld   nextloc: %lld -- offset %lld -- %p %p\n", i, (INT64)key, (INT64)nextloc, offset, piFirstGroup, piGroup );
         piGroup[nextloc] = (OUTDTYPE)(i + offset);
         piFirstGroup[key] = nextloc + 1;
      } // else data intergrity issue
   }
}


//---------------------------------------
// Return a function pointer (NULL on failure)
// to make the igroups
// This routine is used when ikey data is partitioned for low unique count
MAKE_I_GROUP2 GetMakeIGroup2(int iKeyType, int outdtype) {

   // Now build the igroup
   //
   MAKE_I_GROUP2 pBinFunc = NULL;

   if (outdtype == NPY_INT32) {
      switch (iKeyType) {
      case NPY_INT8:
         pBinFunc = MakeiGroup2<INT8, INT32>;
         break;
      case NPY_INT16:
         pBinFunc = MakeiGroup2<INT16, INT32>;
         break;
      CASE_NPY_INT32:
         pBinFunc = MakeiGroup2<INT32, INT32>;
         break;
      CASE_NPY_INT64:
         pBinFunc = MakeiGroup2<INT64, INT32>;
         break;
      default:
         printf("!!!internal error in MakeiGroup\n");
      }
   }
   else {
      switch (iKeyType) {
      case NPY_INT8:
         pBinFunc = MakeiGroup2<INT8, INT64>;
         break;
      case NPY_INT16:
         pBinFunc = MakeiGroup2<INT16, INT64>;
         break;
      CASE_NPY_INT32:
         pBinFunc = MakeiGroup2<INT32, INT64>;
         break;
      CASE_NPY_INT64:
         pBinFunc = MakeiGroup2<INT64, INT64>;
         break;
      default:
         printf("!!!internal error in MakeiGroup\n");
      }
   }
   return pBinFunc;
}




//==================================================================
//------------------------------------------------------------------
typedef void(*MAKE_I_GROUP)(
   void* piKeyT,
   void* pnCountGroupT,
   void* piFirstGroupT,
   void* piGroupT,
   INT64 totalRows,
   INT64 binLow,
   INT64 binHigh);

//-------------------------------------------------------------------
// Parameters
// ----------
// Arg1 iKey from grouping object
// piFirstGroupT a cumsum of pnCountGroup (must be int32 or int64)
//
// Returns
// -------------
// piGroup
// pnCountGroup
// piFirstGroup
template<typename KEYTYPE, typename OUTDTYPE>
void MakeiGroup(
   void* piKeyT,
   void* pnCountGroupT,
   void* piFirstGroupT,
   void* piGroupT,
   INT64 totalRows,
   INT64 binLow,
   INT64 binHigh) {

   KEYTYPE* piKey = (KEYTYPE*)piKeyT;
   OUTDTYPE* pnCountGroup = (OUTDTYPE*)pnCountGroupT;
   OUTDTYPE* piFirstGroup = (OUTDTYPE*)piFirstGroupT;
   OUTDTYPE* piGroup = (OUTDTYPE*)piGroupT;

   // check for our bin range
   // if so, keep taking the scattered groups and putting them together
   // in a fancy index for piGroup
   // as a result, piFirstGroup keeps creeping and we have to subtract it back at the very end
   for (INT64 i = 0; i < totalRows; i++) {
      INT64 key = (INT64)piKey[i];
      if (key >= binLow && key < binHigh) {
         OUTDTYPE nextloc = piFirstGroup[key];
         //printf("%lld %lld -- %p %p\n", (INT64)key, (INT64)nextloc, piFirstGroup, piGroup );
         piGroup[nextloc] = (OUTDTYPE)i;
         piFirstGroup[key] = nextloc + 1;
      }
   }

   // Fixup iFirstGroup by subtracting what we added
   for (INT64 i = binLow; i < binHigh; i++) {
      piFirstGroup[i] -= pnCountGroup[i];
   }
}


//---------------------------------------
// Return a function pointer (NULL on failure)
// to make the igroups
MAKE_I_GROUP GetMakeIGroup(int iKeyType, int outdtype) {

   // Now build the igroup
   //
   MAKE_I_GROUP pBinFunc = NULL;

   if (outdtype == NPY_INT32) {
      switch (iKeyType) {
      case NPY_INT8:
         pBinFunc = MakeiGroup<INT8, INT32>;
         break;
      case NPY_INT16:
         pBinFunc = MakeiGroup<INT16, INT32>;
         break;
      CASE_NPY_INT32:
         pBinFunc = MakeiGroup<INT32, INT32>;
         break;
      CASE_NPY_INT64:
         pBinFunc = MakeiGroup<INT64, INT32>;
         break;
      default:
         printf("!!!internal error in MakeiGroup\n");
      }
   }
   else {
      switch (iKeyType) {
      case NPY_INT8:
         pBinFunc = MakeiGroup<INT8, INT64>;
         break;
      case NPY_INT16:
         pBinFunc = MakeiGroup<INT16, INT64>;
         break;
      CASE_NPY_INT32:
         pBinFunc = MakeiGroup<INT32, INT64>;
         break;
      CASE_NPY_INT64:
         pBinFunc = MakeiGroup<INT64, INT64>;
         break;
      default:
         printf("!!!internal error in MakeiGroup\n");
      }
   }
   return pBinFunc;
}

//-------------------------------------------------------------------
// Parameters
// ----------
// Arg1:iKey from grouping object
// Arg2:nCountGroup or array bincount (must be int32 or int64) see: rc.BinCount
//
// Returns
// -------
// iGroup (int32 or int64) (can be used to make scattered groups contiguous)
// iFirstGroup (int32 or int64)
// note: the Array bincount passed in can be used as the nCountGroup
//
PyObject *
GroupFromBinCount(PyObject *self, PyObject *args)
{
   PyArrayObject *iKey = NULL;
   PyArrayObject *nCountGroup = NULL;

   if (!PyArg_ParseTuple(
      args, "O!O!",
      &PyArray_Type, &iKey,
      &PyArray_Type, &nCountGroup)) {

      return NULL;
   }

   INT32 iKeyType = PyArray_TYPE(iKey);
   INT64 totalRows = ArrayLength(iKey);
   INT64 totalUnique = ArrayLength(nCountGroup);

   PyArrayObject* outiGroup = NULL;
   PyArrayObject* outiFirstGroup = NULL;

   // Check for totalRows > 2100000000 and switch to INT64
   int outdtype = NPY_INT32;

   if (totalRows > 2000000000) {
      outdtype = NPY_INT64;
   }

   if (outdtype != PyArray_TYPE(nCountGroup)) {
      // bail
      PyErr_Format(PyExc_ValueError, "GroupFromBinCount: nCountGroup dtype does not match expected dtype.");
      return NULL;
   }

   outiGroup = AllocateNumpyArray(1, (npy_intp*)&totalRows, outdtype);
   outiFirstGroup = AllocateNumpyArray(1, (npy_intp*)&totalUnique, outdtype);

   if (outiGroup && outiFirstGroup) {
      void* piKey = PyArray_BYTES(iKey);
      void* pnCountGroup = PyArray_BYTES(nCountGroup);
      void* piFirstGroup = PyArray_BYTES(outiFirstGroup);
      void* piGroup = PyArray_BYTES(outiGroup);

      // Generate the location for bin to write the next location
      // Effectively this is a cumsum
      switch (outdtype) {
      case NPY_INT32:
      {
         INT32* pCount = (INT32*)pnCountGroup;
         INT32* pCumSum = (INT32*)piFirstGroup;
         INT32 currentSum = 0;
         for (INT64 i = 0; i < totalUnique; i++) {
            *pCumSum++ = currentSum;
            currentSum += *pCount++;
         }
      }
      break;
      case NPY_INT64:
      {
         INT64* pCount = (INT64*)pnCountGroup;
         INT64* pCumSum = (INT64*)piFirstGroup;
         INT64 currentSum = 0;
         for (INT64 i = 0; i < totalUnique; i++) {
            *pCumSum++ = currentSum;
            currentSum += *pCount++;
         }
      }
      break;
      }

      //
      // Now build the igroup
      //
      MAKE_I_GROUP pMakeIGroup = GetMakeIGroup(iKeyType, outdtype);

      if (!g_cMathWorker->NoThreading && totalRows > 32000) {
         INT64 cores = 0;
         stBinCount* pstBinCount = NULL;

         // Allocate worker segments -- low high bins to work on
         cores = g_cMathWorker->SegmentBins(totalUnique, 0, &pstBinCount);

         struct stBinMaster {
            void* piKey;
            void* pnCountGroup;
            void* piFirstGroup;
            void* piGroup;
            INT64 totalRows;
            stBinCount* pstBinCount;
            MAKE_I_GROUP pMakeIGroup;
         } stMaster;

         // Prepare for multithreading
         stMaster.piKey = piKey;
         stMaster.pnCountGroup = pnCountGroup;
         stMaster.piFirstGroup = piFirstGroup;
         stMaster.piGroup = piGroup;
         stMaster.totalRows = totalRows;
         stMaster.pstBinCount = pstBinCount;
         stMaster.pMakeIGroup = pMakeIGroup;

         auto lambdaBinCountCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
            stBinMaster* callbackArg = (stBinMaster*)callbackArgT;
            INT64 t = workIndex;

            LOGGING("[%d] %lld low %lld  high %lld\n", core, workIndex, callbackArg->pstBinCount[t].BinLow, callbackArg->pstBinCount[t].BinHigh);

            callbackArg->pMakeIGroup(
               callbackArg->piKey,
               callbackArg->pnCountGroup,
               callbackArg->piFirstGroup,
               callbackArg->piGroup,
               callbackArg->totalRows,
               callbackArg->pstBinCount[t].BinLow,
               callbackArg->pstBinCount[t].BinHigh);

            LOGGING("[%d] %lld completed\n", core, workIndex);
            return TRUE;
         };

         LOGGING("multithread makeigroup   %lld cores   %lld unique   %lld rows\n", cores, totalUnique, totalRows);

         g_cMathWorker->DoMultiThreadedWork((int)cores, lambdaBinCountCallback, &stMaster);
         WORKSPACE_FREE(pstBinCount);
      }
      else {

         pMakeIGroup(piKey, pnCountGroup, piFirstGroup, piGroup, totalRows, 0, totalUnique);
      }

      // Return the two arrays we generated
      PyObject* retObject = Py_BuildValue("(OO)", outiGroup, outiFirstGroup);
      Py_DECREF((PyObject*)outiGroup);
      Py_DECREF((PyObject*)outiFirstGroup);

      return (PyObject*)retObject;

   }

   return NULL;
}



typedef void(*BIN_COUNT)(void* pKeysT, void* pOutCount, INT64 startRow, INT64 totalRows, INT64 highbin);

//==================================================================
//-------------------------------------------------------------------
// if ppOutCount is not allocated (set to NULL), we will allocate it
//
template<typename T, typename U>
static void BinCountAlgo(
   void*    pKeysT,    // input the binned keys
   void*    pCountT,   // zero fill and then count
   INT64    startRow,  // ikey start position
   INT64    stopRow,   // ikey stop position
   INT64    total_unique)   // total_unique
{
   T*    pKeys = (T*)pKeysT;
   U*    pCount= (U*)pCountT;

   memset(pCount, 0, total_unique * sizeof(U));

   //printf("start %lld  stop %lld  %p\n", startRow, stopRow, pCount);

   // Move up pKeys to our section so we can loop on i=0
   pKeys += startRow;
   stopRow -= startRow;

   // count the bins
   // NOT: should we unroll this?
   for (INT64 i = 0; i < stopRow; i++) {
      T key = pKeys[i];
      if (key >= 0 && key < total_unique) {
         pCount[key]++;
      }
   }
}

//-------------------------------------------------------------------
// Input the ikeydtype and the output dtype
// ----------
// Return the binning routine
//
BIN_COUNT InternalGetBinFunc(INT32 iKeyType, int outdtype) {
   BIN_COUNT pBinFunc = NULL;

   //------------------------------
   // Pick one of 8 bincount routines based on
   // ikey size and INT32 vs INT64 output
   //
   if (outdtype == NPY_INT32) {
      switch (iKeyType) {
      case NPY_INT8:
         pBinFunc = BinCountAlgo<INT8, INT32>;
         break;
      case NPY_INT16:
         pBinFunc = BinCountAlgo<INT16, INT32>;
         break;
      CASE_NPY_INT32:
         pBinFunc = BinCountAlgo<INT32, INT32>;
         break;
      CASE_NPY_INT64:
         pBinFunc = BinCountAlgo<INT64, INT32>;
         break;
      default:
         printf("!!!internal error in BinCount\n");
      }
   }
   else {
      switch (iKeyType) {
      case NPY_INT8:
         pBinFunc = BinCountAlgo<INT8, INT64>;
         break;
      case NPY_INT16:
         pBinFunc = BinCountAlgo<INT16, INT64>;
         break;
      CASE_NPY_INT32:
         pBinFunc = BinCountAlgo<INT32, INT64>;
         break;
      CASE_NPY_INT64:
         pBinFunc = BinCountAlgo<INT64, INT64>;
         break;
      default:
         printf("!!!internal error in BinCount\n");
      }
   }
   return pBinFunc;
}

//----------------------------------
// Return num of cores used or 0 on failure
// Multithread counts the bins
// Calls: BinCountAlgo
//
INT64 InternalBinCount(
   BIN_COUNT pBinFunc,
   stBinCount** ppstBinCount,      // TO BE FREED LATER
   char** ppMemoryForBins,         // TO BE FREED LATER
   void* piKey,
   INT64 unique_rows,
   INT64 totalRows, 
   INT64 coresRequested,
   INT64 itemSize) {

   INT64 cores = 0;

   // Allocate a low high
   // Using more than 8 cores does not seem to help, we cap at 8

   // pstBinCount needs to be freed
   cores = g_cMathWorker->SegmentBins(totalRows, coresRequested, ppstBinCount);
   stBinCount* pstBinCount = *ppstBinCount;

   // ALLOCATE one large array for all the cores to count bins
   // This array is COL major (2d matrix of [cores, unique_rows])
   *ppMemoryForBins = (char*)WORKSPACE_ALLOC(cores * unique_rows * itemSize);

   if (*ppMemoryForBins == NULL) return 0;
   char* pMemoryForBins = *ppMemoryForBins;

   // Give each core its memory slot
   for (int t = 0; t < cores; t++) {
      pstBinCount[t].pUserMemory = pMemoryForBins + (t *  unique_rows * itemSize);
   }
  
   // Allocate memory for the counting based on
   // cores * unique_rows * sizeof(dtype)

   struct stBinCountMaster {
      BIN_COUNT pBinFunc;
      void* piKey;
      INT64 totalRows;
      INT64 uniqueRows;
      stBinCount* pstBinCount;
   } stMaster;

   //
   // Prepare to multithread the counting of groups
   //
   stMaster.piKey = piKey;
   stMaster.uniqueRows = unique_rows;
   stMaster.totalRows = totalRows;
   stMaster.pstBinCount = pstBinCount;
   stMaster.pBinFunc = pBinFunc;

   auto lambdaBinCountCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
      stBinCountMaster* callbackArg = (stBinCountMaster*)callbackArgT;
      INT64 t = workIndex;

      LOGGING("[%d] %lld low %lld  high %lld\n", core, workIndex, callbackArg->pstBinCount[t].BinLow, callbackArg->pstBinCount[t].BinHigh);

      callbackArg->pBinFunc(
         callbackArg->piKey,
         // memory will be allocated
         callbackArg->pstBinCount[t].pUserMemory,
         callbackArg->pstBinCount[t].BinLow,
         callbackArg->pstBinCount[t].BinHigh,
         callbackArg->uniqueRows);

      LOGGING("[%d] %lld completed\n", core, workIndex);
      return TRUE;
   };

   LOGGING("multithread bincount   %lld cores   %lld unique   %lld rows\n", cores, unique_rows, totalRows);

   // multithread the counting
   g_cMathWorker->DoMultiThreadedWork((int)cores, lambdaBinCountCallback, &stMaster);

   return cores;
}


//----------------------------------------------------
// check for "pack=" 
// returns 0 if pack is not True
INT64 GetPack(PyObject *kwargs) {
   if (!kwargs) return 0;

   PyObject* packObject = PyDict_GetItemString(kwargs, "pack");

   if (packObject && PyBool_Check(packObject)) {
      if (packObject == Py_True) return 1;
   }
   return 0;
}


//-------------------------------------------------------------------
// Parameters
// ----------
// Arg1 iKey from grouping object
// Arg2 unique_count from grouping object
// 
// kwargs
// ------
// pack=None/True/False  True will return 3 values
//
// Returns
// -------
// Array ncountgroup - aka the bincount (int32 or int64)
//
PyObject *
BinCount(PyObject *self, PyObject *args, PyObject* kwargs)
{
   PyArrayObject *iKey = NULL;

   INT64 unique_rows = 0;

   if (!PyArg_ParseTuple(
      args, "O!L",
      &PyArray_Type, &iKey,
      &unique_rows)) {

      return NULL;
   }

   INT32 iKeyType = PyArray_TYPE(iKey);
   INT64 totalRows = ArrayLength(iKey);
   INT64 pack = GetPack(kwargs);

   PyArrayObject* outArray = NULL;
   PyArrayObject* outiGroup = NULL;
   PyArrayObject* outiFirstGroup = NULL;

   // Check for totalRows > 2100000000 and switch to INT64
   int outdtype = NPY_INT32;

   if (unique_rows == 0 || totalRows <=0) {
      PyErr_Format(PyExc_ValueError, "BinCount: unique or totalRows is zero, cannot calculate the bins.");
      return NULL;
   }

   // Check to flip to INT64 based on array length
   if (totalRows > 2000000000) {
      outdtype = NPY_INT64;
   } 

   // Allocate nCountGroup
   outArray = AllocateNumpyArray(1, (npy_intp*)&unique_rows, outdtype);

   if (outArray) {
      void*    piKey = PyArray_BYTES(iKey);
      void*    pnCountGroup = PyArray_BYTES(outArray);
      INT64    itemSize = PyArray_ITEMSIZE(outArray);

      // Get the binning function
      BIN_COUNT pBinFunc = InternalGetBinFunc(iKeyType, outdtype);

      // TJD October 2019
      // Testing this routine with 800,000 uniques on 2M to 200M
      // multithreading did not make this run faster under some circumstance.
      //
      // We wait for a ratio of 50:1  unique to totalrows to kick in multithreading
      //
      if (pack == 1 || (!g_cMathWorker->NoThreading && totalRows > 32000 && (totalRows/unique_rows) > 50)) {
         stBinCount* pstBinCount = NULL;
         char*       pMemoryForBins = NULL;

         // Multithread count the bins and allocate
         INT64 cores =
            InternalBinCount(
               pBinFunc,
               &pstBinCount,
               &pMemoryForBins,
               piKey,
               unique_rows,
               totalRows,
               31,
               itemSize);

         if (pMemoryForBins && pstBinCount) {

            // Copy over the first column and then inplace add all the other cores
            memcpy(pnCountGroup, pstBinCount[0].pUserMemory, unique_rows * itemSize);

            // To get the final counts, we need to add each of the individual counts
            for (int t = 1; t < cores; t++) {
               void *pMem = pstBinCount[t].pUserMemory;
               // Count up what all the worker threads produced
               // This will produce nCountGroup
               // TODO: Could be counted in parallel for large uniques
               if (outdtype == NPY_INT32) {
                  INT32* pMaster = (INT32*)pnCountGroup;
                  INT32* pCount = (INT32*)pMem;

                  // NOTE: this routine can be faster
                  // This is a horizontal add
                  for (INT64 i = 0; i < unique_rows; i++) {
                     pMaster[i] += pCount[i];
                  }
               }
               else {

                  INT64* pMaster = (INT64*)pnCountGroup;
                  INT64* pCount = (INT64*)pMem;
                  for (INT64 i = 0; i < unique_rows; i++) {
                     pMaster[i] += pCount[i];
                  }
               }
            }

            //
            // Check if we are packing and need to return more arrays
            //
            if (pack) {
               outiGroup = AllocateNumpyArray(1, (npy_intp*)&totalRows, outdtype);
               outiFirstGroup = AllocateNumpyArray(1, (npy_intp*)&unique_rows, outdtype);

               //
               // Make sure allocation succeeded
               //
               if (outiGroup && outiFirstGroup) {
                  void* piKey = PyArray_BYTES(iKey);
                  void* piFirstGroup = PyArray_BYTES(outiFirstGroup);
                  void* piGroup = PyArray_BYTES(outiGroup);

                  if (outdtype == NPY_INT32) {
                     // Should be all ZEROS
                     INT32 lastvalue = 0;
                     INT32* pCountArray2d = (INT32*)pstBinCount[0].pUserMemory;

                     // To get the final counts, we need to add each of the individual counts
                     // This is a cumsum on axis=1
                     for (INT64 i = 0; i < unique_rows; i++) {
                        for (INT64 t = 0; t < cores; t++) {
                           INT32* pMem = &pCountArray2d[t*unique_rows + i];
                           // this is is horizontal cumsum
                           INT32 temp = lastvalue;
                           lastvalue += pMem[0];
                           pMem[0] = temp;
                        }
                     }

                     //// Debug printout of [core][group]  first place to write
                     //for (INT64 t = 0; t < cores; t++) {
                     //   INT32 *pMem = (INT32*)pstBinCount[t].pUserMemory;
                     //   for (INT64 i = 0; i < unique_rows; i++) {
                     //      printf("[%lld][%lld]  -- %d \n", t, i, pMem[i]);
                     //   }
                     //}

                  }
                  else {
                     // INT64 routine
                     // Should be all ZEROS
                     INT64 lastvalue = 0;
                     INT64* pCountArray2d = (INT64*)pstBinCount[0].pUserMemory;

                     // To get the final counts, we need to add each of the individual counts
                     for (INT64 i = 0; i < unique_rows; i++) {
                        for (INT64 t = 0; t < cores; t++) {
                           INT64* pMem = &pCountArray2d[t*unique_rows + i];
                           // this is is horizontal cumsum
                           INT64 temp = lastvalue;
                           lastvalue += pMem[0];
                           pMem[0] = temp;
                        }
                     }
                  }

                  // The first column of the 2d matrix is now the iFirstGroup
                  memcpy(piFirstGroup, pstBinCount[0].pUserMemory, unique_rows * itemSize);

                  MAKE_I_GROUP2 pMakeIGroup2 = GetMakeIGroup2(iKeyType, outdtype);

                  struct stBinMaster {
                     void* piKey;
                     void* piGroup;
                     INT64 totalRows;
                     INT64 unique_rows;
                     stBinCount* pstBinCount;
                     MAKE_I_GROUP2 pMakeIGroup2;
                  } stMaster;

                  // Prepare for multithreading
                  stMaster.piKey = piKey;
                  stMaster.piGroup = piGroup;
                  stMaster.totalRows = totalRows;
                  stMaster.unique_rows = unique_rows;
                  stMaster.pstBinCount = pstBinCount;
                  stMaster.pMakeIGroup2 = pMakeIGroup2;

                  auto lambdaBinCountCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
                     stBinMaster* callbackArg = (stBinMaster*)callbackArgT;
                     INT64 t = workIndex;

                     LOGGING("[%d] %lld low %lld  high %lld\n", core, workIndex, callbackArg->pstBinCount[t].BinLow, callbackArg->pstBinCount[t].BinHigh);

                     // Get our ikey data range
                     INT64 start = callbackArg->pstBinCount[t].BinLow;
                     INT64 stop = callbackArg->pstBinCount[t].BinHigh;

                     // call routine to fill in a certain section
                     // 
                     callbackArg->pMakeIGroup2(
                        callbackArg->piKey,
                        callbackArg->pstBinCount[t].pUserMemory,
                        callbackArg->piGroup,
                        stop - start,  // stop - start is the length (totalRows)
                        start,         // binLow is the ikey data shift for this worker thread
                        callbackArg->unique_rows);

                     LOGGING("[%d] %lld completed\n", core, workIndex);
                     return TRUE;
                  };

                  LOGGING("multithread makeigroup2   %lld cores   %lld unique   %lld rows\n", cores, unique_rows, totalRows);

                  g_cMathWorker->DoMultiThreadedWork((int)cores, lambdaBinCountCallback, &stMaster);

                  // Debug printout of [core][group]  first place to write
                  //printf("------------- AFTER COMPUTING ---------------\n");
                  //for (INT64 t = 0; t < cores; t++) {
                  //   INT32 *pMem = (INT32*)pstBinCount[t].pUserMemory;
                  //   for (INT64 i = 0; i < unique_rows; i++) {
                  //      printf("[%lld][%lld]  -- %d \n", t, i, pMem[i]);
                  //   }
                  //}

               }
            }

            //
            // Free what we allocated
            WORKSPACE_FREE(pMemoryForBins);
            WORKSPACE_FREE(pstBinCount);
         }  else {
            // out of memory
            return NULL;

         }
      }
      else {
         // NOT packing and doing one thread
         // Zero out the group so we can count up
         memset(pnCountGroup, 0, unique_rows * itemSize);

         // single threaded
         pBinFunc(piKey, pnCountGroup, 0, totalRows, unique_rows);
      }
   }

   if ( pack==1 && outiGroup && outiFirstGroup) {
      // Return the three arrays we generated
      PyObject* retObject = Py_BuildValue("(OOO)", outArray, outiGroup, outiFirstGroup);
      Py_DECREF((PyObject*)outArray);
      Py_DECREF((PyObject*)outiGroup);
      Py_DECREF((PyObject*)outiFirstGroup);

      return (PyObject*)retObject;
   }

   // return just the nCountGroup
   return (PyObject*)outArray;
}


