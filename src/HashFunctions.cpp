#include "RipTide.h"
#include "HashFunctions.h"
#include "HashLinear.h"
#include "ndarray.h"

//struct ndbuf;
//typedef struct ndbuf {
//   struct ndbuf *next;
//   struct ndbuf *prev;
//   Py_ssize_t len;     /* length of data */
//   Py_ssize_t offset;  /* start of the array relative to data */
//   char *data;         /* raw data */
//   int flags;          /* capabilities of the base buffer */
//   Py_ssize_t exports; /* number of exports */
//   Py_buffer base;     /* base buffer */
//} ndbuf_t;
//
//typedef struct {
//   PyObject_HEAD
//      int flags;          /* ndarray flags */
//   ndbuf_t staticbuf;  /* static buffer for re-exporting mode */
//   ndbuf_t *head;      /* currently active base buffer */
//} NDArrayObject;



#define LOGGING(...)
//#define LOGGING printf




//-----------------------------------------------------------------------------------------
// IsMember
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//    Returns: boolean array and optional INT64 location array
PyObject *
IsMember64(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *inArr2 = NULL;
   int hashMode;
   INT64 hintSize = 0;

   if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &hashMode)) return NULL;

   INT32 arrayType1 = PyArray_TYPE(inArr1);
   INT32 arrayType2 = PyArray_TYPE(inArr2);

   int sizeType1 = (int)NpyItemSize((PyObject*)inArr1);
   int sizeType2 = (int)NpyItemSize((PyObject*)inArr2);

   LOGGING("IsMember32 %s vs %s   size: %d  %d\n", NpyToString(arrayType1), NpyToString(arrayType2), sizeType1, sizeType2);

   if (arrayType1 != arrayType2) {
      // Arguments do not match
      PyErr_Format(PyExc_ValueError, "IsMember32 needs first arg to match %s vs %s", NpyToString(arrayType1), NpyToString(arrayType2));
      return NULL;
   }

   if (sizeType1 == 0) {
      // Weird type
      PyErr_Format(PyExc_ValueError, "IsMember32 needs a type it understands %s vs %s", NpyToString(arrayType1), NpyToString(arrayType2));
      return NULL;
   }

   if (arrayType1 == NPY_OBJECT) {
      PyErr_Format(PyExc_ValueError, "IsMember32 cannot handle unicode strings, please convert to np.chararray");
      return NULL;
   }

   int ndim = PyArray_NDIM(inArr1);
   npy_intp* dims = PyArray_DIMS(inArr1);

   int ndim2 = PyArray_NDIM(inArr2);
   npy_intp* dims2 = PyArray_DIMS(inArr2);

   INT64 arraySize1 = CalcArrayLength(ndim, dims);
   INT64 arraySize2 = CalcArrayLength(ndim2, dims2);

   PyArrayObject* boolArray = AllocateNumpyArray(ndim, dims, NPY_BOOL);
   CHECK_MEMORY_ERROR(boolArray);

   PyArrayObject* indexArray = AllocateNumpyArray(ndim, dims, NPY_INT64);
   CHECK_MEMORY_ERROR(indexArray);

   if (boolArray && indexArray) {
      void* pDataIn1 = PyArray_BYTES(inArr1);
      void* pDataIn2 = PyArray_BYTES(inArr2);
      INT8* pDataOut1 = (INT8*)PyArray_BYTES(boolArray);
      INT64* pDataOut2 = (INT64*)PyArray_BYTES(indexArray);

      //printf("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

      if (arrayType1 >= NPY_STRING) {

         LOGGING("Calling string!\n");
         IsMemberHashString64(arraySize1, sizeType1, (const char*)pDataIn1, arraySize2, sizeType2, (const char*)pDataIn2, pDataOut2, pDataOut1, HASH_MODE(hashMode), hintSize, arrayType1 == NPY_UNICODE);
      }
      else {
         if (arrayType1 == NPY_FLOAT32 || arrayType1 == NPY_FLOAT64) {

            IsMemberHash64(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, pDataOut1, sizeType1 + 100, HASH_MODE(hashMode), hintSize);
         }
         else {
            IsMemberHash64(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, pDataOut1, sizeType1, HASH_MODE(hashMode), hintSize);
         }

         PyObject* retObject = Py_BuildValue("(OO)", boolArray, indexArray);
         Py_DECREF((PyObject*)boolArray);
         Py_DECREF((PyObject*)indexArray);

         return (PyObject*)retObject;
      }
   }
   // out of memory
   return NULL;
}



//-----------------------------------------------------------------------------------
// IsMemberCategorical
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//    Fourth arg: hintSize
//    Returns:
//       missed: 1 or 0
//       an array int8/16/32/64 location array same size as array1
//       index: index location of where first arg found in second arg  (index into second arg)
PyObject *
IsMemberCategorical(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *inArr2 = NULL;
   int hashMode=HASH_MODE_MASK;
   INT64 hintSize=0;

   if (PyTuple_GET_SIZE(args) == 3) {
      if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &hashMode)) return NULL;
   }
   else {
      if (!PyArg_ParseTuple(args, "O!O!iL", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &hashMode, &hintSize)) return NULL;

   }
   INT32 arrayType1 = PyArray_TYPE(inArr1);
   INT32 arrayType2 = PyArray_TYPE(inArr2);

   int sizeType1 = (int)NpyItemSize((PyObject*)inArr1);
   int sizeType2 = (int)NpyItemSize((PyObject*)inArr2);

   LOGGING("IsMember32 %s vs %s   size: %d  %d\n", NpyToString(arrayType1), NpyToString(arrayType2), sizeType1, sizeType2);

   switch (arrayType1) {
   CASE_NPY_INT32:
      arrayType1 = NPY_INT32;
      break;
   CASE_NPY_UINT32:
      arrayType1 = NPY_UINT32;
      break;
   CASE_NPY_INT64:
      arrayType1 = NPY_INT64;
      break;
   CASE_NPY_UINT64:
      arrayType1 = NPY_UINT64;
      break;
   }

   switch (arrayType2) {
   CASE_NPY_INT32:
      arrayType2 = NPY_INT32;
      break;
   CASE_NPY_UINT32:
      arrayType2 = NPY_UINT32;
      break;
   CASE_NPY_INT64:
      arrayType2 = NPY_INT64;
      break;
   CASE_NPY_UINT64:
      arrayType2 = NPY_UINT64;
      break;
   }

   if (arrayType1 != arrayType2) {

      // Arguments do not match
      PyErr_Format(PyExc_ValueError, "IsMemberCategorical needs first arg to match %s vs %s", NpyToString(arrayType1), NpyToString(arrayType2));
      return NULL;
   }

   if (sizeType1 == 0) {
      // Weird type
      PyErr_Format(PyExc_ValueError, "IsMemberCategorical needs a type it understands %s vs %s", NpyToString(arrayType1), NpyToString(arrayType2));
      return NULL;
   }

   if (arrayType1 == NPY_OBJECT) {
      PyErr_Format(PyExc_ValueError, "IsMemberCategorical cannot handle unicode, object, void strings, please convert to np.chararray");
      return NULL;
   }

   INT64 arraySize1 = ArrayLength(inArr1);
   INT64 arraySize2 = ArrayLength(inArr2);

   void* pDataIn1 = PyArray_BYTES(inArr1);
   void* pDataIn2 = PyArray_BYTES(inArr2);

   PyArrayObject* indexArray = NULL;

   LOGGING("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

   INT64 missed = 0;

   if (arrayType1 >= NPY_STRING) {
      LOGGING("Calling string/uni/void!\n");
      missed = IsMemberCategoricalHashStringPre(&indexArray, inArr1, arraySize1, sizeType1, (const char*)pDataIn1, arraySize2, sizeType2, (const char*)pDataIn2, HASH_MODE(hashMode), hintSize, arrayType1==NPY_UNICODE);
   }
   else
   if (arrayType1 == NPY_FLOAT32 || arrayType1 == NPY_FLOAT64 || arrayType1 == NPY_LONGDOUBLE) {

      LOGGING("Calling float!\n");
      if (arraySize1 < 2100000000) {
         indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
         INT32* pDataOut2 = (INT32*)PyArray_BYTES(indexArray);
         missed = IsMemberHashCategorical(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, sizeType1 + 100, HASH_MODE(hashMode), hintSize);
      }
      else {

         indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT64);
         INT64* pDataOut2 = (INT64*)PyArray_BYTES(indexArray);
         missed = IsMemberHashCategorical64(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, sizeType1 + 100, HASH_MODE(hashMode), hintSize);
      }
   }
   else {
      LOGGING("Calling hash!\n");
      if (arraySize1 < 2100000000) {
         indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
         INT32* pDataOut2 = (INT32*)PyArray_BYTES(indexArray);
         missed = IsMemberHashCategorical(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, sizeType1, HASH_MODE(hashMode), hintSize);
      }
      else {
         indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT64);
         INT64* pDataOut2 = (INT64*)PyArray_BYTES(indexArray);
         missed = IsMemberHashCategorical64(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, sizeType1, HASH_MODE(hashMode), hintSize);
      }
   }

   PyObject* retObject = Py_BuildValue("(LO)", missed, indexArray);
   Py_DECREF((PyObject*)indexArray);
   return (PyObject*)retObject;
}


//-----------------------------------------------------------
// Will find the first occurence of an interger in pArray2
// ReverseMap must be size of uniques in array2
// ReIndex must be size of uniques in array1
// Will then fixup reIndex
//
// Returns: Reindex array
//--------------------------------
// bin1   0 3 0 4 3 2 0 2 3  (5 uniques)  [b'b' b'a' b'b' b'd' b'a' b'e' b'b' b'e' b'a']
// bin2   0 2 3 0 1 1 4      (5 uniques)  [b'a' b'c' b'd' b'a' b'e' b'e' b'b']
// uniq   4 2 1 0 3
// revMap 3 2 1 4 0 -1 -1
// reindx 6 1 4 0 2
//
// final result 6 0 6 2 0 4 6 4 0
template<typename T>
void FindFirstOccurence(T* pArray2, INT32* pUnique1, INT32* pReIndex, INT32* pReverseMap, INT32 array2Length, INT32 unique1Length, INT32 unique2Length, INT32 baseOffset2T) {

   T baseOffset2 = (T)baseOffset2T;

   // Put invalid as default
   const INT32 invalid = *(INT32*)GetDefaultForType(NPY_INT32);
   for (INT32 i = 0; i < unique1Length; i++) {
      pReIndex[i] = invalid;
   }

   for (INT32 i = 0; i < unique2Length; i++) {
      pReverseMap[i] = -1;
   }

   // Make reverse map
   INT32 matchedUniqueLength(0);
   for (INT32 i = 0; i < unique1Length; i++) {
      INT32 val = pUnique1[i];
      if (val >= 0 && val < unique2Length) {
         pReverseMap[val] = i;
         ++matchedUniqueLength;
      }
   }

   // leave for debugging
   //for (INT32 i = 0; i < unique2Length; i++) {
   //   printf("rmap [%d] %d\n", i, pReverseMap[i]);
   //}

   // Find first occurence of values in pUnique
   // TODO: early stopping possible
   if (matchedUniqueLength > 0) {
      for (INT32 i = 0; i < array2Length; i++) {
         // N.B. The value can be the minimum integer, need to be checked first before offsetting by baseOffset2.
         T val = pArray2[i];

         // Find first occurence
         if (val >= baseOffset2) {
            val -= baseOffset2;

            // Check if the value in the second array is found in the first array
            INT32 lookup = pReverseMap[val];

            if (lookup >= 0) {
               // Is this the first occurence?
               if (pReIndex[lookup] == invalid) {
                  //printf("first occurence of val:%d  at lookup:%d  pos:%d\n", (int)val, lookup, i);
                  pReIndex[lookup] = i;

                  // check if we found everything possible for early exit
                  --matchedUniqueLength;
                  if (matchedUniqueLength <= 0) break;
               }
            }
         }
      }
   }

   // Leave for debugging
   //for (INT32 i = 0; i < unique1Length; i++) {
   //   printf("ridx [%d] %d\n", i, pReIndex[i]);
   //}

}


//----------------------------------------------------------------------
// IsMemberCategoricalFixup
// Input:
//    First arg: bin array A (array1)
//    Fourth arg: value to replace with
//
// Returns: pOutput and pBoolOutput
// TODO: this routine needs to output INT8/16/32/64 instead of just INT32
template<typename T>
void FinalMatch(T* pArray1, INT32* pOutput, INT8* pBoolOutput, INT32* pReIndex, INT32 array1Length, INT32 baseoffset) {

   const INT32 invalid = *(INT32*)GetDefaultForType(NPY_INT32);

   // TODO: Multithreadable
   // Find first occurence of values in pUnique
   for (INT64 i = 0; i < array1Length; i++) {
      // N.B. The value can be the minimum integer, need to be checked first before offsetting by baseoffset.
      T val = pArray1[i];

      // Find first occurence
      if (val >= baseoffset) {
         val -= baseoffset;
         const INT32 firstoccurence = pReIndex[val];
         pOutput[i] = firstoccurence;
         pBoolOutput[i] = firstoccurence != invalid;
      }
      else {
         pOutput[i] = invalid;
         pBoolOutput[i] = 0;
      }
   }
}


//----------------------------------------------------------------------
// IsMemberCategoricalFixup
//    First arg: bin array A (array1)
//    Second arg: bin array B (array2)
//    Third arg: result of ismember on As unique, B's unique
//    Fourth arg: length of uniques on B
//    Fifth arg: baseoffset 0 or 1
//    Sixth arg: baseoffset 0 or 1
//
//    Returns:
//       boolean
//       an array int8/16/32/64 location array same size as array1
//       index: index location of where first arg found in second arg  (index into second arg)
PyObject *
IsMemberCategoricalFixup(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *inArr2 = NULL;
   PyArrayObject *isMemberUnique = NULL;
   INT32 unique2Length;
   INT32 baseoffset1;
   INT32 baseoffset2;

   if (!PyArg_ParseTuple(args, "O!O!O!iii",
      &PyArray_Type, &inArr1, 
      &PyArray_Type, &inArr2, 
      &PyArray_Type, &isMemberUnique, 
      &unique2Length,
      &baseoffset1,
      &baseoffset2)) return NULL;

   INT32 arraySize1 = (INT32)ArrayLength(inArr1);
   INT32 arraySize2 = (INT32)ArrayLength(inArr2);
   INT32 arraySizeUnique = (INT32)ArrayLength(isMemberUnique);

   INT32 uniqueType = PyArray_TYPE(isMemberUnique);
   INT32 array2Type = PyArray_TYPE(inArr2);
   INT32 array1Type = PyArray_TYPE(inArr1);

   switch (uniqueType) {
   CASE_NPY_INT32:
      break;
   default:
      PyErr_Format(PyExc_ValueError, "IsMemberCategoricalFixup third argument must be type INT32 not %s", NpyToString(uniqueType));
      return NULL;
   }

   LOGGING("IsMemberCategoricalFixup uniqlength:%d   baseoffset1:%d   baseoffset2:%d\n", unique2Length, baseoffset1, baseoffset2);

   // need to reindex this array...
   INT32* reIndexArray = (INT32*)WORKSPACE_ALLOC(arraySizeUnique * sizeof(INT32));
   INT32* reverseMapArray = (INT32*)WORKSPACE_ALLOC(unique2Length * sizeof(INT32));

   PyArrayObject* indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
   PyArrayObject* boolArray = AllocateLikeNumpyArray(inArr1, NPY_BOOL);

   if (reIndexArray && indexArray && boolArray) {

      INT32* pUnique = (INT32*)PyArray_BYTES(isMemberUnique);
      void*  pArray2 = PyArray_BYTES(inArr2);

      switch (array2Type) {
      case NPY_INT8:
         FindFirstOccurence<INT8>((INT8*)pArray2, pUnique, reIndexArray, reverseMapArray, arraySize2, arraySizeUnique, unique2Length, baseoffset2);
         break;
      case NPY_INT16:
         FindFirstOccurence<INT16>((INT16*)pArray2, pUnique, reIndexArray, reverseMapArray, arraySize2, arraySizeUnique, unique2Length, baseoffset2);
         break;
      CASE_NPY_INT32:
         FindFirstOccurence<INT32>((INT32*)pArray2, pUnique, reIndexArray, reverseMapArray, arraySize2, arraySizeUnique, unique2Length, baseoffset2);
         break;
      CASE_NPY_INT64:
         FindFirstOccurence<INT64>((INT64*)pArray2, pUnique, reIndexArray, reverseMapArray, arraySize2, arraySizeUnique, unique2Length, baseoffset2);
         break;
      default:
         PyErr_Format(PyExc_ValueError, "IsMemberCategoricalFixup second argument is not INT8/16/32/64");
         return NULL;
      }

      void*  pArray1 = PyArray_BYTES(inArr1);
      INT32* pIndexOut = (INT32*)PyArray_BYTES(indexArray);
      INT8* pBoolOut = (INT8*)PyArray_BYTES(boolArray);

      switch (array1Type) {
      case NPY_INT8:
         FinalMatch<INT8>((INT8*)pArray1, pIndexOut, pBoolOut, reIndexArray, arraySize1, baseoffset1);
         break;
      case NPY_INT16:
         FinalMatch<INT16>((INT16*)pArray1, pIndexOut, pBoolOut, reIndexArray, arraySize1, baseoffset1);
         break;
      CASE_NPY_INT32:
         FinalMatch<INT32>((INT32*)pArray1, pIndexOut, pBoolOut, reIndexArray, arraySize1, baseoffset1);
         break;
      CASE_NPY_INT64:
         FinalMatch<INT64>((INT64*)pArray1, pIndexOut, pBoolOut, reIndexArray, arraySize1, baseoffset1);
         break;
      default:
         PyErr_Format(PyExc_ValueError, "IsMemberCategoricalFixup first argument is not INT8/16/32/64");
         return NULL;
      }

      WORKSPACE_FREE(reverseMapArray);
      WORKSPACE_FREE(reIndexArray);
      PyObject* retObject = Py_BuildValue("(OO)", boolArray, indexArray);
      Py_DECREF((PyObject*)boolArray);
      Py_DECREF((PyObject*)indexArray);
      return (PyObject*)retObject;
   }

   PyErr_Format(PyExc_ValueError, "IsMemberCategoricalFixup internal memory error");
   return NULL;
}


