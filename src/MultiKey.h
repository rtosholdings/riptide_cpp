#pragma once

//---------------------------------------------------------------------
// NOTE: See SDSArrayInfo and keep same
struct ArrayInfo {

   // Numpy object
   PyArrayObject*   pObject;

   // First bytes
   char*       pData;

   // Width in bytes of one row
   INT64       ItemSize;

   // total number of items
   INT64       ArrayLength;

   INT64       NumBytes;

   INT32       NumpyDType;
   INT32       NDim;

   // When calling ensure contiguous, we might make a copy
   // if so, pObject is the copy and must be deleted.  pOriginal was passed in
   PyArrayObject*   pOriginalObject;

};


extern PyObject *MultiKeyHash(PyObject *self, PyObject *args);
extern PyObject *GroupByPack32(PyObject* self, PyObject* args);
extern PyObject *MultiKeyGroupBy32(PyObject* self, PyObject* args, PyObject *kwargs);
extern PyObject *MultiKeyGroupBy32Super(PyObject* self, PyObject* args);
extern PyObject *MultiKeyUnique32(PyObject* self, PyObject* args);
extern PyObject *MultiKeyIsMember32(PyObject *self, PyObject *args);
extern PyObject *MultiKeyAlign32(PyObject *self, PyObject *args);
extern PyObject *BinCount(PyObject *self, PyObject *args, PyObject* kwargs);
extern PyObject *MakeiNext(PyObject *self, PyObject *args);
extern PyObject *GroupFromBinCount(PyObject *self, PyObject *args);
extern PyObject *MultiKeyRolling(PyObject *self, PyObject *args);

// really found in hashlinear.cpp
extern PyObject *MergeBinnedAndSorted(PyObject *self, PyObject *args);

extern ArrayInfo* BuildArrayInfo(
   PyObject* listObject,
   INT64* pTupleSize,
   INT64* pTotalItemSize,
   BOOL checkrows = TRUE,
   BOOL convert = TRUE);

extern void FreeArrayInfo(ArrayInfo* pArrayInfo);

