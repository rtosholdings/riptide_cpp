#pragma once

#include "CommonInc.h"
#include "Recycler.h"
#include "MultiKey.h"

enum SORT_MODE {
   SORT_MODE_QSORT = 1,
   SORT_MODE_MERGE = 2,
   SORT_MODE_HEAP = 3
};


INT64* GetCutOffs(PyObject* kwargs, INT64& cutoffLength);
PyObject* GroupFromLexSort(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* IsSorted(PyObject* self, PyObject* args);

PyObject* LexSort64(PyObject* self, PyObject* args, PyObject* kwargs);
PyObject* LexSort32(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* Sort(PyObject *self, PyObject *args);
PyObject* SortInPlace(PyObject *self, PyObject *args);
PyObject* SortInPlaceIndirect(PyObject *self, PyObject *args);



//-----------------------------------------------------------------------------------------------
template <typename T>
int heapsort_(T *start, INT64 n);

template <typename T>
int quicksort_(T *start, INT64 num);

template <typename T>
void mergesort0_(T *pl, T *pr, T *pw);

template <typename T>
int mergesort_(T *start, INT64 num);


//===============================================================================
// Helper class for one or more arrays
class CMultiListPrepare {

public:

   Py_ssize_t tupleSize;  // or number of arrays
   ArrayInfo* aInfo;
   INT64 totalItemSize;
   INT64 totalRows;

   CMultiListPrepare(PyObject* args) {
      aInfo = NULL;
      totalItemSize = 0;
      totalRows = 0;

      tupleSize = PyTuple_GET_SIZE(args);

      //MLPLOGGING("Tuple size %llu\n", tupleSize);

      if (tupleSize >= 1) {
         // Check if they passed in a list
         PyObject* listObject = PyTuple_GetItem(args, 0);
         if (PyList_Check(listObject)) {
            args = listObject;
            tupleSize = PyList_GET_SIZE(args);
            //MLPLOGGING("Found list inside tuple size %llu\n", tupleSize);
         }
      }

      INT64 listSize = 0;
      aInfo = BuildArrayInfo(args, &listSize, &totalItemSize);

      if (aInfo) {

         totalRows = aInfo[0].ArrayLength;

         for (INT64 i = 0; i < listSize; i++) {
            if (aInfo[i].ArrayLength != totalRows) {
               PyErr_Format(PyExc_ValueError, "MultiListPrepare all arrays must be same number of rows %llu", totalRows);
               totalRows = 0;
            }
         }
         if (totalRows != 0) {
            //printf("row width %llu   rows %llu\n", totalItemSize, totalRows);
         }
      }
   }

   ~CMultiListPrepare() {
      if (aInfo != NULL) {
         FreeArrayInfo(aInfo);
         aInfo = NULL;
      }
   }

};
