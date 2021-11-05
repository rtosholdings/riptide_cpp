#pragma once

#include "CommonInc.h"

PyObject * ConvertSafeInternal(PyArrayObject * inArr1, int64_t out_dtype);

PyObject * ConvertUnsafeInternal(PyArrayObject * inArr1, int64_t out_dtype);

PyObject * ConvertSafe(PyObject * self, PyObject * args);

PyObject * ConvertUnsafe(PyObject * self, PyObject * args);

PyObject * CombineFilter(PyObject * self, PyObject * args);

PyObject * CombineAccum2Filter(PyObject * self, PyObject * args);

PyObject * CombineAccum1Filter(PyObject * self, PyObject * args);

PyObject * MakeiFirst(PyObject * self, PyObject * args);

PyObject * RemoveTrailingSpaces(PyObject * self, PyObject * args);

PyObject * GetUpcastNum(PyObject * self, PyObject * args);

PyObject * HomogenizeArrays(PyObject * self, PyObject * args);

PyObject * HStack(PyObject * self, PyObject * args);

PyObject * SetItem(PyObject * self, PyObject * args);

PyObject * PutMask(PyObject * self, PyObject * args);

PyObject * ApplyRows(PyObject * self, PyObject * args, PyObject * kwargs);

PyObject * ShiftArrays(PyObject * self, PyObject * args);

int64_t SumBooleanMask(const int8_t * pIn, int64_t length);
