#pragma once

#include "CommonInc.h"
#include "Recycler.h"

PyObject * ReIndex(PyObject * self, PyObject * args);
PyObject * NanInfCountFromSort(PyObject * self, PyObject * args);
PyObject * BinsToCutsBSearch(PyObject * self, PyObject * args);
PyObject * BinsToCutsSorted(PyObject * self, PyObject * args);
