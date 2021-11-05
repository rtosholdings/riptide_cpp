#pragma once

#include "CommonInc.h"

PyObject * MBGet(PyObject * self, PyObject * args);

PyObject * BooleanIndex(PyObject * self, PyObject * args);

PyObject * BooleanSum(PyObject * self, PyObject * args);

extern PyObject * BooleanToFancy(PyObject * self, PyObject * args, PyObject * kwargs);

PyObject * ReIndexGroups(PyObject * self, PyObject * args);

PyObject * ReverseShuffle(PyObject * self, PyObject * args);
