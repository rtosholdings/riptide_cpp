#pragma once

#include "CommonInc.h"

// Basic call for sum, average
PyObject * Rolling(PyObject * self, PyObject * args);

PyObject * EmaAll32(PyObject * self, PyObject * args);

PyObject * InterpExtrap2d(PyObject * self, PyObject * args);