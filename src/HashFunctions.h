#pragma once

#include "CommonInc.h"
#include "Recycler.h"
PyObject * IsMember32(PyObject * self, PyObject * args);
PyObject * IsMember64(PyObject * self, PyObject * args);
PyObject * IsMemberCategorical(PyObject * self, PyObject * args);
PyObject * IsMemberCategoricalFixup(PyObject * self, PyObject * args);
