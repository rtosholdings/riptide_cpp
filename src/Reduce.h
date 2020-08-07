#pragma once

PyObject *
Reduce(PyObject *self, PyObject *args);

PyObject*  ReduceInternal(PyArrayObject *inArr1, INT64 func);