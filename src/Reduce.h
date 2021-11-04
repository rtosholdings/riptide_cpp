#pragma once

PyObject * Reduce(PyObject * self, PyObject * args);

PyObject * ReduceInternal(PyArrayObject * inArr1, int64_t func);
