#pragma once
#include "RipTide.h"


PyObject *CompressFile(PyObject* self, PyObject *args, PyObject *kwargs);
PyObject *DecompressFile(PyObject* self, PyObject *args, PyObject *kwargs);
PyObject *MultiDecompressFiles(PyObject* self, PyObject *args, PyObject *kwargs);
PyObject *MultiStackFiles(PyObject* self, PyObject *args, PyObject *kwargs);
PyObject *MultiPossiblyStackFiles(PyObject* self, PyObject *args, PyObject *kwargs);
PyObject *MultiConcatFiles(PyObject* self, PyObject *args, PyObject *kwargs);
PyObject *SetLustreGateway(PyObject* self, PyObject *args);

