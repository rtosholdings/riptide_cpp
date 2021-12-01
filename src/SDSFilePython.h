#pragma once
#include "RipTide.h"

PyObject * CompressFile(PyObject * self, PyObject * args, PyObject * kwargs);
PyObject * DecompressFile(PyObject * self, PyObject * args, PyObject * kwargs);
PyObject * MultiDecompressFiles(PyObject * self, PyObject * args, PyObject * kwargs);
PyObject * MultiStackFiles(PyObject * self, PyObject * args, PyObject * kwargs);
PyObject * MultiPossiblyStackFiles(PyObject * self, PyObject * args, PyObject * kwargs);
PyObject * MultiConcatFiles(PyObject * self, PyObject * args, PyObject * kwargs);
PyObject * SetLustreGateway(PyObject * self, PyObject * args);

/**
 * @brief Create Python type objects used by some SDS Python functions, then add
 * them to a module's dictionary.
 * @param module_dict The module dictionary for riptide_cpp.
 * @returns bool Indicates whether or not the type registration succeeded.
 */
bool RegisterSdsPythonTypes(PyObject * module_dict);
