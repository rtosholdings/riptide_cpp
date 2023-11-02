#pragma once

#include <Python.h>

void SetupLogging();
void CleanupLogging();
PyObject * EnableLogging(PyObject * self, PyObject * args);
PyObject * DisableLogging(PyObject * self, PyObject * args);