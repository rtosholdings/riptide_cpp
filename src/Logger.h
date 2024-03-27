#pragma once

#include <Python.h>

RT_DLLEXPORT void SetupLogging();
void CleanupLogging();
PyObject * EnableLogging(PyObject * self, PyObject * args);
PyObject * DisableLogging(PyObject * self, PyObject * args);
PyObject * GetRiptideLoggers(PyObject * self, PyObject * args);