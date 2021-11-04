#pragma once

#include "CommonInc.h"

PyObject * TimeStringToNanos(PyObject * self, PyObject * args);

PyObject * DateStringToNanos(PyObject * self, PyObject * args);

PyObject * DateTimeStringToNanos(PyObject * self, PyObject * args);

PyObject * StrptimeToNanos(PyObject * self, PyObject * args);
