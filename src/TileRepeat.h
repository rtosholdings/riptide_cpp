#pragma once

#include "RipTide.h"

// Input1: the recordarray to convert
// Input2: int64 array of offsets
// Input3: list of arrays pre allocated
PyObject* RecordArrayToColMajor(PyObject* self, PyObject* args);

