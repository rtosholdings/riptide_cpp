#ifndef RIPTABLE_CPP_TWO_INPUTS_H
#define RIPTABLE_CPP_TWO_INPUTS_H

#include "RipTide.h"

namespace riptable_cpp
{
    PyObject * calculate_two_inputs(PyObject * self, PyObject * args);

    PyObject * specific_calculate_two_inputs(PyObject * input1, PyObject * input2, PyObject * output, int64_t requested_op);

    PyObject * filtered_calculate(PyObject * self, PyObject * args);
}

#endif
