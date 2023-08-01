#pragma once

#include "CommonInc.h"

// NAN are ODD NUMBERED! follow this rule
enum REDUCE_FUNCTIONS
{
    REDUCE_SUM = 0,
    REDUCE_NANSUM = 1,

    // These output a float/double
    REDUCE_MEAN = 102,
    REDUCE_NANMEAN = 103,

    // ddof =1 for pandas, matlab   =0 for numpy
    REDUCE_VAR = 106,
    REDUCE_NANVAR = 107,
    REDUCE_STD = 108,
    REDUCE_NANSTD = 109,

    REDUCE_MIN = 200,
    REDUCE_NANMIN = 201,
    REDUCE_MAX = 202,
    REDUCE_NANMAX = 203,

    REDUCE_ARGMIN = 204,
    REDUCE_NANARGMIN = 205,
    REDUCE_ARGMAX = 206,
    REDUCE_NANARGMAX = 207,

    // For Jack TODO
    REDUCE_ANY = 208,
    REDUCE_ALL = 209,

    REDUCE_MIN_NANAWARE = 210,
};

PyObject * Reduce(PyObject * self, PyObject * args);

PyObject * ReduceInternal(PyArrayObject * inArr1, int64_t func);
