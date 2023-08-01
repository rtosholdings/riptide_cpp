#pragma once

#include "CommonInc.h"

// These are non-groupby routine -- straight array
enum ROLLING_FUNCTIONS
{
    ROLLING_SUM = 0,
    ROLLING_NANSUM = 1,

    // These output a float/double
    ROLLING_MEAN = 102,
    ROLLING_NANMEAN = 103,
    ROLLING_QUANTILE = 104,

    ROLLING_VAR = 106,
    ROLLING_NANVAR = 107,
    ROLLING_STD = 108,
    ROLLING_NANSTD = 109,
};

// these are functions that output same size
enum EMA_FUNCTIONS
{
    EMA_CUMSUM = 300,
    EMA_DECAY = 301,
    EMA_CUMPROD = 302,
    EMA_FINDNTH = 303,
    EMA_NORMAL = 304,
    EMA_WEIGHTED = 305,
    EMA_CUMNANMAX = 306,
    EMA_CUMNANMIN = 307,
    EMA_CUMMAX = 308,
    EMA_CUMMIN = 309,
};

// Basic call for sum, average
PyObject * Rolling(PyObject * self, PyObject * args);

PyObject * EmaAll32(PyObject * self, PyObject * args);

PyObject * InterpExtrap2d(PyObject * self, PyObject * args);