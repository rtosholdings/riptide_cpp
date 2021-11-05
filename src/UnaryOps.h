#ifndef RIPTIDECPP_UNARYOPS_H
#define RIPTIDECPP_UNARYOPS_H

#include "RipTide.h"

PyObject * BasicMathOneInput(PyObject * self, PyObject * args);

PyObject * BasicMathOneInputFromNumber(PyArrayObject * inObject1, int64_t funcNumber, bool inplace);

PyObject * BasicMathUnaryOp(PyObject * self, PyObject * args, PyObject * kwargs);

//--------------------------------------------------------------------
// multithreaded struct used for calling unary op codes
struct UNARY_CALLBACK
{
    union
    {
        UNARY_FUNC pUnaryCallback;
        UNARY_FUNC_STRIDED pUnaryCallbackStrided;
    };

    char * pDataIn;
    char * pDataOut;

    int64_t itemSizeIn;
    int64_t itemSizeOut;
};

#endif
