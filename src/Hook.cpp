#include "MathWorker.h"
#include "RipTide.h"
#include "ndarray.h"

#include "BasicMath.h"
#include "CommonInc.h"
#include "Compare.h"
#include "Convert.h"
#include "Hook.h"
#include "UnaryOps.h"
#include <cstddef>

//#define LOGGING printf
#define LOGGING(...)

#if defined(__clang__)
    #pragma clang diagnostic ignored "-Wmissing-braces"
#endif

typedef union
{
    void * vfunc;
    binaryfunc bfunc;
    unaryfunc ufunc;
    ternaryfunc tfunc;
} NUMBER_FUNC;

struct stMATH_HOOK
{
    // Place first for easy indexing
    MATH_OPERATION MathOp;

    // Our method that gets called
    NUMBER_FUNC Method;

    // Previous method that we hooked in front of
    NUMBER_FUNC SuperMethod;

    // Offset from PyNumberMethods
    int64_t Offset;
};

// forward reference
extern stMATH_HOOK g_MathHook[];

static PyObject * RiptideTernaryMathFunction(PyObject * self, PyObject * arg1, PyObject * arg2, int64_t index)
{
    LOGGING("in ternary math function %lld %s\n", index, self->ob_type->tp_name);
    // third param is optional and is for applying a modulus after the result
    if (arg2 == Py_None)
    {
        // TODO: intercept a value of 2 to apply square
        if (PyLong_Check(arg1))
        {
            int overflow = 0;
            int64_t power = PyLong_AsLongLongAndOverflow(arg1, &overflow);
            if (power == 1)
            {
                Py_INCREF(self);
                return self;
            }
            if (power == 2 || power == 3 || power == 4)
            {
                PyObject * square = BasicMathTwoInputsFromNumber(self, self, NULL, MATH_OPERATION::MUL);
                if (square != Py_None)
                {
                    if (power == 3)
                    {
                        // inplace multiply
                        BasicMathTwoInputsFromNumber(square, self, square, MATH_OPERATION::MUL);
                        Py_DECREF(square);
                    }
                    else if (power == 4)
                    {
                        // inplace multiply
                        BasicMathTwoInputsFromNumber(square, square, square, MATH_OPERATION::MUL);
                        Py_DECREF(square);
                    }
                    return square;
                }
            }
        }

        PyObject * result = BasicMathTwoInputsFromNumber(self, arg1, NULL, g_MathHook[index].MathOp);
        if (result != Py_None)
        {
            return result;
        }
    }
    // punt to numpy
    return g_MathHook[index].SuperMethod.tfunc(self, arg1, arg2);
}

static PyObject * RiptideTernaryMathFunctionInplace(PyObject * self, PyObject * arg1, PyObject * arg2, int64_t index)
{
    LOGGING("in ternary math inplace function %lld %s\n", index, self->ob_type->tp_name);
    PyObject * result = BasicMathTwoInputsFromNumber(self, arg1, self, g_MathHook[index].MathOp);
    if (arg2 == Py_None)
    {
        if (result != Py_None)
        {
            return result;
        }
    }
    // punt to numpy
    return g_MathHook[index].SuperMethod.tfunc(self, arg1, arg2);
}

static PyObject * RiptideUnaryMathFunction(PyObject * self, int64_t index)
{
    LOGGING("in unary math function %lld %s\n", index, self->ob_type->tp_name);
    PyObject * result = BasicMathOneInputFromNumber((PyArrayObject *)self, g_MathHook[index].MathOp, false);
    if (result != Py_None)
    {
        return result;
    }
    // punt to numpy
    return g_MathHook[index].SuperMethod.ufunc(self);
}

static PyObject * RiptideBinaryMathFunction(PyObject * self, PyObject * right, int64_t index)
{
    //
    // Assumes two inputs, the left always a FastArray
    LOGGING("in binary math function %lld %s\n", index, self->ob_type->tp_name);

    PyObject * result = BasicMathTwoInputsFromNumber(self, right, NULL, g_MathHook[index].MathOp);
    if (result != Py_None)
    {
        return result;
    }
    // punt to numpy
    return g_MathHook[index].SuperMethod.bfunc(self, right);
}

static PyObject * RiptideBinaryMathFunctionInplace(PyObject * self, PyObject * right, int64_t index)
{
    //
    // Assumes two inputs, the left always a FastArray and the operation is
    // inplace
    LOGGING("in binary math function inplace %lld %s\n", index, self->ob_type->tp_name);

    PyObject * result = BasicMathTwoInputsFromNumber(self, right, self, g_MathHook[index].MathOp);
    if (result != Py_None)
    {
        // Check if the output type matches self..
        int self_dtype = PyArray_TYPE((PyArrayObject *)self);
        int out_dtype = PyArray_TYPE((PyArrayObject *)result);
        if (self_dtype != out_dtype)
        {
            PyObject * newarray1 = ConvertSafeInternal((PyArrayObject *)result, self_dtype);
            if (newarray1)
            {
                // swap out for new array
                Py_DecRef(result);
                return newarray1;
            }
        }
        return result;
    }
    LOGGING("punted binary math function inplace %lld %s\n", index, self->ob_type->tp_name);
    // punt to numpy
    return g_MathHook[index].SuperMethod.bfunc(self, right);
}

// Define a way to make a baby math function which calls into larger math
// function with an op index the index is used to lookup in the MathHook table
// to call the proper function
#define DEF_HOOKB(_NAME_, _OP_) \
    PyObject * Riptide##_NAME_(PyObject * self, PyObject * right) \
    { \
        return RiptideBinaryMathFunction(self, right, _OP_); \
    }
#define DEF_HOOKI(_NAME_, _OP_) \
    PyObject * Riptide##_NAME_(PyObject * self, PyObject * right) \
    { \
        return RiptideBinaryMathFunctionInplace(self, right, _OP_); \
    }
#define DEF_HOOKU(_NAME_, _OP_) \
    PyObject * Riptide##_NAME_(PyObject * self) \
    { \
        return RiptideUnaryMathFunction(self, _OP_); \
    }
#define DEF_HOOKT(_NAME_, _OP_) \
    PyObject * Riptide##_NAME_(PyObject * self, PyObject * arg1, PyObject * arg2) \
    { \
        return RiptideTernaryMathFunction(self, arg1, arg2, _OP_); \
    }
#define DEF_HOOKS(_NAME_, _OP_) \
    PyObject * Riptide##_NAME_(PyObject * self, PyObject * arg1, PyObject * arg2) \
    { \
        return RiptideTernaryMathFunctionInplace(self, arg1, arg2, _OP_); \
    }

// Make a baby function for each routine that calls into the larger math
// function This order must match the order in g_MathHook
DEF_HOOKB(ADD, 0);
DEF_HOOKB(SUB, 1);
DEF_HOOKB(MUL, 2);
DEF_HOOKB(DIV, 3);
DEF_HOOKB(FDIV, 4);

DEF_HOOKI(IADD, 5);
DEF_HOOKI(ISUB, 6);
DEF_HOOKI(IMUL, 7);
DEF_HOOKI(IDIV, 8);
DEF_HOOKI(IFDIV, 9);

DEF_HOOKB(BWAND, 10);
DEF_HOOKB(BWOR, 11);
DEF_HOOKB(BWXOR, 12);
DEF_HOOKI(IBWAND, 13);
DEF_HOOKI(IBWOR, 14);
DEF_HOOKI(IBWXOR, 15);

DEF_HOOKB(LSHIFT, 16);
DEF_HOOKB(RSHIFT, 17);
DEF_HOOKI(ILSHIFT, 18);
DEF_HOOKI(IRSHIFT, 19);

DEF_HOOKB(MOD, 20);
DEF_HOOKU(ABS, 21);
DEF_HOOKU(NEG, 22);
DEF_HOOKU(INVERT, 23);

DEF_HOOKT(POWER, 24)
DEF_HOOKS(IPOWER, 25)

// This table is used to go from PyNumberMethods to our hook while recording the
// current function It is like a vtable
stMATH_HOOK g_MathHook[] = {
    { MATH_OPERATION::ADD, (void *)RiptideADD, NULL, offsetof(PyNumberMethods, nb_add) },                // x + y
    { MATH_OPERATION::SUB, (void *)RiptideSUB, NULL, offsetof(PyNumberMethods, nb_subtract) },           // x - y
    { MATH_OPERATION::MUL, (void *)RiptideMUL, NULL, offsetof(PyNumberMethods, nb_multiply) },           // x * y
    { MATH_OPERATION::DIV, (void *)RiptideDIV, NULL, offsetof(PyNumberMethods, nb_true_divide) },        // x / y
    { MATH_OPERATION::FLOORDIV, (void *)RiptideFDIV, NULL, offsetof(PyNumberMethods, nb_floor_divide) }, // x // y

    { MATH_OPERATION::ADD, (void *)RiptideIADD, NULL, offsetof(PyNumberMethods, nb_inplace_add) },                // x += y
    { MATH_OPERATION::SUB, (void *)RiptideISUB, NULL, offsetof(PyNumberMethods, nb_inplace_subtract) },           // x -= y
    { MATH_OPERATION::MUL, (void *)RiptideIMUL, NULL, offsetof(PyNumberMethods, nb_inplace_multiply) },           // x *= y
    { MATH_OPERATION::DIV, (void *)RiptideIDIV, NULL, offsetof(PyNumberMethods, nb_inplace_true_divide) },        // x /= y
    { MATH_OPERATION::FLOORDIV, (void *)RiptideIFDIV, NULL, offsetof(PyNumberMethods, nb_inplace_floor_divide) }, // x //= y

    { MATH_OPERATION::BITWISE_AND, (void *)RiptideBWAND, NULL, offsetof(PyNumberMethods, nb_and) },          // x & y
    { MATH_OPERATION::BITWISE_OR, (void *)RiptideBWOR, NULL, offsetof(PyNumberMethods, nb_or) },             // x | y
    { MATH_OPERATION::BITWISE_XOR, (void *)RiptideBWXOR, NULL, offsetof(PyNumberMethods, nb_xor) },          // x ^ y
    { MATH_OPERATION::BITWISE_AND, (void *)RiptideIBWAND, NULL, offsetof(PyNumberMethods, nb_inplace_and) }, // x &= y
    { MATH_OPERATION::BITWISE_OR, (void *)RiptideIBWOR, NULL, offsetof(PyNumberMethods, nb_inplace_or) },    // x |= y
    { MATH_OPERATION::BITWISE_XOR, (void *)RiptideIBWXOR, NULL, offsetof(PyNumberMethods, nb_inplace_xor) }, // x ^= y

    // TODO: implement
    { MATH_OPERATION::BITWISE_LSHIFT, (void *)RiptideLSHIFT, NULL, offsetof(PyNumberMethods, nb_lshift) },          // x << y
    { MATH_OPERATION::BITWISE_RSHIFT, (void *)RiptideRSHIFT, NULL, offsetof(PyNumberMethods, nb_rshift) },          // x >> y
    { MATH_OPERATION::BITWISE_LSHIFT, (void *)RiptideILSHIFT, NULL, offsetof(PyNumberMethods, nb_inplace_lshift) }, // x <<= y
    { MATH_OPERATION::BITWISE_RSHIFT, (void *)RiptideIRSHIFT, NULL, offsetof(PyNumberMethods, nb_inplace_rshift) }, // x >>= y

    { MATH_OPERATION::MOD, (void *)RiptideMOD, NULL, offsetof(PyNumberMethods, nb_remainder) },    // x % y
    { MATH_OPERATION::ABS, (void *)RiptideABS, NULL, offsetof(PyNumberMethods, nb_absolute) },     // abs(x)
    { MATH_OPERATION::NEG, (void *)RiptideNEG, NULL, offsetof(PyNumberMethods, nb_negative) },     // -x
    { MATH_OPERATION::INVERT, (void *)RiptideINVERT, NULL, offsetof(PyNumberMethods, nb_invert) }, // ~x

    { MATH_OPERATION::POWER, (void *)RiptidePOWER, NULL, offsetof(PyNumberMethods, nb_power) }, // x**y  or pow(x,y,modulus)
    { MATH_OPERATION::POWER, (void *)RiptideIPOWER, NULL, offsetof(PyNumberMethods, nb_inplace_power) }, // x**=y

    { MATH_OPERATION::LAST, NULL }
};

// Comparison functions: '__lt__', '__le__', '__eq__', '__ne__', '__gt__',
// '__ge__', Other: '__iter__', '__add__', '__radd__', '__sub__', '__rsub__',
// '__mul__', '__rmul__',
//'__mod__', '__rmod__', '__divmod__', '__rdivmod__', '__pow__', '__rpow__',
//'__neg__', '__pos__',
//'__abs__', '__bool__', '__invert__', '__lshift__', '__rlshift__',
//'__rshift__', '__rrshift__',
//'__and__', '__rand__', '__xor__', '__rxor__', '__or__', '__ror__', '__iand__',
//'__ixor__', '__ior__',
//'__int__', '__float__', '__iadd__', '__isub__', '__imul__', '__imod__',
//'__ipow__', '__ilshift__', '__irshift__',
//'__floordiv__', '__rfloordiv__', '__truediv__', '__rtruediv__',
//'__ifloordiv__', '__itruediv__',
//'__index__',  for slicing with non integer objects like x[y:z]
//'__matmul__', '__rmatmul__', '__imatmul__', '__len__', '__getitem__',
//'__setitem__', '__delitem__', Rich comparison opcodes
//-----------------------
//#define Py_LT 0
//#define Py_LE 1
//#define Py_EQ 2
//#define Py_NE 3
//#define Py_GT 4
//#define Py_GE 5

// Convert from python opcode to riptide, do not change this order
static const MATH_OPERATION COMP_TABLE[6] = { MATH_OPERATION::CMP_LT, MATH_OPERATION::CMP_LTE, MATH_OPERATION::CMP_EQ,
                                              MATH_OPERATION::CMP_NE, MATH_OPERATION::CMP_GT,  MATH_OPERATION::CMP_GTE };

static richcmpfunc g_CompareFunc = NULL;
static int64_t g_ishooked = 0;

//-----------------------------------------------------------------------------------
// Called when comparisons on arrays
static PyObject * CompareFunction(PyObject * left, PyObject * right, int opcode)
{
    // TODO: Change this to something like opcode < sizeof(COMP_TABLE) -- that'll
    // compile to the same code but still works if COMP_TABLE is ever changed.
    if (opcode < 6)
    {
        PyObject * result = BasicMathTwoInputsFromNumber(left, right, NULL, COMP_TABLE[opcode]);
        if (result != Py_None)
        {
            return result;
        }
    }
    // punt to numpy
    return g_CompareFunc(left, right, opcode);
}

//-----------------------------------------------------------------------------------
// Called once to hook math functions in FastArray class
// Input1: the FastArray class __dict__
// scans the dict
PyObject * BasicMathHook(PyObject * self, PyObject * args)
{
    PyObject * fastArrayClass;
    PyObject * npArrayClass;

    if (! PyArg_ParseTuple(args, "OO", &fastArrayClass, &npArrayClass))
    {
        return NULL;
    }

    if (! PyModule_Check(self))
    {
        PyErr_Format(PyExc_ValueError, "BasicMathHook must call from a module");
    }

    if (g_ishooked == 0)
    {
        PyNumberMethods * numbermethods = fastArrayClass->ob_type->tp_as_number;
        PyNumberMethods * numbermethods_np = npArrayClass->ob_type->tp_as_number;
        richcmpfunc comparefunc = fastArrayClass->ob_type->tp_richcompare;

        if (numbermethods)
        {
            g_ishooked = 1;
            bool hookNumpy = false;

            // Hook python comparison functions
            g_CompareFunc = fastArrayClass->ob_type->tp_richcompare;
            fastArrayClass->ob_type->tp_richcompare = CompareFunction;

            int64_t i = 0;
            while (true)
            {
                if (g_MathHook[i].MathOp == MATH_OPERATION::LAST)
                    break;

                // Use the offset to get to the function ptr for this method
                NUMBER_FUNC * numberfunc = (NUMBER_FUNC *)((const char *)numbermethods + g_MathHook[i].Offset);

                // Chain this function, insert our routine first
                g_MathHook[i].SuperMethod = *numberfunc;
                *numberfunc = g_MathHook[i].Method;

                // Take over numpy also?
                if (hookNumpy)
                {
                    numberfunc = (NUMBER_FUNC *)((const char *)numbermethods_np + g_MathHook[i].Offset);
                    *numberfunc = g_MathHook[i].Method;
                }
                i++;
            }

            if (hookNumpy)
            {
                // npArrayClass->ob_type->tp_richcompare = CompareFunction;
                // g_FastArrayType = NULL;
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}
