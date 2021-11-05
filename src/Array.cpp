#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"

#include "CommonInc.h"
#include "Compare.h"
#include "Array.h"

//#define LOGGING printf
#define LOGGING(...)

static int64_t FindMaxSize(PyObject * object)
{
    int64_t size = Py_SIZE(object);
    int64_t maxsize = 0;
    for (int64_t i = 0; i < size; i++)
    {
        int64_t itemsize = Py_SIZE(PyList_GET_ITEM(object, i));
        if (itemsize > maxsize)
            maxsize = itemsize;
    }
    return maxsize;
}

//---------------------------------------------------------
// arg1: the list or item to be converted to an array
static PyObject * ConvertFloat64(PyObject * object)
{
    npy_intp size = Py_SIZE(object);
    // We have a float
    // If the dtype requested was float32 handle this also
    PyArrayObject * pArray = AllocateNumpyArray(1, &size, NPY_FLOAT64);
    double * pFloat64 = (double *)PyArray_DATA(pArray);

    for (int64_t i = 0; i < size; i++)
    {
        pFloat64[i] = PyFloat_AsDouble(PyList_GET_ITEM(object, i));
    }
    return (PyObject *)pArray;
}

//---------------------------------------------------------
// arg1: the list or item to be converted to an array
/*
static
PyObject* ConvertFloat32(PyObject* object) {
   npy_intp size = Py_SIZE(object);
   // We have a float
   // If the dtype requested was float32 handle this also
   PyArrayObject* pArray = AllocateNumpyArray(1, &size, NPY_FLOAT32);
   float* pFloat32 = (float*)PyArray_DATA(pArray);

   for (int64_t i = 0; i < size; i++) {
      pFloat32[i] = (float)PyFloat_AsDouble(PyList_GET_ITEM(object, i));
   }
   return (PyObject*)pArray;
}
*/

//---------------------------------------------------------
// arg1: the list or item to be converted to an array
// NOTE: may return NULL on failure to convert
static PyObject * ConvertInt32(PyObject * object)
{
    npy_intp size = Py_SIZE(object);
    PyArrayObject * pArray = AllocateNumpyArray(1, &size, NPY_INT32);
    int32_t * pInt32 = (int32_t *)PyArray_DATA(pArray);

    for (int64_t i = 0; i < size; i++)
    {
        int overflow = 0;
        int64_t val = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(object, i), &overflow);
        if (overflow || val < NPY_MIN_INT32 || val > NPY_MAX_INT32)
        {
            // Failure due to out of range
            Py_DecRef((PyObject *)pArray);
            return NULL;
        }
        pInt32[i] = (int32_t)val;
    }
    return (PyObject *)pArray;
}

//---------------------------------------------------------
// arg1: the list or item to be converted to an array
static PyObject * ConvertInt64(PyObject * object)
{
    npy_intp size = Py_SIZE(object);
    PyArrayObject * pArray = AllocateNumpyArray(1, &size, NPY_INT64);
    int64_t * pInt64 = (int64_t *)PyArray_DATA(pArray);

    for (int64_t i = 0; i < size; i++)
    {
        int overflow = 0;
        // if overflow consider uint64?
        pInt64[i] = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(object, i), &overflow);
    }
    return (PyObject *)pArray;
}

//---------------------------------------------------------
// arg1: the list or item to be converted to an array
static PyObject * ConvertBool(PyObject * object)
{
    npy_intp size = Py_SIZE(object);
    // We have a bool (which is a subtype of long)
    PyArrayObject * pArray = AllocateNumpyArray(1, &size, NPY_BOOL);
    bool * pBool = (bool *)PyArray_DATA(pArray);

    for (int64_t i = 0; i < size; i++)
    {
        pBool[i] = PyList_GET_ITEM(object, i) == Py_True;
    }
    return (PyObject *)pArray;
}

//---------------------------------------------------------
// arg1: the list or item to be converted to an array
static PyObject * ConvertBytes(PyObject * object)
{
    npy_intp size = Py_SIZE(object);
    int64_t maxsize = FindMaxSize(object);
    PyArrayObject * pArray = AllocateNumpyArray(1, &size, NPY_STRING, maxsize);
    char * pChar = (char *)PyArray_DATA(pArray);

    for (int64_t i = 0; i < size; i++)
    {
        PyObject * item = PyList_GET_ITEM(object, i);
        int64_t strSize = Py_SIZE(item);

        // get pointer to string
        const char * pString = PyBytes_AS_STRING(item);
        char * pDest = &pChar[i * maxsize];
        char * pEnd = pDest + maxsize;
        char * pEndStr = pDest + strSize;
        while (pDest < pEndStr)
        {
            *pDest++ = *pString++;
        }
        // zero out rest
        while (pDest < pEnd)
        {
            *pDest++ = 0;
        }
    }
    return (PyObject *)pArray;
}

//---------------------------------------------------------
// arg1: the list or item to be converted to an array
static PyObject * ConvertUnicode(PyObject * object)
{
    npy_intp size = Py_SIZE(object);
    int64_t maxsize = FindMaxSize(object);
    PyArrayObject * pArray = AllocateNumpyArray(1, &size, NPY_UNICODE, maxsize * 4);
    uint32_t * pChar = (uint32_t *)PyArray_DATA(pArray);

    for (int64_t i = 0; i < size; i++)
    {
        PyObject * item = PyList_GET_ITEM(object, i);
        int64_t strSize = Py_SIZE(item);
        uint32_t * pDest = &pChar[i * maxsize];

        // get pointer to string
        PyUnicode_AsUCS4(item, pDest, maxsize, 0);

        // zero out rest
        if (strSize < maxsize)
            pDest[strSize] = 0;
    }
    return (PyObject *)pArray;
}

//---------------------------------------------------------
// arg1: kwarg
// searches for 'copy' and 'order'
//
// Returns 0 on error
// else returns flags with some combination of:
// NPY_ARRAY_ENSURECOPY | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS
int ProcessAsArrayKwargs(PyObject * kwargs)
{
    int flags = NPY_ARRAY_ENSUREARRAY;

    if (kwargs && Py_SIZE(kwargs) > 0)
    {
        PyObject * copykwarg = PyDict_GetItemString(kwargs, "copy");
        if (copykwarg)
        {
            if (PyBool_Check(copykwarg))
            {
                if (copykwarg == Py_True)
                {
                    flags |= NPY_ARRAY_ENSURECOPY;
                }
            }
            else
            {
                PyErr_Format(PyExc_ValueError, "The 'copy' argument must be either True or False.");
            }
        }

        PyObject * orderkwarg = PyDict_GetItemString(kwargs, "order");
        if (orderkwarg)
        {
            if (PyUnicode_Check(orderkwarg))
            {
                if (PyUnicode_CompareWithASCIIString(orderkwarg, "F") == 0)
                {
                    flags |= NPY_ARRAY_F_CONTIGUOUS;
                }
                else
                {
                    flags |= NPY_ARRAY_C_CONTIGUOUS;
                }
            }
            else
            {
                PyErr_Format(PyExc_ValueError, "The 'order' argument must be either 'F' or 'C'.");
            }
        }
    }
    return flags;
}

//---------------------------------------------------------
// arg1: the list or item to be converted to an array
// arg2: optional, the dtype
// kwargs:
//    dtype: may also be input as kwarg
//    copy: bool defaults to False
//    order: 'C' or 'F' .  Defaults to 'C'
//
// NOTES:
// A scalar will result in an array of 1 dim (as opposed to 0 dim)
// May return either a numpy array or a fast array
//
PyObject * AsAnyArray(PyObject * self, PyObject * args, PyObject * kwargs)
{
    PyArray_Descr * descr = NULL;

    if (Py_SIZE(args) != 1)
    {
        // NOTE: the third argument could be the copy flag but we want it as kwarg
        // for now
        if (Py_SIZE(args) != 2)
        {
            PyErr_Format(PyExc_TypeError, "AsArray takes 1 or 2 positional arguments but %lld were given", Py_SIZE(args));
            return NULL;
        }
        else
        {
            // convert 2nd arg to dtype descriptor
            descr = DTypeToDescr(PyTuple_GET_ITEM(args, 1));

            // check for error in dtype passed, if NULL error message already set
            if (! descr)
                return NULL;

            LOGGING("user passed dtype of %d  elementsize %d\n", descr->type_num, descr->elsize);
        }
    }

    PyObject * object = PyTuple_GET_ITEM(args, 0);
    int flags = NPY_ARRAY_ENSUREARRAY;

    if (kwargs)
    {
        PyObject * dtypekwarg = PyDict_GetItemString(kwargs, "dtype");
        if (dtypekwarg)
        {
            if (descr)
            {
                PyErr_Format(PyExc_TypeError, "AsArray got multiple values for 'dtype'");
                return NULL;
            }
            else
            {
                descr = DTypeToDescr(dtypekwarg);

                // check for error in dtype passed, if NULL error message already set
                if (! descr)
                    return NULL;
            }
        }
        flags = ProcessAsArrayKwargs(kwargs);
    }

    // TODO: handle when user passes a dtype
    if (descr == NULL)
    {
        // If this is already an array, return the value
        if (IsFastArrayOrNumpy((PyArrayObject *)object))
        {
            // If no transformation or copy requested, return same array
            if (flags == NPY_ARRAY_ENSUREARRAY)
            {
                Py_INCREF(object);
                return object;
            }
        }

        // if this is listlike, we can handle it
        if (PyList_Check(object))
        {
            npy_intp size = Py_SIZE(object);

            if (size == 0)
            {
                // return an empty 1d float64 array
                return (PyObject *)AllocateNumpyArray(1, &size, NPY_FLOAT64);
            }

            // check if all the same type
            PyObject * firstObject = PyList_GET_ITEM(object, 0);
            _typeobject * otype = firstObject->ob_type;

            int64_t countsize = size - 1;
            while (countsize > 0)
            {
                if (otype != PyList_GET_ITEM(object, countsize)->ob_type)
                {
                    LOGGING("Second type is %s\n", PyList_GET_ITEM(object, countsize)->ob_type->tp_name);
                    break;
                }
                countsize--;
            }

            if (countsize == 0)
            {
                // We have one type
                if (PyType_FastSubclass(otype, Py_TPFLAGS_LONG_SUBCLASS))
                {
                    // We have a long

                    if (PyBool_Check(firstObject))
                    {
                        // We have a bool (which is a subtype of long)
                        return ConvertBool(object);
                    }
                    else
                    {
                        // For Windows allocate NPY_INT32 and if out of range, switch to
                        // int64_t
                        if (sizeof(long) == 4)
                        {
                            PyObject * result = ConvertInt32(object);
                            if (result)
                                return result;
                        }
                        return ConvertInt64(object);
                    }
                }
                if (PyFloat_Check(firstObject))
                {
                    // We have a float
                    // If the dtype requested was float32 handle this also
                    return ConvertFloat64(object);
                }
                if (PyBytes_Check(firstObject))
                {
                    return ConvertBytes(object);
                }
                if (PyUnicode_Check(firstObject))
                {
                    return ConvertUnicode(object);
                }
                LOGGING("First type is %s\n", otype->tp_name);
            }
        }
        else
        {
            // TODO: handle scalars ourselves
            //// check for numpy scalar
            // bool isscalar = PyObject_TypeCheck((PyArrayObject*)object,
            // &PyGenericArrType_Type);
            //// check for python scalar
            // bool ispscalar = PyArray_IsPythonScalar(object);
            // LOGGING("scalar check %d %d\n", isscalar, ispscalar);
        }
    }

    // this will likely return a numpy array, not a fastarray
    PyArrayObject * pArray = (PyArrayObject *)PyArray_FromAny(object, descr, 0, 0, flags, NULL);

    // Check for the rank 0 array which numpy produces when it gets a scalar, and
    // change it to 1 dim
    if (! pArray || PyArray_NDIM(pArray) != 0)
    {
        return (PyObject *)pArray;
    }

    // Convert rank 0 to dim 1, length 1 array
    npy_intp size = 1;
    PyArrayObject * pNewArray = AllocateNumpyArray(1, &size, PyArray_TYPE(pArray), PyArray_ITEMSIZE(pArray));

    // copy over the data
    memcpy(PyArray_DATA(pNewArray), PyArray_DATA(pArray), PyArray_ITEMSIZE(pArray));

    // no longer need the numpy array
    Py_DecRef((PyObject *)pArray);

    return (PyObject *)pNewArray;
}

//---------------------------------------------------------
// arg1: the list or item to be converted to an array
//
PyObject * AsFastArray(PyObject * self, PyObject * args, PyObject * kwargs)
{
    PyArrayObject * pArray = (PyArrayObject *)AsAnyArray(self, args, kwargs);

    // Convert to FastArray
    return SetFastArrayView(pArray);
}
