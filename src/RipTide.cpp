// This define must only be declared in one module
#define SHAREDATA_MAIN_C_FILE
#include "ndarray.h"

#include "RipTide.h"
#include "platform_detect.h"
#include "Recycler.h"
#include "HashFunctions.h"
#include "Sort.h"
#include "MultiKey.h"
#include "Convert.h"
#include "MathWorker.h"
#include "Compare.h"
#include "BitCount.h"
#include "UnaryOps.h"
#include "GroupBy.h"
#include "Ema.h"
#include "Reduce.h"
#include "Merge.h"
#include "BasicMath.h"
#include "TimeWindow.h"
#include "Compress.h"
#include "SDSFilePython.h"
#include "Bins.h"
#include "DateTime.h"
#include "Hook.h"
#include "Array.h"

#undef LOGGING
//#define LOGGING printf
#define LOGGING(...)

#define LOG_ALLOC(...)

#ifndef max
    #define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
    #define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

//----------------------------------------------------------------------------------
CMathWorker * g_cMathWorker = new CMathWorker();
PyTypeObject * g_FastArrayType = NULL;
PyObject * g_FastArrayModule = NULL;

PyObject * g_FastArrayCall = NULL;

// original tp_dealloc for FastArray class
destructor g_FastArrayDeallocate = NULL;

// original tp_dealloc for FastArray instance
destructor g_FastArrayInstanceDeallocate = NULL;

PyArray_Descr * g_pDescrLongLong = NULL;
PyArray_Descr * g_pDescrULongLong = NULL;

//----------------------------------------------------------------------------------
// Lookup to go from 1 byte to 8 byte boolean values
int64_t gBooleanLUT64[256];
int32_t gBooleanLUT32[16];

int64_t gBooleanLUT64Inverse[256];
int32_t gBooleanLUT32Inverse[16];

// upcast table from 0-13
struct stUpCast
{
    int dtype1;
    int dtype2;
};

stUpCast gUpcastTable[14 * 14];
stUpCast gUpcastTableComparison[14 * 14];

// BUGBUG this is the windows version
// Linux has NPY_INT / UINT as 8
stScalarType NpTypeObjects[MAX_NUMPY_TYPE] = { { NULL, 1, NPY_BOOL },
                                               { NULL, 1, NPY_BYTE },
                                               { NULL, 1, NPY_UBYTE },
                                               { NULL, 2, NPY_SHORT },
                                               { NULL, 2, NPY_USHORT },
                                               { NULL, 4, NPY_INT },
                                               { NULL, 4, NPY_UINT },
#if defined(RT_OS_WINDOWS)
                                               { NULL, 4, NPY_INT32 },  // believe this is 8 bytes on Linux
                                               { NULL, 4, NPY_UINT32 }, // believe this is 8 bytes on Linux
#else
                                               { NULL, 8, NPY_LONG },  // believe this is 8 bytes on Linux enum 7
                                               { NULL, 8, NPY_ULONG }, // believe this is 8 bytes on Linux enum 8
#endif
                                               { NULL, 8, NPY_INT64 },
                                               { NULL, 8, NPY_UINT64 },
                                               { NULL, 4, NPY_FLOAT },
                                               { NULL, 8, NPY_DOUBLE },
                                               { NULL, sizeof(long double), NPY_LONGDOUBLE },
                                               { NULL, 8, NPY_CFLOAT },
                                               { NULL, 16, NPY_CDOUBLE },
                                               { NULL, sizeof(long double) * 2, NPY_CLONGDOUBLE },
                                               { NULL, 8, NPY_OBJECT },
                                               { NULL, 0, NPY_STRING },  // ? check itemsize
                                               { NULL, 0, NPY_UNICODE }, // ? check itemsize
                                               { NULL, 0, NPY_VOID },
                                               { NULL, 0, NPY_DATETIME },
                                               { NULL, 0, NPY_TIMEDELTA },
                                               { NULL, 2, NPY_HALF } };

const char * gNumpyTypeToString[NUMPY_LAST_TYPE] = {
    "NPY_BOOL",   "NPY_BYTE",   "NPY_UBYTE",   "NPY_SHORT",  "NPY_USHORT",     "NPY_INT",       "NPY_UINT",
#if defined(RT_OS_WINDOWS)
    "NPY_INT32",  // not on linux
    "NPY_UINT32", // not on linux
#else
    "NPY_LONG64",                                                      // not on linux
    "NPY_LONG64",                                                      // not on linux
#endif
    "NPY_INT64",  "NPY_UINT64", "NPY_FLOAT",   "NPY_DOUBLE", "NPY_LONGDOUBLE", "NPY_CFLOAT",    "NPY_CDOUBLE", "NPY_CLONGDOUBLE",
    "NPY_OBJECT", "NPY_STRING", "NPY_UNICODE", "NPY_VOID",   "NPY_DATETIME",   "NPY_TIMEDELTA", "NPY_HALF"
};

int32_t gNumpyTypeToSize[NUMPY_LAST_TYPE] = {
    1, // NPY_BOOL =     0,
    1, // NPY_BYTE,      1
    1, // NPY_UBYTE,     2
    2, // NPY_SHORT,     3
    2, // NPY_USHORT,    4
    4, // NPY_INT,       5 // TJD is this used or just NPY_INT32?
    4, // NPY_UINT,      6 // TJD is this used or just NPY_UINT32?
#if defined(RT_OS_WINDOWS)
    4, // NPY_INT32,      7 // TODO: change to sizeof(long)
    4, // NPY_UINT32,     8
#else
    8,                                                                 // NPY_INT32,      7
    8,                                                                 // NPY_UINT32,     8

#endif
    8,                       // NPY_INT64,  9
    8,                       // NPY_UINT64, 10
    4,                       // NPY_FLOAT,     11
    8,                       // NPY_DOUBLE,    12
    sizeof(long double),     // NPY_LONGDOUBLE,13
    8,                       // NPY_CFLOAT,    14
    16,                      // NPY_CDOUBLE,   15
    sizeof(long double) * 2, // NPY_CLONGDOUBLE,16

    0, // NPY_OBJECT =   17,
    0, // NPY_STRING,    18
    0, // NPY_UNICODE,   19
    0, // NPY_VOID,      20
    0, // NPY_DATETIME,  21
    0, // NPY_TIMEDELTA, 22
    0  // NPY_HALF,      23
};

//---------------------------------------------------------------
// Pass in a type such as np.int32 and a NPY_INT32 will be returned
// returns -1 on failure adn set error string
// otherwise returns NPY_TYPE
int32_t TypeToDtype(PyTypeObject * out_dtype)
{
    if (PyType_Check(out_dtype))
    {
        for (int i = 0; i < 24; i++)
        {
            if (out_dtype == NpTypeObjects[i].type)
            {
                LOGGING("found type %d\n", NpTypeObjects[i].typenum);
                return NpTypeObjects[i].typenum;
            }
        }
    }

    PyErr_SetString(PyExc_ValueError, "DType conversion failed");

    return -1;
}

//---------------------------------------------
// Return the objects dtype
// Returns -1 on types > NPY_VOID suchas NPY_DATETIME,
// otherwise the type number such as NPY_INT32 or NPY_FLOAT
int32_t ObjectToDtype(PyArrayObject * obj)
{
    int32_t result = PyArray_TYPE(obj);

    // NOTE: This code does not handle NPY_DATETIME, NPY_TIMEDELTA, etc.
    if (result < 0 || result > NPY_VOID)
        return -1;
    return result;
}

//------------------------------------------------------------
// gets the itemsize for ndarray, useful for chararray
// int32_t = 4
// int64_t = 8
// FLOAT32 = 4
// '|S5' = 5 for chararry
int64_t NpyItemSize(PyObject * self)
{
    return PyArray_ITEMSIZE((PyArrayObject *)self);
}

//------------------------------------------------------------
// Returns a string description of the numpy type
const char * NpyToString(int32_t numpyType)
{
    if (numpyType < 0 || numpyType >= NUMPY_LAST_TYPE)
    {
        return "<unknown>";
    }
    else
    {
        return gNumpyTypeToString[numpyType];
    }
}

//------------------------------------------------------------
// Returns 0 for chars, strings, or unsupported types
int32_t NpyToSize(int32_t numpyType)
{
    if (numpyType < 0 || numpyType >= NUMPY_LAST_TYPE)
    {
        return 0;
    }
    else
    {
        return gNumpyTypeToSize[numpyType];
    }
}

//-----------------------------------------------------------
// Checks to see if an array is contiguous, if not, it makes a copy
// If a copy is made, the caller is responsible for decrementing the ref count
// Returns NULL on failure
// On Success returns back a contiguous array
PyArrayObject * EnsureContiguousArray(PyArrayObject * inObject)
{
    int arrFlags = PyArray_FLAGS(inObject);

    // make sure C or F contiguous
    if (! (arrFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
    {
        // Have to make a copy (which needs to be deleted later)
        inObject = (PyArrayObject *)PyArray_FromAny((PyObject *)inObject, NULL, 0, 0, NPY_ARRAY_ENSURECOPY, NULL);
        if (! inObject)
        {
            PyErr_Format(PyExc_ValueError, "RipTide: Error converting non-contiguous array");
            return NULL;
        }
    }
    return inObject;
}

//----------------------------------------------------------------
// Calculate the total number of bytes used by the array.
// TODO: Need to extend this to accomodate strided arrays.
int64_t CalcArrayLength(int ndim, npy_intp * dims)
{
    int64_t length = 1;

    // handle case of zero length array
    if (dims && ndim > 0)
    {
        for (int i = 0; i < ndim; i++)
        {
            length *= dims[i];
        }
    }
    else
    {
        // Want to set this to zero, but scalar issue?
        // length = 0;
    }
    return length;
}

//----------------------------------------------------------------
// calcluate the total number of bytes used
int64_t ArrayLength(PyArrayObject * inArr)
{
    int ndim = PyArray_NDIM(inArr);
    npy_intp * dims = PyArray_DIMS(inArr);
    return CalcArrayLength(ndim, dims);
}

//----------------------------------------------------------------
//
void CopyPythonString(PyObject * objColName, char * destBuffer, size_t maxLen)
{
    // we know that object represents an integer - so convert it into C long
    // make it a string

    // Copy over the name of the column
    if (PyUnicode_Check(objColName))
    {
        PyObject * temp2 = PyUnicode_AsASCIIString(objColName);
        if (temp2 != NULL)
        {
            char * str = PyBytes_AsString(temp2);
            LOGGING("str: %s\n", str);

            size_t len = strlen(str);

            // clamp length
            if (len >= maxLen)
                len = maxLen - 1;

            strncpy(destBuffer, str, len);
            destBuffer[len] = 0;

            // Release reference
            Py_DecRef(temp2);
        }
        else
        {
            LogError("Null unicode string\n");
        }
    }
    else
    {
        if (PyBytes_Check(objColName))
        {
            char * str = PyBytes_AsString(objColName);
            LOGGING("str: %s\n", str);

            size_t len = strlen(str);

            // clamp length
            if (len >= maxLen)
                len = maxLen - 1;

            strncpy(destBuffer, str, len);
            destBuffer[len] = 0;
        }
        else
        {
            // NOT a string
            LogInform("!!! str: <error!!>\n");
        }
    }
}

//-----------------------------------------------------------------------------------
extern "C"
{
    // NOTE: Could keep an array counter here to determine how many outstanding
    // arrays there are Called on PyDecRef when refcnt goes to zero
    void FastArrayDestructor(PyObject * object)
    {
        LOG_ALLOC("called %lld %ld\n", object->ob_refcnt, object->ob_type->tp_flags);
        // If we are the base then nothing else is attached to this array object
        // Attempt to recycle, if succeeds, the refnct will be incremented so we can
        // hold on
        if (! RecycleNumpyInternal((PyArrayObject *)object))
        {
            PyArrayObject * pArray = (PyArrayObject *)object;
            LOG_ALLOC("freeing %p %s  len:%lld\n", object, object->ob_type->tp_name, ArrayLength(pArray));
            g_FastArrayInstanceDeallocate(object);
        } // else we are keeping it around
    }
}

//-----------------------------------------------------------------------------------
// Internal use only. Called once from python to set the FastArray type.
// this method now takes over tp_dealloc which gets called
// when Py_DECREF decrement object ref counter to 0.
PyObject * SetFastArrayType(PyObject * self, PyObject * args)
{
    PyObject * arg1 = NULL;
    PyObject * arg2 = NULL;

    if (! PyArg_ParseTuple(args, "OO", &arg1, &arg2))
        return NULL;

    // take over deallocation
    if (g_FastArrayType == NULL)
    {
        g_FastArrayCall = arg2;

        npy_intp length = 1;
        // Now allocate a small array to get the type
        PyArrayObject * arr = AllocateNumpyArray(1, &length, NPY_BOOL);

        g_FastArrayType = arg1->ob_type;
        Py_IncRef((PyObject *)g_FastArrayType);

        // Take over dealloc so we can recycle
        LOGGING("SetFastArrayType dealloc %p %p\n", g_FastArrayType->tp_dealloc, FastArrayDestructor);
        g_FastArrayDeallocate = g_FastArrayType->tp_dealloc;
        g_FastArrayInstanceDeallocate = arr->ob_base.ob_type->tp_dealloc;

        LOGGING("dealloc  %p %p %p %p\n", g_FastArrayDeallocate, g_FastArrayInstanceDeallocate, FastArrayDestructor,
                g_FastArrayType->tp_dealloc);

        // Swap ourselves in
        g_FastArrayType->tp_dealloc = FastArrayDestructor;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

//----------------------------------------------------------------
// Arg1 : numpy array
// Arg2 : numpy array
// Returns True if arrays have same underlying memory address
PyObject * CompareNumpyMemAddress(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    PyArrayObject * inArr2 = NULL;

    if (! PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2))
        return NULL;

    if (PyArray_BYTES(inArr1) == PyArray_BYTES(inArr2))
    {
        Py_INCREF(Py_True);
        return Py_True;
    }
    else
    {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

//----------------------------------------------------------------------------------
// Used to flip numpy arrays to FastArray
// If called with existing FastArray, then same array is returned
PyObject * SetFastArrayView(PyArrayObject * pArray)
{
    if ((g_FastArrayType != NULL && pArray->ob_base.ob_type != g_FastArrayType))
    {
        LOGGING("!! setting view\n");

        // Convert to fast array
        PyObject * result = PyArray_View(pArray, NULL, g_FastArrayType);

        if (result == NULL)
        {
            printf("!!! PyArray_View failed\n");
        }
        else
        {
            // succeeded, remove reference from previous array
            Py_DecRef((PyObject *)pArray);
            return result;
        }
    }
    else
    {
        LOGGING("!!not setting view\n");
    }

    // If already FastArray or failure, return same array
    return (PyObject *)pArray;
}

//---------------------------------------------
// Returns NULL if python object is not callable
// otherwise return the function
PyFunctionObject * GetFunctionObject(PyObject * arg1)
{
    PyFunctionObject * function = NULL;
    if (PyInstanceMethod_Check(arg1))
    {
        PyInstanceMethodObject * f = (PyInstanceMethodObject *)arg1;
        // printf("instance\n");
        function = (PyFunctionObject *)f->func;
    }

    if (PyMethod_Check(arg1))
    {
        function = (PyFunctionObject *)PyMethod_Function(arg1);
    }
    else if (PyFunction_Check(arg1))
    {
        function = (PyFunctionObject *)arg1;
    }

    return function;
}

/*
//-----------------------------------------------------------------------------------
// This routine is not finished
void* GetFunctionName(PyObject *arg1) {
PyFunctionObject* function = NULL;
const char* functionName = NULL;
if (PyCFunction_Check(arg1)) {
PyCFunctionObject* f = (PyCFunctionObject*)arg1;
functionName = f->m_ml->ml_name;
//printf("CFunction %s\n", f->m_ml->ml_name);
}

function = GetFunctionObject(arg1);

if (!function) {

// Supply a temp name here
functionName = "Test";

// TODO: If this is a ufunc, the __call__ wil exist
// Also, can get the function name (a PyUnicodeString pbject)
//functionName = PyObject_GetAttrString(arg1, "__name__")

//// Check for __call__ from ufunc
//PyObject* callfunc = PyObject_GetAttrString(arg1, "__call__");

//LOGGING("call is at %p\n", callfunc);

//if (!callfunc || !PyCallable_Check(callfunc)) {
//   PyTypeObject* type = (PyTypeObject*)PyObject_Type(arg1);

//   PyErr_Format(PyExc_ValueError, "Argument must be a function or a method not
%s.  Call was found at %p\n", type->tp_name, callfunc);
//   return NULL;
//}

//function = GetFunctionObject(callfunc);
//arg1 = callfunc;

//// Dont need to hold onto ref count
//Py_DecRef(callfunc);
}

if (function) {
// copy this immediately?
functionName = DumpPyObject(function->func_qualname);
}
}
*/

//-----------------------------------------------------------------------------------
// Called to time ledger if ledger is on
//
PyObject * LedgerFunction(PyObject * self, PyObject * args, PyObject * kwargs)
{
    Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

    if (tupleSize < 2)
    {
        PyErr_Format(PyExc_TypeError, "LedgerFunction requires two args instead of %llu args", tupleSize);
        return NULL;
    }

    PyObject * arg1 = PyTuple_GET_ITEM(args, 0);

    // Check if callable
    if (! PyCallable_Check(arg1))
    {
        PyTypeObject * type = (PyTypeObject *)PyObject_Type(arg1);

        PyErr_Format(PyExc_TypeError, "Argument must be a function or a method not %s\n", type->tp_name);
        return NULL;
    }

    // New reference
    PyObject * newargs = PyTuple_New(tupleSize - 1);

    // Shift all the arguments over
    for (int i = 0; i < (tupleSize - 1); i++)
    {
        // Getitem steals a reference, it does not increment it
        PyObject * item = PyTuple_GET_ITEM(args, i + 1);

        // Increment ref because this item is now in two tuples
        Py_INCREF(item);

        LOGGING("arg %d refcnt %llu\n", i, item->ob_refcnt);
        PyTuple_SET_ITEM(newargs, i, item);
    }

    PyObject * returnObject = PyObject_Call(arg1, newargs, kwargs);

    Py_DECREF(newargs);

    // Can only handle one return object
    if (returnObject && PyArray_Check(returnObject))
    {
        // possibly convert to fast array
        returnObject = SetFastArrayView((PyArrayObject *)returnObject);
    }
    return returnObject;
}

int64_t g_TotalAllocs = 0;

//-----------------------------------------------------------------------------------
PyArrayObject * AllocateNumpyArray(int ndim, npy_intp * dims, int32_t numpyType, int64_t itemsize, bool fortran_array,
                                   npy_intp * strides)
{
    PyArrayObject * returnObject = nullptr;
    const int64_t len = CalcArrayLength(ndim, dims);
    bool commonArray = false;

    // PyArray_New (and the functions it wraps) don't truly respect the 'flags'
    // argument passed into them; they only check whether it's zero or non-zero,
    // and based on that they set the NPY_ARRAY_CARRAY or NPY_ARRAY_FARRAY flags.
    // Construct our flags value so we end up with an array with the layout the
    // caller requested.
    const int array_flags = fortran_array ? NPY_ARRAY_F_CONTIGUOUS : 0;

    // Don't recycle arrays with certain characteristics; they'd add significant
    // complexity to the caching and aren't the typical case anyway.
    // TODO: Check for non-default striding -- don't want to try to cache those;
    // but 'strides'
    //       can be non-null even for simple arrays (in which case it seems to be
    //       just the element size).
    if (itemsize == 0 && ! PyTypeNum_ISFLEXIBLE(numpyType))
    {
        commonArray = true;
        // try to find recycled array
        returnObject = RecycleFindArray(ndim, numpyType, len);

        // Did we find a recycled array to use? If so, return it.
        if (returnObject)
        {
            LOGGING("got recycled  len:%lld  ref cnt %llu!\n", len, returnObject->ob_base.ob_refcnt);

            // TODO: Need to check here whether the array we got matches the
            // 'fortran_array' parameter;
            //       if it doesn't, we need to return the recycled array to the pool
            //       and continue to the normal allocation path below.

            // Flip to FastArray since we store the base object now
            return (PyArrayObject *)SetFastArrayView(returnObject);
        }
    }

    // Make one dimension size on stack
    volatile int64_t dimensions[1] = { len };

    // This is the safest way...
    if (! dims)
    {
        // Happens with a=FA([]); 100*a;  or  FA([1])[0] / FA([2])
        // printf("dims was null when allocating\n");
        ndim = 1;
        dims = (npy_intp *)dimensions;
    }

    PyTypeObject * const allocType = g_FastArrayType ? g_FastArrayType : &PyArray_Type;

    if (commonArray)
    {
        // NOTE: We now directly allocate a FastArray
        returnObject = (PyArrayObject *)PyArray_New(allocType, ndim, dims, numpyType, NULL, nullptr, 0, array_flags, nullptr);

        if (! returnObject)
        {
            // GCNOW (attempt to free memory) and try again
            GarbageCollect(0, false);
            returnObject = (PyArrayObject *)PyArray_New(allocType, ndim, dims, numpyType, NULL, nullptr, 0, array_flags, nullptr);
        }
    }
    else
    {
        // Special allocation path
        bool bRuntFile = false;

        // probably runt object from matlab -- have to fix this up or it will fail
        // comes from empty strings in matlab - might need to
        if (PyTypeNum_ISFLEXIBLE(numpyType) && itemsize == 0)
        {
            itemsize = 1;
            bRuntFile = true;
        }
        // NOTE: this path taken when we already have data in our own memory
        returnObject = (PyArrayObject *)PyArray_New(allocType, ndim, dims, numpyType,
                                                    strides, // Strides
                                                    nullptr, static_cast<int>(itemsize), array_flags, NULL);

        if (! returnObject)
        {
            // GCNOW and try again
            GarbageCollect(0, false);
            returnObject = (PyArrayObject *)PyArray_New(allocType, ndim, dims, numpyType,
                                                        strides, // Strides
                                                        nullptr, static_cast<int>(itemsize), array_flags, NULL);
        }

        if (returnObject != NULL && bRuntFile)
        {
            char * pData = (char *)PyArray_BYTES(returnObject);
            // nullify string for special matlab runt case
            *pData = 0;
        }
    }

    if (! returnObject)
    {
        printf(
            "!!!out of memory allocating numpy array size:%llu  dims:%d  "
            "dtype:%d  itemsize:%lld  flags:%d  dim0:%lld\n",
            len, ndim, numpyType, itemsize, array_flags, (int64_t)dims[0]);
        return nullptr;
    }

    // No longer doing as we allocate via subtype
    // PyArrayObject* oldObject = returnObject;
    returnObject = (PyArrayObject *)SetFastArrayView(returnObject);
    return returnObject;
}

PyArrayObject * AllocateNumpyArrayForData(int ndim, npy_intp * dims, int32_t numpyType, int64_t itemsize, char * data,
                                          int array_flags, npy_intp * strides)
{
    // If there's no data, short-circuit and return nullptr;
    // we could call the regular AllocateNumpyArray() function but this
    // behavior gives _this_ function more well-defined semantics.
    if (! data)
    {
        return nullptr;
    }

    PyArrayObject * returnObject = nullptr;
    const int64_t len = CalcArrayLength(ndim, dims);

    // This is the safest way...
    if (! dims)
    {
        // Make one dimension size on stack
        int64_t dimensions[1] = { len };

        // Happens with a=FA([]); 100*a;
        // printf("dims was null when allocating\n");
        ndim = 1;
        dims = (npy_intp *)dimensions;
    }

    PyTypeObject * const allocType = g_FastArrayType ? g_FastArrayType : &PyArray_Type;

    // Special allocation path
    bool bRuntFile = false;

    // probably runt object from matlab -- have to fix this up or it will fail
    // comes from empty strings in matlab - might need to
    if (numpyType == NPY_STRING && itemsize == 0)
    {
        itemsize = 1;
        bRuntFile = true;
    }
    // NOTE: this path taken when we already have data in our own memory
    returnObject = (PyArrayObject *)PyArray_New(allocType, ndim, dims, numpyType,
                                                strides, // Strides
                                                data, (int)itemsize, array_flags, NULL);

    if (! returnObject)
    {
        // GCNOW and try again
        GarbageCollect(0, false);
        returnObject = (PyArrayObject *)PyArray_New(allocType, ndim, dims, numpyType,
                                                    strides, // Strides
                                                    data, static_cast<int>(itemsize), array_flags, NULL);
    }

    if (returnObject != NULL && bRuntFile)
    {
        char * pData = (char *)PyArray_BYTES(returnObject);
        // nullify string for special matlab runt case
        *pData = 0;
    }

    if (! returnObject)
    {
        printf(
            "!!!out of memory allocating numpy array size:%llu  dims:%d  "
            "dtype:%d  itemsize:%lld  flags:%d  dim0:%lld\n",
            len, ndim, numpyType, itemsize, array_flags, (int64_t)dims[0]);
        return nullptr;
    }

    // No longer doing as we allocate via subtype
    // PyArrayObject* oldObject = returnObject;
    returnObject = (PyArrayObject *)SetFastArrayView(returnObject);
    return returnObject;
}

//-----------------------------------------------------------------------------------
// Check recycle pool
PyArrayObject * AllocateLikeNumpyArray(PyArrayObject const * inArr, int32_t numpyType)
{
    const int ndim = PyArray_NDIM(inArr);
    npy_intp * const dims = PyArray_DIMS(const_cast<PyArrayObject *>(inArr));

    // If the strides are all "normal", the array is C_CONTIGUOUS,
    // and this is _not_ a string / flexible array, try to re-use an array
    // from the recycler (array pool).
    if ((PyArray_ISNUMBER(const_cast<PyArrayObject *>(inArr)) || PyArray_ISBOOL(const_cast<PyArrayObject *>(inArr))) &&
        PyArray_ISCARRAY(const_cast<PyArrayObject *>(inArr)))
    {
        return AllocateNumpyArray(ndim, dims, numpyType, 0, false, nullptr);
    }

    // If we couldn't re-use an array from the recycler (for whatever reason),
    // allocate a new one based on the old one but override the type.
    // TODO: How to handle the case where either the prototype array is a string
    // array xor numpyType is a string type?
    //       (For the latter, we don't have the target itemsize available here, so
    //       we don't know how to allocate the array.)
    PyArray_Descr * const descr = PyArray_DescrFromType(numpyType);
    if (! descr)
    {
        return nullptr;
    }
    PyArrayObject * returnObject =
        (PyArrayObject *)PyArray_NewLikeArray(const_cast<PyArrayObject *>(inArr), NPY_KEEPORDER, descr, 1);
    CHECK_MEMORY_ERROR(returnObject);

    if (! returnObject)
    {
        return nullptr;
    }

    returnObject = (PyArrayObject *)SetFastArrayView(returnObject);
    return returnObject;
}

//-----------------------------------------------------------------------------------
// Check recycle pool
PyArrayObject * AllocateLikeResize(PyArrayObject * inArr, npy_intp rowSize)
{
    int32_t numpyType = PyArray_TYPE(inArr);

    PyArrayObject * result = NULL;

    if (PyTypeNum_ISOBJECT(numpyType))
    {
        printf("!!! Cannot allocate for object\n");
        return nullptr;
    }
    else if (PyTypeNum_ISSTRING(numpyType))
    {
        int64_t itemSize = NpyItemSize((PyObject *)inArr);
        result = AllocateNumpyArray(1, &rowSize, numpyType, itemSize);
    }
    else
    {
        result = AllocateNumpyArray(1, &rowSize, numpyType);
    }

    CHECK_MEMORY_ERROR(result);
    return result;
}

//-----------------------------------------------------------------------------------
// Allocate a numpy array
// Arg1: List of python ints
// Arg2: dtype number
// Arg3: itemsize for the dtype (useful for strings)
// Arg4: boolean (whether or not to set Fortran flag
// Returns a fastarry
PyObject * Empty(PyObject * self, PyObject * args)
{
    PyObject * inDimensions;
    int dtype;
    int64_t itemsize;
    PyObject * isFortran;

    if (! PyArg_ParseTuple(args, "O!iLO!", &PyList_Type, &inDimensions, &dtype, &itemsize, &PyBool_Type, &isFortran))
        return NULL;

    const bool is_fortran = isFortran == Py_True;

    int64_t dims = PyList_GET_SIZE(inDimensions);
    if (dims != 1)
    {
        Py_IncRef(Py_None);
        return Py_None;
    }

#if NPY_SIZEOF_PY_INTPTR_T == NPY_SIZEOF_LONG
    #define PyLong_AsNPY_INTPAndOverflow PyLong_AsLongAndOverflow
#elif defined(PY_LONG_LONG) && (NPY_SIZEOF_PY_INTPTR_T == NPY_SIZEOF_LONGLONG)
    #define PyLong_AsNPY_INTPAndOverflow PyLong_AsLongLongAndOverflow
#else
    #error Unable to determine how to parse PyLong to npy_intp with overflow-checking.
#endif

    // first item in the list is a python int
    int overflow = 0;
    npy_intp dimension1 = static_cast<npy_intp>(PyLong_AsNPY_INTPAndOverflow(PyList_GET_ITEM(inDimensions, 0), &overflow));
    if (overflow)
    {
        return PyErr_Format(PyExc_ValueError, "Dimension is too large for this system.");
    }

    PyObject * result = (PyObject *)AllocateNumpyArray(1, &dimension1, dtype, itemsize, is_fortran);
    CHECK_MEMORY_ERROR(result);
    return result;
}

int32_t GetArrDType(PyArrayObject * inArr)
{
    int32_t dtype = PyArray_TYPE(inArr);
#if defined(RT_OS_WINDOWS)

    if (dtype == NPY_INT)
    {
        dtype = NPY_INT32;
    }
    if (dtype == NPY_UINT)
    {
        dtype = NPY_UINT32;
    }

#else
    if (dtype == NPY_LONGLONG)
    {
        dtype = NPY_INT64;
    }
    if (dtype == NPY_ULONGLONG)
    {
        dtype = NPY_UINT64;
    }
#endif

    return dtype;
}

//-----------------------------------------------------------------------------------
// Convert python object to
// pInput: pointer to array value that needs to be converted
// numpyInType: dtype of pInput
// pDest: to be filled in
// numpyOutType: type to convert to
//
// Returns true on success, false otherwise
// On Return fills in pDest with int
//
// NOTE: Does not handle strings yet
// NPY_BOOL=0,
// NPY_BYTE     1
// NPY_UBYTE    2
// NPY_SHORT    3
// NPY_USHORT   4
// NPY_INT      5
// NPY_UINT     6
// NPY_INT32      7
// NPY_UINT32     8
// NPY_INT64      9
// NPY_UINT64     10
// NPY_FLOAT      11
// NPY_DOUBLE     12
// NPY_LONGDOUBLE 13
bool ConvertSingleItemArray(void * pInput, int16_t numpyInType, _m256all * pDest, int16_t numpyOutType)
{
    int64_t value = 0;
    double fvalue = 0;

    switch (numpyInType)
    {
    case NPY_BOOL:
        value = (int64_t) * (bool *)pInput;
        fvalue = (double)value;
        break;
    case NPY_INT8:
        value = (int64_t) * (int8_t *)pInput;
        fvalue = (double)value;
        break;
    case NPY_UINT8:
        value = (int64_t) * (uint8_t *)pInput;
        fvalue = (double)value;
        break;
    case NPY_INT16:
        value = (int64_t) * (int16_t *)pInput;
        fvalue = (double)value;
        break;
    case NPY_UINT16:
        value = (int64_t) * (uint16_t *)pInput;
        fvalue = (double)value;
        break;
    CASE_NPY_UINT32:
        value = (int64_t) * (uint32_t *)pInput;
        fvalue = (double)value;
        break;
    CASE_NPY_INT32:
        value = (int64_t) * (int32_t *)pInput;
        fvalue = (double)value;
        break;
    CASE_NPY_UINT64:

        value = (int64_t) * (uint64_t *)pInput;
        fvalue = (double)value;
        break;
    CASE_NPY_INT64:

        value = (int64_t) * (int64_t *)pInput;
        fvalue = (double)value;
        break;
    case NPY_FLOAT32:
        fvalue = (double)*(float *)pInput;
        value = (int64_t)fvalue;
        break;
    case NPY_FLOAT64:
        fvalue = (double)*(double *)pInput;
        value = (int64_t)fvalue;
        break;
    default:
        return false;
    }

    switch (numpyOutType)
    {
    case NPY_BOOL:
    case NPY_INT8:
    case NPY_UINT8:
        pDest->i = _mm256_set1_epi8((int8_t)value);
        break;
    case NPY_INT16:
    case NPY_UINT16:
        pDest->i = _mm256_set1_epi16((int16_t)value);
        break;
    CASE_NPY_UINT32:
    CASE_NPY_INT32:
        pDest->i = _mm256_set1_epi32((int32_t)value);
        break;
    CASE_NPY_UINT64:

    CASE_NPY_INT64:

        pDest->ci.i1 = _mm_set1_epi64x(value);
        pDest->ci.i2 = _mm_set1_epi64x(value);
        break;
    case NPY_FLOAT32:
        pDest->s = _mm256_set1_ps((float)fvalue);
        break;
    case NPY_FLOAT64:
        // printf("setting value to %lf\n", fvalue);
        pDest->d = _mm256_set1_pd((double)fvalue);
        break;
    default:
        printf("unknown scalar type in convertScalarObject %d\n", numpyOutType);
        return false;
        break;
    }

    return true;
}

//---------------------------------------------------------------------------
// Takes as input a scalar object that is a bool, float, or int
// Take as input the numpyOutType you want
// The output of pDest holds the value represented in a 256bit AVX2 register
//
// RETURNS:
// true on success
// *ppDataIn points to scalar object
// pItemSize set to 0 unless a string or unicode
//
// CONVERTS scalar inObject1 to --> numpyOutType filling in pDest
// If inObject1 is a string or unicode, then ppDataIn is filled in with the
// itemsize
//
// NOTE: Cannot handle numpy scalar types like numpy.int32
bool ConvertScalarObject(PyObject * inObject1, _m256all * pDest, int16_t numpyOutType, void ** ppDataIn, int64_t * pItemSize)
{
    *pItemSize = 0;
    *ppDataIn = pDest;

    bool isNumpyScalarInteger = PyArray_IsScalar((inObject1), Integer);
    bool isPyBool = PyBool_Check(inObject1);
    LOGGING("In convert scalar object!  %d %d %d\n", numpyOutType, isNumpyScalarInteger, isPyBool);

    if (isPyBool || PyArray_IsScalar((inObject1), Bool))
    {
        int64_t value = 1;
        if (isPyBool)
        {
            if (inObject1 == Py_False)
                value = 0;
        }
        else
        {
            // Must be a numpy scalar array type, pull the value (see
            // scalartypes.c.src)
            value = ((PyBoolScalarObject *)inObject1)->obval;
        }

        switch (numpyOutType)
        {
        case NPY_BOOL:
        case NPY_INT8:
        case NPY_UINT8:
            pDest->i = _mm256_set1_epi8((int8_t)value);
            break;
        case NPY_INT16:
        case NPY_UINT16:
            pDest->i = _mm256_set1_epi16((int16_t)value);
            break;
        CASE_NPY_UINT32:
        CASE_NPY_INT32:
            pDest->i = _mm256_set1_epi32((int32_t)value);
            break;
        CASE_NPY_UINT64:

        CASE_NPY_INT64:

            pDest->ci.i1 = _mm_set1_epi64x(value);
            pDest->ci.i2 = _mm_set1_epi64x(value);
            break;
        case NPY_FLOAT32:
            pDest->s = _mm256_set1_ps((float)value);
            break;
        case NPY_FLOAT64:
            pDest->d = _mm256_set1_pd((double)value);
            break;
        default:
            printf("unknown bool scalar type in convertScalarObject %d\n", numpyOutType);
            return false;
        }
    }
    else
    {
        if (PyLong_Check(inObject1) || isNumpyScalarInteger)
        {
            int overflow = 0;
            int64_t value;
            uint64_t value2;

            if (! isNumpyScalarInteger)
            {
                value = PyLong_AsLongLongAndOverflow(inObject1, &overflow);

                // overflow of 1 indicates past LONG_MAX
                // overflow of -1 indicate past LONG_MIN which we do not handle
                // PyLong_AsLongLong will RAISE an overflow exception

                // If the value is negative, conversion to unsigned not allowed
                if (value >= 0 || overflow == 1)
                {
                    value2 = PyLong_AsUnsignedLongLongMask(inObject1);
                }
                else
                {
                    value2 = (uint64_t)value;
                }
            }
            else
            {
                PyArray_Descr * dtype = PyArray_DescrFromScalar(inObject1);
                //// NOTE: memory leak here?
                if (dtype->type_num <= NPY_LONGDOUBLE)
                {
                    if (g_pDescrLongLong == NULL)
                    {
                        g_pDescrLongLong = PyArray_DescrNewFromType(NPY_LONGLONG);
                        g_pDescrULongLong = PyArray_DescrNewFromType(NPY_ULONGLONG);
                    }

                    PyArray_CastScalarToCtype(inObject1, &value, g_pDescrLongLong);
                    PyArray_CastScalarToCtype(inObject1, &value2, g_pDescrULongLong);
                }
                else
                {
                    // datetime64 falls into here
                    LOGGING("!!punting on scalar type is %d\n", dtype->type_num);
                    return false;
                }
            }

            switch (numpyOutType)
            {
            case NPY_BOOL:
            case NPY_INT8:
                pDest->i = _mm256_set1_epi8((int8_t)value);
                break;
            case NPY_UINT8:
                pDest->i = _mm256_set1_epi8((uint8_t)value2);
                break;
            case NPY_INT16:
                pDest->i = _mm256_set1_epi16((int16_t)value);
                break;
            case NPY_UINT16:
                pDest->i = _mm256_set1_epi16((uint16_t)value2);
                break;
            CASE_NPY_INT32:
                pDest->i = _mm256_set1_epi32((int32_t)value);
                break;
            CASE_NPY_UINT32:
                pDest->i = _mm256_set1_epi32((uint32_t)value2);
                break;
            CASE_NPY_INT64:

                pDest->ci.i1 = _mm_set1_epi64x(value);
                pDest->ci.i2 = _mm_set1_epi64x(value);
                break;
            CASE_NPY_UINT64:

                pDest->ci.i1 = _mm_set1_epi64x(value2);
                pDest->ci.i2 = _mm_set1_epi64x(value2);
                break;
            case NPY_FLOAT32:
                pDest->s = _mm256_set1_ps((float)value);
                break;
            case NPY_FLOAT64:
                pDest->d = _mm256_set1_pd((double)value);
                break;
            default:
                printf("unknown long scalar type in convertScalarObject %d\n", numpyOutType);
                return false;
            }
        }
        else if (PyFloat_Check(inObject1) || PyArray_IsScalar((inObject1), Floating))
        {
            double value = PyFloat_AsDouble(inObject1);

            switch (numpyOutType)
            {
            case NPY_BOOL:
            case NPY_INT8:
                pDest->i = _mm256_set1_epi8((int8_t)value);
                break;
            case NPY_UINT8:
                pDest->i = _mm256_set1_epi8((uint8_t)value);
                break;
            case NPY_INT16:
                pDest->i = _mm256_set1_epi16((int16_t)value);
                break;
            case NPY_UINT16:
                pDest->i = _mm256_set1_epi16((uint16_t)value);
                break;
            CASE_NPY_UINT32:
                pDest->i = _mm256_set1_epi32((uint32_t)value);
                break;
            CASE_NPY_INT32:
                pDest->i = _mm256_set1_epi32((int32_t)value);
                break;
            CASE_NPY_UINT64:

                pDest->ci.i1 = _mm_set1_epi64x((uint64_t)value);
                pDest->ci.i2 = _mm_set1_epi64x((uint64_t)value);
                break;
            CASE_NPY_INT64:

                pDest->ci.i1 = _mm_set1_epi64x((int64_t)value);
                pDest->ci.i2 = _mm_set1_epi64x((int64_t)value);
                break;
            case NPY_FLOAT32:
                pDest->s = _mm256_set1_ps((float)value);
                break;
            case NPY_FLOAT64:
                pDest->d = _mm256_set1_pd((double)value);
                break;
            case NPY_LONGDOUBLE:
                pDest->d = _mm256_set1_pd((double)(long double)value);
                break;
            default:
                printf("unknown float scalar type in convertScalarObject %d\n", numpyOutType);
                return false;
            }
        }
        else if (PyBytes_Check(inObject1))
        {
            // happens when pass in b'test'
            *pItemSize = PyBytes_GET_SIZE(inObject1);
            *ppDataIn = PyBytes_AS_STRING(inObject1);
            return true;
        }
        else if (PyUnicode_Check(inObject1))
        {
            // happens when pass in 'test'
            if (PyUnicode_READY(inObject1) < 0)
            {
                printf("!!unable to make UNICODE object ready");
                return false;
            }
            *pItemSize = PyUnicode_GET_LENGTH(inObject1) * 4;
            // memory leak needs to be deleted
            *ppDataIn = PyUnicode_AsUCS4Copy(inObject1);
            return true;
        }
        else if (PyArray_IsScalar(inObject1, Generic))
        {
            // only integers are not subclassed in numpy world
            if (PyArray_IsScalar((inObject1), Integer))
            {
                PyArray_Descr * dtype = PyArray_DescrFromScalar(inObject1);

                // NOTE: memory leak here?
                printf("!!integer scalar type is %d\n", dtype->type_num);
                return false;
            }
            else
            {
                printf("!!unknown numpy scalar type in convertScalarObject %d --", numpyOutType);
                return false;
            }
        }

        else
        {
            // Complex types hits here
            LOGGING("!!unknown scalar type in convertScalarObject %d --", numpyOutType);
            PyTypeObject * type = inObject1->ob_type;
            LOGGING("type name is %s\n", type->tp_name);
            return false;
        }
    }
    // printf("returning from scalar conversion\n");
    return true;
}

// Turn threading on or off
static PyObject * ThreadingMode(PyObject * self, PyObject * args)
{
    int64_t threadmode;

    if (! PyArg_ParseTuple(args, "L", &threadmode))
        return NULL;

    bool previous = g_cMathWorker->NoThreading;
    g_cMathWorker->NoThreading = (bool)threadmode;

    return PyLong_FromLong((int)previous);
}

static PyObject * TestNumpy(PyObject * self, PyObject * args)
{
    PyObject * scalarObject = NULL;

    if (! PyArg_ParseTuple(args, "O", &scalarObject))
        return NULL;
    PyTypeObject * type = scalarObject->ob_type;

    printf("type name is %s\n", type->tp_name);
    printf("ref cnt is %zu\n", scalarObject->ob_refcnt);

    if (PyFloat_Check(scalarObject))
    {
        double val = PyFloat_AsDouble(scalarObject);
        printf("float %lf\n", val);
        // PyObject_Type
    }

    if (PyByteArray_Check(scalarObject))
    {
        printf("byte array\n");
    }

    if (PyBytes_Check(scalarObject))
    {
        // happens when pass in b'test'
        printf("bytes\n");
    }

    if (PyUnicode_Check(scalarObject))
    {
        // happens when pass in 'test'
        printf("unicode\n");
    }

    if (PyLong_Check(scalarObject))
    {
        // False/True will come as 0 and 1
        int overflow = 0;
        int64_t val = PyLong_AsLongLongAndOverflow(scalarObject, &overflow);
        printf("long %lld\n", val);
    }

    if (PyBool_Check(scalarObject))
    {
        if (scalarObject == Py_False)
        {
            printf("false");
        }
        printf("bool\n");
    }

    if (PyArray_Check(scalarObject))
    {
        PyArrayObject * arr = (PyArrayObject *)scalarObject;

        printf("array itemsize=%lld, strides=%lld, flags=%d, length=%lld:\n", (int64_t)PyArray_ITEMSIZE(arr),
               (int64_t)(*PyArray_STRIDES(arr)), PyArray_FLAGS(arr), ArrayLength(arr));

        PyObject * pBase = PyArray_BASE((PyArrayObject *)scalarObject);

        printf("object and base  %p vs %p\n", scalarObject, pBase);

        if (pBase)
        {
            printf("refcnt object and base  %zu vs %zu\n", scalarObject->ob_refcnt, pBase->ob_refcnt);

            PyObject * pBaseBase = PyArray_BASE((PyArrayObject *)pBase);

            if (pBaseBase)
            {
                printf("base and basebase  %p vs %p\n", pBase, pBaseBase);
                printf(" %zu vs %zu\n", pBase->ob_refcnt, pBaseBase->ob_refcnt);
            }
        }

        // Py_buffer pb;
    }

    if (PyArray_IsAnyScalar(scalarObject))
    {
        printf("any scalar\n");
    }

    // NOTE this test does not check list,tuples, and other types
    // void PyArray_ScalarAsCtype(PyObject* scalar, void* ctypeptr)
    // Return in ctypeptr a pointer to the actual value in an array scalar.
    // There is no error checking so scalar must be an array - scalar object, and
    // ctypeptr must have enough space to hold the correct type. For flexible -
    // sized types, a pointer to the data is copied into the memory of ctypeptr,
    // for all other types, the actual data is copied into the address pointed to
    // by ctypeptr.

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * CalculateCRC(PyObject * self, PyObject * args);

#if defined(_WIN32)

// global scope
typedef VOID(WINAPI * FuncGetSystemTime)(LPFILETIME);
FuncGetSystemTime g_GetSystemTime;
FILETIME g_TimeStart;
static bool g_IsPreciseTime = false;

//------------------------------------
// Returns windows time in Nanos
__inline static uint64_t GetWindowsTime()
{
    FILETIME timeNow;
    g_GetSystemTime(&timeNow);
    return (*(uint64_t *)&timeNow * 100) - 11644473600000000000L;
}

//-------------------------------------------------------------------
//
class CTimeStamp
{
public:
    CTimeStamp()
    {
        FARPROC fp;

        g_GetSystemTime = GetSystemTimeAsFileTime;

        HMODULE hModule = LoadLibraryW(L"kernel32.dll");

        // Use printf instead of logging because logging is probably not up yet
        // Logging uses the timestamping, so timestamping loads first
        if (hModule != NULL)
        {
            fp = GetProcAddress(hModule, "GetSystemTimePreciseAsFileTime");
            if (fp != NULL)
            {
                g_IsPreciseTime = true;
                // printf("Using precise GetSystemTimePreciseAsFileTime time...\n");
                g_GetSystemTime = (VOID(WINAPI *)(LPFILETIME))fp;
            }
            else
            {
                LOGGING("**Using imprecise GetSystemTimeAsFileTime...\n");
            }
        }
        else
        {
            printf("!! error load kernel32\n");
        }
    }
};

static CTimeStamp * g_TimeStamp = new CTimeStamp();

//---------------------------------------------------------
// Returns and int64_t nanosecs since unix epoch
static PyObject * GetNanoTime(PyObject * self, PyObject * args)
{
    // return nano time since Unix Epoch
    return PyLong_FromLongLong((long long)GetWindowsTime());
}

//---------------------------------------------------------
// Returns and uint64_t timestamp counter
static PyObject * GetTSC(PyObject * self, PyObject * args)
{
    // return tsc
    return PyLong_FromUnsignedLongLong(__rdtsc());
}

#else

    #include <sys/time.h>
    #include <time.h>
    #include <unistd.h>

uint64_t GetTimeStamp()
{
    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    // return tv.tv_sec*(uint64_t)1000000 + tv.tv_usec;

    struct timespec x;
    clock_gettime(CLOCK_REALTIME, &x);
    return x.tv_sec * 1000000000L + x.tv_nsec;
}

static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

//---------------------------------------------------------
// Returns and uint64_t timestamp counter
static PyObject * GetTSC(PyObject * self, PyObject * args)
{
    // return tsc
    return PyLong_FromUnsignedLongLong(rdtsc());
}

//---------------------------------------------------------
// Returns and int64_t nanosecs since unix epoch
static PyObject * GetNanoTime(PyObject * self, PyObject * args)
{
    // return nano time since Unix Epoch
    return PyLong_FromLongLong(GetTimeStamp());
    // return Py_BuildValue("L", GetNanoTimeExtraPrecise());
}

#endif

//---------------------------------------------------------
// Returns nanoseconds since utc epoch
uint64_t GetUTCNanos()
{
#if defined(_WIN32)
    return GetWindowsTime();
#else
    return GetTimeStamp();
#endif
}
//---------------------------------------------------------
// Returns curent thread wakeup setting (does not change it)
static PyObject * GetThreadWakeUp(PyObject * self, PyObject * args)
{
    // return current number of threads that wake up to do work
    return PyLong_FromLongLong((long long)g_cMathWorker->GetFutexWakeup());
}

//---------------------------------------------------------
// Returns curent thread wakeup setting to existing
static PyObject * SetThreadWakeUp(PyObject * self, PyObject * args)
{
    int64_t newWakeup = 1;
    if (! PyArg_ParseTuple(args, "L", &newWakeup))
    {
        return NULL;
    }
    newWakeup = g_cMathWorker->SetFutexWakeup((int)newWakeup);

    // return current number of threads that wake up to do work
    return PyLong_FromLongLong((long long)newWakeup);
}

PyObject * RecordArrayToColMajor(PyObject * self, PyObject * args);

//--------------------------------------------------------
const char * docstring_asarray =
    "Parameters\n"
    "----------\n"
    "a : array_like\n"
    "   Input data, in any form that can be converted to an array.This\n"
    "   includes lists, lists of tuples, tuples, tuples of tuples, tuples\n"
    "   of lists and ndarrays.\n"
    "dtype : data - type, optional\n"
    "   By default, the data - type is inferred from the input data.\n"
    "order : {'C', 'F'}, optional\n"
    "   Whether to use row - major(C - style) or\n"
    "   column - major(Fortran - style) memory representation.\n"
    "   Defaults to 'C'.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "out : ndarray or FastArray\n"
    "   Array interpretation of 'a'.  No copy is performed if the input\n"
    "   is already an ndarray or FastArray with matching dtype and order.\n"
    "   If 'a' is a subclass of ndarray, a base class ndarray is returned.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "asfastarray : Similar function which always returns a FastArray.\n";

const char * docstring_asfastarray =
    "Parameters\n"
    "----------\n"
    "a : array_like\n"
    "   Input data, in any form that can be converted to an array.This\n"
    "   includes lists, lists of tuples, tuples, tuples of tuples, tuples\n"
    "   of lists and ndarrays.\n"
    "dtype : data - type, optional\n"
    "   By default, the data - type is inferred from the input data.\n"
    "order : {'C', 'F'}, optional\n"
    "   Whether to use row - major(C - style) or\n"
    "   column - major(Fortran - style) memory representation.\n"
    "   Defaults to 'C'.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "out : FastArray\n"
    "   Array interpretation of 'a'.  No copy is performed if the input\n"
    "   is already an ndarray or FastArray with matching dtype and order.\n"
    "   If 'a' is a subclass of ndarray, a base class ndarray is returned.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "asarray : Similar function.\n";

/* ==== Set up the methods table ====================== */
// struct PyMethodDef {
//	const char  *ml_name;   /* The name of the built-in function/method */
//	PyCFunction ml_meth;    /* The C function that implements it */
//	int         ml_flags;   /* Combination of METH_xxx flags, which mostly
//							describe the args expected by the C
// func */ 	const char  *ml_doc;    /* The __doc__ attribute, or NULL */
//};

static PyMethodDef CSigMathUtilMethods[] = {
    { "AllocateNumpy", AllocateNumpy, METH_VARARGS, "AllocateNumpy wrapper" },
    { "RecycleNumpy", RecycleNumpy, METH_VARARGS, "RecycleNumpy wrapper" },
    { "RecycleGarbageCollectNow", RecycleGarbageCollectNow, METH_VARARGS,
      "RecycleGarbageCollect wrapper.  arg1 is long integer timespan" },
    { "RecycleSetGarbageCollectTimeout", RecycleSetGarbageCollectTimeout, METH_VARARGS,
      "RecycleSetGarbageCollectTimeout wrapper.  arg1 is long integer timespan" },
    { "RecycleDump", RecycleDump, METH_VARARGS, "Dump all objects in reclaimed memory" },
    { "SetRecycleMode", SetRecycleMode, METH_VARARGS, "Change recycling mode 0=on  1=off" },
    { "Empty", Empty, METH_VARARGS, "similar to np.empty, but tries to allocate from recycler" },

    { "TryRecycleNumpy", TryRecycleNumpy, METH_VARARGS, "Try to allocate from recycled array" },
    { "TestNumpy", TestNumpy, METH_VARARGS, "TestNumpy wrapper" },
    { "CompareNumpyMemAddress", CompareNumpyMemAddress, METH_VARARGS,
      "Compare two numpy arrays to see if same underlying memory address" },

    { "CompressString", CompressString, METH_VARARGS, "Compress a string" },
    { "DecompressString", DecompressString, METH_VARARGS, "Decompress a string" },
    { "CompressDecompressArrays", CompressDecompressArrays, METH_VARARGS, "CompressDecompressArrays one or more arrays" },
    { "CompressFile", (PyCFunction)CompressFile, METH_VARARGS | METH_KEYWORDS, "CompressFile one or more arrays" },
    { "DecompressFile", (PyCFunction)DecompressFile, METH_VARARGS | METH_KEYWORDS, "DecompressFile one or more arrays" },
    { "MultiStackFiles", (PyCFunction)MultiStackFiles, METH_VARARGS | METH_KEYWORDS,
      "MultiStackFiles one or more files with auto hstack .. arrays" },
    { "MultiDecompressFiles", (PyCFunction)MultiDecompressFiles, METH_VARARGS | METH_KEYWORDS,
      "MultiDecompressFile one or more files (returns tuple)" },
    { "MultiPossiblyStackFiles", (PyCFunction)MultiPossiblyStackFiles, METH_VARARGS | METH_KEYWORDS,
      "MultiPossiblyStackFiles one or more files (returns tuple)" },
    { "MultiConcatFiles", (PyCFunction)MultiConcatFiles, METH_VARARGS | METH_KEYWORDS,
      "MultiConcatFiles one or more files with output=filename" },
    { "SetLustreGateway", SetLustreGateway, METH_VARARGS, "SetLustreGateway - give alternate samba servers" },

    { "BitCount", BitCount, METH_VARARGS, "BitCount calculation" },
    { "IsMember32", IsMember32, METH_VARARGS, "IsMember32 calculation" },
    { "IsMember64", IsMember64, METH_VARARGS, "IsMember64 calculation" },
    { "IsMemberCategorical", IsMemberCategorical, METH_VARARGS, "IsMemberCategorical calculation" },
    { "IsMemberCategoricalFixup", IsMemberCategoricalFixup, METH_VARARGS,
      "IsMemberCategoricalFixup used for ismember unique on cat" },

    { "MultiKeyHash", MultiKeyHash, METH_VARARGS, "MultiKeyHash calculation" },
    { "MultiKeyGroupBy32", (PyCFunction)MultiKeyGroupBy32, METH_VARARGS | METH_KEYWORDS, "MultiKeyGroupBy32 calculation" },
    { "MultiKeyGroupBy32Super", MultiKeyGroupBy32Super, METH_VARARGS, "MultiKeyGroupBy32Super calculation" },
    { "MultiKeyUnique32", MultiKeyUnique32, METH_VARARGS, "MultiKeyUnique32 calculation" },
    { "MultiKeyIsMember32", MultiKeyIsMember32, METH_VARARGS, "MultiKeyIsMember32 calculation" },
    { "MultiKeyAlign32", MultiKeyAlign32, METH_VARARGS, "MultiKeyAlign32 calculation" },
    { "MultiKeyRolling", MultiKeyRolling, METH_VARARGS, "MultiKeyRolling calculation" },
    { "BinCount", (PyCFunction)BinCount, METH_VARARGS | METH_KEYWORDS,
      "BinCount calculation, may also return igroup and ifirstgroup" },
    { "MakeiNext", MakeiNext, METH_VARARGS,
      "MakeiNext calculation: arg1: index, arg2: uniquecount, arg3: mode 0 or "
      "1" },
    { "GroupFromBinCount", GroupFromBinCount, METH_VARARGS,
      "GroupFromBinCount calculation: arg1: index, arg2: result from BinCount" },

    { "MergeBinnedAndSorted", MergeBinnedAndSorted, METH_VARARGS, "MergeBinnedAndSorted calculation" },

    { "GroupByPack32", GroupByPack32, METH_VARARGS, "GroupByPack32 data from int to float, etc" },
    //{ "GroupByOp32", GroupByOp32, METH_VARARGS, "GroupByOp32 data from int to
    // float, etc" },
    { "GroupByAll32", GroupByAll32, METH_VARARGS, "GroupByAll32 data from int to float, etc" },
    { "GroupByAll64", GroupByAll64, METH_VARARGS, "GroupByAll64 data from int to float, etc" },
    { "GroupByAllPack32", GroupByAllPack32, METH_VARARGS, "GroupByAllPack32 data from int to float, etc" },

    { "EmaAll32", EmaAll32, METH_VARARGS, "EmaAll32 summation" },
    { "Rolling", Rolling, METH_VARARGS, "Rolling window summation" },
    { "TimeWindow", TimeWindow, METH_VARARGS, "Time window summation/prod" },
    { "InterpExtrap2d", InterpExtrap2d, METH_VARARGS, "Interpolation routine see: np.interp" },

    { "ThreadingMode", ThreadingMode, METH_VARARGS, "Change Threading Mode" },

    { "ConvertSafe", ConvertSafe, METH_VARARGS, "Convert data from int to float, etc, preserving invalids" },
    { "ConvertUnsafe", ConvertUnsafe, METH_VARARGS, "Convert data from int to float, NOT preserving invalids" },
    { "CombineFilter", CombineFilter, METH_VARARGS, "Combine an index and a bool filter" },
    { "CombineAccum2Filter", CombineAccum2Filter, METH_VARARGS, "Combine an index and a bool filter" },
    { "CombineAccum1Filter", CombineAccum1Filter, METH_VARARGS, "Combine an index and a bool filter" },
    { "MakeiFirst", MakeiFirst, METH_VARARGS, "Combine an index and a bool filter, returns iFirst" },

    { "HStack", HStack, METH_VARARGS, "HStack - concats arrays" },
    { "SetItem", SetItem, METH_VARARGS, "takes over numpy array __setitem__" },
    { "PutMask", PutMask, METH_VARARGS, "takes over numpy putmask" },
    { "GetUpcastNum", GetUpcastNum, METH_VARARGS, "GetUpcastNum - pass in list of numpy arrays, get dtype num" },
    { "HomogenizeArrays", HomogenizeArrays, METH_VARARGS, "HomogenizeArrays - pass in list of numpy arrays, get dtype num" },
    { "ShiftArrays", ShiftArrays, METH_VARARGS, "ShiftArrays - pass in list of numpy arrays, shift amoount" },
    { "ApplyRows", (PyCFunction)LedgerFunction, METH_VARARGS | METH_KEYWORDS,
      "ApplyRows calculation, pass in list of arrays, dtype, func" },

    { "MBGet", MBGet, METH_VARARGS, "MBGet fancy index getitem functionality" },
    { "BooleanIndex", BooleanIndex, METH_VARARGS, "BooleanIndex functionality" },
    { "BooleanSum", BooleanSum, METH_VARARGS, "BooleanSum functionality" },
    { "BooleanToFancy", (PyCFunction)BooleanToFancy, METH_VARARGS | METH_KEYWORDS, "BooleanToFancy functionality" },
    { "ReIndexGroups", ReIndexGroups, METH_VARARGS, "reindex categoricals from multistack" },
    { "ReverseShuffle", ReverseShuffle, METH_VARARGS, "ReverseShuffle for indexes algo: out[in[i]] = i" },

    { "Reduce", Reduce, METH_VARARGS, "Reduce add,min,max etc" },

    { "AsAnyArray", (PyCFunction)AsAnyArray, METH_VARARGS | METH_KEYWORDS, docstring_asarray },
    { "AsFastArray", (PyCFunction)AsFastArray, METH_VARARGS | METH_KEYWORDS, docstring_asfastarray },

    { "BasicMathUnaryOp", (PyCFunction)BasicMathUnaryOp, METH_VARARGS | METH_KEYWORDS, "BasicMathUnaryOp functionality" },
    { "BasicMathOneInput", BasicMathOneInput, METH_VARARGS, "BasicMathOneInput functionality" },
    { "BasicMathTwoInputs", BasicMathTwoInputs, METH_VARARGS, "BasicMathTwoInputs functionality" },

    // for low level python hooks
    { "BasicMathHook", BasicMathHook, METH_VARARGS, "BasicMathHook functionality (pass in fastarray class, FA, np.ndarray)" },

    { "LedgerFunction", (PyCFunction)LedgerFunction, METH_VARARGS | METH_KEYWORDS, "LedgerFunction calculation" },
    { "SetFastArrayType", SetFastArrayType, METH_VARARGS, "SetFastArrayType" },

    { "Sort", Sort, METH_VARARGS, "return sorted array, second argument 1=QuickSort, 2=Merge, 3=Heap" },
    { "SortInPlace", SortInPlace, METH_VARARGS, "SortInPlace second argument 1=QuickSort, 2=Merge, 3=Heap" },
    { "SortInPlaceIndirect", SortInPlaceIndirect, METH_VARARGS, "SortInPlaceIndirect" },
    { "ReIndex", ReIndex, METH_VARARGS, "returns data rearranged by index" },

    { "RemoveTrailingSpaces", RemoveTrailingSpaces, METH_VARARGS, "in place removal of trailing spaces (matlab)" },

    { "LexSort64", (PyCFunction)LexSort64, METH_VARARGS | METH_KEYWORDS, "LexSort64 returns int64 indexing" },
    { "LexSort32", (PyCFunction)LexSort32, METH_VARARGS | METH_KEYWORDS, "LexSort32 returns int32 indexing" },
    { "GroupFromLexSort", (PyCFunction)GroupFromLexSort, METH_VARARGS | METH_KEYWORDS,
      "GroupFromLexSort can input int32 or int64 indexing" },

    { "IsSorted", IsSorted, METH_VARARGS, "IsSorted" },

    { "Where", Where, METH_VARARGS, "Where version of np.where" },

    { "RecordArrayToColMajor", RecordArrayToColMajor, METH_VARARGS, "Convert record arrays to col major" },

    { "NanInfCountFromSort", NanInfCountFromSort, METH_VARARGS, "NanInfCountFromSort" },
    { "BinsToCutsBSearch", BinsToCutsBSearch, METH_VARARGS, "BinsToCutsBSearch" },
    { "BinsToCutsSorted", BinsToCutsSorted, METH_VARARGS, "BinsToCutsSorted" },

    { "CalculateCRC", CalculateCRC, METH_VARARGS, "CalculateCRC" },
    { "GetNanoTime", GetNanoTime, METH_NOARGS, "Get int64 nano time since unix epoch" },
    { "GetTSC", GetTSC, METH_NOARGS, "Get int64 time stamp counter from CPU" },
    { "GetThreadWakeUp", GetThreadWakeUp, METH_NOARGS, "Get how many threads wake up to do work." },
    { "SetThreadWakeUp", SetThreadWakeUp, METH_VARARGS, "Set how many threads to wake up.  Return the previous value" },

    { "TimeStringToNanos", TimeStringToNanos, METH_VARARGS,
      "Parse string in HH:MM:SS or HH:MM:SS.mmm format to nanos from midnight" },
    { "DateStringToNanos", DateStringToNanos, METH_VARARGS, "Parse string in YYYYMMDD or YYYY-MM-DD format to UTC epoch nanos" },
    { "DateTimeStringToNanos", DateTimeStringToNanos, METH_VARARGS,
      "Parse string in YYYYMMDD or YYYY-MM-DD format  then HH:MM:SS.mmm to UTC "
      "epoch nanos" },
    { "StrptimeToNanos", StrptimeToNanos, METH_VARARGS, "Parse string in strptime  then .mmm to UTC epoch nanos" },
    //{ "addf32x", addf32x, METH_VARARGS, "addf32 with output calculation" },
    { NULL, NULL, 0, NULL } /* Sentinel - marks the end of this structure */
};

static PyModuleDef CSigMathUtilModule = {
    PyModuleDef_HEAD_INIT,
    "riptide_cpp",                       // Module name
    "Provides functions for math utils", // Module description
    0,
    CSigMathUtilMethods, // Structure that defines the methods
    NULL,                // slots
    NULL,                // GC traverse
    NULL,                // GC
    NULL                 // freefunc
};

// For Python version 3,  PyInit_{module_name} must be used as this function is
// called when the module is imported
PyMODINIT_FUNC PyInit_riptide_cpp()
{
    int32_t count = 0;

    // Count up the
    for (int i = 0; i < 1000; i++)
    {
        if (CSigMathUtilMethods[i].ml_name == NULL)
        {
            break;
        }

        count++;
    }

    LOGGING("FASTMATH: Found %d methods\n", count);

    InitRecycler();

    // allocate plus one because last one is sentinel fulled with nulls
    int64_t allocSize = sizeof(PyMethodDef) * (count + 1);

    // Allocate a new array
    PyMethodDef * pNewMethods = (PyMethodDef *)PYTHON_ALLOC(allocSize);
    memset(pNewMethods, 0, allocSize);
    PyMethodDef * pDest = pNewMethods;

    // Add in the default first
    for (int i = 0; i < count; i++)
    {
        *pDest++ = CSigMathUtilMethods[i];
    }

    // Point to our new list
    CSigMathUtilModule.m_methods = pNewMethods;

    // Build this list on the fly now that we have the table
    PyObject * m = PyModule_Create(&CSigMathUtilModule);

    if (m == NULL)
        return m;

    g_FastArrayModule = m;

    // Load numpy
    import_array();

    // Build conversion
    for (int i = 0; i < NUMPY_LAST_TYPE; i++)
    {
        NpTypeObjects[i].type = (PyTypeObject *)PyArray_TypeObjectFromType(NpTypeObjects[i].typenum);
    }

    // Build LUTs used in comparisons after mask generated
    for (int i = 0; i < 256; i++)
    {
        unsigned char * pDest = (unsigned char *)&gBooleanLUT64[i];
        for (int j = 0; j < 8; j++)
        {
            *pDest++ = ((i >> j) & 1);
        }
    }
    // Build LUTs
    for (int i = 0; i < 16; i++)
    {
        unsigned char * pDest = (unsigned char *)&gBooleanLUT32[i];
        for (int j = 0; j < 4; j++)
        {
            *pDest++ = ((i >> j) & 1);
        }
    }

    // Build LUTs
    for (int i = 0; i < 256; i++)
    {
        gBooleanLUT64Inverse[i] = gBooleanLUT64[i] ^ 0x0101010101010101LL;
    }
    // Build LUTs
    for (int i = 0; i < 16; i++)
    {
        gBooleanLUT32Inverse[i] = gBooleanLUT32[i] ^ 0x01010101;
    }

    // Build upcast table (14 * 14)
    for (int convertType1 = 0; convertType1 < 14; convertType1++)
    {
        for (int convertType2 = 0; convertType2 < 14; convertType2++)
        {
            stUpCast * pRow = &gUpcastTable[convertType1 * 14 + convertType2];

            if (convertType1 == 0)
            {
                // bool converts to anything
                pRow->dtype1 = convertType2;
                pRow->dtype2 = convertType2;
            }
            else if (convertType2 == 0)
            {
                // bool converts to anything
                pRow->dtype1 = convertType1;
                pRow->dtype2 = convertType1;
            }
            else
                // Check for long upcasting?
                if (convertType1 > convertType2)
            {
                if (convertType1 == NPY_ULONGLONG)
                {
                    // check for signed value
                    if ((convertType2 & 1) == 1)
                    {
                        pRow->dtype1 = NPY_FLOAT64;
                        pRow->dtype2 = NPY_FLOAT64;
                    }
                    else
                    {
                        pRow->dtype1 = NPY_ULONGLONG;
                        pRow->dtype2 = NPY_ULONGLONG;
                    }
                }
                else
                {
                    // if the higher is unsigned and the other is signed go up one
                    if (convertType1 < NPY_ULONGLONG && (convertType1 & 1) == 0 && (convertType2 & 1) == 1)
                    {
                        // Choose the higher dtype +1
                        pRow->dtype1 = convertType1 + 1;
                        pRow->dtype2 = convertType1 + 1;

                        // Handle ambiguous dtype upcast (going from int to long does
                        // nothing on some C compilers)
                        if (sizeof(long) == 4)
                        {
                            if (convertType1 == NPY_INT || convertType1 == NPY_UINT)
                            {
                                pRow->dtype1 = convertType1 + 3;
                                pRow->dtype2 = convertType1 + 3;
                            }
                        }
                        else
                        {
                            if (convertType1 == NPY_LONG || convertType1 == NPY_ULONG)
                            {
                                pRow->dtype1 = convertType1 + 3;
                                pRow->dtype2 = convertType1 + 3;
                            }
                        }
                    }
                    else
                    {
                        // Choose the higher dtype
                        pRow->dtype1 = convertType1;
                        pRow->dtype2 = convertType1;
                    }
                }
            }
            else
            {
                if (convertType1 == convertType2)
                {
                    pRow->dtype1 = convertType2;
                    pRow->dtype2 = convertType2;
                }
                else
                {
                    // convertType2 is larger
                    if (convertType2 == NPY_ULONGLONG)
                    {
                        // check for signed value
                        if ((convertType1 & 1) == 1)
                        {
                            pRow->dtype1 = NPY_FLOAT64;
                            pRow->dtype2 = NPY_FLOAT64;
                        }
                        else
                        {
                            pRow->dtype2 = NPY_ULONGLONG;
                            pRow->dtype1 = NPY_ULONGLONG;
                        }
                    }
                    else
                    {
                        // Check for signed/unsigned integer
                        if (convertType2 < NPY_ULONGLONG && (convertType2 & 1) == 0 && (convertType1 & 1) == 1)
                        {
                            // Choose the higher dtype +1
                            pRow->dtype1 = convertType2 + 1;
                            pRow->dtype2 = convertType2 + 1;

                            // Handle ambiguous dtype upcast
                            if (sizeof(long) == 4)
                            {
                                if (convertType2 == NPY_INT || convertType2 == NPY_UINT)
                                {
                                    pRow->dtype1 = convertType2 + 3;
                                    pRow->dtype2 = convertType2 + 3;
                                }
                            }
                            else
                            {
                                if (convertType2 == NPY_LONG || convertType2 == NPY_ULONG)
                                {
                                    pRow->dtype1 = convertType2 + 3;
                                    pRow->dtype2 = convertType2 + 3;
                                }
                            }
                        }
                        else
                        {
                            // Choose the higher dtype
                            pRow->dtype1 = convertType2;
                            pRow->dtype2 = convertType2;
                        }
                    }
                }
            }
        }
    }

    // Comparisons should be able to handle int64 to uint64 specially
    memcpy(&gUpcastTableComparison, &gUpcastTable, sizeof(gUpcastTable));
    stUpCast * pRow;
    if (sizeof(long) == 8)
    {
        pRow = &gUpcastTableComparison[NPY_LONG * 14 + NPY_ULONG];
        pRow->dtype1 = NPY_LONG;
        pRow->dtype2 = NPY_ULONG;
        pRow = &gUpcastTableComparison[NPY_ULONG * 14 + NPY_LONG];
        pRow->dtype1 = NPY_ULONG;
        pRow->dtype2 = NPY_LONG;
    }

    pRow = &gUpcastTableComparison[NPY_LONGLONG * 14 + NPY_ULONGLONG];
    pRow->dtype1 = NPY_LONGLONG;
    pRow->dtype2 = NPY_ULONGLONG;
    pRow = &gUpcastTableComparison[NPY_ULONGLONG * 14 + NPY_LONGLONG];
    pRow->dtype1 = NPY_ULONGLONG;
    pRow->dtype2 = NPY_LONGLONG;

    // Register types defined by this module.
    PyObject * mod_dict = PyModule_GetDict(m);
    if (! mod_dict)
    {
        LOGGING("Unable to get the module dictionary for the riptide_cpp module.\n")
        return NULL;
    }

    if (! RegisterSdsPythonTypes(mod_dict))
    {
        LOGGING("An error occurred when creating/registering SDS Python types.");
        return NULL;
    }

    // start up the worker threads now in case we use them
    g_cMathWorker->StartWorkerThreads(0);

    LOGGING("riptide_cpp loaded\n");
    return m;
}

//-------------------------------------------------------------------------
// int64_t default1 = -9223372036854775808L;
static int64_t gDefaultInt64 = 0x8000000000000000;
static int32_t gDefaultInt32 = 0x80000000;
static uint16_t gDefaultInt16 = 0x8000;
static uint8_t gDefaultInt8 = 0x80;

static uint64_t gDefaultUInt64 = 0xFFFFFFFFFFFFFFFF;
static uint32_t gDefaultUInt32 = 0xFFFFFFFF;
static uint16_t gDefaultUInt16 = 0xFFFF;
static uint8_t gDefaultUInt8 = 0xFF;

static float gDefaultFloat = NAN;
static double gDefaultDouble = NAN;
static int8_t gDefaultBool = 0;
static char gString[1024] = { 0, 0, 0, 0 };

//----------------------------------------------------
// returns pointer to a data type (of same size in memory) that holds the
// invalid value for the type does not yet handle strings
void * GetDefaultForType(int numpyInType)
{
    void * pgDefault = &gDefaultInt64;

    switch (numpyInType)
    {
    case NPY_FLOAT:
        pgDefault = &gDefaultFloat;
        break;
    case NPY_LONGDOUBLE:
    case NPY_DOUBLE:
        pgDefault = &gDefaultDouble;
        break;
        // BOOL should not really have an invalid value inhabiting the type
    case NPY_BOOL:
        pgDefault = &gDefaultBool;
        break;
    case NPY_BYTE:
        pgDefault = &gDefaultInt8;
        break;
    case NPY_INT16:
        pgDefault = &gDefaultInt16;
        break;
    CASE_NPY_INT32:
        //   case NPY_INT: This is the same numeric value as NPY_INT32 above
        pgDefault = &gDefaultInt32;
        break;
    CASE_NPY_INT64:

        pgDefault = &gDefaultInt64;
        break;
    case NPY_UINT8:
        pgDefault = &gDefaultUInt8;
        break;
    case NPY_UINT16:
        pgDefault = &gDefaultUInt16;
        break;
    CASE_NPY_UINT32:
        //   case NPY_UINT: This is the same numeric value as NPY_UINT32 above
        pgDefault = &gDefaultUInt32;
        break;
    CASE_NPY_UINT64:

        pgDefault = &gDefaultUInt64;
        break;
    case NPY_STRING:
        pgDefault = &gString;
        break;
    case NPY_UNICODE:
        pgDefault = &gString;
        break;
    default:
        printf("!!! likely problem in GetDefaultForType\n");
    }

    return pgDefault;
}

int GetNumpyType(bool value)
{
    return NPY_BOOL;
}
int GetNumpyType(int8_t value)
{
    return NPY_INT8;
}
int GetNumpyType(int16_t value)
{
    return NPY_INT16;
}
int GetNumpyType(int32_t value)
{
    return NPY_INT32;
}
int GetNumpyType(int64_t value)
{
    return NPY_INT64;
}
int GetNumpyType(uint8_t value)
{
    return NPY_UINT8;
}
int GetNumpyType(uint16_t value)
{
    return NPY_UINT16;
}
int GetNumpyType(uint32_t value)
{
    return NPY_UINT32;
}
int GetNumpyType(uint64_t value)
{
    return NPY_UINT64;
}
int GetNumpyType(float value)
{
    return NPY_FLOAT;
}
int GetNumpyType(double value)
{
    return NPY_DOUBLE;
}
int GetNumpyType(long double value)
{
    return NPY_LONGDOUBLE;
}
int GetNumpyType(char * value)
{
    return NPY_STRING;
}

//-------------------------------------
// Returns false if cannot upcast
// Input: numpyInType1, numpyInType2
// Output: convertType1, convertType2
bool GetUpcastType(int numpyInType1, int numpyInType2, int & convertType1, int & convertType2, int64_t funcNumber)
{
    if (numpyInType1 == numpyInType2)
    {
        convertType1 = numpyInType1;
        convertType2 = numpyInType1;
        return true;
    }
    if (numpyInType1 >= 0 && numpyInType1 <= NPY_LONGDOUBLE && numpyInType2 >= 0 && numpyInType2 <= NPY_LONGDOUBLE)
    {
        stUpCast * pUpcast;
        if (funcNumber >= MATH_OPERATION::CMP_EQ && funcNumber <= MATH_OPERATION::CMP_GTE)
        {
            pUpcast = &gUpcastTableComparison[numpyInType1 * 14 + numpyInType2];
            LOGGING("special comparison upcast %d %d  to   %d %d\n", numpyInType1, numpyInType2, pUpcast->dtype1, pUpcast->dtype2);
        }
        else
        {
            pUpcast = &gUpcastTable[numpyInType1 * 14 + numpyInType2];
        }
        convertType1 = pUpcast->dtype1;
        convertType2 = pUpcast->dtype2;
        return true;
    }
    else
    {
        // check for strings..
        if (numpyInType1 == NPY_UNICODE)
        {
            if (numpyInType2 == NPY_STRING)
            {
                convertType1 = NPY_UNICODE;
                convertType2 = NPY_UNICODE;
                return true;
            }
        }
        if (numpyInType2 == NPY_UNICODE)
        {
            if (numpyInType1 == NPY_STRING)
            {
                convertType1 = NPY_UNICODE;
                convertType2 = NPY_UNICODE;
                return true;
            }
        }
        convertType1 = -1;
        convertType2 = -1;
        return false;
    }
}

//------------------------------------------------------------------------------
// Determines the if array is contiguous, which allows for one loop
// The stride of the loop is returned
// Each array has 3 possible properties:
// 1) Itemsize contiguous (vector math and threading possible)
//    Example: a=arange(20)  or a=arange(20).reshape(5,4) or
//    a=arange(20).reshape((5,2,2), order='F')
// 2) Strided contiguous (threading possible -- vector math possible only with
// gather)
//    Example: a=arange(20)[::-1] or a=arange(20)[::2]
// 3) Neither contiguous (must be 2 or more dimensions and at least one
// dimension is strided contiguous)
//    Requires multiple loops to process data
//    Example: a=arange(20).reshape(5,4)[::-1] or a=arange(20).reshape(5,4)[::2]
// Returns:
//  ndim:   number of dimensions
//  stride: stride to use if contig is true
//  direction: 0 - neither RowMajor or ColMajor (fully contiguous)
//     > 0 RowMajor with value being the dimension where contiguous breaks
//     < 0 ColMajor with -value being the dimension where contiguous breaks
//  return value 0: one loop can process all data, false = multiple loops
//  NOTE: if return value is 0 and itemsze == stride, then vector math possible
//
int GetStridesAndContig(PyArrayObject const * inArray, int & ndim, int64_t & stride)
{
    stride = PyArray_ITEMSIZE(inArray);
    int direction = 0;
    ndim = PyArray_NDIM(inArray);
    if (ndim > 0)
    {
        stride = PyArray_STRIDE(inArray, 0);
        if (ndim > 1)
        {
            // at least two strides
            int ndims = PyArray_NDIM(inArray);
            int64_t lastStride = PyArray_STRIDE(inArray, ndims - 1);
            if (lastStride == stride)
            {
                // contiguous with one of the dimensions having length 1
            }
            else if (std::abs(lastStride) < std::abs(stride))
            {
                // Row Major - 'C' Style
                // work backwards
                int currentdim = ndims - 1;
                int64_t curStrideLen = lastStride;
                while (currentdim != 0)
                {
                    curStrideLen *= PyArray_DIM(inArray, currentdim);
                    LOGGING("'C' %lld vs %lld  dim: %lld  stride: %lld \n", curStrideLen, PyArray_STRIDE(inArray, currentdim - 1),
                            PyArray_DIM(inArray, currentdim - 1), lastStride);
                    if (PyArray_STRIDE(inArray, currentdim - 1) != curStrideLen)
                        break;
                    currentdim--;
                }
                stride = lastStride;
                direction = currentdim;
            }
            else
            {
                // Col Major - 'F' Style
                int currentdim = 0;
                int64_t curStrideLen = stride;
                while (currentdim != (ndims - 1))
                {
                    curStrideLen *= PyArray_DIM(inArray, currentdim);
                    LOGGING("'F' %lld vs %lld  dim:  %lld   stride: %lld \n", curStrideLen,
                            PyArray_STRIDE(inArray, currentdim + 1), PyArray_DIM(inArray, currentdim + 1), stride);
                    if (PyArray_STRIDE(inArray, currentdim + 1) != curStrideLen)
                        break;
                    currentdim++;
                }
                // think!
                // direction = (ndims - 1) - currentdim;
                direction = currentdim - (ndims - 1);
                // contig = currentdim == (ndims - 1);
            }
        }
    }
    return direction;
}
