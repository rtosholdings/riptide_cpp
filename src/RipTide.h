#pragma once

// Hack because debug builds force python36_d.lib
#ifdef _DEBUG
    #undef _DEBUG
    #include <Python.h>
    #define _DEBUG
#else
    #include <Python.h>
#endif

//#include <x86intrin.h>
//#include "zmmintrin.h"
//#include "immintrin.h"
//#include <intrin.h>

//#define NPY_1_7_API_VERSION 0x00000007
//#define NPY_1_8_API_VERSION 0x00000008
//#define NPY_1_9_API_VERSION 0x00000008
//#define NPY_1_10_API_VERSION 0x00000008
//#define NPY_1_11_API_VERSION 0x00000008

// TJD: Comment out this define to use newer numpy apis
// Such as PyArray_NDIM
//
#define NPY_NO_DEPRECATED_API 0x00000008

// NOTE: See PY_ARRAY_UNIQUE_SYMBOL
// If this is not included, calling PY_ARRAY functions will have a null value
#define PY_ARRAY_UNIQUE_SYMBOL sharedata_ARRAY_API

#ifndef SHAREDATA_MAIN_C_FILE
    #define NO_IMPORT_ARRAY
#endif

#include <numpy/arrayobject.h>

#include "numpy_traits.h"

#include "CommonInc.h"

//------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------
// useful for converting from numpy objects to dtype and getting the size
typedef struct
{
    PyTypeObject * type;
    int size;
    int typenum;
} stScalarType;

// Structs used to hold any type of AVX 256 bit registers
struct _m128comboi
{
    __m128i i1;
    __m128i i2;
};

struct _m256all
{
    union
    {
        __m256i i;
        __m256d d;
        __m256 s;
        _m128comboi ci;
    };
};

static const int NUMPY_LAST_TYPE = 24;
static const int MAX_NUMPY_TYPE = 24;

// Conversion related routines
extern stScalarType NpTypeObjects[MAX_NUMPY_TYPE];
extern int32_t TypeToDtype(PyTypeObject * out_dtype);
extern int32_t ObjectToDtype(PyArrayObject * obj);

extern int32_t gNumpyTypeToSize[NUMPY_LAST_TYPE];

// Boolean look up tables to go from packed bits to 1 bool per byte
extern int64_t gBooleanLUT64[256];
extern int32_t gBooleanLUT32[16];

extern int64_t gBooleanLUT64Inverse[256];
extern int32_t gBooleanLUT32Inverse[16];

extern PyArray_Descr * g_pDescrLongLong;
extern PyArray_Descr * g_pDescrULongLong;

// For detecting npy scalar bools
typedef struct
{
    PyObject_HEAD npy_bool obval;
} PyBoolScalarObject;

extern bool GetUpcastType(int numpyInType1, int numpyInType2, int & convertType1, int & convertType2, int64_t funcNumber);
extern int GetStridesAndContig(PyArrayObject const * inArray, int & ndim, int64_t & stride);

/**
 * @brief Calculate the number of elements in an array with the given
 * dimensions.
 *
 * @param ndim The number of dimensions (i.e. rank) of the array.
 * @param dims A pointer to the dimensions of the array.
 * @return npy_intp The number of elements in the array.
 * @note Similar to the numpy PyArray_MultiplyList() function.
 */
extern int64_t CalcArrayLength(int ndim, npy_intp * dims);

// for speed and to ensure dim of 0 has length of 0
inline static int64_t CALC_ARRAY_LENGTH(int ndim, npy_intp * dims)
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
        length = 1;
    }
    return length;
}

/**
 * @brief Allocate a numpy array (or FastArray), first trying to get the memory
 * from the recycle pool.
 *
 * @param ndim
 * @param dims
 * @param numpyType
 * @param itemsize
 * @param fortran_array When true, the allocated array will be a Fortran-style
 * array rather than a C-style array.
 * @param strides
 * @return PyArrayObject* The allocated array object, or nullptr if an error
 * occurred.
 * @note if the @p dtype is NPY_STRING or NPY_UNICODE -- then @p itemsize is
 * valid.
 */
extern PyArrayObject * AllocateNumpyArray(int ndim, npy_intp * dims, int32_t numpyType, int64_t itemsize = 0,
                                          bool fortran_array = false, npy_intp * strides = nullptr);

/**
 * @brief Allocate a numpy array (or FastArray) backed by an existing data
 * buffer.
 *
 * @param ndim
 * @param dims
 * @param numpyType
 * @param itemsize
 * @param data
 * @param array_flags NPY_ARRAY_* flags
 * @param strides
 * @return PyObject* The allocated PyArrayObject object; or nullptr if @p data
 * is nullptr or an error occurred.
 * @note if the @p dtype is NPY_STRING or NPY_UNICODE -- then @p itemsize is
 * valid.
 */
extern PyArrayObject * AllocateNumpyArrayForData(int ndim, npy_intp * dims, int32_t numpyType, int64_t itemsize, char * data,
                                                 int array_flags, npy_intp * strides = nullptr);

extern PyArrayObject * AllocateLikeNumpyArray(PyArrayObject const * inArr, int32_t numpyType);
extern PyArrayObject * AllocateLikeResize(PyArrayObject * inArr, npy_intp rowSize);

extern int32_t GetArrDType(PyArrayObject * inArr);
extern bool ConvertScalarObject(PyObject * inObject1, _m256all * pDest, int16_t numpyType, void ** pDataIn, int64_t * pItemSize);
extern bool ConvertSingleItemArray(void * pInput, int16_t numpyInType, _m256all * pDest, int16_t numpyType);
extern PyArrayObject * EnsureContiguousArray(PyArrayObject * inObject);

extern int64_t NpyItemSize(PyObject * self);
extern const char * NpyToString(int32_t numpyType);
extern int32_t NpyToSize(int32_t numpyType);

/**
 * @brief Calculate the number of elements in the given array.
 *
 * @param inArr Pointer to a @c PyArrayObject.
 * @return npy_intp The number of elements in the array.
 * @note Similar to the numpy PyArray_Size() function.
 */
extern int64_t ArrayLength(PyArrayObject * inArr);

extern PyTypeObject * g_FastArrayType;
extern PyObject * g_FastArrayModule;

static inline bool IsFastArrayView(PyArrayObject * pArray)
{
    return (pArray->ob_base.ob_type == g_FastArrayType);
}

// Optimistic faster way to check for array (most common check)
static inline bool IsFastArrayOrNumpy(PyArrayObject * pArray)
{
    PyTypeObject * pto = pArray->ob_base.ob_type;
    // consider pto->tp_base for Categoricals
    return ((pto == g_FastArrayType) || (pto == &PyArray_Type) || PyType_IsSubtype(pto, &PyArray_Type));
}

// This is the same as calling np.dtype(inobject)
// It may result in an error and NULL returned
// User may pass in a string such as 'f4' for float32
// User may pass in np.int32 as well
// This comes from looking at numpy/core/src/multiarray/ctors.c
static inline PyArray_Descr * DTypeToDescr(PyObject * inobject)
{
    if (inobject->ob_type == &PyArrayDescr_Type)
    {
        Py_INCREF(inobject);
        return (PyArray_Descr *)inobject;
    }
    return (PyArray_Descr *)PyObject_CallFunctionObjArgs((PyObject *)&PyArrayDescr_Type, inobject, NULL);
}

PyObject * SetFastArrayView(PyArrayObject * pArray);

extern int GetNumpyType(bool value);
extern int GetNumpyType(int8_t value);
extern int GetNumpyType(int16_t value);
extern int GetNumpyType(int32_t value);
extern int GetNumpyType(int64_t value);
extern int GetNumpyType(uint8_t value);
extern int GetNumpyType(uint16_t value);
extern int GetNumpyType(uint32_t value);
extern int GetNumpyType(uint64_t value);
extern int GetNumpyType(float value);
extern int GetNumpyType(double value);
extern int GetNumpyType(long double value);
extern int GetNumpyType(char * value);

//---------------------------------------------------------
// Returns nanoseconds since utc epoch
extern uint64_t GetUTCNanos();

template <typename T>
static void * GetInvalid()
{
    return GetDefaultForType(GetNumpyType((T)0));
}

PyFunctionObject * GetFunctionObject(PyObject * arg1);

// int32_t invalid index used
#define INVALID_INDEX -214783648

// mark invalid index with -1 because it will always
#define GB_INVALID_INDEX -1
#define GB_BASE_INDEX 1
