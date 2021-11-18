#include "RipTide.h"
#include "HashFunctions.h"
#include "HashLinear.h"
#include "ndarray.h"

// struct ndbuf;
// typedef struct ndbuf {
//   struct ndbuf *next;
//   struct ndbuf *prev;
//   Py_ssize_t len;     /* length of data */
//   Py_ssize_t offset;  /* start of the array relative to data */
//   char *data;         /* raw data */
//   int flags;          /* capabilities of the base buffer */
//   Py_ssize_t exports; /* number of exports */
//   Py_buffer base;     /* base buffer */
//} ndbuf_t;
//
// typedef struct {
//   PyObject_HEAD
//      int flags;          /* ndarray flags */
//   ndbuf_t staticbuf;  /* static buffer for re-exporting mode */
//   ndbuf_t *head;      /* currently active base buffer */
//} NDArrayObject;

#define LOGGING(...)
//#define LOGGING printf

//-----------------------------------------------------------------------------------------
// IsMember
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//    Returns: boolean array and optional int64_t location array
PyObject * IsMember64(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    PyArrayObject * inArr2 = NULL;
    int hashMode;
    int64_t hintSize = 0;

    if (! PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &hashMode))
        return NULL;

    int32_t arrayType1 = PyArray_TYPE(inArr1);
    int32_t arrayType2 = PyArray_TYPE(inArr2);

    int sizeType1 = (int)NpyItemSize((PyObject *)inArr1);
    int sizeType2 = (int)NpyItemSize((PyObject *)inArr2);

    LOGGING("IsMember32 %s vs %s   size: %d  %d\n", NpyToString(arrayType1), NpyToString(arrayType2), sizeType1, sizeType2);

    if (arrayType1 != arrayType2)
    {
        // Arguments do not match
        PyErr_Format(PyExc_ValueError, "IsMember32 needs first arg to match %s vs %s", NpyToString(arrayType1),
                     NpyToString(arrayType2));
        return NULL;
    }

    if (sizeType1 == 0)
    {
        // Weird type
        PyErr_Format(PyExc_ValueError, "IsMember32 needs a type it understands %s vs %s", NpyToString(arrayType1),
                     NpyToString(arrayType2));
        return NULL;
    }

    if (arrayType1 == NPY_OBJECT)
    {
        PyErr_Format(PyExc_ValueError,
                     "IsMember32 cannot handle unicode strings, "
                     "please convert to np.chararray");
        return NULL;
    }

    int ndim = PyArray_NDIM(inArr1);
    npy_intp * dims = PyArray_DIMS(inArr1);

    int ndim2 = PyArray_NDIM(inArr2);
    npy_intp * dims2 = PyArray_DIMS(inArr2);

    int64_t arraySize1 = CalcArrayLength(ndim, dims);
    int64_t arraySize2 = CalcArrayLength(ndim2, dims2);

    PyArrayObject * boolArray = AllocateNumpyArray(ndim, dims, NPY_BOOL);
    CHECK_MEMORY_ERROR(boolArray);

    PyArrayObject * indexArray = AllocateNumpyArray(ndim, dims, NPY_INT64);
    CHECK_MEMORY_ERROR(indexArray);

    if (boolArray && indexArray)
    {
        void * pDataIn1 = PyArray_BYTES(inArr1);
        void * pDataIn2 = PyArray_BYTES(inArr2);
        int8_t * pDataOut1 = (int8_t *)PyArray_BYTES(boolArray);
        int64_t * pDataOut2 = (int64_t *)PyArray_BYTES(indexArray);

        // printf("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

        if (arrayType1 >= NPY_STRING)
        {
            LOGGING("Calling string!\n");
            IsMemberHashString64(arraySize1, sizeType1, (const char *)pDataIn1, arraySize2, sizeType2, (const char *)pDataIn2,
                                 pDataOut2, pDataOut1, HASH_MODE(hashMode), hintSize, arrayType1 == NPY_UNICODE);
        }
        else
        {
            if (arrayType1 == NPY_FLOAT32 || arrayType1 == NPY_FLOAT64)
            {
                IsMemberHash64(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, pDataOut1, sizeType1 + 100,
                               HASH_MODE(hashMode), hintSize);
            }
            else
            {
                IsMemberHash64(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, pDataOut1, sizeType1, HASH_MODE(hashMode),
                               hintSize);
            }

            PyObject * retObject = Py_BuildValue("(OO)", boolArray, indexArray);
            Py_DECREF((PyObject *)boolArray);
            Py_DECREF((PyObject *)indexArray);

            return (PyObject *)retObject;
        }
    }
    // out of memory
    return NULL;
}

//-----------------------------------------------------------------------------------
// IsMemberCategorical
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//    Fourth arg: hintSize
//    Returns:
//       missed: 1 or 0
//       an array int8/16/32/64 location array same size as array1
//       index: index location of where first arg found in second arg  (index
//       into second arg)
PyObject * IsMemberCategorical(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    PyArrayObject * inArr2 = NULL;
    int hashMode = HASH_MODE_MASK;
    int64_t hintSize = 0;

    if (PyTuple_GET_SIZE(args) == 3)
    {
        if (! PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &hashMode))
            return NULL;
    }
    else
    {
        if (! PyArg_ParseTuple(args, "O!O!iL", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &hashMode, &hintSize))
            return NULL;
    }
    int32_t arrayType1 = PyArray_TYPE(inArr1);
    int32_t arrayType2 = PyArray_TYPE(inArr2);

    int sizeType1 = (int)NpyItemSize((PyObject *)inArr1);
    int sizeType2 = (int)NpyItemSize((PyObject *)inArr2);

    LOGGING("IsMember32 %s vs %s   size: %d  %d\n", NpyToString(arrayType1), NpyToString(arrayType2), sizeType1, sizeType2);

    switch (arrayType1)
    {
    CASE_NPY_INT32:
        arrayType1 = NPY_INT32;
        break;
    CASE_NPY_UINT32:
        arrayType1 = NPY_UINT32;
        break;
    CASE_NPY_INT64:

        arrayType1 = NPY_INT64;
        break;
    CASE_NPY_UINT64:

        arrayType1 = NPY_UINT64;
        break;
    }

    switch (arrayType2)
    {
    CASE_NPY_INT32:
        arrayType2 = NPY_INT32;
        break;
    CASE_NPY_UINT32:
        arrayType2 = NPY_UINT32;
        break;
    CASE_NPY_INT64:

        arrayType2 = NPY_INT64;
        break;
    CASE_NPY_UINT64:

        arrayType2 = NPY_UINT64;
        break;
    }

    if (arrayType1 != arrayType2)
    {
        // Arguments do not match
        PyErr_Format(PyExc_ValueError, "IsMemberCategorical needs first arg to match %s vs %s", NpyToString(arrayType1),
                     NpyToString(arrayType2));
        return NULL;
    }

    if (sizeType1 == 0)
    {
        // Weird type
        PyErr_Format(PyExc_ValueError, "IsMemberCategorical needs a type it understands %s vs %s", NpyToString(arrayType1),
                     NpyToString(arrayType2));
        return NULL;
    }

    if (arrayType1 == NPY_OBJECT)
    {
        PyErr_Format(PyExc_ValueError,
                     "IsMemberCategorical cannot handle unicode, object, void "
                     "strings, please convert to np.chararray");
        return NULL;
    }

    int64_t arraySize1 = ArrayLength(inArr1);
    int64_t arraySize2 = ArrayLength(inArr2);

    void * pDataIn1 = PyArray_BYTES(inArr1);
    void * pDataIn2 = PyArray_BYTES(inArr2);

    PyArrayObject * indexArray = NULL;

    LOGGING("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

    int64_t missed = 0;

    if (arrayType1 >= NPY_STRING)
    {
        LOGGING("Calling string/uni/void!\n");
        missed = IsMemberCategoricalHashStringPre(&indexArray, inArr1, arraySize1, sizeType1, (const char *)pDataIn1, arraySize2,
                                                  sizeType2, (const char *)pDataIn2, HASH_MODE(hashMode), hintSize,
                                                  arrayType1 == NPY_UNICODE);
    }
    else if (arrayType1 == NPY_FLOAT32 || arrayType1 == NPY_FLOAT64 || arrayType1 == NPY_LONGDOUBLE)
    {
        LOGGING("Calling float!\n");
        if (arraySize1 < 2100000000)
        {
            indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
            int32_t * pDataOut2 = (int32_t *)PyArray_BYTES(indexArray);
            missed = IsMemberHashCategorical(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, sizeType1 + 100,
                                             HASH_MODE(hashMode), hintSize);
        }
        else
        {
            indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT64);
            int64_t * pDataOut2 = (int64_t *)PyArray_BYTES(indexArray);
            missed = IsMemberHashCategorical64(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, sizeType1 + 100,
                                               HASH_MODE(hashMode), hintSize);
        }
    }
    else
    {
        LOGGING("Calling hash!\n");
        if (arraySize1 < 2100000000)
        {
            indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
            int32_t * pDataOut2 = (int32_t *)PyArray_BYTES(indexArray);
            missed = IsMemberHashCategorical(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, sizeType1, HASH_MODE(hashMode),
                                             hintSize);
        }
        else
        {
            indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT64);
            int64_t * pDataOut2 = (int64_t *)PyArray_BYTES(indexArray);
            missed = IsMemberHashCategorical64(arraySize1, pDataIn1, arraySize2, pDataIn2, pDataOut2, sizeType1,
                                               HASH_MODE(hashMode), hintSize);
        }
    }

    PyObject * retObject = Py_BuildValue("(LO)", missed, indexArray);
    Py_DECREF((PyObject *)indexArray);
    return (PyObject *)retObject;
}

//-----------------------------------------------------------
// Will find the first occurence of an interger in pArray2
// ReverseMap must be size of uniques in array2
// ReIndex must be size of uniques in array1
// Will then fixup reIndex
//
// Returns: Reindex array
//--------------------------------
// bin1   0 3 0 4 3 2 0 2 3  (5 uniques)  [b'b' b'a' b'b' b'd' b'a' b'e' b'b' b'e' b'a']
// bin2   0 2 3 0 1 1 4      (5 uniques)  [b'a' b'c' b'd' b'a' b'e' b'e' b'b']
// uniq   4 2 1 0 3
// revMap 3 2 1 4 0 -1 -1
// reindx 6 1 4 0 2
//
// final result 6 0 6 2 0 4 6 4 0
template <typename T>
[[nodiscard]] char const * FindFirstOccurence(T const * pArray2, int32_t const * pUnique1, int32_t * pReIndex,
                                              int32_t * pReverseMap, int64_t array2Length, int32_t unique1Length,
                                              int32_t unique2Length, int32_t baseOffset2T)
{
    T baseOffset2 = (T)baseOffset2T;

    // Put invalid as default
    const int32_t invalid = *(int32_t *)GetDefaultForType(NPY_INT32);
    for (int32_t i = 0; i < unique1Length; i++)
    {
        pReIndex[i] = invalid;
    }

    for (int32_t i = 0; i < unique2Length; i++)
    {
        pReverseMap[i] = -1;
    }

    // Make reverse map
    int32_t matchedUniqueLength(0);
    for (int32_t i = 0; i < unique1Length; i++)
    {
        int32_t val = pUnique1[i];
        if (val == invalid)
        {
            continue;
        }
        if (val >= 0 && val < unique2Length)
        {
            pReverseMap[val] = i;
            ++matchedUniqueLength;
        }
        else
        {
            return "Unexpected out-of-bounds unique value";
        }
    }

    // leave for debugging
    // for (int32_t i = 0; i < unique2Length; i++) {
    //   printf("rmap [%d] %d\n", i, pReverseMap[i]);
    //}

    // Find first occurence of values in pUnique
    // TODO: early stopping possible
    if (matchedUniqueLength > 0)
    {
        for (int64_t i = 0; i < array2Length; i++)
        {
            // N.B. The value can be the minimum integer, need to be checked first
            // before offsetting by baseOffset2.
            T val = pArray2[i];

            // Find first occurence
            if (val >= baseOffset2)
            {
                val -= baseOffset2;

                // Check if the value in the second array is found in the first array
                int32_t lookup = pReverseMap[val];

                if (lookup >= 0)
                {
                    // Is this the first occurence?
                    if (pReIndex[lookup] == invalid)
                    {
                        if (i >= std::numeric_limits<int32_t>::max())
                        {
                            return "Unexpected out-of-bounds first occurrence index";
                        }

                        // printf("first occurence of val:%d  at lookup:%d  pos:%d\n",
                        // (int)val, lookup, i);
                        pReIndex[lookup] = static_cast<int32_t>(i);

                        // check if we found everything possible for early exit
                        --matchedUniqueLength;
                        if (matchedUniqueLength <= 0)
                            break;
                    }
                }
            }
        }
    }

    // Leave for debugging
    // for (int32_t i = 0; i < unique1Length; i++) {
    //   printf("ridx [%d] %d\n", i, pReIndex[i]);
    //}

    return nullptr;
}

//----------------------------------------------------------------------
// IsMemberCategoricalFixup
// Input:
//    First arg: bin array A (array1)
//    Fourth arg: value to replace with
//
// Returns: pOutput and pBoolOutput
// TODO: this routine needs to output INT8/16/32/64 instead of just INT32
template <typename T>
void FinalMatch(T const * pArray1, int32_t * pOutput, int8_t * pBoolOutput, int32_t const * pReIndex, int64_t array1Length,
                int32_t baseoffset)
{
    const int32_t invalid = *(int32_t *)GetDefaultForType(NPY_INT32);

    // TODO: Multithreadable
    // Find first occurence of values in pUnique
    for (int64_t i = 0; i < array1Length; i++)
    {
        // N.B. The value can be the minimum integer, need to be checked first
        // before offsetting by baseoffset.
        T val = pArray1[i];

        // Find first occurence
        if (val >= baseoffset)
        {
            val -= baseoffset;
            const int32_t firstoccurence = pReIndex[val];
            pOutput[i] = firstoccurence;
            pBoolOutput[i] = firstoccurence != invalid;
        }
        else
        {
            pOutput[i] = invalid;
            pBoolOutput[i] = 0;
        }
    }
}

//----------------------------------------------------------------------
// IsMemberCategoricalFixup
//    First arg: bin array A (array1)
//    Second arg: bin array B (array2)
//    Third arg: result of ismember on As unique, B's unique
//    Fourth arg: length of uniques on B
//    Fifth arg: baseoffset 0 or 1
//    Sixth arg: baseoffset 0 or 1
//
//    Returns:
//       boolean
//       an array int8/16/32/64 location array same size as array1
//       index: index location of where first arg found in second arg  (index
//       into second arg)
//
// Currently limited to 2bn unique categoricals.
PyObject * IsMemberCategoricalFixup(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    PyArrayObject * inArr2 = NULL;
    PyArrayObject * isMemberUnique = NULL;
    int32_t unique2Length;
    int32_t baseoffset1;
    int32_t baseoffset2;

    if (! PyArg_ParseTuple(args, "O!O!O!iii", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &PyArray_Type, &isMemberUnique,
                           &unique2Length, &baseoffset1, &baseoffset2))
        return NULL;

    int64_t const arraySize1 = (int64_t)ArrayLength(inArr1);
    int64_t const arraySize2 = (int64_t)ArrayLength(inArr2);
    int64_t const arraySizeUnique64 = (int64_t)ArrayLength(isMemberUnique);
    if (arraySizeUnique64 > std::numeric_limits<int32_t>::max())
    {
        // This catches the case where we have more than 2bn unique categoricals.
        PyErr_Format(PyExc_ValueError, "IsMemberCategoricalFixup third argument size unexpected too large to fit in int32_t");
        return NULL;
    }
    int32_t const arraySizeUnique = static_cast<int32_t>(arraySizeUnique64);

    int32_t const uniqueType = PyArray_TYPE(isMemberUnique);
    int32_t const array2Type = PyArray_TYPE(inArr2);
    int32_t const array1Type = PyArray_TYPE(inArr1);

    switch (uniqueType)
    {
    CASE_NPY_INT32:
        break;
    default:
        PyErr_Format(PyExc_ValueError, "IsMemberCategoricalFixup third argument must be type int32_t not %s",
                     NpyToString(uniqueType));
        return NULL;
    }

    LOGGING(
        "IsMemberCategoricalFixup uniqlength:%d   baseoffset1:%d   "
        "baseoffset2:%d\n",
        unique2Length, baseoffset1, baseoffset2);

    // need to reindex this array...
    int32_t * reIndexArray = (int32_t *)WORKSPACE_ALLOC(arraySizeUnique * sizeof(int32_t));
    int32_t * reverseMapArray = (int32_t *)WORKSPACE_ALLOC(unique2Length * sizeof(int32_t));

    PyArrayObject * indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
    PyArrayObject * boolArray = AllocateLikeNumpyArray(inArr1, NPY_BOOL);

    if (reIndexArray && indexArray && boolArray)
    {
        int32_t * pUnique = (int32_t *)PyArray_BYTES(isMemberUnique);
        void * pArray2 = PyArray_BYTES(inArr2);

        char const * err{ nullptr };
        switch (array2Type)
        {
        case NPY_INT8:
            err = FindFirstOccurence<int8_t>((int8_t *)pArray2, pUnique, reIndexArray, reverseMapArray, arraySize2,
                                             arraySizeUnique, unique2Length, baseoffset2);
            break;
        case NPY_INT16:
            err = FindFirstOccurence<int16_t>((int16_t *)pArray2, pUnique, reIndexArray, reverseMapArray, arraySize2,
                                              arraySizeUnique, unique2Length, baseoffset2);
            break;
        CASE_NPY_INT32:
            err = FindFirstOccurence<int32_t>((int32_t *)pArray2, pUnique, reIndexArray, reverseMapArray, arraySize2,
                                              arraySizeUnique, unique2Length, baseoffset2);
            break;
        CASE_NPY_INT64:
            err = FindFirstOccurence<int64_t>((int64_t *)pArray2, pUnique, reIndexArray, reverseMapArray, arraySize2,
                                              arraySizeUnique, unique2Length, baseoffset2);
            break;
        default:
            PyErr_Format(PyExc_ValueError, "IsMemberCategoricalFixup second argument is not INT8/16/32/64");
            return NULL;
        }
        if (err)
        {
            PyErr_Format(PyExc_RuntimeError, err);
            return NULL;
        }

        void * pArray1 = PyArray_BYTES(inArr1);
        int32_t * pIndexOut = (int32_t *)PyArray_BYTES(indexArray);
        int8_t * pBoolOut = (int8_t *)PyArray_BYTES(boolArray);

        switch (array1Type)
        {
        case NPY_INT8:
            FinalMatch<int8_t>((int8_t *)pArray1, pIndexOut, pBoolOut, reIndexArray, arraySize1, baseoffset1);
            break;
        case NPY_INT16:
            FinalMatch<int16_t>((int16_t *)pArray1, pIndexOut, pBoolOut, reIndexArray, arraySize1, baseoffset1);
            break;
        CASE_NPY_INT32:
            FinalMatch<int32_t>((int32_t *)pArray1, pIndexOut, pBoolOut, reIndexArray, arraySize1, baseoffset1);
            break;
        CASE_NPY_INT64:
            FinalMatch<int64_t>((int64_t *)pArray1, pIndexOut, pBoolOut, reIndexArray, arraySize1, baseoffset1);
            break;
        default:
            PyErr_Format(PyExc_ValueError, "IsMemberCategoricalFixup first argument is not INT8/16/32/64");
            return NULL;
        }

        WORKSPACE_FREE(reverseMapArray);
        WORKSPACE_FREE(reIndexArray);
        PyObject * retObject = Py_BuildValue("(OO)", boolArray, indexArray);
        Py_DECREF((PyObject *)boolArray);
        Py_DECREF((PyObject *)indexArray);
        return (PyObject *)retObject;
    }

    PyErr_Format(PyExc_ValueError, "IsMemberCategoricalFixup internal memory error");
    return NULL;
}
