// Python Interface to SDS File

#include "Compress.h"
#include "FileReadWrite.h"
#include "SDSFile.h"
#include "SharedMemory.h"
#include "bytesobject.h"
#include <Python.h>
#include <stdlib.h>

#include "MathWorker.h"
#include "MultiKey.h"

// For sumbooleanmask
#include "Convert.h"

#ifndef OUT
    #define OUT
#endif

#define LOGGING(...)
//#define LOGGING printf

// For supporting older versions of PyPy.
// Source:
// https://github.com/numpy/numpy/blob/504fd7b2eedb90dd3aa0b326ac8c3120118b5f2d/numpy/core/src/multiarray/typeinfo.c#L8
#if (defined(PYPY_VERSION_NUM) && (PYPY_VERSION_NUM <= 0x07030000))
    // PyPy issue 3160
    #include <structseq.h>
#endif

// PyTypeObject for the 'SDSArrayCutoffs' StructSequence type.
static PyTypeObject PyType_SDSArrayCutoffs;
// PyTypeObject for the 'SDSFileInfo' StructSequence type.
static PyTypeObject PyType_SDSFileInfo;
// PyTypeObject for the 'SDSContainerItem' StructSequence type.
static PyTypeObject PyType_SDSContainerItem;
// PyTypeObject for the 'SDSArrayInfo' StructSequence type.
static PyTypeObject PyType_SDSArrayInfo;

static PyStructSequence_Field SDSArrayCutoffs_fields[] = {
    // typing.Tuple[np.ndarray, ...]
    { "arrays",
      "A sequence of arrays created by stacking the columns of one or "
      "more files or one or more sections of a single file." },
    // typing.Sequence[riptide_cpp.SDSContainerItem]
    { "array_infos",
      "A sequence of the same length as the sequence provided in the 'arrays' "
      "component, where each element of the sequence contains an object with "
      "the name and SDS flags of the corresponding stacked array in 'arrays'." },
    // typing.Sequence[np.ndarray]
    { "array_cutoffs",
      "A sequence of non-negative integer arrays; each array corresponds to the "
      "same-indexed array in the 'arrays' component and contains the cumsum of "
      "the lengths of the arrays that were stacked to create the output array." },
    // typing.Sequence[bytes]
    { "file_metadata",
      "A sequence where each element is the raw SDS file metadata for the "
      "corresponding element of the 'files' component." },
    // typing.Sequence[bytes]
    { "filenames",
      "A sequence containing the ASCII-encoded filenames of SDS files from "
      "which data was read from during the stacking operation." },
    // typing.Optional[typing.Mapping[str, typing.Any]]
    { "first_file_header",
      "A dictionary created from the SDS file header in "
      "the first file being stacked." },
    { NULL, NULL }
};

static PyStructSequence_Desc SDSArrayCutoffs_desc = {
    // name
    "riptide_cpp.SDSArrayCutoffs",
    // doc
    "Information about the contents of one or more SDS files to be stacked.",
    // fields
    SDSArrayCutoffs_fields,
    // n_in_sequence
    6,
};

static PyStructSequence_Field SDSFileInfo_fields[] = {
    // bytes
    { "column_metadata",
      "JSON-formatted metadata for columns (arrays) within the Dataset which "
      "have types subclassing FastArray (such as Categorical or DateTimeNano)." },
    // Union[Sequence[np.ndarray], Sequence[riptide_cpp.SDSArrayInfo]]
    //   Sequence[np.ndarray] when loading data;
    //   Sequence[riptide_cpp.SDSArrayInfo] when only loading metadata.
    { "columns",
      "A sequence of ndarrays (or derived array types) stored in the "
      "file; or, if reading an SDS file in info-only mode, a "
      "sequence of objects describing the type, shape, etc. of each "
      "of the columns stored in the file." },
    // Sequence[riptide_cpp.SDSContainerItem]
    { "container_items",
      "A sequence of objects, each describing an item stored "
      "within the SDS file." },
    // bytes
    { "file_header", "A dictionary created from the SDS file header." },
    { NULL, NULL }
};

static PyStructSequence_Desc SDSFileInfo_desc = {
    // name
    "riptide_cpp.SDSFileInfo",
    // doc
    "Information about the contents of an SDS file.",
    // fields
    SDSFileInfo_fields,
    // n_in_sequence
    4,
};

static PyStructSequence_Field SDSContainerItem_fields[] = {
    // bytes
    { "itemname", "The item name as an ASCII 'bytes' string." },
    // SDSFlag
    { "flags", "The SDS flags for this item." },
    { NULL, NULL }
};

static PyStructSequence_Desc SDSContainerItem_desc = {
    // name
    "riptide_cpp.SDSContainerItem",
    // doc
    "The name and flags for an item (e.g. a dataset, array, nested struct) "
    "stored within an SDS file.",
    // fields
    SDSContainerItem_fields,
    // n_in_sequence
    2,
};

static PyStructSequence_Field SDSArrayInfo_fields[] = {
    // Tuple[int, ...]
    { "shape", "The array shape, given as a tuple of non-negative integers." },
    // int
    { "dtype", "The numeric code for the array's numpy dtype." },
    // int
    { "flags", "The numpy array flags for the array." },
    // int
    { "itemsize", "The size, in bytes, of each element in the array." },
    { NULL, NULL }
};

static PyStructSequence_Desc SDSArrayInfo_desc = {
    // name
    "riptide_cpp.SDSArrayInfo",
    // doc
    "Description of an array stored within an SDS file.",
    // fields
    SDSArrayInfo_fields,
    // n_in_sequence
    4,
};

//
// Factory functions for creating instances of SDS Python types.
//

static PyObject * Create_SDSArrayCutoffs(PyObject * const returnArrayTuple, PyObject * const pyListName,
                                         PyObject * const pyArrayCutoffs, PyObject * const pyMeta, PyObject * const pyFiles,
                                         PyObject * const firstFileHeader)
{
    PyObject * entry = PyStructSequence_New(&PyType_SDSArrayCutoffs);

    if (! entry)
        return NULL;

    // Py_BuildValue docs:
    // https://docs.python.org/3/c-api/arg.html#c.Py_BuildValue
    PyStructSequence_SET_ITEM(entry, 0, returnArrayTuple);
    PyStructSequence_SET_ITEM(entry, 1, pyListName);
    PyStructSequence_SET_ITEM(entry, 2, pyArrayCutoffs);
    PyStructSequence_SET_ITEM(entry, 3, pyMeta);
    PyStructSequence_SET_ITEM(entry, 4, pyFiles);
    PyStructSequence_SET_ITEM(entry, 5, firstFileHeader);

    if (PyErr_Occurred())
    {
        Py_DECREF(entry);
        return NULL;
    }

    return entry;
}

static PyObject * Create_SDSArrayInfo(PyObject * const arrayShapeTuple, const int32_t dtype, const int32_t flags,
                                      const int32_t itemsize)
{
    PyObject * entry = PyStructSequence_New(&PyType_SDSArrayInfo);

    if (! entry)
        return NULL;

    // Py_BuildValue docs:
    // https://docs.python.org/3/c-api/arg.html#c.Py_BuildValue
    PyStructSequence_SET_ITEM(entry, 0, arrayShapeTuple);
    PyStructSequence_SET_ITEM(entry, 1, PyLong_FromLong(dtype));
    PyStructSequence_SET_ITEM(entry, 2, PyLong_FromLong(flags));
    PyStructSequence_SET_ITEM(entry, 3, PyLong_FromLong(itemsize));

    if (PyErr_Occurred())
    {
        Py_DECREF(entry);
        return NULL;
    }

    return entry;
}

static PyObject * Create_SDSContainerItem(const char * const arrayName, const unsigned char sdsFlags)
{
    PyObject * entry = PyStructSequence_New(&PyType_SDSContainerItem);

    if (! entry)
        return NULL;

    // Py_BuildValue docs:
    // https://docs.python.org/3/c-api/arg.html#c.Py_BuildValue
    PyStructSequence_SET_ITEM(entry, 0, PyBytes_FromString(arrayName));
    // TODO: If we're able to define the SDSFlags enum (as an enum.IntFlags)
    // within the riptide_cpp module,
    //       we should create an instance of that type for the next tuple element
    //       instead of a plain integer.
    PyStructSequence_SET_ITEM(entry, 1, PyLong_FromUnsignedLong(sdsFlags));

    if (PyErr_Occurred())
    {
        Py_DECREF(entry);
        return NULL;
    }

    return entry;
}

static PyObject * Create_SDSFileInfo(PyObject * const columnMetadataBytes, PyObject * const columns,
                                     PyObject * const containerItems, PyObject * const fileMetadataDict)
{
    PyObject * entry = PyStructSequence_New(&PyType_SDSFileInfo);

    if (! entry)
        return NULL;

    // Py_BuildValue docs:
    // https://docs.python.org/3/c-api/arg.html#c.Py_BuildValue
    PyStructSequence_SET_ITEM(entry, 0, columnMetadataBytes);
    PyStructSequence_SET_ITEM(entry, 1, columns);
    PyStructSequence_SET_ITEM(entry, 2, containerItems);
    PyStructSequence_SET_ITEM(entry, 3, fileMetadataDict);

    if (PyErr_Occurred())
    {
        Py_DECREF(entry);
        return NULL;
    }

    return entry;
}

// TODO: It could be helpful for this function to also define enum types used by
// SDS and currently repeated
//       in riptable/rt_enum.py. rt_enum.py could still import + re-export these
//       for backwards-compatibility, but it would allow the definitions to be
//       centralized within riptide_cpp. A partial solution is implemented here
//       (but it'd be better if we could define the IntEnum itself via the C
//       API):
//       https://stackoverflow.com/questions/61451232/how-to-return-a-python-enum-from-c-extension/61454030#61454030
// TODO: Should this function accept a pointer to the module itself (instead of
// the module dictionary) and
//       use the PyModule_AddObject() function to add the types to the module?
bool RegisterSdsPythonTypes(PyObject * module_dict)
{
    // For each of the Struct Sequence (C-API version of namedtuple) types defined
    // for SDS, initialize the PyTypeObject then add it to the module dictionary
    // provided by the caller.

    if (PyStructSequence_InitType2(&PyType_SDSArrayCutoffs, &SDSArrayCutoffs_desc) < 0)
    {
        return false;
    }
    if (PyDict_SetItemString(module_dict, "SDSArrayCutoffs", (PyObject *)&PyType_SDSArrayCutoffs) < 0)
    {
        return false;
    }

    if (PyStructSequence_InitType2(&PyType_SDSFileInfo, &SDSFileInfo_desc) < 0)
    {
        return false;
    }
    if (PyDict_SetItemString(module_dict, "SDSFileInfo", (PyObject *)&PyType_SDSFileInfo) < 0)
    {
        return false;
    }

    // TODO: Do we need to Py_DECREF(&SDSFileInfo_desc) here to keep the refcounts
    // correct?

    if (PyStructSequence_InitType2(&PyType_SDSContainerItem, &SDSContainerItem_desc) < 0)
    {
        return false;
    }
    if (PyDict_SetItemString(module_dict, "SDSContainerItem", (PyObject *)&PyType_SDSContainerItem) < 0)
    {
        return false;
    }

    if (PyStructSequence_InitType2(&PyType_SDSArrayInfo, &SDSArrayInfo_desc) < 0)
    {
        return false;
    }
    if (PyDict_SetItemString(module_dict, "SDSArrayInfo", (PyObject *)&PyType_SDSArrayInfo) < 0)
    {
        return false;
    }

    // TODO: Can we add type hints to these types by defining an __annotations__
    // field on them (using PyObject_SetAttrString())?
    //       (And perhaps also _fields and some others defined by
    //       typing.NamedTuple-derived classes.)

    // In [1]: import numpy as np
    //
    // In [2]: import riptable as rt
    //
    // In [3]: from typing import NamedTuple, Sequence, Tuple, Union
    //
    // In [4]: class SDSArrayInfo(NamedTuple):
    //   ...:     shape: Tuple[int, ...]
    //   ...:     dtype: int
    //   ...:     flags: int
    //   ...:     """numpy array flags"""
    //   ...:     itemsize: int
    //   ...:
    //
    // In [5]: class SDSContainerItem(NamedTuple):
    //   ...:     itemname: bytes
    //   ...:     flags: rt.rt_enum.SDSFlag
    //   ...:
    //
    // In [6]: class SDSFileInfo(NamedTuple):
    //   ...:     column_metadata: bytes
    //   ...:     """JSON-formatted metadata for some columns in the dataset."""
    //   ...:     columns: Union[Sequence[np.ndarray], Sequence[SDSArrayInfo]]
    //   ...:     container_items: Sequence[SDSContainerItem]
    //   ...:     file_metadata: bytes
    //   ...:     """JSON-formatted file metadata."""
    //   ...:

    return true;
}

// Call to clear any previous errors
static void ClearErrors()
{
    g_lastexception = 0;
}

// check to see if any errors were recorded
// Returns true if there was an error
static bool CheckErrors()
{
    if (g_lastexception)
    {
        PyErr_Format(PyExc_ValueError, g_errorbuffer);
        return true;
    }
    return false;
}

//--------------------------------------------------
//
void * BeginAllowThreads()
{
    return PyEval_SaveThread();
}

//--------------------------------------------------
//
void EndAllowThreads(void * saveObject)
{
    return PyEval_RestoreThread((PyThreadState *)saveObject);
}

//--------------------------------------------------
//
static char * GetMemoryOffset(char * BaseAddress, int64_t offset)
{
    return (BaseAddress + offset);
}

//--------------------------------------------------
//
static SDS_ARRAY_BLOCK * GetArrayBlock(char * baseOffset, int64_t arrayNum)
{
    SDS_ARRAY_BLOCK * pArrayBlock =
        (SDS_ARRAY_BLOCK *)GetMemoryOffset(baseOffset, ((SDS_FILE_HEADER *)baseOffset)->ArrayBlockOffset);
    return &pArrayBlock[arrayNum];
}

void DestroyStringList(SDS_STRING_LIST * pStringList)
{
    for (const char * filename : *pStringList)
    {
        WORKSPACE_FREE((void *)filename);
    }
    delete pStringList;
}

//----------------------------------
// caller must delete STRING_LIST
// handles both BYTE and UNICODE
SDS_STRING_LIST * StringListToVector(PyObject * listFilenames)
{
    SDS_STRING_LIST * returnList = new SDS_STRING_LIST;

    if (PyList_Check(listFilenames))
    {
        int64_t filenameCount = PyList_GET_SIZE(listFilenames);
        returnList->reserve(filenameCount);

        for (int64_t i = 0; i < filenameCount; i++)
        {
            PyObject * pBytes = PyList_GET_ITEM(listFilenames, i);
            const char * fileName = NULL;
            if (PyBytes_Check(pBytes))
            {
                fileName = PyBytes_AsString(pBytes);

                int64_t strSize = strlen(fileName);
                char * pNewString = (char *)WORKSPACE_ALLOC(strSize + 1);
                memcpy(pNewString, fileName, strSize);
                pNewString[strSize] = 0;

                // add to our own list
                returnList->push_back(pNewString);
            }
            else if (PyUnicode_Check(pBytes))
            {
                PyObject * temp2 = PyUnicode_AsASCIIString(pBytes);
                if (temp2 != NULL)
                {
                    fileName = PyBytes_AsString(temp2);
                    int64_t strSize = strlen(fileName);
                    char * pNewString = (char *)WORKSPACE_ALLOC(strSize + 1);
                    memcpy(pNewString, fileName, strSize);
                    pNewString[strSize] = 0;

                    returnList->push_back(pNewString);

                    // Release reference
                    Py_DecRef(temp2);
                }
            }
        }
    }

    return returnList;
}

//----------------------------------------------------
// Input: Python List of Tuples(asciiz string,int)
// Output: Write to pListNames.
//         ASCIIZ strings follow by UINT8 enum (string1, 0, enum1, string2, 0,
//         enum2, etc.)
// Returns: Length of data in pListNames
//
int64_t BuildListInfo(PyListObject * inListNames, OUT char * pListNames)
{
    int64_t listNameCount = PyList_GET_SIZE(inListNames);
    char * pStart = pListNames;

    for (int i = 0; i < listNameCount; i++)
    {
        PyObject * pTuple = PyList_GET_ITEM(inListNames, i);

        if (pTuple && PyTuple_Check(pTuple))
        {
            PyObject * pBytes = PyTuple_GET_ITEM(pTuple, 0);

            if (PyBytes_Check(pBytes))
            {
                int overflow = 0;
                int64_t value = PyLong_AsLongLongAndOverflow(PyTuple_GET_ITEM(pTuple, 1), &overflow);
                int64_t strSize = Py_SIZE(pBytes);

                char * pName = PyBytes_AS_STRING(pBytes);
                LOGGING("Name is %s -- size %d  value %d\n", pName, (int)strSize, (int)value);

                while ((*pListNames++ = *pName++))
                    ;

                // Store the 1 byte enum type
                *pListNames++ = (uint8_t)value;
            }
            else
            {
                printf(
                    "!!internal error processing, check that list is in bytes "
                    "instead of unicode\n");
            }
        }
        else
        {
            printf("!!internal error2 processing, is it a list of tuples?\n");
        }
    }

    return pListNames - pStart;
}

/**
 * @brief Create and return a Python list of SDSContainerItem objects.
 * @param pArrayNames pointer to string, null terminated, followed by UINT8 enum
 * value
 * @param nameBlockCount how many names
 * @param nameSize the size of @p pArrayNames (all of the names)
 * @returns A Python list of SDSContainerItem objects.
 */
PyObject * MakeListNames(const char * pArrayNames, int64_t nameBlockCount, const int64_t nameSize)
{
    const char * nameData = pArrayNames;
    PyObject * pyListName = PyList_New(nameBlockCount);

    int64_t curPos = 0;
    // for every name
    while (nameBlockCount)
    {
        nameBlockCount--;
        const char * pStart = pArrayNames;

        // skip to end (search for 0 terminating char)
        while (*pArrayNames++)
            ;

        // Read the 1-byte SDSFlags enum value that's stored just after the NUL
        // ('\0') character.
        uint8_t value = *pArrayNames++;

        LOGGING("makelist file name is %s, %lld\n", pStart, nameBlockCount);

        // Create an SDSContainerItem instance for this array.
        PyObject * item_obj = Create_SDSContainerItem(pStart, value);

        // PyList_Append() will add a reference count but PyList_SetItem() will not.
        PyList_SetItem(pyListName, curPos, item_obj);

        curPos++;

        // If we ran too far, break
        if ((pArrayNames - nameData) >= nameSize)
            break;
    }
    return pyListName;
}

//------------------------------------------------------
// Input: file already opened
//
// returns list of names or empty list
PyObject * ReadListNamesPython(char * nameData, SDS_FILE_HEADER * pFileHeader)
{
    LOGGING("ReadListNames %p\n", nameData);

    PyObject * pListName = NULL;
    if (nameData)
    {
        // return list of SDSContainerItem
        // GIL must be held to create the list
        int64_t nameSize = pFileHeader->NameBlockSize;

        pListName = MakeListNames(nameData, pFileHeader->NameBlockCount, nameSize);
    }
    else
    {
        // make empty list
        LOGGING("empty list!\n");
        pListName = PyList_New(0);
    }
    return pListName;
}

//---------------------------------------------------------
// Linux: long = 64 bits
// Windows: long = 32 bits
// TODO: This should be 'constexpr' but can't be as long as we want to support
// old versions of GCC.
/*constexpr*/ static /*NPY_TYPES*/ int FixupDType(const /*NPY_TYPES*/ int dtype, const int64_t itemsize)
{
    if (dtype == NPY_LONG)
    {
        // types 7 (NPY_LONG) and 8 (NPY_ULONG) are ambiguous due to differences
        // in 64-bit OS memory models (LLP64 vs. LP64).
        // https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
        if (itemsize == 4)
        {
            return NPY_INT;
        }
        else
        {
            return NPY_LONGLONG;
        }
    }

    if (dtype == NPY_ULONG)
    {
        // types 7 and 8 are ambiguous
        if (itemsize == 4)
        {
            return NPY_UINT;
        }
        else
        {
            return NPY_ULONGLONG;
        }
    }
    return dtype;
}

//-----------------------------------------
// Return empty string on failure
PyObject * GetMetaData(const char * const metaData, const int64_t metaSize)
{
    if (metaData)
    {
        // caller wants a pystring
        // this will make a copy of the data
        return PyBytes_FromStringAndSize(metaData, metaSize);
    }
    printf("Possible error -returning null on metadata\n");
    const char * const temp = "{}";
    return PyBytes_FromStringAndSize(temp, 2);
}

int SetStringLong(PyObject * pDict, const char * strkey, long long value)
{
    return PyDict_SetItemString(pDict, strkey, (PyObject *)PyLong_FromLongLong(value));
}

//----------------------------------------------------
//
PyObject * GetFileHeaderDict(const SDS_FILE_HEADER * const pFileHeader, SDS_FINAL_CALLBACK * const pSDSFinalCallback)
{
    PyObject * pDict = PyDict_New();
    if (pFileHeader && pDict)
    {
        SetStringLong(pDict, "VersionHigh", (long long)(pFileHeader->VersionHigh));
        SetStringLong(pDict, "VersionLow", (long long)(pFileHeader->VersionLow));
        SetStringLong(pDict, "CompMode", (long long)(pFileHeader->CompMode));
        SetStringLong(pDict, "CompType", (long long)(pFileHeader->CompType));
        SetStringLong(pDict, "CompLevel", (long long)(pFileHeader->CompLevel));
        SetStringLong(pDict, "FileType", (long long)(pFileHeader->FileType));
        SetStringLong(pDict, "StackType", (long long)(pFileHeader->StackType));
        SetStringLong(pDict, "AuthorId", (long long)(pFileHeader->AuthorId));
        SetStringLong(pDict, "TotalArrayCompressedSize", (long long)(pFileHeader->TotalArrayCompressedSize));
        SetStringLong(pDict, "TotalArrayUncompressedSize", (long long)(pFileHeader->TotalArrayUncompressedSize));

        SetStringLong(pDict, "ArrayBlockSize", (long long)(pFileHeader->ArrayBlockSize));
        SetStringLong(pDict, "ArrayBlockOffset", (long long)(pFileHeader->ArrayBlockOffset));

        SetStringLong(pDict, "ArraysWritten", (long long)(pFileHeader->ArraysWritten));
        SetStringLong(pDict, "ArrayFirstOffset", (long long)(pFileHeader->ArrayFirstOffset));

        SetStringLong(pDict, "NameBlockSize", (long long)(pFileHeader->NameBlockSize));
        SetStringLong(pDict, "NameBlockOffset", (long long)(pFileHeader->NameBlockOffset));
        SetStringLong(pDict, "NameBlockCount", (long long)(pFileHeader->NameBlockCount));

        SetStringLong(pDict, "BandBlockSize", (long long)(pFileHeader->BandBlockSize));
        SetStringLong(pDict, "BandBlockOffset", (long long)(pFileHeader->BandBlockOffset));
        SetStringLong(pDict, "BandBlockCount", (long long)(pFileHeader->BandBlockCount));
        SetStringLong(pDict, "BandSize", (long long)(pFileHeader->BandSize));

        SetStringLong(pDict, "SectionBlockSize", (long long)(pFileHeader->SectionBlockSize));
        SetStringLong(pDict, "SectionBlockOffset", (long long)(pFileHeader->SectionBlockOffset));
        SetStringLong(pDict, "SectionBlockCount", (long long)(pFileHeader->SectionBlockCount));
        SetStringLong(pDict, "SectionBlockReservedSize", (long long)(pFileHeader->SectionBlockReservedSize));

        SetStringLong(pDict, "FileOffset", (long long)(pFileHeader->FileOffset));
        SetStringLong(pDict, "TimeStampUTCNanos", (long long)(pFileHeader->TimeStampUTCNanos));
        if (pSDSFinalCallback && pSDSFinalCallback->strFileName)
        {
            PyDict_SetItemString(pDict, "Filename", PyUnicode_FromString(pSDSFinalCallback->strFileName));
            SDSSectionName * pSection = pSDSFinalCallback->pSectionName;

            // The sectionoffset is used to reorder
            if (pSection)
                SetStringLong(pDict, "SectionOffset", pSection->SectionOffset);

            // Check if we have sections
            if (pSection && pSection->SectionCount)
            {
                // Create a new list of strings containing section names
                PyObject * pSectionList = PyList_New(pSection->SectionCount);
                PyObject * pSectionListOffset = PyList_New(pSection->SectionCount);
                PyDict_SetItemString(pDict, "Sections", pSectionList);
                PyDict_SetItemString(pDict, "SectionOffsets", pSectionListOffset);

                for (int64_t n = 0; n < pSection->SectionCount; n++)
                {
                    PyList_SetItem(pSectionList, n, PyUnicode_FromString(pSection->pSectionNames[n]));
                    PyList_SetItem(pSectionListOffset, n, (PyObject *)PyLong_FromLongLong(pSection->pSectionOffsets[n]));
                }
            }
        }
    }

    // TODO: Use PyDictProxy_New(pDict) to wrap the dictionary in a read-only
    // proxy so it can't be modified after being returned.
    return pDict;
}

/**
 * @brief Create an SDSFileInfo Python object.
 * @param[in] pListName A possibly-null Python list of SDSContainerItem objects.
 * @param[in] pystring A possibly-null Python 'bytes' object containing
 * JSON-encoded column metadata.
 * @param arrayCount Length (# of elements) of @p pArrayBlockFirst.
 * @param[in] pArrayBlockFirst (allocated array block)
 * @param[in] pFileHeader TODO
 * @param[in] pSDSFinalCallback TODO
 * @returns python object (SDSFileInfo) to return to user
 */
PyObject * GetSDSFileInfo(PyObject * const pListName, PyObject * const pystring, const int64_t arrayCount,
                          const SDS_ARRAY_BLOCK * const pArrayBlockFirst, const SDS_FILE_HEADER * const pFileHeader,
                          // NULL possible
                          SDS_FINAL_CALLBACK * const pSDSFinalCallback = nullptr)
{
    PyObject * numpyArrayTuple = PyTuple_New(arrayCount);

    LOGGING("In GetSDSFileInfo -- %lld   %p\n", arrayCount, pArrayBlockFirst);

    // Insert all the arrays
    for (int64_t i = 0; i < arrayCount; i++)
    {
        const SDS_ARRAY_BLOCK * pArrayBlock = &pArrayBlockFirst[i];

        // LOGGING("Array block %lld at %p  compsize:%lld  %lld\n", i, pArrayBlock,
        // pArrayBlock->ArrayCompressedSize, pArrayBlock->ArrayUncompressedSize);

        PyObject * shapeTuple = PyTuple_New(pArrayBlock->NDim);
        for (int64_t j = 0; j < pArrayBlock->NDim; j++)
        {
            PyTuple_SET_ITEM(shapeTuple, j, PyLong_FromLongLong(pArrayBlock->Dimensions[j]));
        }

        // dtype fixup for Windows (LLP64) vs Linux (LP64).
        // INT/UINT/LONG/ULONG dtypes are tied directly to the OS' definition of
        // 'int' and 'long', so they have OS-specific sizes and aren't stable for
        // use when serializing arrays to disk, where they could be read from
        // another OS. Such arrays should be caught (and the dtype transformed) when
        // writing out the arrays; but in case such an array has been written to
        // disk already, we try to fix it up here.
        // TODO: For returning metadata (as this function does), it seems like it'd
        // be better to _not_ do this here
        //       and instead return *exactly* what's in the file, to aid in
        //       diagnosing issues if we hit such an array. SDS readers should
        //       probably also refuse to load any arrays saved out with these dtypes
        //       rather than trying to fix them up -- the onus should be on the
        //       writer to handle this dtype "fixup" so the on-disk format is stable
        //       across OSes.
        const int FixedUpDtype = FixupDType(pArrayBlock->DType, pArrayBlock->ItemSize);

        // Build the SDSArrayInfo struct sequence (C-API namedtuple) instance for
        // this array.
        PyObject * array_info = Create_SDSArrayInfo(shapeTuple, FixedUpDtype, pArrayBlock->Flags, pArrayBlock->ItemSize);

        PyTuple_SET_ITEM(numpyArrayTuple, i, array_info);
    }

    // Read the file header and create a Python dictionary object from the header
    // data.
    PyObject * pDict = GetFileHeaderDict(pFileHeader, pSDSFinalCallback);

    // Create and return an SDSFileInfo Struct Sequence (C-API namedtuple).
    PyObject * returnFileInfo = Create_SDSFileInfo(pystring, numpyArrayTuple, pListName, pDict);
    return returnFileInfo;
}

//-------------------------------------------------------------
// Tries to find the '!' char in the string
// if it finds the ! it returns the location
// otherwise it returns NULL
const char * FindBang(const char * pString)
{
    while (*pString)
    {
        if (*pString == '!')
            return pString;
        pString++;
    }
    return NULL;
}
//----------------------------------------------------
// Returns true if included
bool IsIncluded(PyObject * pInclusionList, const char * pArrayName)
{
    // If there is no inclusion list, assume all are included
    if (pInclusionList)
    {
        PyObject * includeDict = pInclusionList;
        PyObject * exists = PyDict_GetItemString(includeDict, pArrayName);

        // NOTE: to do... check for !
        if (! exists)
        {
            // The pArrayName might be a categorical column
            // If so, it will be in the format categoricalname!col0

            char * pHasBang = (char *)FindBang(pArrayName);
            if (pHasBang)
            {
                // temp remove bang
                *pHasBang = 0;

                // Now check dictionary again to see if we have a match
                exists = PyDict_GetItemString(includeDict, pArrayName);

                // replace bang
                *pHasBang = '!';

                // if we matched return true
                if (exists)
                {
                    LOGGING("categorical columns !array was included %s\n", pArrayName);
                    return true;
                }
            }

            LOGGING("!array was excluded %s\n", pArrayName);
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------
// Clears both FastArray and base array WRITEABLE flags (marked readonly)
void ClearWriteableFlag(PyArrayObject * pArray)
{
    // Flip off writeable flag at low level if came from shared memory
    PyArray_CLEARFLAGS(pArray, NPY_ARRAY_WRITEABLE);

    // Also clear the base flag
    PyArrayObject * pBase = (PyArrayObject *)PyArray_BASE(pArray);
    while (pBase != NULL)
    {
        // make sure base object is a numpy array object
        if (! PyArray_Check(pBase))
        {
            break;
        }
        pArray = pBase;
        pBase = (PyArrayObject *)PyArray_BASE(pArray);
    }
    PyArray_CLEARFLAGS(pArray, NPY_ARRAY_WRITEABLE);
}

//----------------------------------------------------
// Input: sharedmemory struct we are reading from
// Output: python object (tuple of 4 objects) to return to user
// Python ONLY ROUTINE
// pSharedMemory->GetFileHeader()
void * ReadFromSharedMemory(SDS_SHARED_MEMORY_CALLBACK * pSMCB)
{
    SDS_FILE_HEADER * pFileHeader = pSMCB->pFileHeader;
    char * baseOffset = pSMCB->baseOffset;
    int mode = pSMCB->mode;

    PyObject * pystring = NULL;
    PyObject * pListName = NULL;
    LOGGING("Reading from shared memory\n");

    //----------- LOAD ARRAY NAMES -------------------------
    int64_t nameSize = pFileHeader->NameBlockSize;
    if (nameSize)
    {
        char * nameData = GetMemoryOffset(baseOffset, pFileHeader->NameBlockOffset);
        pListName = MakeListNames(nameData, pFileHeader->NameBlockCount, nameSize);
    }
    else
    {
        pListName = PyList_New(0);
    }
    LOGGING("Number of names %lld\n", PyList_GET_SIZE(pListName));

    //------------- META DATA -------------------------------
    // Python will make a copy of this string
    pystring = PyBytes_FromStringAndSize(GetMemoryOffset(baseOffset, pFileHeader->MetaBlockOffset), pFileHeader->MetaBlockSize);

    //--------------- LOAD ARRAYS ---------------------------
    int64_t arrayCount = pFileHeader->ArraysWritten;

    if (mode == COMPRESSION_MODE_INFO)
    {
        return GetSDSFileInfo(pListName, pystring, arrayCount, GetArrayBlock(baseOffset, 0), pFileHeader);
    }

    PyObject * returnTuple = PyTuple_New(arrayCount);

    LOGGING("Number of arrays %lld\n", arrayCount);

    // Insert all the arrays
    for (int64_t i = 0; i < arrayCount; i++)
    {
        SDS_ARRAY_BLOCK * pArrayBlock = GetArrayBlock(baseOffset, i);

        // scalars
        // if (pArrayBlock->Dimensions ==0)

        char * data = GetMemoryOffset(baseOffset, pArrayBlock->ArrayDataOffset);

        // TODO: dtype fixup for Windows vs Linux
        int dtype = FixupDType(pArrayBlock->DType, pArrayBlock->ItemSize);

        // Use our own data in shared memory
        // TODO: The pArrayBlock->Dimensions cast to npy_intp* is incorrect here if
        // running on a 32-bit system;
        //       need to check the dimension values fit into npy_intp and convert
        //       them if needed. Can use CPP or C++ constexpr-if so we only pay the
        //       cost on 32-bit systems.
        PyArrayObject * pArray =
            AllocateNumpyArrayForData(pArrayBlock->NDim, (npy_intp *)pArrayBlock->Dimensions, dtype, pArrayBlock->ItemSize, data,
                                      pArrayBlock->Flags, (npy_intp *)pArrayBlock->Strides);
        CHECK_MEMORY_ERROR(pArray);

        if (pArray)
        {
            // Make it read only since in shared memory
            ClearWriteableFlag(pArray);
            PyTuple_SetItem(returnTuple, i, (PyObject *)pArray);
        }
        else
        {
            Py_IncRef(Py_None);
            PyTuple_SetItem(returnTuple, i, Py_None);
        }
    }

    PyObject * pDict = GetFileHeaderDict(pFileHeader, NULL);

    // Create and return an SDSFileInfo namedtuple.
    PyObject * returnTupleTuple = Create_SDSFileInfo(pystring, returnTuple, pListName, pDict);
    return returnTupleTuple;
}

//--------------------------------------------
// Wrap arrays
PyObject * ReadFinalStackArrays(SDS_STACK_CALLBACK * pSDSFinalCallback, int64_t arraysWritten,
                                SDS_STACK_CALLBACK_FILES * pSDSFileInfo, SDS_FILTER * pSDSFilter, int64_t fileCount)
{
    //---------- BUILD PYTHON RETURN OBJECTS ---------
    PyObject * returnArrayTuple = PyTuple_New(arraysWritten);
    PyObject * pyListName = PyList_New(arraysWritten);
    PyObject * pyArrayOffset = PyList_New(arraysWritten);

    // not currently used
    // int hasFilter = (pSDSFilter && pSDSFilter->pBoolMask &&
    // pSDSFilter->pFilterInfo);

    for (int t = 0; t < arraysWritten; t++)
    {
        PyObject * item = NULL;

        item = (PyObject *)(pSDSFinalCallback[t].pArrayObject);

        LOGGING("Setting item %d  %p\n", t, item);

        // Return NONE for any arrays with memory issues
        if (item == NULL)
        {
            LOGGING("!! removed item %d -- setting to PyNone\n", t);
            Py_INCREF(Py_None);
            item = Py_None;
        }

        // printf("ref %d  %llu\n", i, item->ob_refcnt);
        PyTuple_SET_ITEM(returnArrayTuple, t, item);

        //================
        // Create an SDSContainerItem for this array.
        PyObject * sdsContainerItem = Create_SDSContainerItem(pSDSFinalCallback[t].ArrayName, pSDSFinalCallback[t].ArrayEnum);

        // pylist_append will add a reference count but setitem will not
        PyList_SET_ITEM(pyListName, t, sdsContainerItem);

        //==============
        PyArrayObject * pOffsetArray = AllocateNumpyArray(1, (npy_intp *)&fileCount, NPY_LONGLONG);
        if (pOffsetArray)
        {
            int64_t * pOffsets = (int64_t *)PyArray_GETPTR1(pOffsetArray, 0);

            LOGGING("arary hasfilter:%d  offsets%lld  %d  name:%s\n", pSDSFilter && pSDSFilter->pBoolMask, fileCount,
                    pSDSFinalCallback[t].ArrayEnum, pSDSFinalCallback[t].ArrayName);

            // copy over our array offsets (skip past first element which is 0)
            memcpy(pOffsets, pSDSFinalCallback[t].pArrayOffsets + 1, fileCount * sizeof(int64_t));

            PyList_SET_ITEM(pyArrayOffset, t, (PyObject *)pOffsetArray);
        }
        else
        {
            PyList_SET_ITEM(pyArrayOffset, t, Py_None);
            Py_INCREF(Py_None);
        }
    }

    PyObject * pyFiles = PyList_New(fileCount);
    PyObject * pyMeta = PyList_New(fileCount);

    for (int f = 0; f < fileCount; f++)
    {
        // printf("filename: %s\n  meta: %s\n ", pSDSFileInfo[f].Filename,
        // pSDSFileInfo[f].MetaData);
        PyList_SET_ITEM(pyFiles, f, PyBytes_FromString(pSDSFileInfo[f].Filename));
        PyList_SET_ITEM(pyMeta, f, PyBytes_FromStringAndSize(pSDSFileInfo[f].MetaData, pSDSFileInfo[f].MetaDataSize));
    }

    // Return the first fileheader to help autodetect the type of file when
    // stacking
    PyObject * pDict{};
    if (fileCount > 0)
    {
        pDict = GetFileHeaderDict(pSDSFileInfo[0].pFileHeader, NULL);
    }
    else
    {
        pDict = Py_None;
        Py_INCREF(Py_None);
    }

    // Create and return an SDSArrayCutoffs namedtuple.
    PyObject * returnTupleTuple = Create_SDSArrayCutoffs(returnArrayTuple, pyListName, pyArrayOffset, pyMeta, pyFiles, pDict);

    return returnTupleTuple;
}

//--------------------------------------------
// Wrap arrays
PyObject * ReadFinalArrays(int64_t arraysWritten, SDSArrayInfo * pArrayInfo)
{
    //---------- BUILD PYTHON RETURN OBJECTS ---------
    PyObject * returnArrayTuple = PyTuple_New(arraysWritten);

    // Decompression
    for (int t = 0; t < arraysWritten; t++)
    {
        PyObject * item = NULL;

        item = (PyObject *)(pArrayInfo[t].pArrayObject);

        LOGGING("Setting item %d  %p\n", t, item);

        // Return NONE for any arrays with memory issues
        if (item == NULL)
        {
            LOGGING("!! removed item %d -- setting to PyNone\n", t);
            Py_INCREF(Py_None);
            item = Py_None;
        }

        // printf("ref %d  %llu\n", i, item->ob_refcnt);
        PyTuple_SET_ITEM(returnArrayTuple, t, item);
    }

    return returnArrayTuple;
}

//-------------------------
// Wrap one file
// May pass in NULL
PyObject * ReadFinalWrap(SDS_FINAL_CALLBACK * pSDSFinalCallback)
{
    if (pSDSFinalCallback == NULL)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    int32_t mode = pSDSFinalCallback->mode;
    int64_t arraysWritten = pSDSFinalCallback->arraysWritten;
    SDS_ARRAY_BLOCK * pArrayBlocks = pSDSFinalCallback->pArrayBlocks;
    SDSArrayInfo * pArrayInfo = pSDSFinalCallback->pArrayInfo;
    SDS_FILE_HEADER * pFileHeader = pSDSFinalCallback->pFileHeader;

    PyObject * pListName = ReadListNamesPython(pSDSFinalCallback->nameData, pFileHeader);
    PyObject * pystring = GetMetaData(pSDSFinalCallback->metaData, pSDSFinalCallback->metaSize);

    // -- STOP EARLY IF THE USER JUST WANTS THE INFORMATION -----------------
    if (mode == COMPRESSION_MODE_INFO)
    {
        LOGGING("Returning just the info\n");
        PyObject * returnObject = GetSDSFileInfo(pListName, pystring, arraysWritten, pArrayBlocks, pFileHeader, pSDSFinalCallback);
        return returnObject;
    }

    LOGGING("Building return object of arrays %lld  %p\n", arraysWritten, pArrayInfo);

    //---------- BUILD PYTHON RETURN OBJECTS ---------
    PyObject * returnArrayTuple = ReadFinalArrays(arraysWritten, pArrayInfo);
    PyObject * pDict = GetFileHeaderDict(pFileHeader, pSDSFinalCallback);

    // Create and return an SDSFileInfo namedtuple.
    PyObject * returnWrap = Create_SDSFileInfo(pystring, returnArrayTuple, pListName, pDict);

    // Soon after this returns, files will be closed, memory deallocated
    return returnWrap;
}

//----------------------------------------
// CALLBACK2 - can wrap more than one file
// finalCount is how many info sections to return
// if there are sections inside a single file, the finalCount > 1
void * ReadFinal(SDS_FINAL_CALLBACK * pSDSFinalCallback, int64_t finalCount)
{
    PyObject * returnItem = NULL;

    if (finalCount <= 0)
    {
        // No valid files found, just return None
        Py_INCREF(Py_None);
        returnItem = Py_None;
    }
    else
    {
        // Return a list of all the data
        returnItem = PyList_New(finalCount);

        // Wrap the item for every file
        for (int64_t file = 0; file < finalCount; file++)
        {
            PyObject * item = ReadFinalWrap(&pSDSFinalCallback[file]);

            // TODO: Check if 'item' is nullptr; if so, need to set it to Py_None and
            // increment Py_None refcount,
            //       so we don't potentially end up segfaulting later.

            // Steals a reference
            PyList_SET_ITEM(returnItem, file, item);
        }
    }
    LOGGING("End ReadFinal %p.  finalCount %lld\n", returnItem, finalCount);
    return returnItem;
}

//----------------------------------------
// CALLBACK2 - all files were stacked into one column
void * ReadFinalStack(SDS_STACK_CALLBACK * pSDSFinalCallback, int64_t finalCount, SDS_STACK_CALLBACK_FILES * pSDSFileInfo,
                      SDS_FILTER * pSDSFilter, int64_t fileCount)
{
    PyObject * returnItem = NULL;

    if (finalCount <= 0)
    {
        Py_INCREF(Py_None);
        returnItem = Py_None;
    }
    else
    {
        PyObject * returnArrayTuple = ReadFinalStackArrays(pSDSFinalCallback, finalCount, pSDSFileInfo, pSDSFilter, fileCount);

        returnItem = returnArrayTuple;
    }

    LOGGING("End ReadFinalStack %p\n", returnItem);
    return returnItem;
}

//--------------------------------------
// free string with WORKSPACE_FREE
void CopyUnicodeString(PyObject * pUnicode, char ** returnString, int64_t * returnSize)
{
    PyObject * temp2 = PyUnicode_AsASCIIString(pUnicode);
    if (temp2 != NULL)
    {
        *returnString = PyBytes_AsString(temp2);
        *returnSize = strlen(*returnString);
        char * pNewString = (char *)WORKSPACE_ALLOC(*returnSize + 1);
        memcpy(pNewString, *returnString, *returnSize);
        pNewString[*returnSize] = 0;

        *returnString = pNewString;
        // Release reference
        Py_DecRef(temp2);
    }
    else
    {
        *returnString = NULL;
        *returnSize = strlen(*returnString);
    }
}

//----------------------------------------------------
// check for "section=" (NOT the same as sections)
// must be a unicode string
// returns 0 if no section
int64_t GetStringFromDict(const char * dictstring, PyObject * kwargs, char ** returnString, int64_t * returnSize)
{
    if (! kwargs)
        return 0;

    PyObject * sectionObject = PyDict_GetItemString(kwargs, dictstring);

    if (sectionObject && PyUnicode_Check(sectionObject))
    {
        CopyUnicodeString(sectionObject, returnString, returnSize);
        return *returnSize;
    }
    else
    {
        *returnString = NULL;
        *returnSize = 0;
    }

    return 0;
}

//----------------------------------------------------
// check for "sections=" (NOT the same as section)
// must be a list of strings
// returns NULL if no list of strings found
SDS_STRING_LIST * GetSectionsName(PyObject * kwargs)
{
    if (! kwargs)
        return NULL;

    PyObject * sectionNameObject = PyDict_GetItemString(kwargs, "sections");

    if (sectionNameObject && PyList_Check(sectionNameObject))
    {
        return StringListToVector(sectionNameObject);
    }

    return NULL;
}

//----------------------------------------------------
// check for "bandsize="
// must be an INT
// returns 0 if no bandsize
int64_t GetBandSize(PyObject * kwargs)
{
    if (! kwargs)
        return 0;

    PyObject * bandsizeObject = PyDict_GetItemString(kwargs, "bandsize");

    if (bandsizeObject && PyLong_Check(bandsizeObject))
    {
        int64_t result = PyLong_AsLongLong(bandsizeObject);

        // minimum bandsize is 10K
        if (result < 0)
            result = 0;
        return result;
    }

    return 0;
}

//----------------------------------------------------
// check for "folders="
// must be a list of strings
// returns NULL of no list of strings found
SDS_STRING_LIST * GetFoldersName(PyObject * kwargs)
{
    if (! kwargs)
        return NULL;

    PyObject * folderNameObject = PyDict_GetItemString(kwargs, "folders");

    if (folderNameObject && PyList_Check(folderNameObject))
    {
        return StringListToVector(folderNameObject);
    }

    return NULL;
}

//----------------------------------------------------
// Arg1: BYTES - filename (UNICODE not allowed)
// Arg2: BYTES - metadata
// Arg3: Pass in list of numpy arrays
// Arg4: Pass in list of tuples (arrayname/int)
// Arg5: compType
// Arg6: compression level
// Arg7: <optional> sharename
//
// Kwargs
// folders=
// bandsize=
//
// Returns: None
//
// File is created with path
// NOTE: If sharename is specified, NO file is writtern
// NOTE: If the user wants both the sharename and the file, this API must be
// called twice (with and without sharename)

PyObject * CompressFile(PyObject * self, PyObject * args, PyObject * kwargs)
{
    PyObject * inListArrays = NULL;
    PyListObject * inListNames = NULL;

    const char * fileName;
    uint32_t fileNameSize;

    const char * metaData;
    uint32_t metaDataSize;

    const char * shareName = NULL;
    uint32_t shareNameSize = 0;

    int32_t mode = COMPRESSION_MODE_COMPRESS_FILE;
    int32_t compType = COMPRESSION_TYPE_ZSTD;
    int32_t level = ZSTD_CLEVEL_DEFAULT;
    int32_t fileType = 0;

    if (! PyArg_ParseTuple(args, "y#y#OO!iii|y#", &fileName, &fileNameSize, &metaData, &metaDataSize, &inListArrays, &PyList_Type,
                           &inListNames, &compType, &level, &fileType, &shareName, &shareNameSize))
    {
        return NULL;
    }

    LOGGING("In CompressFile %s\n", fileName);

    // Check for kwargs: folders, bandsize, section
    //
    SDS_STRING_LIST * folderName = GetFoldersName(kwargs);
    int64_t bandSize = GetBandSize(kwargs);

    char * pSectionName = NULL;
    int64_t sectionSize = 0;

    GetStringFromDict("section", kwargs, &pSectionName, &sectionSize);

    if (sectionSize >= SDS_MAX_SECTIONNAME)
    {
        PyErr_Format(PyExc_ValueError, "section name too large: %lld", sectionSize);
        return NULL;
    }

    // printf("section name: %s\n", sectionName);

    ClearErrors();

    // Handle list of names ------------------------------------------------------
    int64_t listNameCount = PyList_GET_SIZE(inListNames);

    LOGGING("Name count is %d\n", (int)listNameCount);

    // alloc worst case scenario
    char * pListNames = (char *)WORKSPACE_ALLOC((SDS_MAX_FILENAME * listNameCount) + 8);
    if (! pListNames)
    {
        return NULL;
    }

    // Process list of names tuples
    int64_t listNameSize = BuildListInfo(inListNames, pListNames);

    // Handle list of numpy arrays -----------------------------------
    int64_t totalItemSize = 0;
    int64_t arrayCount = 0;

    ArrayInfo * aInfo = BuildArrayInfo(inListArrays, &arrayCount, &totalItemSize, false);

    // CHECK FOR ERRORS
    if (aInfo)
    {
        SDS_WRITE_CALLBACKS SDSWriteCallbacks;

        SDSWriteCallbacks.BeginAllowThreads = BeginAllowThreads;
        SDSWriteCallbacks.EndAllowThreads = EndAllowThreads;

        SDS_WRITE_INFO SDSWriteInfo;
        SDSWriteInfo.aInfo = (SDSArrayInfo *)WORKSPACE_ALLOC(sizeof(SDSArrayInfo) * arrayCount);

        //============================================
        // Convert from ArrayInfo* to SDSArrayInfo*
        //
        SDSArrayInfo * pDest = SDSWriteInfo.aInfo;
        ArrayInfo * pSrc = aInfo;

        for (int64_t i = 0; i < arrayCount; i++)
        {
            pDest->ArrayLength = pSrc->ArrayLength;
            pDest->ItemSize = (int32_t)pSrc->ItemSize;
            pDest->pArrayObject = pSrc->pObject; // We do not need this..
            pDest->NumBytes = pSrc->NumBytes;
            pDest->NumpyDType = pSrc->NumpyDType;
            pDest->pData = pSrc->pData;

            int32_t ndim = pSrc->NDim;
            if (ndim > SDS_MAX_DIMS)
            {
                printf("!!!SDS: array dimensions too high: %d\n", ndim);
                ndim = SDS_MAX_DIMS;
            }
            // if (ndim < 1) {
            //   printf("!!!SDS: array dimensions too low: %d\n", ndim);
            //   ndim = 1;
            //}
            pDest->NDim = ndim;

            for (int dim_idx = 0; dim_idx < SDS_MAX_DIMS; dim_idx++)
            {
                pDest->Dimensions[dim_idx] = 0;
                pDest->Strides[dim_idx] = 0;
            }

            // None can be passed in for an array now
            if (pSrc->pObject)
            {
                npy_intp * pdims = ((PyArrayObject_fields *)pSrc->pObject)->dimensions;
                npy_intp * pstrides = ((PyArrayObject_fields *)pSrc->pObject)->strides;

                for (int dim_idx = 0; dim_idx < ndim; dim_idx++)
                {
                    pDest->Dimensions[dim_idx] = pdims[dim_idx];
                    pDest->Strides[dim_idx] = pstrides[dim_idx];
                }

                pDest->Flags = PyArray_FLAGS((PyArrayObject *)(pSrc->pObject));

                // make sure C or F contiguous
                if (! (pDest->Flags & (SDS_ARRAY_C_CONTIGUOUS | SDS_ARRAY_F_CONTIGUOUS)))
                {
                    // pSrc->pObject = PyArray_FromAny(pSrc->pObject, NULL, 0, 0,
                    // NPY_ARRAY_ENSURECOPY, NULL);

                    printf("!!!SDS: array is not C or F contiguous: %d\n", pDest->Flags);
                }
            }
            else
            {
                pDest->Flags = 0;
            }

            pDest++;
            pSrc++;
        }

        SDSWriteInfo.arrayCount = arrayCount;

        // meta information
        SDSWriteInfo.metaData = metaData;
        SDSWriteInfo.metaDataSize = metaDataSize;

        // names of arrays information
        SDSWriteInfo.pListNames = pListNames;
        SDSWriteInfo.listNameSize = listNameSize;
        SDSWriteInfo.listNameCount = listNameCount;

        // compressed or uncompressed
        SDSWriteInfo.mode = mode;
        SDSWriteInfo.compType = compType;
        SDSWriteInfo.level = level;

        // NEED TO SEND in
        SDSWriteInfo.sdsFileType = fileType;
        SDSWriteInfo.sdsAuthorId = SDS_AUTHOR_ID_PYTHON;

        // section and append information
        SDSWriteInfo.appendFileHeadersMode = false;
        SDSWriteInfo.appendRowsMode = false;
        SDSWriteInfo.appendColumnsMode = false;
        SDSWriteInfo.bandSize = bandSize;

        SDSWriteInfo.sectionName = NULL;
        SDSWriteInfo.sectionNameSize = 0;

        // if the kwarg section exists,
        if (pSectionName)
        {
            SDSWriteInfo.appendRowsMode = true;
            SDSWriteInfo.sectionName = pSectionName;
            SDSWriteInfo.sectionNameSize = sectionSize;
        }

        bool result = SDSWriteFile(fileName,
                                   shareName, // can be NULL
                                   folderName, &SDSWriteInfo, &SDSWriteCallbacks);

        // FREE workspace allocations
        WORKSPACE_FREE(SDSWriteInfo.aInfo);
        FreeArrayInfo(aInfo);
    }

    WORKSPACE_FREE(pListNames);

    if (pSectionName)
    {
        WORKSPACE_FREE(pSectionName);
    }
    if (folderName)
    {
        DestroyStringList(folderName);
    }

    // If there are errors, return NULL
    if (CheckErrors())
    {
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

//==================================
// Called back when reading in data
void AllocateArrayCallback(SDS_ALLOCATE_ARRAY * pAllocateArray)
{
    SDSArrayInfo * pDestInfo = pAllocateArray->pDestInfo;

    int ndim = pAllocateArray->ndim;

    const char * pArrayName = pAllocateArray->pArrayName;

    LOGGING("Allocate array name: %s\n", pArrayName);

    pDestInfo->pArrayObject = NULL;

    // if (pAllocateArray->data)
    pDestInfo->pData = NULL;

    // if (IsIncluded((PyObject*)pAllocateArray->pInclusionList, pArrayName)) {
    if (true)
    {
        // If we have no dimensions, do not allocate
        if (ndim)
        {
            int64_t * dims = pAllocateArray->dims;
            int64_t * strides = pAllocateArray->strides;

            if (pAllocateArray->data)
            {
                LOGGING("Shared memory was set to %p\n", pAllocateArray->data);
            }

            LOGGING(
                "Allocating ndim:%d  dim0:%lld  type:%d  itemsize:%lld  flags:%d "
                " strde0:%lld",
                pAllocateArray->ndim, dims[0], pAllocateArray->numpyType, pAllocateArray->itemsize, pAllocateArray->numpyFlags,
                strides[0]);

            // Use different array-creation functions based on whether we're
            // allocating new (or reclaiming recycled) memory, or we have some
            // existing backing memory (e.g. shared memory) and we're creating the new
            // array object to just wrap that memory.
            if (pAllocateArray->data)
            {
                pDestInfo->pArrayObject = AllocateNumpyArrayForData(pAllocateArray->ndim, (npy_intp *)dims,
                                                                    pAllocateArray->numpyType, pAllocateArray->itemsize,
                                                                    pAllocateArray->data, // set for shared memory?
                                                                    pAllocateArray->numpyFlags, (npy_intp *)strides);

                // TODO: Check whether pDestInfo->pArrayObject is nullptr here; if it
                // is, something went wrong
                //       with the allocation and now's the chance to gracefully handle
                //       it; otherwise we'll segfault later once we try to read from
                //       that memory.
            }
            else
            {
                pDestInfo->pArrayObject =
                    AllocateNumpyArray(pAllocateArray->ndim, (npy_intp *)dims, pAllocateArray->numpyType, pAllocateArray->itemsize,
                                       pAllocateArray->numpyFlags & NPY_ARRAY_F_CONTIGUOUS, (npy_intp *)strides);
            }

            // check for successful allocation
            if (pDestInfo->pArrayObject)
            {
                if (pAllocateArray->data)
                {
                    // Flip off writeable flag at low level if came from shared memory
                    ClearWriteableFlag((PyArrayObject *)(pDestInfo->pArrayObject));
                }

                // Fill in pDestInfo
                pDestInfo->pData = (char *)PyArray_GETPTR1((PyArrayObject *)(pDestInfo->pArrayObject), 0);
            }
        }
    }
}

//----------------------------------------------------
// Return the maskLength and pBooleanMask
void GetFilters(PyObject * kwargs, SDS_READ_CALLBACKS * pRCB)
{
    pRCB->Filter.BoolMaskLength = 0;
    pRCB->Filter.pBoolMask = NULL;
    pRCB->Filter.BoolMaskTrueCount = 0;
    pRCB->Filter.pFilterInfo = NULL;

    pRCB->MustExist = false;

    if (kwargs)
    {
        //-------------------
        // This DOES NOT change the ref count (borrowed reference)
        // PyObject* filterItem = PyDict_GetItemString(kwargs, "filter");
        //
        // if (filterItem && PyArray_Check(filterItem)) {
        //   if (PyArray_TYPE((PyArrayObject*)filterItem) == NPY_INT ||
        //   PyArray_TYPE((PyArrayObject*)filterItem) == NPY_LONG) {
        //      pRCB->Filter.FancyLength = ArrayLength((PyArrayObject*)filterItem);
        //      pRCB->Filter.pFancyMask =
        //      (INT32*)PyArray_GETPTR1((PyArrayObject*)filterItem, 0);
        //      LOGGING("Found valid filter.  length: %lld\n",
        //      pRCB->Filter.FancyLength);
        //   }
        //}

        PyObject * maskItem = PyDict_GetItemString(kwargs, "mask");

        if (maskItem && PyArray_Check(maskItem))
        {
            if (PyArray_TYPE((PyArrayObject *)maskItem) == NPY_BOOL)
            {
                pRCB->Filter.BoolMaskLength = ArrayLength((PyArrayObject *)maskItem);
                pRCB->Filter.pBoolMask = (bool *)PyArray_GETPTR1((PyArrayObject *)maskItem, 0);
                // Needed
                pRCB->Filter.BoolMaskTrueCount = SumBooleanMask((int8_t *)pRCB->Filter.pBoolMask, pRCB->Filter.BoolMaskLength);
                LOGGING("Found valid mask.  length: %lld\n", pRCB->Filter.BoolMaskLength);
            }
        }

        PyObject * mustexist = PyDict_GetItemString(kwargs, "mustexist");
        if (mustexist && PyBool_Check(mustexist))
        {
            if (mustexist == Py_True)
            {
                pRCB->MustExist = true;
            }
        }
    }
}

//----------------------------------------------------
// Python Interface
// Arg1: BYTES - filename (UNICODE not allowed)
// Arg2: mode: integer defaults to COMPRESSION_MODE_DECOMPRESS_FILE, also
// allowed: COMPRESSION_MODE_INFO Arg3: <optional> shared memory prefix (UNICODE
// not allowed)
//
// Kwargs
// include=
// folder=
//
// Returns tuple <metadata, list of arrays compressed, list of array
// names/enums>
//
PyObject * DecompressFile(PyObject * self, PyObject * args, PyObject * kwargs)
{
    const char * fileName;
    uint32_t fileNameSize;

    const char * shareName = NULL;
    SDS_STRING_LIST * folderName = NULL;
    SDS_STRING_LIST * sectionsName = NULL;

    uint32_t shareNameSize = 0;

    int32_t mode = COMPRESSION_MODE_DECOMPRESS_FILE;

    // UNSUPPORTED: PyObject* includeDict = NULL;

    if (! PyArg_ParseTuple(args, "y#i|y#", &fileName, &fileNameSize, &mode, &shareName, &shareNameSize))
    {
        return NULL;
    }

    if (kwargs && PyDict_Check(kwargs))
    {
        // Borrowed reference
        // Returns NULL if key not present
        PyObject * includedItem = PyDict_GetItemString(kwargs, "include");

        if (includedItem && PyDict_Check(includedItem))
        {
            LOGGING("Found valid inclusion dict\n");
            // UNSUPPORTED: includeDict = includedItem;
        }
        else
        {
            // LOGGING("did not like dict!  %p\n", includedItem);
        }

        folderName = GetFoldersName(kwargs);
        sectionsName = GetSectionsName(kwargs);
    }

    //==============================================
    // Build callback table
    SDS_READ_CALLBACKS sdsRCB;

    sdsRCB.ReadFinalCallback = ReadFinal;
    sdsRCB.StackFinalCallback = ReadFinalStack;
    sdsRCB.ReadMemoryCallback = ReadFromSharedMemory;
    sdsRCB.AllocateArrayCallback = AllocateArrayCallback;
    sdsRCB.BeginAllowThreads = BeginAllowThreads;
    sdsRCB.EndAllowThreads = EndAllowThreads;
    sdsRCB.pInclusionList = NULL; // NO LONGER SUPPORTED includeDict;
    sdsRCB.pExclusionList = NULL;
    sdsRCB.pFolderInclusionList = NULL;

    // new for filtering
    GetFilters(kwargs, &sdsRCB);

    // new for boolean mask
    // GetFilters(kwargs, &sdsRCB.pBooleanMask, &sdsRCB.MaskLength);

    SDS_READ_INFO sdsRI;

    sdsRI.mode = mode;

    //==============================================
    void * result = SDSReadFile(fileName, shareName, folderName, sectionsName, &sdsRI, &sdsRCB);

    if (folderName)
    {
        DestroyStringList(folderName);
    }

    if (sectionsName)
    {
        DestroyStringList(sectionsName);
    }

    // If there are errors, return NULL
    if (! result && CheckErrors())
    {
        return NULL;
    }

    if (! result)
    {
        PyErr_Format(PyExc_ValueError, "NULL is returned from SDSReadFile but no error string was found");
    }
    return (PyObject *)result;
}

//----------------------------------------------------
// Arg1: List of [filenames]
// Arg2: mode: integer defaults to COMPRESSION_MODE_DECOMPRESS_FILE, also
// allowed: COMPRESSION_MODE_INFO multimode = stacking or reading multiple files
// Returns tuple <metadata, list of arrays compressed, list of array
// names/enums>
//
PyObject * InternalDecompressFiles(PyObject * self, PyObject * args, PyObject * kwargs, int multiMode)
{
    PyObject * listFilenames;
    PyObject * includeDict = NULL;
    void * result = NULL;
    double reserveSpace = 0.0;

    int32_t mode = COMPRESSION_MODE_DECOMPRESS_FILE;

    if (! PyArg_ParseTuple(args, "O!|i", &PyList_Type, &listFilenames, &mode))
    {
        return NULL;
    }

    //--------------------------------------------------
    // Check if we are flipping into info modes
    //
    if (multiMode == SDS_MULTI_MODE_READ_MANY && mode != COMPRESSION_MODE_DECOMPRESS_FILE)
    {
        multiMode = SDS_MULTI_MODE_READ_MANY_INFO;
    }
    if (multiMode == SDS_MULTI_MODE_STACK_MANY && mode != COMPRESSION_MODE_DECOMPRESS_FILE)
    {
        multiMode = SDS_MULTI_MODE_STACK_MANY_INFO;
    }

    char * pOutputName = NULL;
    int64_t outputSize = 0;

    SDS_STRING_LIST * pInclusionList = NULL;
    SDS_STRING_LIST * pFolderList = NULL;
    SDS_STRING_LIST * pSectionsList = NULL;
    int64_t maskLength = 0;
    bool * pBooleanMask = NULL;

    if (multiMode == SDS_MULTI_MODE_CONCAT_MANY)
    {
        GetStringFromDict("output", kwargs, &pOutputName, &outputSize);
        if (! pOutputName)
        {
            PyErr_Format(PyExc_ValueError, "The output= must be a filename when concatenating files");
        }
    }

    if (kwargs && PyDict_Check(kwargs))
    {
        // Borrowed reference
        // Returns NULL if key not present
        PyObject * includedItem = PyDict_GetItemString(kwargs, "include");

        if (includedItem && PyList_Check(includedItem))
        {
            LOGGING("Found valid inclusion list\n");
            pInclusionList = StringListToVector(includedItem);
        }
        else
        {
            // LOGGING("did not like dict!  %p\n", includedItem);
        }
        //-------------------

        PyObject * reserveItem = PyDict_GetItemString(kwargs, "reserve");

        if (reserveItem && PyFloat_Check(reserveItem))
        {
            reserveSpace = PyFloat_AsDouble(reserveItem);
            LOGGING("Found valid reserve space as %lf\n", reserveSpace);
        }

        pFolderList = GetFoldersName(kwargs);
        pSectionsList = GetSectionsName(kwargs);
    }

    SDS_STRING_LIST * pFilenames = StringListToVector(listFilenames);
    int64_t fileCount = pFilenames->size();

    LOGGING("InternalDecompress filecount: %lld  mode: %d", fileCount, multiMode);
    if (fileCount)
    {
        SDS_MULTI_READ * pMultiRead = (SDS_MULTI_READ *)WORKSPACE_ALLOC(sizeof(SDS_MULTI_READ) * fileCount);

        if (pMultiRead)
        {
            int64_t i = 0;

            // loop over all filenames and build callback
            for (const char * filename : *pFilenames)
            {
                memset(&pMultiRead[i].FinalCallback, 0, sizeof(SDS_FINAL_CALLBACK));
                pMultiRead[i].pFileName = filename;
                pMultiRead[i].Index = i;
                i++;
            }

            //==============================================
            // Build callback table
            SDS_READ_CALLBACKS sdsRCB;

            sdsRCB.ReadFinalCallback = ReadFinal;

            // called only if in stack mode
            sdsRCB.StackFinalCallback = ReadFinalStack;

            // called only when reading from shared memory
            sdsRCB.ReadMemoryCallback = ReadFromSharedMemory;
            sdsRCB.AllocateArrayCallback = AllocateArrayCallback;
            sdsRCB.BeginAllowThreads = BeginAllowThreads;
            sdsRCB.EndAllowThreads = EndAllowThreads;
            sdsRCB.pInclusionList = pInclusionList;
            sdsRCB.pExclusionList = NULL;
            sdsRCB.pFolderInclusionList = pFolderList;
            sdsRCB.ReserveSpace = reserveSpace;
            sdsRCB.strOutputFilename = pOutputName;

            // new for filtering
            GetFilters(kwargs, &sdsRCB);

            //==============================================
            // may return NULL if not all files existed
            result = SDSReadManyFiles(pMultiRead, pInclusionList, pFolderList, pSectionsList, fileCount, multiMode, &sdsRCB);

            WORKSPACE_FREE(pMultiRead);
        }
    }

    // Free what might have been allocated
    if (pOutputName)
    {
        WORKSPACE_FREE(pOutputName);
    }

    DestroyStringList(pFilenames);

    if (pInclusionList)
    {
        DestroyStringList(pInclusionList);
    }
    if (pFolderList)
    {
        DestroyStringList(pFolderList);
    }

    LOGGING("Multistack after destroy string list\n");

    bool isThereAnError = CheckErrors();

    // If there are errors, return NULL
    if (! result && isThereAnError)
    {
        return NULL;
    }

    if (isThereAnError && fileCount == 1)
    {
        // result is good, but there is an error.  decrement the result so we can
        // get rid of it
        Py_DecRef((PyObject *)result);
        return NULL;
    }

    // We might have a partial error if we get here
    if (! result)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    LOGGING("Multistack returning a good result %p\n", result);
    return (PyObject *)result;
}

//----------------------------------------------------
// Arg1: List of [filenames]
// Arg2: mode: integer defaults to COMPRESSION_MODE_DECOMPRESS_FILE, also
// allowed: COMPRESSION_MODE_INFO Arg3: <optional> shared memory prefix (UNICODE
// not allowed) Returns tuple <metadata, list of arrays compressed, list of
// array names/enums>
//
PyObject * MultiDecompressFiles(PyObject * self, PyObject * args, PyObject * kwargs)
{
    return InternalDecompressFiles(self, args, kwargs, SDS_MULTI_MODE_READ_MANY);
}

PyObject * MultiStackFiles(PyObject * self, PyObject * args, PyObject * kwargs)
{
    return InternalDecompressFiles(self, args, kwargs, SDS_MULTI_MODE_STACK_MANY);
}

PyObject * MultiPossiblyStackFiles(PyObject * self, PyObject * args, PyObject * kwargs)
{
    return InternalDecompressFiles(self, args, kwargs, SDS_MULTI_MODE_STACK_OR_READMANY);
}

//----------------------------------------------------
// Arg1: List of [filenames]
// kwarg: output = string full path of file
PyObject * MultiConcatFiles(PyObject * self, PyObject * args, PyObject * kwargs)
{
    return InternalDecompressFiles(self, args, kwargs, SDS_MULTI_MODE_CONCAT_MANY);
}

//----------------------------------------------------
// Arg1: BYTES - filename
// Arg2: tuple of byte strings (UNICODE not allowed) which are replacements
//
PyObject * SetLustreGateway(PyObject * self, PyObject * args)
{
    const char * fileName;
    uint32_t fileNameSize;

    PyObject * tuple;

    if (! PyArg_ParseTuple(args, "y#O!", &fileName, &fileNameSize, &PyTuple_Type, &tuple))
    {
        return NULL;
    }

    // printf("hint: %s\n", fileName);

    g_gatewaylist.clear();
    int64_t tupleLength = PyTuple_GET_SIZE(tuple);
    g_gatewaylist.reserve(tupleLength);

    for (int64_t i = 0; i < tupleLength; i++)
    {
        const char * gateway = PyBytes_AsString(PyTuple_GET_ITEM(tuple, i));
        // printf("gw: %s\n", gateway);
        g_gatewaylist.push_back(std::string(gateway));
    }

    Py_INCREF(Py_None);
    return Py_None;
}
