// Python Interface to SDS File

#include <stdlib.h>
#include <Python.h>
#include "bytesobject.h"
#include "Compress.h"
#include "SDSFile.h"
#include "SharedMemory.h"
#include "FileReadWrite.h"

#include "MultiKey.h"
#include "MathWorker.h"

// For sumbooleanmask
#include "Convert.h"

#ifndef OUT
#define OUT
#endif

#define LOGGING(...)
//#define LOGGING printf

// Call to clear any previous errors
static void ClearErrors() {
   g_lastexception = 0;
}

// check to see if any errors were recorded
// Returns TRUE if there was an error
static BOOL CheckErrors() {

   if (g_lastexception) {
      PyErr_Format(PyExc_ValueError, g_errorbuffer);
      return TRUE;
   }
   return FALSE;
}


//--------------------------------------------------
//
void* BeginAllowThreads() {
   return PyEval_SaveThread();
}

//--------------------------------------------------
//
void EndAllowThreads(void* saveObject) {
   return PyEval_RestoreThread((PyThreadState*)saveObject);
}

//--------------------------------------------------
//
static char*   GetMemoryOffset(char* BaseAddress, INT64 offset) {
   return (BaseAddress + offset);
}

//--------------------------------------------------
//
static SDS_ARRAY_BLOCK*  GetArrayBlock(char* baseOffset, INT64 arrayNum) {
   SDS_ARRAY_BLOCK* pArrayBlock = (SDS_ARRAY_BLOCK*)GetMemoryOffset(baseOffset, ((SDS_FILE_HEADER*)baseOffset)->ArrayBlockOffset);
   return &pArrayBlock[arrayNum];
}


void
DestroyStringList(SDS_STRING_LIST* pStringList) {
   for (const char* filename : *pStringList) {
      WORKSPACE_FREE((void*)filename);
   }
   delete pStringList;
}


//----------------------------------
// caller must delete STRING_LIST
// handles both BYTE and UNICODE
SDS_STRING_LIST*
StringListToVector(PyObject* listFilenames) {

   SDS_STRING_LIST* returnList = new SDS_STRING_LIST;

   if (PyList_Check(listFilenames)) {
      INT64 filenameCount = PyList_GET_SIZE(listFilenames);
      returnList->reserve(filenameCount);

      for (INT64 i = 0; i < filenameCount; i++) {

         PyObject* pBytes = PyList_GET_ITEM(listFilenames, i);
         const char *fileName = NULL;
         if (PyBytes_Check(pBytes)) {
            fileName = PyBytes_AsString(pBytes);

            INT64 strSize = strlen(fileName) ;
            char* pNewString = (char*)WORKSPACE_ALLOC(strSize+ 1);
            memcpy(pNewString, fileName, strSize);
            pNewString[strSize] = 0;

            // add to our own list
            returnList->push_back(pNewString);
         }
         else
            if (PyUnicode_Check(pBytes)) {
               PyObject* temp2 = PyUnicode_AsASCIIString(pBytes);
               if (temp2 != NULL) {
                  fileName = PyBytes_AsString(temp2);
                  INT64 strSize = strlen(fileName);
                  char* pNewString = (char*)WORKSPACE_ALLOC(strSize + 1);
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
//         ASCIIZ strings follow by UINT8 enum (string1, 0, enum1, string2, 0, enum2, etc.)
// Returns: Length of data in pListNames
//
INT64 BuildListInfo(PyListObject *inListNames, OUT char* pListNames) {
   INT64 listNameCount = PyList_GET_SIZE(inListNames);
   char* pStart = pListNames;

   for (int i = 0; i < listNameCount; i++) {
      PyObject* pTuple = PyList_GET_ITEM(inListNames, i);

      if (pTuple && PyTuple_Check(pTuple)) {

         PyObject* pBytes = PyTuple_GET_ITEM(pTuple, 0);

         if (PyBytes_Check(pBytes)) {
            int overflow = 0;
            INT64 value = PyLong_AsLongLongAndOverflow(PyTuple_GET_ITEM(pTuple, 1), &overflow);
            INT64 strSize = Py_SIZE(pBytes);

            char* pName = PyBytes_AS_STRING(pBytes);
            LOGGING("Name is %s -- size %d  value %d\n", pName, (int)strSize, (int)value);

            while ((*pListNames++ = *pName++));

            // Store the 1 byte enum type
            *pListNames++ = (UINT8)value;
         }
         else {
            printf("!!internal error processing, check that list is in bytes instead of unicode\n");
         }

      }
      else {
         printf("!!internal error2 processing, is it a list of tuples?\n");

      }
   }

   return pListNames - pStart;
}


//--------------------------------------------
// Returns a list of string (Column names)
//
// Entry: 
// arg1: pointer to string, null terminated, followed by UINT8 enum value
// arg2: how many names
// arg3: the size of pArrayNames (all of the names)

PyObject* MakeListNames(const char* pArrayNames, INT64 nameBlockCount, INT64 nameSize) {
   const char* nameData = pArrayNames;
   PyObject* pyListName = PyList_New(nameBlockCount);

   INT64 curPos = 0;
   // for every name
   while (nameBlockCount) {
      nameBlockCount--;
      const char* pStart = pArrayNames;

      // skip to end (search for 0 terminating char)
      while (*pArrayNames++);

      // get the enum
      UINT8 value = *pArrayNames++;

      PyObject* pTuple = PyTuple_New(2);
      LOGGING("makelist file name is %s, %lld\n", pStart, nameBlockCount);

      PyTuple_SetItem(pTuple, 0, PyBytes_FromString(pStart));
      PyTuple_SetItem(pTuple, 1, PyLong_FromLong((long)value));

      // pylist_append will add a reference count but setitem will not
      PyList_SetItem(pyListName, curPos, pTuple);

      curPos++;

      // If we ran too far, break
      if ((pArrayNames - nameData) >= nameSize) break;
   }
   return pyListName;
}


//------------------------------------------------------
// Input: file already openeded
//
// returns list of names or empty list
PyObject* ReadListNamesPython(char* nameData, SDS_FILE_HEADER *pFileHeader) {
   LOGGING("ReadListNames %p\n", nameData);

   PyObject* pListName = NULL;
   if (nameData) {
      // return list of strings
      // GIL must be held to create the list
      INT64 nameSize = pFileHeader->NameBlockSize;

      pListName = MakeListNames(nameData, pFileHeader->NameBlockCount, nameSize);
   }
   else {
      // make empty list
      LOGGING("empty list!\n");
      pListName = PyList_New(0);
   }
   return pListName;

}

//---------------------------------------------------------
// Linux: long = 64 bits
// Windows: long = 32 bits
static int FixupDType(int dtype, INT64 itemsize) {

   if (dtype == NPY_LONG) {
      // types 7 and 8 are ambiguous because of different compilers
      if (itemsize == 4) {
         dtype = NPY_INT;
      }
      else {
         dtype = NPY_LONGLONG;
      }
   }

   if (dtype == NPY_ULONG) {
      // types 7 and 8 are ambiguous
      if (itemsize == 4) {
         dtype = NPY_UINT;
      }
      else {
         dtype = NPY_ULONGLONG;
      }
   }
   return dtype;
}

//-----------------------------------------
// Return empty string on failure
PyObject*  GetMetaData(char* metaData, INT64 metaSize) {
   if (metaData) {
      // caller wants a pystring
      // this will make a copy of the data
      return PyBytes_FromStringAndSize(metaData, metaSize);
   }
   printf("Possible error -returning null on metadata\n");
   const char* const temp = "{}";
   return PyBytes_FromStringAndSize(temp, 2);
}

int SetStringLong(PyObject* pDict, const char* strkey, long long value) {
   return PyDict_SetItemString(pDict, strkey, (PyObject*)PyLong_FromLongLong(value));
}

//----------------------------------------------------
//
PyObject* GetFileHeaderDict(
   SDS_FILE_HEADER* pFileHeader,
   SDS_FINAL_CALLBACK* pSDSFinalCallback) {
   PyObject* pDict = PyDict_New();
   if (pFileHeader) {
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
      if (pSDSFinalCallback && pSDSFinalCallback->strFileName) {
         PyDict_SetItemString(pDict, "Filename", PyUnicode_FromString(pSDSFinalCallback->strFileName));
         SDSSectionName* pSection = pSDSFinalCallback->pSectionName;

         // The sectionoffset is used to reorder
         if (pSection)  SetStringLong(pDict, "SectionOffset", pSection->SectionOffset);

         // Check if we have sections
         if (pSection && pSection->SectionCount) {
            // Create a new list of strings containing section names
            PyObject* pSectionList = PyList_New(pSection->SectionCount);
            PyObject* pSectionListOffset = PyList_New(pSection->SectionCount);
            PyDict_SetItemString(pDict, "Sections", pSectionList);
            PyDict_SetItemString(pDict, "SectionOffsets", pSectionListOffset);

            for (INT64 n = 0; n < pSection->SectionCount; n++) {
               PyList_SetItem(pSectionList, n, PyUnicode_FromString(pSection->pSectionNames[n]));
               PyList_SetItem(pSectionListOffset, n, (PyObject*)PyLong_FromLongLong(pSection->pSectionOffsets[n]));
            }
         }
      }
   }
   return pDict;
}


//----------------------------------------------------
// Input: 
// pListName == maybe NULL
// pystring == maybe NULL
// arrayCount
// ppArrayBlock (allocated array block)
//
// Output: python object (tuple of 3 objects) to return to user
PyObject* GetSDSFileInfo(
   PyObject* pListName, 
   PyObject* pystring, 
   INT64 arrayCount, 
   SDS_ARRAY_BLOCK* pArrayBlockFirst,
   SDS_FILE_HEADER* pFileHeader,
   // NULL possible
   SDS_FINAL_CALLBACK* pSDSFinalCallback = NULL) {

   PyObject* numpyArrayTuple = Py_None;
   numpyArrayTuple = PyTuple_New(arrayCount);

   LOGGING("In GetSDSFileInfo -- %lld   %p\n", arrayCount, pArrayBlockFirst);

   // Insert all the arrays
   for (INT64 i = 0; i < arrayCount; i++) {
      SDS_ARRAY_BLOCK* pArrayBlock = &pArrayBlockFirst[i];

      //LOGGING("Array block %lld at %p  compsize:%lld  %lld\n", i, pArrayBlock, pArrayBlock->ArrayCompressedSize, pArrayBlock->ArrayUncompressedSize);

      PyObject* numpyTuple = PyTuple_New(4);
      PyObject* shapeTuple = PyTuple_New(pArrayBlock->NDim);
      
      for (INT64 j = 0; j < pArrayBlock->NDim; j++) {
         PyTuple_SET_ITEM(shapeTuple, j, PyLong_FromLongLong(pArrayBlock->Dimensions[j]));
      }

      PyTuple_SET_ITEM(numpyTuple, 0, shapeTuple);

      // dtype fixup for Windows vs Linux
      PyTuple_SET_ITEM(numpyTuple, 1, PyLong_FromLongLong((long long)FixupDType(pArrayBlock->DType, pArrayBlock->ItemSize)));
      PyTuple_SET_ITEM(numpyTuple, 2, PyLong_FromLongLong((long long)pArrayBlock->Flags));
      PyTuple_SET_ITEM(numpyTuple, 3, PyLong_FromLongLong((long long)pArrayBlock->ItemSize));

      PyTuple_SET_ITEM(numpyArrayTuple, i, numpyTuple);
   }

   PyObject* pDict = GetFileHeaderDict(pFileHeader, pSDSFinalCallback);

   // return a tuple with a string and a tuple of arrays
   PyObject* returnFileInfo = PyTuple_New(4);

   PyTuple_SET_ITEM(returnFileInfo, 0, pystring);
   PyTuple_SET_ITEM(returnFileInfo, 1, numpyArrayTuple);
   PyTuple_SET_ITEM(returnFileInfo, 2, pListName);
   PyTuple_SET_ITEM(returnFileInfo, 3, pDict);
   return returnFileInfo;
}

//-------------------------------------------------------------
// Tries to find the '!' char in the string
// if it finds the ! it returns the location
// otherwise it returns NULL
const char* FindBang(const char *pString) {
   while (*pString) {
      if (*pString == '!') return pString;
      pString++;
   }
   return NULL;
}
//----------------------------------------------------
// Returns TRUE if included
BOOL IsIncluded(PyObject* pInclusionList, const char* pArrayName) {
  
   // If there is no inclusion list, assume all are included
    if (pInclusionList) {

      PyObject* includeDict = pInclusionList;
      PyObject* exists = PyDict_GetItemString(includeDict, pArrayName);

      // NOTE: to do... check for !
      if (!exists) {
         // The pArrayName might be a categorical column
         // If so, it will be in the format categoricalname!col0

         char* pHasBang = (char*)FindBang(pArrayName);
         if (pHasBang) {
            // temp remove bang
            *pHasBang = 0;

            // Now check dictionary again to see if we have a match
            exists = PyDict_GetItemString(includeDict, pArrayName);

            // replace bang
            *pHasBang = '!';

            // if we matched return TRUE
            if (exists) {
               LOGGING("categorical columns !array was included %s\n", pArrayName);
               return TRUE;
            }
         }

         LOGGING("!array was excluded %s\n", pArrayName);
         return FALSE;
      }
   }
   return TRUE;
}

//---------------------------------------------------------
// Clears both FastArray and base array WRITEABLE flags (marked readonly)
void ClearWriteableFlag(PyArrayObject* pArray) {
   // Flip off writeable flag at low level if came from shared memory
   PyArray_CLEARFLAGS(pArray, NPY_ARRAY_WRITEABLE);

   // Also clear the base flag
   PyArrayObject* pBase = (PyArrayObject*)PyArray_BASE(pArray);
   while (pBase != NULL) {
      // make sure base object is a numpy array object
      if (!PyArray_Check(pBase)) {
         break;
      }
      pArray = pBase;
      pBase = (PyArrayObject*)PyArray_BASE(pArray);
   }
   PyArray_CLEARFLAGS(pArray, NPY_ARRAY_WRITEABLE);

}

//----------------------------------------------------
// Input: sharedmemory struct we are reading from
// Output: python object (tuple of 4 objects) to return to user
// Python ONLY ROUTINE
// pSharedMemory->GetFileHeader()
void* ReadFromSharedMemory(SDS_SHARED_MEMORY_CALLBACK* pSMCB) {

   SDS_FILE_HEADER* pFileHeader = pSMCB->pFileHeader;
   char* baseOffset = pSMCB->baseOffset;
   int mode = pSMCB->mode;

   PyObject* pystring = NULL;
   PyObject* pListName = NULL;
   LOGGING("Reading from shared memory\n");    

   //----------- LOAD ARRAY NAMES -------------------------
   INT64 nameSize = pFileHeader->NameBlockSize;
   if (nameSize) {
      char *nameData = GetMemoryOffset(baseOffset, pFileHeader->NameBlockOffset);
      pListName = MakeListNames(nameData, pFileHeader->NameBlockCount, nameSize);
   }
   else {
      pListName = PyList_New(0);
   }
   LOGGING("Number of names %lld\n", PyList_GET_SIZE(pListName));

   //------------- META DATA -------------------------------
   // Python will make a copy of this string
   pystring = PyBytes_FromStringAndSize(
      GetMemoryOffset(baseOffset, pFileHeader->MetaBlockOffset),
      pFileHeader->MetaBlockSize);

   //--------------- LOAD ARRAYS ---------------------------
   INT64 arrayCount = pFileHeader->ArraysWritten;

   if (mode == COMPRESSION_MODE_INFO) {
      return GetSDSFileInfo(pListName, pystring, arrayCount, GetArrayBlock(baseOffset, 0), pFileHeader);
   }

   PyObject* returnTuple = Py_None;
   returnTuple = PyTuple_New(arrayCount);

   LOGGING("Number of arrays %lld\n", arrayCount);

   // Insert all the arrays
   for (INT64 i = 0; i < arrayCount; i++) {
      SDS_ARRAY_BLOCK* pArrayBlock = GetArrayBlock(baseOffset, i);

      // scalars
      //if (pArrayBlock->Dimensions ==0)

      char* data = GetMemoryOffset(baseOffset, pArrayBlock->ArrayDataOffset);

      // TODO: dtype fixup for Windows vs Linux
      int dtype = FixupDType(pArrayBlock->DType, pArrayBlock->ItemSize);

      // Use our own data in shared memory
      // TODO: The pArrayBlock->Dimensions cast to npy_intp* is incorrect here if running on a 32-bit system;
      //       need to check the dimension values fit into npy_intp and convert them if needed.
      //       Can use CPP or C++ constexpr-if so we only pay the cost on 32-bit systems.
      PyArrayObject* pArray = AllocateNumpyArrayForData(
         pArrayBlock->NDim,
         (npy_intp*)pArrayBlock->Dimensions,
         dtype, pArrayBlock->ItemSize,
         data,
         pArrayBlock->Flags,
         (npy_intp*)pArrayBlock->Strides);
      CHECK_MEMORY_ERROR(pArray);

      if (pArray) {
         // Make it read only since in shared memory
         ClearWriteableFlag(pArray);
         PyTuple_SetItem(returnTuple, i, (PyObject*)pArray);
      }
      else {
         Py_IncRef(Py_None);
         PyTuple_SetItem(returnTuple, i, Py_None);
      }
   }

   PyObject* pDict = GetFileHeaderDict(pFileHeader, NULL);

   // return a tuple with a string and a tuple of arrays
   PyObject* returnTupleTuple = PyTuple_New(4);
   PyTuple_SET_ITEM(returnTupleTuple, 0, pystring);
   PyTuple_SET_ITEM(returnTupleTuple, 1, returnTuple);
   PyTuple_SET_ITEM(returnTupleTuple, 2, pListName);
   PyTuple_SET_ITEM(returnTupleTuple, 3, pDict);
   return returnTupleTuple;
}


//--------------------------------------------
// Wrap arrays
PyObject* ReadFinalStackArrays(
   SDS_STACK_CALLBACK* pSDSFinalCallback,
   INT64 arraysWritten,
   SDS_STACK_CALLBACK_FILES *pSDSFileInfo,
   SDS_FILTER*    pSDSFilter,
   INT64 fileCount) {

   //---------- BUILD PYTHON RETURN OBJECTS ---------
   PyObject* returnArrayTuple = PyTuple_New(arraysWritten);
   PyObject* pyListName = PyList_New(arraysWritten);
   PyObject* pyArrayOffset = PyList_New(arraysWritten);

   // not currently used
   //int hasFilter = (pSDSFilter && pSDSFilter->pBoolMask && pSDSFilter->pFilterInfo);

   for (int t = 0; t < arraysWritten; t++) {
      PyObject* item = NULL;

      item = (PyObject*)(pSDSFinalCallback[t].pArrayObject);

      LOGGING("Setting item %d  %p\n", t, item);

      // Return NONE for any arrays with memory issues
      if (item == NULL) {
         LOGGING("!! removed item %d -- setting to PyNone\n", t);
         Py_INCREF(Py_None);
         item = Py_None;
      }

      //printf("ref %d  %llu\n", i, item->ob_refcnt);
      PyTuple_SET_ITEM(returnArrayTuple, t, item);

      //================
      PyObject* pTuple = PyTuple_New(2);

      PyTuple_SET_ITEM(pTuple, 0, PyBytes_FromString(pSDSFinalCallback[t].ArrayName));
      PyTuple_SET_ITEM(pTuple, 1, PyLong_FromLong((long)pSDSFinalCallback[t].ArrayEnum));

      // pylist_append will add a reference count but setitem will not
      PyList_SET_ITEM(pyListName, t, pTuple);

      //==============
      PyArrayObject* pOffsetArray = AllocateNumpyArray(1, (npy_intp*)&fileCount, NPY_LONGLONG);
      if (pOffsetArray) {
         INT64* pOffsets = (INT64*)PyArray_GETPTR1(pOffsetArray, 0);

         LOGGING("arary hasfilter:%d  offsets%lld  %d  name:%s\n", pSDSFilter && pSDSFilter->pBoolMask, fileCount, pSDSFinalCallback[t].ArrayEnum, pSDSFinalCallback[t].ArrayName);

         // copy over our array offsets (skip past first element which is 0)
         memcpy(pOffsets, pSDSFinalCallback[t].pArrayOffsets + 1, fileCount * sizeof(INT64));

         PyList_SET_ITEM(pyArrayOffset, t, (PyObject*)pOffsetArray);
      }
      else {
         PyList_SET_ITEM(pyArrayOffset, t, Py_None);
         Py_INCREF(Py_None);
      }
   }

   PyObject* pyFiles = PyList_New(fileCount);
   PyObject* pyMeta = PyList_New(fileCount);

   for (int f = 0; f < fileCount; f++) {
      //printf("filename: %s\n  meta: %s\n ", pSDSFileInfo[f].Filename, pSDSFileInfo[f].MetaData);
      PyList_SET_ITEM(pyFiles, f, PyBytes_FromString(pSDSFileInfo[f].Filename));
      PyList_SET_ITEM(pyMeta, f, PyBytes_FromStringAndSize(pSDSFileInfo[f].MetaData, pSDSFileInfo[f].MetaDataSize));
   }

   // return a tuple with a string and a tuple of arrays
   PyObject* returnTupleTuple = PyTuple_New(6);
   PyTuple_SET_ITEM(returnTupleTuple, 0, returnArrayTuple);
   PyTuple_SET_ITEM(returnTupleTuple, 1, pyListName);
   PyTuple_SET_ITEM(returnTupleTuple, 2, pyArrayOffset);
   PyTuple_SET_ITEM(returnTupleTuple, 3, pyMeta);
   PyTuple_SET_ITEM(returnTupleTuple, 4, pyFiles);

   // Return the first fileheader to help autodetect the type of file when stecking
   if (fileCount > 0) {
      PyObject* pDict = GetFileHeaderDict(pSDSFileInfo[0].pFileHeader, NULL);
      PyTuple_SET_ITEM(returnTupleTuple, 5, pDict);
   }
   else {
      PyTuple_SET_ITEM(returnTupleTuple, 5, Py_None);
      Py_INCREF(Py_None);
   }


   return returnTupleTuple;
}


//--------------------------------------------
// Wrap arrays
PyObject* ReadFinalArrays(
   INT64 arraysWritten,
   SDSArrayInfo* pArrayInfo
   ) {

   //---------- BUILD PYTHON RETURN OBJECTS ---------
   PyObject* returnArrayTuple =
      PyTuple_New(arraysWritten);

   // Decompression
   for (int t = 0; t < arraysWritten; t++) {
      PyObject* item = NULL;

      item = (PyObject*)(pArrayInfo[t].pArrayObject);

      LOGGING("Setting item %d  %p\n", t, item);

      // Return NONE for any arrays with memory issues
      if (item == NULL) {
         LOGGING("!! removed item %d -- setting to PyNone\n", t);
         Py_INCREF(Py_None);
         item = Py_None;
      }

      //printf("ref %d  %llu\n", i, item->ob_refcnt);
      PyTuple_SET_ITEM(returnArrayTuple, t, item);
   }

   return returnArrayTuple;
}


//-------------------------
// Wrap one file
// May pass in NULL
PyObject* ReadFinalWrap(SDS_FINAL_CALLBACK* pSDSFinalCallback) {

   if (pSDSFinalCallback == NULL) {
      Py_INCREF(Py_None);
      return Py_None;
   }

   INT32 mode = pSDSFinalCallback->mode;
   INT64 arraysWritten = pSDSFinalCallback->arraysWritten;
   SDS_ARRAY_BLOCK* pArrayBlocks = pSDSFinalCallback->pArrayBlocks;
   SDSArrayInfo* pArrayInfo = pSDSFinalCallback->pArrayInfo;
   SDS_FILE_HEADER* pFileHeader = pSDSFinalCallback->pFileHeader;

   PyObject* pListName = ReadListNamesPython(pSDSFinalCallback->nameData, pFileHeader);
   PyObject* pystring = GetMetaData(pSDSFinalCallback->metaData, pSDSFinalCallback->metaSize);

   // -- STOP EARLY IF THE USER JUST WANTS THE INFORMATION -----------------
   if (mode == COMPRESSION_MODE_INFO) {

      LOGGING("Returning just the info\n");
      PyObject* returnObject = GetSDSFileInfo(pListName, pystring, arraysWritten, pArrayBlocks, pFileHeader, pSDSFinalCallback);
      return returnObject;
   }

   LOGGING("Building return object of arrays %lld  %p\n", arraysWritten, pArrayInfo);

   //---------- BUILD PYTHON RETURN OBJECTS ---------
   PyObject* returnArrayTuple = ReadFinalArrays(arraysWritten, pArrayInfo);
   PyObject* pDict = GetFileHeaderDict(pFileHeader, pSDSFinalCallback);

   // return a tuple with a string and a tuple of arrays
   PyObject* returnWrap = PyTuple_New(4);
   PyTuple_SET_ITEM(returnWrap, 0, pystring);
   PyTuple_SET_ITEM(returnWrap, 1, returnArrayTuple);
   PyTuple_SET_ITEM(returnWrap, 2, pListName);
   PyTuple_SET_ITEM(returnWrap, 3, pDict);

   // Soon after this returns, files will be closed, memory deallocated
   return returnWrap;
}


//----------------------------------------
// CALLBACK2 - can wrap more than one file
// finalCount is how many info sections to return
// if there are sections inside a single file, the finalCount > 1
void* ReadFinal(SDS_FINAL_CALLBACK* pSDSFinalCallback, INT64 finalCount) {     

   PyObject* returnItem = NULL;

   if (finalCount <= 0) {
      // No valid files found, just return None
      Py_INCREF(Py_None);
      returnItem= Py_None;

   } else {
      // Return a list of all the data
      returnItem = PyList_New(finalCount);

      // Wrap the item for every file
      for (INT64 file = 0; file < finalCount; file++) {
         PyObject* item = ReadFinalWrap(&pSDSFinalCallback[file]);

         // Steals a reference
         PyList_SET_ITEM(returnItem, file, item);
      }
   }
   LOGGING("End ReadFinal %p.  finalCount %lld\n", returnItem, finalCount);
   return returnItem;
}


//----------------------------------------
// CALLBACK2 - all files were stacked into one column
void* ReadFinalStack(
   SDS_STACK_CALLBACK* pSDSFinalCallback, 
   INT64 finalCount, 
   SDS_STACK_CALLBACK_FILES *pSDSFileInfo, 
   SDS_FILTER*    pSDSFilter,
   INT64 fileCount) {

   PyObject* returnItem = NULL;

   if (finalCount <= 0) {
      Py_INCREF(Py_None);
      returnItem = Py_None;

   }
   else {

      PyObject* returnArrayTuple =
         ReadFinalStackArrays(
            pSDSFinalCallback,
            finalCount,
            pSDSFileInfo,
            pSDSFilter,
            fileCount);

      returnItem = returnArrayTuple;
   }

   LOGGING("End ReadFinalStack %p\n", returnItem);
   return returnItem;
}

//--------------------------------------
// free string with WORKSPACE_FREE
void
CopyUnicodeString(PyObject* pUnicode, char** returnString, INT64* returnSize) {
   PyObject* temp2 = PyUnicode_AsASCIIString(pUnicode);
   if (temp2 != NULL) {
      *returnString = PyBytes_AsString(temp2);
      *returnSize = strlen(*returnString);
      char* pNewString = (char*)WORKSPACE_ALLOC(*returnSize + 1);
      memcpy(pNewString, *returnString, *returnSize);
      pNewString[*returnSize] = 0;

      *returnString = pNewString;
      // Release reference
      Py_DecRef(temp2);
   }
   else {
      *returnString = NULL;
      *returnSize = strlen(*returnString);
   }
}



//----------------------------------------------------
// check for "section=" (NOT the same as sections)
// must be a unicode string
// returns 0 if no section
INT64 GetStringFromDict(const char* dictstring, PyObject *kwargs, char** returnString, INT64* returnSize ) {
   if (!kwargs) return 0;

   PyObject* sectionObject = PyDict_GetItemString(kwargs, dictstring);

   if (sectionObject && PyUnicode_Check(sectionObject)) {
      CopyUnicodeString(sectionObject, returnString, returnSize);
      return *returnSize;
   }
   else {
      *returnString = NULL;
      *returnSize = 0;
   }

   return 0;
}

//----------------------------------------------------
// check for "sections=" (NOT the same as section)
// must be a list of strings
// returns NULL of no list of strings found
SDS_STRING_LIST* GetSectionsName(PyObject *kwargs) {
   if (!kwargs) return NULL;

   PyObject* sectionNameObject = PyDict_GetItemString(kwargs, "sections");

   if (sectionNameObject && PyList_Check(sectionNameObject)) {
      return StringListToVector(sectionNameObject);
   }

   return NULL;
}

//----------------------------------------------------
// check for "bandsize="
// must be an INT
// returns 0 if no bandsize
INT64 GetBandSize(PyObject *kwargs) {
   if (!kwargs) return 0;

   PyObject* bandsizeObject = PyDict_GetItemString(kwargs, "bandsize");

   if (bandsizeObject && PyLong_Check(bandsizeObject)) {
      INT64 result = PyLong_AsLongLong (bandsizeObject);

      // minimum bandsize is 10K
      if (result < 0) result = 0;
      return result;
   }

   return 0;
}

//----------------------------------------------------
// check for "folders="
// must be a list of strings
// returns NULL of no list of strings found
SDS_STRING_LIST* GetFoldersName(PyObject *kwargs) {
   if (!kwargs) return NULL;

   PyObject* folderNameObject = PyDict_GetItemString(kwargs, "folders");

   if (folderNameObject && PyList_Check(folderNameObject)) {
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
// NOTE: If the user wants both the sharename and the file, this API must be called twice (with and without sharename)

PyObject *CompressFile(PyObject* self, PyObject *args, PyObject *kwargs)
{
   PyObject *inListArrays = NULL;
   PyListObject *inListNames = NULL;

   const char *fileName;
   UINT32 fileNameSize;

   const char *metaData;
   UINT32 metaDataSize;

   const char *shareName = NULL;
   UINT32 shareNameSize = 0;

   INT32 mode = COMPRESSION_MODE_COMPRESS_FILE;
   INT32 compType = COMPRESSION_TYPE_ZSTD;
   INT32 level = ZSTD_CLEVEL_DEFAULT;
   INT32 fileType = 0;

   if (!PyArg_ParseTuple(
      args, "y#y#OO!iii|y#",
      &fileName, &fileNameSize,
      &metaData, &metaDataSize,
      &inListArrays,
      &PyList_Type, &inListNames,
      &compType,
      &level,
      &fileType,
      &shareName, &shareNameSize
   )) {

      return NULL;
   }

   LOGGING("In CompressFile %s\n", fileName);

   // Check for kwargs: folders, bandsize, section
   //
   SDS_STRING_LIST* folderName = GetFoldersName(kwargs);
   INT64 bandSize = GetBandSize(kwargs);

   char* pSectionName = NULL;
   INT64 sectionSize = 0;

   GetStringFromDict("section", kwargs, &pSectionName, &sectionSize);

   if (sectionSize >= SDS_MAX_SECTIONNAME) {
      PyErr_Format(PyExc_ValueError, "section name too large: %lld", sectionSize);
      return NULL;
   }

   //printf("section name: %s\n", sectionName);

   ClearErrors();

   // Handle list of names ------------------------------------------------------
   INT64 listNameCount = PyList_GET_SIZE(inListNames);

   LOGGING("Name count is %d\n", (int)listNameCount);

   // alloc worst case scenario
   char* pListNames = (char*)WORKSPACE_ALLOC((SDS_MAX_FILENAME * listNameCount) + 8);
   if (!pListNames) {
      return NULL;
   }

   // Process list of names tuples
   INT64 listNameSize = BuildListInfo(inListNames, pListNames);

   // Handle list of numpy arrays -----------------------------------
   INT64 totalItemSize = 0;
   INT64 arrayCount = 0;

   ArrayInfo* aInfo = BuildArrayInfo(inListArrays, &arrayCount, &totalItemSize, FALSE);

   // CHECK FOR ERRORS
   if (aInfo) {

      SDS_WRITE_CALLBACKS SDSWriteCallbacks;

      SDSWriteCallbacks.BeginAllowThreads = BeginAllowThreads;
      SDSWriteCallbacks.EndAllowThreads = EndAllowThreads;

      SDS_WRITE_INFO SDSWriteInfo;
      SDSWriteInfo.aInfo = (SDSArrayInfo*)WORKSPACE_ALLOC(sizeof(SDSArrayInfo) * arrayCount);

      //============================================
      // Convert from ArrayInfo* to SDSArrayInfo*
      //
      SDSArrayInfo* pDest = SDSWriteInfo.aInfo;
      ArrayInfo* pSrc = aInfo;

      for (INT64 i = 0; i < arrayCount; i++) {

         pDest->ArrayLength = pSrc->ArrayLength;
         pDest->ItemSize = (INT32)pSrc->ItemSize;
         pDest->pArrayObject = pSrc->pObject; // We do not need this..
         pDest->NumBytes = pSrc->NumBytes;
         pDest->NumpyDType = pSrc->NumpyDType;
         pDest->pData = pSrc->pData;

         INT32 ndim = pSrc->NDim;
         if (ndim > SDS_MAX_DIMS) {
            printf("!!!SDS: array dimensions too high: %d\n", ndim);
            ndim = SDS_MAX_DIMS;
         }
         //if (ndim < 1) {
         //   printf("!!!SDS: array dimensions too low: %d\n", ndim);
         //   ndim = 1;
         //}
         pDest->NDim = ndim;

         for (int dim_idx = 0; dim_idx < SDS_MAX_DIMS; dim_idx++) {
            pDest->Dimensions[dim_idx] = 0;
            pDest->Strides[dim_idx] = 0;
         }

         // None can be passed in for an array now
         if (pSrc->pObject) {

            npy_intp* pdims = ((PyArrayObject_fields *)pSrc->pObject)->dimensions;
            npy_intp* pstrides = ((PyArrayObject_fields *)pSrc->pObject)->strides;

            for (int dim_idx = 0; dim_idx < ndim; dim_idx++) {
               pDest->Dimensions[dim_idx] = pdims[dim_idx];
               pDest->Strides[dim_idx] = pstrides[dim_idx];
            }

            pDest->Flags = PyArray_FLAGS((PyArrayObject*)(pSrc->pObject));

            // make sure C or F contiguous
            if (!(pDest->Flags & (SDS_ARRAY_C_CONTIGUOUS | SDS_ARRAY_F_CONTIGUOUS))) {
               //pSrc->pObject = PyArray_FromAny(pSrc->pObject, NULL, 0, 0, NPY_ARRAY_ENSURECOPY, NULL);

               printf("!!!SDS: array is not C or F contiguous: %d\n", pDest->Flags);
            }
         }
         else {
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
      SDSWriteInfo.appendFileHeadersMode = FALSE;
      SDSWriteInfo.appendRowsMode = FALSE;
      SDSWriteInfo.appendColumnsMode = FALSE;
      SDSWriteInfo.bandSize = bandSize;

      SDSWriteInfo.sectionName = NULL;
      SDSWriteInfo.sectionNameSize = 0;

      // if the kwarg section exists,
      if (pSectionName) {
         SDSWriteInfo.appendRowsMode = TRUE;
         SDSWriteInfo.sectionName = pSectionName;
         SDSWriteInfo.sectionNameSize = sectionSize;
      }

      BOOL result =
         SDSWriteFile(
            fileName,
            shareName,  // can be NULL
            folderName, 
            &SDSWriteInfo,
            &SDSWriteCallbacks );

      // FREE workspace allocations
      WORKSPACE_FREE(SDSWriteInfo.aInfo);
      FreeArrayInfo(aInfo);
   }

   WORKSPACE_FREE(pListNames);

   if (pSectionName) {
      WORKSPACE_FREE(pSectionName);
   }
   if (folderName) {
      DestroyStringList(folderName);
   }

   // If there are errors, return NULL
   if (CheckErrors()) {
      return NULL;
   }

   Py_INCREF(Py_None);
   return Py_None;
}


//==================================
// Called back when reading in data
void AllocateArrayCallback(SDS_ALLOCATE_ARRAY *pAllocateArray) {

   SDSArrayInfo* pDestInfo = pAllocateArray->pDestInfo;

   int ndim = pAllocateArray->ndim;

   const char* pArrayName = pAllocateArray->pArrayName;

   LOGGING("Allocate array name: %s\n", pArrayName);

   pDestInfo->pArrayObject = NULL;

   //if (pAllocateArray->data)
   pDestInfo->pData = NULL;

   //if (IsIncluded((PyObject*)pAllocateArray->pInclusionList, pArrayName)) {
   if (TRUE) {
      // If we have no dimensions, do not allocate
      if (ndim) {

         INT64* dims = pAllocateArray->dims;
         INT64* strides = pAllocateArray->strides;

         if (pAllocateArray->data) {
            LOGGING("Shared memory was set to %p\n", pAllocateArray->data);
         }

         LOGGING("Allocating ndim:%d  dim0:%lld  type:%d  itemsize:%lld  flags:%d  strde0:%lld", pAllocateArray->ndim, dims[0], pAllocateArray->numpyType, pAllocateArray->itemsize, pAllocateArray->numpyFlags, strides[0]);

         // Use different array-creation functions based on whether we're allocating new (or reclaiming recycled) memory,
         // or we have some existing backing memory (e.g. shared memory) and we're creating the new array object to just
         // wrap that memory.
         if (pAllocateArray->data)
         {
            pDestInfo->pArrayObject = AllocateNumpyArrayForData(
               pAllocateArray->ndim,
               (npy_intp*)dims,
               pAllocateArray->numpyType,
               pAllocateArray->itemsize,
               pAllocateArray->data,  // set for shared memory?
               pAllocateArray->numpyFlags,
               (npy_intp*)strides);

            // TODO: Check whether pDestInfo->pArrayObject is nullptr here; if it is, something went wrong
            //       with the allocation and now's the chance to gracefully handle it; otherwise we'll segfault
            //       later once we try to read from that memory.
         }
         else
         {
            pDestInfo->pArrayObject = AllocateNumpyArray(
               pAllocateArray->ndim,
               (npy_intp*)dims,
               pAllocateArray->numpyType,
               pAllocateArray->itemsize,
               pAllocateArray->numpyFlags & NPY_ARRAY_F_CONTIGUOUS,
               (npy_intp*)strides);
         }

         // check for successful allocation
         if (pDestInfo->pArrayObject) {
            if (pAllocateArray->data) {
               // Flip off writeable flag at low level if came from shared memory
               ClearWriteableFlag((PyArrayObject*)(pDestInfo->pArrayObject));
            }

            // Fill in pDestInfo
            pDestInfo->pData = (char*)PyArray_GETPTR1((PyArrayObject*)(pDestInfo->pArrayObject), 0);
         }
      }
   }
}

//----------------------------------------------------
// Return the maskLength and pBooleanMask
void
GetFilters(PyObject* kwargs, SDS_READ_CALLBACKS* pRCB) {

   pRCB->Filter.BoolMaskLength = 0;
   pRCB->Filter.pBoolMask = NULL;
   pRCB->Filter.BoolMaskTrueCount = 0;
   pRCB->Filter.pFilterInfo = NULL;

   pRCB->MustExist = FALSE;

   if (kwargs) {
      //-------------------
      // This DOES NOT change the ref count (borrowed reference)
      //PyObject* filterItem = PyDict_GetItemString(kwargs, "filter");
      //
      //if (filterItem && PyArray_Check(filterItem)) {
      //   if (PyArray_TYPE((PyArrayObject*)filterItem) == NPY_INT || PyArray_TYPE((PyArrayObject*)filterItem) == NPY_LONG) {
      //      pRCB->Filter.FancyLength = ArrayLength((PyArrayObject*)filterItem);
      //      pRCB->Filter.pFancyMask = (INT32*)PyArray_GETPTR1((PyArrayObject*)filterItem, 0);
      //      LOGGING("Found valid filter.  length: %lld\n", pRCB->Filter.FancyLength);
      //   }
      //}

      PyObject* maskItem = PyDict_GetItemString(kwargs, "mask");

      if (maskItem && PyArray_Check(maskItem)) {
         if (PyArray_TYPE((PyArrayObject*)maskItem) == NPY_BOOL) {         
            pRCB->Filter.BoolMaskLength = ArrayLength((PyArrayObject*)maskItem);
            pRCB->Filter.pBoolMask = (bool*)PyArray_GETPTR1((PyArrayObject*)maskItem, 0);
            // Needed
            pRCB->Filter.BoolMaskTrueCount = SumBooleanMask((INT8*)pRCB->Filter.pBoolMask, pRCB->Filter.BoolMaskLength);
            LOGGING("Found valid mask.  length: %lld\n", pRCB->Filter.BoolMaskLength);
         }
      }

      PyObject* mustexist = PyDict_GetItemString(kwargs, "mustexist");
      if (mustexist && PyBool_Check(mustexist)) {
         if (mustexist == Py_True) {
            pRCB->MustExist = TRUE;
         }
      }
   }

}


//----------------------------------------------------
// Python Interface
// Arg1: BYTES - filename (UNICODE not allowed)
// Arg2: mode: integer defaults to COMPRESSION_MODE_DECOMPRESS_FILE, also allowed: COMPRESSION_MODE_INFO
// Arg3: <optional> shared memory prefix (UNICODE not allowed)
//
// Kwargs
// include=
// folder=
//
// Returns tuple <metadata, list of arrays compressed, list of array names/enums>
//
PyObject *DecompressFile(PyObject* self, PyObject *args, PyObject *kwargs) {
   const char *fileName;
   UINT32 fileNameSize;

   const char *shareName = NULL;
   SDS_STRING_LIST *folderName = NULL;
   SDS_STRING_LIST *sectionsName = NULL;

   UINT32 shareNameSize = 0;

   INT32 mode = COMPRESSION_MODE_DECOMPRESS_FILE;

   PyObject* includeDict = NULL;

   if (!PyArg_ParseTuple(
      args, "y#i|y#",
      &fileName, &fileNameSize,
      &mode,
      &shareName, &shareNameSize)) {

      return NULL;
   }

   if (kwargs && PyDict_Check(kwargs)) {
      // Borrowed reference
      // Returns NULL if key not present
      PyObject* includedItem =PyDict_GetItemString(kwargs, "include");

      if (includedItem && PyDict_Check(includedItem)) {
         LOGGING("Found valid inclusion dict\n");
         includeDict = includedItem;
      }
      else {
         //LOGGING("did not like dict!  %p\n", includedItem);
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
   //GetFilters(kwargs, &sdsRCB.pBooleanMask, &sdsRCB.MaskLength);

   SDS_READ_INFO  sdsRI;

   sdsRI.mode = mode;

   //==============================================
   void*  result =
      SDSReadFile(
         fileName,
         shareName,
         folderName,
         sectionsName,
         &sdsRI,
         &sdsRCB);

   if (folderName) {
      DestroyStringList(folderName);
   }

   if (sectionsName) {
      DestroyStringList(sectionsName);
   }

   // If there are errors, return NULL
   if (!result && CheckErrors()) {
      return NULL;
   }

   if (!result) {
      PyErr_Format(PyExc_ValueError, "NULL is returned from SDSReadFile but no error string was found");
   }
   return (PyObject*)result;
}



//----------------------------------------------------
// Arg1: List of [filenames]
// Arg2: mode: integer defaults to COMPRESSION_MODE_DECOMPRESS_FILE, also allowed: COMPRESSION_MODE_INFO
// multimode = stacking or reading multiple files
// Returns tuple <metadata, list of arrays compressed, list of array names/enums>
//
PyObject*
InternalDecompressFiles(
   PyObject*   self, 
   PyObject*   args,
   PyObject*   kwargs,
   int         multiMode) {

   PyObject*   listFilenames;
   PyObject*   includeDict = NULL;
   void*       result = NULL;
   double      reserveSpace = 0.0;

   INT32 mode = COMPRESSION_MODE_DECOMPRESS_FILE;

   if (!PyArg_ParseTuple(
      args, "O!|i",
      &PyList_Type, &listFilenames,
      &mode)) {

      return NULL;
   }

   //--------------------------------------------------
   // Check if we are flipping into info modes
   //
   if (multiMode == SDS_MULTI_MODE_READ_MANY && mode != COMPRESSION_MODE_DECOMPRESS_FILE) {
      multiMode = SDS_MULTI_MODE_READ_MANY_INFO;
   }
   if (multiMode == SDS_MULTI_MODE_STACK_MANY && mode != COMPRESSION_MODE_DECOMPRESS_FILE) {
      multiMode = SDS_MULTI_MODE_STACK_MANY_INFO;
   }

   char* pOutputName = NULL;
   INT64 outputSize = 0;

   SDS_STRING_LIST*  pInclusionList = NULL;
   SDS_STRING_LIST*  pFolderList = NULL;
   SDS_STRING_LIST*  pSectionsList = NULL;
   INT64             maskLength = 0;
   bool*             pBooleanMask = NULL;

   if (multiMode == SDS_MULTI_MODE_CONCAT_MANY) {
      GetStringFromDict("output", kwargs, &pOutputName, &outputSize);
      if (!pOutputName) {
         PyErr_Format(PyExc_ValueError, "The output= must be a filename when concatenating files");
      }
   }


   if (kwargs && PyDict_Check(kwargs)) {
      // Borrowed reference
      // Returns NULL if key not present
      PyObject* includedItem = PyDict_GetItemString(kwargs, "include");

      if (includedItem && PyList_Check(includedItem)) {
         LOGGING("Found valid inclusion list\n");
         pInclusionList = StringListToVector(includedItem);
      }
      else {
         //LOGGING("did not like dict!  %p\n", includedItem);
      }
      //-------------------

      PyObject* reserveItem = PyDict_GetItemString(kwargs, "reserve");

      if (reserveItem && PyFloat_Check(reserveItem)) {
         reserveSpace = PyFloat_AsDouble(reserveItem);
         LOGGING("Found valid reserve space as %lf\n", reserveSpace);
      }

      pFolderList = GetFoldersName(kwargs);
      pSectionsList = GetSectionsName(kwargs);


   }

   SDS_STRING_LIST* pFilenames = StringListToVector(listFilenames);
   INT64 fileCount = pFilenames->size();

   LOGGING("InternalDecompress filecount: %lld  mode: %d", fileCount, multiMode);
   if (fileCount) {

      SDS_MULTI_READ* pMultiRead = (SDS_MULTI_READ * )WORKSPACE_ALLOC(sizeof(SDS_MULTI_READ) * fileCount);

      if (pMultiRead) {
         INT64 i = 0;

         // loop over all filenames and build callback
         for (const char* filename : *pFilenames) {
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
         result =
            SDSReadManyFiles(
               pMultiRead,
               pInclusionList,
               pFolderList,
               pSectionsList,
               fileCount,
               multiMode,
               &sdsRCB);

         WORKSPACE_FREE(pMultiRead);
      }
   }

   // Free what might have been allocated
   if (pOutputName) {
      WORKSPACE_FREE(pOutputName);
   }

   DestroyStringList(pFilenames);

   if (pInclusionList) {
      DestroyStringList(pInclusionList);
   }
   if (pFolderList) {
      DestroyStringList(pFolderList);
   }

   LOGGING("Multistack after destroy string list\n");

   BOOL isThereAnError = CheckErrors();

   // If there are errors, return NULL
   if (!result && isThereAnError) {
      return NULL;
   }

   if (isThereAnError && fileCount == 1) {
      // result is good, but there is an error.  decrement the result so we can get rid of it
      Py_DecRef((PyObject*)result);
      return NULL;
   }

   // We might have a partial error if we get here
   if (!result) {
      Py_INCREF(Py_None);
      return Py_None;
   }

   LOGGING("Multistack returning a good result %p\n", result);
   return (PyObject*)result;
}



//----------------------------------------------------
// Arg1: List of [filenames]
// Arg2: mode: integer defaults to COMPRESSION_MODE_DECOMPRESS_FILE, also allowed: COMPRESSION_MODE_INFO
// Arg3: <optional> shared memory prefix (UNICODE not allowed)
// Returns tuple <metadata, list of arrays compressed, list of array names/enums>
//
PyObject *MultiDecompressFiles(PyObject* self, PyObject *args, PyObject *kwargs) {
   return InternalDecompressFiles(self, args, kwargs, SDS_MULTI_MODE_READ_MANY);
}

PyObject *MultiStackFiles(PyObject* self, PyObject *args, PyObject *kwargs) {
   return InternalDecompressFiles(self, args, kwargs, SDS_MULTI_MODE_STACK_MANY);
}

PyObject *MultiPossiblyStackFiles(PyObject* self, PyObject *args, PyObject *kwargs) {
   return InternalDecompressFiles(self, args, kwargs, SDS_MULTI_MODE_STACK_OR_READMANY);
}

//----------------------------------------------------
// Arg1: List of [filenames]
// kwarg: output = string full path of file
PyObject *MultiConcatFiles(PyObject* self, PyObject *args, PyObject *kwargs) {
   return InternalDecompressFiles(self, args, kwargs, SDS_MULTI_MODE_CONCAT_MANY);
}

//----------------------------------------------------
// Arg1: BYTES - filename
// Arg2: tuple of byte strings (UNICODE not allowed) which are replacements
//
PyObject *SetLustreGateway(PyObject* self, PyObject *args) {
   const char *fileName;
   UINT32 fileNameSize;

   PyObject*   tuple;

   if (!PyArg_ParseTuple(
      args, "y#O!",
      &fileName, &fileNameSize,
      &PyTuple_Type, &tuple)) {

      return NULL;
   }

   //printf("hint: %s\n", fileName);

   g_gatewaylist.empty();
   INT64 tupleLength = PyTuple_GET_SIZE(tuple);
   g_gatewaylist.reserve(tupleLength);

   for (INT64 i = 0; i < tupleLength; i++) {
      const char *gateway = PyBytes_AsString(PyTuple_GET_ITEM(tuple, i));
      //printf("gw: %s\n", gateway);
      g_gatewaylist.push_back(std::string(gateway));
   }

   Py_INCREF(Py_None);
   return Py_None;
}

