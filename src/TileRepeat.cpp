#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"

#include "CommonInc.h"

//#define LOGGING printf
#define LOGGING(...)


PyObject*
ToBeDone(PyObject* self, PyObject* args, PyObject* kwargs) {
   return NULL;
}

struct stOffsets {
   char* pData;
   INT64    itemsize;
};
static const INT64 CHUNKSIZE = 16384;

// This is used to multiply the strides
SFW_ALIGN(64)
const union
{
   INT32 i[8];
   __m256i m;
//} __vindex8_strides = { 7, 6, 5, 4, 3, 2, 1, 0 };
} __vindex8_strides = { 0, 1, 2, 3, 4, 5, 6, 7 };

//-----------------------------------
//
void ConvertRecArray(char* pStartOffset, INT64 startRow, INT64 totalRows, stOffsets* pstOffset, INT64* pOffsets, INT64 numArrays, INT64 itemSize)
{
   // Try to keep everything in L1Cache
   const INT64 L1CACHE = 32768;
   INT64 CHUNKROWS = L1CACHE / (itemSize * 2);
   if (CHUNKROWS < 1) {
      CHUNKROWS = 1;
   }

   __m256i vindex = _mm256_mullo_epi32(_mm256_set1_epi32((INT32)itemSize), _mm256_loadu_si256(&__vindex8_strides.m));
   __m128i vindex128 = _mm256_extracti128_si256(vindex, 0);

   while (startRow < totalRows) {

      // Calc how many rows to process in this pass
      INT64 endRow = startRow + CHUNKROWS;
      if (endRow > totalRows) {
         endRow = totalRows;
      }

      INT64 origRow = startRow;

      //printf("processing %lld\n", startRow);
      for (INT64 i = 0; i < numArrays; i++) {

         startRow = origRow;

         // Calculate place to read
         char* pRead = pStartOffset + pOffsets[i];
         char* pWrite = pstOffset[i].pData;

         INT64 arrItemSize = pstOffset[i].itemsize;

         //printf("processing  start:%lld  end:%lld   pRead:%p  %p  itemsize: %lld\n", startRow, endRow, pRead, pWrite, arrItemSize);

         switch (pstOffset[i].itemsize) {
         case 1:
            while (startRow < endRow) {
               INT8 data = *(INT8*)(pRead + (startRow * itemSize));
               *(INT8*)(pWrite + startRow) = data;
               startRow++;
            }
            break;
         case 2:
            while (startRow < endRow) {
               INT16 data = *(INT16*)(pRead + (startRow * itemSize));
               *(INT16*)(pWrite + startRow * arrItemSize) = data;
               startRow++;
            }
            break;
         case 4:
            // ??? use _mm256_i32gather_epi32 to speed up
            {
               INT64 endSubRow = endRow - 8;
               while (startRow < endSubRow) {
                  __m256i m0 = _mm256_i32gather_epi32((INT32*)(pRead + (startRow * itemSize)), vindex, 1);
                  _mm256_storeu_si256((__m256i*)(pWrite + (startRow * arrItemSize)), m0);
                  startRow += 8;
               }
               while (startRow < endRow) {
                  INT32 data = *(INT32*)(pRead + (startRow * itemSize));
                  *(INT32*)(pWrite + startRow * arrItemSize) = data;
                  startRow++;
               }
            }
            break;
         case 8:
            {
               INT64 endSubRow = endRow - 4;
               while (startRow < endSubRow) {
                  __m256i m0 = _mm256_i32gather_epi64((INT64*)(pRead + (startRow * itemSize)), vindex128, 1);
                  _mm256_storeu_si256((__m256i*)(pWrite + (startRow * arrItemSize)), m0);
                  startRow += 4;
               }
               while (startRow < endRow) {
                  INT64 data = *(INT64*)(pRead + (startRow * itemSize));
                  *(INT64*)(pWrite + startRow * arrItemSize) = data;
                  startRow++;
               }
            }
            break;
         default:
            while (startRow < endRow) {
               char* pSrc = pRead + (startRow * itemSize);
               char* pDest = pWrite + (startRow * arrItemSize);
               char* pEnd = pSrc + arrItemSize;
               while ((pSrc + 8) < pEnd) {
                  *(INT64*)pDest = *(INT64*)pSrc;
                  pDest += 8;
                  pSrc += 8;
               }
               while (pSrc < pEnd) {
                  *pDest++ = *pSrc++;
               }
               startRow++;
            }
            break;

         }

      }
   }
}

//-----------------------------------
// Input1: the recordarray to convert
// Input2: int64 array of offsets
// Input3: list of arrays pre allocated
PyObject*
RecordArrayToColMajor(PyObject* self, PyObject* args) {
   PyArrayObject* inArr = NULL;
   PyArrayObject* offsetArr = NULL;
   PyArrayObject* arrArr = NULL;

   if (!PyArg_ParseTuple(args, "O!O!O!:RecordArrayToColMajor",
      &PyArray_Type, &inArr,
      &PyArray_Type, &offsetArr,
      &PyArray_Type, &arrArr)) {
      return NULL;
   }

   INT64 itemSize = PyArray_ITEMSIZE(inArr);

   if (itemSize != PyArray_STRIDE(inArr, 0)) {
      PyErr_Format(PyExc_ValueError, "RecordArrayToColMajor cannot handle strides");
      return NULL;
   }

   INT64 length = ArrayLength(inArr);
   INT64 numArrays = ArrayLength(arrArr);

   if (numArrays != ArrayLength(offsetArr)) {
      PyErr_Format(PyExc_ValueError, "RecordArrayToColMajor inputs do not match");
      return NULL;
   }


   INT64 totalRows = length;
   INT64* pOffsets = (INT64*)PyArray_BYTES(offsetArr);
   PyArrayObject** ppArrays = (PyArrayObject**)PyArray_BYTES(arrArr);

   stOffsets *pstOffset;
   
   pstOffset = (stOffsets*)WORKSPACE_ALLOC(sizeof(stOffsets) * numArrays);

   for (INT64 i = 0; i < numArrays; i++) {
      pstOffset[i].pData = PyArray_BYTES(ppArrays[i]);
      pstOffset[i].itemsize = PyArray_ITEMSIZE(ppArrays[i]);
   }

   //printf("chunkrows is %lld\n", CHUNKROWS);

   char* pStartOffset = PyArray_BYTES(inArr);
   INT64 startRow = 0;

   if (totalRows > 16384) {
      // Prepare for multithreading
      struct stConvertRec {
         char* pStartOffset;
         INT64 startRow;
         INT64 totalRows;
         stOffsets* pstOffset;
         INT64* pOffsets;
         INT64 numArrays;
         INT64 itemSize;
         INT64 lastRow;
      } stConvert;

      INT64 items = (totalRows + (CHUNKSIZE - 1)) / CHUNKSIZE;

      stConvert.pStartOffset = pStartOffset;
      stConvert.startRow = startRow;
      stConvert.totalRows = totalRows;
      stConvert.pstOffset = pstOffset;
      stConvert.pOffsets = pOffsets;
      stConvert.numArrays = numArrays;
      stConvert.itemSize = itemSize;
      stConvert.lastRow = items - 1;

      auto lambdaConvertRecCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
         stConvertRec* callbackArg = (stConvertRec*)callbackArgT;
         INT64 startRow = callbackArg->startRow + (workIndex * CHUNKSIZE);
         INT64 totalRows = startRow + CHUNKSIZE;

         if (totalRows > callbackArg->totalRows) {
            totalRows = callbackArg->totalRows;
         }

         ConvertRecArray(
            callbackArg->pStartOffset,
            startRow,
            totalRows,
            callbackArg->pstOffset,
            callbackArg->pOffsets,
            callbackArg->numArrays,
            callbackArg->itemSize);

         LOGGING("[%d] %lld completed\n", core, workIndex);
         return TRUE;
      };

      g_cMathWorker->DoMultiThreadedWork((int)items, lambdaConvertRecCallback, &stConvert);

   }
   else {
      ConvertRecArray(pStartOffset, startRow, totalRows, pstOffset, pOffsets, numArrays, itemSize);
   }

   WORKSPACE_FREE(pstOffset);

   RETURN_NONE;
}

