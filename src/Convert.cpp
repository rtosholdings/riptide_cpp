#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "Convert.h"
#include "MultiKey.h"
#include "Recycler.h"
#include "Reduce.h"

#include <algorithm>

#define LOGGING(...)
//#define LOGGING printf

#if defined(_MSC_VER) && _MSC_VER < 1910
// The VS2015 compiler doesn't provide the _mm256_extract_epi64() intrinsic, even though
// that intrinsic is supposed to be available as part of AVX support. Define a compatible
// version of the function using intrinsics that _are_ available in that compiler.

// Extract a 64-bit integer from a, selected with imm8.
#define _mm256_extract_epi64(a, imm8) _mm_extract_epi64(_mm256_extracti128_si256((a), imm8 / 2), imm8 % 2)

#endif   // _MSC_VER


// SIMD conversion functions for integers
//extern __m256i __cdecl _mm256_cvtepi8_epi16(__m128i);
//extern __m256i __cdecl _mm256_cvtepi8_epi32(__m128i);
//extern __m256i __cdecl _mm256_cvtepi8_epi64(__m128i);
//extern __m256i __cdecl _mm256_cvtepi16_epi32(__m128i);
//extern __m256i __cdecl _mm256_cvtepi16_epi64(__m128i);
//extern __m256i __cdecl _mm256_cvtepi32_epi64(__m128i);
//
//extern __m256i __cdecl _mm256_cvtepu8_epi16(__m128i);
//extern __m256i __cdecl _mm256_cvtepu8_epi32(__m128i);
//extern __m256i __cdecl _mm256_cvtepu8_epi64(__m128i);
//extern __m256i __cdecl _mm256_cvtepu16_epi32(__m128i);
//extern __m256i __cdecl _mm256_cvtepu16_epi64(__m128i);
//extern __m256i __cdecl _mm256_cvtepu32_epi64(__m128i);
typedef void(*CONVERT_SAFE)(void* pDataIn, void* pDataOut, INT64 len, void* pBadInput1, void* pBadOutput1, INT64 strideIn, INT64 strideOut);
typedef void(*MASK_CONVERT_SAFE)(void* pDataIn, void* pDataOut, INT8* pMask, INT64 len, void* pBadInput1, void* pBadOutput1);
typedef void(*CONVERT_SAFE_STRING)(void* pDataIn, void* pDataOut, INT64 len, INT64 inputItemSize, INT64 outputItemSize);

static void ConvertSafeStringCopy(void* pDataIn, void* pDataOut, INT64 len, INT64 inputItemSize, INT64 outputItemSize) {
   LOGGING("String convert %lld %lld\n", inputItemSize, outputItemSize);
   if (inputItemSize == outputItemSize) {
      // straight memcpy
      memcpy(pDataOut, pDataIn, len * inputItemSize);
   }
   else {
      if (inputItemSize < outputItemSize) {
         char* pOut = (char*)pDataOut;
         char* pIn = (char*)pDataIn;
         INT64 remain = outputItemSize - inputItemSize;

         if (inputItemSize >= 8) {
            for (INT64 i = 0; i < len; i++) {
               memcpy(pOut, pIn, inputItemSize);
               pOut += inputItemSize;
               for (INT64 j = 0; j < remain; j++) {
                  pOut[j] = 0;
               }

               pOut += remain;
               pIn += inputItemSize;
            }

         }
         else {
            for (INT64 i = 0; i < len; i++) {
               for (INT64 j = 0; j < inputItemSize; j++) {
                  pOut[j] = pIn[j];
               }
               pOut += inputItemSize;

               // consider memset
               for (INT64 j = 0; j < remain; j++) {
                  pOut[j] = 0;
               }

               pOut += remain;
               pIn += inputItemSize;
            }

         }
      }
      else {
         // currently not possible (clipping input)
         char* pOut = (char*)pDataOut;
         char* pIn = (char*)pDataIn;

         for (INT64 i = 0; i < len; i++) {
            memcpy(pOut, pIn, outputItemSize);
            pOut += outputItemSize;
            pIn += inputItemSize;
         }

      }
   }
}

//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// thus <float, int32> converts a float to an int32
template<typename T, typename U>
class ConvertBase {
public:
   ConvertBase() {};
   ~ConvertBase() {};

   static void PutMaskFast(void* pDataIn, void* pDataOut, INT8* pMask, INT64 len, void* pBadInput1, void* pBadOutput1) {
      T* pIn = (T*)pDataIn;
      T* pOut = (T*)pDataOut;

      // TODO can be made faster by pulling 8 bytes at once
      for (int i = 0; i < len; i++) {
         if (pMask[i]) {
            pOut[i] = pIn[i];
         }
      }
   }

   static void PutMaskCopy(void* pDataIn, void* pDataOut, INT8* pMask, INT64 len, void* pBadInput1, void* pBadOutput1) {
      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;
      T pBadValueIn = *(T*)pBadInput1;
      U pBadValueOut = *(U*)pBadOutput1;

      for (int i = 0; i < len; i++) {
         if (pMask[i]) {
            if (pIn[i] != pBadValueIn) {
               pOut[i] = (U)pIn[i];
            }
            else {
               pOut[i] = pBadValueOut;
            }
         }
      }
   }

   static void PutMaskCopyBool(void* pDataIn, void* pDataOut, INT8* pMask, INT64 len, void* pBadInput1, void* pBadOutput1) {
      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;
      T pBadValueIn = *(T*)pBadInput1;
      U pBadValueOut = *(U*)pBadOutput1;

      for (int i = 0; i < len; i++) {
         if (pMask[i]) {
            pOut[i] = pIn[i] != 0;
         }
      }
   }


   static void PutMaskCopyFloat(void* pDataIn, void* pDataOut, INT8* pMask, INT64 len, void* pBadInput1, void* pBadOutput1) {
      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;
      U pBadValueOut = *(U*)pBadOutput1;

      for (int i = 0; i < len; i++) {
         if (pMask[i]) {
            if (pIn[i] == pIn[i]) {
               pOut[i] = (U)pIn[i];
            }
            else {
               pOut[i] = pBadValueOut;
            }
         }
      }
   }

   // Pass in one vector and returns converted vector
   // Used for operations like C = A + B
   //typedef void(*ANY_TWO_FUNC)(void* pDataIn, void* pDataIn2, void* pDataOut, INT64 len, INT32 scalarMode);
   //typedef void(*ANY_ONE_FUNC)(void* pDataIn, void* pDataOut, INT64 len);
   static void OneStubConvert(void* pDataIn, void* pDataOut, INT64 len, INT64 strideIn, INT64 strideOut) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      if (strideIn == sizeof(T) && strideOut == sizeof(U)) {

         // How to handle nan conversions?
         // NAN converts to MININT (for float --> int conversion)
         // then the reverse, MIININT converts to NAN (for int --> float conversion)
         // convert from int --> float
         // check for NPY_MIN_INT64
         for (INT64 i = 0; i < len; i++) {
            pOut[i] = (U)pIn[i];
         }
      }
      else {
         // Strided loop
         U* pEndOut = (U*)((char*)pOut + (len * strideOut));
         while (pOut != pEndOut) {
            *pOut = (U)*pIn;
            pIn = STRIDE_NEXT(T, pIn, strideIn);
            pOut = STRIDE_NEXT(U, pOut, strideOut);
         }
      }
   }


   static void OneStubConvertSafeCopy(void* pDataIn, void* pDataOut, INT64 len, void* pBadInput1, void* pBadOutput1, INT64 strideIn, INT64 strideOut) {
      // include memcpy with stride
      if (strideIn == sizeof(T) && strideOut == sizeof(U)) {
         memcpy(pDataOut, pDataIn, len * sizeof(U));
      }
      else {
         T* pIn = (T*)pDataIn;
         U* pOut = (U*)pDataOut;

         // Strided loop
         U* pEndOut = (U*)((char*)pOut + (len * strideOut));
         while (pOut != pEndOut) {
            *pOut = *pIn;
            pIn = STRIDE_NEXT(T, pIn, strideIn);
            pOut = STRIDE_NEXT(U, pOut, strideOut);
         }
      }
   }

   //------------------------------------------------------------------------
   // Designed NOT to preserve sentinels
   static void OneStubConvertUnsafe(void* pDataIn, void* pDataOut, INT64 len, void* pBadInput1, void* pBadOutput1, INT64 strideIn, INT64 strideOut) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      if (strideIn == sizeof(T) && strideOut == sizeof(U)) {
         for (INT64 i = 0; i < len; i++) {
            pOut[i] = (U)pIn[i];
         }
      }
      else {
         // Strided loop
         U* pEndOut = (U*)((char*)pOut + (len * strideOut));
         while (pOut != pEndOut) {
            *pOut = (U)*pIn;
            pIn = STRIDE_NEXT(T, pIn, strideIn);
            pOut = STRIDE_NEXT(U, pOut, strideOut);
         }
      }

   }

   //------------------------------------------------------------------------
   // Designed to preserve sentinels
   // NOTE: discussion on on what happens with UINT8 conversion (may or may not ignore 0xFF on conversion since so common)
   static void OneStubConvertSafe(void* pDataIn, void* pDataOut, INT64 len, void* pBadInput1, void* pBadOutput1, INT64 strideIn, INT64 strideOut) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;
      T pBadValueIn = *(T*)pBadInput1;
      U pBadValueOut = *(U*)pBadOutput1;

      // How to handle nan conversions?
      // NAN converts to MININT (for float --> int conversion)
      // then the reverse, MIININT converts to NAN (for int --> float conversion)
      // convert from int --> float
      // check for NPY_MIN_INT64
      if (strideIn == sizeof(T) && strideOut == sizeof(U)) {
         for (INT64 i = 0; i < len; i++) {
            if (pIn[i] != pBadValueIn) {
               pOut[i] = (U)pIn[i];
            }
            else {
               pOut[i] = pBadValueOut;
            }
         }
      }
      else {
         // Strided loop
         U* pEndOut = (U*)((char*)pOut + (len * strideOut));
         while (pOut != pEndOut) {
            if (*pIn != pBadValueIn) {
               *pOut = (U)*pIn;
            }
            else {
               *pOut = pBadValueOut;
            }
            pIn = STRIDE_NEXT(T, pIn, strideIn);
            pOut = STRIDE_NEXT(U, pOut, strideOut);
         }
      }
   }


   static void OneStubConvertSafeFloatToDouble(void* pDataIn, void* pDataOut, INT64 len, void* pBadInput1, void* pBadOutput1, INT64 strideIn, INT64 strideOut) {

      float* pIn = (float*)pDataIn;
      double* pOut = (double*)pDataOut;

      if (strideIn == sizeof(float) && strideOut == sizeof(double)) {
         const double* pEndOut = (double*)((char*)pOut + (len * strideOut));
         const double* pEndOut8 = pEndOut - 8;
         while (pOut <= pEndOut8) {
            __m256 m0 = _mm256_loadu_ps(pIn);
            _mm256_storeu_pd(pOut, _mm256_cvtps_pd(_mm256_extractf128_ps(m0,0)));
            _mm256_storeu_pd(pOut + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(m0,1)));
            pOut += 8;
            pIn += 8;
         }
         while (pOut != pEndOut) {
            *pOut++ = (double)*pIn++;
         }
      }
      else {
         // Strided loop
         double* pEndOut = (double*)((char*)pOut + (len * strideOut));
         while (pOut != pEndOut) {
            *pOut = (double)*pIn;
            pIn = STRIDE_NEXT(float, pIn, strideIn);
            pOut = STRIDE_NEXT(double, pOut, strideOut);
         }
      }
   }


   static void OneStubConvertSafeDoubleToFloat(void* pDataIn, void* pDataOut, INT64 len, void* pBadInput1, void* pBadOutput1, INT64 strideIn, INT64 strideOut) {

      double* pIn = (double*)pDataIn;
      float* pOut = (float*)pDataOut;

      if (strideIn == sizeof(double) && strideOut == sizeof(float)) {
         for (INT64 i = 0; i < len; i++) {
            pOut[i] = (float)pIn[i];
         }
      }
      else {
         // Strided loop
         float* pEndOut = (float*)((char*)pOut + (len * strideOut));
         while (pOut != pEndOut) {
            *pOut = (float)*pIn;
            pIn = STRIDE_NEXT(double, pIn, strideIn);
            pOut = STRIDE_NEXT(float, pOut, strideOut);
         }
      }
   }


   static void OneStubConvertSafeFloat(void* pDataIn, void* pDataOut, INT64 len, void* pBadInput1, void* pBadOutput1, INT64 strideIn, INT64 strideOut) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;
      T pBadValueIn = *(T*)pBadInput1;
      U pBadValueOut = *(U*)pBadOutput1;

      // How to handle nan conversions?
      // NAN converts to MININT (for float --> int conversion)
      // then the reverse, MIININT converts to NAN (for int --> float conversion)
      // convert from int --> float
      // check for NPY_MIN_INT64
      if (strideIn == sizeof(T) && strideOut == sizeof(U)) {
         for (INT64 i = 0; i < len; i++) {

            if (std::isfinite(pIn[i]) && (pIn[i] != pBadValueIn)) {
               pOut[i] = (U)pIn[i];
            }
            else {
               pOut[i] = pBadValueOut;
            }
         }
      }
      else {
         // Strided loop
         U* pEndOut = (U*)((char*)pOut + (len * strideOut));
         while (pOut != pEndOut) {
            if (std::isfinite(*pIn) && (*pIn != pBadValueIn)) {
               *pOut = (U)*pIn;
            }
            else {
               *pOut = pBadValueOut;
            }
            pIn = STRIDE_NEXT(T, pIn, strideIn);
            pOut = STRIDE_NEXT(U, pOut, strideOut);
         }
      }
   }


   static void OneStubConvertSafeBool(void* pDataIn, void* pDataOut, INT64 len, void* pBadInput1, void* pBadOutput1, INT64 strideIn, INT64 strideOut) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      // How to handle nan conversions?
      // NAN converts to MININT (for float --> int conversion)
      // then the reverse, MIININT converts to NAN (for int --> float conversion)
      // convet from float --> int
      if (strideIn == sizeof(T) && strideOut == sizeof(U)) {
         for (INT64 i = 0; i < len; i++) {
            pOut[i] = (U)(pIn[i] != 0);
         }
      }
      else {
         // Strided loop
         U* pEndOut = (U*)((char*)pOut + (len * strideOut));
         while (pOut != pEndOut) {
            *pOut = (U)(*pIn != 0);
            pIn = STRIDE_NEXT(T, pIn, strideIn);
            pOut = STRIDE_NEXT(U, pOut, strideOut);
         }
      }
   }


   static void OneStubConvertBool(void* pDataIn, void* pDataOut, INT64 len, INT64 strideIn, INT64 strideOut) {

      T* pIn = (T*)pDataIn;
      U* pOut = (U*)pDataOut;

      if (strideIn == sizeof(T) && strideOut == sizeof(U)) {
         // How to handle nan conversions?
         // NAN converts to MININT (for float --> int conversion)
         // then the reverse, MIININT converts to NAN (for int --> float conversion)
         // convet from float --> int
         for (INT64 i = 0; i < len; i++) {
            pOut[i] = (U)(pIn[i] != 0);
         }
      }
      else {
         // Strided loop
         U* pEndOut = (U*)((char*)pOut + (len * strideOut));
         while (pOut != pEndOut) {
            *pOut = (U)(*pIn != 0);
            pIn = STRIDE_NEXT(T, pIn, strideIn);
            pOut = STRIDE_NEXT(U, pOut, strideOut);
         }
      }
   }

};


template<typename T>
static UNARY_FUNC GetConversionStep2(int outputType) {
   switch (outputType) {
   case NPY_BOOL:   return ConvertBase<T, bool>::OneStubConvertBool;
   case NPY_FLOAT:  return ConvertBase<T, float>::OneStubConvert;
   case NPY_DOUBLE: return ConvertBase<T, double>::OneStubConvert;
   case NPY_LONGDOUBLE: return ConvertBase<T, long double>::OneStubConvert;
   case NPY_BYTE:   return ConvertBase<T, INT8>::OneStubConvert;
   case NPY_INT16:  return ConvertBase<T, INT16>::OneStubConvert;
   CASE_NPY_INT32:  return ConvertBase<T, INT32>::OneStubConvert;
   CASE_NPY_INT64:  return ConvertBase<T, INT64>::OneStubConvert;
   case NPY_UBYTE:  return ConvertBase<T, UINT8>::OneStubConvert;
   case NPY_UINT16: return ConvertBase<T, UINT16>::OneStubConvert;
   CASE_NPY_UINT32: return ConvertBase<T, UINT32>::OneStubConvert;
   CASE_NPY_UINT64: return ConvertBase<T, UINT64>::OneStubConvert;
   }
   return NULL;

}

template<typename T>
static CONVERT_SAFE GetConversionStep2Safe(int outputType) {
   switch (outputType) {
   case NPY_BOOL:   return ConvertBase<T, bool>::OneStubConvertSafeBool;
   case NPY_FLOAT:  return ConvertBase<T, float>::OneStubConvertSafe;
   case NPY_DOUBLE: return ConvertBase<T, double>::OneStubConvertSafe;
   case NPY_LONGDOUBLE: return ConvertBase<T, long double>::OneStubConvertSafe;
   case NPY_BYTE:   return ConvertBase<T, INT8>::OneStubConvertSafe;
   case NPY_INT16:  return ConvertBase<T, INT16>::OneStubConvertSafe;
   CASE_NPY_INT32:  return ConvertBase<T, INT32>::OneStubConvertSafe;
   CASE_NPY_INT64:  return ConvertBase<T, INT64>::OneStubConvertSafe;
   case NPY_UBYTE:  return ConvertBase<T, UINT8>::OneStubConvertSafe;
   case NPY_UINT16: return ConvertBase<T, UINT16>::OneStubConvertSafe;
   CASE_NPY_UINT32: return ConvertBase<T, UINT32>::OneStubConvertSafe;
   CASE_NPY_UINT64: return ConvertBase<T, UINT64>::OneStubConvertSafe;
   }
   return NULL;

}

//-----------------------------------------------------------
// Used when converting from a UINT8 which has no sentinel (discussion point)
template<typename T>
static CONVERT_SAFE GetConversionStep2Unsafe(int outputType) {
   switch (outputType) {
   case NPY_BOOL:   return ConvertBase<T, bool>::OneStubConvertSafeBool;
   case NPY_FLOAT:  return ConvertBase<T, float>::OneStubConvertUnsafe;
   case NPY_DOUBLE: return ConvertBase<T, double>::OneStubConvertUnsafe;
   case NPY_LONGDOUBLE: return ConvertBase<T, long double>::OneStubConvertUnsafe;
   case NPY_BYTE:   return ConvertBase<T, INT8>::OneStubConvertUnsafe;
   case NPY_INT16:  return ConvertBase<T, INT16>::OneStubConvertUnsafe;
   CASE_NPY_INT32:  return ConvertBase<T, INT32>::OneStubConvertUnsafe;
   CASE_NPY_INT64:  return ConvertBase<T, INT64>::OneStubConvertUnsafe;
   case NPY_UBYTE:  return ConvertBase<T, UINT8>::OneStubConvertUnsafe;
   case NPY_UINT16: return ConvertBase<T, UINT16>::OneStubConvertUnsafe;
   CASE_NPY_UINT32: return ConvertBase<T, UINT32>::OneStubConvertUnsafe;
   CASE_NPY_UINT64: return ConvertBase<T, UINT64>::OneStubConvertUnsafe;
   }
   return NULL;

}


template<typename T>
static CONVERT_SAFE GetConversionStep2SafeFromFloat(int outputType) {
   switch (outputType) {
   case NPY_BOOL:   return ConvertBase<T, bool>::OneStubConvertSafeBool;
   case NPY_FLOAT:  return ConvertBase<T, float>::OneStubConvertSafeFloat;
   case NPY_DOUBLE: return ConvertBase<T, double>::OneStubConvertSafeFloatToDouble;  // very common
   case NPY_LONGDOUBLE: return ConvertBase<T, long double>::OneStubConvertSafeFloat;
   case NPY_BYTE:   return ConvertBase<T, INT8>::OneStubConvertSafeFloat;
   case NPY_INT16:  return ConvertBase<T, INT16>::OneStubConvertSafeFloat;
   CASE_NPY_INT32:  return ConvertBase<T, INT32>::OneStubConvertSafeFloat;
   CASE_NPY_INT64:  return ConvertBase<T, INT64>::OneStubConvertSafeFloat;
   case NPY_UBYTE:  return ConvertBase<T, UINT8>::OneStubConvertSafeFloat;
   case NPY_UINT16: return ConvertBase<T, UINT16>::OneStubConvertSafeFloat;
   CASE_NPY_UINT32: return ConvertBase<T, UINT32>::OneStubConvertSafeFloat;
   CASE_NPY_UINT64: return ConvertBase<T, UINT64>::OneStubConvertSafeFloat;
   }
   return NULL;

}

template<typename T>
static CONVERT_SAFE GetConversionStep2SafeFromDouble(int outputType) {
   switch (outputType) {
   case NPY_BOOL:   return ConvertBase<T, bool>::OneStubConvertSafeBool;
   case NPY_FLOAT:  return ConvertBase<T, float>::OneStubConvertSafeDoubleToFloat;   // very common
   case NPY_DOUBLE: return ConvertBase<T, double>::OneStubConvertSafeFloat;
   case NPY_LONGDOUBLE: return ConvertBase<T, long double>::OneStubConvertSafeFloat;
   case NPY_BYTE:   return ConvertBase<T, INT8>::OneStubConvertSafeFloat;
   case NPY_INT16:  return ConvertBase<T, INT16>::OneStubConvertSafeFloat;
   CASE_NPY_INT32:  return ConvertBase<T, INT32>::OneStubConvertSafeFloat;
   CASE_NPY_INT64:  return ConvertBase<T, INT64>::OneStubConvertSafeFloat;
   case NPY_UBYTE:  return ConvertBase<T, UINT8>::OneStubConvertSafeFloat;
   case NPY_UINT16: return ConvertBase<T, UINT16>::OneStubConvertSafeFloat;
   CASE_NPY_UINT32: return ConvertBase<T, UINT32>::OneStubConvertSafeFloat;
   CASE_NPY_UINT64: return ConvertBase<T, UINT64>::OneStubConvertSafeFloat;
   }
   return NULL;

}


template<typename T>
static CONVERT_SAFE GetConversionStep2SafeFloat(int outputType) {
   switch (outputType) {
   case NPY_BOOL:   return ConvertBase<T, bool>::OneStubConvertSafeBool;
   case NPY_FLOAT:  return ConvertBase<T, float>::OneStubConvertSafeFloat;
   case NPY_DOUBLE: return ConvertBase<T, double>::OneStubConvertSafeFloat;
   case NPY_LONGDOUBLE: return ConvertBase<T, long double>::OneStubConvertSafeFloat;
   case NPY_BYTE:   return ConvertBase<T, INT8>::OneStubConvertSafeFloat;
   case NPY_INT16:  return ConvertBase<T, INT16>::OneStubConvertSafeFloat;
   CASE_NPY_INT32:  return ConvertBase<T, INT32>::OneStubConvertSafeFloat;
   CASE_NPY_INT64:  return ConvertBase<T, INT64>::OneStubConvertSafeFloat;
   case NPY_UBYTE:  return ConvertBase<T, UINT8>::OneStubConvertSafeFloat;
   case NPY_UINT16: return ConvertBase<T, UINT16>::OneStubConvertSafeFloat;
   CASE_NPY_UINT32: return ConvertBase<T, UINT32>::OneStubConvertSafeFloat;
   CASE_NPY_UINT64: return ConvertBase<T, UINT64>::OneStubConvertSafeFloat;
   }
   return NULL;

}


static CONVERT_SAFE GetConversionFunctionSafeCopy(int inputType) {

   switch (inputType) {
   case NPY_BYTE:
   case NPY_UBYTE:
   case NPY_BOOL:   return ConvertBase<INT8, INT8>::OneStubConvertSafeCopy;

   case NPY_INT16:
   case NPY_UINT16:  return ConvertBase<INT16, INT16>::OneStubConvertSafeCopy;

   CASE_NPY_INT32:
   CASE_NPY_UINT32:
   case NPY_FLOAT:  return ConvertBase<INT32, INT32>::OneStubConvertSafeCopy;

   CASE_NPY_INT64:
   CASE_NPY_UINT64:
   case NPY_DOUBLE: return ConvertBase<INT64, INT64>::OneStubConvertSafeCopy;

   case NPY_LONGDOUBLE: return ConvertBase<long double, long double>::OneStubConvertSafeCopy;

   }
   return NULL;
}


static CONVERT_SAFE GetConversionFunctionSafe(int inputType, int outputType) {

   // check for same type -- which is shorthand for copy
   if (inputType == outputType) {
      return GetConversionFunctionSafeCopy(inputType);
   }

   switch (inputType) {
   //case NPY_BOOL:   return GetConversionStep2Safe<bool>(outputType);
   case NPY_BOOL:   return GetConversionStep2Safe<INT8>(outputType);
   case NPY_FLOAT:  return GetConversionStep2SafeFromFloat<float>(outputType);
   case NPY_DOUBLE: return GetConversionStep2SafeFromDouble<double>(outputType);
   case NPY_LONGDOUBLE: return GetConversionStep2SafeFloat<long double>(outputType);
   case NPY_BYTE:   return GetConversionStep2Safe<INT8>(outputType);
   case NPY_INT16:  return GetConversionStep2Safe<INT16>(outputType);
   CASE_NPY_INT32:  return GetConversionStep2Safe<INT32>(outputType);
   CASE_NPY_INT64:  return GetConversionStep2Safe<INT64>(outputType);

   // DISCUSSION -- UINT8 and the value 255 or 0xFF will not be a sentinel
   //case NPY_UBYTE:  return GetConversionStep2Unsafe<UINT8>(outputType);
   case NPY_UBYTE:  return GetConversionStep2Safe<UINT8>(outputType);

   case NPY_UINT16: return GetConversionStep2Safe<UINT16>(outputType);
   CASE_NPY_UINT32: return GetConversionStep2Safe<UINT32>(outputType);
   CASE_NPY_UINT64: return GetConversionStep2Safe<UINT64>(outputType);

   }
   return NULL;
}


static CONVERT_SAFE GetConversionFunctionUnsafe(int inputType, int outputType) {

   // check for same type -- which is shorthand for copy
   if (inputType == outputType) {
      return GetConversionFunctionSafeCopy(inputType);
   }

   switch (inputType) {
      //case NPY_BOOL:   return GetConversionStep2Safe<bool>(outputType);
   case NPY_BOOL:   return GetConversionStep2Unsafe<INT8>(outputType);
   case NPY_FLOAT:  return GetConversionStep2Unsafe<float>(outputType);
   case NPY_DOUBLE: return GetConversionStep2Unsafe<double>(outputType);
   case NPY_LONGDOUBLE: return GetConversionStep2Unsafe<long double>(outputType);
   case NPY_BYTE:   return GetConversionStep2Unsafe<INT8>(outputType);
   case NPY_INT16:  return GetConversionStep2Unsafe<INT16>(outputType);
   CASE_NPY_INT32:  return GetConversionStep2Unsafe<INT32>(outputType);
   CASE_NPY_INT64:  return GetConversionStep2Unsafe<INT64>(outputType);
   case NPY_UBYTE:  return GetConversionStep2Unsafe<UINT8>(outputType);
   case NPY_UINT16: return GetConversionStep2Unsafe<UINT16>(outputType);
   CASE_NPY_UINT32: return GetConversionStep2Unsafe<UINT32>(outputType);
   CASE_NPY_UINT64: return GetConversionStep2Unsafe<UINT64>(outputType);

   }
   return NULL;
}



template<typename T>
static MASK_CONVERT_SAFE GetConversionPutMask2Float(int outputType) {
   switch (outputType) {
   case NPY_BOOL:   return ConvertBase<T, bool>::PutMaskCopyBool;
   case NPY_FLOAT:  return ConvertBase<T, float>::PutMaskCopyFloat;
   case NPY_DOUBLE: return ConvertBase<T, double>::PutMaskCopyFloat;
   case NPY_LONGDOUBLE: return ConvertBase<T, long double>::PutMaskCopyFloat;
   case NPY_BYTE:   return ConvertBase<T, INT8>::PutMaskCopyFloat;
   case NPY_INT16:  return ConvertBase<T, INT16>::PutMaskCopyFloat;
   CASE_NPY_INT32:  return ConvertBase<T, INT32>::PutMaskCopyFloat;
   CASE_NPY_INT64:  return ConvertBase<T, INT64>::PutMaskCopyFloat;
   case NPY_UBYTE:  return ConvertBase<T, UINT8>::PutMaskCopyFloat;
   case NPY_UINT16: return ConvertBase<T, UINT16>::PutMaskCopyFloat;
   CASE_NPY_UINT32: return ConvertBase<T, UINT32>::PutMaskCopyFloat;
   CASE_NPY_UINT64: return ConvertBase<T, UINT64>::PutMaskCopyFloat;
   }
   return NULL;

}


template<typename T>
static MASK_CONVERT_SAFE GetConversionPutMask2(int outputType) {
   switch (outputType) {
   case NPY_BOOL:   return ConvertBase<T, bool>::PutMaskCopyBool;
   case NPY_FLOAT:  return ConvertBase<T, float>::PutMaskCopy;
   case NPY_DOUBLE: return ConvertBase<T, double>::PutMaskCopy;
   case NPY_LONGDOUBLE: return ConvertBase<T, long double>::PutMaskCopy;
   case NPY_BYTE:   return ConvertBase<T, INT8>::PutMaskCopy;
   case NPY_INT16:  return ConvertBase<T, INT16>::PutMaskCopy;
   CASE_NPY_INT32:  return ConvertBase<T, INT32>::PutMaskCopy;
   CASE_NPY_INT64:  return ConvertBase<T, INT64>::PutMaskCopy;
   case NPY_UBYTE:  return ConvertBase<T, UINT8>::PutMaskCopy;
   case NPY_UINT16: return ConvertBase<T, UINT16>::PutMaskCopy;
   CASE_NPY_UINT32: return ConvertBase<T, UINT32>::PutMaskCopy;
   CASE_NPY_UINT64: return ConvertBase<T, UINT64>::PutMaskCopy;
   }
   return NULL;

}

static MASK_CONVERT_SAFE GetConversionPutMask(int inputType, int outputType) {

   // check for same type -- which is shorthand for copy
   if (inputType == outputType) {
      switch (inputType) {
      case NPY_BYTE:
      case NPY_UBYTE:
      case NPY_BOOL:   return ConvertBase<INT8, INT8>::PutMaskFast;

      case NPY_INT16:
      case NPY_UINT16:  return ConvertBase<INT16, INT16>::PutMaskFast;

      CASE_NPY_INT32:
      CASE_NPY_UINT32:
      case NPY_FLOAT:  return ConvertBase<INT32, INT32>::PutMaskFast;

      CASE_NPY_INT64:
      CASE_NPY_UINT64:
      case NPY_DOUBLE: return ConvertBase<INT64, INT64>::PutMaskFast;

      case NPY_LONGDOUBLE: return ConvertBase<long double, long double>::PutMaskFast;

      }
   }

   switch (inputType) {
      //case NPY_BOOL:   return GetConversionStep2Safe<bool>(outputType);
   case NPY_BOOL:   return GetConversionPutMask2<INT8>(outputType);
   case NPY_FLOAT:  return GetConversionPutMask2Float<float>(outputType);
   case NPY_DOUBLE: return GetConversionPutMask2Float<double>(outputType);
   case NPY_LONGDOUBLE: return GetConversionPutMask2Float<long double>(outputType);
   case NPY_BYTE:   return GetConversionPutMask2<INT8>(outputType);
   case NPY_INT16:  return GetConversionPutMask2<INT16>(outputType);
   CASE_NPY_INT32:  return GetConversionPutMask2<INT32>(outputType);
   CASE_NPY_INT64:  return GetConversionPutMask2<INT64>(outputType);

      // DISCUSSION -- UINT8 and the value 255 or 0xFF will not be a sentinel
      //case NPY_UBYTE:  return GetConversionStep2Unsafe<UINT8>(outputType);
   case NPY_UBYTE:  return GetConversionPutMask2<UINT8>(outputType);

   case NPY_UINT16: return GetConversionPutMask2<UINT16>(outputType);
   CASE_NPY_UINT32: return GetConversionPutMask2<UINT32>(outputType);
   CASE_NPY_UINT64: return GetConversionPutMask2<UINT64>(outputType);

   }
   return NULL;
}


//=====================================================================================
//--------------------------------------------------------------------
struct CONVERT_CALLBACK {
   CONVERT_SAFE anyConvertCallback;
   char* pDataIn;
   char* pDataOut;
   void* pBadInput1;
   void* pBadOutput1;

   INT64 typeSizeIn;
   INT64 typeSizeOut;

} stConvertCallback;

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static BOOL ConvertThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex) {
   BOOL didSomeWork = FALSE;
   CONVERT_CALLBACK* Callback = (CONVERT_CALLBACK*)pstWorkerItem->WorkCallbackArg;

   char* pDataIn = (char *)Callback->pDataIn;
   char* pDataOut = (char*)Callback->pDataOut;
   INT64 lenX;
   INT64 workBlock;

   // As long as there is work to do
   while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

      INT64 inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeIn;
      INT64 outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;

      Callback->anyConvertCallback(pDataIn + inputAdj, pDataOut + outputAdj, lenX, Callback->pBadInput1, Callback->pBadOutput1, Callback->typeSizeIn, Callback->typeSizeOut);

      // Indicate we completed a block
      didSomeWork = TRUE;

      // tell others we completed this work block
      pstWorkerItem->CompleteWorkBlock();
      //printf("|%d %d", core, (int)workBlock);
   }

   return didSomeWork;

}

//=====================================================================================
//
//
void* GetInvalid(int dtype) {
   void* pBadInput = GetDefaultForType(dtype);
   if (dtype == NPY_BOOL) {
      // We do not want FALSE to become a sentinel
      pBadInput = GetDefaultForType(NPY_INT8);
   }
   return pBadInput;
}

//=====================================================================================
// Input: Two parameters
// Arg1: array to convert
// Arg2: dtype.num of the output array
//
// Returns converted array (or NULL, if an error occurs).
// The returned array is expected to have the same shape as the input array;
// when the input array has either or both the C_CONTIGUOUS or F_CONTIGUOUS flags values
// set, the returned array is expected to have the same flags set.
// NOTE: if they are the same type, special fast routine called
PyObject *
ConvertSafeInternal(
   PyArrayObject* const inArr1,
   const INT64 out_dtype) {

   const INT32 numpyOutType = (INT32)out_dtype;
   const INT32 numpyInType = PyArray_TYPE(inArr1);

   if (numpyOutType < 0 || numpyInType > NPY_LONGDOUBLE || numpyOutType > NPY_LONGDOUBLE) {
      return PyErr_Format(PyExc_ValueError, "ConvertSafe: Don't know how to convert these types %d %d", numpyInType, numpyOutType);
   }

   // TODO: Do we still need the check above? Or can we just rely on GetConversionFunctionSafe() to do any necessary checks?
   const CONVERT_SAFE pFunction = GetConversionFunctionSafe(numpyInType, numpyOutType);
   if (!pFunction)
   {
      return PyErr_Format(PyExc_ValueError, "ConvertSafe: Don't know how to convert these types %d %d", numpyInType, numpyOutType);
   }

   LOGGING("ConvertSafe converting type %d to type %d\n", numpyInType, numpyOutType);

   void* const pDataIn = PyArray_BYTES(inArr1);
   int ndim = PyArray_NDIM(inArr1);
   npy_intp* const dims = PyArray_DIMS(inArr1);

   // Make sure array lengths match
   const INT64 arraySize1 = CalcArrayLength(ndim, dims);
   const INT64 len = arraySize1;

   // Allocate the output array.
   // TODO: Consider using AllocateLikeNumpyArray here instead for simplicity.
   PyArrayObject* outArray = AllocateNumpyArray(ndim, dims, numpyOutType, 0, PyArray_IS_F_CONTIGUOUS(inArr1));
   CHECK_MEMORY_ERROR(outArray);

   // Check if we're out of memory.
   if (!outArray)
   {
      return PyErr_Format(PyExc_MemoryError, "ConvertSafe out of memory");
   }

   void* pDataOut = PyArray_BYTES(outArray);
   void* pBadInput1 = GetInvalid(numpyInType);

   // if output is boolean, bad means FALSE
   void* pBadOutput1 = GetDefaultForType(numpyOutType);

   // Check the strides of both the input and output to make sure we can handle
   INT64 strideIn;
   int directionIn = GetStridesAndContig(inArr1, ndim, strideIn);

   int ndimOut;
   INT64 strideOut;
   int directionOut = GetStridesAndContig(outArray, ndimOut, strideOut);

   // If the input is C and/or F-contiguous, the output should have
   // the same flag(s) set.
   if (directionIn != 0 || directionOut !=0)
   {
      // non-contiguous loop
      // Walk the input, dimension by dimension, getting the stride
      // Check if we can process, else punt to numpy
      if (directionIn == 1 && directionOut == 0) {
         // Row Major 2dim like array with output being fully contiguous
         INT64 innerLen = 1;
         for (int i = directionIn; i < ndim; i++) {
            innerLen *= PyArray_DIM(inArr1, i);
         }
         // TODO: consider dividing the work over multiple threads if innerLen is large enough
         const INT64 outerLen = PyArray_DIM(inArr1, 0);
         const INT64 outerStride = PyArray_STRIDE(inArr1, 0);

         LOGGING("Row Major  innerLen:%lld  outerLen:%lld  outerStride:%lld\n", innerLen, outerLen, outerStride);

         for (INT64 j = 0; j < outerLen; j++) {
            pFunction((char*)pDataIn + (j * outerStride), (char*)pDataOut + (j * innerLen * strideOut), innerLen, pBadInput1, pBadOutput1, strideIn, strideOut);
         }

      }
      else if (directionIn == -1 && directionOut == 0) {
         // Col Major 2dim like array with output being fully contiguous
         INT64 innerLen = 1;
         directionIn = -directionIn;
         for (int i = 0; i < directionIn; i++) {
            innerLen *= PyArray_DIM(inArr1, i);
         }
         // TODO: consider dividing the work over multiple threads if innerLen is large enough
         const INT64 outerLen = PyArray_DIM(inArr1, (ndim - 1));
         const INT64 outerStride = PyArray_STRIDE(inArr1, (ndim - 1));

         LOGGING("Col Major  innerLen:%lld  outerLen:%lld  outerStride:%lld\n", innerLen, outerLen, outerStride);

         for (INT64 j = 0; j < outerLen; j++) {
            pFunction((char*)pDataIn + (j * outerStride), (char*)pDataOut + (j * innerLen * strideOut), innerLen, pBadInput1, pBadOutput1, strideIn, strideOut);
         }
      }
      else {

         // Don't leak the memory we allocated -- free it before raising the Python error and returning.
         RecycleNumpyInternal(outArray);
         // have numpy do the work
         outArray = (PyArrayObject*)PyArray_FROM_OT((PyObject*)inArr1, numpyOutType);
         //return PyErr_Format(PyExc_RuntimeError, "ConvertSafe allocated an output array whose *_CONTIGUOUS flags were not set even though the input array was contiguous. %d %d   out:%d %d  strideIn:%lld  strideOut:%lld", contigIn, ndim, contigOut, ndimOut, strideIn, strideOut);
      }
   }
   else {

      stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(len);

      if (!pWorkItem) {
         // Threading not allowed for this work item, call it directly from main thread
         pFunction(pDataIn, pDataOut, len, pBadInput1, pBadOutput1, strideIn, strideOut);
      }
      else {
         // Each thread will call this routine with the callbackArg
         pWorkItem->DoWorkCallback = ConvertThreadCallback;
         pWorkItem->WorkCallbackArg = &stConvertCallback;

         stConvertCallback.anyConvertCallback = pFunction;
         stConvertCallback.pDataOut = (char*)pDataOut;
         stConvertCallback.pDataIn = (char*)pDataIn;
         stConvertCallback.pBadInput1 = pBadInput1;
         stConvertCallback.pBadOutput1 = pBadOutput1;

         stConvertCallback.typeSizeIn = strideIn;
         stConvertCallback.typeSizeOut = strideOut;

         // This will notify the worker threads of a new work item
         // TODO: Calc how many threads we need to do the conversion (possibly just 3 worker threads is enough)
         g_cMathWorker->WorkMain(pWorkItem, len, 0);
      }
   }
   return (PyObject*)outArray;
}


//=====================================================================================
PyObject *
ConvertSafe(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;
   INT64 out_dtype = 0;

   if (Py_SIZE(args) > 1) {
      PyArrayObject* inObject = (PyArrayObject*)PyTuple_GET_ITEM(args, 0);
      PyObject* inNumber = PyTuple_GET_ITEM(args, 1);
      if (PyLong_CheckExact(inNumber)) {
         INT64 dtypeNum = PyLong_AsLongLong(inNumber);

         if (IsFastArrayOrNumpy(inObject)) {
            PyObject* result =
               ConvertSafeInternal(inObject, dtypeNum);
            return result;
         }
         else {

            PyErr_Format(PyExc_ValueError, "ConvertSafe first argument must be an array not type %s", ((PyObject*)inObject)->ob_type->tp_name);
         }
      }
      else {
         PyErr_Format(PyExc_ValueError, "ConvertSafe second argument must be an integer not type %s", inNumber->ob_type->tp_name);
      }
   }
   else {
      PyErr_Format(PyExc_ValueError, "ConvertSafe must have at least two arguments");

   }

   return NULL;
}


//=====================================================================================
// Input: Two parameters
// Arg1: array to convert
// Arg2: dtype.num of the output array
//
// Returns converted array or nullptr on error.
// NOTE: if they are the same type, special fast routine called
// TODO: Combine ConvertSafeInternal and ConvertUnsafeInternal into a single, templated function --
//       they only differ in whether they use GetConversionFunctionSafe / GetConversionFunctionUnsafe.
PyObject *
ConvertUnsafeInternal(
   PyArrayObject *inArr1,
   INT64 out_dtype) {

   const INT32 numpyOutType = (INT32)out_dtype;
   const INT32 numpyInType = ObjectToDtype(inArr1);

   if (numpyOutType < 0 || numpyInType < 0 || numpyInType > NPY_LONGDOUBLE || numpyOutType > NPY_LONGDOUBLE) {
      return PyErr_Format(PyExc_ValueError, "ConvertUnsafe: Don't know how to convert these types %d %d", numpyInType, numpyOutType);
   }

   // TODO: Do we still need the check above? Or can we just rely on GetConversionFunctionUnsafe() to do any necessary checks?
   CONVERT_SAFE pFunction = GetConversionFunctionUnsafe(numpyInType, numpyOutType);
   if (!pFunction)
   {
      return PyErr_Format(PyExc_ValueError, "ConvertUnsafe: Don't know how to convert these types %d %d", numpyInType, numpyOutType);
   }

   LOGGING("ConvertUnsafe converting type %d to type %d\n", numpyInType, numpyOutType);

   void* pDataIn = PyArray_BYTES(inArr1);

   const int ndim = PyArray_NDIM(inArr1);
   npy_intp* const dims = PyArray_DIMS(inArr1);
   //auto* const inArr1_strides = PyArray_STRIDES(inArr1);
   // TODO: CalcArrayLength probably needs to be fixed to account for any non-default striding
   const INT64 arraySize1 = CalcArrayLength(ndim, dims);
   const INT64 len = arraySize1;

   // Allocate the output array.
   // TODO: Consider using AllocateLikeNumpyArray here instead for simplicity.
   PyArrayObject* outArray = AllocateNumpyArray(ndim, dims, numpyOutType, 0, PyArray_IS_F_CONTIGUOUS(inArr1));
   CHECK_MEMORY_ERROR(outArray);
   if (!outArray)
   {
      return PyErr_Format(PyExc_MemoryError, "ConvertUnsafe out of memory");
   }


   void* pDataOut = PyArray_BYTES(outArray);

   void* pBadInput1 = GetInvalid(numpyInType);

   // if output is boolean, bad means FALSE
   void* pBadOutput1 = GetDefaultForType(numpyOutType);

   stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(len);

   if (!pWorkItem) {
      // Threading not allowed for this work item, call it directly from main thread
      pFunction(pDataIn, pDataOut, len, pBadInput1, pBadOutput1, PyArray_STRIDE(inArr1, 0), PyArray_STRIDE(outArray, 0));
   }
   else {
      // Each thread will call this routine with the callbackArg
      pWorkItem->DoWorkCallback = ConvertThreadCallback;
      pWorkItem->WorkCallbackArg = &stConvertCallback;

      stConvertCallback.anyConvertCallback = pFunction;
      stConvertCallback.pDataOut = (char*)pDataOut;
      stConvertCallback.pDataIn = (char*)pDataIn;
      stConvertCallback.pBadInput1 = pBadInput1;
      stConvertCallback.pBadOutput1 = pBadOutput1;

      stConvertCallback.typeSizeIn = PyArray_STRIDE(inArr1, 0);
      stConvertCallback.typeSizeOut = PyArray_STRIDE(outArray, 0);

      // This will notify the worker threads of a new work item
      g_cMathWorker->WorkMain(pWorkItem, len, 0);
   }

   return (PyObject*)outArray;
}


//=====================================================================================
// Input: Two parameters
// Arg1: array to convert
// Arg2: dtype.num of the output array
//
// Returns converted array
// NOTE: if they are the same type, special fast routine called
PyObject *
ConvertUnsafe(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;
   INT64 out_dtype = 0;

   if (!PyArg_ParseTuple(
      args, "O!L:ConvertUnsafe",
      &PyArray_Type, &inArr1,
      &out_dtype)) {

      return NULL;
   }
   return ConvertUnsafeInternal(inArr1, out_dtype);
}


//=====================================================================================
// COMBINE MASK------------------------------------------------------------------------
//=====================================================================================
typedef void(*COMBINE_MASK)(void* pDataIn, void* pDataOut, INT64 len, INT8* pFilter);

template<typename T>
static void CombineMask(void* pDataInT, void* pDataOutT, INT64 len, INT8* pFilter) {
   T*    pDataIn = (T*)pDataInT;
   T*    pDataOut = (T*)pDataOutT;

   for (INT64 i = 0; i < len; i++) {
      pDataOut[i] = pDataIn[i] * (T)pFilter[i];
   }

}


static COMBINE_MASK GetCombineFunction(int outputType) {
   switch (outputType) {
   case NPY_INT8:   return CombineMask<INT8>;
   case NPY_INT16:  return CombineMask<INT16>;
   CASE_NPY_INT32:  return CombineMask<INT32>;
   CASE_NPY_INT64:  return CombineMask<INT64>;
   }
   return NULL;

}




//--------------------------------------------------------------------
struct COMBINE_CALLBACK {
   COMBINE_MASK anyCombineCallback;
   char* pDataIn;
   char* pDataOut;
   INT8* pFilter;

   INT64 typeSizeOut;

} stCombineCallback;

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static BOOL CombineThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex) {
   BOOL didSomeWork = FALSE;
   COMBINE_CALLBACK* Callback = (COMBINE_CALLBACK*)pstWorkerItem->WorkCallbackArg;

   char* pDataIn = (char *)Callback->pDataIn;
   char* pDataOut = (char*)Callback->pDataOut;
   INT64 lenX;
   INT64 workBlock;

   // As long as there is work to do
   while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

      INT64 inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;
      INT64 filterAdj = pstWorkerItem->BlockSize * workBlock;

      Callback->anyCombineCallback(pDataIn + inputAdj, pDataOut + inputAdj, lenX, Callback->pFilter + filterAdj);

      // Indicate we completed a block
      didSomeWork = TRUE;

      // tell others we completed this work block
      pstWorkerItem->CompleteWorkBlock();
      //printf("|%d %d", core, (int)workBlock);
   }

   return didSomeWork;

}



//=====================================================================================
// Input: Two parameters
// Arg1: Index array
// Arg2: Boolean array to merge
//
// Returns new index array with 0
PyObject *
CombineFilter(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *inFilter = NULL;

   INT64 out_dtype = 0;

   if (!PyArg_ParseTuple(
      args, "O!O!:CombineFilter",
      &PyArray_Type, &inArr1,
      &PyArray_Type, &inFilter)) {

      return NULL;
   }

   INT32 numpyOutType = PyArray_TYPE(inArr1);
   void* pDataIn = PyArray_BYTES(inArr1);
   int ndim = PyArray_NDIM(inArr1);
   npy_intp* dims = PyArray_DIMS(inArr1);
   INT64 arraySize1 = CalcArrayLength(ndim, dims);
   INT64 len = arraySize1;

   if (arraySize1 != ArrayLength(inFilter)) {
      PyErr_Format(PyExc_ValueError, "CombineFilter: Filter size not the same %lld", arraySize1);
      return NULL;
   }

   if (PyArray_TYPE(inFilter) != NPY_BOOL) {
      PyErr_Format(PyExc_ValueError, "CombineFilter: Filter is not type NPY_BOOL");
      return NULL;
   }

   // SWTICH
   COMBINE_MASK  pFunction = NULL;

   switch (numpyOutType) {
   case NPY_INT8:
      pFunction = GetCombineFunction(numpyOutType);
      break;

   case NPY_INT16:
      pFunction = GetCombineFunction(numpyOutType);
      break;

   CASE_NPY_INT32:
      pFunction = GetCombineFunction(numpyOutType);
      break;

   CASE_NPY_INT64:
      pFunction = GetCombineFunction(numpyOutType);
      break;
   }


   if (pFunction != NULL) {
      PyArrayObject* outArray = AllocateNumpyArray(ndim, dims, numpyOutType);
      CHECK_MEMORY_ERROR(outArray);

      if (outArray) {
         void* pDataOut = PyArray_BYTES(outArray);
         INT8* pFilter = (INT8*)PyArray_BYTES(inFilter);

         stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(len);

         if (pWorkItem == NULL) {

            // Threading not allowed for this work item, call it directly from main thread
            pFunction(pDataIn, pDataOut, len, pFilter);
         }
         else {
            // Each thread will call this routine with the callbackArg
            pWorkItem->DoWorkCallback = CombineThreadCallback;
            pWorkItem->WorkCallbackArg = &stCombineCallback;

            stCombineCallback.anyCombineCallback = pFunction;
            stCombineCallback.pDataOut = (char*)pDataOut;
            stCombineCallback.pDataIn = (char*)pDataIn;
            stCombineCallback.pFilter = pFilter;
            stCombineCallback.typeSizeOut = PyArray_ITEMSIZE(inArr1);

            // This will notify the worker threads of a new work item
            g_cMathWorker->WorkMain(pWorkItem, len, 0);
         }

         return (PyObject*)outArray;
      }
      PyErr_Format(PyExc_ValueError, "Combine out of memory");
      return NULL;
   }

   PyErr_Format(PyExc_ValueError, "Dont know how to combine these types %d", numpyOutType);
   return NULL;
}



//=====================================================================================
// COMBINE ACCUM 2 MASK----------------------------------------------------------------
//=====================================================================================
typedef void(*COMBINE_ACCUM2_MASK)(void* pDataIn1, void* pDataIn2, void* pDataOut, const INT64 multiplier, const INT64 maxbin, INT64 len, INT32* pCountOut, INT8* pFilter);

template<typename T, typename U, typename V>
static void CombineAccum2Mask(void* pDataIn1T, void* pDataIn2T, void* pDataOutT, const INT64 multiplier, const INT64 maxbinT, INT64 len, INT32* pCountOut, INT8* pFilter) {
   T*    pDataIn1 = (T*)pDataIn1T;
   U*    pDataIn2 = (U*)pDataIn2T;
   V*    pDataOut = (V*)pDataOutT;

   const V maxbin = (V)maxbinT;

   if (pCountOut) {
      if (pFilter) {
         for (INT64 i = 0; i < len; i++) {
            if (pFilter[i]) {
               V bin = (V)(pDataIn2[i] * multiplier + pDataIn1[i]);
               if (bin >= 0 && bin < maxbin) {
                  pCountOut[bin]++;
                  pDataOut[i] = bin;
               }
               else {
                  pCountOut[0]++;
                  pDataOut[i] = 0;
               }
            }
            else {
               pCountOut[0]++;
               pDataOut[i] = 0;
            }
         }
      }
      else {
         for (INT64 i = 0; i < len; i++) {
            V bin = (V)(pDataIn2[i] * multiplier + pDataIn1[i]);
            if (bin >= 0 && bin < maxbin) {
               pCountOut[bin]++;
               pDataOut[i] = bin;
            }
            else {
               pCountOut[0]++;
               pDataOut[i] = 0;
            }
         }

      }
   }
   else {
      // NO COUNT
      if (pFilter) {
         for (INT64 i = 0; i < len; i++) {
            if (pFilter[i]) {
               V bin = (V)(pDataIn2[i] * multiplier + pDataIn1[i]);
               if (bin >= 0 && bin < maxbin) {
                  pDataOut[i] = bin;
               }
               else {
                  pDataOut[i] = 0;
               }
            }
            else {
               pDataOut[i] = 0;
            }
         }
      }
      else {
         for (INT64 i = 0; i < len; i++) {
            V bin = (V)(pDataIn2[i] * multiplier + pDataIn1[i]);
            pDataOut[i] = bin;
            //if (bin >= 0 && bin < maxbin) {
            //   pDataOut[i] = bin;
            //}
            //else {
            //   pDataOut[i] = 0;
            //}
         }

      }

   }
}


template<typename T, typename U>
static COMBINE_ACCUM2_MASK GetCombineAccum2Function(int outputType) {

   //printf("GetCombine -- %lld %lld\n", sizeof(T), sizeof(U));

   switch (outputType) {
   case NPY_INT8:   return CombineAccum2Mask<T,U,INT8>;
   case NPY_INT16:  return CombineAccum2Mask<T,U,INT16>;
   CASE_NPY_INT32:  return CombineAccum2Mask<T,U,INT32>;
   CASE_NPY_INT64:  return CombineAccum2Mask<T,U,INT64>;
   }
   return NULL;

}


//--------------------------------------------------------------------
struct COMBINE_ACCUM2_CALLBACK {
   COMBINE_ACCUM2_MASK anyCombineCallback;
   char* pDataIn1;
   char* pDataIn2;
   char* pDataOut;
   INT8* pFilter;

   INT32* pCountOut;
   INT64 typeSizeIn1;
   INT64 typeSizeIn2;
   INT64 typeSizeOut;
   INT64 multiplier;
   INT64 maxbin;
   INT32* pCountWorkSpace;

} stCombineAccum2Callback;


//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static BOOL CombineThreadAccum2Callback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex) {
   BOOL didSomeWork = FALSE;
   COMBINE_ACCUM2_CALLBACK* Callback = (COMBINE_ACCUM2_CALLBACK*)pstWorkerItem->WorkCallbackArg;

   char* pDataIn1 = (char *)Callback->pDataIn1;
   char* pDataIn2 = (char *)Callback->pDataIn2;
   char* pDataOut = (char*)Callback->pDataOut;
   INT64 lenX;
   INT64 workBlock;


   // As long as there is work to do
   while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

      INT64 inputAdj1 = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeIn1;
      INT64 inputAdj2 = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeIn2;
      INT64 outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;
      INT64 filterAdj = pstWorkerItem->BlockSize * workBlock;

      //pFunction(pDataIn1, pDataIn2, pDataOut, inArr1Max, hashSize, arraySize1, pCountArray, pFilterIn);

      Callback->anyCombineCallback(
         pDataIn1 + inputAdj1,
         pDataIn2 + inputAdj2,
         pDataOut + outputAdj,
         Callback->multiplier,
         Callback->maxbin,
         lenX,
         // based on core, pick a counter
         Callback->pCountWorkSpace ? &Callback->pCountWorkSpace[(core+1) * Callback->maxbin] : NULL,
         Callback->pFilter ? (Callback->pFilter + filterAdj) : NULL);

      // Indicate we completed a block
      didSomeWork = TRUE;

      // tell others we completed this work block
      pstWorkerItem->CompleteWorkBlock();
      //printf("|%d %d %lld", core, (int)workBlock, lenX);
   }

   return didSomeWork;

}




//=====================================================================================
// Input: Five parameters
// Arg1: First Index array
// Arg2: Second Index array
// Arg3: Max value first index array
// Arg4: Max value second index array
// Arg5: Boolean array to merge <optional: can set to none>
//
// Returns new index array and unique count array
PyObject *
CombineAccum2Filter(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *inArr2 = NULL;
   INT64 inArr1Max = 0;
   INT64 inArr2Max = 0;
   PyObject *inFilter = NULL;

   INT64 out_dtype = 0;

   if (!PyArg_ParseTuple(
      args, "O!O!LLO:CombineAccum2Filter",
      &PyArray_Type, &inArr1,
      &PyArray_Type, &inArr2,
      &inArr1Max,
      &inArr2Max,
      &inFilter
      )) {

      return NULL;
   }

   inArr1Max++;
   inArr2Max++;

   INT64 hashSize = inArr1Max * inArr2Max ;
   //printf("Combine hashsize is %lld     %lld x %lld\n", hashSize, inArr1Max, inArr2Max);

   void* pDataIn1 = PyArray_BYTES(inArr1);
   void* pDataIn2 = PyArray_BYTES(inArr2);
   INT8* pFilterIn = NULL;
   int ndim = PyArray_NDIM(inArr1);
   npy_intp* dims = PyArray_DIMS(inArr1);
   INT64 arraySize1 = CalcArrayLength(ndim, dims);
   INT64 arraySize2 = ArrayLength(inArr2);

   if (arraySize1 != arraySize2) {
      PyErr_Format(PyExc_ValueError, "CombineAccum2Filter: array sizes not the same %lld", arraySize1);
      return NULL;
   }

   if (PyArray_Check(inFilter)) {
      if (arraySize1 != ArrayLength((PyArrayObject*)inFilter)) {
         PyErr_Format(PyExc_ValueError, "CombineAccum2Filter: Filter size not the same %lld", arraySize1);
         return NULL;
      }
      if (PyArray_TYPE((PyArrayObject*)inFilter) != NPY_BOOL) {
         PyErr_Format(PyExc_ValueError, "CombineAccum2Filter: Filter is not type NPY_BOOL");
         return NULL;
      }
      pFilterIn = (INT8*)PyArray_BYTES((PyArrayObject*)inFilter);
   }


   if (hashSize < 0 || hashSize > 2000000000) {
      PyErr_Format(PyExc_ValueError, "CombineAccum2Filter: Index sizes are either 0, negative, or produce more than 2 billion results %lld", hashSize);
      return NULL;
   }

   INT32 numpyOutType = NPY_INT64;
   INT64 typeSizeOut = 8;

   if (hashSize < 2000000000) {
      numpyOutType = NPY_INT32;
      typeSizeOut = 4;
   }
   if (hashSize < 32000) {
      numpyOutType = NPY_INT16;
      typeSizeOut = 2;
   }
   if (hashSize < 120) {
      numpyOutType = NPY_INT8;
      typeSizeOut = 1;
   }

   // SWTICH
   COMBINE_ACCUM2_MASK  pFunction = NULL;

   INT32 type2 = PyArray_TYPE(inArr2);

   switch (PyArray_TYPE(inArr1)) {
   case NPY_INT8:
      switch (type2) {
      case NPY_INT8:   pFunction = GetCombineAccum2Function<INT8, INT8>(numpyOutType);  break;
      case NPY_INT16:  pFunction = GetCombineAccum2Function<INT8, INT16>(numpyOutType);  break;
      CASE_NPY_INT32:  pFunction = GetCombineAccum2Function<INT8, INT32>(numpyOutType);  break;
      CASE_NPY_INT64:  pFunction = GetCombineAccum2Function<INT8, INT64>(numpyOutType);  break;
      }
      break;

   case NPY_INT16:
      switch (type2) {
      case NPY_INT8:   pFunction = GetCombineAccum2Function<INT16, INT8>(numpyOutType);  break;
      case NPY_INT16:  pFunction = GetCombineAccum2Function<INT16, INT16>(numpyOutType);  break;
      CASE_NPY_INT32:  pFunction = GetCombineAccum2Function<INT16, INT32>(numpyOutType);  break;
      CASE_NPY_INT64:  pFunction = GetCombineAccum2Function<INT16, INT64>(numpyOutType);  break;
      }
      break;

   CASE_NPY_INT32:
      switch (type2) {
      case NPY_INT8:   pFunction = GetCombineAccum2Function<INT32, INT8>(numpyOutType);  break;
      case NPY_INT16:  pFunction = GetCombineAccum2Function<INT32, INT16>(numpyOutType);  break;
      CASE_NPY_INT32:  pFunction = GetCombineAccum2Function<INT32, INT32>(numpyOutType);  break;
      CASE_NPY_INT64:  pFunction = GetCombineAccum2Function<INT32, INT64>(numpyOutType);  break;
      }
      break;

   CASE_NPY_INT64:
      switch (type2) {
      case NPY_INT8:   pFunction = GetCombineAccum2Function<INT64, INT8>(numpyOutType);  break;
      case NPY_INT16:  pFunction = GetCombineAccum2Function<INT64, INT16>(numpyOutType);  break;
      CASE_NPY_INT32:  pFunction = GetCombineAccum2Function<INT64, INT32>(numpyOutType);  break;
      CASE_NPY_INT64:  pFunction = GetCombineAccum2Function<INT64, INT64>(numpyOutType);  break;
      }
      break;
   }

   BOOL bWantCount = FALSE;

   if (pFunction != NULL) {
      PyArrayObject* outArray = AllocateNumpyArray(ndim, dims, numpyOutType, 0, PyArray_IS_F_CONTIGUOUS(inArr1));
      CHECK_MEMORY_ERROR(outArray);

      if (outArray) {
         void* pDataOut = PyArray_BYTES(outArray);

         // 32 bit count limitation here
         PyArrayObject* countArray = NULL;
         INT32* pCountArray = NULL;

         if (bWantCount) {
            countArray = AllocateNumpyArray(1, (npy_intp*)&hashSize, NPY_INT32);
            CHECK_MEMORY_ERROR(countArray);
            if (countArray) {
               pCountArray = (INT32*)PyArray_BYTES(countArray);
               memset(pCountArray, 0, hashSize * sizeof(INT32));
            }
         }

         stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(arraySize1);

         if (pWorkItem == NULL) {

            // Threading not allowed for this work item, call it directly from main thread
            pFunction(pDataIn1, pDataIn2, pDataOut, inArr1Max, hashSize, arraySize1, pCountArray, pFilterIn);
         }
         else {

            // TODO: steal from hash
            INT numCores = g_cMathWorker->WorkerThreadCount + 1;
            INT64 sizeToAlloc = numCores * hashSize * sizeof(INT32);
            PVOID pWorkSpace = 0;

            if (bWantCount) {
               pWorkSpace = WORKSPACE_ALLOC(sizeToAlloc);
               memset(pWorkSpace, 0, sizeToAlloc);
            }

            // Each thread will call this routine with the callbackArg
            pWorkItem->DoWorkCallback = CombineThreadAccum2Callback;
            pWorkItem->WorkCallbackArg = &stCombineAccum2Callback;

            stCombineAccum2Callback.anyCombineCallback = pFunction;
            stCombineAccum2Callback.pDataOut = (char*)pDataOut;
            stCombineAccum2Callback.pDataIn1 = (char*)pDataIn1;
            stCombineAccum2Callback.pDataIn2 = (char*)pDataIn2;
            stCombineAccum2Callback.pFilter = pFilterIn;
            stCombineAccum2Callback.pCountOut = pCountArray;

            stCombineAccum2Callback.typeSizeIn1 = PyArray_ITEMSIZE(inArr1);
            stCombineAccum2Callback.typeSizeIn2 = PyArray_ITEMSIZE(inArr2);
            stCombineAccum2Callback.typeSizeOut = typeSizeOut;
            stCombineAccum2Callback.multiplier = inArr1Max;
            stCombineAccum2Callback.maxbin = hashSize;
            stCombineAccum2Callback.pCountWorkSpace = (INT32*)pWorkSpace;

            LOGGING("**array: %lld      out sizes: %lld %lld %lld\n", arraySize1, stCombineAccum2Callback.typeSizeIn1, stCombineAccum2Callback.typeSizeIn2, stCombineAccum2Callback.typeSizeOut);

            // This will notify the worker threads of a new work item
            g_cMathWorker->WorkMain(pWorkItem, arraySize1, 0);

            if (bWantCount && pCountArray) {
               // Collect the results
               INT32* pCoreCountArray = (INT32*)pWorkSpace;

               for (int j = 0; j < numCores; j++) {

                  for (int i = 0; i < hashSize; i++) {
                     pCountArray[i] += pCoreCountArray[i];
                  }

                  // go to next core
                  pCoreCountArray += hashSize;
               }
            }

            if (bWantCount) {
               WORKSPACE_FREE(pWorkSpace);
            }
         }

         if (bWantCount) {
            PyObject* retObject = Py_BuildValue("(OO)", outArray, countArray);
            Py_DecRef((PyObject*)outArray);
            Py_DecRef((PyObject*)countArray);
            return (PyObject*)retObject;
         }
         else {
            Py_INCREF(Py_None);
            PyObject* retObject = Py_BuildValue("(OO)", outArray, Py_None);
            Py_DecRef((PyObject*)outArray);
            return (PyObject*)retObject;
         }
      }
      PyErr_Format(PyExc_ValueError, "CombineFilter out of memory");
      return NULL;
   }

   PyErr_Format(PyExc_ValueError, "Dont know how to combine filter these types %d.  Please make sure all bins are INT8, INT16, INT32, or INT64.", numpyOutType);
   return NULL;
}

//==========================================
// Old  Filter First  ==> NewIndex NewFirst
// 1      T     0           1        0
// 1      F     2           0        4
// 2      F     3           0
// 3      F                 0
// 3      T                 2
// 3      T                 2
//
// Input:  InputIndex
//         Filter
// Output: OutputIndex (the new iKey)
//         NewFirst    (the new iFirstKey)
//
//
typedef INT64(*COMBINE_1_FILTER)(
   void*    pInputIndex,
   void*    pOutputIndex,  // newly allocated
   INT32*   pNewFirst,     // newly allocated
   INT8*    pFilter,       // may be null
   INT64    arrayLength,   // index array size
   INT64    hashLength);   // max uniques + 1 (for 0 bin)

template<typename INDEX>
INT64 Combine1Filter(
   void*    pInputIndex,
   void*    pOutputIndex,  // newly allocated
   INT32*   pNewFirst,     // newly allocated NOTE: for > 2e9 should be INT64
   INT8*    pFilter,       // may be null
   INT64    arrayLength,
   INT64    hashLength) {

   INDEX*      pInput = (INDEX*)pInputIndex;
   INDEX*      pOutput = (INDEX*)pOutputIndex;

   //WORKSPACE_ALLOC
   INT64 allocSize = hashLength * sizeof(INT32);

   INT32* pHash = (INT32*)WorkSpaceAllocLarge(allocSize);
   memset(pHash, 0, allocSize);

   INT32       uniquecount = 0;
   if (pFilter) {
      for (INT64 i = 0; i < arrayLength; i++) {
         if (pFilter[i]) {
            INDEX index = pInput[i];
            //printf("[%lld] got index for %lld\n", (INT64)index, i);

            if (index != 0) {
               // Check hash
               if (pHash[index] == 0) {
                  // First time, assign FirstKey
                  pNewFirst[uniquecount] = (INT32)i;
                  uniquecount++;

                  //printf("reassign index:%lld to bin:%d\n", (INT64)index, uniquecount);

                  // ReassignKey
                  pHash[index] = uniquecount;
                  pOutput[i] = (INDEX)uniquecount;
               }
               else {
                  // Get reassigned key
                  //printf("exiting  index:%lld to bin:%d\n", (INT64)index, (INT32)pHash[index]);
                  pOutput[i] = (INDEX)pHash[index];
               }
            }
            else {
               // was already 0 bin
               pOutput[i] = 0;
            }
         }
         else {
            // filtered out
            pOutput[i] = 0;
         }
      }
   }
   else {
      // When no filter provided
      for (INT64 i = 0; i < arrayLength; i++) {
         INDEX index = pInput[i];
         //printf("[%lld] got index\n", (INT64)index);

         if (index != 0) {
            // Check hash
            if (pHash[index] == 0) {
               // First time, assign FirstKey
               pNewFirst[uniquecount] = (INT32)i;
               uniquecount++;

               // ReassignKey
               pHash[index] = uniquecount;
               pOutput[i] = (INDEX)uniquecount;
            }
            else {
               // Get reassigned key
               pOutput[i] = (INDEX)pHash[index];
            }
         }
         else {
            // was already 0 bin
            pOutput[i] = 0;
         }
      }
   }

   void* pHashVoid = pHash;
   WorkSpaceFreeAllocLarge(pHashVoid, allocSize);
   return uniquecount;
}

//=====================================================================================
// Input:
// Arg1: Index array
// Arg2: Max uniques
// Arg3: Boolean array to filter on (or None)
//
// Output:
// New Index Array
// New First Array (can use to pull in key names)
// UniqueCount (should be size of FirstArray)... possibly 0 if everything removed
// Returns new index array and unique count array
PyObject *
CombineAccum1Filter(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;
   INT64 inArr1Max = 0;
   PyObject *inFilter = NULL;

   INT64 out_dtype = 0;

   if (!PyArg_ParseTuple(
      args, "O!LO:CombineAccum1Filter",
      &PyArray_Type, &inArr1,
      &inArr1Max,
      &inFilter
   )) {

      return NULL;
   }

   void* pDataIn1 = PyArray_BYTES(inArr1);
   INT8* pFilterIn = NULL;

   int ndim = PyArray_NDIM(inArr1);
   npy_intp* dims = PyArray_DIMS(inArr1);
   INT64 arraySize1 = CalcArrayLength(ndim, dims);

   if (PyArray_Check(inFilter)) {
      if (arraySize1 != ArrayLength((PyArrayObject*)inFilter)) {
         PyErr_Format(PyExc_ValueError, "CombineAccum1Filter: Filter size not the same %lld", arraySize1);
         return NULL;
      }
      if (PyArray_TYPE((PyArrayObject*)inFilter) != NPY_BOOL) {
         PyErr_Format(PyExc_ValueError, "CombineAccum1Filter: Filter is not type NPY_BOOL");
         return NULL;
      }
      pFilterIn = (INT8*)PyArray_BYTES((PyArrayObject*)inFilter);
   }

   inArr1Max++;
   INT64 hashSize = inArr1Max;
   //printf("Combine hashsize is %lld     %lld\n", hashSize, inArr1Max);
   if (hashSize < 0 || hashSize > 2000000000) {
      PyErr_Format(PyExc_ValueError, "CombineAccum1Filter: Index sizes are either 0, negative, or produce more than 2 billion results %lld", hashSize);
      return NULL;
   }

   int dtype= PyArray_TYPE(inArr1);

   COMBINE_1_FILTER pFunction = NULL;
   switch (dtype) {
   case NPY_INT8:
      pFunction = Combine1Filter<INT8>;
      break;
   case NPY_INT16:
      pFunction = Combine1Filter<INT16>;
      break;
   CASE_NPY_INT32:
      pFunction = Combine1Filter<INT32>;
      break;
   CASE_NPY_INT64:
      pFunction = Combine1Filter<INT64>;
      break;
   }

   if (pFunction != NULL) {
      PyArrayObject* outArray = AllocateNumpyArray(ndim, dims, dtype, 0, PyArray_IS_F_CONTIGUOUS(inArr1));
      CHECK_MEMORY_ERROR(outArray);

      PyArrayObject* firstArray = AllocateNumpyArray(1, (npy_intp*)&arraySize1, NPY_INT32);  // TODO: bump up to INT64 for large arrays
      CHECK_MEMORY_ERROR(firstArray);

      if (outArray && firstArray) {
         INT32* pFirst = (INT32*)PyArray_BYTES(firstArray);

         INT64 uniqueCount =
            pFunction(
               pDataIn1,
               PyArray_BYTES(outArray),
               pFirst,
               pFilterIn,
               arraySize1,
               hashSize);

         if (uniqueCount < arraySize1) {
            // fixup first to hold only the uniques
            PyArrayObject* firstArrayReduced = AllocateNumpyArray(1, (npy_intp*)&uniqueCount, NPY_INT32);  // TODO: bump up to INT64 for large arrays
            CHECK_MEMORY_ERROR(firstArrayReduced);

            if (firstArrayReduced) {
               INT32* pFirstReduced = (INT32*)PyArray_BYTES(firstArrayReduced);

               memcpy(pFirstReduced, pFirst, uniqueCount * sizeof(INT32));
            }
            Py_DecRef((PyObject*)firstArray);
            firstArray = firstArrayReduced;
         }

         PyObject* returnObject = PyList_New(3);
         PyList_SET_ITEM(returnObject, 0, (PyObject*)outArray);
         PyList_SET_ITEM(returnObject, 1, (PyObject*)firstArray);
         PyList_SET_ITEM(returnObject, 2, (PyObject*)PyLong_FromLongLong(uniqueCount));
         return returnObject;
      }
   }


   return NULL;

}







typedef INT64(*IFIRST_FILTER)(
   void*    pInputIndex,
   void*    pNewFirstIndex,     // newly allocated
   INT8*    pFilter,       // may be null
   INT64    arrayLength,   // index array size
   INT64    hashLength);   // max uniques + 1 (for 0 bin)

template<typename INDEX>
INT64 iFirstFilter(
   void*    pInputIndex,
   void*    pNewFirstIndex,     // newly allocated NOTE: for > 2e9 should be INT64
   INT8*    pFilter,       // may be null
   INT64    arrayLength,
   INT64    hashLength) {

   INDEX*      pInput = (INDEX*)pInputIndex;
   INT64*      pNewFirst = (INT64*)pNewFirstIndex;
   INT64       invalid = (INT64)(1LL <<  (sizeof(INT64) * 8 - 1));

   // Fill with invalid
   for (INT64 i = 0; i < hashLength; i++) {
      pNewFirst[i] = invalid;
   }

   // NOTE: the uniquecount is currently not used
   INT32       uniquecount = 0;
   if (pFilter) {
      for (INT64 i = 0; i < arrayLength; i++) {
         if (pFilter[i]) {
            INDEX index = pInput[i];
            //printf("[%lld] got index for %lld\n", (INT64)index, i);

            if (index > 0 && index < hashLength) {
               // Check hash
               if (pNewFirst[index] == invalid) {
                  // First time, assign FirstKey
                  pNewFirst[index] = i;
                  uniquecount++;

               }
            }
         }
      }
   }
   else {
      // When no filter provided
      for (INT64 i = 0; i < arrayLength; i++) {
         INDEX index = pInput[i];

         if (index > 0 && index < hashLength) {
            // Check hash
            if (pNewFirst[index] == invalid) {
               // First time, assign FirstKey
               pNewFirst[index] = i;
               uniquecount++;

            }
         }
      }
   }
   return uniquecount;
}

template<typename INDEX>
INT64 iLastFilter(
   void*    pInputIndex,
   void*    pNewLastIndex,     // newly allocated NOTE: for > 2e9 should be INT64
   INT8*    pFilter,           // may be null
   INT64    arrayLength,
   INT64    hashLength) {

   INDEX*      pInput = (INDEX*)pInputIndex;
   INT64*      pNewLast = (INT64*)pNewLastIndex;
   INT64       invalid = (INT64)(1LL << (sizeof(INT64) * 8 - 1));

   // Fill with invalid
   for (INT64 i = 0; i < hashLength; i++) {
      pNewLast[i] = invalid;
   }

   if (pFilter) {
      for (INT64 i = 0; i < arrayLength; i++) {
         if (pFilter[i]) {
            INDEX index = pInput[i];
            if (index > 0 && index < hashLength) {
               // assign current LastKey
               pNewLast[index] = i;
            }
         }
      }
   }
   else {
      // When no filter provided
      for (INT64 i = 0; i < arrayLength; i++) {
         INDEX index = pInput[i];

         if (index > 0 && index < hashLength) {
            // assign current LastKey
            pNewLast[index] = i;
         }
      }
   }
   // last does not keep track of uniquecount
   return 0;
}


//=====================================================================================
// Input:
// Arg1: Index array
// Arg2: Max uniques
// Arg3: Boolean array to filter on (or None)
// Arg4: integer set to 0 for first, 1 for last 
//
// Output:
// New First Array (can use to pull in key names)
// UniqueCount (should be size of FirstArray)... possibly 0 if everything removed
PyObject *
MakeiFirst(PyObject *self, PyObject *args) {
   PyArrayObject *inArr1 = NULL;
   INT64 inArr1Max = 0;
   PyObject *inFilter = NULL;
   INT64 isLast = 0;

   INT64 out_dtype = 0;

   if (!PyArg_ParseTuple(
      args, "O!LOL:MakeiFirst",
      &PyArray_Type, &inArr1,
      &inArr1Max,
      &inFilter,
      &isLast
   )) {

      return NULL;
   }

   void* pDataIn1 = PyArray_BYTES(inArr1);
   INT8* pFilterIn = NULL;

   int ndim = PyArray_NDIM(inArr1);
   npy_intp* dims = PyArray_DIMS(inArr1);
   INT64 arraySize1 = CalcArrayLength(ndim, dims);

   if (PyArray_Check(inFilter)) {
      if (arraySize1 != ArrayLength((PyArrayObject*)inFilter)) {
         PyErr_Format(PyExc_ValueError, "MakeiFirst: Filter size not the same %lld", arraySize1);
         return NULL;
      }
      if (PyArray_TYPE((PyArrayObject*)inFilter) != NPY_BOOL) {
         PyErr_Format(PyExc_ValueError, "MakeiFirst: Filter is not type NPY_BOOL");
         return NULL;
      }
      pFilterIn = (INT8*)PyArray_BYTES((PyArrayObject*)inFilter);
   }

   inArr1Max++;
   INT64 hashSize = inArr1Max;
   //printf("Combine hashsize is %lld     %lld\n", hashSize, inArr1Max);
   if (hashSize < 0 || hashSize > 20000000000LL) {
      PyErr_Format(PyExc_ValueError, "MakeiFirst: Index sizes are either 0, negative, or produce more than 20 billion results %lld", hashSize);
      return NULL;
   }

   int dtype = PyArray_TYPE(inArr1);

   IFIRST_FILTER pFunction = NULL;

   if (isLast) {
      switch (dtype) {
      case NPY_INT8:
         pFunction = iLastFilter<INT8>;
         break;
      case NPY_INT16:
         pFunction = iLastFilter<INT16>;
         break;
      case NPY_INT32:
         pFunction = iLastFilter<INT32>;
         break;
      case NPY_INT64:
         pFunction = iLastFilter<INT64>;
         break;
      }

   }
   else {
      switch (dtype) {
      case NPY_INT8:
         pFunction = iFirstFilter<INT8>;
         break;
      case NPY_INT16:
         pFunction = iFirstFilter<INT16>;
         break;
      case NPY_INT32:
         pFunction = iFirstFilter<INT32>;
         break;
      case NPY_INT64:
         pFunction = iFirstFilter<INT64>;
         break;
      }
   }

   if (pFunction != NULL) {
      PyArrayObject* firstArray = AllocateNumpyArray(1, (npy_intp*)&hashSize, NPY_INT64); 
      CHECK_MEMORY_ERROR(firstArray);

      if (firstArray) {
         void* pFirst = PyArray_BYTES(firstArray);

         INT64 uniqueCount =
            pFunction(
               pDataIn1,
               pFirst,
               pFilterIn,
               arraySize1,
               hashSize);

         return (PyObject*)firstArray;
      }
   }
   return NULL;

}







//=====================================================================================
//
void TrailingSpaces(char* pStringArray, INT64 length, INT64 itemSize) {
   for (INT64 i = 0; i < length; i++) {
      char* pStart = pStringArray + (i * itemSize);
      char* pEnd = pStart + itemSize - 1;
      while (pEnd >= pStart && (*pEnd == ' ' || *pEnd == 0)) {
         *pEnd-- = 0;
      }
   }
}

//=====================================================================================
//
void TrailingSpacesUnicode(UINT32* pUnicodeArray, INT64 length, INT64 itemSize) {
   itemSize = itemSize / 4;

   for (INT64 i = 0; i < length; i++) {
      UINT32* pStart = pUnicodeArray + (i * itemSize);
      UINT32* pEnd = pStart + itemSize - 1;
      while (pEnd >= pStart && (*pEnd == 32 || *pEnd == 0)) {
         *pEnd-- = 0;
      }
   }
}

//=====================================================================================
// Arg1: array to strip trailing spaces
//
// Returns converted array or NULL
PyObject *
RemoveTrailingSpaces(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;

   if (!PyArg_ParseTuple(
      args, "O!",
      &PyArray_Type, &inArr1)) {

      return NULL;
   }

   int dtype = PyArray_TYPE(inArr1);

   if (dtype == NPY_STRING || dtype == NPY_UNICODE) {
      void* pDataIn1 = PyArray_BYTES(inArr1);
      INT64 arraySize1 = ArrayLength(inArr1);
      INT64 itemSize = PyArray_ITEMSIZE(inArr1);

      if (dtype == NPY_STRING) {
         TrailingSpaces((char *)pDataIn1, arraySize1, itemSize);
      }
      else {
         TrailingSpacesUnicode((UINT32 *)pDataIn1, arraySize1, itemSize);
      }

      Py_IncRef((PyObject*)inArr1);
      return (PyObject*)inArr1;
   }

   PyErr_Format(PyExc_ValueError, "Dont know how to convert these types %d.  Please make sure to pass a string.", dtype);
   return NULL;

}


int GetUpcastDtype(ArrayInfo* aInfo, INT64 tupleSize) {
   int maxfloat = -1;
   int maxint = -1;
   int maxuint = -1;
   int maxstring = -1;
   int maxobject = -1;
   int abort = 0;

   for (int t = 0; t < tupleSize; t++) {
      int tempdtype = aInfo[t].NumpyDType;
      if (tempdtype <= NPY_LONGDOUBLE) {
         if (tempdtype >= NPY_FLOAT) {
            if (tempdtype > maxfloat) {
               maxfloat = tempdtype;
            }
         }
         else {
            if (tempdtype & 1 || tempdtype ==0) {
               if (tempdtype > maxint) {
                  maxint = tempdtype;
               }
            }
            else {
               if (tempdtype > maxuint) {
                  maxuint = tempdtype;
               }
            }
         }
      }
      else {
         if (tempdtype == NPY_OBJECT) {
            maxobject = NPY_OBJECT;
         } else
         if (tempdtype == NPY_UNICODE) {
            maxstring = NPY_UNICODE;
         } else
         if (tempdtype == NPY_STRING) {
            if (maxstring < NPY_STRING) {
               maxstring = NPY_STRING;
            }
         }
         else {
            abort = tempdtype;
         }
      }
   }

   if (abort > 0) {
      return -1;
   }

   // Return in this order:
   // OBJECT
   // UNICODE
   // STRING
   if (maxobject == NPY_OBJECT) {
      return NPY_OBJECT;
   }

   if (maxstring > 0) {
      // return either NPY_UNICODE or NPY_STRING
      return maxstring;
   }

   if (maxfloat > 0) {
      // do we have a float?
      if (maxfloat > NPY_FLOAT) {
         return maxfloat;
      }

      // we have a float... see if we have integers that force a double
      if (maxint > NPY_INT16 || maxuint > NPY_UINT16) {
         return NPY_DOUBLE;
      }

      return maxfloat;
   }
   else {
      if (maxuint > 0) {
         // Do we have a uint and no floats?
         if (maxint > maxuint) {
            // we can safely upcast the uint to maxint
            return maxint;
         }

         // check if any ints
         if (maxint == -1) {
            // no integers and no floats
            return maxuint;
         }

         if (sizeof(long) == 8) {
            // gcc/linux path
            // if maxuint is hit and we have integers, force to go to double
            if (maxuint == NPY_ULONGLONG || maxuint == NPY_ULONG) {
               return NPY_DOUBLE;
            }
            if (maxint == NPY_LONG || maxint == NPY_LONGLONG) {
               return NPY_DOUBLE;
            }

            // we have ints, go to next higher int
            return (maxuint + 1);
         }
         else {
            if (maxuint == NPY_ULONGLONG) {
               return NPY_DOUBLE;
            }
            if (maxint == NPY_LONG) {
               return NPY_DOUBLE;
            }

            // we have ints, go to next higher int
            return (maxuint + 1);
         }

         // should not get here
         return maxuint;
      }
      else {
         // we have just ints or bools
         return maxint;
      }
   }
}



//----------------------------------------------------
// Arg1: Pass in list of arrays
//
// Returns: dtype num to upcast to
//          may return -1 on impossible
PyObject *GetUpcastNum(PyObject* self, PyObject *args)
{
   PyObject *inList1 = NULL;

   if (!PyArg_ParseTuple(
      args, "O",
      &inList1)) {

      return NULL;
   }

   INT64 totalItemSize = 0;
   INT64 tupleSize = 0;

   // Allow jagged rows
   // Do not copy
   ArrayInfo* aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, FALSE, FALSE);

   if (aInfo) {
      int dtype = GetUpcastDtype(aInfo, tupleSize);
      FreeArrayInfo(aInfo);
      return PyLong_FromLong(dtype);
   }
   return NULL;
}

//----------------------------------------------------
// Arg1: Pass in list of arrays
// Arg2: (optional) default dtype num (for possible upcast)
// Returns single array concatenatedessed
//
PyObject *HStack(PyObject* self, PyObject *args)
{
   PyObject *inList1 = NULL;
   INT32 dtype = -1;

   if (!PyArg_ParseTuple(
      args, "O|i",
      &inList1,
      &dtype)) {

      return NULL;
   }

   if (dtype != -1) {
      if (dtype < 0 || dtype > NPY_LONGDOUBLE) {
         PyErr_Format(PyExc_ValueError, "Dont know how to convert dtype num %d.  Please make sure all arrays are ints or floats.", dtype);
         return NULL;
      }
   }

   INT64 totalItemSize = 0;
   INT64 tupleSize = 0;

   // Allow jagged rows
   ArrayInfo* aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, FALSE);

   if (aInfo) {

      INT64 totalLength = 0;
      INT64 maxItemSize = 0;
      PyArrayObject* outputArray = NULL;

      if (dtype == -1) {
         dtype = GetUpcastDtype(aInfo, tupleSize);

         if (dtype <  0 || dtype > NPY_LONGDOUBLE) {

            BOOL isSameDtype = TRUE;

            // Check for all strings or all unicode which we know how to stack
            for (int t = 0; t < tupleSize; t++) {
               if (dtype != aInfo[t].NumpyDType) {
                  isSameDtype = FALSE;
                  break;
               }
               // track max itemsize since for a string we must match it
               if (aInfo[t].ItemSize > maxItemSize) {
                  maxItemSize = aInfo[t].ItemSize;
               }
            }

            // Check for strings
            if ((dtype == NPY_STRING || dtype == NPY_UNICODE) && isSameDtype) {
               // they are all strings/unicode

            }
            else {

               PyErr_Format(PyExc_ValueError, "Dont know how to convert dtype num %d.  Please make sure all arrays are ints or floats.", dtype);
               return NULL;
            }
         }
      }

      if (dtype == NPY_STRING || dtype == NPY_UNICODE) {

         //
         // Path for strings
         //
         struct stHSTACK_STRING {
            INT64                   Offset;
            INT64                   ItemSize;
            CONVERT_SAFE_STRING     ConvertSafeString;
         };

         stHSTACK_STRING*  pHStack = (stHSTACK_STRING*)WORKSPACE_ALLOC(sizeof(stHSTACK_STRING) * tupleSize);
         // calculate total size and get conversion function for each row
         for (int t = 0; t < tupleSize; t++) {
            int a_dtype = aInfo[t].NumpyDType;

            pHStack[t].ConvertSafeString = ConvertSafeStringCopy;
            pHStack[t].Offset = totalLength;
            pHStack[t].ItemSize = aInfo[t].ItemSize;
            totalLength += aInfo[t].ArrayLength;
         }

         if (dtype == NPY_STRING || dtype == NPY_UNICODE) {
            // string allocation
            outputArray = AllocateNumpyArray(1, (npy_intp*)&totalLength, dtype, maxItemSize);
         }
         else {
            outputArray = AllocateNumpyArray(1, (npy_intp*)&totalLength, dtype);
         }

         CHECK_MEMORY_ERROR(outputArray);
         if (outputArray) {
            INT64 itemSize = PyArray_ITEMSIZE(outputArray);
            char* pOutput = (char*)PyArray_BYTES(outputArray);

            if (tupleSize < 2 || totalLength <= g_cMathWorker->WORK_ITEM_CHUNK) {

               for (int t = 0; t < tupleSize; t++) {
                  pHStack[t].ConvertSafeString(aInfo[t].pData, pOutput + (pHStack[t].Offset * itemSize), aInfo[t].ArrayLength, pHStack[t].ItemSize, itemSize);
               }
            }
            else {

               // Callback routine from multithreaded worker thread (items just count up from 0,1,2,...)
               //typedef BOOL(*MTWORK_CALLBACK)(void* callbackArg, int core, INT64 workIndex);
               struct stSHSTACK {
                  stHSTACK_STRING*  pHStack;
                  ArrayInfo*        aInfo;
                  char*             pOutput;
                  INT64             ItemSizeOutput;
               } myhstack;

               myhstack.pHStack = pHStack;
               myhstack.aInfo = aInfo;
               myhstack.pOutput = pOutput;
               myhstack.ItemSizeOutput = itemSize;

               LOGGING("MT string hstack work on %lld\n", tupleSize);

               auto lambdaHSCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
                  stSHSTACK* callbackArg = (stSHSTACK*)callbackArgT;
                  INT64 t = workIndex;
                  callbackArg->pHStack[t].ConvertSafeString(
                     callbackArg->aInfo[t].pData,
                     callbackArg->pOutput + (callbackArg->pHStack[t].Offset * callbackArg->ItemSizeOutput),
                     callbackArg->aInfo[t].ArrayLength,
                     callbackArg->aInfo[t].ItemSize,
                     callbackArg->ItemSizeOutput);

                  return TRUE;
               };

               g_cMathWorker->DoMultiThreadedWork((int)tupleSize, lambdaHSCallback, &myhstack);

            }
         }

         WORKSPACE_FREE(pHStack);
      }
      else {

         // Path for non-strings
         struct stHSTACK {
            INT64          Offset;
            void*          pBadInput1;
            CONVERT_SAFE   ConvertSafe;
         };

         stHSTACK*  pHStack = (stHSTACK*)WORKSPACE_ALLOC(sizeof(stHSTACK) * tupleSize);

         // calculate total size and get conversion function for each row
         for (int t = 0; t < tupleSize; t++) {
            int a_dtype = aInfo[t].NumpyDType;

            if (a_dtype > NPY_LONGDOUBLE || aInfo[t].NDim != 1) {
               FreeArrayInfo(aInfo);
               PyErr_Format(PyExc_ValueError, "Dont know how to convert dtype num %d or more than 1 dimension.  Please make sure all arrays are ints or floats.", a_dtype);
               return NULL;
            }

            pHStack[t].ConvertSafe = GetConversionFunctionSafe(aInfo[t].NumpyDType, dtype);
            pHStack[t].Offset = totalLength;
            pHStack[t].pBadInput1 = GetInvalid(aInfo[t].NumpyDType);
            totalLength += aInfo[t].ArrayLength;
         }

         outputArray = AllocateNumpyArray(1, (npy_intp*)&totalLength, dtype);
         CHECK_MEMORY_ERROR(outputArray);

         if (outputArray) {

            // if output is boolean, bad means FALSE
            void* pBadOutput1 = GetDefaultForType(dtype);

            INT64 strideOut = PyArray_STRIDE(outputArray, 0);
            char* pOutput = (char*)PyArray_BYTES(outputArray);

            if (tupleSize < 2 || totalLength <= g_cMathWorker->WORK_ITEM_CHUNK) {

               for (int t = 0; t < tupleSize; t++) {
                  pHStack[t].ConvertSafe(
                     aInfo[t].pData, pOutput + (pHStack[t].Offset * strideOut),
                     aInfo[t].ArrayLength, 
                     pHStack[t].pBadInput1, 
                     pBadOutput1,
                     PyArray_STRIDE(aInfo[t].pObject,0),
                     strideOut);
               }
            }
            else {

               // Callback routine from multithreaded worker thread (items just count up from 0,1,2,...)
               //typedef BOOL(*MTWORK_CALLBACK)(void* callbackArg, int core, INT64 workIndex);
               struct stSHSTACK {
                  stHSTACK*      pHStack;
                  ArrayInfo*     aInfo;
                  char*          pOutput;
                  INT64          StrideOut;
                  void*          pBadOutput1;
               } myhstack;

               myhstack.pHStack = pHStack;
               myhstack.aInfo = aInfo;
               myhstack.pOutput = pOutput;
               myhstack.StrideOut = strideOut;
               myhstack.pBadOutput1 = pBadOutput1;

               LOGGING("MT hstack work on %lld\n", tupleSize);

               auto lambdaHSCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
                  stSHSTACK* callbackArg = (stSHSTACK*)callbackArgT;
                  INT64 t = workIndex;
                  callbackArg->pHStack[t].ConvertSafe(
                     callbackArg->aInfo[t].pData,
                     callbackArg->pOutput + (callbackArg->pHStack[t].Offset * callbackArg->StrideOut),
                     callbackArg->aInfo[t].ArrayLength,
                     callbackArg->pHStack[t].pBadInput1,
                     callbackArg->pBadOutput1,
                     PyArray_STRIDE(callbackArg->aInfo[t].pObject, 0),
                     callbackArg->StrideOut);

                  return TRUE;
               };

               g_cMathWorker->DoMultiThreadedWork((int)tupleSize, lambdaHSCallback, &myhstack);

            }
         }
         WORKSPACE_FREE(pHStack);
      }

      FreeArrayInfo(aInfo);

      if (!outputArray) {
         PyErr_Format(PyExc_ValueError, "hstack out of memory");
         return NULL;
      }

      return (PyObject*)outputArray;
   }

   return NULL;
}



//----------------------------------------------------
// Arg1: Pass in list of arrays
// Arg2: +/- amount to shift
// Returns each array shifted
//
PyObject *ShiftArrays(PyObject* self, PyObject *args)
{
   PyObject *inList1 = NULL;
   INT64 shiftAmount = 0;

   if (!PyArg_ParseTuple(
      args, "OL",
      &inList1,
      &shiftAmount)) {

      return NULL;
   }

   INT64 totalItemSize = 0;
   INT64 tupleSize = 0;

   // Allow jagged rows
   ArrayInfo* aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, FALSE);

   if (aInfo) {

      INT64 totalLength = 0;

      // Callback routine from multithreaded worker thread (items just count up from 0,1,2,...)
      //typedef BOOL(*MTWORK_CALLBACK)(void* callbackArg, int core, INT64 workIndex);
      struct stSHIFT {
         ArrayInfo*     aInfo;
         INT64          shiftAmount;
      } myshift;

      myshift.aInfo = aInfo;
      myshift.shiftAmount = shiftAmount;

      auto lambdaShiftCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
         stSHIFT* pShift = (stSHIFT*)callbackArgT;
         INT64 t = workIndex;

         ArrayInfo* pArrayInfo = &pShift->aInfo[t];

         if (pArrayInfo->pData) {
            npy_intp* const pStrides= PyArray_STRIDES(pArrayInfo->pObject);

            // Check for fortran style
            if (pArrayInfo->NDim >= 2 && pStrides[0] < pStrides[1]) {
               npy_intp* const pDims = PyArray_DIMS(pArrayInfo->pObject);


               //-- BEFORE SHIFT rows:5  cols: 3
               // a d g j m   0  1  2  3  4
               // b e h k n   5  6  7  8  9
               // c f i l o  10 11 12 13 14
               //
               //-- AFTER SHIFT of 2
               // g j m x x   0  1  2  3  4
               // h k n x x   5  6  7  8  9
               // i l o x x  10 11 12 13 14
               //
               INT64 rows = pDims[0];
               INT64 cols = pDims[1];

               LOGGING("!! encountered fortran array while shifting! %lld x %lld\n", rows, cols);

               if (pArrayInfo->NDim >= 3) {
                  printf("!! too many dimensions to shift! %lld x %lld\n", rows, cols);
               }
               else {

                  char* pDst = pArrayInfo->pData;
                  char* pSrc = pDst + (pShift->shiftAmount * pArrayInfo->ItemSize);

                  INT64 rowsToMove = rows - pShift->shiftAmount;

                  // Check for negative shift value
                  if (pShift->shiftAmount < 0) {
                     rowsToMove = rows + pShift->shiftAmount;
                     pSrc = pArrayInfo->pData;
                     pDst = pSrc - (pShift->shiftAmount * pArrayInfo->ItemSize);
                  }

                  if (rowsToMove > 0) {
                     INT64 rowsToMoveSize = rowsToMove * pArrayInfo->ItemSize;
                     INT64 rowSize = rows * pArrayInfo->ItemSize;

                     // 2d shift
                     for (INT64 i = 0; i < cols; i++) {
                        memmove(pDst, pSrc, rowsToMoveSize);
                        pDst += rowSize;
                        pSrc += rowSize;
                     }
                  }
               }
            }
            else {
               // Example:
               // ArrayLength: 10000
               // shiftAmount:  1000
               // deltaShift:   9000 items to move
               //
               INT64 deltaShift = pArrayInfo->ArrayLength - pShift->shiftAmount;
               if (pShift->shiftAmount < 0) {
                  deltaShift = pArrayInfo->ArrayLength + pShift->shiftAmount;
               }
               // make sure something to shift
               if (deltaShift > 0) {

                  char* pTop1 = pArrayInfo->pData;

                  // make sure something to shift
                  if (pShift->shiftAmount < 0) {
                     char* pTop2 = pTop1 - (pShift->shiftAmount * pArrayInfo->ItemSize);
                     LOGGING("[%d] neg shifting %p %p  size: %lld  itemsize: %lld\n", core, pTop2, pTop1, deltaShift, pArrayInfo->ItemSize);
                     memmove(pTop2, pTop1, deltaShift * pArrayInfo->ItemSize);
                  }
                  else {
                     char* pTop2 = pTop1 + (pShift->shiftAmount * pArrayInfo->ItemSize);
                     LOGGING("[%d] pos shifting %p %p  size: %lld  itemsize: %lld\n", core, pTop1, pTop2, deltaShift, pArrayInfo->ItemSize);
                     memmove(pTop1, pTop2, deltaShift * pArrayInfo->ItemSize);
                  }
               }
            }
         }
         return TRUE;
      };

      g_cMathWorker->DoMultiThreadedWork((int)tupleSize, lambdaShiftCallback, &myshift);

      FreeArrayInfo(aInfo);
      Py_IncRef(inList1);
      return inList1;
   }

   PyErr_Format(PyExc_ValueError, "Unable to shift arrays");
   return NULL;
}



//-----------------------
// HomogenizeArrays
// Arg1: List of numpy arrays
// Arg2: Optional final dtype
//
// Returns: list of homogenized arrays
PyObject *HomogenizeArrays(PyObject* self, PyObject *args) {
   if (!PyTuple_Check(args)) {
      PyErr_Format(PyExc_ValueError, "HomogenizeArrays arguments needs to be a tuple");
      return NULL;
   }

   Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

   if (argTupleSize < 2) {
      PyErr_Format(PyExc_ValueError, "HomogenizeArrays requires two args instead of %llu args", argTupleSize);
      return NULL;
   }

   PyObject* inList1 = PyTuple_GetItem(args, 0);
   PyObject* dtypeObject = PyTuple_GetItem(args, 1);

   INT64 totalItemSize = 0;
   INT64 tupleSize = 0;

   // Do not allow jagged rows
   ArrayInfo* aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, TRUE);

   if (aInfo) {
      INT32 dtype = (INT32)PyLong_AsLong(dtypeObject);

      if (dtype != -1) {
         if (dtype < 0 || dtype > NPY_LONGDOUBLE) {
            PyErr_Format(PyExc_ValueError, "HomogenizeArrays: Dont know how to convert dtype num %d.  Please make sure all arrays are ints or floats.", dtype);
            return NULL;
         }
      }
      else {
         dtype = GetUpcastDtype(aInfo, tupleSize);
         if (dtype == -1) {
            return NULL;
         }
      }

      // Now convert?
      // if output is boolean, bad means FALSE
      void* pBadOutput1 = GetDefaultForType(dtype);

      PyObject*  returnList = PyList_New(0);

      // Convert any different types... build a new list...
      for (int t = 0; t < tupleSize; t++) {

         if (dtype != aInfo[t].NumpyDType) {
            CONVERT_SAFE convertSafe = GetConversionFunctionSafe(aInfo[t].NumpyDType, dtype);
            void* pBadInput1 = GetDefaultForType(aInfo[t].NumpyDType);
            PyArrayObject* pOutput = AllocateLikeNumpyArray(aInfo[t].pObject, dtype);

            // TODO: multithread this
            if (pOutput) {
               // preserve sentinels
               convertSafe(
                  aInfo[t].pData,
                  PyArray_BYTES(pOutput),
                  aInfo[t].ArrayLength,
                  pBadInput1,
                  pBadOutput1,
                  PyArray_STRIDE(aInfo[t].pObject, 0),
                  PyArray_STRIDE(pOutput, 0));

               // pylist_append will add a reference count but setitem will not
               PyList_Append(returnList, (PyObject*)pOutput);
               Py_DecRef((PyObject*)pOutput);
            }
         }
         else {
            // add a refernce
            PyList_Append(returnList, (PyObject*)aInfo[t].pObject);
         }
      }

      // Figure out which arrays will be recast

      FreeArrayInfo(aInfo);
      return returnList;
   }

   return NULL;
}

/**
 * Count the number of 'True' (nonzero) 1-byte bool values in an array,
 * using an AVX2-based implementation.
 *
 * @param pData Array of 1-byte bool values.
 * @param length The number of elements in the array.
 * @return The number of nonzero 1-byte bool values in the array.
 */
// TODO: When we support runtime CPU detection/dispatching, bring back the original popcnt-based implementation
//       of this function for systems that don't support AVX2. Also consider implementing an SSE-based version
//       of this function for the same reason (logic will be very similar, just using __m128i instead).
// TODO: Consider changing `length` to UINT64 here so it agrees better with the result of sizeof().
INT64 SumBooleanMask(const INT8* const pData, const INT64 length) {
   // Basic input validation.
   if (!pData)
   {
      return 0;
   }
   else if (length < 0)
   {
      return 0;
   }

   // Now that we know length is >= 0, it's safe to convert it to unsigned so it agrees with
   // the sizeof() math in the logic below.
   // Make sure to use this instead of 'length' in the code below to avoid signed/unsigned
   // arithmetic warnings.
   const auto ulength = static_cast<size_t>(length);

   // Holds the accumulated result value.
   INT64 result = 0;

   // YMM (32-byte) vector packed with 32 byte values, each set to 1.
   // NOTE: The obvious thing here would be to use _mm256_set1_epi8(1),
   //       but many compilers (e.g. MSVC) store the data for this vector
   //       then load it here, which unnecessarily wastes cache space we could be
   //       using for something else.
   //       Generate the constants using a few intrinsics, it's faster than even an L1 cache hit anyway.
   const auto zeros_ = _mm256_setzero_si256();
   // compare 0 to 0 returns 0xFF; treated as an int8_t, 0xFF = -1, so abs(-1) = 1.
   const auto ones = _mm256_abs_epi8(_mm256_cmpeq_epi8(zeros_, zeros_));

   //
   // Convert each byte in the input to a 0 or 1 byte according to C-style boolean semantics.
   //

   // This first loop does the bulk of the processing for large vectors -- it doesn't use popcount
   // instructions and instead relies on the fact we can sum 0/1 values to acheive the same result,
   // up to CHAR_MAX. This allows us to use very inexpensive instructions for most of the accumulation
   // so we're primarily limited by memory bandwidth.
   const size_t vector_length = ulength / sizeof(__m256i);
   const auto pVectorData = (__m256i*)pData;
   for (size_t i = 0; i < vector_length;)
   {
      // Determine how much we can process in _this_ iteration of the loop.
      // The maximum number of "inner" iterations here is CHAR_MAX (255),
      // because otherwise our byte-sized counters would overflow.
      const auto inner_loop_iters =
         std::min(
            static_cast<size_t>(std::numeric_limits<uint8_t>::max()),
            vector_length - i);

      // Holds the current per-vector-lane (i.e. per-byte-within-vector) popcount.
      // PERF: If necessary, the loop below can be manually unrolled to ensure we saturate memory bandwidth.
      auto byte_popcounts = _mm256_setzero_si256();
      for (size_t j = 0; j < inner_loop_iters; j++)
      {
         // Use an unaligned load to grab a chunk of data;
         // then call _mm256_min_epu8 where one operand is the register we set
         // earlier containing packed byte-sized 1 values (e.g. 0x01010101...).
         // This effectively converts each byte in the input to a 0 or 1 byte value.
         const auto cstyle_bools = _mm256_min_epu8(ones, _mm256_loadu_si256(&pVectorData[i + j]));

         // Since each byte in the converted vector now contains either a 0 or 1,
         // we can simply add it to the running per-byte sum to simulate a popcount.
         byte_popcounts = _mm256_add_epi8(byte_popcounts, cstyle_bools);
      }

      // Sum the per-byte-lane popcounts, then add them to the overall result.
      // For the vectorized partial sums, it's important the 'zeros' argument is used as the second operand
      // so that the zeros are 'unpacked' into the high byte(s) of each packed element in the result.
      const auto zeros = _mm256_setzero_si256();

      // Sum 32x 1-byte counts -> 16x 2-byte counts
      const auto byte_popcounts_8a = _mm256_unpacklo_epi8(byte_popcounts, zeros);
      const auto byte_popcounts_8b = _mm256_unpackhi_epi8(byte_popcounts, zeros);
      const auto byte_popcounts_16 = _mm256_add_epi16(byte_popcounts_8a, byte_popcounts_8b);

      // Sum 16x 2-byte counts -> 8x 4-byte counts
      const auto byte_popcounts_16a = _mm256_unpacklo_epi16(byte_popcounts_16, zeros);
      const auto byte_popcounts_16b = _mm256_unpackhi_epi16(byte_popcounts_16, zeros);
      const auto byte_popcounts_32 = _mm256_add_epi32(byte_popcounts_16a, byte_popcounts_16b);

      // Sum 8x 4-byte counts -> 4x 8-byte counts
      const auto byte_popcounts_32a = _mm256_unpacklo_epi32(byte_popcounts_32, zeros);
      const auto byte_popcounts_32b = _mm256_unpackhi_epi32(byte_popcounts_32, zeros);
      const auto byte_popcounts_64 = _mm256_add_epi64(byte_popcounts_32a, byte_popcounts_32b);

      // Sum 4x 8-byte counts -> 1x 32-byte count.
      const auto byte_popcount_256 =
         _mm256_extract_epi64(byte_popcounts_64, 0)
         + _mm256_extract_epi64(byte_popcounts_64, 1)
         + _mm256_extract_epi64(byte_popcounts_64, 2)
         + _mm256_extract_epi64(byte_popcounts_64, 3);

      // Add the accumulated popcount from this loop iteration (for 32*255 bytes) to the overall result.
      result += byte_popcount_256;

      // Increment the outer loop counter by the number of inner iterations we performed.
      i += inner_loop_iters;
   }

   // Handle the last few bytes, if any, that couldn't be handled with the vectorized loop.
   const auto vectorized_length = vector_length * sizeof(__m256i);
   for (size_t i = vectorized_length; i < ulength; i++)
   {
      if (pData[i])
      {
         result++;
      }
   }

   return result;
}


//--------------------------------------------------------------------------
// Array copy from one to another using boolean mask
void CopyItemBooleanMask(void* pSrcV, void* pDestV, INT8* pBoolMask, INT64 arrayLength, INT64 itemsize) {
   switch (itemsize) {

   case 1:
   {
      INT8* pSrc = (INT8*)pSrcV;
      INT8* pDest = (INT8*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            pDest[i] = *pSrc++;
         }
      }
      break;
   }
   case 2:
   {
      INT16* pSrc = (INT16*)pSrcV;
      INT16* pDest = (INT16*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            pDest[i] = *pSrc++;
         }
      }
      break;
   }
   case 4:
   {
      INT32* pSrc = (INT32*)pSrcV;
      INT32* pDest = (INT32*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            pDest[i] = *pSrc++;
         }
      }
      break;
   }
   case 8:
   {
      INT64* pSrc = (INT64*)pSrcV;
      INT64* pDest = (INT64*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            pDest[i] = *pSrc++;
         }
      }
      break;
   }
   default:
   {
      char* pSrc = (char*)pSrcV;
      char* pDest = (char*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            memcpy(pDest + (itemsize* i), pSrc, itemsize);
            pSrc += itemsize;
         }
      }
      break;

   }
   }
}


//--------------------------------------------------------------------------
// Copying scalars to array with boolean mask
void CopyItemBooleanMaskScalar(void* pSrcV, void* pDestV, INT8* pBoolMask, INT64 arrayLength, INT64 itemsize) {
   switch (itemsize) {

   case 1:
   {
      INT8* pSrc = (INT8*)pSrcV;
      INT8* pDest = (INT8*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            pDest[i] = *pSrc;
         }
      }
      break;
   }
   case 2:
   {
      INT16* pSrc = (INT16*)pSrcV;
      INT16* pDest = (INT16*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            pDest[i] = *pSrc;
         }
      }
      break;
   }
   case 4:
   {
      INT32* pSrc = (INT32*)pSrcV;
      INT32* pDest = (INT32*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            pDest[i] = *pSrc;
         }
      }
      break;
   }
   case 8:
   {
      INT64* pSrc = (INT64*)pSrcV;
      INT64* pDest = (INT64*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            pDest[i] = *pSrc;
         }
      }
      break;
   }
   default:
   {
      char* pSrc = (char*)pSrcV;
      char* pDest = (char*)pDestV;

      for (INT64 i = 0; i < arrayLength; i++) {
         if (*pBoolMask++) {
            memcpy(pDest + (itemsize* i), pSrc, itemsize);
         }
      }
      break;

   }
   }
}



//--------------------------------------------------------------------------
//
PyObject* SetItemBooleanMask(PyArrayObject* arr, PyArrayObject* mask, PyArrayObject* inValues, INT64 arrayLength) {

   //PyObject* boolsum =
   //   ReduceInternal(mask, REDUCE_SUM);
   INT8* pBoolMask = (INT8*)PyArray_BYTES(mask);
   INT64 bsum = SumBooleanMask(pBoolMask, ArrayLength(mask));

   // Sum the boolean array
   if (bsum > 0 && bsum == ArrayLength(inValues)) {

      void* pSrc = PyArray_BYTES(inValues);
      void* pDest = PyArray_BYTES(arr);

      CopyItemBooleanMask(pSrc, pDest, pBoolMask, arrayLength, PyArray_ITEMSIZE(arr));

      Py_IncRef(Py_True);
      return Py_True;
   }

   Py_IncRef(Py_False);
   return Py_False;
}


//--------------------------------------------------------------------------
// Must be a power of 2
// For smaller arrays, change this size
#define SETITEM_PARTITION_SIZE   16384

//--------------------------------------------------------------------------
// TODO: Refactor this method and SetItemBooleanMask
PyObject* SetItemBooleanMaskLarge(PyArrayObject* arr, PyArrayObject* mask, PyArrayObject* inValues, INT64 arrayLength) {
   struct ST_BOOLCOUNTER {
      INT8*    pBoolMask;
      INT64*   pCounts;
      INT64    sections;
      INT64    maskLength;
      char*    pSrc;
      char*    pDest;
      INT64    itemSize;

   } stBoolCounter;

   stBoolCounter.pSrc = (char*)PyArray_BYTES(inValues);
   stBoolCounter.pDest = (char*)PyArray_BYTES(arr);
   stBoolCounter.itemSize = PyArray_ITEMSIZE(arr);

   stBoolCounter.maskLength = ArrayLength(mask);
   stBoolCounter.sections = (stBoolCounter.maskLength + (SETITEM_PARTITION_SIZE-1)) / SETITEM_PARTITION_SIZE;

   INT64 allocSize = stBoolCounter.sections * sizeof(INT64);
   const INT64 maxStackAlloc = 1024 * 1024;  // 1 MB

   if (allocSize > maxStackAlloc) {
      stBoolCounter.pCounts = (INT64*)WORKSPACE_ALLOC(allocSize);
   }
   else {
      stBoolCounter.pCounts = (INT64*)alloca(allocSize);
   }
   if (stBoolCounter.pCounts) {
      stBoolCounter.pBoolMask = (INT8*)PyArray_BYTES(mask);

      auto lambdaCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
         ST_BOOLCOUNTER* pstBoolCounter = (ST_BOOLCOUNTER*)callbackArgT;
         INT64 t = workIndex;

         INT64 lastCount = SETITEM_PARTITION_SIZE;
         if (t == pstBoolCounter->sections - 1) {
            lastCount = pstBoolCounter->maskLength & (SETITEM_PARTITION_SIZE - 1);
            if (lastCount == 0) lastCount = SETITEM_PARTITION_SIZE;
         }
         pstBoolCounter->pCounts[workIndex] = SumBooleanMask(pstBoolCounter->pBoolMask + (SETITEM_PARTITION_SIZE * workIndex), lastCount);
         //printf("isum %lld %lld %lld\n", workIndex, pstBoolCounter->pCounts[workIndex], lastCount);

         return TRUE;
      };

      g_cMathWorker->DoMultiThreadedWork((int)stBoolCounter.sections, lambdaCallback, &stBoolCounter);

      // calculate the sum for each section
      INT64 bsum = 0;
      for (int i = 0; i < stBoolCounter.sections; i++) {
         INT64 temp = bsum;
         bsum += stBoolCounter.pCounts[i];
         stBoolCounter.pCounts[i] = temp;
      }

      INT64 arrlength = ArrayLength(inValues);

      if (bsum > 0 && bsum == arrlength) {

         auto lambda2Callback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
            ST_BOOLCOUNTER* pstBoolCounter = (ST_BOOLCOUNTER*)callbackArgT;
            INT64 t = workIndex;

            INT64 lastCount = SETITEM_PARTITION_SIZE;
            if (t == pstBoolCounter->sections - 1) {
               lastCount = pstBoolCounter->maskLength & (SETITEM_PARTITION_SIZE - 1);
               if (lastCount == 0) lastCount = SETITEM_PARTITION_SIZE;
            }
            INT64 adjustment = (SETITEM_PARTITION_SIZE * workIndex * pstBoolCounter->itemSize);
            CopyItemBooleanMask(
               pstBoolCounter->pSrc + (pstBoolCounter->pCounts[workIndex] * pstBoolCounter->itemSize),
               pstBoolCounter->pDest + adjustment,
               pstBoolCounter->pBoolMask + (SETITEM_PARTITION_SIZE * workIndex),
               lastCount,
               pstBoolCounter->itemSize);

            return TRUE;
         };

         g_cMathWorker->DoMultiThreadedWork((int)stBoolCounter.sections, lambda2Callback, &stBoolCounter);

         if (allocSize > maxStackAlloc) WORKSPACE_FREE(stBoolCounter.pCounts);
         Py_IncRef(Py_True);
         return Py_True;

      }

      if (bsum > 0 && arrlength == 1) {

         auto lambda2Callback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
            ST_BOOLCOUNTER* pstBoolCounter = (ST_BOOLCOUNTER*)callbackArgT;
            INT64 t = workIndex;

            INT64 lastCount = SETITEM_PARTITION_SIZE;
            if (t == pstBoolCounter->sections - 1) {
               lastCount = pstBoolCounter->maskLength & (SETITEM_PARTITION_SIZE - 1);
               if (lastCount == 0) lastCount = SETITEM_PARTITION_SIZE;
            }
            INT64 adjustment = (SETITEM_PARTITION_SIZE * workIndex * pstBoolCounter->itemSize);
            CopyItemBooleanMaskScalar(
               pstBoolCounter->pSrc,
               pstBoolCounter->pDest + adjustment,
               pstBoolCounter->pBoolMask + (SETITEM_PARTITION_SIZE * workIndex),
               lastCount,
               pstBoolCounter->itemSize);

            return TRUE;
         };

         g_cMathWorker->DoMultiThreadedWork((int)stBoolCounter.sections, lambda2Callback, &stBoolCounter);

         if (allocSize > maxStackAlloc) WORKSPACE_FREE(stBoolCounter.pCounts);
         Py_IncRef(Py_True);
         return Py_True;
      }
      if (allocSize > maxStackAlloc) WORKSPACE_FREE(stBoolCounter.pCounts);

   }
   LOGGING("bsum problem %lld  %lld\n", bsum, arrlength);
   Py_IncRef(Py_False);
   return Py_False;
}


//--------------------------------------------------------------------------
//def __setitem__(self, fld, value) :
//   : param fld : boolean or fancy index mask
//   : param value : scalar, sequence or dataset value as follows
//
// returns TRUE if it worked
// returns FALSE
// NOTE: This routine is not finished yet
PyObject *SetItem(PyObject* self, PyObject *args)
{
   //if (!PyTuple_Check(args)) {
   //   PyErr_Format(PyExc_ValueError, "SetItem arguments needs to be a tuple");
   //   return NULL;
   //}

   Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

   if (argTupleSize < 3) {
      PyErr_Format(PyExc_ValueError, "SetItem requires three args instead of %llu args", argTupleSize);
      return NULL;
   }

   PyArrayObject* arr = (PyArrayObject*)PyTuple_GetItem(args, 0);
   PyArrayObject* mask = (PyArrayObject*)PyTuple_GetItem(args, 1);

   // Try to convert value if we have to
   PyObject* value = PyTuple_GetItem(args, 2);
   if (!PyArray_Check(value)) {
      value = PyArray_FromAny(value, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
   }

   // Make sure from any worked
   if (value) {
      if (PyArray_Check(arr) && PyArray_Check(mask) && PyArray_Check(value)) {
         PyArrayObject* inValues = (PyArrayObject*)value;

         // check for strides, same itemsize, 1 dimensional
         // TODO: improvement when string itemsize is different length -- we can do a custom string copy
         if (PyArray_NDIM(arr) == 1 &&
            PyArray_ITEMSIZE(arr) > 0 &&
            PyArray_NDIM(inValues) == 1 &&
            PyArray_ITEMSIZE(inValues) == PyArray_ITEMSIZE(arr) &&
            PyArray_TYPE(arr) == PyArray_TYPE(inValues)) {

            // Boolean path...
            int arrType = PyArray_TYPE(mask);

            if (arrType == NPY_BOOL) {
               INT64 arrayLength = ArrayLength(arr);

               if (arrayLength == ArrayLength(mask)) {

                  if (arrayLength <= SETITEM_PARTITION_SIZE) {
                     return SetItemBooleanMask(arr, mask, inValues, arrayLength);

                  }
                  else {
                     // special count
                     return SetItemBooleanMaskLarge(arr, mask, inValues, arrayLength);
                  }
               }
            }
            else if (arrType <= NPY_LONGLONG) {
               // Assume int even if uint
               // Fancy index
               switch (PyArray_ITEMSIZE(arr)) {
               case 1:
                  break;
               case 2:
                  break;
               case 4:
                  break;
               case 8:
                  break;
               }
            }
         }
      }
     LOGGING("SetItem Could not convert value to array %d  %lld  %d  %lld\n", PyArray_NDIM(arr), PyArray_ITEMSIZE(arr), PyArray_NDIM((PyArrayObject*)value), PyArray_ITEMSIZE((PyArrayObject*)value));
   }
   else {
      LOGGING("SetItem Could not convert value to array\n");
   }
   // punt to numpy
   Py_IncRef(Py_False);
   return Py_False;

}

//----------------------------------------------------
// rough equivalvent arr[mask] = value[mask]
//
// returns TRUE if it worked
// returns FALSE
PyObject *PutMask(PyObject* self, PyObject *args)
{
   Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

   if (argTupleSize < 3) {
      PyErr_Format(PyExc_ValueError, "SetItem requires three args instead of %llu args", argTupleSize);
      return NULL;
   }

   PyArrayObject* arr = (PyArrayObject*)PyTuple_GetItem(args, 0);
   PyArrayObject* mask = (PyArrayObject*)PyTuple_GetItem(args, 1);

   // Try to convert value if we have to
   PyObject* value = PyTuple_GetItem(args, 2);
   if (!PyArray_Check(value)) {
      value = PyArray_FromAny(value, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
   }

   if (PyArray_Check(arr) && PyArray_Check(mask) && PyArray_Check(value)) {
      PyArrayObject* inValues = (PyArrayObject*)value;

      if (PyArray_TYPE(mask) == NPY_BOOL) {

         INT64 itemSizeOut = PyArray_ITEMSIZE(arr);
         INT64 itemSizeIn = PyArray_ITEMSIZE(inValues);

         // check for strides... ?
         INT64 arrayLength = ArrayLength(arr);
         if (arrayLength == ArrayLength(mask) && itemSizeOut == PyArray_STRIDE(arr,0)) {
            INT64 valLength = ArrayLength(inValues);

            if (arrayLength == valLength) {
               int outDType = PyArray_TYPE(arr);
               int inDType = PyArray_TYPE(inValues);
               MASK_CONVERT_SAFE maskSafe = GetConversionPutMask(inDType, outDType);

               if (maskSafe) {

                  // MT callback
                  struct MASK_CALLBACK_STRUCT {
                     MASK_CONVERT_SAFE maskSafe;
                     char* pIn;
                     char* pOut;
                     INT64 itemSizeOut;
                     INT64 itemSizeIn;
                     INT8* pMask;
                     void* pBadInput1;
                     void* pBadOutput1;

                  };

                  MASK_CALLBACK_STRUCT stMask;

                  // This is the routine that will be called back from multiple threads
                  auto lambdaMaskCallback = [](void* callbackArgT, int core, INT64 start, INT64 length) -> BOOL {
                     MASK_CALLBACK_STRUCT* callbackArg = (MASK_CALLBACK_STRUCT*)callbackArgT;

                     //printf("[%d] Mask %lld %lld\n", core, start, length);
                     //maskSafe(pIn, pOut, (INT8*)pMask, length, pBadInput1, pBadOutput1);
                     // Auto adjust pointers
                     callbackArg->maskSafe(
                        callbackArg->pIn + (start * callbackArg->itemSizeIn),
                        callbackArg->pOut + (start * callbackArg->itemSizeOut),
                        callbackArg->pMask + start,
                        length,
                        callbackArg->pBadInput1,
                        callbackArg->pBadOutput1);

                     return TRUE;
                  };

                  stMask.itemSizeIn = itemSizeIn;
                  stMask.itemSizeOut = itemSizeOut;
                  stMask.pBadInput1 = GetDefaultForType(inDType);
                  stMask.pBadOutput1 = GetDefaultForType(outDType);

                  stMask.pIn = (char*)PyArray_BYTES(inValues);
                  stMask.pOut = (char*)PyArray_BYTES(arr);
                  stMask.pMask = (INT8*)PyArray_BYTES(mask);
                  stMask.maskSafe = maskSafe;

                  g_cMathWorker->DoMultiThreadedChunkWork(arrayLength, lambdaMaskCallback, &stMask);

                  Py_IncRef(Py_True);
                  return Py_True;
               }
            }
         }
      }
   }

   // punt to numpy
   Py_IncRef(Py_False);
   return Py_False;

}



//-----------------------
// NOTE: Not completed
// Idea is to horizontally retrieve rows (from multiple columns) and to make a numpy array
//
// apply rows
// Arg1: List of numpy arrays
// Arg2: Optional final dtype
// Arg3: Function to call
// Arg4 +: args to pass
PyObject *ApplyRows(PyObject* self, PyObject *args, PyObject* kwargs)
{

   Py_ssize_t argTupleSize = PyTuple_GET_SIZE(args);

   if (argTupleSize < 2) {
      PyErr_Format(PyExc_ValueError, "ApplyRows requires two args instead of %llu args", argTupleSize);
      return NULL;
   }

   PyObject* arg1 = PyTuple_GetItem(args, 2);

   // Check if callable
   if (!PyCallable_Check(arg1)) {
      PyTypeObject* type = (PyTypeObject*)PyObject_Type(arg1);

      PyErr_Format(PyExc_ValueError, "Argument must be a function or a method not %s\n", type->tp_name);
      return NULL;
   }

   PyFunctionObject* function = GetFunctionObject(arg1);

   if (function) {
      PyObject* inList1 = PyTuple_GetItem(args, 0);
      PyObject* dtypeObject= PyTuple_GetItem(args, 1);

      INT64 totalItemSize = 0;
      INT64 tupleSize = 0;

      // Do not allow jagged rows
      ArrayInfo* aInfo = BuildArrayInfo(inList1, &tupleSize, &totalItemSize, TRUE);

      if (aInfo) {
         INT32 dtype = (INT32)PyLong_AsLong(dtypeObject);

         if (dtype != -1) {
            if (dtype < 0 || dtype > NPY_LONGDOUBLE) {
               PyErr_Format(PyExc_ValueError, "Dont know how to convert dtype num %d.  Please make sure all arrays are ints or floats.", dtype);
               return NULL;
            }
         }
         else {
            dtype = GetUpcastDtype(aInfo, tupleSize);
         }

         // Now convert?
         // if output is boolean, bad means FALSE
         void* pBadOutput1 = GetDefaultForType(dtype);

         // Convert any different types... build a new list...
         // Figure out which arrays will be recast

         FreeArrayInfo(aInfo);
      }
   }

   return NULL;

}