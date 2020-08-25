#include "stdafx.h"
//#include "..\TestSD\stdafx.h"
//#include <Windows.h>
//#include <stdio.h>
#include <memory>
#include "CommonInc.h"
#include "RipTide.h"
#include "HashLinear.h"
#include "MathWorker.h"
#include "Recycler.h"

#ifndef LogError
#define LogError(...)
#endif

#ifndef LogInform
#define LogInform printf
//#define LogInform(...)
#endif

#define LOGGING(...)
//#define LOGGING printf

#define GB_INVALID_INDEX -1

// TODO: Use C++14 variable templates here once we no longer need to support GCC 4.x.
//template<typename T>
//static constexpr int numpy_type_code = -1;   // -1 is an invalid code, because if something uses this specialization we want it to break.
//
//template<>
//static constexpr int numpy_type_code<INT32> = NPY_INT32;
//
//template<>
//static constexpr int numpy_type_code<INT64> = NPY_INT64;

/* Template-based, compile-time mapping between C++ types and numpy type codes (e.g. NPY_FLOAT64). */

template<typename T>
struct numpy_type_code
{
   static constexpr int value = -1;    // -1 is an invalid code, because if something uses this specialization we want it to break.
};

template<>
struct numpy_type_code<INT32>
{
   static constexpr int value = NPY_INT32;
};

template<>
struct numpy_type_code<INT64>
{
   static constexpr int value = NPY_INT64;
};



// Use this table to find a suitable hash size
INT64 PRIME_NUMBERS[] = {
   53,
   97,
   193,
   389,
   769,
   1543,
   3079,
   6151,
   12289,
   24593,
   49157,
   98317,
   196613,
   393241,
   786433,
   1572869,
   3145739,
   6291469,
   12582917,
   25165843,
   50331653,
   100663319,
   201326611,
   402653189,
   805306457,
   1610612741,
   3002954501,
   4294967291,
   8589934583,
   17179869143,
   34359738337,
   68719476731,
   0
};

#define MEMCMP_NEW(ARG1, ARG2, ARG3, ARG4) \
{ ARG1 = 0; const char* pSrc1 = ARG2; const char* pSrc2 = ARG3; INT64 length = ARG4; \
   while (length >= 8) { \
      UINT64* p1 = (UINT64*)pSrc1; \
      UINT64* p2 = (UINT64*)pSrc2;\
      if (*p1 != *p2) {ARG1=1; length=0; break;}\
      length -= 8;\
      pSrc1 += 8;\
      pSrc2 += 8;\
   }\
   if (length >= 4) { \
      UINT32* p1 = (UINT32*)pSrc1; \
      UINT32* p2 = (UINT32*)pSrc2; \
      if (*p1 != *p2) { \
         ARG1 = 1; length = 0; \
      } \
      else { \
         length -= 4; \
         pSrc1 += 4; \
         pSrc2 += 4;\
      }\
   }\
   while (length > 0) { \
      if (*pSrc1 != *pSrc2) {ARG1 = 1; break;}\
      length -= 1;\
      pSrc1 += 1;\
      pSrc2 += 1;\
   }\
}


//static inline UINT32  sse42_crc32(const uint8_t *bytes, size_t len)
//{
//   UINT32 hash = 0;
//   INT64 i = 0;
//   for (i = 0; i<len; i++) {
//      hash = _mm_crc32_u8(hash, bytes[i]);
//   }
//   return hash;
//}

static FORCEINLINE UINT64 crchash64(const char *buf, size_t count) {
   UINT64 h = 0;

   while (count >= 8) {
      h = _mm_crc32_u64(h, *(UINT64*)buf);
      count -= 8;
      buf += 8;
   }

   UINT64 t = 0;

   switch (count) {
   case 7: t ^= (UINT64)buf[6] << 48;
   case 6: t ^= (UINT64)buf[5] << 40;
   case 5: t ^= (UINT64)buf[4] << 32;
   case 4: t ^= (UINT64)buf[3] << 24;
   case 3: t ^= (UINT64)buf[2] << 16;
   case 2: t ^= (UINT64)buf[1] << 8;
   case 1: t ^= (UINT64)buf[0];
      return _mm_crc32_u64(h, t);
   }
   return h;
}



// This algorithm comes from
// https://code.google.com/archive/p/fast-hash/
//
// See also"
// https://github.com/rurban/smhasher/issues/5
// Compression function for Merkle-Damgard construction.
// This function is generated using the framework provided.
#define mix(h) h ^= h >> 23; h *= 0x2127599bf4325c37ULL;	 h ^= h >> 47;

static FORCEINLINE UINT64 fasthash64(const void *buf, size_t len)
{
   UINT64 seed = 0;
   const UINT64    m = 0x880355f21e6d1965ULL;
   const UINT64 *pos = (const UINT64 *)buf;
   const unsigned char *pos2;
   UINT64 h = seed ^ (len * m);
   UINT64 v;

   while (len >=8) {
      v = *pos++;
      h ^= mix(v);
      h *= m;
      len -= 8;
   }

   pos2 = (const unsigned char*)pos;
   v = 0;

   switch (len) {
   case 7: v ^= (UINT64)pos2[6] << 48;
   case 6: v ^= (UINT64)pos2[5] << 40;
   case 5: v ^= (UINT64)pos2[4] << 32;
   case 4: v ^= (UINT64)pos2[3] << 24;
   case 3: v ^= (UINT64)pos2[2] << 16;
   case 2: v ^= (UINT64)pos2[1] << 8;
   case 1: v ^= (UINT64)pos2[0];
      h ^= mix(v);
      h *= m;
   }

   return mix(h);
}


// --------------------------------------------
// Used when we know 8byte hash
FORCEINLINE UINT64 fasthash64_8(UINT64 v)
{
   UINT64 seed = 0;
   const UINT64    m = 0x880355f21e6d1965ULL;
   UINT64 h = seed ^  m;

   h ^= mix(v);
   h *= m;

   return mix(h);
}

// --------------------------------------------
// Used when we know 8byte hash
FORCEINLINE UINT64 fasthash64_16(UINT64* v)
{
   UINT64 seed = 0;
   const UINT64    m = 0x880355f21e6d1965ULL;
   UINT64 h = seed ^ m;

   h ^= mix(v[0]);
   h *= m;
   h ^= mix(v[0]);
   h *= m;

   return mix(h);
}

//===============================================================================================
// NOTE: Both of these routines are fast
#define DEFAULT_HASH64 crchash64
//#define DEFAULT_HASH64 fasthash64
//===============================================================================================


//=====================================================================================================================
// STRING
//=====================================================================================================================


#define HASH_STRING()          \
const char* strStart = (pHashList+(i*strWidth)); \
const char* str = strStart; \
const char* strEnd = str + strWidth; \
while (*str !=0) if (++str >= strEnd) break;         \
UINT64 hash = fasthash64(strStart, str - strStart);


#define HASH_UNICODE()          \
const char* strStart = (pHashList+(i*strWidth)); \
const UINT32* str = (UINT32*)strStart; \
const UINT32* strEnd = str + strWidth/4; \
UINT32 hash = 0;                  \
UINT32  c;                            \
while ((c = *str) !=0) { \
   str++;                            \
   hash = _mm_crc32_u32(hash, c);    \
   if (str >= strEnd) break;         \
 }


FORCE_INLINE
BOOLEAN
UNICODE_MATCH(const char* str1T, const char* str2T, INT64 strWidth1) {
   const UINT32* str1 = (const UINT32*)str1T;
   const UINT32* str2 = (const UINT32*)str2T;
   strWidth1 /= 4;
   while (strWidth1 > 0) {
      if (*str1 != *str2) return FALSE;
      ++str1;
      ++str2;
      --strWidth1;
   }
   return TRUE;
}

FORCE_INLINE
BOOLEAN
STRING_MATCH(const char* str1, const char* str2, INT64 strWidth1) {
   while (strWidth1 > 0) {
      if (*str1 != *str2) return FALSE;
      ++str1;
      ++str2;
      --strWidth1;
   }
   return TRUE;
}


FORCE_INLINE
BOOLEAN
UNICODE_MATCH2(const char* str1T, const char* str2T, INT64 strWidth1, INT64 strWidth2) {
   const UINT32* str1 = (const UINT32*)str1T;
   const UINT32* str2 = (const UINT32*)str2T;
   strWidth1 /= 4;
   strWidth2 /= 4;
   while (1) {

      // Check for when one string ends and the other has not yet
      if (strWidth1 == 0) {
         if (*str2 == 0) return TRUE;
         return FALSE;
      }
      if (strWidth2 == 0) {
         if (*str1 == 0) return TRUE;
         return FALSE;
      }

      if (*str1 != *str2) return FALSE;
      ++str1;
      ++str2;
      --strWidth1;
      --strWidth2;
   }
   return TRUE;
}


FORCE_INLINE
BOOLEAN
STRING_MATCH2(const char* str1, const char* str2, INT64 strWidth1, INT64 strWidth2) {
   while (1) {

      // Check for when one string ends and the other has not yet
      if (strWidth1 == 0) {
         if (*str2 == 0) return TRUE;
         return FALSE;
      }
      if (strWidth2 == 0) {
         if (*str1 == 0) return TRUE;
         return FALSE;
      }

      if (*str1 != *str2) return FALSE;
++str1;
++str2;
--strWidth1;
--strWidth2;
   }
   return TRUE;
}


template<typename T, typename U>
INT64 CHashLinear<T, U>::GetHashSize(
   INT64 numberOfEntries) {

   // Check for perfect hashes first when type size is small
   //if (sizeof(T) == 1) {
   //   return 256;
   //}

   //if (sizeof(T) == 2) {
   //   return 65536;
   //}

   if (HashMode == HASH_MODE_PRIME) {
      int i = 0;
      while (PRIME_NUMBERS[i] != 0) {
         if (PRIME_NUMBERS[i] > numberOfEntries) {
            return PRIME_NUMBERS[i];
         }
         i++;
      }
      LogError("**Failed to find prime number for hash size %lld\n", numberOfEntries);
      return 0;
   }
   else {
      // Power of 2 search
      int i = 0;
      while ((1LL << i) < numberOfEntries) i++;

      INT64 result= (1LL << i);
      INT64 maxhashsize = (1LL << 31);

      // TODO: really need to change strategies if high unique count and high row count
      if (result > maxhashsize) {
         result = maxhashsize;
      }
      return result;
   }
}



//-----------------------------------------------
template<typename T, typename U>
void CHashLinear<T, U>::FreeMemory(BOOL forceDeallocate) {

   if (forceDeallocate || Deallocate) {
      //printf("deallocating\n");
      WorkSpaceFreeAllocLarge(pHashTableAny, HashTableAllocSize);
   }
   else {
      //printf("not deallocating\n");
   }

   void* pTemp = pBitFields;
   WorkSpaceFreeAllocSmall(pTemp, BitAllocSize);
   pBitFields = NULL;
}

//-----------------------------------------------
template<typename T, typename U>
void CHashLinear<T, U>::AllocAndZeroBitFields(UINT64 hashSize) {

   // Calculate how many bytes we need (in 64 bit/8 byte increments)
   BitAllocSize = (HashSize + 63) / 64;
   BitAllocSize *= sizeof(UINT64);

   pBitFields = (UINT64*)WorkSpaceAllocSmall(BitAllocSize);

   if (pBitFields) {
      // Fill with zeros to indicate no hash position is used yet
      RtlZeroMemory(pBitFields, BitAllocSize);
   }
}


//-----------------------------------------------
// returns pointer to allocated array
template<typename T, typename U>
char* CHashLinear<T, U>::AllocHashTable(size_t allocSize) {

   HashTableAllocSize = allocSize;
   pHashTableAny = WorkSpaceAllocLarge(HashTableAllocSize);

   return (char*)pHashTableAny;

   //// Do not zero it
   ////RtlZeroMemory(pHashTableAny, HashTableAllocSize);
}


//-----------------------------------------------
// Allocates based on HashMode
// and size of structure
//
// Returns: pointer to extra section if requested
// Returns NULL does not mean failure if sizeofExtraArray=0
template<typename T, typename U>
void* CHashLinear<T, U>::AllocMemory(
   INT64 numberOfEntries,
   INT64 sizeofStruct,
   INT64 sizeofExtraArray,
   BOOL  isFloat) {

   if (sizeofStruct == -1) {
      sizeofStruct = sizeof(SingleKeyEntry);
   }
   if (sizeofStruct == -2) {
      sizeofStruct = sizeof(MultiKeyEntry);
   }

   NumEntries = numberOfEntries;
   //HashSize= GetHashSize(NumEntries*2);

   if (HashMode == HASH_MODE_MASK) {
      // For float *8 seems to help
      // For INT, it does not
      if (isFloat) {
         HashSize = GetHashSize(NumEntries * 8);
      }
      else {
         HashSize= GetHashSize(NumEntries*2);
      }

      // Helps avoid collisions for low number of strings
      LOGGING("Checking to up HashSize %llu  %llu  %llu\n", HashSize, sizeof(T), sizeof(U));

      // NOTE for 1 byte AND 2 bytes we try to do a perfect hash
      if (HashSize < 65536) HashSize = 65536;
   }
   else {
      HashSize = GetHashSize(NumEntries * 2);
   }

   size_t allocSize = HashSize * sizeofStruct;

   FreeMemory(TRUE);

   LOGGING("Hash size selected %llu for NumEntries %llu -- allocation size %llu\n", HashSize, NumEntries, allocSize);

   AllocAndZeroBitFields(HashSize);

   if (pBitFields) {
      if (sizeofExtraArray) {
         // 128 byte padding
         INT64 padSize = (allocSize + 127) & ~127;
         char* pRootBuffer = AllocHashTable(padSize + sizeofExtraArray);

         if (pRootBuffer) {
            // return pointer to extra section
            return pRootBuffer + padSize;
         }
         else {
            //memory fail
            CHECK_MEMORY_ERROR(NULL);
            return NULL;
         }
      }
      else {
         // ok
         AllocHashTable(allocSize);
         return NULL;
      }
   }
   // memory fail
   CHECK_MEMORY_ERROR(NULL);
   return NULL;
}



//-----------------------------------------------
// looks for the index of set location
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T,U>::InternalGetLocation(
   U  i,
   HashLocation* pLocation,
   INT8* pBooleanOutput,
   U*    pLocationOutput,
   T     item,
   UINT64 hash) {

   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      if (pLocation[hash].value == item) {
         // return the first location
         pLocationOutput[i] = pLocation[hash].Location;
         pBooleanOutput[i] = 1;
         return;
      }

      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;
      }
   }
   // Not found
   pLocationOutput[i] = BAD_INDEX ;
   pBooleanOutput[i] = 0;
}



//-----------------------------------------------
// looks for the index of set location
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalGetLocationCategorical(
   U  i,
   HashLocation* pLocation,
   U*    pLocationOutput,
   T     item,
   UINT64 hash,
   INT64 *missed) {

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      if (pLocation[hash].value == item) {
         // return the first location
         pLocationOutput[i] = pLocation[hash].Location + 1;
         return;
      }

      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;
      }
   }
   // Not found
   pLocationOutput[i] = 0;
   *missed = 1;
}



//-----------------------------------------------
// stores the index of the first location
// remove the forceline to make debugging easier
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalSetLocation(
   U  i,
   HashLocation* pLocation,
   T  item,
   UINT64 hash) {

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      if (pLocation[hash].value == item) {
         // Duplicate
         return;
      }


      // This entry is not us so we must have collided
      //++NumCollisions;

      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;

         //if (NumCollisions > (HashSize * 2)) {
         //   LogError("hash collision error %d %llu\n", i, NumCollisions);
         //   LogError("Bad hash function, too many collisions");
         //   return;
         //}
      }
   }
   // Failed to find hash
   SetBit(hash);
   //++NumUnique;
   pLocation[hash].Location = i;
   pLocation[hash].value = item;

}

#define INTERNAL_SET_LOCATION \
while (1) { \
   if (pBitFields[hash >> 6] & (1LL << (hash & 63))) { \
      /* Check if we have a match from before */ \
      if (pLocation[hash].value == item) { \
         /* Duplicate */ \
         break; \
      } \
      /* This entry is not us so we must have collided */ \
      /* Linear goes to next position */ \
      if (++hash >= HashSize) {  \
         hash = 0; \
      } \
   } \
   else { \
      /* Failed to find hash */ \
      /* Set the visit bit */ \
      pBitFields[hash >> 6] |= (1LL << (hash & 63)); \
      pLocation[hash].Location = i; \
      pLocation[hash].value = item; \
      break; \
   } \
}



//-----------------------------------------------
// stores the index of the first location
//
template<typename T, typename U>
void CHashLinear<T, U>::MakeHashLocationMK(
   INT64 arraySize,
   T*    pInput,
   INT64 totalItemSize,
   INT64 hintSize) {

   if (hintSize == 0) {
      hintSize = arraySize;
   }

   AllocMemory(hintSize, sizeof(HashLocationMK), 0, FALSE);
   //UINT64 NumUnique = 0;
   //UINT64 NumCollisions = 0;

   HashLocationMK* pLocation = (HashLocationMK*)pHashTableAny;

   UINT64*     pBitFields = this->pBitFields;

   if (!pLocation || !pBitFields) {
      return;
   }

   LOGGING("in mkhashlocationmk asize: %llu   isize: %llu  HashSize:%lld  sizeofU:%d\n", arraySize, totalItemSize, (long long)HashSize, (int)sizeof(U));

   for (U i = 0; i < arraySize; i++) {
      const char* pMatch = pInput + (totalItemSize*i);
      UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);
      hash = hash & (HashSize - 1);

      //printf("[%d] \n", (int)i);

      while (1) {
         UINT64 index = hash >> 6;
         if (pBitFields[index] & (1LL << (hash & 63))) {
            // Check if we have a match from before
            U Last = pLocation[hash].Location;
            const char* pMatch2 = pInput + (totalItemSize*Last);

            // TJD: August 2018 --unrolled MEMCMP and got 10-15% speedup
            const char* pSrc1 = pMatch;
            const char* pSrc2 = pMatch2;
            INT64 length = totalItemSize;

            while (length >= 8) {
               UINT64* p1 = (UINT64*)pSrc1;
               UINT64* p2 = (UINT64*)pSrc2;

               if (*p1 != *p2) goto FAIL_MATCH;
               length -= 8;
               pSrc1 += 8;
               pSrc2 += 8;
            }
            if (length >= 4) {
               UINT32* p1 = (UINT32*)pSrc1;
               UINT32* p2 = (UINT32*)pSrc2;

               if (*p1 != *p2) goto FAIL_MATCH;
               length -= 4;
               pSrc1 += 4;
               pSrc2 += 4;
            }

            while (length > 0) {
               if (*pSrc1 != *pSrc2) goto FAIL_MATCH;
               length -= 1;
               pSrc1 += 1;
               pSrc2 += 1;
            }

            //printf("[%d] matched \n", (int)i);
            // Matched
            break;

         FAIL_MATCH:
            //printf("[%d] collided \n", (int)i);

            // This entry is not us so we must have collided
            //++NumCollisions;

            // Linear goes to next position
            if (++hash >= HashSize) {
               hash = 0;
            }
            continue;
         }


         // Failed to find hash
         //printf("[%d] fail \n", (int)i);
         pBitFields[hash >> 6] |= (1LL << (hash & 63));

         //++NumUnique;
         pLocation[hash].Location = i;
         break;
      }
   }

   //LOGGING("%llu entries had %llu collisions   %llu unique\n", arraySize, NumCollisions, NumUnique);
   //printf("%llu entries had %llu collisions   %llu unique\n", arraySize, NumCollisions, NumUnique);

}


//-----------------------------------------------
// stores the index of the last matching index
// ASSUMES that pVal1 and pVal2 are SORTED!!
//  U is INT32 or INT64
//  V is INT32, INT64, float, or double
template<typename U, typename V>
void FindLastMatchCategorical(
   INT64    arraySize1,
   INT64    arraySize2,
   U*       pKey1,
   U*       pKey2,
   V*       pVal1,
   V*       pVal2,
   U*       pLocationOutput,
   INT64    totalUniqueSize) {

   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));
   U*       pLocation = (U*)WORKSPACE_ALLOC(totalUniqueSize * sizeof(U));

   for (U k = 0; k < totalUniqueSize; k++) {
      pLocation[k] = -1;
   }

   U i = 0;
   U j = 0;
   while (i < arraySize1 && j < arraySize2) {
      if (pVal1[i] < pVal2[j]) {
         // key1 is first
         U lookup = pKey1[i];

         if (pLocation[lookup] != -1) {
            // We have a match from before, update with key1
            pLocationOutput[i] = pLocation[lookup];
         }
         else {
            // failed to match
            pLocationOutput[i] = BAD_INDEX;
         }
         i++;
      }
      else {
         //key2 is first
         U lookup = pKey2[i];
         pLocation[lookup] = j;
         j++;
      }
   }

   while (i < arraySize1) {
      U lookup = pKey1[i];

      if (pLocation[lookup] != -1) {
         pLocationOutput[i] = pLocation[lookup];
      }
      else {
         pLocationOutput[i] = BAD_INDEX;
      }
      i++;
   }

   WORKSPACE_FREE(pLocation);

}


//-----------------------------------------------
// stores the index of the last matching index
// ASSUMES that pVal1 and pVal2 are SORTED!!
template<typename T, typename U> template<typename V>
void CHashLinear<T, U>::FindLastMatchMK(
   INT64 arraySize1,
   INT64 arraySize2,
   T*    pKey1,
   T*    pKey2,
   V*    pVal1,
   V*    pVal2,
   U*    pLocationOutput,
   INT64 totalItemSize,
   bool allowExact) {

   AllocMemory(arraySize2, sizeof(HashLocationMK), 0, FALSE);

   HashLocationMK* pLocation = (HashLocationMK*)pHashTableAny;

   if (!pLocation || !pBitFields) {
      return;
   }

   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   U i = 0;
   U j = 0;
   while (i < arraySize1 && j < arraySize2) {
      if (pVal1[i] < pVal2[j] || (!allowExact &&  pVal1[i] == pVal2[j]) ) {
         const char* pMatch1 = pKey1 + (totalItemSize*i);
         UINT64 hash = DEFAULT_HASH64(pMatch1, totalItemSize);
         hash = hash & (HashSize - 1);

         //TODO: should maybe put begin .. end into function implementing lookup for hashmap
         //begin
         while (1) {
            if (IsBitSet(hash)) {
               // Check if we have a match from before
               U Last = pLocation[hash].Location;
               const char* pMatch2 = pKey2 + (totalItemSize*Last);
               int mresult;
               MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
               if (!mresult) {
                  pLocationOutput[i] = pLocation[hash].Location;
                  break;
               }

               // Linear goes to next position
               if (++hash >= HashSize) {
                  hash = 0;
               }
               continue;
            }
            // Not found
            pLocationOutput[i] = BAD_INDEX;
            break;
         }
         //end
         i++;

      }
      else {
         const char* pMatch1 = pKey2 + (totalItemSize*j);
         UINT64 hash = DEFAULT_HASH64(pMatch1, totalItemSize);
         hash = hash & (HashSize - 1);

         //TODO: should maybe start .. end into a function implementing insertion for hashmap
         //begin
         while (1) {
            if (IsBitSet(hash)) {
               // Check if we have a match from before
               U Last = pLocation[hash].Location;
               const char* pMatch2 = pKey2 + (totalItemSize*Last);
               int mresult;
               MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
               if (!mresult) {
                  pLocation[hash].Location = j;
                  break;
               }

               // Linear goes to next position
               if (++hash >= HashSize) {
                  hash = 0;
               }
               continue;
            }
            // Failed to find hash
            SetBit(hash);
            pLocation[hash].Location = j;
            break;
         }
         //end

         j++;
      }

   }
   while (i < arraySize1) {
      const char* pMatch1 = pKey1 + (totalItemSize*i);
      UINT64 hash = DEFAULT_HASH64(pMatch1, totalItemSize);
      hash = hash & (HashSize - 1);

      //TODO: should maybe put begin .. end into function implementing lookup for hashmap
      //begin
      while (1) {
         if (IsBitSet(hash)) {
            // Check if we have a match from before
            U Last = pLocation[hash].Location;
            const char* pMatch2 = pKey2 + (totalItemSize*Last);
            int mresult;
            MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
            if (!mresult) {
               // return the first location
               pLocationOutput[i] = pLocation[hash].Location;
               break;
            }

            // Linear goes to next position
            if (++hash >= HashSize) {
               hash = 0;
            }
            continue;
         }
         // Not found
         //pLocationOutput[i] = i;
         pLocationOutput[i] = BAD_INDEX;
         break;
      }
      //end
      i++;
   }

}


//-----------------------------------------------
// stores the index of the next matching index
// ASSUMES that pVal1 and pVal2 are SORTED!!
// TODO: Clean this up and merge with FindLastMatchMk
template<typename T, typename U> template<typename V>
void CHashLinear<T, U>::FindNextMatchMK(
   INT64 arraySize1,
   INT64 arraySize2,
   T*    pKey1,
   T*    pKey2,
   V*    pVal1,
   V*    pVal2,
   U*    pLocationOutput,
   INT64 totalItemSize,
   bool allowExact) {

   AllocMemory(arraySize2, sizeof(HashLocationMK), 0, FALSE);
   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   HashLocationMK* pLocation = (HashLocationMK*)pHashTableAny;
   if (!pLocation || !pBitFields) {
      return;
   }

   U i = (U)(arraySize1 - 1);
   U j = (U)(arraySize2 - 1);
   while (i >= 0 && j >= 0) {
      if (pVal1[i] > pVal2[j] || (!allowExact &&  pVal1[i] == pVal2[j])) {
         const char* pMatch1 = pKey1 + (totalItemSize*i);
         UINT64 hash = DEFAULT_HASH64(pMatch1, totalItemSize);
         hash = hash & (HashSize - 1);

         //TODO: should maybe put begin .. end into function implementing lookup for hashmap
         //begin
         while (1) {
            if (IsBitSet(hash)) {
               // Check if we have a match from before
               U Last = pLocation[hash].Location;
               const char* pMatch2 = pKey2 + (totalItemSize*Last);
               int mresult;
               MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
               if (!mresult) {
                  pLocationOutput[i] = pLocation[hash].Location;
                  break;
               }

               // Linear goes to next position
               if (++hash >= HashSize) {
                  hash = 0;
               }
               continue;
            }
            // Not found
            pLocationOutput[i] = BAD_INDEX;
            break;
         }
         //end
         i--;

      }
      else {
         const char* pMatch1 = pKey2 + (totalItemSize*j);
         UINT64 hash = DEFAULT_HASH64(pMatch1, totalItemSize);
         hash = hash & (HashSize - 1);

         //TODO: should maybe start .. end into a function implementing insertion for hashmap
         //begin
         while (1) {
            if (IsBitSet(hash)) {
               // Check if we have a match from before
               U Last = pLocation[hash].Location;
               const char* pMatch2 = pKey2 + (totalItemSize*Last);
               int mresult;
               MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
               if (!mresult) {
                  pLocation[hash].Location = j;
                  break;
               }

               // Linear goes to next position
               if (++hash >= HashSize) {
                  hash = 0;
               }
               continue;
            }
            // Failed to find hash
            SetBit(hash);
            pLocation[hash].Location = j;
            break;
         }
         //end

         j--;
      }

   }
   while (i >= 0) {
      const char* pMatch1 = pKey1 + (totalItemSize*i);
      UINT64 hash = DEFAULT_HASH64(pMatch1, totalItemSize);
      hash = hash & (HashSize - 1);

      //TODO: should maybe put begin .. end into function implementing lookup for hashmap
      //begin
      while (1) {
         if (IsBitSet(hash)) {
            // Check if we have a match from before
            U Last = pLocation[hash].Location;
            const char* pMatch2 = pKey2 + (totalItemSize*Last);
            int mresult;
            MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
            if (!mresult) {
               // return the first location
               pLocationOutput[i] = pLocation[hash].Location;
               break;
            }

            // Linear goes to next position
            if (++hash >= HashSize) {
               hash = 0;
            }
            continue;
         }
         // Not found
         //pLocationOutput[i] = i;
         pLocationOutput[i] = BAD_INDEX;
         break;
      }
      //end
      i--;
   }

}



//-----------------------------------------------
// T is the input type
// U is the index output type (int8/16/32/64)
//
// outputs boolean array
// outputs location array
//
template<typename T, typename U>
static void IsMemberMK(
   void* pHashLinearVoid,
   INT64 arraySize,
   void* pInputT,
   void* pInput2T,
   INT8* pBooleanOutput,
   void* pLocationOutputU,
   INT64 totalItemSize) {

   struct HashLocationMK
   {
      U        Location;
   };

   CHashLinear<T, U>* pHashLinear = (CHashLinear<T, U>*)pHashLinearVoid;

   HashLocationMK* pLocation = (HashLocationMK*)pHashLinear->pHashTableAny;

   U* pLocationOutput = (U*)pLocationOutputU;
   T* pInput = (T*)pInputT;
   T* pInput2 = (T*)pInput2T;

   UINT64 HashSize = pHashLinear->HashSize;

   // to determine if hash location has been visited
   UINT64*     pBitFields = pHashLinear->pBitFields;
   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   LOGGING("in ismembermk  asize: %llu   isize: %llu  %p   %p  %p\n", arraySize, totalItemSize, pBitFields, pLocationOutput, pBooleanOutput);

   for (U i = 0; i < arraySize; i++) {
      const char* pMatch = pInput + (totalItemSize*i);
      UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);
      hash = hash & (HashSize - 1);

      //printf("[%d] %llu\n", (int)i, hash);

      while (1) {
         UINT64 index = hash >> 6;
         if (pBitFields[index] & (1LL << (hash & 63))) {
            // Check if we have a match from before
            U Last = pLocation[hash].Location;
            const char* pMatch2 = pInput2 + (totalItemSize*Last);
            int mresult;
            MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
            if (!mresult) {
               // return the first location
               pLocationOutput[i] = pLocation[hash].Location;
               pBooleanOutput[i] = 1;
               break;
            }

            // Linear goes to next position
            if (++hash >= HashSize) {
               hash = 0;
            }
            //printf("[%d] continue\n", (int)i);
            continue;
         }
         //printf("[%d] Not found\n", (int)i);
         // Not found
         pLocationOutput[i] = BAD_INDEX;
         pBooleanOutput[i] = 0;
         break;
      }
   }

   LOGGING("Done with ismembermk\n");

}



//=============================================
#define HASH_INT1  INT64 h = (INT64)item; h= h % HashSize;

//#define HASH_INT32  UINT64 h = (UINT64)item; h ^= (h >> 16); h= h & (HashSize-1);
//#define HASH_INT32 UINT64 h= fasthash64_8(item) & (HashSize-1);
#define HASH_INT32  UINT64 h= _mm_crc32_u32(0, (UINT32)item) & (HashSize-1);

//#define HASH_INT64  UINT64 h= _mm_crc32_u64(0, item) & (HashSize-1);
#define HASH_INT64 UINT64 h= fasthash64_8(item) & (HashSize-1);

#define HASH_INT128 UINT64 h= fasthash64_16((UINT64*)&pInput[i]) & (HashSize-1);

//=============================================
//#define HASH_FLOAT1  h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16;   h = h % HashSize;
//#define HASH_FLOAT2  h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16;   h = h & (HashSize - 1);
//#define HASH_FLOAT3  h ^= h >> 33; h *= 0xff51afd7ed558ccd; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53; h ^= h >> 33; h = h % HashSize;
//#define HASH_FLOAT4  h ^= h >> 33; h *= 0xff51afd7ed558ccd; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53; h ^= h >> 33; h = h & (HashSize - 1);

//------------------
// WHEN MULT SIZE *4
//28000000 entries had 1018080 collisions   22388609 unique
//Elapsed time 5.043842 seconds.
//--------------------------------------------------
// WHEN MULT SIZE *2
//28000000 entries had 2311199 collisions   22388609 unique
//Elapsed time 4.661492 seconds.

//#define HASH_FLOAT1   h = h % HashSize;
//#define HASH_FLOAT2  h ^= h >> 3; h = h & (HashSize - 1);
//#define HASH_FLOAT1  h ^= (h >> 20) ^ (h >> 12) ^ (h >> 7) ^ (h >> 4); h = h % HashSize;
// SCORE 5 seconds #define HASH_FLOAT1  h ^= (h >> 23) ^ (h << 32); h = h % HashSize;
#define HASH_FLOAT1  h ^= (h >> 20); h = h % HashSize;
//#define HASH_FLOAT2  h ^= (h >> 23); h = h & (HashSize - 1);
// -- NOT BAD 11 #define HASH_FLOAT2  h ^= (h >> 23); h = h & (HashSize - 1);
// -- NOT BAD 10.9 #define HASH_FLOAT2  h ^= (h >> 16); h = h & (HashSize - 1);
#define HASH_FLOAT2  h ^= (h >> 20); h = h & (HashSize - 1);
#define HASH_FLOAT3  h ^= h >> 32; h = h % HashSize;

//------------------------------------------------------------------------------
// hash for 64 bit float
//#define HASH_FLOAT4  h ^= h >> 32; h = h & (HashSize - 1);
#define HASH_FLOAT4  h= fasthash64_8(h) & (HashSize-1);
//#define HASH_FLOAT4  h ^= (h >> 44) ^ (h >> 32) ^ (h >> 17) ^ (h >> 4); h = h & (HashSize - 1);



//-----------------------------------------------
// stores the index of the first location
//
template<typename T, typename U>
void CHashLinear<T,U>::MakeHashLocation(
   INT64 arraySize,
   T*    pHashList,
   INT64 hintSize) {

   if (hintSize == 0) {
      hintSize = arraySize;
   }

   AllocMemory(hintSize, sizeof(HashLocation), 0, FALSE);
   NumUnique = 0;

   LOGGING("MakeHashLocation: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

   HashLocation* pLocation = (HashLocation*)pHashTableAny;
   UINT64*     pBitFields = this->pBitFields;

   if (!pLocation || !pBitFields) {
      return;
   }

   LOGGING("IsMember index size is %zu    output size is %zu  HashSize is %llu    hashmode %d\n", sizeof(U), sizeof(T), HashSize, (int)HashMode);

   //for (int i = 0; i < ((HashSize + 63) / 64); i++) {
   //   printf("%llu |", pBitFields[i]);
   //}


   if (sizeof(T) <= 2) {
      for (U i = 0; i < arraySize; i++) {
         // perfect hash
         T item = pHashList[i];
         UINT64 hash = item;
         INTERNAL_SET_LOCATION;
         //InternalSetLocation(i, pLocation, item, item);
      }
   }
   else {
      if (sizeof(T) == 4) {
         if (HashMode == HASH_MODE_PRIME) {
            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT1;
               UINT64 hash = h;
               INTERNAL_SET_LOCATION;
               //InternalSetLocation(i, pLocation, item, h);
            }
         }
         else {

            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT32;
               UINT64 hash = h;
               INTERNAL_SET_LOCATION;
               //InternalSetLocation(i, pLocation, item, h);
            }
         }
      } else
      if (sizeof(T) == 8) {
         if (HashMode == HASH_MODE_PRIME) {
            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT1;
               UINT64 hash = h;
               INTERNAL_SET_LOCATION;
               //InternalSetLocation(i, pLocation, item, h);
            }
         }
         else {

            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT64;
               UINT64 hash = h;
               INTERNAL_SET_LOCATION;
               //InternalSetLocation(i, pLocation, item, h);
            }
         }
      }
      else {
         printf("!!! MakeHashLocation -- hash item size is not 1,2,4, or 8!  %zu\n", sizeof(T));
      }

   }
   LOGGING("%llu entries had %llu collisions   %llu unique\n", arraySize, NumCollisions, NumUnique);

   LOGGING("IsMember index size is %zu    output size is %zu  HashSize is %llu\n", sizeof(U), sizeof(T), HashSize);

   //for (int i = 0; i < ((HashSize + 63) / 64); i++) {
   //   printf("%llu |", pBitFields[i]);
   //}
   //printf("\n");

   //printf("%llu entries had %llu collisions   %llu unique\n", arraySize, NumCollisions, NumUnique);

}





//-----------------------------------------------
// outputs boolean array
// outputs location array
//
template<typename T, typename U>
INT64 CHashLinear<T, U>::IsMemberCategorical(
   INT64 arraySize,
   T*    pHashList,
   U*    pLocationOutput) {

   HashLocation* pLocation = (HashLocation*)pHashTableAny;
   INT64 missed = 0;
   if (sizeof(T) <= 2) {
      // perfect hash
      for (U i = 0; i < arraySize; i++) {
         T item = pHashList[i];
         InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, item, &missed);
      }
   }
   else {

      if (sizeof(T) == 4) {

         if (HashMode == HASH_MODE_PRIME) {
            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT1;
               InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
            }
         }
         else {

            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT32;
               InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
            }
         }
      } else
      if (sizeof(T) == 8) {

         if (HashMode == HASH_MODE_PRIME) {
            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT1;
               InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
            }
         }
         else {

            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT64;
               InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
            }
         }
      }
      else {
         printf("!!IsMemberCategorical hash size error! %zu\n", sizeof(T));
      }
   }
   return missed;
}



#define INNER_GET_LOCATION_PERFECT       \
   if (pBitFields[hash >> 6] & (1LL << (hash & 63))) { \
      pLocationOutput[i] = pLocation[hash].Location; \
      pBooleanOutput[i] = 1; \
   } \
   else { \
      /* Not found */ \
      pLocationOutput[i] = BAD_INDEX; \
      pBooleanOutput[i] = 0; \
   }



#define INNER_GET_LOCATION      \
while (1) {             \
   if (pBitFields[hash >> 6] & (1LL << (hash & 63))) { \
      /* Check if we have a match from before*/ \
      if (pLocation[hash].value == item) {      \
         /* return the first location */ \
         pLocationOutput[i] = pLocation[hash].Location; \
         pBooleanOutput[i] = 1; \
         break; \
      } \
      /* Linear goes to next position */ \
      if (++hash >= HashSize) { \
         hash = 0; \
      } \
      continue; \
   } \
   else { \
      /* Not found */ \
      pLocationOutput[i] = BAD_INDEX; \
      pBooleanOutput[i] = 0; \
      break; \
   } \
}


//-----------------------------------------------
// T is the input type byte/float32/float64/int8/uint8/int16/uint16/int32/...
// U is the index output type (int8/16/32/64)
//
// outputs boolean array
// outputs location array
//
template<typename T, typename U>
void IsMember(
   void* pHashLinearVoid,
   INT64 arraySize,
   void* pHashListT,
   INT8* pBooleanOutput,
   void* pLocationOutputU) {


   struct HashLocation
   {
      T        value;
      U        Location;
   };

   CHashLinear<T, U>* pHashLinear = (CHashLinear<T, U>*)pHashLinearVoid;
   UINT64 HashSize = pHashLinear->HashSize;

   HashLocation* pLocation = (HashLocation*)pHashLinear->pHashTableAny;
   U* pLocationOutput = (U*)pLocationOutputU;
   T* pHashList = (T*)pHashListT;

   // make local reference on stack
   UINT64* pBitFields = pHashLinear->pBitFields;

   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   LOGGING("IsMember arraySize %lld   get index size is %zu    output size is %zu  HashSize is %llu\n", arraySize, sizeof(U), sizeof(T), HashSize);

   //for (int i = 0; i < ((HashSize+63) / 64); i++) {
   //   printf("%llu |", pBitFields[i]);
   //}

   //printf("\n");

   if (sizeof(T) <= 2) {
      LOGGING("Perfect hash\n");
      // perfect hash
      for (INT64 i = 0; i < arraySize; i++) {
         T item = pHashList[i];
         UINT64 hash = (UINT64)item;
         INNER_GET_LOCATION_PERFECT;
      }
   }
   else {
      HASH_MODE   HashMode = pHashLinear->HashMode;

      if (sizeof(T) == 4) {
         if (HashMode == HASH_MODE_PRIME) {
            for (INT64 i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT1;
               UINT64 hash = (UINT64)h;
               INNER_GET_LOCATION;
               //InternalGetLocation(i, pLocation, pBooleanOutput, pLocationOutput, item, h);
            }
         }
         else {

            for (INT64 i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT32;
               UINT64 hash = h;
               INNER_GET_LOCATION;
               //InternalGetLocation(i, pLocation, pBooleanOutput, pLocationOutput, item, h);
            }
         }
      }
      else
      if (sizeof(T) == 8) {
         if (HashMode == HASH_MODE_PRIME) {
            for (INT64 i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT1;
               UINT64 hash = (UINT64)h;
               INNER_GET_LOCATION;
               //InternalGetLocation(i, pLocation, pBooleanOutput, pLocationOutput, item, h);
            }
         }
         else {

            for (INT64 i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               HASH_INT64;
               UINT64 hash = h;
               INNER_GET_LOCATION;
            }
         }
      }
      else {
         printf("!!! IsMember hash item size not valid %zu\n", sizeof(T));
      }
   }

}



//-----------------------------------------
// bits 32-51 appear to be sweet spot
void CalculateHashBits64(
   INT64 arraySize,
   UINT64* pHashList) {

   INT64 position[64];
   for (int i = 0; i < 64; i++) position[i] = 0;

   while (arraySize--) {
      for (int i = 0; i < 64; i++) {
         // check if bit is set
         if ((1LL << i) & pHashList[arraySize])
            position[i]++;
      }
   }

   for (int i = 0; i < 64; i++) {
      printf("%d  %llu\n", i, position[i]);
   }
}

//-----------------------------------------------
// bits 3-22 appear to be sweet spot
void CalculateHashBits32(
   INT64 arraySize,
   UINT32* pHashList) {

   INT64 position[32];
   for (int i = 0; i < 32; i++) position[i] = 0;

   while (arraySize--) {
      for (int i = 0; i < 32; i++) {
         // check if bit is set
         if ((1LL << i) & pHashList[arraySize])
            position[i]++;
      }
   }

   for (int i = 0; i < 32; i++) {
      printf("%d  %llu\n", i, position[i]);
   }
}

//-----------------------------------------------
// stores the index of the first location
//
template<typename T, typename U>
void CHashLinear<T, U>::MakeHashLocationFloat(
   INT64 arraySize,
   T*    pHashList,
   INT64 hintSize ) {

   if (hintSize == 0) {
      hintSize = arraySize;
   }

   AllocMemory(hintSize, sizeof(HashLocation), 0, TRUE);
   NumUnique = 0;

   LOGGING("MakeHashLocationFloat: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

   HashLocation* pLocation = (HashLocation*)pHashTableAny;
   UINT64*     pBitFields = this->pBitFields;

   if (!pLocation || !pBitFields) {
      return;
   }

   if (sizeof(T) == 8) {
      //printf("**double %llu\n", HashSize);
      //CalculateHashBits64(arraySize, (UINT64*)pHashList);

      if (HashMode == HASH_MODE_PRIME) {

         for (U i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            // NAN CHECK FIRST
            if (item == item) {
               UINT64* pHashList2 = (UINT64*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT3;
               UINT64 hash = h;
               INTERNAL_SET_LOCATION;
               //InternalSetLocation(i, pLocation, item, h);
            }
         }
      }
      else {
         for (U i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            // NAN CHECK FIRST
            if (item == item) {
               UINT64* pHashList2 = (UINT64*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT4;
               UINT64 hash = h;
               INTERNAL_SET_LOCATION;
               //InternalSetLocation(i, pLocation, item, h);
            }
         }
      }
   }
   if (sizeof(T) == 4) {
      //printf("**single  %llu\n",HashSize);
      //CalculateHashBits32(arraySize,(UINT32*)pHashList);

      if (HashMode == HASH_MODE_PRIME) {

         for (U i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            // NAN CHECK FIRST
            if (item == item) {
               UINT32* pHashList2 = (UINT32*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT1;
               UINT64 hash = h;
               INTERNAL_SET_LOCATION;
               //InternalSetLocation(i, pLocation, item, h);
            }
         }
      }
      else {
         for (U i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            // NAN CHECK FIRST
            if (item == item) {
               UINT32* pHashList2 = (UINT32*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT2;
               UINT64 hash = h;
               INTERNAL_SET_LOCATION;
               //InternalSetLocation(i, pLocation, item, h);
            }
         }
      }
   }
   LOGGING("%llu entries had %llu collisions   %llu unique\n", arraySize, NumCollisions, NumUnique);
   //printf("%llu entries had %llu collisions   %llu unique\n", arraySize, NumCollisions, NumUnique);

}



//-----------------------------------------------
// outputs boolean array
// outputs location array
//
template<typename T, typename U>
void IsMemberFloat(
   void* pHashLinearVoid,
   INT64 arraySize,
   void* pHashListT,
   INT8* pBooleanOutput,
   void* pLocationOutputU) {

   LOGGING("IsMemberFloat: arraySize %lld \n", arraySize);


   struct HashLocation
   {
      T        value;
      U        Location;
   };

   CHashLinear<T, U>* pHashLinear = (CHashLinear<T, U>*)pHashLinearVoid;
   UINT64 HashSize = pHashLinear->HashSize;
   int HashMode = pHashLinear->HashMode;


   HashLocation* pLocation = (HashLocation*)pHashLinear->pHashTableAny;
   U* pLocationOutput = (U*)pLocationOutputU;
   T* pHashList = (T*)pHashListT;

   // make local reference on stack
   UINT64* pBitFields = pHashLinear->pBitFields;

   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   if (sizeof(T) == 8) {

      if (HashMode == HASH_MODE_PRIME) {
         for (INT64 i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            if (item == item) {
               UINT64* pHashList2 = (UINT64*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT3;
               UINT64 hash = (UINT64)h;
               INNER_GET_LOCATION;
               //InternalGetLocation(i, pLocation, pBooleanOutput, pLocationOutput, item, h);
            }
            else {
               // Not found
               pLocationOutput[i] = BAD_INDEX;
               pBooleanOutput[i] = 0;
            }
         }
      }
      else {

         for (INT64 i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            if (item == item) {
               UINT64* pHashList2 = (UINT64*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT4;
               UINT64 hash = (UINT64)h;
               INNER_GET_LOCATION;
               //InternalGetLocation(i, pLocation, pBooleanOutput, pLocationOutput, item, h);
            }
            else {

               // Not found
               pLocationOutput[i] = BAD_INDEX;
               pBooleanOutput[i] = 0;
            }
         }
      }
   } else
   if (sizeof(T) == 4) {

      if (HashMode == HASH_MODE_PRIME) {
         for (INT64 i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            if (item == item) {
               UINT32* pHashList2 = (UINT32*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT1;
               UINT64 hash = (UINT64)h;
               INNER_GET_LOCATION;
               //InternalGetLocation(i, pLocation, pBooleanOutput, pLocationOutput, item, h);
            }
            else {
               // Not found
               pLocationOutput[i] = BAD_INDEX;
               pBooleanOutput[i] = 0;
            }
         }
      }
      else {

         for (INT64 i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            if (item == item) {
               UINT32* pHashList2 = (UINT32*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT2;
               UINT64 hash = (UINT64)h;
               INNER_GET_LOCATION;
               //InternalGetLocation(i, pLocation, pBooleanOutput, pLocationOutput, item, h);
            }
            else {

               // Not found
               pLocationOutput[i] = BAD_INDEX;
               pBooleanOutput[i] = 0;
            }
         }
      }
   }

}




//-----------------------------------------------
// outputs location array + 1
//
template<typename T, typename U>
INT64 CHashLinear<T, U>::IsMemberFloatCategorical(
   INT64 arraySize,
   T*    pHashList,
   U*    pLocationOutput) {

   HashLocation* pLocation = (HashLocation*)pHashTableAny;
   INT64 missed = 0;

   // BUG BUG -- LONG DOUBLE on Linux size 16

   if (sizeof(T) == 8) {

      if (HashMode == HASH_MODE_PRIME) {
         for (U i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            if (item == item) {
               UINT64* pHashList2 = (UINT64*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT3;
               InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
            }
            else {
               // Not found
               pLocationOutput[i] = 0;
               missed =1;
            }
         }
      }
      else {

         for (U i = 0; i < arraySize; i++) {
            T item = pHashList[i];
            if (item == item) {
               UINT64* pHashList2 = (UINT64*)pHashList;
               UINT64 h = pHashList2[i];
               HASH_FLOAT4;
               InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
            }
            else {

               // Not found
               pLocationOutput[i] = 0;
               missed =1;
            }
         }
      }
   }
   else
      if (sizeof(T) == 4) {

         if (HashMode == HASH_MODE_PRIME) {
            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               if (item == item) {
                  UINT32* pHashList2 = (UINT32*)pHashList;
                  UINT64 h = pHashList2[i];
                  HASH_FLOAT1;
                  InternalGetLocationCategorical(i, pLocation,  pLocationOutput, item, h, &missed);
               }
               else {
                  // Not found
                  pLocationOutput[i] = 0;
                  missed =1;
               }
            }
         }
         else {

            for (U i = 0; i < arraySize; i++) {
               T item = pHashList[i];
               if (item == item) {
                  UINT32* pHashList2 = (UINT32*)pHashList;
                  UINT64 h = pHashList2[i];
                  HASH_FLOAT2;
                  InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
               }
               else {

                  // Not found
                  pLocationOutput[i] = 0;
                  missed =1;
               }
            }
         }
      }


   return missed;
}


//===========================================================================
//===========================================================================

#define GROUPBY_INNER_LOOP_PERFECT \
UINT64 hash = (UINT64)item & (HashSize -1); \
SingleKeyEntry* pKey = &pLocation[hash]; \
   if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63))) { \
       pIndexArray[i] = pKey->UniqueKey;            \
   }                                            \
   else {                                          \
      pBitFieldsX[hash >> 6] |= (1LL << (hash & 63));        \
      pFirstArray[numUnique] = i;                         \
      numUnique++;                                            \
      pKey->UniqueKey = numUnique;                         \
      pIndexArray[i] = numUnique;                     \
   }


#define GROUPBY_INNER_LOOP \
UINT64 hash = h; \
while (1) { \
   if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63))) { \
      /* Check if we have a match from before */     \
      if (item == pLocation[hash].value) {           \
         /* 2nd+ Match */                      \
         /* Make the key the same */         \
         pIndexArray[i] = pLocation[hash].UniqueKey;            \
         break;                        \
      }                                         \
      ++hash;                              \
      if (hash >= HashSize) {                           \
         hash = 0;                               \
      }                               \
   }                                            \
   else {                                          \
      /* Failed to find hash */                       \
      /* Must be first item */                           \
      pBitFieldsX[hash >> 6] |= (1LL << (hash & 63));        \
      /* Check if we have a match from before */        \
      pLocation[hash].value = item;                         \
      pFirstArray[numUnique] = i;                         \
      /*base index of 1 so increment first */              \
      numUnique++;                                            \
      pLocation[hash].UniqueKey = numUnique;                         \
      pIndexArray[i] = numUnique;                     \
      break;                                 \
   }                                \
}

typedef PyArrayObject*(COPY_TO_SMALLER_ARRAY)(void* pFirstArrayIndex, INT64 numUnique, INT64 totalRows);
//-----------------------------------------------------------------------------------------
// Returns AN ALLOCATED numpy array
// if firstArray is NULL, it will allocate
template<typename INDEX_TYPE>
PyArrayObject* CopyToSmallerArray(void* pFirstArrayIndex, INT64 numUnique, INT64 totalRows, PyArrayObject* firstArray = NULL) {

   // check for out of memory
   if (!pFirstArrayIndex) {
      Py_IncRef(Py_None);
      // caller should check, this is additional safety
      return (PyArrayObject*)Py_None;
   }

   INDEX_TYPE* pFirstArray = (INDEX_TYPE*)pFirstArrayIndex;

   // Once we know the number of unique, we can allocate the smaller array
   if (firstArray == NULL) {

      switch (sizeof(INDEX_TYPE)) {
      case 1:
         firstArray = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT8);
         break;
      case 2:
         firstArray = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT16);
         break;
      case 4:
         firstArray = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT32);
         break;
      case 8:
         firstArray = AllocateNumpyArray(1, (npy_intp*)&numUnique, NPY_INT64);
         break;
      default:
         printf("!!!Internal error CopyToSmallerArray\n");
         break;
      }
   }
   CHECK_MEMORY_ERROR(firstArray);

   LOGGING("after alloc numpy copy to smaller %p  %lld %lld %lld\n", pFirstArray, numUnique, totalRows, numUnique * sizeof(INDEX_TYPE));

   if (firstArray != NULL && numUnique <= totalRows) {

      INDEX_TYPE* pFirstArrayDest = (INDEX_TYPE*)PyArray_BYTES(firstArray);
      memcpy(pFirstArrayDest, pFirstArray, numUnique * sizeof(INDEX_TYPE));
   }
   else {
      printf("!!! error allocating copytosmallerarray %lld %lld\n", numUnique, totalRows);
   }

   return firstArray;
}

//------------------------------------------------------------------
// NOTE pFirstArray is allocated and must be deallocated and reduced later
//    // return to caller the first array that we reduced
// *pFirstArrayObject = CopyToSmallerArray<U>(pFirstArray, numUnique, totalRows);

template<typename T, typename U>
UINT64 CHashLinear<T, U>::GroupByFloat(
   INT64 totalRows,
   INT64 totalItemSize,
   T* pInput,
   int coreType, // -1 when unknown  indicates only one array
                 // Return values
   U* pIndexArray,
   U* &pFirstArray,
   HASH_MODE hashMode,
   INT64 hintSize,
   bool* pBoolFilter) {

   LOGGING("GroupByFloat: hintSize %lld   HashSize %llu  totalRows %lld\n", hintSize, HashSize, totalRows);

   U numUnique = 0;
   SingleKeyEntry* pLocation = (SingleKeyEntry*)pHashTableAny;

   // make local reference on stack
   UINT64* pBitFieldsX = pBitFields;

   switch (sizeof(T)) {
   case 4:
      if (pBoolFilter == NULL) {
         for (U i = 0; i < totalRows; i++) {
            T item = pInput[i];
            UINT64 h = ((UINT32*)pInput)[i];
            HASH_FLOAT2;
            GROUPBY_INNER_LOOP;
         }
      }
      else
      for (U i = 0; i < totalRows; i++) {
         // check to see if in filter
         if (pBoolFilter[i]) {
            T item = pInput[i];
            UINT64 h = ((UINT32*)pInput)[i];
            HASH_FLOAT2;
            GROUPBY_INNER_LOOP;
         }
         else {
            // not in filter set to zero bin
            pIndexArray[i] = 0;
         }
      }
      break;
   case 8:
      if (pBoolFilter == NULL) {
         for (U i = 0; i < totalRows; i++) {
            T item = pInput[i];
            UINT64 h = ((UINT64*)pInput)[i];
            HASH_FLOAT4;
            GROUPBY_INNER_LOOP;
         }
      }
      else
         for (U i = 0; i < totalRows; i++) {
            // check to see if in filter
            if (pBoolFilter[i]) {
               T item = pInput[i];
               UINT64 h = ((UINT64*)pInput)[i];
               HASH_FLOAT4;
               GROUPBY_INNER_LOOP;
            }
            else {
               // not in filter set to zero bin
               pIndexArray[i] = 0;
            }
         }
      break;
   }

   LOGGING("GroupByFloat end! %I64d\n", (INT64)numUnique);

   return numUnique;
}



//------------------------------------------------------------------
// Returns pFirstArray
template<typename T, typename U>
UINT64 CHashLinear<T, U>::GroupByItemSize(
   INT64 totalRows,
   INT64 totalItemSize,
   T* pInput,
   int coreType, // -1 when unknown  indicates only one array
                 // Return values
   U* pIndexArray,
   U* &pFirstArray,
   HASH_MODE hashMode,
   INT64 hintSize,
   bool* pBoolFilter) {

   LOGGING("GroupByItem: hintSize %lld   HashSize %llu  totalRows %lld   sizeofT:%lld   sizeofU:%lld\n", hintSize, HashSize, totalRows, sizeof(T), sizeof(U));

   U numUnique = 0;
   SingleKeyEntry* pLocation = (SingleKeyEntry*)pHashTableAny;

   // make local reference on stack
   UINT64* pBitFieldsX = pBitFields;

   switch (sizeof(T)) {
   case 1:
      // TODO: Specially handle bools here -- they're 1-byte but logically have only two buckets (zero/nonzero).
      //if (coreType == NPY_BOOL)
      //{
      //   //
      //}

      if (pBoolFilter == NULL) {
         for (U i = 0; i < totalRows; i++) {
            T item = pInput[i];
            GROUPBY_INNER_LOOP_PERFECT;
         }
      }
      else
      for (U i = 0; i < totalRows; i++) {
         // check to see if in filter
         if (pBoolFilter[i]) {
            T item = pInput[i];
            GROUPBY_INNER_LOOP_PERFECT;
         }
         else {
            // not in filter set to zero bin
            pIndexArray[i] = 0;
         }
      }
      break;
   case 2:
      if (pBoolFilter == NULL) {
         for (U i = 0; i < totalRows; i++) {
            T item = pInput[i];
            GROUPBY_INNER_LOOP_PERFECT;
         }
      }
      else
         for (U i = 0; i < totalRows; i++) {
            // check to see if in filter
            if (pBoolFilter[i]) {
               T item = pInput[i];
               GROUPBY_INNER_LOOP_PERFECT;
            }
            else {
               // not in filter set to zero bin
               pIndexArray[i] = 0;
            }
         }
      break;
   case 4:
      if (pBoolFilter == NULL) {
         for (U i = 0; i < totalRows; i++) {
            T item = pInput[i];
            HASH_INT32;
            GROUPBY_INNER_LOOP;
         }
      }
      else
         for (U i = 0; i < totalRows; i++) {
            // check to see if in filter
            if (pBoolFilter[i]) {
               T item = pInput[i];
               HASH_INT32;
               GROUPBY_INNER_LOOP;
            }
            else {
               // not in filter set to zero bin
               pIndexArray[i] = 0;
            }
         }
      break;
   case 8:
      if (pBoolFilter == NULL) {
         for (U i = 0; i < totalRows; i++) {
            T item = pInput[i];
            HASH_INT64;
            GROUPBY_INNER_LOOP;
         }
      }
      else
         for (U i = 0; i < totalRows; i++) {
            // check to see if in filter
            if (pBoolFilter[i]) {
               T item = pInput[i];
               HASH_INT64;
               GROUPBY_INNER_LOOP;
            }
            else {
               // not in filter set to zero bin
               pIndexArray[i] = 0;
            }
         }
      break;
   case 16:
      // TO BE WORKED ON...
      if (pBoolFilter == NULL) {
         for (U i = 0; i < totalRows; i++) {
            T item = pInput[i];
            HASH_INT128;
            GROUPBY_INNER_LOOP;
         }
      }
      else
         for (U i = 0; i < totalRows; i++) {
            // check to see if in filter
            if (pBoolFilter[i]) {
               T item = pInput[i];
               HASH_INT128;
               GROUPBY_INNER_LOOP;
            }
            else {
               // not in filter set to zero bin
               pIndexArray[i] = 0;
            }
         }
      break;
   }

   //printf("GroupByItem end! %d\n", numUnique);
   // return to caller the first array that we reduced
   //*pFirstArrayObject = CopyToSmallerArray<U>(pFirstArray, numUnique, totalRows);

   return numUnique;
}



//-----------------------------------------------
// stores the index of the first location
// hintSize can be passed if # unique items is KNOWN or GUESSTIMATED in ADVNACE
// hintSize can be 0 which will default to totalRows
// pBoolFilter can be NULL
template<typename T, typename U>
UINT64 CHashLinear<T, U>::GroupBy(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput,

   int coreType, // -1 when unknown  indicates only one array

                 // Return values
   U* pIndexArray,
   U* &pFirstArray,

   HASH_MODE hashMode,
   INT64 hintSize,
   bool* pBoolFilter) {

   LOGGING("GroupBy: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

   U numUnique = 0;
   U numCollisions = 0;
   MultiKeyEntry* pLocation = (MultiKeyEntry*)pHashTableAny;

   // make local reference on stack
   UINT64* pBitFieldsX = pBitFields;
   if (pBoolFilter == NULL) {
      // make local reference on stack

      for (U i = 0; i < totalRows; i++) {

         const char* pMatch = pInput + (totalItemSize*i);
         UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);

         // Use and mask to strip off high bits
         hash = hash & (HashSize - 1);
         while (1) {
            if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63))) {
               // Check if we have a match from before
               U Last = pLocation[hash].Last;
               const char* pMatch2 = pInput + (totalItemSize*Last);
               int mresult;
               MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
               if (!mresult) {
                  // 2nd+ Match
                  // Make the key the same
                  pIndexArray[i] = pIndexArray[Last];
                  break;
               }

               // Linear goes to next position
               ++hash;
               if (hash >= HashSize) {
                  hash = 0;
               }
            }
            else {
               // Failed to find hash
               // Must be first item
               pBitFieldsX[hash >> 6] |= (1LL << (hash & 63));
               pLocation[hash].Last = i;
               pFirstArray[numUnique] = i;

               // base index of 1 so increment first
               numUnique++;
               pIndexArray[i] = (U)numUnique;
               break;
            }
         }
      }
   }
   else {

      for (U i = 0; i < totalRows; i++) {

         // check to see if in filter
         if (pBoolFilter[i]) {

            const char* pMatch = pInput + (totalItemSize*i);
            UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);
            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            while (1) {
               if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63))) {
                  // Check if we have a match from before
                  U Last = pLocation[hash].Last;
                  const char* pMatch2 = pInput + (totalItemSize*Last);
                  int mresult;
                  MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                  if (!mresult) {
                     // 2nd+ Match
                     // Make the key the same
                     pIndexArray[i] = pIndexArray[Last];
                     break;
                  }

                  // Linear goes to next position
                  ++hash;
                  if (hash >= HashSize) {
                     hash = 0;
                  }

               }
               else {
                  // Failed to find hash
                  // Must be first item
                  pBitFieldsX[hash >> 6] |= (1LL << (hash & 63));
                  pLocation[hash].Last = i;
                  pFirstArray[numUnique] = i;
                  // base index of 1 so increment first
                  numUnique++;
                  pIndexArray[i] = (U)numUnique;
                  break;
               }
            }
         }
         else {
            // not in filter set to zero bin
            pIndexArray[i] = 0;
         }
      }
   }

   LOGGING("%lld entries   hashSize %llu   %lld unique\n", totalRows, HashSize, (INT64)numUnique);

   // return to caller the first array that we reduced
   //*pFirstArrayObject = CopyToSmallerArray<U>(pFirstArray, numUnique, totalRows);

   return numUnique;
}



//-----------------------------------------------
// stores the index of the first location
// hintSize can be passed if # unique items is KNOWN or GUESSTIMATED in ADVNACE
// hintSize can be 0 which will default to totalRows
// pBoolFilter can be NULL
template<typename T, typename U>
UINT64 CHashLinear<T, U>::GroupBySuper(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput,

   int coreType, // -1 when unknown  indicates only one array

   // Return values
   U* pIndexArray,
   U* pNextArray,
   U* pUniqueArray,
   U* pUniqueCountArray,
   HASH_MODE hashMode,
   INT64 hintSize,
   bool* pBoolFilter) {

   if (hintSize == 0) {
      hintSize = totalRows;
   }
   AllocMemory(hintSize, sizeof(MultiKeyEntrySuper), 0, FALSE);

   LOGGING("GroupBySuper: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

   U numUnique = 0;
   U numCollisions = 0;
   MultiKeyEntrySuper* pLocation = (MultiKeyEntrySuper*)pHashTableAny;

   if (!pLocation || !pBitFields) {
      return 0;
   }

   // pre-fill for invalid bin
   //pUniqueCountArray[0] = 0;
   //pUniqueArray[0] = GB_INVALID_INDEX;

   // make local reference on stack
   UINT64* pBitFieldsX = pBitFields;
   if (pBoolFilter == NULL) {
      // make local reference on stack

      for (U i = 0; i < totalRows; i++) {

         const char* pMatch = pInput + (totalItemSize*i);
         UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);
         //UINT64 hash = mHashOld(pMatch, totalItemSize);
         //UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);

         //printf("%d", hash);
         // Use and mask to strip off high bits
         hash = hash & (HashSize - 1);
         while (1) {
            if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63))) {
               // Check if we have a match from before
               U Last = pLocation[hash].Last;
               const char* pMatch2 = pInput + (totalItemSize*Last);
               int mresult;
               MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
               if (!mresult) {
                  // 2nd+ Match

                  // Make the key the same
                  pIndexArray[i] = pIndexArray[Last];

                  // The next item is unknown
                  pNextArray[i] = GB_INVALID_INDEX;

                  // if we go back to previous, the next item is us
                  pNextArray[Last] = i;

                  //update last item - TJD NOTE: DO NOT think this helps or is nec.
                  pLocation[hash].Last = i;
                  pUniqueCountArray[pLocation[hash].UniqueKey]++;
                  break;
               }


               // This entry is not us so we must have collided
               ++numCollisions;

               if (numCollisions < 0) {
                  NumCollisions = numCollisions;
                  NumUnique = numUnique;

                  printf("!!! error in groupby collisions too high -- trying to match size %lld\n", totalItemSize);
                  printf("%llu entries   hashSize %llu  had %llu collisions   %llu unique\n", totalRows, HashSize, NumCollisions, NumUnique);
                  return NumUnique;
               }

               // Linear goes to next position
               ++hash;
               if (hash >= HashSize) {
                  hash = 0;
               }

            }
            else {
               // Failed to find hash
               // Must be first item
               pBitFieldsX[hash >> 6] |= (1LL << (hash & 63));

               pLocation[hash].Last = i;

               pLocation[hash].UniqueKey = numUnique;
               pUniqueCountArray[numUnique] = 1;
               pUniqueArray[numUnique] = i;

               // base index of 1 so increment first
               numUnique++;

               pIndexArray[i] = (U)numUnique;
               pNextArray[i] = GB_INVALID_INDEX;

               break;
            }
         }
      }
   }
   else {

      U InvalidLast = GB_INVALID_INDEX;

      for (U i = 0; i < totalRows; i++) {

         // check to see if in filter
         if (pBoolFilter[i]) {

            const char* pMatch = pInput + (totalItemSize*i);
            UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);
            //UINT64 hash = mHashOld(pMatch, totalItemSize);
            //UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);

            //printf("%d", hash);
            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            while (1) {
               if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63))) {
                  // Check if we have a match from before
                  U Last = pLocation[hash].Last;
                  const char* pMatch2 = pInput + (totalItemSize*Last);
                  int mresult;
                  MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                  if (!mresult) {
                     // 2nd+ Match

                     // Make the key the same
                     pIndexArray[i] = pIndexArray[Last];

                     // The next item is unknown
                     pNextArray[i] = GB_INVALID_INDEX;

                     // if we go back to previous, the next item is us
                     pNextArray[Last] = i;

                     //update last item
                     pLocation[hash].Last = i;
                     pUniqueCountArray[pLocation[hash].UniqueKey]++;
                     break;
                  }

                  // This entry is not us so we must have collided
                  ++numCollisions;

                  if (numCollisions < 0) {
                     NumCollisions = numCollisions;
                     NumUnique = numUnique;

                     printf("!!! error in groupby collisions too high -- trying to match size %lld\n", totalItemSize);
                     printf("%llu entries   hashSize %llu  had %llu collisions   %llu unique\n", totalRows, HashSize, NumCollisions, NumUnique);
                     return NumUnique;
                  }

                  // Linear goes to next position
                  ++hash;
                  if (hash >= HashSize) {
                     hash = 0;
                  }

               }
               else {
                  // Failed to find hash
                  // Must be first item
                  pBitFieldsX[hash >> 6] |= (1LL << (hash & 63));

                  pLocation[hash].Last = i;

                  pLocation[hash].UniqueKey = numUnique;
                  pUniqueCountArray[numUnique] = 1;
                  pUniqueArray[numUnique] = i;

                  // base index of 1 so increment first
                  numUnique++;
                  pIndexArray[i] = (U)numUnique;
                  pNextArray[i] = GB_INVALID_INDEX;

                  // 0 based key
                  //numUnique++;

                  break;

               }

            }
         }
         else {
            // not in filter set to zero bin
            pIndexArray[i] = 0;
            pNextArray[i] = GB_INVALID_INDEX;

            // First location of invalid bin
            if (InvalidLast != GB_INVALID_INDEX) {
               pNextArray[InvalidLast] = i;
            }
            InvalidLast = i;

         }
      }
   }

   NumCollisions = numCollisions;
   NumUnique = numUnique;

   LOGGING("%llu entries   hashSize %llu  had %llu collisions   %llu unique\n", totalRows, HashSize, NumCollisions, NumUnique);

   return NumUnique;

}



//-----------------------------------------------
// stores the index of the first location
// hintSize can be passed if # unique items is KNOWN or GUESSTIMATED in ADVNACE
// hintSize can be 0 which will default to totalRows
// pBoolFilter option (can be NULL)
template<typename T, typename U>
UINT64 CHashLinear<T, U>::Unique(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput,

   // Return values
   U* pIndexArray,

   // Return count values
   U* pCountArray,

   // inpuys
   HASH_MODE hashMode,
   INT64 hintSize,
   bool* pBoolFilter) {

   if (hintSize == 0) {
      hintSize = totalRows;
   }

   AllocMemory(hintSize, sizeof(UniqueEntry), 0, FALSE);

   LOGGING("Unique: hintSize:%lld   HashSize:%llu  sizeoftypeU:%lld   sizeoftypeT:%lld\n", hintSize, HashSize, sizeof(U), sizeof(T));

   UniqueEntry* pLocation = (UniqueEntry*)pHashTableAny;

   if (!pLocation || !pBitFields) {
      return 0;
   }

   U NumUnique = 0;

   if (pBoolFilter) {

      for (U i = 0; i < totalRows; i++) {

         // Make sure in the filter
         if (pBoolFilter[i]) {
            const char* pMatch = pInput + (totalItemSize*i);
            UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);
            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            while (1) {
               if (IsBitSet(hash)) {
                  // Check if we have a match from before
                  const char* pMatch2 = pLocation[hash].Last;

                  int mresult;
                  MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                  if (!mresult) {
                     // 2nd+ Match
                     pCountArray[pLocation[hash].UniqueKey]++;
                     break;
                  }

                  // This entry is not us so we must have collided
                  //++NumCollisions;

                  // Linear goes to next position
                  if (++hash >= HashSize) {
                     hash = 0;
                  }

               }
               else {
                  // Failed to find hash
                  // Must be first item
                  SetBit(hash);

                  pLocation[hash].Last = pMatch;
                  pLocation[hash].UniqueKey = NumUnique;
                  pIndexArray[NumUnique] = i;

                  // First count
                  pCountArray[NumUnique] = 1;
                  NumUnique++;
                  break;
               }
            }
         }
      }


   }
   else {
      for (U i = 0; i < totalRows; i++) {
         const char* pMatch = pInput + (totalItemSize*i);
         UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);
         // Use and mask to strip off high bits
         hash = hash & (HashSize - 1);
         while (1) {
            if (IsBitSet(hash)) {
               // Check if we have a match from before
               const char* pMatch2 = pLocation[hash].Last;

               int mresult;
               MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
               if (!mresult) {
                  // 2nd+ Match
                  pCountArray[pLocation[hash].UniqueKey]++;
                  break;
               }

               // This entry is not us so we must have collided
               //++NumCollisions;

               // Linear goes to next position
               if (++hash >= HashSize) {
                  hash = 0;
               }

            }
            else {
               // Failed to find hash
               // Must be first item
               SetBit(hash);

               pLocation[hash].Last = pMatch;
               pLocation[hash].UniqueKey = NumUnique;
               pIndexArray[NumUnique] = i;

               // First count
               pCountArray[NumUnique] = 1;
               NumUnique++;
               break;
            }
         }

      }
   }

   LOGGING("%llu entries   hashSize %lld    %lld unique\n", totalRows, (INT64)HashSize, (INT64)NumUnique);

   return NumUnique;

}


//-----------------------------------------------
// Remembers previous values
// Set hintSize < 0 if second pass so it will not allocate
template<typename T, typename U>
void CHashLinear<T, U>::MultiKeyRolling(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput,

   // Return values
   U* pIndexArray,
   U* pRunningCountArray,
   HASH_MODE hashMode,
   INT64 hintSize) {   // pass in -value to indicate reusing

   if (totalItemSize > 16) { //sizeof(MultiKeyEntryRolling.Key))
      printf("!!!rolling key is too wide %lld\n", totalItemSize);
      return;
   }

   if (hintSize >= 0) {
      if (hintSize == 0) {
         hintSize = totalRows;
      }
      AllocMemory(hintSize, sizeof(MultiKeyEntryRolling), 0, FALSE);
      NumUnique = 0;
   }

   LOGGING("MakeHashLocationMultiKey: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

   MultiKeyEntryRolling* pLocation = (MultiKeyEntryRolling*)pHashTableAny;

   if (!pLocation || !pBitFields) {
      return;
   }

   for (U i = 0; i < totalRows; i++) {
      const char* pMatch = pInput + (totalItemSize*i);
      UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);

      // Use and mask to strip off high bits
      hash = hash & (HashSize - 1);
      while (1) {
         if (IsBitSet(hash)) {
            // Check if we have a match from before
            int mresult;
            MEMCMP_NEW(mresult, pMatch, pLocation[hash].Key, totalItemSize);
            if (!mresult) {
               // 2nd+ Match
               // 2nd+ Match
               pIndexArray[i] = pLocation[hash].Last;
               pRunningCountArray[i] = ++pLocation[hash].RunningCount;
               break;
            }

            // This entry is not us so we must have collided
            ++NumCollisions;

            // Bail on too many collisions (could return FALSE)
            if ((INT64)NumCollisions > hintSize)
               break;

            // Linear goes to next position
            if (++hash >= HashSize) {
               hash = 0;
            }

         }
         else {
            // Failed to find hash
            // Must be first item
            SetBit(hash);

            memcpy(pLocation[hash].Key, pMatch, totalItemSize);
            pLocation[hash].Last = (U)NumUnique;
            pLocation[hash].RunningCount = 1;
            pIndexArray[i] = (U)NumUnique;
            pRunningCountArray[i] = 1;
            NumUnique++;
            break;
         }

      }
   }
}


//-----------------------------------------------
// stores the index of the first location
// pBoolFilter can be NULL
template<typename T, typename U>
void CHashLinear<T, U>::MakeHashLocationMultiKey(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput,

   // Return values
   U* pIndexArray,
   U* pRunningCountArray,
   U* pPrevArray,
   U* pNextArray,
   U* pFirstArray,
   HASH_MODE hashMode,
   INT64 hintSize,
   bool* pBoolFilter) {

   //INT64 arraySize,
   //const char* pHashList,
   //INT32 strWidth) {

   if (hintSize == 0) {
      hintSize = totalRows;
   }
   AllocMemory(hintSize, sizeof(MultiKeyEntry), 0, FALSE);
   NumUnique = 0;

   LOGGING("MakeHashLocationMultiKey: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

   MultiKeyEntry* pLocation = (MultiKeyEntry*)pHashTableAny;

   if (!pLocation || !pBitFields) {
      return;
   }


   for (U i = 0; i < totalRows; i++) {
      const char* pMatch = pInput + (totalItemSize*i);
      UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);
      //UINT64 hash = mHashOld(pMatch, totalItemSize);
      //UINT64 hash = DEFAULT_HASH64(pMatch, totalItemSize);

      //printf("%d", hash);
      // Use and mask to strip off high bits
      hash = hash & (HashSize - 1);
      while (1) {
         if (IsBitSet(hash)) {
            // Check if we have a match from before
            U Last = pLocation[hash].Last;
            const char* pMatch2 = pInput + (totalItemSize*Last);
            int mresult;
            MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
            if (!mresult) {
               // 2nd+ Match
               pIndexArray[i] = pIndexArray[Last];
               pFirstArray[i] = pFirstArray[Last];
               pRunningCountArray[i] = pRunningCountArray[Last]+1;
               pPrevArray[i] = Last;
               pNextArray[i] = GB_INVALID_INDEX;
               pNextArray[Last] = i;
               pLocation[hash].Last = i;
               break;
            }


            // This entry is not us so we must have collided
            ++NumCollisions;

            // Linear goes to next position
            if (++hash >= HashSize) {
               hash = 0;
            }

         }
         else {
            // Failed to find hash
            // Must be first item
            SetBit(hash);
            NumUnique++;

            pLocation[hash].Last = i;
            pFirstArray[i] = i;
            pIndexArray[i] = (U)NumUnique;
            pRunningCountArray[i] = 1;
            pPrevArray[i] = GB_INVALID_INDEX;
            pNextArray[i] = GB_INVALID_INDEX;
            break;

         }

      }
   }

   LOGGING("%llu entries   hashSize %llu  had %llu collisions   %llu unique\n", totalRows, HashSize, NumCollisions, NumUnique);

}



//-----------------------------------------------
// stores the index of the first location
//
template<typename T, typename U>
void CHashLinear<T, U>::MakeHashLocationString(
   INT64 arraySize,
   const char* pHashList,
   INT64 strWidth,
   INT64 hintSize,
   BOOL  isUnicode) {

   if (hintSize == 0) {
      hintSize = arraySize;
   }

   AllocMemory(hintSize, sizeof(HashLocation), 0, FALSE);
   NumUnique = 0;

   LOGGING("MakeHashLocationString: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

   HashLocation* pLocation = (HashLocation*)pHashTableAny;

   if (!pLocation || !pBitFields) {
      return;
   }

   if (isUnicode) {
      for (INT64 i = 0; i < arraySize; i++) {
         HASH_UNICODE()
         // Use and mask to strip off high bits
         hash = hash & (HashSize - 1);
         InternalSetLocationUnicode((U)i, pLocation, strStart, strWidth, hash);
      }
   }
   else {
      for (INT64 i = 0; i < arraySize; i++) {
         HASH_STRING()
         // Use and mask to strip off high bits
         hash = hash & (HashSize - 1);
         InternalSetLocationString((U)i, pLocation, strStart, strWidth, hash);
      }
   }
   LOGGING("%llu entries had %llu collisions   %llu unique\n", arraySize, NumCollisions, NumUnique);

}

//-----------------------------------------------
// stores the index of the first location
// remove the forceline to make debugging easier
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalSetLocationString(
   U  i,
   HashLocation* pLocation,
   const char* strValue,
   INT64 strWidth,
   UINT64 hash) {

   //printf("**set %llu  width: %d  string: %s\n", hash, strWidth, strValue);

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      if (STRING_MATCH((const char*)pLocation[hash].value, strValue, strWidth)) {
         return;
      }

      //printf("Collide \n");

      // This entry is not us so we must have collided
      ++NumCollisions;

      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;

         if (NumCollisions > (HashSize * 2)) {
            //LogError("hash collision error %d %llu\n", i, NumCollisions);
            LogError("Bad hash function, too many collisions");
            return;
         }
      }
   }
   // Failed to find hash
   SetBit(hash);
   ++NumUnique;
   pLocation[hash].Location = i;
   pLocation[hash].value = (INT64)strValue;

}



//-----------------------------------------------
// stores the index of the first location
// remove the forceline to make debugging easier
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalSetLocationUnicode(
   U  i,
   HashLocation* pLocation,
   const char* strValue,
   INT64 strWidth,
   UINT64 hash) {

   //printf("**set %llu  width: %d  string: %s\n", hash, strWidth, strValue);

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      if (UNICODE_MATCH((const char*)pLocation[hash].value, strValue, strWidth)) {
         return;
      }

      //printf("Collide \n");

      // This entry is not us so we must have collided
      ++NumCollisions;

      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;

         if (NumCollisions > (HashSize * 2)) {
            //LogError("hash collision error %d %llu\n", i, NumCollisions);
            LogError("Bad hash function, too many collisions");
            return;
         }
      }
   }
   // Failed to find hash
   SetBit(hash);
   ++NumUnique;
   pLocation[hash].Location = i;
   pLocation[hash].value = (INT64)strValue;

}


//-----------------------------------------------
// looks for the index of set location
// strings must be same width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalGetLocationString(
   INT64  i,
   HashLocation* pLocation,
   INT8* pBooleanOutput,
   U*    pLocationOutput,
   const char* strValue,
   INT64 strWidth,
   UINT64 hash) {

   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      // Check if we have a match from before
      if (STRING_MATCH((const char*)pLocation[hash].value, strValue, strWidth)) {
         // return the first location
         pLocationOutput[i] = pLocation[hash].Location;
         pBooleanOutput[i] = 1;
         return;
      }

      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;
      }
   }
   // Not found
   pLocationOutput[i] = BAD_INDEX;
   pBooleanOutput[i] = 0;
}


//-----------------------------------------------
// looks for the index of set location
// strings must be same width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalGetLocationUnicode(
   INT64  i,
   HashLocation* pLocation,
   INT8* pBooleanOutput,
   U*    pLocationOutput,
   const char* strValue,
   INT64 strWidth,
   UINT64 hash) {

   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      // Check if we have a match from before
      if (UNICODE_MATCH((const char*)pLocation[hash].value, strValue, strWidth)) {
         // return the first location
         pLocationOutput[i] = pLocation[hash].Location;
         pBooleanOutput[i] = 1;
         return;
      }

      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;
      }
   }
   // Not found
   pLocationOutput[i] = BAD_INDEX;
   pBooleanOutput[i] = 0;
}


//-----------------------------------------------
// looks for the index of set location
// strings must be diff width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalGetLocationString2(
   INT64  i,
   HashLocation* pLocation,
   INT8* pBooleanOutput,
   U*    pLocationOutput,
   const char* strValue,
   INT64 strWidth,
   INT64 strWidth2,
   UINT64 hash) {

   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      // Check if we have a match from before
      //if ((i % 1000) == 0) printf("%d  Comparing2 %s to %s\n", (int)i, (const char*)pLocation[hash].value, strValue);
      if (STRING_MATCH2((const char*)pLocation[hash].value, strValue, strWidth, strWidth2)) {
         //printf("match\n");
         // return the first location
         pLocationOutput[i] = pLocation[hash].Location;
         pBooleanOutput[i] = 1;
         return;
      }

      //printf("no match checking next\n");
      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;
      }
   }
   // Not found
   pLocationOutput[i] = BAD_INDEX;
   pBooleanOutput[i] = 0;
}



//-----------------------------------------------
// looks for the index of set location
// strings must be diff width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalGetLocationUnicode2(
   INT64  i,
   HashLocation* pLocation,
   INT8* pBooleanOutput,
   U*    pLocationOutput,
   const char* strValue,
   INT64 strWidth,
   INT64 strWidth2,
   UINT64 hash) {

   const U        BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      // Check if we have a match from before
      //if ((i % 1000) == 0) printf("%d  Comparing2 %s to %s\n", (int)i, (const char*)pLocation[hash].value, strValue);
      if (UNICODE_MATCH2((const char*)pLocation[hash].value, strValue, strWidth, strWidth2)) {
         //printf("match\n");
         // return the first location
         pLocationOutput[i] = pLocation[hash].Location;
         pBooleanOutput[i] = 1;
         return;
      }

      //printf("no match checking next\n");
      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;
      }
   }
   // Not found
   pLocationOutput[i] = BAD_INDEX;
   pBooleanOutput[i] = 0;
}




//-----------------------------------------------
// looks for the index of set location
// strings must be same width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalGetLocationStringCategorical(
   INT64  i,
   HashLocation* pLocation,
   U*    pLocationOutput,
   const char* strValue,
   INT64 strWidth,
   UINT64 hash,
   INT64* missed) {

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      // Check if we have a match from before
      if (STRING_MATCH((const char*)pLocation[hash].value, strValue, strWidth)) {
         // return the first location
         pLocationOutput[i] = pLocation[hash].Location + 1;
         return;
      }

      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;
      }
   }
   // Not found
   pLocationOutput[i] = 0;
   *missed = 1;
}


//-----------------------------------------------
// looks for the index of set location
// strings must be diff width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE
void CHashLinear<T, U>::InternalGetLocationString2Categorical(
   INT64  i,
   HashLocation* pLocation,
   U*    pLocationOutput,
   const char* strValue,
   INT64 strWidth,
   INT64 strWidth2,
   UINT64 hash,
   INT64* missed) {

   while (IsBitSet(hash)) {
      // Check if we have a match from before
      // Check if we have a match from before
      //if ((i % 1000) == 0) printf("%d  Comparing2 %s to %s\n", (int)i, (const char*)pLocation[hash].value, strValue);
      if (STRING_MATCH2((const char*)pLocation[hash].value, strValue, strWidth, strWidth2)) {
         //printf("match\n");
         // return the first location
         pLocationOutput[i] = pLocation[hash].Location + 1;
         return;
      }

      //printf("no match checking next\n");
      // Linear goes to next position
      if (++hash >= HashSize) {
         hash = 0;
      }
   }
   // Not found
   pLocationOutput[i] = 0;
   *missed = 1;
}



//-----------------------------------------------
// outputs misses as 0 or 1
// outputs location  + 1, if not found places in 0
//
// strWidth is the width for pHashList and the first argument called passed
// strWidth2 was used in InternalSetLocationString and refers to second argument

template<typename T, typename U>
static INT64 IsMemberStringCategorical(
   void* pHashLinearVoid,
   INT64 arraySize,
   INT64 strWidth,
   INT64 strWidth2,
   const char* pHashList,
   void*  pLocationOutputU,
   BOOL   isUnicode) {

   struct HashLocation
   {
      T        value;
      U        Location;
   };

   CHashLinear<T, U>* pHashLinear = (CHashLinear<T, U>*)pHashLinearVoid;

   HashLocation* pLocation = (HashLocation*)pHashLinear->pHashTableAny;
   U* pLocationOutput = (U*)pLocationOutputU;

   INT64 missed = 0;
   UINT64 HashSize = pHashLinear->HashSize;

   // to determine if hash location has been visited
   UINT64*     pBitFields = pHashLinear->pBitFields;

   if (strWidth == strWidth2) {
      //-------------------------------------------------------------------
      // STRINGS are SAME SIZE --------------------------------------------
      if (isUnicode) {
         for (INT64 i = 0; i < arraySize; i++) {
            HASH_UNICODE()

            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);

            while (1) {
               UINT64 index = hash >> 6;
               if (pBitFields[index] & (1LL << (hash & 63))) {
                  // Check if we have a match from before
                  // Check if we have a match from before
                  if (UNICODE_MATCH((const char*)pLocation[hash].value, strStart, strWidth)) {
                     // return the first location
                     pLocationOutput[i] = pLocation[hash].Location + 1;
                     break;
                  }

                  // Linear goes to next position
                  if (++hash >= HashSize) {
                     hash = 0;
                  }
                  continue;

               }
               else {

                  // Not found
                  pLocationOutput[i] = 0;
                  missed = 1;
                  break;
               }
            }

         }
      }
      else {

         for (INT64 i = 0; i < arraySize; i++) {
            HASH_STRING()

               // Use and mask to strip off high bits
               hash = hash & (HashSize - 1);
            while (1) {
               UINT64 index = hash >> 6;
               if (pBitFields[index] & (1LL << (hash & 63))) {
                  // Check if we have a match from before
                  if (STRING_MATCH((const char*)pLocation[hash].value, strStart, strWidth)) {
                     // return the first location
                     pLocationOutput[i] = pLocation[hash].Location + 1;
                     break;
                  }

                  // Linear goes to next position
                  if (++hash >= HashSize) {
                     hash = 0;
                  }
                  continue;

               }
               else {

                  // Not found
                  pLocationOutput[i] = 0;
                  missed = 1;
                  break;
               }
            }
         }
      }
   }
   else {
      //-------------------------------------------------------------------
      // STRINGS are DIFFERENT SIZE --------------------------------------------
      if (isUnicode) {
         for (INT64 i = 0; i < arraySize; i++) {
            HASH_UNICODE()

            // Use and mask to strip off high bits
            hash = hash & (pHashLinear->HashSize - 1);

            while (1) {
               UINT64 index = hash >> 6;
               if (pBitFields[index] & (1LL << (hash & 63))) {
                  // Check if we have a match from before
                  // Check if we have a match from before
                  if (UNICODE_MATCH2((const char*)pLocation[hash].value, strStart, strWidth2, strWidth)) {
                     // return the first location
                     pLocationOutput[i] = pLocation[hash].Location + 1;
                     break;
                  }

                  // Linear goes to next position
                  if (++hash >= HashSize) {
                     hash = 0;
                  }
                  continue;

               }
               else {

                  // Not found
                  pLocationOutput[i] = 0;
                  missed = 1;
                  break;
               }
            }

         }
      }
      else {

         for (INT64 i = 0; i < arraySize; i++) {
            HASH_STRING()

            // Use and mask to strip off high bits
            hash = hash & (pHashLinear->HashSize - 1);

            while (1) {
               UINT64 index = hash >> 6;
               if (pBitFields[index] & (1LL << (hash & 63))) {
                  // Check if we have a match from before
                  if (STRING_MATCH2((const char*)pLocation[hash].value, strStart, strWidth2, strWidth)) {
                     // return the first location
                     pLocationOutput[i] = pLocation[hash].Location + 1;
                     break;
                  }

                  // Linear goes to next position
                  if (++hash >= pHashLinear->HashSize) {
                     hash = 0;
                  }
                  continue;

               }
               else {

                  // Not found
                  pLocationOutput[i] = 0;
                  missed = 1;
                  break;
               }
            }
         }
      }
   }

   return missed;
}



//-----------------------------------------------
// outputs boolean array
// outputs location array
//
// strWidth is the width for pHashList and the first argument called passed
// strWidth2 was used in InternalSetLocationString and refers to second argument

template<typename T, typename U>
void CHashLinear<T, U>::IsMemberString(
   INT64 arraySize,
   INT64 strWidth,
   INT64 strWidth2,
   const char* pHashList,
   INT8* pBooleanOutput,
   U*    pLocationOutput,
   BOOL  isUnicode) {

   HashLocation* pLocation = (HashLocation*)pHashTableAny;

   if (strWidth == strWidth2) {
      //-------------------------------------------------------------------
      // STRINGS are SAME SIZE --------------------------------------------
      if (isUnicode) {
         for (INT64 i = 0; i < arraySize; i++) {
            HASH_UNICODE()

            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            //printf("**uni hash %lld\n", (long long)hash);
            InternalGetLocationUnicode(i, pLocation, pBooleanOutput, pLocationOutput, strStart, strWidth,  hash);
         }
      }
      else {

         for (INT64 i = 0; i < arraySize; i++) {
            HASH_STRING()

            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            InternalGetLocationString(i, pLocation, pBooleanOutput, pLocationOutput, strStart, strWidth,  hash);
         }
      }
   }
   else {
      //-------------------------------------------------------------------
      // STRINGS are DIFFERENT SIZE --------------------------------------------
      if (isUnicode) {
         for (INT64 i = 0; i < arraySize; i++) {
            HASH_UNICODE()

            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            InternalGetLocationUnicode2(i, pLocation, pBooleanOutput, pLocationOutput, strStart, strWidth2, strWidth, hash);
         }
      }
      else {

         for (INT64 i = 0; i < arraySize; i++) {
            HASH_STRING()

            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            InternalGetLocationString2(i, pLocation, pBooleanOutput, pLocationOutput, strStart, strWidth2, strWidth, hash);
         }
      }
   }
}


//===============================================================================
//===============================================================================

typedef void(*ISMEMBER_MT)(
   void* pHashLinearVoid,
   INT64 arraySize,
   void* pHashList,
   INT8* pBooleanOutput,
   void* pLocationOutputU);

//--------------------------------------------------------------------
struct IMMT_CALLBACK {
   ISMEMBER_MT anyIMMTCallback;

   void* pHashLinearVoid;

   INT64 size1;
   void* pHashList;
   INT8* pBooleanOutput;
   void* pOutput;
   INT64 typeSizeIn;
   INT64 typeSizeOut;

} stIMMTCallback;



//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static BOOL IMMTThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex) {
   BOOL didSomeWork = FALSE;
   IMMT_CALLBACK* Callback = (IMMT_CALLBACK*)pstWorkerItem->WorkCallbackArg;

   char* pInput1 = (char *)Callback->pHashList;
   char* pOutput = (char*)Callback->pOutput;
   INT64 lenX;
   INT64 workBlock;

   // As long as there is work to do
   while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

      INT64 inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeIn;
      INT64 outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;
      INT64 booleanAdj = pstWorkerItem->BlockSize * workBlock;

      Callback->anyIMMTCallback(Callback->pHashLinearVoid, lenX, pInput1 + inputAdj, Callback->pBooleanOutput + booleanAdj, pOutput + outputAdj);

      // Indicate we completed a block
      didSomeWork = TRUE;

      // tell others we completed this work block
      pstWorkerItem->CompleteWorkBlock();
      //printf("|%d %d", core, (int)workBlock);
   }

   return didSomeWork;

}



static void IsMemberMultiThread(
   ISMEMBER_MT pFunction,
   void* pHashLinearVoid,
   INT64 arraySize,
   void* pHashList,
   INT8* pBooleanOutput,
   void* pLocationOutputU,
   INT64 sizeInput,
   INT64 sizeOutput) {


   stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(arraySize);

   if (pWorkItem == NULL) {

      // Threading not allowed for this work item, call it directly from main thread
      pFunction(pHashLinearVoid, arraySize, pHashList, pBooleanOutput, pLocationOutputU);
   }
   else {
      // Each thread will call this routine with the callbackArg
      pWorkItem->DoWorkCallback = IMMTThreadCallback;

      pWorkItem->WorkCallbackArg = &stIMMTCallback;

      stIMMTCallback.pHashLinearVoid = pHashLinearVoid;
      stIMMTCallback.anyIMMTCallback = pFunction;
      stIMMTCallback.size1 = arraySize;
      stIMMTCallback.pHashList = pHashList;
      stIMMTCallback.pBooleanOutput = pBooleanOutput;
      stIMMTCallback.pOutput = pLocationOutputU;
      stIMMTCallback.typeSizeIn = sizeInput;
      stIMMTCallback.typeSizeOut = sizeOutput;

      // This will notify the worker threads of a new work item
      g_cMathWorker->WorkMain(pWorkItem, arraySize, 0);
   }

}



//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
// floats are 4 bytes and will be handled like INT32 or UIN32
//
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//
// Returns in pOutput: index location of second arg -- where first arg found in second arg
// Returns in pBooleanOutput: True if found, False otherwise
template<typename U>
void* IsMemberHash32(
   INT64 size1,
   void* pInput1,

   INT64 size2,
   void* pInput2,

   U* pOutput,

   INT8* pBooleanOutput,
   INT sizeType,
   HASH_MODE hashMode,
   INT64 hintSize) {


   // Allocate hash

   switch (sizeType) {
   case 1: {
      CHashLinear<UINT8, U>* pHashLinear = new CHashLinear<UINT8, U>(HASH_MODE_MASK);
      pHashLinear->MakeHashLocation(size2, (UINT8*)pInput2, 256 / 2);
      IsMemberMultiThread(IsMember<UINT8, U>, pHashLinear, size1, (UINT8*)pInput1, pBooleanOutput, (U*)pOutput, sizeof(UINT8), sizeof(U));
      delete pHashLinear;
      return NULL;
   }
           break;
   case 2: {
      CHashLinear<UINT16, U>* pHashLinear = new CHashLinear<UINT16, U>(HASH_MODE_MASK);
      pHashLinear->MakeHashLocation(size2, (UINT16*)pInput2, 65536 / 2);
      IsMemberMultiThread(IsMember<UINT16, U>, pHashLinear, size1, (UINT16*)pInput1, pBooleanOutput, (U*)pOutput, sizeof(UINT16), sizeof(U));
      delete pHashLinear;
      return NULL;
   }
           break;

   case 4:
   {
      CHashLinear<UINT32, U>* pHashLinear = new CHashLinear<UINT32, U>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT32*)pInput2, hintSize);
      IsMemberMultiThread(IsMember<UINT32, U>, pHashLinear, size1, (UINT32*)pInput1, pBooleanOutput, (U*)pOutput, sizeof(UINT32), sizeof(U));
      delete pHashLinear;
      return NULL;
   }
   break;
   case 8:
   {
      CHashLinear<UINT64, U>* pHashLinear = new CHashLinear<UINT64, U>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT64*)pInput2, hintSize);
      IsMemberMultiThread(IsMember<UINT64, U>, pHashLinear, size1, (UINT64*)pInput1, pBooleanOutput, (U*)pOutput, sizeof(UINT64), sizeof(U));
      delete pHashLinear;
      return NULL;
   }
   break;
   case 104:
   {
      CHashLinear<FLOAT, U>* pHashLinear = new CHashLinear<FLOAT, U>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (FLOAT*)pInput2, hintSize);
      IsMemberMultiThread(IsMemberFloat<FLOAT, U>, pHashLinear, size1, (FLOAT*)pInput1, pBooleanOutput, (U*)pOutput, sizeof(FLOAT), sizeof(U));
      delete pHashLinear;
      return NULL;
   }
   break;
   case 108:
   {
      CHashLinear<DOUBLE, U>* pHashLinear = new CHashLinear<DOUBLE, U>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (DOUBLE*)pInput2, hintSize);
      IsMemberMultiThread(IsMemberFloat<DOUBLE, U>, pHashLinear, size1, (DOUBLE*)pInput1, pBooleanOutput, (U*)pOutput, sizeof(DOUBLE), sizeof(U));
      delete pHashLinear;
      return NULL;
   }
   case 116:
   {
      printf("!!! Linux long double case not handled\n");
   }
   break;

   }


   return NULL;
}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
// floats are + 100 and will be handled differnt from INT64 or UIN32
void* IsMemberHash64(
   INT64 size1,
   void* pInput1,
   INT64 size2,
   void* pInput2,
   INT64* pOutput,
   INT8* pBooleanOutput,
   INT sizeType,
   HASH_MODE hashMode,
   INT64 hintSize) {

   switch (sizeType) {
   case 1: {
      CHashLinear<UINT8, INT64>* pHashLinear = new CHashLinear<UINT8, INT64>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT8*)pInput2, hintSize);
      IsMember<UINT8, INT64>(pHashLinear, size1, (UINT8*)pInput1, pBooleanOutput, (INT64*)pOutput);
      delete pHashLinear;
      return NULL;
   }
           break;
   case 2: {
      CHashLinear<UINT16, INT64>* pHashLinear = new CHashLinear<UINT16, INT64>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT16*)pInput2, hintSize);
      IsMember<UINT16, INT64>(pHashLinear, size1, (UINT16*)pInput1, pBooleanOutput, (INT64*)pOutput);
      delete pHashLinear;
      return NULL;
   }
           break;

   case 4:
   {
      CHashLinear<UINT32, INT64>* pHashLinear = new CHashLinear<UINT32, INT64>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT32*)pInput2, hintSize);
      IsMember<UINT32, INT64>(pHashLinear, size1, (UINT32*)pInput1, pBooleanOutput, (INT64*)pOutput);
      delete pHashLinear;
      return NULL;
   }
   break;
   case 8:
   {
      CHashLinear<UINT64, INT64>* pHashLinear = new CHashLinear<UINT64, INT64>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT64*)pInput2, hintSize);
      IsMember<UINT64, INT64>(pHashLinear, size1, (UINT64*)pInput1, pBooleanOutput, (INT64*)pOutput);
      delete pHashLinear;
      return NULL;
   }
   break;
   case 104:
   {
      CHashLinear<FLOAT, INT64>* pHashLinear = new CHashLinear<FLOAT, INT64>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (FLOAT*)pInput2, hintSize);
      IsMemberFloat<FLOAT, INT64>(pHashLinear, size1, (FLOAT*)pInput1, pBooleanOutput, (INT64*)pOutput);
      delete pHashLinear;
      return NULL;
   }
   break;
   case 108:
   {
      CHashLinear<DOUBLE, INT64>* pHashLinear = new CHashLinear<DOUBLE, INT64>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (DOUBLE*)pInput2, hintSize);
      IsMemberFloat<DOUBLE, INT64>(pHashLinear, size1, (DOUBLE*)pInput1, pBooleanOutput, (INT64*)pOutput);
      delete pHashLinear;
      return NULL;
   }
   break;

   }


   return NULL;
}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
// floats are 4 bytes and will be handled like INT32 or UIN32
//
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//
// Returns in pOutput: index location of second arg -- where first arg found in second arg
// Returns in pBooleanOutput: True if found, False otherwise
INT64 IsMemberHashCategorical(
   INT64 size1,
   void* pInput1,
   INT64 size2,
   void* pInput2,
   INT32* pOutput,
   INT sizeType,
   HASH_MODE hashMode,
   INT64 hintSize) {


   INT64 missed = 0;
   // Allocate hash

   switch (sizeType) {
   case 1: {
      CHashLinear<UINT8, INT32>* pHashLinear = new CHashLinear<UINT8, INT32>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT8*)pInput2, hintSize);
      missed = pHashLinear->IsMemberCategorical(size1, (UINT8*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;

   case 2: {
      CHashLinear<UINT16, INT32>* pHashLinear = new CHashLinear<UINT16, INT32>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT16*)pInput2, hintSize);
      missed = pHashLinear->IsMemberCategorical(size1, (UINT16*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;

   case 4:
   {
      CHashLinear<UINT32, INT32>* pHashLinear = new CHashLinear<UINT32, INT32>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT32*)pInput2, hintSize);
      missed = pHashLinear->IsMemberCategorical(size1, (UINT32*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;
   case 8:
   {
      CHashLinear<UINT64, INT32>* pHashLinear = new CHashLinear<UINT64, INT32>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT64*)pInput2, hintSize);
      missed = pHashLinear->IsMemberCategorical(size1, (UINT64*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;
   case 104:
   {
      CHashLinear<FLOAT, INT32>* pHashLinear = new CHashLinear<FLOAT, INT32>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (FLOAT*)pInput2, hintSize);
      missed = pHashLinear->IsMemberFloatCategorical(size1, (FLOAT*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;
   case 108:
   {
      CHashLinear<DOUBLE, INT32>* pHashLinear = new CHashLinear<DOUBLE, INT32>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (DOUBLE*)pInput2, hintSize);
      missed = pHashLinear->IsMemberFloatCategorical(size1, (DOUBLE*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;
   case 116:
   {
      CHashLinear<long double, INT32>* pHashLinear = new CHashLinear<long double, INT32>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (long double*)pInput2, hintSize);
      missed = pHashLinear->IsMemberFloatCategorical(size1, (long double*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;

   }


   return missed;
}



//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
// floats are 4 bytes and will be handled like INT32 or UIN32
//
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//
// Returns in pOutput: index location of second arg -- where first arg found in second arg
// Returns in pBooleanOutput: True if found, False otherwise
INT64 IsMemberHashCategorical64(
   INT64 size1,
   void* pInput1,
   INT64 size2,
   void* pInput2,
   INT64* pOutput,
   INT sizeType,
   HASH_MODE hashMode,
   INT64 hintSize) {


   INT64 missed = 0;
   // Allocate hash

   switch (sizeType) {
   case 1: {
      CHashLinear<UINT8, INT64>* pHashLinear = new CHashLinear<UINT8, INT64>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT8*)pInput2, hintSize);
      missed = pHashLinear->IsMemberCategorical(size1, (UINT8*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
           break;

   case 2: {
      CHashLinear<UINT16, INT64>* pHashLinear = new CHashLinear<UINT16, INT64>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT16*)pInput2, hintSize);
      missed = pHashLinear->IsMemberCategorical(size1, (UINT16*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
           break;

   case 4:
   {
      CHashLinear<UINT32, INT64>* pHashLinear = new CHashLinear<UINT32, INT64>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT32*)pInput2, hintSize);
      missed = pHashLinear->IsMemberCategorical(size1, (UINT32*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;
   case 8:
   {
      CHashLinear<UINT64, INT64>* pHashLinear = new CHashLinear<UINT64, INT64>(hashMode);
      pHashLinear->MakeHashLocation(size2, (UINT64*)pInput2, hintSize);
      missed = pHashLinear->IsMemberCategorical(size1, (UINT64*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;
   case 104:
   {
      CHashLinear<FLOAT, INT64>* pHashLinear = new CHashLinear<FLOAT, INT64>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (FLOAT*)pInput2, hintSize);
      missed = pHashLinear->IsMemberFloatCategorical(size1, (FLOAT*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;
   case 108:
   {
      CHashLinear<DOUBLE, INT64>* pHashLinear = new CHashLinear<DOUBLE, INT64>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (DOUBLE*)pInput2, hintSize);
      missed = pHashLinear->IsMemberFloatCategorical(size1, (DOUBLE*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;
   case 116:
   {
      CHashLinear<long double, INT64>* pHashLinear = new CHashLinear<long double, INT64>(hashMode);
      pHashLinear->MakeHashLocationFloat(size2, (long double*)pInput2, hintSize);
      missed = pHashLinear->IsMemberFloatCategorical(size1, (long double*)pInput1, pOutput);
      delete pHashLinear;
      return missed;
   }
   break;

   }


   return missed;
}



//===================================================================================================

typedef INT64(*ISMEMBER_STRING)(void* pHashLinearVoid, INT64 arraySize, INT64 strWidth1, INT64 strWidth2, const char* pHashList, void*    pLocationOutputU, BOOL isUnicode);

//--------------------------------------------------------------------
struct IMS_CALLBACK {
   ISMEMBER_STRING anyIMSCallback;

   void* pHashLinearVoid;

   INT64 size1;
   INT64 strWidth1;
   const char* pInput1;
   INT64 size2;
   INT64 strWidth2;
   void* pOutput;
   INT64 typeSizeOut;
   INT64 missed;

   BOOL  isUnicode;

} stIMSCallback;



//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static BOOL IMSThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex) {
   BOOL didSomeWork = FALSE;
   IMS_CALLBACK* Callback = (IMS_CALLBACK*)pstWorkerItem->WorkCallbackArg;

   BOOL isUnicode = Callback->isUnicode;
   char* pInput1 = (char *)Callback->pInput1;
   char* pOutput = (char*)Callback->pOutput;
   INT64 lenX;
   INT64 workBlock;

   // As long as there is work to do
   while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

      INT64 inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->strWidth1;
      INT64 outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;

      INT64 missed = Callback->anyIMSCallback(Callback->pHashLinearVoid, lenX, Callback->strWidth1, Callback->strWidth2, pInput1 + inputAdj, pOutput + outputAdj, isUnicode);

      // Careful with multithreading -- only set it to 1
      if (missed) {
         Callback->missed = 1;
      }

      // Indicate we completed a block
      didSomeWork = TRUE;

      // tell others we completed this work block
      pstWorkerItem->CompleteWorkBlock();
      //printf("|%d %d", core, (int)workBlock);
   }

   return didSomeWork;

}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
template<typename U>
INT64 IsMemberHashStringCategorical(
   INT64 size1,
   INT64 strWidth1,
   const char* pInput1,
   INT64 size2,
   INT64 strWidth2,
   const char* pInput2,
   U* pOutput,
   HASH_MODE hashMode,
   INT64 hintSize,
   BOOL isUnicode) {

   CHashLinear<UINT64, U>* pHashLinear = new CHashLinear<UINT64, U>(hashMode);

   LOGGING("MakeHashLocationString  %lld  %p  strdwidth2: %lld  hashMode %d\n", size2, pInput2, strWidth2, (int)hashMode);

   // First pass build hash table of second string input
   pHashLinear->MakeHashLocationString(size2, pInput2, strWidth2, hintSize, isUnicode);

   LOGGING("IsMemberString  %lld  %lld  strdwidth2: %lld\n", size1, strWidth1, strWidth2);

   // Second pass find matches
   // We can multithread it
   INT64 missed;

   stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(size1);
   ISMEMBER_STRING pFunction = IsMemberStringCategorical<UINT64, U>;

   if (pWorkItem == NULL) {

      // Threading not allowed for this work item, call it directly from main thread
      missed =
         pFunction(pHashLinear, size1, strWidth1, strWidth2, pInput1, pOutput, isUnicode);

   }
   else {
      // Each thread will call this routine with the callbackArg
      pWorkItem->DoWorkCallback = IMSThreadCallback;

      pWorkItem->WorkCallbackArg = &stIMSCallback;

      stIMSCallback.pHashLinearVoid = pHashLinear;
      stIMSCallback.anyIMSCallback = pFunction;
      stIMSCallback.strWidth1 = strWidth1;
      stIMSCallback.pInput1 = pInput1;
      stIMSCallback.size2 = size2;
      stIMSCallback.strWidth2 = strWidth2;
      stIMSCallback.pOutput = pOutput;
      stIMSCallback.typeSizeOut = sizeof(U);
      stIMSCallback.missed = 0;
      stIMSCallback.isUnicode = isUnicode;

      // This will notify the worker threads of a new work item
      g_cMathWorker->WorkMain(pWorkItem, size1, 0);
      missed = stIMSCallback.missed;
   }

   LOGGING("IsMemberHashStringCategorical  done\n");

   delete pHashLinear;
   return missed;
}



//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
template<typename U>
void IsMemberHashString32(
   INT64 size1,
   INT64 strWidth1,
   const char* pInput1,
   INT64 size2,
   INT64 strWidth2,
   const char* pInput2,
   U* pOutput,
   INT8* pBooleanOutput,
   HASH_MODE hashMode,
   INT64 hintSize,
   BOOL isUnicode) {

   CHashLinear<UINT64, U>* pHashLinear = new CHashLinear<UINT64, U>(hashMode);

   LOGGING("MakeHashLocationString  %lld  %p  strdwidth2: %lld  hashMode %d\n", size2, pInput2, strWidth2, (int)hashMode);

   // First pass build hash table of second string input
   pHashLinear->MakeHashLocationString(size2, pInput2, strWidth2, hintSize, isUnicode);

   LOGGING("IsMemberString  %lld  %lld  strdwidth2: %lld\n", size1, strWidth1, strWidth2);

   // Second pass find matches
   // We can multithread it
   pHashLinear->IsMemberString(size1, strWidth1, strWidth2, pInput1, pBooleanOutput, pOutput, isUnicode);

   LOGGING("IsMemberHashString32  done\n");

   delete pHashLinear;
}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void IsMemberHashString64(
   INT64 size1,
   INT64 strWidth1,
   const char* pInput1,
   INT64 size2,
   INT64 strWidth2,
   const char* pInput2,
   INT64* pOutput,
   INT8* pBooleanOutput,
   HASH_MODE hashMode,
   INT64 hintSize,
   BOOL isUnicode) {

   CHashLinear<UINT64, INT64>* pHashLinear = new CHashLinear<UINT64, INT64>(hashMode);
   pHashLinear->MakeHashLocationString(size2, pInput2, strWidth2, hintSize, isUnicode);
   pHashLinear->IsMemberString(size1, strWidth1, strWidth2, pInput1, pBooleanOutput, (INT64*)pOutput, isUnicode);
   delete pHashLinear;
}


//-----------------------------------------------------------------------------------------
// Returns 8/16/32/64 bit indexes
template<typename KEY_TYPE>
BOOL MergePreBinned(
   INT64 size1,
   KEY_TYPE*    pKey1,
   void* pInVal1,
   INT64 size2,
   KEY_TYPE*    pKey2,
   void* pInVal2,
   KEY_TYPE*    pOutput,
   INT64 totalUniqueSize,
   HASH_MODE hashMode,
   INT dtype) {

   BOOL success = TRUE;

   LOGGING("AlignCategorical32 dtype: %d  size1: %lld  size2: %lld\n", dtype, size1, size2);

   switch (dtype) {
   CASE_NPY_INT64:
      FindLastMatchCategorical<KEY_TYPE, INT64>(size1, size2, pKey1, pKey2, (INT64*)pInVal1, (INT64*)pInVal2, pOutput, totalUniqueSize);
      break;
   CASE_NPY_INT32:
      FindLastMatchCategorical<KEY_TYPE, INT32>(size1, size2, pKey1, pKey2, (INT32*)pInVal1, (INT32*)pInVal2, pOutput, totalUniqueSize);
      break;
   case NPY_FLOAT64:
      FindLastMatchCategorical<KEY_TYPE, double>(size1, size2, pKey1, pKey2, (double*)pInVal1, (double*)pInVal2, pOutput, totalUniqueSize);
      break;
   case NPY_FLOAT32:
      FindLastMatchCategorical<KEY_TYPE, float>(size1, size2, pKey1, pKey2, (float*)pInVal1, (float*)pInVal2, pOutput, totalUniqueSize);
      break;
   default:
      success = FALSE;
      break;
   }
   return success;

}





//Based on input type, calls different functions
//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------
// Returns 32 bit indexes
BOOL AlignHashMK32(
   INT64 size1,
   void* pInput1,
   void* pInVal1,
   INT64 size2,
   void* pInput2,
   void* pInVal2,
   INT32* pOutput,
   INT64 totalItemSize,
   HASH_MODE hashMode,
   INT dtype,
   bool isForward,
   bool allowExact) {

   BOOL success = TRUE;
   CHashLinear<char, INT32>* pHashLinear = new CHashLinear<char, INT32>(hashMode);

   switch (dtype) {
   CASE_NPY_INT64:
      if (isForward) {
         pHashLinear->FindNextMatchMK<INT64>(size1, size2, (char*)pInput1, (char*)pInput2, (INT64*)pInVal1, (INT64*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      else {
         pHashLinear->FindLastMatchMK<INT64>(size1, size2, (char*)pInput1, (char*)pInput2, (INT64*)pInVal1, (INT64*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      break;
   CASE_NPY_INT32:
      if (isForward) {
         pHashLinear->FindNextMatchMK<INT32>(size1, size2, (char*)pInput1, (char*)pInput2, (INT32*)pInVal1, (INT32*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      else {
         pHashLinear->FindLastMatchMK<INT32>(size1, size2, (char*)pInput1, (char*)pInput2, (INT32*)pInVal1, (INT32*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      break;
   case NPY_FLOAT64:
      if (isForward) {
         pHashLinear->FindNextMatchMK<double>(size1, size2, (char*)pInput1, (char*)pInput2, (double*)pInVal1, (double*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      else {
         pHashLinear->FindLastMatchMK<double>(size1, size2, (char*)pInput1, (char*)pInput2, (double*)pInVal1, (double*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      break;
   case NPY_FLOAT32:
      if (isForward) {
         pHashLinear->FindNextMatchMK<float>(size1, size2, (char*)pInput1, (char*)pInput2, (float*)pInVal1, (float*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      else {
         pHashLinear->FindLastMatchMK<float>(size1, size2, (char*)pInput1, (char*)pInput2, (float*)pInVal1, (float*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      break;
   default:
      success = FALSE;
      break;
   }
   delete pHashLinear;
   return success;

}

//-----------------------------------------------------------------------------------------
// Returns 64 bit indexes
BOOL AlignHashMK64(
   INT64 size1,
   void* pInput1,
   void* pInVal1,
   INT64 size2,
   void* pInput2,
   void* pInVal2,
   INT64* pOutput,
   INT64 totalItemSize,
   HASH_MODE hashMode,
   INT dtype,
   bool isForward,
   bool allowExact) {


   BOOL success = TRUE;
   CHashLinear<char, INT64>* pHashLinear = new CHashLinear<char, INT64>(hashMode);

   switch (dtype) {
   CASE_NPY_INT64:
      if (isForward) {
         pHashLinear->FindNextMatchMK<INT64>(size1, size2, (char*)pInput1, (char*)pInput2, (INT64*)pInVal1, (INT64*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      else {
         pHashLinear->FindLastMatchMK<INT64>(size1, size2, (char*)pInput1, (char*)pInput2, (INT64*)pInVal1, (INT64*)pInVal2, pOutput, totalItemSize, allowExact);
      }

      break;
   CASE_NPY_INT32:
      if (isForward) {
         pHashLinear->FindNextMatchMK<INT32>(size1, size2, (char*)pInput1, (char*)pInput2, (INT32*)pInVal1, (INT32*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      else {
         pHashLinear->FindLastMatchMK<INT32>(size1, size2, (char*)pInput1, (char*)pInput2, (INT32*)pInVal1, (INT32*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      break;
   case NPY_FLOAT64:
      if (isForward) {
         pHashLinear->FindNextMatchMK<double>(size1, size2, (char*)pInput1, (char*)pInput2, (double*)pInVal1, (double*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      else {
         pHashLinear->FindLastMatchMK<double>(size1, size2, (char*)pInput1, (char*)pInput2, (double*)pInVal1, (double*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      break;
   case NPY_FLOAT32:
      if (isForward) {
         pHashLinear->FindNextMatchMK<float>(size1, size2, (char*)pInput1, (char*)pInput2, (float*)pInVal1, (float*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      else {
         pHashLinear->FindLastMatchMK<float>(size1, size2, (char*)pInput1, (char*)pInput2, (float*)pInVal1, (float*)pInVal2, pOutput, totalItemSize, allowExact);
      }
      break;
   default:
      success = FALSE;
      break;
   }
   delete pHashLinear;
   return success;

}

//----------------------------------------------
// any non standard size
template<typename HASH_TYPE, typename INDEX_TYPE>
UINT64
DoLinearHash(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   INDEX_TYPE* pIndexArray,
   void** pFirstArrayVoid,
   void** pHashTableAny,
   INT64* hashTableSize,
   HASH_MODE hashMode,
   INT64  hintSize,
   bool* pBoolFilter) {

   UINT64 numUnique = 0;
   CHashLinear<HASH_TYPE, INDEX_TYPE>* pHashLinear = new CHashLinear<HASH_TYPE, INDEX_TYPE>(hashMode, FALSE);
   INDEX_TYPE* pFirstArray = (INDEX_TYPE*)pHashLinear->AllocMemory(hintSize, -2, sizeof(INDEX_TYPE) * (totalRows + 1), FALSE);

   // Handles any size
   numUnique =
      pHashLinear->GroupBy(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray, hashMode, hintSize, pBoolFilter);

   *pHashTableAny = pHashLinear->pHashTableAny;
   *hashTableSize = pHashLinear->HashTableAllocSize;
   *pFirstArrayVoid = pFirstArray;
   delete pHashLinear;
   return numUnique;
}


//----------------------------------------------
// common float
template<typename HASH_TYPE, typename INDEX_TYPE>
UINT64
DoLinearHashFloat(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   INDEX_TYPE* pIndexArray,
   void** pFirstArrayVoid,
   void** pHashTableAny,
   INT64* hashTableSize,
   HASH_MODE hashMode,
   INT64  hintSize,
   bool* pBoolFilter) {

   UINT64 numUnique = 0;
   CHashLinear<HASH_TYPE, INDEX_TYPE>* pHashLinear = new CHashLinear<HASH_TYPE, INDEX_TYPE>(hashMode, FALSE);
   INDEX_TYPE* pFirstArray = (INDEX_TYPE*)pHashLinear->AllocMemory(hintSize, -1, sizeof(INDEX_TYPE) * (totalRows + 1), FALSE);

   numUnique =
      pHashLinear->GroupByFloat(totalRows, totalItemSize, (HASH_TYPE*)pInput1, coreType, pIndexArray, pFirstArray, hashMode, hintSize, pBoolFilter);

   // Copy these before they get deleted
   *pHashTableAny = pHashLinear->pHashTableAny;
   *hashTableSize = pHashLinear->HashTableAllocSize;
   *pFirstArrayVoid = pFirstArray;
   delete pHashLinear;
   return numUnique;
}

//----------------------------------------------
// common types non-float
template<typename HASH_TYPE, typename INDEX_TYPE>
UINT64
DoLinearHashItemSize(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,  // This is the numpy type code, e.g. NPY_FLOAT32
   INDEX_TYPE* pIndexArray,
   void** pFirstArrayVoid,
   void** pHashTableAny,
   INT64* hashTableSize,
   HASH_MODE hashMode,
   INT64  hintSize,
   bool* pBoolFilter) {

   UINT64 numUnique = 0;

   CHashLinear<HASH_TYPE, INDEX_TYPE>* pHashLinear = new CHashLinear<HASH_TYPE, INDEX_TYPE>(hashMode, FALSE);
   INDEX_TYPE* pFirstArray = (INDEX_TYPE*)pHashLinear->AllocMemory(hintSize, -1, sizeof(INDEX_TYPE) * (totalRows + 1), FALSE);

   if (pFirstArray) {
      numUnique =
         pHashLinear->GroupByItemSize(totalRows, totalItemSize, (HASH_TYPE*)pInput1, coreType, pIndexArray, pFirstArray, hashMode, hintSize, pBoolFilter);
   }

   // Copy these before they get deleted
   *pHashTableAny = pHashLinear->pHashTableAny;
   *hashTableSize = pHashLinear->HashTableAllocSize;
   *pFirstArrayVoid = pFirstArray;

   delete pHashLinear;
   return numUnique;
}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
template<typename INDEX_TYPE>
UINT64 GroupByInternal(
   void** pFirstArray,
   void** pHashTableAny,
   INT64* hashTableSize,

   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   INDEX_TYPE* pIndexArray,
   HASH_MODE hashMode,
   INT64  hintSize,
   bool* pBoolFilter) {

   UINT64 numUnique = 0;
   BOOL calculated = FALSE;

   if (hintSize == 0) {
      hintSize = totalRows;
   }

   //
   // TODO: Need to add special handling for bools
   //

   // Calling the hash function will return the FirstArray

   switch (coreType) {
   case NPY_FLOAT32:
   {
      // so that nans compare, we tell it is uint32
      numUnique =
         DoLinearHashFloat<UINT32, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray, pHashTableAny, hashTableSize, hashMode, hintSize, pBoolFilter);
      calculated = TRUE;
   }
   break;
   case NPY_FLOAT64:
   {
      // so that nans compare, we tell it is uint64
      numUnique =
         DoLinearHashFloat<UINT64, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray, pHashTableAny, hashTableSize, hashMode, hintSize, pBoolFilter);
      calculated = TRUE;
   }
   break;
   }

   // Now go based on size
   if (calculated == FALSE) {
      switch (totalItemSize) {
      case 1:
      {
         numUnique =
            DoLinearHashItemSize<UINT8, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray, pHashTableAny, hashTableSize, hashMode, 256/2, pBoolFilter);
         calculated = TRUE;
      }
      break;
      case 2:
      {
         numUnique =
            DoLinearHashItemSize<UINT16, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray, pHashTableAny, hashTableSize, hashMode, 65536/2, pBoolFilter);
         calculated = TRUE;
      }
      break;
      case 4:
      {
         numUnique =
            DoLinearHashItemSize<UINT32, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray, pHashTableAny, hashTableSize, hashMode, hintSize, pBoolFilter);
         calculated = TRUE;
      }
      break;
      case 8:
      {
         numUnique =
            DoLinearHashItemSize<UINT64, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray, pHashTableAny, hashTableSize, hashMode, hintSize, pBoolFilter);
         calculated = TRUE;
      }
      break;
      }
   }

   if (calculated == FALSE) {
      numUnique =
         DoLinearHash<UINT32, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray, pHashTableAny, hashTableSize, hashMode, hintSize, pBoolFilter);
   }

   return numUnique;
}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
UINT64 GroupBy32Super(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   INT32* pIndexArray,
   INT32* pNextArray,
   INT32* pUniqueArray,
   INT32* pUniqueCountArray,
   HASH_MODE hashMode,
   INT64  hintSize,
   bool* pBoolFilter) {

   UINT64 numUnique = 0;

   CHashLinear<UINT32, INT32>* pHashLinear = new CHashLinear<UINT32, INT32>(hashMode);
   numUnique =
      pHashLinear->GroupBySuper(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pNextArray, pUniqueArray, pUniqueCountArray, hashMode, hintSize, pBoolFilter);
   delete pHashLinear;

   return numUnique;
}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
UINT64 GroupBy64Super(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   INT64* pIndexArray,
   INT64* pNextArray,
   INT64* pUniqueArray,
   INT64* pUniqueCountArray,
   HASH_MODE hashMode,
   INT64  hintSize,
   bool* pBoolFilter) {

   CHashLinear<UINT64, INT64>* pHashLinear = new CHashLinear<UINT64, INT64>(hashMode);
   UINT64 numUnique =
      pHashLinear->GroupBySuper(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pNextArray, pUniqueArray, pUniqueCountArray, hashMode, hintSize, pBoolFilter);

   delete pHashLinear;
   return numUnique;
}



//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
UINT64 Unique32(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,

   INT32* pIndexArray,
   INT32* pCountArray,
   HASH_MODE hashMode,
   INT64  hintSize,
   bool* pBoolFilter) {

   CHashLinear<UINT32, INT32>* pHashLinear = new CHashLinear<UINT32, INT32>(hashMode);
   UINT64 numUnique =
      pHashLinear->Unique(totalRows, totalItemSize, pInput1, pIndexArray, pCountArray, hashMode, hintSize, pBoolFilter);

   delete pHashLinear;
   return numUnique;
}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
UINT64 Unique64(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,

   INT64* pIndexArray,
   INT64* pCountArray,
   HASH_MODE hashMode,
   INT64  hintSize,
   bool* pBoolFilter) {

   CHashLinear<UINT32, INT64>* pHashLinear = new CHashLinear<UINT32, INT64>(hashMode);
   UINT64 numUnique =
      pHashLinear->Unique(totalRows, totalItemSize, pInput1, pIndexArray, pCountArray, hashMode, hintSize, pBoolFilter);

   delete pHashLinear;
   return numUnique;
}

//-----------------------------------------------------------------------------------------
void MultiKeyRollingStep2Delete(
   void* pHashLinearLast) {
   CHashLinear<UINT64, INT64>* pHashLinear;

   // If we are rolling, they will pass back what we returned
   if (pHashLinearLast) {
      pHashLinear = (CHashLinear<UINT64, INT64>*)pHashLinearLast;
      delete pHashLinear;
   }
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void* MultiKeyRollingStep2(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,

   INT64* pIndexArray,
   INT64* pRunningCountArray,
   HASH_MODE hashMode,
   INT64 hintSize,
   UINT64* numUnique,     // returned back
   void* pHashLinearLast) {

   CHashLinear<UINT64, INT64>* pHashLinear;

   // If we are rolling, they will pass back what we returned
   if (pHashLinearLast) {
      pHashLinear = (CHashLinear<UINT64, INT64>*)pHashLinearLast;
      hintSize = -1;
      LOGGING("Rolling using existing! %llu\n", pHashLinear->NumUnique);
   }
   else {
      pHashLinear  = new CHashLinear<UINT64, INT64>(hashMode);
   }

   pHashLinear->MultiKeyRolling(totalRows, totalItemSize, pInput1, pIndexArray, pRunningCountArray, hashMode, hintSize);
   *numUnique = pHashLinear->NumUnique;

   // Allow to keep rolling
   return pHashLinear;
}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void* MultiKeyHash32(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,

   INT32* pIndexArray,
   INT32* pRunningCountArray,
   INT32* pPrevArray,
   INT32* pNextArray,
   INT32* pFirstArray,
   HASH_MODE hashMode,
   INT64 hintSize,
   bool* pBoolFilter) {

   CHashLinear<UINT32, INT32>* pHashLinear = new CHashLinear<UINT32, INT32>(hashMode);
   pHashLinear->MakeHashLocationMultiKey(totalRows, totalItemSize, pInput1, pIndexArray, pRunningCountArray, pPrevArray, pNextArray, pFirstArray, hashMode, hintSize, pBoolFilter);
   delete pHashLinear;
   return NULL;
}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void* MultiKeyHash64(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,

   INT64* pIndexArray,
   INT64* pRunningCountArray,
   INT64* pPrevArray,
   INT64* pNextArray,
   INT64* pFirstArray,
   HASH_MODE hashMode,
   INT64 hintSize,
   bool* pBoolFilter) {

   CHashLinear<UINT32, INT64>* pHashLinear = new CHashLinear<UINT32, INT64>(hashMode);
   pHashLinear->MakeHashLocationMultiKey(totalRows, totalItemSize, pInput1, pIndexArray, pRunningCountArray, pPrevArray, pNextArray, pFirstArray, hashMode, hintSize, pBoolFilter);
   delete pHashLinear;
   return NULL;
}




//-----------------------------------------------------------------------------------------
// Should follow categorical size checks
//
void IsMemberHashString32Pre(
   PyArrayObject** indexArray,
   PyArrayObject* inArr1,
   INT64 size1,
   INT64 strWidth1,
   const char* pInput1,
   INT64 size2,
   INT64 strWidth2,
   const char* pInput2,
   INT8* pBooleanOutput,
   HASH_MODE hashMode,
   INT64 hintSize,
   BOOL isUnicode) {

   INT64 size = size1;
   if (size2 > size) {
      size = size2;
   }

   if (size < 100) {
      *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT8);
      INT8* pDataOut2 = (INT8*)PyArray_BYTES(*indexArray);
      IsMemberHashString32<INT8>(size1, strWidth1, (const char*)pInput1, size2, strWidth2, (const char*)pInput2, pDataOut2, pBooleanOutput, HASH_MODE(hashMode), hintSize, isUnicode);

   }
   else if (size < 30000) {
      *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT16);
      INT16* pDataOut2 = (INT16*)PyArray_BYTES(*indexArray);
      IsMemberHashString32<INT16>(size1, strWidth1, (const char*)pInput1, size2, strWidth2, (const char*)pInput2, pDataOut2, pBooleanOutput, HASH_MODE(hashMode), hintSize, isUnicode);

   }
   else if (size < 2000000000) {
      *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
      INT32* pDataOut2 = (INT32*)PyArray_BYTES(*indexArray);
      IsMemberHashString32<INT32>(size1, strWidth1, (const char*)pInput1, size2, strWidth2, (const char*)pInput2, pDataOut2, pBooleanOutput, HASH_MODE(hashMode), hintSize, isUnicode);

   }
   else {
      *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT64);
      INT64* pDataOut2 = (INT64*)PyArray_BYTES(*indexArray);
      IsMemberHashString32<INT64>(size1, strWidth1, (const char*)pInput1, size2, strWidth2, (const char*)pInput2, pDataOut2, pBooleanOutput, HASH_MODE(hashMode), hintSize, isUnicode);
   }

}



//-----------------------------------------------------------------------------------------
// Should follow categorical size checks
//
INT64 IsMemberCategoricalHashStringPre(
   PyArrayObject** indexArray,
   PyArrayObject* inArr1,
   INT64 size1,
   INT64 strWidth1,
   const char* pInput1,
   INT64 size2,
   INT64 strWidth2,
   const char* pInput2,
   HASH_MODE hashMode,
   INT64 hintSize,
   BOOL isUnicode) {

   INT64 missed = 0;

   if (size2 < 100) {
      *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT8);
      INT8* pDataOut2 = (INT8*)PyArray_BYTES(*indexArray);
      missed = IsMemberHashStringCategorical<INT8>(size1, strWidth1, (const char*)pInput1, size2, strWidth2, (const char*)pInput2, pDataOut2, HASH_MODE(hashMode), hintSize, isUnicode);

   }
   else if (size2 < 30000) {
      *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT16);
      INT16* pDataOut2 = (INT16*)PyArray_BYTES(*indexArray);
      missed = IsMemberHashStringCategorical<INT16>(size1, strWidth1, (const char*)pInput1, size2, strWidth2, (const char*)pInput2, pDataOut2, HASH_MODE(hashMode), hintSize, isUnicode);

   }
   else if (size2 < 2000000000) {
      *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
      INT32* pDataOut2 = (INT32*)PyArray_BYTES(*indexArray);
      missed = IsMemberHashStringCategorical<INT32>(size1, strWidth1, (const char*)pInput1, size2, strWidth2, (const char*)pInput2, pDataOut2, HASH_MODE(hashMode), hintSize, isUnicode);

   }
   else {
      *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT64);
      INT64* pDataOut2 = (INT64*)PyArray_BYTES(*indexArray);
      missed = IsMemberHashStringCategorical<INT64>(size1, strWidth1, (const char*)pInput1, size2, strWidth2, (const char*)pInput2, pDataOut2, HASH_MODE(hashMode), hintSize, isUnicode);

   }

   return missed;
}


//===================================================================================================

typedef void(*ISMEMBER_MK)(
   void* pHashLinearVoid,
   INT64 arraySize,
   void* pInputT,
   void* pInput2T,
   INT8* pBooleanOutput,
   void* pLocationOutputU,
   INT64 totalItemSize);

//--------------------------------------------------------------------
struct IMMK_CALLBACK {
   ISMEMBER_MK anyIMMKCallback;

   void* pHashLinearVoid;

   INT64 size1;
   void* pInput1;
   INT64 size2;         // size of the second argument
   void* pInput2;
   INT8* pBooleanOutput;
   void* pOutput;
   INT64 totalItemSize;
   INT64 typeSizeOut;

} stIMMKCallback;



//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static BOOL IMMKThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex) {
   BOOL didSomeWork = FALSE;
   IMMK_CALLBACK* Callback = (IMMK_CALLBACK*)pstWorkerItem->WorkCallbackArg;

   char* pInput1 = (char *)Callback->pInput1;
   char* pOutput = (char*)Callback->pOutput;
   INT64 lenX;
   INT64 workBlock;

   // As long as there is work to do
   while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

      INT64 inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->totalItemSize;
      INT64 outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;
      INT64 booleanAdj = pstWorkerItem->BlockSize * workBlock ;

      Callback->anyIMMKCallback(Callback->pHashLinearVoid, lenX, pInput1 + inputAdj, Callback->pInput2, Callback->pBooleanOutput + booleanAdj, pOutput + outputAdj, Callback->totalItemSize);

      // Indicate we completed a block
      didSomeWork = TRUE;

      // tell others we completed this work block
      pstWorkerItem->CompleteWorkBlock();
      //printf("|%d %d", core, (int)workBlock);
   }

   return didSomeWork;

}


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
template<typename U>
void IsMemberHashMK(
   INT64 size1,
   void* pInput1,
   INT64 size2,         // size of the second argument
   void* pInput2,
   INT8* pBooleanOutput,
   U*    pOutput,
   INT64 totalItemSize,
   INT64 hintSize,
   HASH_MODE hashMode) {

   LOGGING("ismember hash  sz1:%lld  sz2:%lld   totalitemsize:%lld  %p  %p\n", size1, size2, totalItemSize, pInput1, pInput2);

   CHashLinear<char, U>* pHashLinear = new CHashLinear<char, U>(hashMode);
   pHashLinear->MakeHashLocationMK(size2, (char*)pInput2, totalItemSize, hintSize);

   stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(size1);
   ISMEMBER_MK pFunction = IsMemberMK<char, U>;

   if (pWorkItem == NULL) {

      // Threading not allowed for this work item, call it directly from main thread
      pFunction(pHashLinear, size1, (char*)pInput1, (char*)pInput2, pBooleanOutput, pOutput, totalItemSize);

   }
   else {
      // Each thread will call this routine with the callbackArg
      pWorkItem->DoWorkCallback = IMMKThreadCallback;

      pWorkItem->WorkCallbackArg = &stIMMKCallback;

      stIMMKCallback.pHashLinearVoid = pHashLinear;
      stIMMKCallback.anyIMMKCallback = pFunction;
      stIMMKCallback.size1 = size1;
      stIMMKCallback.pInput1 = pInput1;
      stIMMKCallback.size2 = size2;
      stIMMKCallback.pInput2 = pInput2;
      stIMMKCallback.pBooleanOutput = pBooleanOutput;
      stIMMKCallback.pOutput = pOutput;
      stIMMKCallback.totalItemSize = totalItemSize;
      stIMMKCallback.typeSizeOut = sizeof(U);

      // This will notify the worker threads of a new work item
      g_cMathWorker->WorkMain(pWorkItem, size1, 0);
   }

   delete pHashLinear;

}






//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
// floats are 4 bytes and will be handled like INT32 or UIN32
void IsMemberHashMKPre(
   PyArrayObject** indexArray,
   INT64 size1,
   void* pInput1,
   INT64 size2,         // size of the second argument
   void* pInput2,
   INT8* pBooleanOutput,
   INT64 totalItemSize,
   INT64 hintSize,
   HASH_MODE hashMode) {


   INT64 size = size1;

   if (size2 > size1) {
      size = size2;
   }

   LOGGING("!!! in multikey ismember %lld  %d   size2:%lld  size1:%lld  size:%lld\n", hintSize, hashMode, size2, size1, size);

   if (size < 100) {
      *indexArray = AllocateNumpyArray(1, (npy_intp*)&size1, NPY_INT8);
      if (*indexArray) {
         INT8* pOutput = (INT8*)PyArray_BYTES(*indexArray);
         IsMemberHashMK<INT8>(size1, pInput1, size2, pInput2, pBooleanOutput, pOutput, totalItemSize, hintSize, hashMode);
      }
   }
   else if (size < 30000) {
      *indexArray = AllocateNumpyArray(1, (npy_intp*)&size1, NPY_INT16);
      if (*indexArray) {
         INT16* pOutput = (INT16*)PyArray_BYTES(*indexArray);
         IsMemberHashMK<INT16>(size1, pInput1, size2, pInput2, pBooleanOutput, pOutput, totalItemSize, hintSize, hashMode);
      }
   }
   else if (size < 2000000000) {
      *indexArray = AllocateNumpyArray(1, (npy_intp*)&size1, NPY_INT32);
      if (*indexArray) {
         INT32* pOutput = (INT32*)PyArray_BYTES(*indexArray);
         IsMemberHashMK<INT32>(size1, pInput1, size2, pInput2, pBooleanOutput, pOutput, totalItemSize, hintSize, hashMode);
      }
   }
   else {
      *indexArray = AllocateNumpyArray(1, (npy_intp*)&size1, NPY_INT64);
      if (*indexArray) {
         INT64* pOutput = (INT64*)PyArray_BYTES(*indexArray);
         IsMemberHashMK<INT64>(size1, pInput1, size2, pInput2, pBooleanOutput, pOutput, totalItemSize, hintSize, hashMode);
      }
   }
   CHECK_MEMORY_ERROR(*indexArray);
}
















//-----------------------------------------------------------------------------------------
// IsMember
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//    Fourth arg: <optional> hashSize (default to 0)
//    Returns: boolean array and optional INT32 location array
//       boolean array: True if first arg found in second arg
//       index: index location of where first arg found in second arg  (index into second arg)

/*
NOTE ON 'row' parameter
appears to take all the numbers in the a and check to see if it exists in b ?
has to be an exact match-- all the elements in row 1 of A have to match all the elements of any row in B in the same order?

>> b

b =

1.00          2.00          3.00          4.00          5.00          6.00          7.00          8.00          9.00         10.00
4.00          5.00          6.00          7.00          8.00          9.00         10.00         11.00         12.00         13.00
14.00         15.00         16.00         17.00         18.00         19.00         20.00         21.00         22.00         23.00
11.00         12.00         13.00         14.00         15.00         16.00         17.00         18.00         19.00         20.00

>> a

a =

1.00          2.00          3.00          4.00          5.00          6.00          7.00          8.00          9.00         10.00
1.00          2.00          3.00          4.00          5.00          6.00          7.00          8.00          9.00         10.00
11.00         12.00         13.00         14.00         15.00         16.00         17.00         18.00         19.00         20.00

>> [c,d]=ismember(a,b,'rows');
>> c

c =

3×1 logical array

1
1
1

>> d

d =

1.00
1.00
4.00
*/


PyObject *
IsMember32(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *inArr2 = NULL;
   int hashMode = 2;
   INT64 hintSize = 0;

   Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

   LOGGING("IsMember32 called with %lld args\n", tupleSize);

   if (tupleSize <= 1) {
      return NULL;
   }

   if (tupleSize == 2) {
      if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2)) return NULL;
   }
   else if (tupleSize == 3) {
      if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &hashMode)) return NULL;
   }
   else {
      if (!PyArg_ParseTuple(args, "O!O!iL", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2, &hashMode, &hintSize)) return NULL;

   }
   INT32 arrayType1 = PyArray_TYPE(inArr1);
   INT32 arrayType2 = PyArray_TYPE(inArr2);

   int sizeType1 = (int)NpyItemSize((PyObject*)inArr1);
   int sizeType2 = (int)NpyItemSize((PyObject*)inArr2);

   LOGGING("IsMember32 %s vs %s   size: %d  %d\n", NpyToString(arrayType1), NpyToString(arrayType2), sizeType1, sizeType2);

   switch (arrayType1) {
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

   switch (arrayType2) {
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

   if (arrayType1 != arrayType2) {

      // Arguments do not match
      PyErr_Format(PyExc_ValueError, "IsMember32 needs first arg to match %s vs %s", NpyToString(arrayType1), NpyToString(arrayType2));
      return NULL;
   }

   if (sizeType1 == 0) {
      // Weird type
      PyErr_Format(PyExc_ValueError, "IsMember32 needs a type it understands %s vs %s", NpyToString(arrayType1), NpyToString(arrayType2));
      return NULL;
   }

   if (arrayType1 == NPY_OBJECT) {
      PyErr_Format(PyExc_ValueError, "IsMember32 cannot handle unicode, object, void strings, please convert to np.chararray");
      return NULL;
   }

   INT64 arraySize1 = ArrayLength(inArr1);
   INT64 arraySize2 = ArrayLength(inArr2);

   PyArrayObject* boolArray = AllocateLikeNumpyArray(inArr1, NPY_BOOL);

   if (boolArray) {
      void* pDataIn1 = PyArray_BYTES(inArr1);
      void* pDataIn2 = PyArray_BYTES(inArr2);

      INT8* pDataOut1 = (INT8*)PyArray_BYTES(boolArray);

      PyArrayObject* indexArray = NULL;

      LOGGING("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

      if (arrayType1 >= NPY_STRING) {
         LOGGING("Calling string!\n");

         // Performance gain: if STRING and itemsize matches and itemsize is 1 or 2 --> Send to IsMemberHash32
         IsMemberHashString32Pre(&indexArray, inArr1, arraySize1, sizeType1, (const char*)pDataIn1, arraySize2, sizeType2, (const char*)pDataIn2, pDataOut1, HASH_MODE(hashMode), hintSize, arrayType1 == NPY_UNICODE);
      }
      else {
         if (arrayType1 == NPY_FLOAT32 || arrayType1 == NPY_FLOAT64) {

            LOGGING("Calling float!\n");
            sizeType1 += 100;
         }

         int dtype = NPY_INT8;

         if (arraySize2 < 100) {
            dtype = NPY_INT8;
         }
         else
         if (arraySize2 < 30000) {
            dtype = NPY_INT16;
         } else
         if (arraySize2 < 2000000000) {
            dtype = NPY_INT32;
         }
         else {
            dtype = NPY_INT64;
         }

         indexArray = AllocateLikeNumpyArray(inArr1, dtype);

         // make sure allocation succeeded
         if (indexArray) {
            void* pDataOut2 = PyArray_BYTES(indexArray);
            switch (dtype) {
            case NPY_INT8:
               IsMemberHash32<INT8>(arraySize1, pDataIn1, arraySize2, pDataIn2, (INT8*)pDataOut2, pDataOut1, sizeType1, HASH_MODE(hashMode), hintSize);
               break;
            case NPY_INT16:
               IsMemberHash32<INT16>(arraySize1, pDataIn1, arraySize2, pDataIn2, (INT16*)pDataOut2, pDataOut1, sizeType1, HASH_MODE(hashMode), hintSize);
               break;
            case NPY_INT32:
               IsMemberHash32<INT32>(arraySize1, pDataIn1, arraySize2, pDataIn2, (INT32*)pDataOut2, pDataOut1, sizeType1, HASH_MODE(hashMode), hintSize);
               break;
            case NPY_INT64:
               IsMemberHash32<INT64>(arraySize1, pDataIn1, arraySize2, pDataIn2, (INT64*)pDataOut2, pDataOut1, sizeType1, HASH_MODE(hashMode), hintSize);
               break;
            }
         }
      }

      if (indexArray) {
         PyObject* retObject = Py_BuildValue("(OO)", boolArray, indexArray);
         Py_DECREF((PyObject*)boolArray);
         Py_DECREF((PyObject*)indexArray);

         return (PyObject*)retObject;
      }
   }
   // out of memory
   return NULL;
}


/**
 * @brief
 *
 * @tparam _Index The type of the integer indices used and returned by this function. Should be INT32 or INT64.
 * @param partitionLength
 * @param pCutOffs
 * @param totalRows
 * @param totalItemSize
 * @param pInput1
 * @param coreType
 * @param pIndexArray
 * @param pFirstArrayObject
 * @param hashMode
 * @param hintSize
 * @param pBoolFilter
 * @return UINT64
 */
#if defined(__GNUC__) && __GNUC__ < 5
// Workaround for old versions of gcc which don't have enable_if_t
template <typename _Index>
#else
// removed std::is_integral due to debian compilers
template <typename _Index>
//template <typename _Index,
//   std::enable_if_t<std::is_integral<_Index>::value, int> = 0>
#endif
static UINT64 GroupByImpl(
   const INT64  partitionLength, // may be 0
   INT64* const pCutOffs,        // may be NULL
   const INT64 totalRows,
   const INT64 totalItemSize,
   const char* const pInput1,
   const int coreType,
   _Index* const pIndexArray,
   PyArrayObject** pFirstArrayObject,
   const HASH_MODE hashMode,
   const INT64   hintSize,
   bool* const pBoolFilter) {

   _Index* pFirstArray = nullptr;
   void* pHashTableAny = nullptr;
   INT64  hashTableSize = 0;

   if (partitionLength) {
      // turn off threading? or memory allocations?
      // need to pass more info
      // If this is a partitioned groupby then
      // the pIndexArray must be divided
      // the firstArray
      // the pSuperArray based on totalItemSize
      // when groupby is complete, all 0s must be kept as 0s
      // otherwise the unique count PERDAY is used this returned in another array to get the slices
      //
      // pFirstArray -- copytosmallerarray needs to change
      //
      struct PARTITION_GB {
         _Index* pFirstArray;
         void* pHashTableAny;
         INT64  HashTableSize;
         INT64  NumUnique;
         INT64  TotalRows;
      };

      // MT callback
      struct MKGBCallbackStruct {
         PARTITION_GB* pPartitions;
         INT64             PartitionLength;
         INT64* pCutOffs;

         INT64             TotalRows;
         INT64             TotalItemSize;
         const char* pInput1;

         int               CoreType;
         _Index* pIndexArray;
         HASH_MODE         HashMode;
         INT64             HintSize;
         bool* pBoolFilter;
      };

      // This is the routine that will be called back from multiple threads
      auto lambdaMKGBCallback = [](void* callbackArgT, int core, INT64 count) -> BOOL {
         auto* cb = static_cast<MKGBCallbackStruct*>(callbackArgT);

         INT64* pCutOffs = cb->pCutOffs;
         PARTITION_GB* pPartition = &cb->pPartitions[count];

         INT64             partOffset = 0;
         INT64             partLength = pCutOffs[count];
         bool* pBoolFilter = cb->pBoolFilter;
         auto* pIndexArray = cb->pIndexArray;
         const char* pInput1 = cb->pInput1;

         // use the cutoffs to calculate partition length
         if (count > 0) {
            partOffset = pCutOffs[count - 1];
         }
         partLength -= partOffset;
         pPartition->TotalRows = partLength;

         LOGGING("[%d] MKGB %lld  cutoff:%lld  offset: %lld length:%lld  hintsize:%lld\n", core, count, pCutOffs[count], partOffset, partLength, cb->HintSize);

         // NOW SHIFT THE DATA ---------------
         if (pBoolFilter) {
               pBoolFilter += partOffset;
         }
         pIndexArray += partOffset;
         pInput1 += (partOffset * cb->TotalItemSize);

         // NOW HASH the data
         pPartition->NumUnique = (INT64)
            GroupByInternal<_Index>(
               // These three are returned, they have to be deallocated
               reinterpret_cast<void**>(&pPartition->pFirstArray), &pPartition->pHashTableAny, &pPartition->HashTableSize,

               partLength,
               cb->TotalItemSize,
               pInput1,
               cb->CoreType,          // set to -1 for unknown
               pIndexArray,
               cb->HashMode,
               cb->HintSize,
               pBoolFilter);

         return TRUE;
      };

      PARTITION_GB* pPartitions = (PARTITION_GB*)WORKSPACE_ALLOC(partitionLength * sizeof(PARTITION_GB));

      // TODO: Initialize the struct using different syntax so fields which aren't meant to be modified can be marked 'const'.
      MKGBCallbackStruct stMKGBCallback;

      stMKGBCallback.pPartitions = pPartitions;
      stMKGBCallback.PartitionLength = partitionLength;
      stMKGBCallback.pCutOffs = pCutOffs;
      stMKGBCallback.TotalRows = totalRows;
      stMKGBCallback.TotalItemSize = totalItemSize;
      stMKGBCallback.pInput1 = pInput1;
      stMKGBCallback.CoreType = coreType;
      stMKGBCallback.pIndexArray = pIndexArray;
      stMKGBCallback.HashMode = HASH_MODE(hashMode);
      stMKGBCallback.HintSize = hintSize;
      stMKGBCallback.pBoolFilter = pBoolFilter;

      // turn off caching since multiple threads will allocate ----------
      g_cMathWorker->NoCaching = TRUE;

      g_cMathWorker->DoMultiThreadedWork(static_cast<int>(partitionLength), lambdaMKGBCallback, &stMKGBCallback);

      //firstArray = *stMKGBCallback.pFirstArray;
      // NOW COLLECT ALL THE RESULTS

      PyArrayObject* cutoffsArray = AllocateNumpyArray(1, (npy_intp*)&partitionLength, NPY_INT64);
      CHECK_MEMORY_ERROR(cutoffsArray);
      if (!cutoffsArray) return 0;

      INT64* pCutOffs = (INT64*)PyArray_BYTES(cutoffsArray);

      INT64 totalUniques = 0;
      for (int i = 0; i < partitionLength; i++) {
         totalUniques += pPartitions[i].NumUnique;
         pCutOffs[i] = totalUniques;
      }

      PyArrayObject* firstArray = AllocateNumpyArray(1, (npy_intp*)&totalUniques, numpy_type_code<_Index>::value);
      CHECK_MEMORY_ERROR(firstArray);
      if (!firstArray) return 0;

      _Index* pFirstArray = (_Index*)PyArray_BYTES(firstArray);

      INT64 startpos = 0;

      // Clean up------------------------------
      for (int i = 0; i < partitionLength; i++) {
         memcpy(&pFirstArray[startpos], pPartitions[i].pFirstArray, pPartitions[i].NumUnique * sizeof(_Index));
         startpos += pPartitions[i].NumUnique;
      }

      // Clean up------------------------------
      for (int i = 0; i < partitionLength; i++) {
         WorkSpaceFreeAllocLarge(pPartitions[i].pHashTableAny, pPartitions[i].HashTableSize);
      }

      // turn caching back on -----------------------------------------
      g_cMathWorker->NoCaching = FALSE;

      WORKSPACE_FREE(pPartitions);

      PyObject* pyFirstList = PyList_New(2);
      PyList_SET_ITEM(pyFirstList, 0, (PyObject*)firstArray);
      PyList_SET_ITEM(pyFirstList, 1, (PyObject*)cutoffsArray);

      *pFirstArrayObject = (PyArrayObject*)pyFirstList;
      return totalUniques;

   }
   else {

      // NOTE: because the linear is heavily optimized, it knows how to reuse a large memory allocation
      // This makes for a more complicated GroupBy as in parallel mode, it has to shut down the low level caching
      // Further, the size of the first array is not known until the unique count is known

      UINT64 numUnique =
         GroupByInternal<_Index>(

            reinterpret_cast<void**>(&pFirstArray), &pHashTableAny, &hashTableSize,

            totalRows,
            totalItemSize,
            pInput1,
            coreType,
            pIndexArray,
            hashMode,
            hintSize,
            pBoolFilter);

      // Move uniques into proper array size
      // Free HashTableAllocSize
      //printf("Got back %p %lld\n", pFirstArray, hashTableSize);
      *pFirstArrayObject = CopyToSmallerArray<_Index>(pFirstArray, numUnique, totalRows);
      WorkSpaceFreeAllocLarge(pHashTableAny, hashTableSize);
      return numUnique;
   }
}

//===================================================================================================
UINT64 GroupBy32(
   INT64  partitionLength, // may be 0
   INT64* pCutOffs,        // may be NULL
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   void* pIndexArray,
   PyArrayObject** pFirstArrayObject,
   HASH_MODE hashMode,
   INT64   hintSize,
   bool* pBoolFilter) {
   // Call the templated implementation of this function.
   using index_type = INT32;
   return GroupByImpl<index_type>(
      partitionLength, pCutOffs, totalRows, totalItemSize, pInput1, coreType,
      static_cast<index_type*>(pIndexArray), pFirstArrayObject, hashMode, hintSize, pBoolFilter);
}

//===================================================================================================
UINT64 GroupBy64(
   INT64  partitionLength, // may be 0
   INT64* pCutOffs,        // may be NULL
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   void* pIndexArray,
   PyArrayObject** pFirstArrayObject,
   HASH_MODE hashMode,
   INT64   hintSize,
   bool* pBoolFilter) {
   // Call the templated implementation of this function.
   using index_type = INT64;
   return GroupByImpl<index_type>(
      partitionLength, pCutOffs, totalRows, totalItemSize, pInput1, coreType,
      static_cast<index_type*>(pIndexArray), pFirstArrayObject, hashMode, hintSize, pBoolFilter);
}


//------------------------------------------------------------------------
// NOTE: Look at this code... fastpath for merge_asof
PyObject *MergeBinnedAndSorted(PyObject *self, PyObject *args)
{
   PyArrayObject* key1;
   PyArrayObject* key2;
   PyArrayObject* pvalArray1;
   PyArrayObject* pvalArray2;
   INT64 totalUniqueSize;

   if (!PyArg_ParseTuple(
      args, "O!O!O!O!L",
      &PyArray_Type, &key1,
      &PyArray_Type, &key2,
      &PyArray_Type, &pvalArray1,
      &PyArray_Type, &pvalArray2,
      &totalUniqueSize
   )) {

      return NULL;
   }

   LOGGING("Unique size %lld\n", totalUniqueSize);
   INT32 dtype1 = ObjectToDtype((PyArrayObject*)pvalArray1);
   INT32 dtype2 = ObjectToDtype((PyArrayObject*)pvalArray2);

   if (dtype1 < 0) {
      PyErr_Format(PyExc_ValueError, "MergeBinnedAndSorted data types are not understood dtype.num: %d vs %d", dtype1, dtype2);
      return NULL;
   }

   if (dtype1 != dtype2) {
      // Check for when numpy has 7==9 or 8==10 on Linux 5==7, 6==8 on Windows
      if (!((dtype1 <= NPY_ULONGLONG && dtype2 <= NPY_ULONGLONG) &&
            ((dtype1 & 1) == (dtype2 & 1)) &&
            PyArray_ITEMSIZE((PyArrayObject*)pvalArray1) == PyArray_ITEMSIZE((PyArrayObject*)pvalArray2))) {

         PyErr_Format(PyExc_ValueError, "MergeBinnedAndSorted data types are not the same dtype.num: %d vs %d", dtype1, dtype2);
         return NULL;
      }
   }

   void* pVal1 = PyArray_BYTES(pvalArray1);
   void* pVal2 = PyArray_BYTES(pvalArray2);
   void* pKey1 = PyArray_BYTES(key1);
   void* pKey2 = PyArray_BYTES(key2);

   PyArrayObject* indexArray = (PyArrayObject*)Py_None;
   bool isIndex32 = TRUE;
   BOOL success = FALSE;

   indexArray = AllocateLikeNumpyArray(key1, dtype1);

   if (indexArray) {
      switch (dtype1) {
      case NPY_INT8:
         success = MergePreBinned<INT8>(ArrayLength(key1), (INT8*)pKey1, pVal1, ArrayLength(key2), (INT8*)pKey2, pVal2, (INT8*)PyArray_BYTES(indexArray), totalUniqueSize, HASH_MODE_MASK, dtype1);
         break;
      case NPY_INT16:
         success = MergePreBinned<INT16>(ArrayLength(key1), (INT16*)pKey1, pVal1, ArrayLength(key2), (INT16*)pKey2, pVal2, (INT16*)PyArray_BYTES(indexArray), totalUniqueSize, HASH_MODE_MASK, dtype1);
         break;
      CASE_NPY_INT32:
         success = MergePreBinned<INT32>(ArrayLength(key1), (INT32*)pKey1, pVal1, ArrayLength(key2), (INT32*)pKey2, pVal2, (INT32*)PyArray_BYTES(indexArray), totalUniqueSize, HASH_MODE_MASK, dtype1);
         break;
      CASE_NPY_INT64:
         success = MergePreBinned<INT64>(ArrayLength(key1), (INT64*)pKey1, pVal1, ArrayLength(key2), (INT64*)pKey2, pVal2, (INT64*)PyArray_BYTES(indexArray), totalUniqueSize, HASH_MODE_MASK, dtype1);
         break;
      }
   }

   if (!success) {
      PyErr_Format(PyExc_ValueError, "MultiKeyAlign failed.  Only accepts INT32,INT64,FLOAT32,FLOAT64");
      return NULL;
   }
   return (PyObject*)indexArray;

}
