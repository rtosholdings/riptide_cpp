#include "stdafx.h"
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

// TODO: Use C++14 variable templates here once we no longer need to support
// GCC 4.x.
// template<typename T>
// static constexpr int numpy_type_code = -1;   // -1 is an invalid code,
// because if something uses this specialization we want it to break.
//
// template<>
// static constexpr int numpy_type_code<int32_t> = NPY_INT32;
//
// template<>
// static constexpr int numpy_type_code<int64_t> = NPY_INT64;

/* Template-based, compile-time mapping between C++ types and numpy type codes
 * (e.g. NPY_FLOAT64). */

template <typename T>
struct numpy_type_code
{
    static constexpr int value = -1; // -1 is an invalid code, because if something uses this
                                     // specialization we want it to break.
};

template <>
struct numpy_type_code<int32_t>
{
    static constexpr int value = NPY_INT32;
};

template <>
struct numpy_type_code<int64_t>
{
    static constexpr int value = NPY_INT64;
};

// Use this table to find a suitable hash size
int64_t PRIME_NUMBERS[] = { 53,         97,          193,         389,         769,        1543,       3079,
                            6151,       12289,       24593,       49157,       98317,      196613,     393241,
                            786433,     1572869,     3145739,     6291469,     12582917,   25165843,   50331653,
                            100663319,  201326611,   402653189,   805306457,   1610612741, 3002954501, 4294967291,
                            8589934583, 17179869143, 34359738337, 68719476731, 0 };

#define MEMCMP_NEW(ARG1, ARG2, ARG3, ARG4) \
    { \
        ARG1 = 0; \
        const char * pSrc1 = ARG2; \
        const char * pSrc2 = ARG3; \
        int64_t length = ARG4; \
        while (length >= 8) \
        { \
            uint64_t * p1 = (uint64_t *)pSrc1; \
            uint64_t * p2 = (uint64_t *)pSrc2; \
            if (*p1 != *p2) \
            { \
                ARG1 = 1; \
                length = 0; \
                break; \
            } \
            length -= 8; \
            pSrc1 += 8; \
            pSrc2 += 8; \
        } \
        if (length >= 4) \
        { \
            uint32_t * p1 = (uint32_t *)pSrc1; \
            uint32_t * p2 = (uint32_t *)pSrc2; \
            if (*p1 != *p2) \
            { \
                ARG1 = 1; \
                length = 0; \
            } \
            else \
            { \
                length -= 4; \
                pSrc1 += 4; \
                pSrc2 += 4; \
            } \
        } \
        while (length > 0) \
        { \
            if (*pSrc1 != *pSrc2) \
            { \
                ARG1 = 1; \
                break; \
            } \
            length -= 1; \
            pSrc1 += 1; \
            pSrc2 += 1; \
        } \
    }

// static inline uint32_t  sse42_crc32(const uint8_t *bytes, size_t len)
//{
//   uint32_t hash = 0;
//   int64_t i = 0;
//   for (i = 0; i<len; i++) {
//      hash = _mm_crc32_u8(hash, bytes[i]);
//   }
//   return hash;
//}

static FORCEINLINE uint64_t crchash64(const char * buf, size_t count)
{
    uint64_t h = 0;

    while (count >= 8)
    {
        h = _mm_crc32_u64(h, *(uint64_t *)buf);
        count -= 8;
        buf += 8;
    }

    uint64_t t = 0;

    switch (count)
    {
    case 7:
        t ^= (uint64_t)buf[6] << 48;
    case 6:
        t ^= (uint64_t)buf[5] << 40;
    case 5:
        t ^= (uint64_t)buf[4] << 32;
    case 4:
        t ^= (uint64_t)buf[3] << 24;
    case 3:
        t ^= (uint64_t)buf[2] << 16;
    case 2:
        t ^= (uint64_t)buf[1] << 8;
    case 1:
        t ^= (uint64_t)buf[0];
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
#define mix(h) \
    h ^= h >> 23; \
    h *= 0x2127599bf4325c37ULL; \
    h ^= h >> 47;

static FORCEINLINE uint64_t fasthash64(const void * buf, size_t len)
{
    uint64_t seed = 0;
    const uint64_t m = 0x880355f21e6d1965ULL;
    const uint64_t * pos = (const uint64_t *)buf;
    const unsigned char * pos2;
    uint64_t h = seed ^ (len * m);
    uint64_t v;

    while (len >= 8)
    {
        v = *pos++;
        h ^= mix(v);
        h *= m;
        len -= 8;
    }

    pos2 = (const unsigned char *)pos;
    v = 0;

    switch (len)
    {
    case 7:
        v ^= (uint64_t)pos2[6] << 48;
    case 6:
        v ^= (uint64_t)pos2[5] << 40;
    case 5:
        v ^= (uint64_t)pos2[4] << 32;
    case 4:
        v ^= (uint64_t)pos2[3] << 24;
    case 3:
        v ^= (uint64_t)pos2[2] << 16;
    case 2:
        v ^= (uint64_t)pos2[1] << 8;
    case 1:
        v ^= (uint64_t)pos2[0];
        h ^= mix(v);
        h *= m;
    }

    return mix(h);
}

// --------------------------------------------
// Used when we know 8byte hash
FORCEINLINE uint64_t fasthash64_8(uint64_t v)
{
    uint64_t seed = 0;
    const uint64_t m = 0x880355f21e6d1965ULL;
    uint64_t h = seed ^ m;

    h ^= mix(v);
    h *= m;

    return mix(h);
}

// --------------------------------------------
// Used when we know 8byte hash
FORCEINLINE uint64_t fasthash64_16(uint64_t * v)
{
    uint64_t seed = 0;
    const uint64_t m = 0x880355f21e6d1965ULL;
    uint64_t h = seed ^ m;

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

#define HASH_STRING() \
    const char * strStart = (pHashList + (i * strWidth)); \
    const char * str = strStart; \
    const char * strEnd = str + strWidth; \
    while (*str != 0) \
        if (++str >= strEnd) \
            break; \
    uint64_t hash = fasthash64(strStart, str - strStart);

#define HASH_UNICODE() \
    const char * strStart = (pHashList + (i * strWidth)); \
    const uint32_t * str = (uint32_t *)strStart; \
    const uint32_t * strEnd = str + strWidth / 4; \
    uint32_t hash = 0; \
    uint32_t c; \
    while ((c = *str) != 0) \
    { \
        str++; \
        hash = _mm_crc32_u32(hash, c); \
        if (str >= strEnd) \
            break; \
    }

FORCE_INLINE
bool UNICODE_MATCH(const char * str1T, const char * str2T, int64_t strWidth1)
{
    const uint32_t * str1 = (const uint32_t *)str1T;
    const uint32_t * str2 = (const uint32_t *)str2T;
    strWidth1 /= 4;
    while (strWidth1 > 0)
    {
        if (*str1 != *str2)
            return false;
        ++str1;
        ++str2;
        --strWidth1;
    }
    return true;
}

FORCE_INLINE
bool STRING_MATCH(const char * str1, const char * str2, int64_t strWidth1)
{
    while (strWidth1 > 0)
    {
        if (*str1 != *str2)
            return false;
        ++str1;
        ++str2;
        --strWidth1;
    }
    return true;
}

FORCE_INLINE
bool UNICODE_MATCH2(const char * str1T, const char * str2T, int64_t strWidth1, int64_t strWidth2)
{
    const uint32_t * str1 = (const uint32_t *)str1T;
    const uint32_t * str2 = (const uint32_t *)str2T;
    strWidth1 /= 4;
    strWidth2 /= 4;
    while (1)
    {
        // Check for when one string ends and the other has not yet
        if (strWidth1 == 0)
        {
            if (*str2 == 0)
                return true;
            return false;
        }
        if (strWidth2 == 0)
        {
            if (*str1 == 0)
                return true;
            return false;
        }

        if (*str1 != *str2)
            return false;
        ++str1;
        ++str2;
        --strWidth1;
        --strWidth2;
    }
    return true;
}

FORCE_INLINE
bool STRING_MATCH2(const char * str1, const char * str2, int64_t strWidth1, int64_t strWidth2)
{
    while (1)
    {
        // Check for when one string ends and the other has not yet
        if (strWidth1 == 0)
        {
            if (*str2 == 0)
                return true;
            return false;
        }
        if (strWidth2 == 0)
        {
            if (*str1 == 0)
                return true;
            return false;
        }

        if (*str1 != *str2)
            return false;
        ++str1;
        ++str2;
        --strWidth1;
        --strWidth2;
    }
    return true;
}

template <typename T, typename U>
int64_t CHashLinear<T, U>::GetHashSize(int64_t numberOfEntries)
{
    // Check for perfect hashes first when type size is small
    // if (sizeof(T) == 1) {
    //   return 256;
    //}

    // if (sizeof(T) == 2) {
    //   return 65536;
    //}

    if (HashMode == HASH_MODE_PRIME)
    {
        int i = 0;
        while (PRIME_NUMBERS[i] != 0)
        {
            if (PRIME_NUMBERS[i] > numberOfEntries)
            {
                return PRIME_NUMBERS[i];
            }
            i++;
        }
        LogError("**Failed to find prime number for hash size %lld\n", numberOfEntries);
        return 0;
    }
    else
    {
        // Power of 2 search
        int i = 0;
        while ((1LL << i) < numberOfEntries)
            i++;

        int64_t result = (1LL << i);
        int64_t maxhashsize = (1LL << 31);

        // TODO: really need to change strategies if high unique count and high row
        // count
        if (result > maxhashsize)
        {
            result = maxhashsize;
        }
        return result;
    }
}

//-----------------------------------------------
template <typename T, typename U>
void CHashLinear<T, U>::FreeMemory(bool forceDeallocate)
{
    if (forceDeallocate || Deallocate)
    {
        // printf("deallocating\n");
        WorkSpaceFreeAllocLarge(pHashTableAny, HashTableAllocSize);
    }
    else
    {
        // printf("not deallocating\n");
    }

    void * pTemp = pBitFields;
    WorkSpaceFreeAllocSmall(pTemp, BitAllocSize);
    pBitFields = NULL;
}

//-----------------------------------------------
template <typename T, typename U>
void CHashLinear<T, U>::AllocAndZeroBitFields(uint64_t hashSize)
{
    // Calculate how many bytes we need (in 64 bit/8 byte increments)
    BitAllocSize = (HashSize + 63) / 64;
    BitAllocSize *= sizeof(uint64_t);

    pBitFields = (uint64_t *)WorkSpaceAllocSmall(BitAllocSize);

    if (pBitFields)
    {
        // Fill with zeros to indicate no hash position is used yet
        RtlZeroMemory(pBitFields, BitAllocSize);
    }
}

//-----------------------------------------------
// returns pointer to allocated array
template <typename T, typename U>
char * CHashLinear<T, U>::AllocHashTable(size_t allocSize)
{
    HashTableAllocSize = allocSize;
    pHashTableAny = WorkSpaceAllocLarge(HashTableAllocSize);

    return (char *)pHashTableAny;

    //// Do not zero it
    ////RtlZeroMemory(pHashTableAny, HashTableAllocSize);
}

//-----------------------------------------------
// Allocates based on HashMode
// and size of structure
//
// Returns: pointer to extra section if requested
// Returns NULL does not mean failure if sizeofExtraArray=0
template <typename T, typename U>
void * CHashLinear<T, U>::AllocMemory(int64_t numberOfEntries, int64_t sizeofStruct, int64_t sizeofExtraArray, bool isFloat)
{
    if (sizeofStruct == -1)
    {
        sizeofStruct = sizeof(SingleKeyEntry);
    }
    if (sizeofStruct == -2)
    {
        sizeofStruct = sizeof(MultiKeyEntry);
    }

    NumEntries = numberOfEntries;
    // HashSize= GetHashSize(NumEntries*2);

    if (HashMode == HASH_MODE_MASK)
    {
        // For float *8 seems to help
        // For INT, it does not
        if (isFloat)
        {
            HashSize = GetHashSize(NumEntries * 8);
        }
        else
        {
            HashSize = GetHashSize(NumEntries * 2);
        }

        // Helps avoid collisions for low number of strings
        LOGGING("Checking to up HashSize %llu  %llu  %llu\n", HashSize, sizeof(T), sizeof(U));

        // NOTE for 1 byte AND 2 bytes we try to do a perfect hash
        if (HashSize < 65536)
            HashSize = 65536;
    }
    else
    {
        HashSize = GetHashSize(NumEntries * 2);
    }

    size_t allocSize = HashSize * sizeofStruct;

    FreeMemory(true);

    LOGGING("Hash size selected %llu for NumEntries %llu -- allocation size %llu\n", HashSize, NumEntries, allocSize);

    AllocAndZeroBitFields(HashSize);

    if (pBitFields)
    {
        if (sizeofExtraArray)
        {
            // 128 byte padding
            int64_t padSize = (allocSize + 127) & ~127;
            char * pRootBuffer = AllocHashTable(padSize + sizeofExtraArray);

            if (pRootBuffer)
            {
                // return pointer to extra section
                return pRootBuffer + padSize;
            }
            else
            {
                // memory fail
                CHECK_MEMORY_ERROR(NULL);
                return NULL;
            }
        }
        else
        {
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
FORCE_INLINE void CHashLinear<T, U>::InternalGetLocation(U i, HashLocation * pLocation, int8_t * pBoolOutput, U * pLocationOutput,
                                                         T item, uint64_t hash)
{
    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        if (pLocation[hash].value == item)
        {
            // return the first location
            pLocationOutput[i] = pLocation[hash].Location;
            pBoolOutput[i] = 1;
            return;
        }

        // Linear goes to next position
        if (++hash >= HashSize)
        {
            hash = 0;
        }
    }
    // Not found
    pLocationOutput[i] = BAD_INDEX;
    pBoolOutput[i] = 0;
}

//-----------------------------------------------
// looks for the index of set location
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE void CHashLinear<T, U>::InternalGetLocationCategorical(U i, HashLocation * pLocation, U * pLocationOutput, T item,
                                                                    uint64_t hash, int64_t * missed)
{
    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        if (pLocation[hash].value == item)
        {
            // return the first location
            pLocationOutput[i] = pLocation[hash].Location + 1;
            return;
        }

        // Linear goes to next position
        if (++hash >= HashSize)
        {
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
FORCE_INLINE void CHashLinear<T, U>::InternalSetLocation(U i, HashLocation * pLocation, T item, uint64_t hash)
{
    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        if (pLocation[hash].value == item)
        {
            // Duplicate
            return;
        }

        // This entry is not us so we must have collided
        //++NumCollisions;

        // Linear goes to next position
        if (++hash >= HashSize)
        {
            hash = 0;

            // if (NumCollisions > (HashSize * 2)) {
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
    while (1) \
    { \
        if (pBitFields[hash >> 6] & (1LL << (hash & 63))) \
        { \
            /* Check if we have a match from before */ \
            if (pLocation[hash].value == item) \
            { \
                /* Duplicate */ \
                break; \
            } \
            /* This entry is not us so we must have collided */ \
            /* Linear goes to next position */ \
            if (++hash >= HashSize) \
            { \
                hash = 0; \
            } \
        } \
        else \
        { \
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
template <typename T, typename U>
void CHashLinear<T, U>::MakeHashLocationMK(int64_t arraySize, T * pInput, int64_t totalItemSize, int64_t hintSize)
{
    if (hintSize == 0)
    {
        hintSize = arraySize;
    }

    AllocMemory(hintSize, sizeof(HashLocationMK), 0, false);
    // uint64_t NumUnique = 0;
    // uint64_t NumCollisions = 0;

    HashLocationMK * pLocation = (HashLocationMK *)pHashTableAny;

    uint64_t * pBitFields = this->pBitFields;

    if (! pLocation || ! pBitFields)
    {
        return;
    }

    LOGGING(
        "in mkhashlocationmk asize: %llu   isize: %llu  HashSize:%lld  "
        "sizeofU:%d\n",
        arraySize, totalItemSize, (long long)HashSize, (int)sizeof(U));

    for (U i = 0; i < arraySize; i++)
    {
        const char * pMatch = pInput + (totalItemSize * i);
        uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);
        hash = hash & (HashSize - 1);

        // printf("[%d] \n", (int)i);

        while (1)
        {
            uint64_t index = hash >> 6;
            if (pBitFields[index] & (1LL << (hash & 63)))
            {
                // Check if we have a match from before
                U Last = pLocation[hash].Location;
                const char * pMatch2 = pInput + (totalItemSize * Last);

                // TJD: August 2018 --unrolled MEMCMP and got 10-15% speedup
                const char * pSrc1 = pMatch;
                const char * pSrc2 = pMatch2;
                int64_t length = totalItemSize;

                while (length >= 8)
                {
                    uint64_t * p1 = (uint64_t *)pSrc1;
                    uint64_t * p2 = (uint64_t *)pSrc2;

                    if (*p1 != *p2)
                        goto FAIL_MATCH;
                    length -= 8;
                    pSrc1 += 8;
                    pSrc2 += 8;
                }
                if (length >= 4)
                {
                    uint32_t * p1 = (uint32_t *)pSrc1;
                    uint32_t * p2 = (uint32_t *)pSrc2;

                    if (*p1 != *p2)
                        goto FAIL_MATCH;
                    length -= 4;
                    pSrc1 += 4;
                    pSrc2 += 4;
                }

                while (length > 0)
                {
                    if (*pSrc1 != *pSrc2)
                        goto FAIL_MATCH;
                    length -= 1;
                    pSrc1 += 1;
                    pSrc2 += 1;
                }

                // printf("[%d] matched \n", (int)i);
                // Matched
                break;

            FAIL_MATCH:
                // printf("[%d] collided \n", (int)i);

                // This entry is not us so we must have collided
                //++NumCollisions;

                // Linear goes to next position
                if (++hash >= HashSize)
                {
                    hash = 0;
                }
                continue;
            }

            // Failed to find hash
            // printf("[%d] fail \n", (int)i);
            pBitFields[hash >> 6] |= (1LL << (hash & 63));

            //++NumUnique;
            pLocation[hash].Location = i;
            break;
        }
    }

    // LOGGING("%llu entries had %llu collisions   %llu unique\n", arraySize,
    // NumCollisions, NumUnique); printf("%llu entries had %llu collisions   %llu
    // unique\n", arraySize, NumCollisions, NumUnique);
}

//-----------------------------------------------
// stores the index of the last matching index
// ASSUMES that pVal1 and pVal2 are SORTED!!
//  U is int32_t or int64_t
//  V is int32_t, int64_t, float, or double
template <typename U, typename V>
void FindLastMatchCategorical(int64_t arraySize1, int64_t arraySize2, U * pKey1, U * pKey2, V * pVal1, V * pVal2,
                              U * pLocationOutput, int64_t totalUniqueSize)
{
    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));
    U * pLocation = (U *)WORKSPACE_ALLOC(totalUniqueSize * sizeof(U));

    for (U k = 0; k < totalUniqueSize; k++)
    {
        pLocation[k] = -1;
    }

    U i = 0;
    U j = 0;
    while (i < arraySize1 && j < arraySize2)
    {
        if (pVal1[i] < pVal2[j])
        {
            // key1 is first
            U lookup = pKey1[i];

            if (pLocation[lookup] != -1)
            {
                // We have a match from before, update with key1
                pLocationOutput[i] = pLocation[lookup];
            }
            else
            {
                // failed to match
                pLocationOutput[i] = BAD_INDEX;
            }
            i++;
        }
        else
        {
            // key2 is first
            U lookup = pKey2[i];
            pLocation[lookup] = j;
            j++;
        }
    }

    while (i < arraySize1)
    {
        U lookup = pKey1[i];

        if (pLocation[lookup] != -1)
        {
            pLocationOutput[i] = pLocation[lookup];
        }
        else
        {
            pLocationOutput[i] = BAD_INDEX;
        }
        i++;
    }

    WORKSPACE_FREE(pLocation);
}

//-----------------------------------------------
// stores the index of the last matching index
// ASSUMES that pVal1 and pVal2 are SORTED!!
template <typename T, typename U>
template <typename V>
void CHashLinear<T, U>::FindLastMatchMK(int64_t arraySize1, int64_t arraySize2, T * pKey1, T * pKey2, V * pVal1, V * pVal2,
                                        U * pLocationOutput, int64_t totalItemSize, bool allowExact)
{
    AllocMemory(arraySize2, sizeof(HashLocationMK), 0, false);

    HashLocationMK * pLocation = (HashLocationMK *)pHashTableAny;

    if (! pLocation || ! pBitFields)
    {
        return;
    }

    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    U i = 0;
    U j = 0;
    while (i < arraySize1 && j < arraySize2)
    {
        if (pVal1[i] < pVal2[j] || (! allowExact && pVal1[i] == pVal2[j]))
        {
            const char * pMatch1 = pKey1 + (totalItemSize * i);
            uint64_t hash = DEFAULT_HASH64(pMatch1, totalItemSize);
            hash = hash & (HashSize - 1);

            // TODO: should maybe put begin .. end into function implementing lookup
            // for hashmap begin
            while (1)
            {
                if (IsBitSet(hash))
                {
                    // Check if we have a match from before
                    U Last = pLocation[hash].Location;
                    const char * pMatch2 = pKey2 + (totalItemSize * Last);
                    int mresult;
                    MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
                    if (! mresult)
                    {
                        pLocationOutput[i] = pLocation[hash].Location;
                        break;
                    }

                    // Linear goes to next position
                    if (++hash >= HashSize)
                    {
                        hash = 0;
                    }
                    continue;
                }
                // Not found
                pLocationOutput[i] = BAD_INDEX;
                break;
            }
            // end
            i++;
        }
        else
        {
            const char * pMatch1 = pKey2 + (totalItemSize * j);
            uint64_t hash = DEFAULT_HASH64(pMatch1, totalItemSize);
            hash = hash & (HashSize - 1);

            // TODO: should maybe start .. end into a function implementing insertion
            // for hashmap begin
            while (1)
            {
                if (IsBitSet(hash))
                {
                    // Check if we have a match from before
                    U Last = pLocation[hash].Location;
                    const char * pMatch2 = pKey2 + (totalItemSize * Last);
                    int mresult;
                    MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
                    if (! mresult)
                    {
                        pLocation[hash].Location = j;
                        break;
                    }

                    // Linear goes to next position
                    if (++hash >= HashSize)
                    {
                        hash = 0;
                    }
                    continue;
                }
                // Failed to find hash
                SetBit(hash);
                pLocation[hash].Location = j;
                break;
            }
            // end

            j++;
        }
    }
    while (i < arraySize1)
    {
        const char * pMatch1 = pKey1 + (totalItemSize * i);
        uint64_t hash = DEFAULT_HASH64(pMatch1, totalItemSize);
        hash = hash & (HashSize - 1);

        // TODO: should maybe put begin .. end into function implementing lookup for
        // hashmap begin
        while (1)
        {
            if (IsBitSet(hash))
            {
                // Check if we have a match from before
                U Last = pLocation[hash].Location;
                const char * pMatch2 = pKey2 + (totalItemSize * Last);
                int mresult;
                MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
                if (! mresult)
                {
                    // return the first location
                    pLocationOutput[i] = pLocation[hash].Location;
                    break;
                }

                // Linear goes to next position
                if (++hash >= HashSize)
                {
                    hash = 0;
                }
                continue;
            }
            // Not found
            // pLocationOutput[i] = i;
            pLocationOutput[i] = BAD_INDEX;
            break;
        }
        // end
        i++;
    }
}

//-----------------------------------------------
// stores the index of the next matching index
// ASSUMES that pVal1 and pVal2 are SORTED!!
// TODO: Clean this up and merge with FindLastMatchMk
template <typename T, typename U>
template <typename V>
void CHashLinear<T, U>::FindNextMatchMK(int64_t arraySize1, int64_t arraySize2, T * pKey1, T * pKey2, V * pVal1, V * pVal2,
                                        U * pLocationOutput, int64_t totalItemSize, bool allowExact)
{
    AllocMemory(arraySize2, sizeof(HashLocationMK), 0, false);
    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    HashLocationMK * pLocation = (HashLocationMK *)pHashTableAny;
    if (! pLocation || ! pBitFields)
    {
        return;
    }

    U i = (U)(arraySize1 - 1);
    U j = (U)(arraySize2 - 1);
    while (i >= 0 && j >= 0)
    {
        if (pVal1[i] > pVal2[j] || (! allowExact && pVal1[i] == pVal2[j]))
        {
            const char * pMatch1 = pKey1 + (totalItemSize * i);
            uint64_t hash = DEFAULT_HASH64(pMatch1, totalItemSize);
            hash = hash & (HashSize - 1);

            // TODO: should maybe put begin .. end into function implementing lookup
            // for hashmap begin
            while (1)
            {
                if (IsBitSet(hash))
                {
                    // Check if we have a match from before
                    U Last = pLocation[hash].Location;
                    const char * pMatch2 = pKey2 + (totalItemSize * Last);
                    int mresult;
                    MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
                    if (! mresult)
                    {
                        pLocationOutput[i] = pLocation[hash].Location;
                        break;
                    }

                    // Linear goes to next position
                    if (++hash >= HashSize)
                    {
                        hash = 0;
                    }
                    continue;
                }
                // Not found
                pLocationOutput[i] = BAD_INDEX;
                break;
            }
            // end
            i--;
        }
        else
        {
            const char * pMatch1 = pKey2 + (totalItemSize * j);
            uint64_t hash = DEFAULT_HASH64(pMatch1, totalItemSize);
            hash = hash & (HashSize - 1);

            // TODO: should maybe start .. end into a function implementing insertion
            // for hashmap begin
            while (1)
            {
                if (IsBitSet(hash))
                {
                    // Check if we have a match from before
                    U Last = pLocation[hash].Location;
                    const char * pMatch2 = pKey2 + (totalItemSize * Last);
                    int mresult;
                    MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
                    if (! mresult)
                    {
                        pLocation[hash].Location = j;
                        break;
                    }

                    // Linear goes to next position
                    if (++hash >= HashSize)
                    {
                        hash = 0;
                    }
                    continue;
                }
                // Failed to find hash
                SetBit(hash);
                pLocation[hash].Location = j;
                break;
            }
            // end

            j--;
        }
    }
    while (i >= 0)
    {
        const char * pMatch1 = pKey1 + (totalItemSize * i);
        uint64_t hash = DEFAULT_HASH64(pMatch1, totalItemSize);
        hash = hash & (HashSize - 1);

        // TODO: should maybe put begin .. end into function implementing lookup for
        // hashmap begin
        while (1)
        {
            if (IsBitSet(hash))
            {
                // Check if we have a match from before
                U Last = pLocation[hash].Location;
                const char * pMatch2 = pKey2 + (totalItemSize * Last);
                int mresult;
                MEMCMP_NEW(mresult, pMatch1, pMatch2, totalItemSize);
                if (! mresult)
                {
                    // return the first location
                    pLocationOutput[i] = pLocation[hash].Location;
                    break;
                }

                // Linear goes to next position
                if (++hash >= HashSize)
                {
                    hash = 0;
                }
                continue;
            }
            // Not found
            // pLocationOutput[i] = i;
            pLocationOutput[i] = BAD_INDEX;
            break;
        }
        // end
        i--;
    }
}

//-----------------------------------------------
// T is the input type
// U is the index output type (int8/16/32/64)
//
// outputs bool array
// outputs location array
//
template <typename T, typename U>
static void IsMemberMK(void * pHashLinearVoid, int64_t arraySize, void * pInputT, void * pInput2T, int8_t * pBoolOutput,
                       void * pLocationOutputU, int64_t totalItemSize)
{
    struct HashLocationMK
    {
        U Location;
    };

    CHashLinear<T, U> * pHashLinear = (CHashLinear<T, U> *)pHashLinearVoid;

    HashLocationMK * pLocation = (HashLocationMK *)pHashLinear->pHashTableAny;

    U * pLocationOutput = (U *)pLocationOutputU;
    T * pInput = (T *)pInputT;
    T * pInput2 = (T *)pInput2T;

    uint64_t HashSize = pHashLinear->HashSize;

    // to determine if hash location has been visited
    uint64_t * pBitFields = pHashLinear->pBitFields;
    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    LOGGING("in ismembermk  asize: %llu   isize: %llu  %p   %p  %p\n", arraySize, totalItemSize, pBitFields, pLocationOutput,
            pBoolOutput);

    for (U i = 0; i < arraySize; i++)
    {
        const char * pMatch = pInput + (totalItemSize * i);
        uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);
        hash = hash & (HashSize - 1);

        // printf("[%d] %llu\n", (int)i, hash);

        while (1)
        {
            uint64_t index = hash >> 6;
            if (pBitFields[index] & (1LL << (hash & 63)))
            {
                // Check if we have a match from before
                U Last = pLocation[hash].Location;
                const char * pMatch2 = pInput2 + (totalItemSize * Last);
                int mresult;
                MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                if (! mresult)
                {
                    // return the first location
                    pLocationOutput[i] = pLocation[hash].Location;
                    pBoolOutput[i] = 1;
                    break;
                }

                // Linear goes to next position
                if (++hash >= HashSize)
                {
                    hash = 0;
                }
                // printf("[%d] continue\n", (int)i);
                continue;
            }
            // printf("[%d] Not found\n", (int)i);
            // Not found
            pLocationOutput[i] = BAD_INDEX;
            pBoolOutput[i] = 0;
            break;
        }
    }

    LOGGING("Done with ismembermk\n");
}

//=============================================
#define HASH_INT1 \
    int64_t h = (int64_t)item; \
    h = h % HashSize;

//#define HASH_int32_t  uint64_t h = (uint64_t)item; h ^= (h >> 16); h= h &
//(HashSize-1); #define HASH_int32_t uint64_t h= fasthash64_8(item) &
//(HashSize-1);
#define HASH_int32_t uint64_t h = _mm_crc32_u32(0, (uint32_t)item) & (HashSize - 1);

//#define HASH_int64_t  uint64_t h= _mm_crc32_u64(0, item) & (HashSize-1);
#define HASH_int64_t uint64_t h = fasthash64_8(item) & (HashSize - 1);

#define HASH_INT128 uint64_t h = fasthash64_16((uint64_t *)&pInput[i]) & (HashSize - 1);

//=============================================
//#define HASH_FLOAT1  h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *=
// 0xc2b2ae35; h ^= h >> 16;   h = h % HashSize; #define HASH_FLOAT2  h ^= h >>
// 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16;   h = h &
//(HashSize - 1); #define HASH_FLOAT3  h ^= h >> 33; h *= 0xff51afd7ed558ccd; h
//^= h >> 33; h *= 0xc4ceb9fe1a85ec53; h ^= h >> 33; h = h % HashSize; #define
// HASH_FLOAT4  h ^= h >> 33; h *= 0xff51afd7ed558ccd; h ^= h >> 33; h *=
// 0xc4ceb9fe1a85ec53; h ^= h >> 33; h = h & (HashSize - 1);

//------------------
// WHEN MULT SIZE *4
// 28000000 entries had 1018080 collisions   22388609 unique
// Elapsed time 5.043842 seconds.
//--------------------------------------------------
// WHEN MULT SIZE *2
// 28000000 entries had 2311199 collisions   22388609 unique
// Elapsed time 4.661492 seconds.

//#define HASH_FLOAT1   h = h % HashSize;
//#define HASH_FLOAT2  h ^= h >> 3; h = h & (HashSize - 1);
//#define HASH_FLOAT1  h ^= (h >> 20) ^ (h >> 12) ^ (h >> 7) ^ (h >> 4); h = h %
// HashSize;
// SCORE 5 seconds #define HASH_FLOAT1  h ^= (h >> 23) ^ (h << 32); h = h %
// HashSize;
#define HASH_FLOAT1 \
    h ^= (h >> 20); \
    h = h % HashSize;
//#define HASH_FLOAT2  h ^= (h >> 23); h = h & (HashSize - 1);
// -- NOT BAD 11 #define HASH_FLOAT2  h ^= (h >> 23); h = h & (HashSize - 1);
// -- NOT BAD 10.9 #define HASH_FLOAT2  h ^= (h >> 16); h = h & (HashSize - 1);
#define HASH_FLOAT2 \
    h ^= (h >> 20); \
    h = h & (HashSize - 1);
#define HASH_FLOAT3 \
    h ^= h >> 32; \
    h = h % HashSize;

//------------------------------------------------------------------------------
// hash for 64 bit float
//#define HASH_FLOAT4  h ^= h >> 32; h = h & (HashSize - 1);
#define HASH_FLOAT4 h = fasthash64_8(h) & (HashSize - 1);
//#define HASH_FLOAT4  h ^= (h >> 44) ^ (h >> 32) ^ (h >> 17) ^ (h >> 4); h = h
//& (HashSize - 1);

//-----------------------------------------------
// stores the index of the first location
//
template <typename T, typename U>
void CHashLinear<T, U>::MakeHashLocation(int64_t arraySize, T * pHashList, int64_t hintSize)
{
    if (hintSize == 0)
    {
        hintSize = arraySize;
    }

    AllocMemory(hintSize, sizeof(HashLocation), 0, false);
    NumUnique = 0;

    LOGGING("MakeHashLocation: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

    HashLocation * pLocation = (HashLocation *)pHashTableAny;
    uint64_t * pBitFields = this->pBitFields;

    if (! pLocation || ! pBitFields)
    {
        return;
    }

    LOGGING(
        "IsMember index size is %zu    output size is %zu  HashSize is %llu  "
        "  hashmode %d\n",
        sizeof(U), sizeof(T), HashSize, (int)HashMode);

    // for (int i = 0; i < ((HashSize + 63) / 64); i++) {
    //   printf("%llu |", pBitFields[i]);
    //}

    if (sizeof(T) <= 2)
    {
        for (U i = 0; i < arraySize; i++)
        {
            // perfect hash
            T item = pHashList[i];
            uint64_t hash = item;
            INTERNAL_SET_LOCATION;
            // InternalSetLocation(i, pLocation, item, item);
        }
    }
    else
    {
        if (sizeof(T) == 4)
        {
            if (HashMode == HASH_MODE_PRIME)
            {
                for (U i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_INT1;
                    uint64_t hash = h;
                    INTERNAL_SET_LOCATION;
                    // InternalSetLocation(i, pLocation, item, h);
                }
            }
            else
            {
                for (U i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_int32_t;
                    uint64_t hash = h;
                    INTERNAL_SET_LOCATION;
                    // InternalSetLocation(i, pLocation, item, h);
                }
            }
        }
        else if (sizeof(T) == 8)
        {
            if (HashMode == HASH_MODE_PRIME)
            {
                for (U i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_INT1;
                    uint64_t hash = h;
                    INTERNAL_SET_LOCATION;
                    // InternalSetLocation(i, pLocation, item, h);
                }
            }
            else
            {
                for (U i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_int64_t;
                    uint64_t hash = h;
                    INTERNAL_SET_LOCATION;
                    // InternalSetLocation(i, pLocation, item, h);
                }
            }
        }
        else
        {
            printf("!!! MakeHashLocation -- hash item size is not 1,2,4, or 8!  %zu\n", sizeof(T));
        }
    }
    LOGGING("%llu entries had %llu collisions   %llu unique\n", arraySize, NumCollisions, NumUnique);

    LOGGING("IsMember index size is %zu    output size is %zu  HashSize is %llu\n", sizeof(U), sizeof(T), HashSize);

    // for (int i = 0; i < ((HashSize + 63) / 64); i++) {
    //   printf("%llu |", pBitFields[i]);
    //}
    // printf("\n");

    // printf("%llu entries had %llu collisions   %llu unique\n", arraySize,
    // NumCollisions, NumUnique);
}

//-----------------------------------------------
// outputs bool array
// outputs location array
//
template <typename T, typename U>
int64_t CHashLinear<T, U>::IsMemberCategorical(int64_t arraySize, T * pHashList, U * pLocationOutput)
{
    HashLocation * pLocation = (HashLocation *)pHashTableAny;
    int64_t missed = 0;
    if (sizeof(T) <= 2)
    {
        // perfect hash
        for (U i = 0; i < arraySize; i++)
        {
            T item = pHashList[i];
            InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, item, &missed);
        }
    }
    else
    {
        if (sizeof(T) == 4)
        {
            if (HashMode == HASH_MODE_PRIME)
            {
                for (U i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_INT1;
                    InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
                }
            }
            else
            {
                for (U i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_int32_t;
                    InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
                }
            }
        }
        else if (sizeof(T) == 8)
        {
            if (HashMode == HASH_MODE_PRIME)
            {
                for (U i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_INT1;
                    InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
                }
            }
            else
            {
                for (U i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_int64_t;
                    InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
                }
            }
        }
        else
        {
            printf("!!IsMemberCategorical hash size error! %zu\n", sizeof(T));
        }
    }
    return missed;
}

#define INNER_GET_LOCATION_PERFECT \
    if (pBitFields[hash >> 6] & (1LL << (hash & 63))) \
    { \
        pLocationOutput[i] = pLocation[hash].Location; \
        pBoolOutput[i] = 1; \
    } \
    else \
    { \
        /* Not found */ \
        pLocationOutput[i] = BAD_INDEX; \
        pBoolOutput[i] = 0; \
    }

#define INNER_GET_LOCATION \
    while (1) \
    { \
        if (pBitFields[hash >> 6] & (1LL << (hash & 63))) \
        { \
            /* Check if we have a match from before*/ \
            if (pLocation[hash].value == item) \
            { \
                /* return the first location */ \
                pLocationOutput[i] = pLocation[hash].Location; \
                pBoolOutput[i] = 1; \
                break; \
            } \
            /* Linear goes to next position */ \
            if (++hash >= HashSize) \
            { \
                hash = 0; \
            } \
            continue; \
        } \
        else \
        { \
            /* Not found */ \
            pLocationOutput[i] = BAD_INDEX; \
            pBoolOutput[i] = 0; \
            break; \
        } \
    }

//-----------------------------------------------
// T is the input type byte/float32/float64/int8/uint8/int16/uint16/int32/...
// U is the index output type (int8/16/32/64)
//
// outputs bool array
// outputs location array
//
template <typename T, typename U>
void IsMember(void * pHashLinearVoid, int64_t arraySize, void * pHashListT, int8_t * pBoolOutput, void * pLocationOutputU)
{
    struct HashLocation
    {
        T value;
        U Location;
    };

    CHashLinear<T, U> * pHashLinear = (CHashLinear<T, U> *)pHashLinearVoid;
    uint64_t HashSize = pHashLinear->HashSize;

    HashLocation * pLocation = (HashLocation *)pHashLinear->pHashTableAny;
    U * pLocationOutput = (U *)pLocationOutputU;
    T * pHashList = (T *)pHashListT;

    // make local reference on stack
    uint64_t * pBitFields = pHashLinear->pBitFields;

    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    LOGGING(
        "IsMember arraySize %lld   get index size is %zu    output size is "
        "%zu  HashSize is %llu\n",
        arraySize, sizeof(U), sizeof(T), HashSize);

    // for (int i = 0; i < ((HashSize+63) / 64); i++) {
    //   printf("%llu |", pBitFields[i]);
    //}

    // printf("\n");

    if (sizeof(T) <= 2)
    {
        LOGGING("Perfect hash\n");
        // perfect hash
        for (int64_t i = 0; i < arraySize; i++)
        {
            T item = pHashList[i];
            uint64_t hash = (uint64_t)item;
            INNER_GET_LOCATION_PERFECT;
        }
    }
    else
    {
        HASH_MODE HashMode = pHashLinear->HashMode;

        if (sizeof(T) == 4)
        {
            if (HashMode == HASH_MODE_PRIME)
            {
                for (int64_t i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_INT1;
                    uint64_t hash = (uint64_t)h;
                    INNER_GET_LOCATION;
                    // InternalGetLocation(i, pLocation, pBoolOutput, pLocationOutput,
                    // item, h);
                }
            }
            else
            {
                for (int64_t i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_int32_t;
                    uint64_t hash = h;
                    INNER_GET_LOCATION;
                    // InternalGetLocation(i, pLocation, pBoolOutput, pLocationOutput,
                    // item, h);
                }
            }
        }
        else if (sizeof(T) == 8)
        {
            if (HashMode == HASH_MODE_PRIME)
            {
                for (int64_t i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_INT1;
                    uint64_t hash = (uint64_t)h;
                    INNER_GET_LOCATION;
                    // InternalGetLocation(i, pLocation, pBoolOutput, pLocationOutput,
                    // item, h);
                }
            }
            else
            {
                for (int64_t i = 0; i < arraySize; i++)
                {
                    T item = pHashList[i];
                    HASH_int64_t;
                    uint64_t hash = h;
                    INNER_GET_LOCATION;
                }
            }
        }
        else
        {
            printf("!!! IsMember hash item size not valid %zu\n", sizeof(T));
        }
    }
}

//-----------------------------------------
// bits 32-51 appear to be sweet spot
void CalculateHashBits64(int64_t arraySize, uint64_t * pHashList)
{
    int64_t position[64];
    for (int i = 0; i < 64; i++)
        position[i] = 0;

    while (arraySize--)
    {
        for (int i = 0; i < 64; i++)
        {
            // check if bit is set
            if ((1LL << i) & pHashList[arraySize])
                position[i]++;
        }
    }

    for (int i = 0; i < 64; i++)
    {
        printf("%d  %llu\n", i, position[i]);
    }
}

//-----------------------------------------------
// bits 3-22 appear to be sweet spot
void CalculateHashBits32(int64_t arraySize, uint32_t * pHashList)
{
    int64_t position[32];
    for (int i = 0; i < 32; i++)
        position[i] = 0;

    while (arraySize--)
    {
        for (int i = 0; i < 32; i++)
        {
            // check if bit is set
            if ((1LL << i) & pHashList[arraySize])
                position[i]++;
        }
    }

    for (int i = 0; i < 32; i++)
    {
        printf("%d  %llu\n", i, position[i]);
    }
}

//-----------------------------------------------
// stores the index of the first location
//
template <typename T, typename U>
void CHashLinear<T, U>::MakeHashLocationFloat(int64_t arraySize, T * pHashList, int64_t hintSize)
{
    if (hintSize == 0)
    {
        hintSize = arraySize;
    }

    AllocMemory(hintSize, sizeof(HashLocation), 0, true);
    NumUnique = 0;

    LOGGING("MakeHashLocationFloat: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

    HashLocation * pLocation = (HashLocation *)pHashTableAny;
    uint64_t * pBitFields = this->pBitFields;

    if (! pLocation || ! pBitFields)
    {
        return;
    }

    if (sizeof(T) == 8)
    {
        // printf("**double %llu\n", HashSize);
        // CalculateHashBits64(arraySize, (uint64_t*)pHashList);

        if (HashMode == HASH_MODE_PRIME)
        {
            for (U i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                // NAN CHECK FIRST
                if (item == item)
                {
                    uint64_t * pHashList2 = (uint64_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT3;
                    uint64_t hash = h;
                    INTERNAL_SET_LOCATION;
                    // InternalSetLocation(i, pLocation, item, h);
                }
            }
        }
        else
        {
            for (U i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                // NAN CHECK FIRST
                if (item == item)
                {
                    uint64_t * pHashList2 = (uint64_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT4;
                    uint64_t hash = h;
                    INTERNAL_SET_LOCATION;
                    // InternalSetLocation(i, pLocation, item, h);
                }
            }
        }
    }
    if (sizeof(T) == 4)
    {
        // printf("**single  %llu\n",HashSize);
        // CalculateHashBits32(arraySize,(uint32_t*)pHashList);

        if (HashMode == HASH_MODE_PRIME)
        {
            for (U i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                // NAN CHECK FIRST
                if (item == item)
                {
                    uint32_t * pHashList2 = (uint32_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT1;
                    uint64_t hash = h;
                    INTERNAL_SET_LOCATION;
                    // InternalSetLocation(i, pLocation, item, h);
                }
            }
        }
        else
        {
            for (U i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                // NAN CHECK FIRST
                if (item == item)
                {
                    uint32_t * pHashList2 = (uint32_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT2;
                    uint64_t hash = h;
                    INTERNAL_SET_LOCATION;
                    // InternalSetLocation(i, pLocation, item, h);
                }
            }
        }
    }
    LOGGING("%llu entries had %llu collisions   %llu unique\n", arraySize, NumCollisions, NumUnique);
    // printf("%llu entries had %llu collisions   %llu unique\n", arraySize,
    // NumCollisions, NumUnique);
}

//-----------------------------------------------
// outputs bool array
// outputs location array
//
template <typename T, typename U>
void IsMemberFloat(void * pHashLinearVoid, int64_t arraySize, void * pHashListT, int8_t * pBoolOutput, void * pLocationOutputU)
{
    LOGGING("IsMemberFloat: arraySize %lld \n", arraySize);

    struct HashLocation
    {
        T value;
        U Location;
    };

    CHashLinear<T, U> * pHashLinear = (CHashLinear<T, U> *)pHashLinearVoid;
    uint64_t HashSize = pHashLinear->HashSize;
    int HashMode = pHashLinear->HashMode;

    HashLocation * pLocation = (HashLocation *)pHashLinear->pHashTableAny;
    U * pLocationOutput = (U *)pLocationOutputU;
    T * pHashList = (T *)pHashListT;

    // make local reference on stack
    uint64_t * pBitFields = pHashLinear->pBitFields;

    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    if (sizeof(T) == 8)
    {
        if (HashMode == HASH_MODE_PRIME)
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                if (item == item)
                {
                    uint64_t * pHashList2 = (uint64_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT3;
                    uint64_t hash = (uint64_t)h;
                    INNER_GET_LOCATION;
                    // InternalGetLocation(i, pLocation, pBoolOutput, pLocationOutput,
                    // item, h);
                }
                else
                {
                    // Not found
                    pLocationOutput[i] = BAD_INDEX;
                    pBoolOutput[i] = 0;
                }
            }
        }
        else
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                if (item == item)
                {
                    uint64_t * pHashList2 = (uint64_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT4;
                    uint64_t hash = (uint64_t)h;
                    INNER_GET_LOCATION;
                    // InternalGetLocation(i, pLocation, pBoolOutput, pLocationOutput,
                    // item, h);
                }
                else
                {
                    // Not found
                    pLocationOutput[i] = BAD_INDEX;
                    pBoolOutput[i] = 0;
                }
            }
        }
    }
    else if (sizeof(T) == 4)
    {
        if (HashMode == HASH_MODE_PRIME)
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                if (item == item)
                {
                    uint32_t * pHashList2 = (uint32_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT1;
                    uint64_t hash = (uint64_t)h;
                    INNER_GET_LOCATION;
                    // InternalGetLocation(i, pLocation, pBoolOutput, pLocationOutput,
                    // item, h);
                }
                else
                {
                    // Not found
                    pLocationOutput[i] = BAD_INDEX;
                    pBoolOutput[i] = 0;
                }
            }
        }
        else
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                if (item == item)
                {
                    uint32_t * pHashList2 = (uint32_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT2;
                    uint64_t hash = (uint64_t)h;
                    INNER_GET_LOCATION;
                    // InternalGetLocation(i, pLocation, pBoolOutput, pLocationOutput,
                    // item, h);
                }
                else
                {
                    // Not found
                    pLocationOutput[i] = BAD_INDEX;
                    pBoolOutput[i] = 0;
                }
            }
        }
    }
}

//-----------------------------------------------
// outputs location array + 1
//
template <typename T, typename U>
int64_t CHashLinear<T, U>::IsMemberFloatCategorical(int64_t arraySize, T * pHashList, U * pLocationOutput)
{
    HashLocation * pLocation = (HashLocation *)pHashTableAny;
    int64_t missed = 0;

    // BUG BUG -- LONG DOUBLE on Linux size 16

    if (sizeof(T) == 8)
    {
        if (HashMode == HASH_MODE_PRIME)
        {
            for (U i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                if (item == item)
                {
                    uint64_t * pHashList2 = (uint64_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT3;
                    InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
                }
                else
                {
                    // Not found
                    pLocationOutput[i] = 0;
                    missed = 1;
                }
            }
        }
        else
        {
            for (U i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                if (item == item)
                {
                    uint64_t * pHashList2 = (uint64_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT4;
                    InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
                }
                else
                {
                    // Not found
                    pLocationOutput[i] = 0;
                    missed = 1;
                }
            }
        }
    }
    else if (sizeof(T) == 4)
    {
        if (HashMode == HASH_MODE_PRIME)
        {
            for (U i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                if (item == item)
                {
                    uint32_t * pHashList2 = (uint32_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT1;
                    InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
                }
                else
                {
                    // Not found
                    pLocationOutput[i] = 0;
                    missed = 1;
                }
            }
        }
        else
        {
            for (U i = 0; i < arraySize; i++)
            {
                T item = pHashList[i];
                if (item == item)
                {
                    uint32_t * pHashList2 = (uint32_t *)pHashList;
                    uint64_t h = pHashList2[i];
                    HASH_FLOAT2;
                    InternalGetLocationCategorical(i, pLocation, pLocationOutput, item, h, &missed);
                }
                else
                {
                    // Not found
                    pLocationOutput[i] = 0;
                    missed = 1;
                }
            }
        }
    }

    return missed;
}

//===========================================================================
//===========================================================================

#define GROUPBY_INNER_LOOP_PERFECT \
    uint64_t hash = (uint64_t)item & (HashSize - 1); \
    SingleKeyEntry * pKey = &pLocation[hash]; \
    if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63))) \
    { \
        pIndexArray[i] = pKey->UniqueKey; \
    } \
    else \
    { \
        pBitFieldsX[hash >> 6] |= (1LL << (hash & 63)); \
        pFirstArray[numUnique] = i; \
        numUnique++; \
        pKey->UniqueKey = numUnique; \
        pIndexArray[i] = numUnique; \
    }

#define GROUPBY_INNER_LOOP \
    uint64_t hash = h; \
    while (1) \
    { \
        if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63))) \
        { \
            /* Check if we have a match from before */ \
            if (item == pLocation[hash].value) \
            { \
                /* 2nd+ Match */ \
                /* Make the key the same */ \
                pIndexArray[i] = pLocation[hash].UniqueKey; \
                break; \
            } \
            ++hash; \
            if (hash >= HashSize) \
            { \
                hash = 0; \
            } \
        } \
        else \
        { \
            /* Failed to find hash */ \
            /* Must be first item */ \
            pBitFieldsX[hash >> 6] |= (1LL << (hash & 63)); \
            /* Check if we have a match from before */ \
            pLocation[hash].value = item; \
            pFirstArray[numUnique] = i; \
            /*base index of 1 so increment first */ \
            numUnique++; \
            pLocation[hash].UniqueKey = numUnique; \
            pIndexArray[i] = numUnique; \
            break; \
        } \
    }

typedef PyArrayObject *(COPY_TO_SMALLER_ARRAY)(void * pFirstArrayIndex, int64_t numUnique, int64_t totalRows);
//-----------------------------------------------------------------------------------------
// Returns AN ALLOCATED numpy array
// if firstArray is NULL, it will allocate
template <typename INDEX_TYPE>
PyArrayObject * CopyToSmallerArray(void * pFirstArrayIndex, int64_t numUnique, int64_t totalRows,
                                   PyArrayObject * firstArray = NULL)
{
    // check for out of memory
    if (! pFirstArrayIndex)
    {
        Py_IncRef(Py_None);
        // caller should check, this is additional safety
        return (PyArrayObject *)Py_None;
    }

    INDEX_TYPE * pFirstArray = (INDEX_TYPE *)pFirstArrayIndex;

    // Once we know the number of unique, we can allocate the smaller array
    if (firstArray == NULL)
    {
        switch (sizeof(INDEX_TYPE))
        {
        case 1:
            firstArray = AllocateNumpyArray(1, (npy_intp *)&numUnique, NPY_INT8);
            break;
        case 2:
            firstArray = AllocateNumpyArray(1, (npy_intp *)&numUnique, NPY_INT16);
            break;
        case 4:
            firstArray = AllocateNumpyArray(1, (npy_intp *)&numUnique, NPY_INT32);
            break;
        case 8:
            firstArray = AllocateNumpyArray(1, (npy_intp *)&numUnique, NPY_INT64);
            break;
        default:
            printf("!!!Internal error CopyToSmallerArray\n");
            break;
        }
    }
    CHECK_MEMORY_ERROR(firstArray);

    LOGGING("after alloc numpy copy to smaller %p  %lld %lld %lld\n", pFirstArray, numUnique, totalRows,
            numUnique * sizeof(INDEX_TYPE));

    if (firstArray != NULL && numUnique <= totalRows)
    {
        INDEX_TYPE * pFirstArrayDest = (INDEX_TYPE *)PyArray_BYTES(firstArray);
        memcpy(pFirstArrayDest, pFirstArray, numUnique * sizeof(INDEX_TYPE));
    }
    else
    {
        printf("!!! error allocating copytosmallerarray %lld %lld\n", numUnique, totalRows);
    }

    return firstArray;
}

//------------------------------------------------------------------
// NOTE pFirstArray is allocated and must be deallocated and reduced later
//    // return to caller the first array that we reduced
// *pFirstArrayObject = CopyToSmallerArray<U>(pFirstArray, numUnique,
// totalRows);

template <typename T, typename U>
uint64_t CHashLinear<T, U>::GroupByFloat(int64_t totalRows, int64_t totalItemSize, T * pInput,
                                         int coreType, // -1 when unknown  indicates only one array
                                                       // Return values
                                         U * pIndexArray, U *& pFirstArray, HASH_MODE hashMode, int64_t hintSize,
                                         bool * pBoolFilter)
{
    LOGGING("GroupByFloat: hintSize %lld   HashSize %llu  totalRows %lld\n", hintSize, HashSize, totalRows);

    U numUnique = 0;
    SingleKeyEntry * pLocation = (SingleKeyEntry *)pHashTableAny;

    // make local reference on stack
    uint64_t * pBitFieldsX = pBitFields;

    switch (sizeof(T))
    {
    case 4:
        if (pBoolFilter == NULL)
        {
            for (U i = 0; i < totalRows; i++)
            {
                T item = pInput[i];
                uint64_t h = ((uint32_t *)pInput)[i];
                HASH_FLOAT2;
                GROUPBY_INNER_LOOP;
            }
        }
        else
            for (U i = 0; i < totalRows; i++)
            {
                // check to see if in filter
                if (pBoolFilter[i])
                {
                    T item = pInput[i];
                    uint64_t h = ((uint32_t *)pInput)[i];
                    HASH_FLOAT2;
                    GROUPBY_INNER_LOOP;
                }
                else
                {
                    // not in filter set to zero bin
                    pIndexArray[i] = 0;
                }
            }
        break;
    case 8:
        if (pBoolFilter == NULL)
        {
            for (U i = 0; i < totalRows; i++)
            {
                T item = pInput[i];
                uint64_t h = ((uint64_t *)pInput)[i];
                HASH_FLOAT4;
                GROUPBY_INNER_LOOP;
            }
        }
        else
            for (U i = 0; i < totalRows; i++)
            {
                // check to see if in filter
                if (pBoolFilter[i])
                {
                    T item = pInput[i];
                    uint64_t h = ((uint64_t *)pInput)[i];
                    HASH_FLOAT4;
                    GROUPBY_INNER_LOOP;
                }
                else
                {
                    // not in filter set to zero bin
                    pIndexArray[i] = 0;
                }
            }
        break;
    }

    LOGGING("GroupByFloat end! %I64d\n", (int64_t)numUnique);

    return numUnique;
}

//------------------------------------------------------------------
// Returns pFirstArray
template <typename T, typename U>
uint64_t CHashLinear<T, U>::GroupByItemSize(int64_t totalRows, int64_t totalItemSize, T * pInput,
                                            int coreType, // -1 when unknown  indicates only one array
                                                          // Return values
                                            U * pIndexArray, U *& pFirstArray, HASH_MODE hashMode, int64_t hintSize,
                                            bool * pBoolFilter)
{
    LOGGING(
        "GroupByItem: hintSize %lld   HashSize %llu  totalRows %lld   "
        "sizeofT:%lld   sizeofU:%lld\n",
        hintSize, HashSize, totalRows, sizeof(T), sizeof(U));

    U numUnique = 0;
    SingleKeyEntry * pLocation = (SingleKeyEntry *)pHashTableAny;

    // make local reference on stack
    uint64_t * pBitFieldsX = pBitFields;

    switch (sizeof(T))
    {
    case 1:
        // TODO: Specially handle bools here -- they're 1-byte but logically have
        // only two buckets (zero/nonzero).
        // if (coreType == NPY_BOOL)
        //{
        //   //
        //}

        if (pBoolFilter == NULL)
        {
            for (U i = 0; i < totalRows; i++)
            {
                T item = pInput[i];
                GROUPBY_INNER_LOOP_PERFECT;
            }
        }
        else
            for (U i = 0; i < totalRows; i++)
            {
                // check to see if in filter
                if (pBoolFilter[i])
                {
                    T item = pInput[i];
                    GROUPBY_INNER_LOOP_PERFECT;
                }
                else
                {
                    // not in filter set to zero bin
                    pIndexArray[i] = 0;
                }
            }
        break;
    case 2:
        if (pBoolFilter == NULL)
        {
            for (U i = 0; i < totalRows; i++)
            {
                T item = pInput[i];
                GROUPBY_INNER_LOOP_PERFECT;
            }
        }
        else
            for (U i = 0; i < totalRows; i++)
            {
                // check to see if in filter
                if (pBoolFilter[i])
                {
                    T item = pInput[i];
                    GROUPBY_INNER_LOOP_PERFECT;
                }
                else
                {
                    // not in filter set to zero bin
                    pIndexArray[i] = 0;
                }
            }
        break;
    case 4:
        if (pBoolFilter == NULL)
        {
            for (U i = 0; i < totalRows; i++)
            {
                T item = pInput[i];
                HASH_int32_t;
                GROUPBY_INNER_LOOP;
            }
        }
        else
            for (U i = 0; i < totalRows; i++)
            {
                // check to see if in filter
                if (pBoolFilter[i])
                {
                    T item = pInput[i];
                    HASH_int32_t;
                    GROUPBY_INNER_LOOP;
                }
                else
                {
                    // not in filter set to zero bin
                    pIndexArray[i] = 0;
                }
            }
        break;
    case 8:
        if (pBoolFilter == NULL)
        {
            for (U i = 0; i < totalRows; i++)
            {
                T item = pInput[i];
                HASH_int64_t;
                GROUPBY_INNER_LOOP;
            }
        }
        else
            for (U i = 0; i < totalRows; i++)
            {
                // check to see if in filter
                if (pBoolFilter[i])
                {
                    T item = pInput[i];
                    HASH_int64_t;
                    GROUPBY_INNER_LOOP;
                }
                else
                {
                    // not in filter set to zero bin
                    pIndexArray[i] = 0;
                }
            }
        break;
    case 16:
        // TO BE WORKED ON...
        if (pBoolFilter == NULL)
        {
            for (U i = 0; i < totalRows; i++)
            {
                T item = pInput[i];
                HASH_INT128;
                GROUPBY_INNER_LOOP;
            }
        }
        else
            for (U i = 0; i < totalRows; i++)
            {
                // check to see if in filter
                if (pBoolFilter[i])
                {
                    T item = pInput[i];
                    HASH_INT128;
                    GROUPBY_INNER_LOOP;
                }
                else
                {
                    // not in filter set to zero bin
                    pIndexArray[i] = 0;
                }
            }
        break;
    }

    // printf("GroupByItem end! %d\n", numUnique);
    // return to caller the first array that we reduced
    //*pFirstArrayObject = CopyToSmallerArray<U>(pFirstArray, numUnique,
    // totalRows);

    return numUnique;
}

//-----------------------------------------------
// stores the index of the first location
// hintSize can be passed if # unique items is KNOWN or GUESSTIMATED in ADVNACE
// hintSize can be 0 which will default to totalRows
// pBoolFilter can be NULL
template <typename T, typename U>
uint64_t CHashLinear<T, U>::GroupBy(int64_t totalRows, int64_t totalItemSize, const char * pInput,

                                    int coreType, // -1 when unknown  indicates only one array

                                    // Return values
                                    U * pIndexArray, U *& pFirstArray,

                                    HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    LOGGING("GroupBy: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

    U numUnique = 0;
    U numCollisions = 0;
    MultiKeyEntry * pLocation = (MultiKeyEntry *)pHashTableAny;

    // make local reference on stack
    uint64_t * pBitFieldsX = pBitFields;
    if (pBoolFilter == NULL)
    {
        // make local reference on stack

        for (U i = 0; i < totalRows; i++)
        {
            const char * pMatch = pInput + (totalItemSize * i);
            uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);

            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            while (1)
            {
                if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63)))
                {
                    // Check if we have a match from before
                    U Last = pLocation[hash].Last;
                    const char * pMatch2 = pInput + (totalItemSize * Last);
                    int mresult;
                    MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                    if (! mresult)
                    {
                        // 2nd+ Match
                        // Make the key the same
                        pIndexArray[i] = pIndexArray[Last];
                        break;
                    }

                    // Linear goes to next position
                    ++hash;
                    if (hash >= HashSize)
                    {
                        hash = 0;
                    }
                }
                else
                {
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
    else
    {
        for (U i = 0; i < totalRows; i++)
        {
            // check to see if in filter
            if (pBoolFilter[i])
            {
                const char * pMatch = pInput + (totalItemSize * i);
                uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);
                // Use and mask to strip off high bits
                hash = hash & (HashSize - 1);
                while (1)
                {
                    if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63)))
                    {
                        // Check if we have a match from before
                        U Last = pLocation[hash].Last;
                        const char * pMatch2 = pInput + (totalItemSize * Last);
                        int mresult;
                        MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                        if (! mresult)
                        {
                            // 2nd+ Match
                            // Make the key the same
                            pIndexArray[i] = pIndexArray[Last];
                            break;
                        }

                        // Linear goes to next position
                        ++hash;
                        if (hash >= HashSize)
                        {
                            hash = 0;
                        }
                    }
                    else
                    {
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
            else
            {
                // not in filter set to zero bin
                pIndexArray[i] = 0;
            }
        }
    }

    LOGGING("%lld entries   hashSize %llu   %lld unique\n", totalRows, HashSize, (int64_t)numUnique);

    // return to caller the first array that we reduced
    //*pFirstArrayObject = CopyToSmallerArray<U>(pFirstArray, numUnique,
    // totalRows);

    return numUnique;
}

//-----------------------------------------------
// stores the index of the first location
// hintSize can be passed if # unique items is KNOWN or GUESSTIMATED in ADVNACE
// hintSize can be 0 which will default to totalRows
// pBoolFilter can be NULL
template <typename T, typename U>
uint64_t CHashLinear<T, U>::GroupBySuper(int64_t totalRows, int64_t totalItemSize, const char * pInput,

                                         int coreType, // -1 when unknown  indicates only one array

                                         // Return values
                                         U * pIndexArray, U * pNextArray, U * pUniqueArray, U * pUniqueCountArray,
                                         HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    if (hintSize == 0)
    {
        hintSize = totalRows;
    }
    AllocMemory(hintSize, sizeof(MultiKeyEntrySuper), 0, false);

    LOGGING("GroupBySuper: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

    U numUnique = 0;
    U numCollisions = 0;
    MultiKeyEntrySuper * pLocation = (MultiKeyEntrySuper *)pHashTableAny;

    if (! pLocation || ! pBitFields)
    {
        return 0;
    }

    // pre-fill for invalid bin
    // pUniqueCountArray[0] = 0;
    // pUniqueArray[0] = GB_INVALID_INDEX;

    // make local reference on stack
    uint64_t * pBitFieldsX = pBitFields;
    if (pBoolFilter == NULL)
    {
        // make local reference on stack

        for (U i = 0; i < totalRows; i++)
        {
            const char * pMatch = pInput + (totalItemSize * i);
            uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);
            // uint64_t hash = mHashOld(pMatch, totalItemSize);
            // uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);

            // printf("%d", hash);
            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            while (1)
            {
                if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63)))
                {
                    // Check if we have a match from before
                    U Last = pLocation[hash].Last;
                    const char * pMatch2 = pInput + (totalItemSize * Last);
                    int mresult;
                    MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                    if (! mresult)
                    {
                        // 2nd+ Match

                        // Make the key the same
                        pIndexArray[i] = pIndexArray[Last];

                        // The next item is unknown
                        pNextArray[i] = GB_INVALID_INDEX;

                        // if we go back to previous, the next item is us
                        pNextArray[Last] = i;

                        // update last item - TJD NOTE: DO NOT think this helps or is nec.
                        pLocation[hash].Last = i;
                        pUniqueCountArray[pLocation[hash].UniqueKey]++;
                        break;
                    }

                    // This entry is not us so we must have collided
                    ++numCollisions;

                    if (numCollisions < 0)
                    {
                        NumCollisions = numCollisions;
                        NumUnique = numUnique;

                        printf(
                            "!!! error in groupby collisions too high -- trying to "
                            "match size %lld\n",
                            totalItemSize);
                        printf(
                            "%llu entries   hashSize %llu  had %llu collisions   %llu "
                            "unique\n",
                            totalRows, HashSize, NumCollisions, NumUnique);
                        return NumUnique;
                    }

                    // Linear goes to next position
                    ++hash;
                    if (hash >= HashSize)
                    {
                        hash = 0;
                    }
                }
                else
                {
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
    else
    {
        U InvalidLast = GB_INVALID_INDEX;

        for (U i = 0; i < totalRows; i++)
        {
            // check to see if in filter
            if (pBoolFilter[i])
            {
                const char * pMatch = pInput + (totalItemSize * i);
                uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);
                // uint64_t hash = mHashOld(pMatch, totalItemSize);
                // uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);

                // printf("%d", hash);
                // Use and mask to strip off high bits
                hash = hash & (HashSize - 1);
                while (1)
                {
                    if (pBitFieldsX[hash >> 6] & (1LL << (hash & 63)))
                    {
                        // Check if we have a match from before
                        U Last = pLocation[hash].Last;
                        const char * pMatch2 = pInput + (totalItemSize * Last);
                        int mresult;
                        MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                        if (! mresult)
                        {
                            // 2nd+ Match

                            // Make the key the same
                            pIndexArray[i] = pIndexArray[Last];

                            // The next item is unknown
                            pNextArray[i] = GB_INVALID_INDEX;

                            // if we go back to previous, the next item is us
                            pNextArray[Last] = i;

                            // update last item
                            pLocation[hash].Last = i;
                            pUniqueCountArray[pLocation[hash].UniqueKey]++;
                            break;
                        }

                        // This entry is not us so we must have collided
                        ++numCollisions;

                        if (numCollisions < 0)
                        {
                            NumCollisions = numCollisions;
                            NumUnique = numUnique;

                            printf(
                                "!!! error in groupby collisions too high -- trying to "
                                "match size %lld\n",
                                totalItemSize);
                            printf(
                                "%llu entries   hashSize %llu  had %llu collisions   %llu "
                                "unique\n",
                                totalRows, HashSize, NumCollisions, NumUnique);
                            return NumUnique;
                        }

                        // Linear goes to next position
                        ++hash;
                        if (hash >= HashSize)
                        {
                            hash = 0;
                        }
                    }
                    else
                    {
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
                        // numUnique++;

                        break;
                    }
                }
            }
            else
            {
                // not in filter set to zero bin
                pIndexArray[i] = 0;
                pNextArray[i] = GB_INVALID_INDEX;

                // First location of invalid bin
                if (InvalidLast != GB_INVALID_INDEX)
                {
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
template <typename T, typename U>
uint64_t CHashLinear<T, U>::Unique(int64_t totalRows, int64_t totalItemSize, const char * pInput,

                                   // Return values
                                   U * pIndexArray,

                                   // Return count values
                                   U * pCountArray,

                                   // inpuys
                                   HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    if (hintSize == 0)
    {
        hintSize = totalRows;
    }

    AllocMemory(hintSize, sizeof(UniqueEntry), 0, false);

    LOGGING(
        "Unique: hintSize:%lld   HashSize:%llu  sizeoftypeU:%lld   "
        "sizeoftypeT:%lld\n",
        hintSize, HashSize, sizeof(U), sizeof(T));

    UniqueEntry * pLocation = (UniqueEntry *)pHashTableAny;

    if (! pLocation || ! pBitFields)
    {
        return 0;
    }

    U NumUnique = 0;

    if (pBoolFilter)
    {
        for (U i = 0; i < totalRows; i++)
        {
            // Make sure in the filter
            if (pBoolFilter[i])
            {
                const char * pMatch = pInput + (totalItemSize * i);
                uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);
                // Use and mask to strip off high bits
                hash = hash & (HashSize - 1);
                while (1)
                {
                    if (IsBitSet(hash))
                    {
                        // Check if we have a match from before
                        const char * pMatch2 = pLocation[hash].Last;

                        int mresult;
                        MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                        if (! mresult)
                        {
                            // 2nd+ Match
                            pCountArray[pLocation[hash].UniqueKey]++;
                            break;
                        }

                        // This entry is not us so we must have collided
                        //++NumCollisions;

                        // Linear goes to next position
                        if (++hash >= HashSize)
                        {
                            hash = 0;
                        }
                    }
                    else
                    {
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
    else
    {
        for (U i = 0; i < totalRows; i++)
        {
            const char * pMatch = pInput + (totalItemSize * i);
            uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);
            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            while (1)
            {
                if (IsBitSet(hash))
                {
                    // Check if we have a match from before
                    const char * pMatch2 = pLocation[hash].Last;

                    int mresult;
                    MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                    if (! mresult)
                    {
                        // 2nd+ Match
                        pCountArray[pLocation[hash].UniqueKey]++;
                        break;
                    }

                    // This entry is not us so we must have collided
                    //++NumCollisions;

                    // Linear goes to next position
                    if (++hash >= HashSize)
                    {
                        hash = 0;
                    }
                }
                else
                {
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

    LOGGING("%llu entries   hashSize %lld    %lld unique\n", totalRows, (int64_t)HashSize, (int64_t)NumUnique);

    return NumUnique;
}

//-----------------------------------------------
// Remembers previous values
// Set hintSize < 0 if second pass so it will not allocate
template <typename T, typename U>
void CHashLinear<T, U>::MultiKeyRolling(int64_t totalRows, int64_t totalItemSize, const char * pInput,

                                        // Return values
                                        U * pIndexArray, U * pRunningCountArray, HASH_MODE hashMode, int64_t hintSize)
{ // pass in -value to indicate reusing

    if (totalItemSize > 16)
    { // sizeof(MultiKeyEntryRolling.Key))
        printf("!!!rolling key is too wide %lld\n", totalItemSize);
        return;
    }

    if (hintSize >= 0)
    {
        if (hintSize == 0)
        {
            hintSize = totalRows;
        }
        AllocMemory(hintSize, sizeof(MultiKeyEntryRolling), 0, false);
        NumUnique = 0;
    }

    LOGGING("MakeHashLocationMultiKey: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

    MultiKeyEntryRolling * pLocation = (MultiKeyEntryRolling *)pHashTableAny;

    if (! pLocation || ! pBitFields)
    {
        return;
    }

    for (U i = 0; i < totalRows; i++)
    {
        const char * pMatch = pInput + (totalItemSize * i);
        uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);

        // Use and mask to strip off high bits
        hash = hash & (HashSize - 1);
        while (1)
        {
            if (IsBitSet(hash))
            {
                // Check if we have a match from before
                int mresult;
                MEMCMP_NEW(mresult, pMatch, pLocation[hash].Key, totalItemSize);
                if (! mresult)
                {
                    // 2nd+ Match
                    // 2nd+ Match
                    pIndexArray[i] = pLocation[hash].Last;
                    pRunningCountArray[i] = ++pLocation[hash].RunningCount;
                    break;
                }

                // This entry is not us so we must have collided
                ++NumCollisions;

                // Bail on too many collisions (could return false)
                if ((int64_t)NumCollisions > hintSize)
                    break;

                // Linear goes to next position
                if (++hash >= HashSize)
                {
                    hash = 0;
                }
            }
            else
            {
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
template <typename T, typename U>
void CHashLinear<T, U>::MakeHashLocationMultiKey(int64_t totalRows, int64_t totalItemSize, const char * pInput,

                                                 // Return values
                                                 U * pIndexArray, U * pRunningCountArray, U * pPrevArray, U * pNextArray,
                                                 U * pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    // int64_t arraySize,
    // const char* pHashList,
    // int32_t strWidth) {

    if (hintSize == 0)
    {
        hintSize = totalRows;
    }
    AllocMemory(hintSize, sizeof(MultiKeyEntry), 0, false);
    NumUnique = 0;

    LOGGING("MakeHashLocationMultiKey: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

    MultiKeyEntry * pLocation = (MultiKeyEntry *)pHashTableAny;

    if (! pLocation || ! pBitFields)
    {
        return;
    }

    for (U i = 0; i < totalRows; i++)
    {
        const char * pMatch = pInput + (totalItemSize * i);
        uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);
        // uint64_t hash = mHashOld(pMatch, totalItemSize);
        // uint64_t hash = DEFAULT_HASH64(pMatch, totalItemSize);

        // printf("%d", hash);
        // Use and mask to strip off high bits
        hash = hash & (HashSize - 1);
        while (1)
        {
            if (IsBitSet(hash))
            {
                // Check if we have a match from before
                U Last = pLocation[hash].Last;
                const char * pMatch2 = pInput + (totalItemSize * Last);
                int mresult;
                MEMCMP_NEW(mresult, pMatch, pMatch2, totalItemSize);
                if (! mresult)
                {
                    // 2nd+ Match
                    pIndexArray[i] = pIndexArray[Last];
                    pFirstArray[i] = pFirstArray[Last];
                    pRunningCountArray[i] = pRunningCountArray[Last] + 1;
                    pPrevArray[i] = Last;
                    pNextArray[i] = GB_INVALID_INDEX;
                    pNextArray[Last] = i;
                    pLocation[hash].Last = i;
                    break;
                }

                // This entry is not us so we must have collided
                ++NumCollisions;

                // Linear goes to next position
                if (++hash >= HashSize)
                {
                    hash = 0;
                }
            }
            else
            {
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
template <typename T, typename U>
void CHashLinear<T, U>::MakeHashLocationString(int64_t arraySize, const char * pHashList, int64_t strWidth, int64_t hintSize,
                                               bool isUnicode)
{
    if (hintSize == 0)
    {
        hintSize = arraySize;
    }

    AllocMemory(hintSize, sizeof(HashLocation), 0, false);
    NumUnique = 0;

    LOGGING("MakeHashLocationString: hintSize %lld   HashSize %llu\n", hintSize, HashSize);

    HashLocation * pLocation = (HashLocation *)pHashTableAny;

    if (! pLocation || ! pBitFields)
    {
        return;
    }

    if (isUnicode)
    {
        for (int64_t i = 0; i < arraySize; i++)
        {
            HASH_UNICODE()
            // Use and mask to strip off high bits
            hash = hash & (HashSize - 1);
            InternalSetLocationUnicode((U)i, pLocation, strStart, strWidth, hash);
        }
    }
    else
    {
        for (int64_t i = 0; i < arraySize; i++)
        {
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
FORCE_INLINE void CHashLinear<T, U>::InternalSetLocationString(U i, HashLocation * pLocation, const char * strValue,
                                                               int64_t strWidth, uint64_t hash)
{
    // printf("**set %llu  width: %d  string: %s\n", hash, strWidth, strValue);

    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        if (STRING_MATCH((const char *)pLocation[hash].value, strValue, strWidth))
        {
            return;
        }

        // printf("Collide \n");

        // This entry is not us so we must have collided
        ++NumCollisions;

        // Linear goes to next position
        if (++hash >= HashSize)
        {
            hash = 0;

            if (NumCollisions > (HashSize * 2))
            {
                // LogError("hash collision error %d %llu\n", i, NumCollisions);
                LogError("Bad hash function, too many collisions");
                return;
            }
        }
    }
    // Failed to find hash
    SetBit(hash);
    ++NumUnique;
    pLocation[hash].Location = i;
    pLocation[hash].value = (int64_t)strValue;
}

//-----------------------------------------------
// stores the index of the first location
// remove the forceline to make debugging easier
template <typename T, typename U>
FORCE_INLINE void CHashLinear<T, U>::InternalSetLocationUnicode(U i, HashLocation * pLocation, const char * strValue,
                                                                int64_t strWidth, uint64_t hash)
{
    // printf("**set %llu  width: %d  string: %s\n", hash, strWidth, strValue);

    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        if (UNICODE_MATCH((const char *)pLocation[hash].value, strValue, strWidth))
        {
            return;
        }

        // printf("Collide \n");

        // This entry is not us so we must have collided
        ++NumCollisions;

        // Linear goes to next position
        if (++hash >= HashSize)
        {
            hash = 0;

            if (NumCollisions > (HashSize * 2))
            {
                // LogError("hash collision error %d %llu\n", i, NumCollisions);
                LogError("Bad hash function, too many collisions");
                return;
            }
        }
    }
    // Failed to find hash
    SetBit(hash);
    ++NumUnique;
    pLocation[hash].Location = i;
    pLocation[hash].value = (int64_t)strValue;
}

//-----------------------------------------------
// looks for the index of set location
// strings must be same width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE void CHashLinear<T, U>::InternalGetLocationString(int64_t i, HashLocation * pLocation, int8_t * pBoolOutput,
                                                               U * pLocationOutput, const char * strValue, int64_t strWidth,
                                                               uint64_t hash)
{
    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        // Check if we have a match from before
        if (STRING_MATCH((const char *)pLocation[hash].value, strValue, strWidth))
        {
            // return the first location
            pLocationOutput[i] = pLocation[hash].Location;
            pBoolOutput[i] = 1;
            return;
        }

        // Linear goes to next position
        if (++hash >= HashSize)
        {
            hash = 0;
        }
    }
    // Not found
    pLocationOutput[i] = BAD_INDEX;
    pBoolOutput[i] = 0;
}

//-----------------------------------------------
// looks for the index of set location
// strings must be same width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE void CHashLinear<T, U>::InternalGetLocationUnicode(int64_t i, HashLocation * pLocation, int8_t * pBoolOutput,
                                                                U * pLocationOutput, const char * strValue, int64_t strWidth,
                                                                uint64_t hash)
{
    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        // Check if we have a match from before
        if (UNICODE_MATCH((const char *)pLocation[hash].value, strValue, strWidth))
        {
            // return the first location
            pLocationOutput[i] = pLocation[hash].Location;
            pBoolOutput[i] = 1;
            return;
        }

        // Linear goes to next position
        if (++hash >= HashSize)
        {
            hash = 0;
        }
    }
    // Not found
    pLocationOutput[i] = BAD_INDEX;
    pBoolOutput[i] = 0;
}

//-----------------------------------------------
// looks for the index of set location
// strings must be diff width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE void CHashLinear<T, U>::InternalGetLocationString2(int64_t i, HashLocation * pLocation, int8_t * pBoolOutput,
                                                                U * pLocationOutput, const char * strValue, int64_t strWidth,
                                                                int64_t strWidth2, uint64_t hash)
{
    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        // Check if we have a match from before
        // if ((i % 1000) == 0) printf("%d  Comparing2 %s to %s\n", (int)i, (const
        // char*)pLocation[hash].value, strValue);
        if (STRING_MATCH2((const char *)pLocation[hash].value, strValue, strWidth, strWidth2))
        {
            // printf("match\n");
            // return the first location
            pLocationOutput[i] = pLocation[hash].Location;
            pBoolOutput[i] = 1;
            return;
        }

        // printf("no match checking next\n");
        // Linear goes to next position
        if (++hash >= HashSize)
        {
            hash = 0;
        }
    }
    // Not found
    pLocationOutput[i] = BAD_INDEX;
    pBoolOutput[i] = 0;
}

//-----------------------------------------------
// looks for the index of set location
// strings must be diff width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE void CHashLinear<T, U>::InternalGetLocationUnicode2(int64_t i, HashLocation * pLocation, int8_t * pBoolOutput,
                                                                 U * pLocationOutput, const char * strValue, int64_t strWidth,
                                                                 int64_t strWidth2, uint64_t hash)
{
    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        // Check if we have a match from before
        // if ((i % 1000) == 0) printf("%d  Comparing2 %s to %s\n", (int)i, (const
        // char*)pLocation[hash].value, strValue);
        if (UNICODE_MATCH2((const char *)pLocation[hash].value, strValue, strWidth, strWidth2))
        {
            // printf("match\n");
            // return the first location
            pLocationOutput[i] = pLocation[hash].Location;
            pBoolOutput[i] = 1;
            return;
        }

        // printf("no match checking next\n");
        // Linear goes to next position
        if (++hash >= HashSize)
        {
            hash = 0;
        }
    }
    // Not found
    pLocationOutput[i] = BAD_INDEX;
    pBoolOutput[i] = 0;
}

//-----------------------------------------------
// looks for the index of set location
// strings must be same width
// remove the forceline to make debugging easier
// Like Matlab IsMember
// Returns two arrays
template <typename T, typename U>
FORCE_INLINE void CHashLinear<T, U>::InternalGetLocationStringCategorical(int64_t i, HashLocation * pLocation, U * pLocationOutput,
                                                                          const char * strValue, int64_t strWidth, uint64_t hash,
                                                                          int64_t * missed)
{
    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        // Check if we have a match from before
        if (STRING_MATCH((const char *)pLocation[hash].value, strValue, strWidth))
        {
            // return the first location
            pLocationOutput[i] = pLocation[hash].Location + 1;
            return;
        }

        // Linear goes to next position
        if (++hash >= HashSize)
        {
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
FORCE_INLINE void CHashLinear<T, U>::InternalGetLocationString2Categorical(int64_t i, HashLocation * pLocation,
                                                                           U * pLocationOutput, const char * strValue,
                                                                           int64_t strWidth, int64_t strWidth2, uint64_t hash,
                                                                           int64_t * missed)
{
    while (IsBitSet(hash))
    {
        // Check if we have a match from before
        // Check if we have a match from before
        // if ((i % 1000) == 0) printf("%d  Comparing2 %s to %s\n", (int)i, (const
        // char*)pLocation[hash].value, strValue);
        if (STRING_MATCH2((const char *)pLocation[hash].value, strValue, strWidth, strWidth2))
        {
            // printf("match\n");
            // return the first location
            pLocationOutput[i] = pLocation[hash].Location + 1;
            return;
        }

        // printf("no match checking next\n");
        // Linear goes to next position
        if (++hash >= HashSize)
        {
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

template <typename T, typename U>
static int64_t IsMemberStringCategorical(void * pHashLinearVoid, int64_t arraySize, int64_t strWidth, int64_t strWidth2,
                                         const char * pHashList, void * pLocationOutputU, bool isUnicode)
{
    struct HashLocation
    {
        T value;
        U Location;
    };

    CHashLinear<T, U> * pHashLinear = (CHashLinear<T, U> *)pHashLinearVoid;

    HashLocation * pLocation = (HashLocation *)pHashLinear->pHashTableAny;
    U * pLocationOutput = (U *)pLocationOutputU;

    int64_t missed = 0;
    uint64_t HashSize = pHashLinear->HashSize;

    // to determine if hash location has been visited
    uint64_t * pBitFields = pHashLinear->pBitFields;

    if (strWidth == strWidth2)
    {
        //-------------------------------------------------------------------
        // STRINGS are SAME SIZE --------------------------------------------
        if (isUnicode)
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                HASH_UNICODE()

                // Use and mask to strip off high bits
                hash = hash & (HashSize - 1);

                while (1)
                {
                    uint64_t index = hash >> 6;
                    if (pBitFields[index] & (1LL << (hash & 63)))
                    {
                        // Check if we have a match from before
                        // Check if we have a match from before
                        if (UNICODE_MATCH((const char *)pLocation[hash].value, strStart, strWidth))
                        {
                            // return the first location
                            pLocationOutput[i] = pLocation[hash].Location + 1;
                            break;
                        }

                        // Linear goes to next position
                        if (++hash >= HashSize)
                        {
                            hash = 0;
                        }
                        continue;
                    }
                    else
                    {
                        // Not found
                        pLocationOutput[i] = 0;
                        missed = 1;
                        break;
                    }
                }
            }
        }
        else
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                HASH_STRING()

                // Use and mask to strip off high bits
                hash = hash & (HashSize - 1);
                while (1)
                {
                    uint64_t index = hash >> 6;
                    if (pBitFields[index] & (1LL << (hash & 63)))
                    {
                        // Check if we have a match from before
                        if (STRING_MATCH((const char *)pLocation[hash].value, strStart, strWidth))
                        {
                            // return the first location
                            pLocationOutput[i] = pLocation[hash].Location + 1;
                            break;
                        }

                        // Linear goes to next position
                        if (++hash >= HashSize)
                        {
                            hash = 0;
                        }
                        continue;
                    }
                    else
                    {
                        // Not found
                        pLocationOutput[i] = 0;
                        missed = 1;
                        break;
                    }
                }
            }
        }
    }
    else
    {
        //-------------------------------------------------------------------
        // STRINGS are DIFFERENT SIZE --------------------------------------------
        if (isUnicode)
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                HASH_UNICODE()

                // Use and mask to strip off high bits
                hash = hash & (pHashLinear->HashSize - 1);

                while (1)
                {
                    uint64_t index = hash >> 6;
                    if (pBitFields[index] & (1LL << (hash & 63)))
                    {
                        // Check if we have a match from before
                        // Check if we have a match from before
                        if (UNICODE_MATCH2((const char *)pLocation[hash].value, strStart, strWidth2, strWidth))
                        {
                            // return the first location
                            pLocationOutput[i] = pLocation[hash].Location + 1;
                            break;
                        }

                        // Linear goes to next position
                        if (++hash >= HashSize)
                        {
                            hash = 0;
                        }
                        continue;
                    }
                    else
                    {
                        // Not found
                        pLocationOutput[i] = 0;
                        missed = 1;
                        break;
                    }
                }
            }
        }
        else
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                HASH_STRING()

                // Use and mask to strip off high bits
                hash = hash & (pHashLinear->HashSize - 1);

                while (1)
                {
                    uint64_t index = hash >> 6;
                    if (pBitFields[index] & (1LL << (hash & 63)))
                    {
                        // Check if we have a match from before
                        if (STRING_MATCH2((const char *)pLocation[hash].value, strStart, strWidth2, strWidth))
                        {
                            // return the first location
                            pLocationOutput[i] = pLocation[hash].Location + 1;
                            break;
                        }

                        // Linear goes to next position
                        if (++hash >= pHashLinear->HashSize)
                        {
                            hash = 0;
                        }
                        continue;
                    }
                    else
                    {
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
// outputs bool array
// outputs location array
//
// strWidth is the width for pHashList and the first argument called passed
// strWidth2 was used in InternalSetLocationString and refers to second argument

template <typename T, typename U>
void CHashLinear<T, U>::IsMemberString(int64_t arraySize, int64_t strWidth, int64_t strWidth2, const char * pHashList,
                                       int8_t * pBoolOutput, U * pLocationOutput, bool isUnicode)
{
    HashLocation * pLocation = (HashLocation *)pHashTableAny;

    if (strWidth == strWidth2)
    {
        //-------------------------------------------------------------------
        // STRINGS are SAME SIZE --------------------------------------------
        if (isUnicode)
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                HASH_UNICODE()

                // Use and mask to strip off high bits
                hash = hash & (HashSize - 1);
                // printf("**uni hash %lld\n", (long long)hash);
                InternalGetLocationUnicode(i, pLocation, pBoolOutput, pLocationOutput, strStart, strWidth, hash);
            }
        }
        else
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                HASH_STRING()

                // Use and mask to strip off high bits
                hash = hash & (HashSize - 1);
                InternalGetLocationString(i, pLocation, pBoolOutput, pLocationOutput, strStart, strWidth, hash);
            }
        }
    }
    else
    {
        //-------------------------------------------------------------------
        // STRINGS are DIFFERENT SIZE --------------------------------------------
        if (isUnicode)
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                HASH_UNICODE()

                // Use and mask to strip off high bits
                hash = hash & (HashSize - 1);
                InternalGetLocationUnicode2(i, pLocation, pBoolOutput, pLocationOutput, strStart, strWidth2, strWidth, hash);
            }
        }
        else
        {
            for (int64_t i = 0; i < arraySize; i++)
            {
                HASH_STRING()

                // Use and mask to strip off high bits
                hash = hash & (HashSize - 1);
                InternalGetLocationString2(i, pLocation, pBoolOutput, pLocationOutput, strStart, strWidth2, strWidth, hash);
            }
        }
    }
}

//===============================================================================
//===============================================================================

typedef void (*ISMEMBER_MT)(void * pHashLinearVoid, int64_t arraySize, void * pHashList, int8_t * pBoolOutput,
                            void * pLocationOutputU);

//--------------------------------------------------------------------
struct IMMT_CALLBACK
{
    ISMEMBER_MT anyIMMTCallback;

    void * pHashLinearVoid;

    int64_t size1;
    void * pHashList;
    int8_t * pBoolOutput;
    void * pOutput;
    int64_t typeSizeIn;
    int64_t typeSizeOut;

} stIMMTCallback;

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static bool IMMTThreadCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    bool didSomeWork = false;
    IMMT_CALLBACK * Callback = (IMMT_CALLBACK *)pstWorkerItem->WorkCallbackArg;

    char * pInput1 = (char *)Callback->pHashList;
    char * pOutput = (char *)Callback->pOutput;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
    {
        int64_t inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeIn;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;
        int64_t boolAdj = pstWorkerItem->BlockSize * workBlock;

        Callback->anyIMMTCallback(Callback->pHashLinearVoid, lenX, pInput1 + inputAdj, Callback->pBoolOutput + boolAdj,
                                  pOutput + outputAdj);

        // Indicate we completed a block
        didSomeWork = true;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        // printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}

static void IsMemberMultiThread(ISMEMBER_MT pFunction, void * pHashLinearVoid, int64_t arraySize, void * pHashList,
                                int8_t * pBoolOutput, void * pLocationOutputU, int64_t sizeInput, int64_t sizeOutput)
{
    stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(arraySize);

    if (pWorkItem == NULL)
    {
        // Threading not allowed for this work item, call it directly from main
        // thread
        pFunction(pHashLinearVoid, arraySize, pHashList, pBoolOutput, pLocationOutputU);
    }
    else
    {
        // Each thread will call this routine with the callbackArg
        pWorkItem->DoWorkCallback = IMMTThreadCallback;

        pWorkItem->WorkCallbackArg = &stIMMTCallback;

        stIMMTCallback.pHashLinearVoid = pHashLinearVoid;
        stIMMTCallback.anyIMMTCallback = pFunction;
        stIMMTCallback.size1 = arraySize;
        stIMMTCallback.pHashList = pHashList;
        stIMMTCallback.pBoolOutput = pBoolOutput;
        stIMMTCallback.pOutput = pLocationOutputU;
        stIMMTCallback.typeSizeIn = sizeInput;
        stIMMTCallback.typeSizeOut = sizeOutput;

        // This will notify the worker threads of a new work item
        g_cMathWorker->WorkMain(pWorkItem, arraySize, 0);
    }
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
// floats are 4 bytes and will be handled like int32_t or UIN32
//
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//
// Returns in pOutput: index location of second arg -- where first arg found in
// second arg Returns in pBoolOutput: True if found, False otherwise
template <typename U>
void * IsMemberHash32(int64_t size1, void * pInput1,

                      int64_t size2, void * pInput2,

                      U * pOutput,

                      int8_t * pBoolOutput, int32_t sizeType, HASH_MODE hashMode, int64_t hintSize)
{
    // Allocate hash

    switch (sizeType)
    {
    case 1:
        {
            CHashLinear<uint8_t, U> * pHashLinear = new CHashLinear<uint8_t, U>(HASH_MODE_MASK);
            pHashLinear->MakeHashLocation(size2, (uint8_t *)pInput2, 256 / 2);
            IsMemberMultiThread(IsMember<uint8_t, U>, pHashLinear, size1, (uint8_t *)pInput1, pBoolOutput, (U *)pOutput,
                                sizeof(uint8_t), sizeof(U));
            delete pHashLinear;
            return NULL;
        }
        break;
    case 2:
        {
            CHashLinear<uint16_t, U> * pHashLinear = new CHashLinear<uint16_t, U>(HASH_MODE_MASK);
            pHashLinear->MakeHashLocation(size2, (uint16_t *)pInput2, 65536 / 2);
            IsMemberMultiThread(IsMember<uint16_t, U>, pHashLinear, size1, (uint16_t *)pInput1, pBoolOutput, (U *)pOutput,
                                sizeof(uint16_t), sizeof(U));
            delete pHashLinear;
            return NULL;
        }
        break;

    case 4:
        {
            CHashLinear<uint32_t, U> * pHashLinear = new CHashLinear<uint32_t, U>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint32_t *)pInput2, hintSize);
            IsMemberMultiThread(IsMember<uint32_t, U>, pHashLinear, size1, (uint32_t *)pInput1, pBoolOutput, (U *)pOutput,
                                sizeof(uint32_t), sizeof(U));
            delete pHashLinear;
            return NULL;
        }
        break;
    case 8:
        {
            CHashLinear<uint64_t, U> * pHashLinear = new CHashLinear<uint64_t, U>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint64_t *)pInput2, hintSize);
            IsMemberMultiThread(IsMember<uint64_t, U>, pHashLinear, size1, (uint64_t *)pInput1, pBoolOutput, (U *)pOutput,
                                sizeof(uint64_t), sizeof(U));
            delete pHashLinear;
            return NULL;
        }
        break;
    case 104:
        {
            CHashLinear<float, U> * pHashLinear = new CHashLinear<float, U>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (float *)pInput2, hintSize);
            IsMemberMultiThread(IsMemberFloat<float, U>, pHashLinear, size1, (float *)pInput1, pBoolOutput, (U *)pOutput,
                                sizeof(float), sizeof(U));
            delete pHashLinear;
            return NULL;
        }
        break;
    case 108:
        {
            CHashLinear<double, U> * pHashLinear = new CHashLinear<double, U>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (double *)pInput2, hintSize);
            IsMemberMultiThread(IsMemberFloat<double, U>, pHashLinear, size1, (double *)pInput1, pBoolOutput, (U *)pOutput,
                                sizeof(double), sizeof(U));
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
// floats are + 100 and will be handled differnt from int64_t or UIN32
void * IsMemberHash64(int64_t size1, void * pInput1, int64_t size2, void * pInput2, int64_t * pOutput, int8_t * pBoolOutput,
                      int32_t sizeType, HASH_MODE hashMode, int64_t hintSize)
{
    switch (sizeType)
    {
    case 1:
        {
            CHashLinear<uint8_t, int64_t> * pHashLinear = new CHashLinear<uint8_t, int64_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint8_t *)pInput2, hintSize);
            IsMember<uint8_t, int64_t>(pHashLinear, size1, (uint8_t *)pInput1, pBoolOutput, (int64_t *)pOutput);
            delete pHashLinear;
            return NULL;
        }
        break;
    case 2:
        {
            CHashLinear<uint16_t, int64_t> * pHashLinear = new CHashLinear<uint16_t, int64_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint16_t *)pInput2, hintSize);
            IsMember<uint16_t, int64_t>(pHashLinear, size1, (uint16_t *)pInput1, pBoolOutput, (int64_t *)pOutput);
            delete pHashLinear;
            return NULL;
        }
        break;

    case 4:
        {
            CHashLinear<uint32_t, int64_t> * pHashLinear = new CHashLinear<uint32_t, int64_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint32_t *)pInput2, hintSize);
            IsMember<uint32_t, int64_t>(pHashLinear, size1, (uint32_t *)pInput1, pBoolOutput, (int64_t *)pOutput);
            delete pHashLinear;
            return NULL;
        }
        break;
    case 8:
        {
            CHashLinear<uint64_t, int64_t> * pHashLinear = new CHashLinear<uint64_t, int64_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint64_t *)pInput2, hintSize);
            IsMember<uint64_t, int64_t>(pHashLinear, size1, (uint64_t *)pInput1, pBoolOutput, (int64_t *)pOutput);
            delete pHashLinear;
            return NULL;
        }
        break;
    case 104:
        {
            CHashLinear<float, int64_t> * pHashLinear = new CHashLinear<float, int64_t>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (float *)pInput2, hintSize);
            IsMemberFloat<float, int64_t>(pHashLinear, size1, (float *)pInput1, pBoolOutput, (int64_t *)pOutput);
            delete pHashLinear;
            return NULL;
        }
        break;
    case 108:
        {
            CHashLinear<double, int64_t> * pHashLinear = new CHashLinear<double, int64_t>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (double *)pInput2, hintSize);
            IsMemberFloat<double, int64_t>(pHashLinear, size1, (double *)pInput1, pBoolOutput, (int64_t *)pOutput);
            delete pHashLinear;
            return NULL;
        }
        break;
    }

    return NULL;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
// floats are 4 bytes and will be handled like int32_t or UIN32
//
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//
// Returns in pOutput: index location of second arg -- where first arg found in
// second arg Returns in pBoolOutput: True if found, False otherwise
int64_t IsMemberHashCategorical(int64_t size1, void * pInput1, int64_t size2, void * pInput2, int32_t * pOutput, int32_t sizeType,
                                HASH_MODE hashMode, int64_t hintSize)
{
    int64_t missed = 0;
    // Allocate hash

    switch (sizeType)
    {
    case 1:
        {
            CHashLinear<uint8_t, int32_t> * pHashLinear = new CHashLinear<uint8_t, int32_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint8_t *)pInput2, hintSize);
            missed = pHashLinear->IsMemberCategorical(size1, (uint8_t *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;

    case 2:
        {
            CHashLinear<uint16_t, int32_t> * pHashLinear = new CHashLinear<uint16_t, int32_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint16_t *)pInput2, hintSize);
            missed = pHashLinear->IsMemberCategorical(size1, (uint16_t *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;

    case 4:
        {
            CHashLinear<uint32_t, int32_t> * pHashLinear = new CHashLinear<uint32_t, int32_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint32_t *)pInput2, hintSize);
            missed = pHashLinear->IsMemberCategorical(size1, (uint32_t *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    case 8:
        {
            CHashLinear<uint64_t, int32_t> * pHashLinear = new CHashLinear<uint64_t, int32_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint64_t *)pInput2, hintSize);
            missed = pHashLinear->IsMemberCategorical(size1, (uint64_t *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    case 104:
        {
            CHashLinear<float, int32_t> * pHashLinear = new CHashLinear<float, int32_t>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (float *)pInput2, hintSize);
            missed = pHashLinear->IsMemberFloatCategorical(size1, (float *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    case 108:
        {
            CHashLinear<double, int32_t> * pHashLinear = new CHashLinear<double, int32_t>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (double *)pInput2, hintSize);
            missed = pHashLinear->IsMemberFloatCategorical(size1, (double *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    case 116:
        {
            CHashLinear<long double, int32_t> * pHashLinear = new CHashLinear<long double, int32_t>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (long double *)pInput2, hintSize);
            missed = pHashLinear->IsMemberFloatCategorical(size1, (long double *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    }

    return missed;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
// floats are 4 bytes and will be handled like int32_t or UIN32
//
//    First arg: existing  numpy array
//    Second arg: existing  numpy array
//    Third arg: hashmode (1 or 2)
//
// Returns in pOutput: index location of second arg -- where first arg found in
// second arg Returns in pBoolOutput: True if found, False otherwise
int64_t IsMemberHashCategorical64(int64_t size1, void * pInput1, int64_t size2, void * pInput2, int64_t * pOutput,
                                  int32_t sizeType, HASH_MODE hashMode, int64_t hintSize)
{
    int64_t missed = 0;
    // Allocate hash

    switch (sizeType)
    {
    case 1:
        {
            CHashLinear<uint8_t, int64_t> * pHashLinear = new CHashLinear<uint8_t, int64_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint8_t *)pInput2, hintSize);
            missed = pHashLinear->IsMemberCategorical(size1, (uint8_t *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;

    case 2:
        {
            CHashLinear<uint16_t, int64_t> * pHashLinear = new CHashLinear<uint16_t, int64_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint16_t *)pInput2, hintSize);
            missed = pHashLinear->IsMemberCategorical(size1, (uint16_t *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;

    case 4:
        {
            CHashLinear<uint32_t, int64_t> * pHashLinear = new CHashLinear<uint32_t, int64_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint32_t *)pInput2, hintSize);
            missed = pHashLinear->IsMemberCategorical(size1, (uint32_t *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    case 8:
        {
            CHashLinear<uint64_t, int64_t> * pHashLinear = new CHashLinear<uint64_t, int64_t>(hashMode);
            pHashLinear->MakeHashLocation(size2, (uint64_t *)pInput2, hintSize);
            missed = pHashLinear->IsMemberCategorical(size1, (uint64_t *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    case 104:
        {
            CHashLinear<float, int64_t> * pHashLinear = new CHashLinear<float, int64_t>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (float *)pInput2, hintSize);
            missed = pHashLinear->IsMemberFloatCategorical(size1, (float *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    case 108:
        {
            CHashLinear<double, int64_t> * pHashLinear = new CHashLinear<double, int64_t>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (double *)pInput2, hintSize);
            missed = pHashLinear->IsMemberFloatCategorical(size1, (double *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    case 116:
        {
            CHashLinear<long double, int64_t> * pHashLinear = new CHashLinear<long double, int64_t>(hashMode);
            pHashLinear->MakeHashLocationFloat(size2, (long double *)pInput2, hintSize);
            missed = pHashLinear->IsMemberFloatCategorical(size1, (long double *)pInput1, pOutput);
            delete pHashLinear;
            return missed;
        }
        break;
    }

    return missed;
}

//===================================================================================================

typedef int64_t (*ISMEMBER_STRING)(void * pHashLinearVoid, int64_t arraySize, int64_t strWidth1, int64_t strWidth2,
                                   const char * pHashList, void * pLocationOutputU, bool isUnicode);

//--------------------------------------------------------------------
struct IMS_CALLBACK
{
    ISMEMBER_STRING anyIMSCallback;

    void * pHashLinearVoid;

    int64_t size1;
    int64_t strWidth1;
    const char * pInput1;
    int64_t size2;
    int64_t strWidth2;
    void * pOutput;
    int64_t typeSizeOut;
    int64_t missed;

    bool isUnicode;

} stIMSCallback;

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static bool IMSThreadCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    bool didSomeWork = false;
    IMS_CALLBACK * Callback = (IMS_CALLBACK *)pstWorkerItem->WorkCallbackArg;

    bool isUnicode = Callback->isUnicode;
    char * pInput1 = (char *)Callback->pInput1;
    char * pOutput = (char *)Callback->pOutput;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
    {
        int64_t inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->strWidth1;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;

        int64_t missed = Callback->anyIMSCallback(Callback->pHashLinearVoid, lenX, Callback->strWidth1, Callback->strWidth2,
                                                  pInput1 + inputAdj, pOutput + outputAdj, isUnicode);

        // Careful with multithreading -- only set it to 1
        if (missed)
        {
            Callback->missed = 1;
        }

        // Indicate we completed a block
        didSomeWork = true;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        // printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
template <typename U>
int64_t IsMemberHashStringCategorical(int64_t size1, int64_t strWidth1, const char * pInput1, int64_t size2, int64_t strWidth2,
                                      const char * pInput2, U * pOutput, HASH_MODE hashMode, int64_t hintSize, bool isUnicode)
{
    CHashLinear<uint64_t, U> * pHashLinear = new CHashLinear<uint64_t, U>(hashMode);

    LOGGING("MakeHashLocationString  %lld  %p  strdwidth2: %lld  hashMode %d\n", size2, pInput2, strWidth2, (int)hashMode);

    // First pass build hash table of second string input
    pHashLinear->MakeHashLocationString(size2, pInput2, strWidth2, hintSize, isUnicode);

    LOGGING("IsMemberString  %lld  %lld  strdwidth2: %lld\n", size1, strWidth1, strWidth2);

    // Second pass find matches
    // We can multithread it
    int64_t missed;

    stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(size1);
    ISMEMBER_STRING pFunction = IsMemberStringCategorical<uint64_t, U>;

    if (pWorkItem == NULL)
    {
        // Threading not allowed for this work item, call it directly from main
        // thread
        missed = pFunction(pHashLinear, size1, strWidth1, strWidth2, pInput1, pOutput, isUnicode);
    }
    else
    {
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
template <typename U>
void IsMemberHashString32(int64_t size1, int64_t strWidth1, const char * pInput1, int64_t size2, int64_t strWidth2,
                          const char * pInput2, U * pOutput, int8_t * pBoolOutput, HASH_MODE hashMode, int64_t hintSize,
                          bool isUnicode)
{
    CHashLinear<uint64_t, U> * pHashLinear = new CHashLinear<uint64_t, U>(hashMode);

    LOGGING("MakeHashLocationString  %lld  %p  strdwidth2: %lld  hashMode %d\n", size2, pInput2, strWidth2, (int)hashMode);

    // First pass build hash table of second string input
    pHashLinear->MakeHashLocationString(size2, pInput2, strWidth2, hintSize, isUnicode);

    LOGGING("IsMemberString  %lld  %lld  strdwidth2: %lld\n", size1, strWidth1, strWidth2);

    // Second pass find matches
    // We can multithread it
    pHashLinear->IsMemberString(size1, strWidth1, strWidth2, pInput1, pBoolOutput, pOutput, isUnicode);

    LOGGING("IsMemberHashString32  done\n");

    delete pHashLinear;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void IsMemberHashString64(int64_t size1, int64_t strWidth1, const char * pInput1, int64_t size2, int64_t strWidth2,
                          const char * pInput2, int64_t * pOutput, int8_t * pBoolOutput, HASH_MODE hashMode, int64_t hintSize,
                          bool isUnicode)
{
    CHashLinear<uint64_t, int64_t> * pHashLinear = new CHashLinear<uint64_t, int64_t>(hashMode);
    pHashLinear->MakeHashLocationString(size2, pInput2, strWidth2, hintSize, isUnicode);
    pHashLinear->IsMemberString(size1, strWidth1, strWidth2, pInput1, pBoolOutput, (int64_t *)pOutput, isUnicode);
    delete pHashLinear;
}

//-----------------------------------------------------------------------------------------
// Returns 8/16/32/64 bit indexes
template <typename KEY_TYPE>
bool MergePreBinned(int64_t size1, KEY_TYPE * pKey1, void * pInVal1, int64_t size2, KEY_TYPE * pKey2, void * pInVal2,
                    KEY_TYPE * pOutput, int64_t totalUniqueSize, HASH_MODE hashMode, int32_t dtype)
{
    bool success = true;

    LOGGING("AlignCategorical32 dtype: %d  size1: %lld  size2: %lld\n", dtype, size1, size2);

    switch (dtype)
    {
    CASE_NPY_INT64:

        FindLastMatchCategorical<KEY_TYPE, int64_t>(size1, size2, pKey1, pKey2, (int64_t *)pInVal1, (int64_t *)pInVal2, pOutput,
                                                    totalUniqueSize);
        break;
    CASE_NPY_INT32:
        FindLastMatchCategorical<KEY_TYPE, int32_t>(size1, size2, pKey1, pKey2, (int32_t *)pInVal1, (int32_t *)pInVal2, pOutput,
                                                    totalUniqueSize);
        break;
    case NPY_FLOAT64:
        FindLastMatchCategorical<KEY_TYPE, double>(size1, size2, pKey1, pKey2, (double *)pInVal1, (double *)pInVal2, pOutput,
                                                   totalUniqueSize);
        break;
    case NPY_FLOAT32:
        FindLastMatchCategorical<KEY_TYPE, float>(size1, size2, pKey1, pKey2, (float *)pInVal1, (float *)pInVal2, pOutput,
                                                  totalUniqueSize);
        break;
    default:
        success = false;
        break;
    }
    return success;
}

// Based on input type, calls different functions
//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------
// Returns 32 bit indexes
bool AlignHashMK32(int64_t size1, void * pInput1, void * pInVal1, int64_t size2, void * pInput2, void * pInVal2, int32_t * pOutput,
                   int64_t totalItemSize, HASH_MODE hashMode, int32_t dtype, bool isForward, bool allowExact)
{
    bool success = true;
    CHashLinear<char, int32_t> * pHashLinear = new CHashLinear<char, int32_t>(hashMode);

    switch (dtype)
    {
    CASE_NPY_INT64:

        if (isForward)
        {
            pHashLinear->FindNextMatchMK<int64_t>(size1, size2, (char *)pInput1, (char *)pInput2, (int64_t *)pInVal1,
                                                  (int64_t *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        else
        {
            pHashLinear->FindLastMatchMK<int64_t>(size1, size2, (char *)pInput1, (char *)pInput2, (int64_t *)pInVal1,
                                                  (int64_t *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        break;
    CASE_NPY_INT32:
        if (isForward)
        {
            pHashLinear->FindNextMatchMK<int32_t>(size1, size2, (char *)pInput1, (char *)pInput2, (int32_t *)pInVal1,
                                                  (int32_t *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        else
        {
            pHashLinear->FindLastMatchMK<int32_t>(size1, size2, (char *)pInput1, (char *)pInput2, (int32_t *)pInVal1,
                                                  (int32_t *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        break;
    case NPY_FLOAT64:
        if (isForward)
        {
            pHashLinear->FindNextMatchMK<double>(size1, size2, (char *)pInput1, (char *)pInput2, (double *)pInVal1,
                                                 (double *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        else
        {
            pHashLinear->FindLastMatchMK<double>(size1, size2, (char *)pInput1, (char *)pInput2, (double *)pInVal1,
                                                 (double *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        break;
    case NPY_FLOAT32:
        if (isForward)
        {
            pHashLinear->FindNextMatchMK<float>(size1, size2, (char *)pInput1, (char *)pInput2, (float *)pInVal1, (float *)pInVal2,
                                                pOutput, totalItemSize, allowExact);
        }
        else
        {
            pHashLinear->FindLastMatchMK<float>(size1, size2, (char *)pInput1, (char *)pInput2, (float *)pInVal1, (float *)pInVal2,
                                                pOutput, totalItemSize, allowExact);
        }
        break;
    default:
        success = false;
        break;
    }
    delete pHashLinear;
    return success;
}

//-----------------------------------------------------------------------------------------
// Returns 64 bit indexes
bool AlignHashMK64(int64_t size1, void * pInput1, void * pInVal1, int64_t size2, void * pInput2, void * pInVal2, int64_t * pOutput,
                   int64_t totalItemSize, HASH_MODE hashMode, int32_t dtype, bool isForward, bool allowExact)
{
    bool success = true;
    CHashLinear<char, int64_t> * pHashLinear = new CHashLinear<char, int64_t>(hashMode);

    switch (dtype)
    {
    CASE_NPY_INT64:

        if (isForward)
        {
            pHashLinear->FindNextMatchMK<int64_t>(size1, size2, (char *)pInput1, (char *)pInput2, (int64_t *)pInVal1,
                                                  (int64_t *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        else
        {
            pHashLinear->FindLastMatchMK<int64_t>(size1, size2, (char *)pInput1, (char *)pInput2, (int64_t *)pInVal1,
                                                  (int64_t *)pInVal2, pOutput, totalItemSize, allowExact);
        }

        break;
    CASE_NPY_INT32:
        if (isForward)
        {
            pHashLinear->FindNextMatchMK<int32_t>(size1, size2, (char *)pInput1, (char *)pInput2, (int32_t *)pInVal1,
                                                  (int32_t *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        else
        {
            pHashLinear->FindLastMatchMK<int32_t>(size1, size2, (char *)pInput1, (char *)pInput2, (int32_t *)pInVal1,
                                                  (int32_t *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        break;
    case NPY_FLOAT64:
        if (isForward)
        {
            pHashLinear->FindNextMatchMK<double>(size1, size2, (char *)pInput1, (char *)pInput2, (double *)pInVal1,
                                                 (double *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        else
        {
            pHashLinear->FindLastMatchMK<double>(size1, size2, (char *)pInput1, (char *)pInput2, (double *)pInVal1,
                                                 (double *)pInVal2, pOutput, totalItemSize, allowExact);
        }
        break;
    case NPY_FLOAT32:
        if (isForward)
        {
            pHashLinear->FindNextMatchMK<float>(size1, size2, (char *)pInput1, (char *)pInput2, (float *)pInVal1, (float *)pInVal2,
                                                pOutput, totalItemSize, allowExact);
        }
        else
        {
            pHashLinear->FindLastMatchMK<float>(size1, size2, (char *)pInput1, (char *)pInput2, (float *)pInVal1, (float *)pInVal2,
                                                pOutput, totalItemSize, allowExact);
        }
        break;
    default:
        success = false;
        break;
    }
    delete pHashLinear;
    return success;
}

//----------------------------------------------
// any non standard size
template <typename HASH_TYPE, typename INDEX_TYPE>
uint64_t DoLinearHash(int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, INDEX_TYPE * pIndexArray,
                      void ** pFirstArrayVoid, void ** pHashTableAny, int64_t * hashTableSize, HASH_MODE hashMode,
                      int64_t hintSize, bool * pBoolFilter)
{
    uint64_t numUnique = 0;
    CHashLinear<HASH_TYPE, INDEX_TYPE> * pHashLinear = new CHashLinear<HASH_TYPE, INDEX_TYPE>(hashMode, false);
    INDEX_TYPE * pFirstArray = (INDEX_TYPE *)pHashLinear->AllocMemory(hintSize, -2, sizeof(INDEX_TYPE) * (totalRows + 1), false);

    // Handles any size
    numUnique = pHashLinear->GroupBy(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray, hashMode, hintSize,
                                     pBoolFilter);

    *pHashTableAny = pHashLinear->pHashTableAny;
    *hashTableSize = pHashLinear->HashTableAllocSize;
    *pFirstArrayVoid = pFirstArray;
    delete pHashLinear;
    return numUnique;
}

//----------------------------------------------
// common float
template <typename HASH_TYPE, typename INDEX_TYPE>
uint64_t DoLinearHashFloat(int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, INDEX_TYPE * pIndexArray,
                           void ** pFirstArrayVoid, void ** pHashTableAny, int64_t * hashTableSize, HASH_MODE hashMode,
                           int64_t hintSize, bool * pBoolFilter)
{
    uint64_t numUnique = 0;
    CHashLinear<HASH_TYPE, INDEX_TYPE> * pHashLinear = new CHashLinear<HASH_TYPE, INDEX_TYPE>(hashMode, false);
    INDEX_TYPE * pFirstArray = (INDEX_TYPE *)pHashLinear->AllocMemory(hintSize, -1, sizeof(INDEX_TYPE) * (totalRows + 1), false);

    numUnique = pHashLinear->GroupByFloat(totalRows, totalItemSize, (HASH_TYPE *)pInput1, coreType, pIndexArray, pFirstArray,
                                          hashMode, hintSize, pBoolFilter);

    // Copy these before they get deleted
    *pHashTableAny = pHashLinear->pHashTableAny;
    *hashTableSize = pHashLinear->HashTableAllocSize;
    *pFirstArrayVoid = pFirstArray;
    delete pHashLinear;
    return numUnique;
}

//----------------------------------------------
// common types non-float
template <typename HASH_TYPE, typename INDEX_TYPE>
uint64_t DoLinearHashItemSize(int64_t totalRows, int64_t totalItemSize, const char * pInput1,
                              int coreType, // This is the numpy type code, e.g. NPY_FLOAT32
                              INDEX_TYPE * pIndexArray, void ** pFirstArrayVoid, void ** pHashTableAny, int64_t * hashTableSize,
                              HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    uint64_t numUnique = 0;

    CHashLinear<HASH_TYPE, INDEX_TYPE> * pHashLinear = new CHashLinear<HASH_TYPE, INDEX_TYPE>(hashMode, false);
    INDEX_TYPE * pFirstArray = (INDEX_TYPE *)pHashLinear->AllocMemory(hintSize, -1, sizeof(INDEX_TYPE) * (totalRows + 1), false);

    if (pFirstArray)
    {
        numUnique = pHashLinear->GroupByItemSize(totalRows, totalItemSize, (HASH_TYPE *)pInput1, coreType, pIndexArray,
                                                 pFirstArray, hashMode, hintSize, pBoolFilter);
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
template <typename INDEX_TYPE>
uint64_t GroupByInternal(void ** pFirstArray, void ** pHashTableAny, int64_t * hashTableSize,

                         int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, INDEX_TYPE * pIndexArray,
                         HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    uint64_t numUnique = 0;
    bool calculated = false;

    if (hintSize == 0)
    {
        hintSize = totalRows;
    }

    //
    // TODO: Need to add special handling for bools
    //

    // Calling the hash function will return the FirstArray

    switch (coreType)
    {
    case NPY_FLOAT32:
        {
            // so that nans compare, we tell it is uint32
            numUnique =
                DoLinearHashFloat<uint32_t, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray,
                                                        pHashTableAny, hashTableSize, hashMode, hintSize, pBoolFilter);
            calculated = true;
        }
        break;
    case NPY_FLOAT64:
        {
            // so that nans compare, we tell it is uint64
            numUnique =
                DoLinearHashFloat<uint64_t, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray,
                                                        pHashTableAny, hashTableSize, hashMode, hintSize, pBoolFilter);
            calculated = true;
        }
        break;
    }

    // Now go based on size
    if (calculated == false)
    {
        switch (totalItemSize)
        {
        case 1:
            {
                numUnique = DoLinearHashItemSize<uint8_t, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray,
                                                                      pFirstArray, pHashTableAny, hashTableSize, hashMode, 256 / 2,
                                                                      pBoolFilter);
                calculated = true;
            }
            break;
        case 2:
            {
                numUnique = DoLinearHashItemSize<uint16_t, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray,
                                                                       pFirstArray, pHashTableAny, hashTableSize, hashMode,
                                                                       65536 / 2, pBoolFilter);
                calculated = true;
            }
            break;
        case 4:
            {
                numUnique = DoLinearHashItemSize<uint32_t, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray,
                                                                       pFirstArray, pHashTableAny, hashTableSize, hashMode,
                                                                       hintSize, pBoolFilter);
                calculated = true;
            }
            break;
        case 8:
            {
                numUnique = DoLinearHashItemSize<uint64_t, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray,
                                                                       pFirstArray, pHashTableAny, hashTableSize, hashMode,
                                                                       hintSize, pBoolFilter);
                calculated = true;
            }
            break;
        }
    }

    if (calculated == false)
    {
        numUnique = DoLinearHash<uint32_t, INDEX_TYPE>(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pFirstArray,
                                                       pHashTableAny, hashTableSize, hashMode, hintSize, pBoolFilter);
    }

    return numUnique;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
uint64_t GroupBy32Super(int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, int32_t * pIndexArray,
                        int32_t * pNextArray, int32_t * pUniqueArray, int32_t * pUniqueCountArray, HASH_MODE hashMode,
                        int64_t hintSize, bool * pBoolFilter)
{
    uint64_t numUnique = 0;

    CHashLinear<uint32_t, int32_t> * pHashLinear = new CHashLinear<uint32_t, int32_t>(hashMode);
    numUnique = pHashLinear->GroupBySuper(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pNextArray, pUniqueArray,
                                          pUniqueCountArray, hashMode, hintSize, pBoolFilter);
    delete pHashLinear;

    return numUnique;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
uint64_t GroupBy64Super(int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, int64_t * pIndexArray,
                        int64_t * pNextArray, int64_t * pUniqueArray, int64_t * pUniqueCountArray, HASH_MODE hashMode,
                        int64_t hintSize, bool * pBoolFilter)
{
    CHashLinear<uint64_t, int64_t> * pHashLinear = new CHashLinear<uint64_t, int64_t>(hashMode);
    uint64_t numUnique = pHashLinear->GroupBySuper(totalRows, totalItemSize, pInput1, coreType, pIndexArray, pNextArray,
                                                   pUniqueArray, pUniqueCountArray, hashMode, hintSize, pBoolFilter);

    delete pHashLinear;
    return numUnique;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
uint64_t Unique32(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                  int32_t * pIndexArray, int32_t * pCountArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    CHashLinear<uint32_t, int32_t> * pHashLinear = new CHashLinear<uint32_t, int32_t>(hashMode);
    uint64_t numUnique =
        pHashLinear->Unique(totalRows, totalItemSize, pInput1, pIndexArray, pCountArray, hashMode, hintSize, pBoolFilter);

    delete pHashLinear;
    return numUnique;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
uint64_t Unique64(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                  int64_t * pIndexArray, int64_t * pCountArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    CHashLinear<uint32_t, int64_t> * pHashLinear = new CHashLinear<uint32_t, int64_t>(hashMode);
    uint64_t numUnique =
        pHashLinear->Unique(totalRows, totalItemSize, pInput1, pIndexArray, pCountArray, hashMode, hintSize, pBoolFilter);

    delete pHashLinear;
    return numUnique;
}

//-----------------------------------------------------------------------------------------
void MultiKeyRollingStep2Delete(void * pHashLinearLast)
{
    CHashLinear<uint64_t, int64_t> * pHashLinear;

    // If we are rolling, they will pass back what we returned
    if (pHashLinearLast)
    {
        pHashLinear = (CHashLinear<uint64_t, int64_t> *)pHashLinearLast;
        delete pHashLinear;
    }
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void * MultiKeyRollingStep2(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                            int64_t * pIndexArray, int64_t * pRunningCountArray, HASH_MODE hashMode, int64_t hintSize,
                            uint64_t * numUnique, // returned back
                            void * pHashLinearLast)
{
    CHashLinear<uint64_t, int64_t> * pHashLinear;

    // If we are rolling, they will pass back what we returned
    if (pHashLinearLast)
    {
        pHashLinear = (CHashLinear<uint64_t, int64_t> *)pHashLinearLast;
        hintSize = -1;
        LOGGING("Rolling using existing! %llu\n", pHashLinear->NumUnique);
    }
    else
    {
        pHashLinear = new CHashLinear<uint64_t, int64_t>(hashMode);
    }

    pHashLinear->MultiKeyRolling(totalRows, totalItemSize, pInput1, pIndexArray, pRunningCountArray, hashMode, hintSize);
    *numUnique = pHashLinear->NumUnique;

    // Allow to keep rolling
    return pHashLinear;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void * MultiKeyHash32(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                      int32_t * pIndexArray, int32_t * pRunningCountArray, int32_t * pPrevArray, int32_t * pNextArray,
                      int32_t * pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    CHashLinear<uint32_t, int32_t> * pHashLinear = new CHashLinear<uint32_t, int32_t>(hashMode);
    pHashLinear->MakeHashLocationMultiKey(totalRows, totalItemSize, pInput1, pIndexArray, pRunningCountArray, pPrevArray,
                                          pNextArray, pFirstArray, hashMode, hintSize, pBoolFilter);
    delete pHashLinear;
    return NULL;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void * MultiKeyHash64(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                      int64_t * pIndexArray, int64_t * pRunningCountArray, int64_t * pPrevArray, int64_t * pNextArray,
                      int64_t * pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    CHashLinear<uint32_t, int64_t> * pHashLinear = new CHashLinear<uint32_t, int64_t>(hashMode);
    pHashLinear->MakeHashLocationMultiKey(totalRows, totalItemSize, pInput1, pIndexArray, pRunningCountArray, pPrevArray,
                                          pNextArray, pFirstArray, hashMode, hintSize, pBoolFilter);
    delete pHashLinear;
    return NULL;
}

//-----------------------------------------------------------------------------------------
// Should follow categorical size checks
//
void IsMemberHashString32Pre(PyArrayObject ** indexArray, PyArrayObject * inArr1, int64_t size1, int64_t strWidth1,
                             const char * pInput1, int64_t size2, int64_t strWidth2, const char * pInput2, int8_t * pBoolOutput,
                             HASH_MODE hashMode, int64_t hintSize, bool isUnicode)
{
    int64_t size = size1;
    if (size2 > size)
    {
        size = size2;
    }

    if (size < 100)
    {
        *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT8);
        int8_t * pDataOut2 = (int8_t *)PyArray_BYTES(*indexArray);
        IsMemberHashString32<int8_t>(size1, strWidth1, (const char *)pInput1, size2, strWidth2, (const char *)pInput2, pDataOut2,
                                     pBoolOutput, HASH_MODE(hashMode), hintSize, isUnicode);
    }
    else if (size < 30000)
    {
        *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT16);
        int16_t * pDataOut2 = (int16_t *)PyArray_BYTES(*indexArray);
        IsMemberHashString32<int16_t>(size1, strWidth1, (const char *)pInput1, size2, strWidth2, (const char *)pInput2, pDataOut2,
                                      pBoolOutput, HASH_MODE(hashMode), hintSize, isUnicode);
    }
    else if (size < 2000000000)
    {
        *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
        int32_t * pDataOut2 = (int32_t *)PyArray_BYTES(*indexArray);
        IsMemberHashString32<int32_t>(size1, strWidth1, (const char *)pInput1, size2, strWidth2, (const char *)pInput2, pDataOut2,
                                      pBoolOutput, HASH_MODE(hashMode), hintSize, isUnicode);
    }
    else
    {
        *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT64);
        int64_t * pDataOut2 = (int64_t *)PyArray_BYTES(*indexArray);
        IsMemberHashString32<int64_t>(size1, strWidth1, (const char *)pInput1, size2, strWidth2, (const char *)pInput2, pDataOut2,
                                      pBoolOutput, HASH_MODE(hashMode), hintSize, isUnicode);
    }
}

//-----------------------------------------------------------------------------------------
// Should follow categorical size checks
//
int64_t IsMemberCategoricalHashStringPre(PyArrayObject ** indexArray, PyArrayObject * inArr1, int64_t size1, int64_t strWidth1,
                                         const char * pInput1, int64_t size2, int64_t strWidth2, const char * pInput2,
                                         HASH_MODE hashMode, int64_t hintSize, bool isUnicode)
{
    int64_t missed = 0;

    if (size2 < 100)
    {
        *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT8);
        int8_t * pDataOut2 = (int8_t *)PyArray_BYTES(*indexArray);
        missed = IsMemberHashStringCategorical<int8_t>(size1, strWidth1, (const char *)pInput1, size2, strWidth2,
                                                       (const char *)pInput2, pDataOut2, HASH_MODE(hashMode), hintSize, isUnicode);
    }
    else if (size2 < 30000)
    {
        *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT16);
        int16_t * pDataOut2 = (int16_t *)PyArray_BYTES(*indexArray);
        missed =
            IsMemberHashStringCategorical<int16_t>(size1, strWidth1, (const char *)pInput1, size2, strWidth2,
                                                   (const char *)pInput2, pDataOut2, HASH_MODE(hashMode), hintSize, isUnicode);
    }
    else if (size2 < 2000000000)
    {
        *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT32);
        int32_t * pDataOut2 = (int32_t *)PyArray_BYTES(*indexArray);
        missed =
            IsMemberHashStringCategorical<int32_t>(size1, strWidth1, (const char *)pInput1, size2, strWidth2,
                                                   (const char *)pInput2, pDataOut2, HASH_MODE(hashMode), hintSize, isUnicode);
    }
    else
    {
        *indexArray = AllocateLikeNumpyArray(inArr1, NPY_INT64);
        int64_t * pDataOut2 = (int64_t *)PyArray_BYTES(*indexArray);
        missed =
            IsMemberHashStringCategorical<int64_t>(size1, strWidth1, (const char *)pInput1, size2, strWidth2,
                                                   (const char *)pInput2, pDataOut2, HASH_MODE(hashMode), hintSize, isUnicode);
    }

    return missed;
}

//===================================================================================================

typedef void (*ISMEMBER_MK)(void * pHashLinearVoid, int64_t arraySize, void * pInputT, void * pInput2T, int8_t * pBoolOutput,
                            void * pLocationOutputU, int64_t totalItemSize);

//--------------------------------------------------------------------
struct IMMK_CALLBACK
{
    ISMEMBER_MK anyIMMKCallback;

    void * pHashLinearVoid;

    int64_t size1;
    void * pInput1;
    int64_t size2; // size of the second argument
    void * pInput2;
    int8_t * pBoolOutput;
    void * pOutput;
    int64_t totalItemSize;
    int64_t typeSizeOut;

} stIMMKCallback;

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static bool IMMKThreadCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
{
    bool didSomeWork = false;
    IMMK_CALLBACK * Callback = (IMMK_CALLBACK *)pstWorkerItem->WorkCallbackArg;

    char * pInput1 = (char *)Callback->pInput1;
    char * pOutput = (char *)Callback->pOutput;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
    {
        int64_t inputAdj = pstWorkerItem->BlockSize * workBlock * Callback->totalItemSize;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->typeSizeOut;
        int64_t boolAdj = pstWorkerItem->BlockSize * workBlock;

        Callback->anyIMMKCallback(Callback->pHashLinearVoid, lenX, pInput1 + inputAdj, Callback->pInput2,
                                  Callback->pBoolOutput + boolAdj, pOutput + outputAdj, Callback->totalItemSize);

        // Indicate we completed a block
        didSomeWork = true;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        // printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
template <typename U>
void IsMemberHashMK(int64_t size1, void * pInput1,
                    int64_t size2, // size of the second argument
                    void * pInput2, int8_t * pBoolOutput, U * pOutput, int64_t totalItemSize, int64_t hintSize, HASH_MODE hashMode)
{
    LOGGING("ismember hash  sz1:%lld  sz2:%lld   totalitemsize:%lld  %p  %p\n", size1, size2, totalItemSize, pInput1, pInput2);

    CHashLinear<char, U> * pHashLinear = new CHashLinear<char, U>(hashMode);
    pHashLinear->MakeHashLocationMK(size2, (char *)pInput2, totalItemSize, hintSize);

    stMATH_WORKER_ITEM * pWorkItem = g_cMathWorker->GetWorkItem(size1);
    ISMEMBER_MK pFunction = IsMemberMK<char, U>;

    if (pWorkItem == NULL)
    {
        // Threading not allowed for this work item, call it directly from main
        // thread
        pFunction(pHashLinear, size1, (char *)pInput1, (char *)pInput2, pBoolOutput, pOutput, totalItemSize);
    }
    else
    {
        // Each thread will call this routine with the callbackArg
        pWorkItem->DoWorkCallback = IMMKThreadCallback;

        pWorkItem->WorkCallbackArg = &stIMMKCallback;

        stIMMKCallback.pHashLinearVoid = pHashLinear;
        stIMMKCallback.anyIMMKCallback = pFunction;
        stIMMKCallback.size1 = size1;
        stIMMKCallback.pInput1 = pInput1;
        stIMMKCallback.size2 = size2;
        stIMMKCallback.pInput2 = pInput2;
        stIMMKCallback.pBoolOutput = pBoolOutput;
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
// floats are 4 bytes and will be handled like int32_t or UIN32
void IsMemberHashMKPre(PyArrayObject ** indexArray, int64_t size1, void * pInput1,
                       int64_t size2, // size of the second argument
                       void * pInput2, int8_t * pBoolOutput, int64_t totalItemSize, int64_t hintSize, HASH_MODE hashMode)
{
    int64_t size = size1;

    if (size2 > size1)
    {
        size = size2;
    }

    LOGGING("!!! in multikey ismember %lld  %d   size2:%lld  size1:%lld  size:%lld\n", hintSize, hashMode, size2, size1, size);

    if (size < 100)
    {
        *indexArray = AllocateNumpyArray(1, (npy_intp *)&size1, NPY_INT8);
        if (*indexArray)
        {
            int8_t * pOutput = (int8_t *)PyArray_BYTES(*indexArray);
            IsMemberHashMK<int8_t>(size1, pInput1, size2, pInput2, pBoolOutput, pOutput, totalItemSize, hintSize, hashMode);
        }
    }
    else if (size < 30000)
    {
        *indexArray = AllocateNumpyArray(1, (npy_intp *)&size1, NPY_INT16);
        if (*indexArray)
        {
            int16_t * pOutput = (int16_t *)PyArray_BYTES(*indexArray);
            IsMemberHashMK<int16_t>(size1, pInput1, size2, pInput2, pBoolOutput, pOutput, totalItemSize, hintSize, hashMode);
        }
    }
    else if (size < 2000000000)
    {
        *indexArray = AllocateNumpyArray(1, (npy_intp *)&size1, NPY_INT32);
        if (*indexArray)
        {
            int32_t * pOutput = (int32_t *)PyArray_BYTES(*indexArray);
            IsMemberHashMK<int32_t>(size1, pInput1, size2, pInput2, pBoolOutput, pOutput, totalItemSize, hintSize, hashMode);
        }
    }
    else
    {
        *indexArray = AllocateNumpyArray(1, (npy_intp *)&size1, NPY_INT64);
        if (*indexArray)
        {
            int64_t * pOutput = (int64_t *)PyArray_BYTES(*indexArray);
            IsMemberHashMK<int64_t>(size1, pInput1, size2, pInput2, pBoolOutput, pOutput, totalItemSize, hintSize, hashMode);
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
//    Returns: bool array and optional int32_t location array
//       bool array: True if first arg found in second arg
//       index: index location of where first arg found in second arg  (index
//       into second arg)

/*
NOTE ON 'row' parameter
appears to take all the numbers in the a and check to see if it exists in b ?
has to be an exact match-- all the elements in row 1 of A have to match all the
elements of any row in B in the same order?

>> b

b =

1.00          2.00          3.00          4.00          5.00          6.00 7.00
8.00          9.00         10.00 4.00          5.00          6.00          7.00
8.00          9.00         10.00         11.00         12.00         13.00 14.00
15.00         16.00         17.00         18.00         19.00         20.00
21.00         22.00         23.00
11.00         12.00         13.00         14.00         15.00         16.00
17.00         18.00         19.00         20.00

>> a

a =

1.00          2.00          3.00          4.00          5.00          6.00 7.00
8.00          9.00         10.00 1.00          2.00          3.00          4.00
5.00          6.00          7.00          8.00          9.00         10.00 11.00
12.00         13.00         14.00         15.00         16.00         17.00
18.00         19.00         20.00

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

PyObject * IsMember32(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr1 = NULL;
    PyArrayObject * inArr2 = NULL;
    int hashMode = 2;
    int64_t hintSize = 0;

    Py_ssize_t tupleSize = PyTuple_GET_SIZE(args);

    LOGGING("IsMember32 called with %lld args\n", tupleSize);

    if (tupleSize <= 1)
    {
        return NULL;
    }

    if (tupleSize == 2)
    {
        if (! PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &inArr1, &PyArray_Type, &inArr2))
            return NULL;
    }
    else if (tupleSize == 3)
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
                     "IsMember32 cannot handle unicode, object, void strings, "
                     "please convert to np.chararray");
        return NULL;
    }

    int64_t arraySize1 = ArrayLength(inArr1);
    int64_t arraySize2 = ArrayLength(inArr2);

    PyArrayObject * boolArray = AllocateLikeNumpyArray(inArr1, NPY_BOOL);

    if (boolArray)
    {
        void * pDataIn1 = PyArray_BYTES(inArr1);
        void * pDataIn2 = PyArray_BYTES(inArr2);

        int8_t * pDataOut1 = (int8_t *)PyArray_BYTES(boolArray);

        PyArrayObject * indexArray = NULL;

        LOGGING("Size array1: %llu   array2: %llu\n", arraySize1, arraySize2);

        if (arrayType1 >= NPY_STRING)
        {
            LOGGING("Calling string!\n");

            // Performance gain: if STRING and itemsize matches and itemsize is 1 or 2
            // --> Send to IsMemberHash32
            IsMemberHashString32Pre(&indexArray, inArr1, arraySize1, sizeType1, (const char *)pDataIn1, arraySize2, sizeType2,
                                    (const char *)pDataIn2, pDataOut1, HASH_MODE(hashMode), hintSize, arrayType1 == NPY_UNICODE);
        }
        else
        {
            if (arrayType1 == NPY_FLOAT32 || arrayType1 == NPY_FLOAT64)
            {
                LOGGING("Calling float!\n");
                sizeType1 += 100;
            }

            int dtype = NPY_INT8;

            if (arraySize2 < 100)
            {
                dtype = NPY_INT8;
            }
            else if (arraySize2 < 30000)
            {
                dtype = NPY_INT16;
            }
            else if (arraySize2 < 2000000000)
            {
                dtype = NPY_INT32;
            }
            else
            {
                dtype = NPY_INT64;
            }

            indexArray = AllocateLikeNumpyArray(inArr1, dtype);

            // make sure allocation succeeded
            if (indexArray)
            {
                void * pDataOut2 = PyArray_BYTES(indexArray);
                switch (dtype)
                {
                case NPY_INT8:
                    IsMemberHash32<int8_t>(arraySize1, pDataIn1, arraySize2, pDataIn2, (int8_t *)pDataOut2, pDataOut1, sizeType1,
                                           HASH_MODE(hashMode), hintSize);
                    break;
                case NPY_INT16:
                    IsMemberHash32<int16_t>(arraySize1, pDataIn1, arraySize2, pDataIn2, (int16_t *)pDataOut2, pDataOut1, sizeType1,
                                            HASH_MODE(hashMode), hintSize);
                    break;
                CASE_NPY_INT32:
                    IsMemberHash32<int32_t>(arraySize1, pDataIn1, arraySize2, pDataIn2, (int32_t *)pDataOut2, pDataOut1, sizeType1,
                                            HASH_MODE(hashMode), hintSize);
                    break;
                CASE_NPY_INT64:
                    IsMemberHash32<int64_t>(arraySize1, pDataIn1, arraySize2, pDataIn2, (int64_t *)pDataOut2, pDataOut1, sizeType1,
                                            HASH_MODE(hashMode), hintSize);
                    break;
                }
            }
        }

        if (indexArray)
        {
            PyObject * retObject = Py_BuildValue("(OO)", boolArray, indexArray);
            Py_DECREF((PyObject *)boolArray);
            Py_DECREF((PyObject *)indexArray);

            return (PyObject *)retObject;
        }
    }
    // out of memory
    return NULL;
}

/**
 * @brief
 *
 * @tparam _Index The type of the integer indices used and returned by this
 * function. Should be int32_t or int64_t.
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
 * @return uint64_t
 */
#if defined(__GNUC__) && __GNUC__ < 5
// Workaround for old versions of gcc which don't have enable_if_t
template <typename _Index>
#else
// removed std::is_integral due to debian compilers
template <typename _Index>
// template <typename _Index,
//   std::enable_if_t<std::is_integral<_Index>::value, int> = 0>
#endif
static uint64_t GroupByImpl(const int64_t partitionLength, // may be 0
                            int64_t * const pCutOffs,      // may be NULL
                            const int64_t totalRows, const int64_t totalItemSize, const char * const pInput1, const int coreType,
                            _Index * const pIndexArray, PyArrayObject ** pFirstArrayObject, const HASH_MODE hashMode,
                            const int64_t hintSize, bool * const pBoolFilter)
{
    _Index * pFirstArray = nullptr;
    void * pHashTableAny = nullptr;
    int64_t hashTableSize = 0;

    if (partitionLength)
    {
        // turn off threading? or memory allocations?
        // need to pass more info
        // If this is a partitioned groupby then
        // the pIndexArray must be divided
        // the firstArray
        // the pSuperArray based on totalItemSize
        // when groupby is complete, all 0s must be kept as 0s
        // otherwise the unique count PERDAY is used this returned in another array
        // to get the slices
        //
        // pFirstArray -- copytosmallerarray needs to change
        //
        struct PARTITION_GB
        {
            _Index * pFirstArray;
            void * pHashTableAny;
            int64_t HashTableSize;
            int64_t NumUnique;
            int64_t TotalRows;
        };

        // MT callback
        struct MKGBCallbackStruct
        {
            PARTITION_GB * pPartitions;
            int64_t PartitionLength;
            int64_t * pCutOffs;

            int64_t TotalRows;
            int64_t TotalItemSize;
            const char * pInput1;

            int CoreType;
            _Index * pIndexArray;
            HASH_MODE HashMode;
            int64_t HintSize;
            bool * pBoolFilter;
        };

        // This is the routine that will be called back from multiple threads
        auto lambdaMKGBCallback = [](void * callbackArgT, int core, int64_t count) -> bool
        {
            auto * cb = static_cast<MKGBCallbackStruct *>(callbackArgT);

            int64_t * pCutOffs = cb->pCutOffs;
            PARTITION_GB * pPartition = &cb->pPartitions[count];

            int64_t partOffset = 0;
            int64_t partLength = pCutOffs[count];
            bool * pBoolFilter = cb->pBoolFilter;
            auto * pIndexArray = cb->pIndexArray;
            const char * pInput1 = cb->pInput1;

            // use the cutoffs to calculate partition length
            if (count > 0)
            {
                partOffset = pCutOffs[count - 1];
            }
            partLength -= partOffset;
            pPartition->TotalRows = partLength;

            LOGGING(
                "[%d] MKGB %lld  cutoff:%lld  offset: %lld length:%lld  "
                "hintsize:%lld\n",
                core, count, pCutOffs[count], partOffset, partLength, cb->HintSize);

            // NOW SHIFT THE DATA ---------------
            if (pBoolFilter)
            {
                pBoolFilter += partOffset;
            }
            pIndexArray += partOffset;
            pInput1 += (partOffset * cb->TotalItemSize);

            // NOW HASH the data
            pPartition->NumUnique = (int64_t)GroupByInternal<_Index>(
                // These three are returned, they have to be deallocated
                reinterpret_cast<void **>(&pPartition->pFirstArray), &pPartition->pHashTableAny, &pPartition->HashTableSize,

                partLength, cb->TotalItemSize, pInput1,
                cb->CoreType, // set to -1 for unknown
                pIndexArray, cb->HashMode, cb->HintSize, pBoolFilter);

            return true;
        };

        PARTITION_GB * pPartitions = (PARTITION_GB *)WORKSPACE_ALLOC(partitionLength * sizeof(PARTITION_GB));

        // TODO: Initialize the struct using different syntax so fields which aren't
        // meant to be modified can be marked 'const'.
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
        g_cMathWorker->NoCaching = true;

        g_cMathWorker->DoMultiThreadedWork(static_cast<int>(partitionLength), lambdaMKGBCallback, &stMKGBCallback);

        // firstArray = *stMKGBCallback.pFirstArray;
        // NOW COLLECT ALL THE RESULTS

        PyArrayObject * cutoffsArray = AllocateNumpyArray(1, (npy_intp *)&partitionLength, NPY_INT64);
        CHECK_MEMORY_ERROR(cutoffsArray);
        if (! cutoffsArray)
            return 0;

        int64_t * pCutOffs = (int64_t *)PyArray_BYTES(cutoffsArray);

        int64_t totalUniques = 0;
        for (int i = 0; i < partitionLength; i++)
        {
            totalUniques += pPartitions[i].NumUnique;
            pCutOffs[i] = totalUniques;
        }

        PyArrayObject * firstArray = AllocateNumpyArray(1, (npy_intp *)&totalUniques, numpy_type_code<_Index>::value);
        CHECK_MEMORY_ERROR(firstArray);
        if (! firstArray)
            return 0;

        _Index * pFirstArray = (_Index *)PyArray_BYTES(firstArray);

        int64_t startpos = 0;

        // Clean up------------------------------
        for (int i = 0; i < partitionLength; i++)
        {
            memcpy(&pFirstArray[startpos], pPartitions[i].pFirstArray, pPartitions[i].NumUnique * sizeof(_Index));
            startpos += pPartitions[i].NumUnique;
        }

        // Clean up------------------------------
        for (int i = 0; i < partitionLength; i++)
        {
            WorkSpaceFreeAllocLarge(pPartitions[i].pHashTableAny, pPartitions[i].HashTableSize);
        }

        // turn caching back on -----------------------------------------
        g_cMathWorker->NoCaching = false;

        WORKSPACE_FREE(pPartitions);

        PyObject * pyFirstList = PyList_New(2);
        PyList_SET_ITEM(pyFirstList, 0, (PyObject *)firstArray);
        PyList_SET_ITEM(pyFirstList, 1, (PyObject *)cutoffsArray);

        *pFirstArrayObject = (PyArrayObject *)pyFirstList;
        return totalUniques;
    }
    else
    {
        // NOTE: because the linear is heavily optimized, it knows how to reuse a
        // large memory allocation This makes for a more complicated GroupBy as in
        // parallel mode, it has to shut down the low level caching Further, the
        // size of the first array is not known until the unique count is known

        uint64_t numUnique = GroupByInternal<_Index>(

            reinterpret_cast<void **>(&pFirstArray), &pHashTableAny, &hashTableSize,

            totalRows, totalItemSize, pInput1, coreType, pIndexArray, hashMode, hintSize, pBoolFilter);

        // Move uniques into proper array size
        // Free HashTableAllocSize
        // printf("Got back %p %lld\n", pFirstArray, hashTableSize);
        *pFirstArrayObject = CopyToSmallerArray<_Index>(pFirstArray, numUnique, totalRows);
        WorkSpaceFreeAllocLarge(pHashTableAny, hashTableSize);
        return numUnique;
    }
}

//===================================================================================================
uint64_t GroupBy32(int64_t partitionLength, // may be 0
                   int64_t * pCutOffs,      // may be NULL
                   int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, void * pIndexArray,
                   PyArrayObject ** pFirstArrayObject, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    // Call the templated implementation of this function.
    using index_type = int32_t;
    return GroupByImpl<index_type>(partitionLength, pCutOffs, totalRows, totalItemSize, pInput1, coreType,
                                   static_cast<index_type *>(pIndexArray), pFirstArrayObject, hashMode, hintSize, pBoolFilter);
}

//===================================================================================================
uint64_t GroupBy64(int64_t partitionLength, // may be 0
                   int64_t * pCutOffs,      // may be NULL
                   int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, void * pIndexArray,
                   PyArrayObject ** pFirstArrayObject, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter)
{
    // Call the templated implementation of this function.
    using index_type = int64_t;
    return GroupByImpl<index_type>(partitionLength, pCutOffs, totalRows, totalItemSize, pInput1, coreType,
                                   static_cast<index_type *>(pIndexArray), pFirstArrayObject, hashMode, hintSize, pBoolFilter);
}

//------------------------------------------------------------------------
// NOTE: Look at this code... fastpath for merge_asof
PyObject * MergeBinnedAndSorted(PyObject * self, PyObject * args)
{
    PyArrayObject * key1;
    PyArrayObject * key2;
    PyArrayObject * pvalArray1;
    PyArrayObject * pvalArray2;
    int64_t totalUniqueSize;

    if (! PyArg_ParseTuple(args, "O!O!O!O!L", &PyArray_Type, &key1, &PyArray_Type, &key2, &PyArray_Type, &pvalArray1,
                           &PyArray_Type, &pvalArray2, &totalUniqueSize))
    {
        return NULL;
    }

    LOGGING("Unique size %lld\n", totalUniqueSize);
    int32_t dtype1 = ObjectToDtype((PyArrayObject *)pvalArray1);
    int32_t dtype2 = ObjectToDtype((PyArrayObject *)pvalArray2);

    if (dtype1 < 0)
    {
        PyErr_Format(PyExc_ValueError,
                     "MergeBinnedAndSorted data types are not understood "
                     "dtype.num: %d vs %d",
                     dtype1, dtype2);
        return NULL;
    }

    if (dtype1 != dtype2)
    {
        // Check for when numpy has 7==9 or 8==10 on Linux 5==7, 6==8 on Windows
        if (! ((dtype1 <= NPY_ULONGLONG && dtype2 <= NPY_ULONGLONG) && ((dtype1 & 1) == (dtype2 & 1)) &&
               PyArray_ITEMSIZE((PyArrayObject *)pvalArray1) == PyArray_ITEMSIZE((PyArrayObject *)pvalArray2)))
        {
            PyErr_Format(PyExc_ValueError,
                         "MergeBinnedAndSorted data types are not the same "
                         "dtype.num: %d vs %d",
                         dtype1, dtype2);
            return NULL;
        }
    }

    void * pVal1 = PyArray_BYTES(pvalArray1);
    void * pVal2 = PyArray_BYTES(pvalArray2);
    void * pKey1 = PyArray_BYTES(key1);
    void * pKey2 = PyArray_BYTES(key2);

    PyArrayObject * indexArray = (PyArrayObject *)Py_None;
    bool isIndex32 = true;
    bool success = false;

    indexArray = AllocateLikeNumpyArray(key1, dtype1);

    if (indexArray)
    {
        switch (dtype1)
        {
        case NPY_INT8:
            success = MergePreBinned<int8_t>(ArrayLength(key1), (int8_t *)pKey1, pVal1, ArrayLength(key2), (int8_t *)pKey2, pVal2,
                                             (int8_t *)PyArray_BYTES(indexArray), totalUniqueSize, HASH_MODE_MASK, dtype1);
            break;
        case NPY_INT16:
            success =
                MergePreBinned<int16_t>(ArrayLength(key1), (int16_t *)pKey1, pVal1, ArrayLength(key2), (int16_t *)pKey2, pVal2,
                                        (int16_t *)PyArray_BYTES(indexArray), totalUniqueSize, HASH_MODE_MASK, dtype1);
            break;
        CASE_NPY_INT32:
            success =
                MergePreBinned<int32_t>(ArrayLength(key1), (int32_t *)pKey1, pVal1, ArrayLength(key2), (int32_t *)pKey2, pVal2,
                                        (int32_t *)PyArray_BYTES(indexArray), totalUniqueSize, HASH_MODE_MASK, dtype1);
            break;
        CASE_NPY_INT64:

            success =
                MergePreBinned<int64_t>(ArrayLength(key1), (int64_t *)pKey1, pVal1, ArrayLength(key2), (int64_t *)pKey2, pVal2,
                                        (int64_t *)PyArray_BYTES(indexArray), totalUniqueSize, HASH_MODE_MASK, dtype1);
            break;
        }
    }

    if (! success)
    {
        PyErr_Format(PyExc_ValueError, "MultiKeyAlign failed.  Only accepts int32_t,int64_t,float32,float64");
        return NULL;
    }
    return (PyObject *)indexArray;
}
