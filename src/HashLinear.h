#pragma once
#include <assert.h>

PyObject * IsMember32(PyObject * self, PyObject * args);
PyObject * IsMember64(PyObject * self, PyObject * args);
PyObject * IsMemberCategorical(PyObject * self, PyObject * args);

enum HASH_MODE
{
    HASH_MODE_PRIME = 1,
    HASH_MODE_MASK = 2
};

struct UINT128
{
    uint64_t _val1;
    uint64_t _val2;
};

void * IsMemberHashMK32(int64_t size1, void * pInput1, int64_t size2, void * pInput2, int32_t * pOutput, int8_t * pBooleanOutput,
                        int64_t sizeType, int64_t hintSize, HASH_MODE hashMode);

int64_t IsMemberHashCategorical(int64_t size1, void * pInput1, int64_t size2, void * pInput2, int32_t * pOutput, int32_t sizeType,
                                HASH_MODE hashMode, int64_t hintSize);

int64_t IsMemberHashCategorical64(int64_t size1, void * pInput1, int64_t size2, void * pInput2, int64_t * pOutput,
                                  int32_t sizeType, HASH_MODE hashMode, int64_t hintSize);

template <typename U>
void * IsMemberHash32(int64_t size1, void * pInput1, int64_t size2, void * pInput2, U * pOutput, int8_t * pBooleanOutput,
                      int32_t sizeType, HASH_MODE hashMode, int64_t hintSize);

void * IsMemberHash64(int64_t size1, void * pInput1, int64_t size2, void * pInput2, int64_t * pOutput, int8_t * pBooleanOutput,
                      int32_t sizeType, HASH_MODE hashMode, int64_t hintSize);

int64_t IsMemberCategoricalHashStringPre(PyArrayObject ** indexArray, PyArrayObject * inArr1, int64_t size1, int64_t strWidth1,
                                         const char * pInput1, int64_t size2, int64_t strWidth2, const char * pInput2,
                                         HASH_MODE hashMode, int64_t hintSize, bool isUnicode);

void IsMemberHashString32Pre(PyArrayObject ** indexArray, PyArrayObject * inArr1, int64_t size1, int64_t strWidth1,
                             const char * pInput1, int64_t size2, int64_t strWidth2, const char * pInput2, int8_t * pBooleanOutput,
                             HASH_MODE hashMode, int64_t hintSize, bool isUnicode);

void IsMemberHashMKPre(PyArrayObject ** indexArray, int64_t size1, void * pInput1, int64_t size2, void * pInput2,
                       int8_t * pBooleanOutput, int64_t totalItemSize, int64_t hintSize, HASH_MODE hashMode);

void IsMemberHashString64(int64_t size1, int64_t strWidth1, const char * pInput1, int64_t size2, int64_t strWidth2,
                          const char * pInput2, int64_t * pOutput, int8_t * pBooleanOutput, HASH_MODE hashMode, int64_t hintSize,
                          bool isUnicode);

template <typename U, typename V>
void FindLastMatchCategorical(int64_t arraySize1, int64_t arraySize2, U * pKey1, U * pKey2, V * pVal1, V * pVal2,
                              U * pLocationOutput, int64_t totalUniqueSize);

template <typename KEY_TYPE>
bool MergePreBinned(int64_t size1, KEY_TYPE * pKey1, void * pInVal1, int64_t size2, KEY_TYPE * pKey2, void * pInVal2,
                    KEY_TYPE * pOutput, int64_t totalUniqueSize, HASH_MODE hashMode, int32_t dtype);

bool AlignHashMK32(int64_t size1, void * pInput1, void * pInVal1, int64_t size2, void * pInput2, void * pInVal2, int32_t * pOutput,
                   int64_t totalItemSize, HASH_MODE hashMode, int32_t dtype, bool isForward, bool allowExact);

bool AlignHashMK64(int64_t size1, void * pInput1, void * pInVal1, int64_t size2, void * pInput2, void * pInVal2, int64_t * pOutput,
                   int64_t totalItemSize, HASH_MODE hashMode, int32_t dtype, bool isForward, bool allowExact);

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
template <typename INDEX_TYPE>
uint64_t GroupByInternal(void ** pFirstArrayVoid, void ** pHashTableAny, int64_t * hashTableSize,

                         int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, INDEX_TYPE * pIndexArray,
                         HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

//===================================================================================================
// Declare callback for either int32_t or int64_t
typedef uint64_t (*GROUPBYCALL)(int64_t partitionLength, // may be 0
                                int64_t * pCutOffs,      // may be NULL
                                int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, void * pIndexArray,
                                PyArrayObject ** pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

uint64_t GroupBy32(int64_t partitionLength, // may be 0
                   int64_t * pCutOffs,      // may be NULL
                   int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, void * pIndexArray,
                   PyArrayObject ** pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

uint64_t GroupBy64(int64_t partitionLength, // may be 0
                   int64_t * pCutOffs,      // may be NULL
                   int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, void * pIndexArray,
                   PyArrayObject ** pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
uint64_t GroupBy32Super(int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, int32_t * pIndexArray,
                        int32_t * pNextArray, int32_t * pUniqueArray, int32_t * pUniqueCountArray, HASH_MODE hashMode,
                        int64_t hintSize, bool * pBoolFilter);

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
uint64_t GroupBy64Super(int64_t totalRows, int64_t totalItemSize, const char * pInput1, int coreType, int64_t * pIndexArray,
                        int64_t * pNextArray, int64_t * pUniqueArray, int64_t * pUniqueCountArray, HASH_MODE hashMode,
                        int64_t hintSize, bool * pBoolFilter);

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
uint64_t Unique32(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                  int32_t * pIndexArray, int32_t * pCountArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
uint64_t Unique64(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                  int64_t * pIndexArray, int64_t * pCountArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

//-----------------------------------------------------------------------------------------
void MultiKeyRollingStep2Delete(void * pHashLinearLast);

//-----------------------------------------------------------------------------------------
void * MultiKeyRollingStep2(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                            int64_t * pIndexArray, int64_t * pRunningCountArray, HASH_MODE hashMode, int64_t hintSize,
                            uint64_t * numUnique, void * pHashLinearLast);

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void * MultiKeyHash32(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                      int32_t * pIndexArray, int32_t * pRunningCountArray, int32_t * pPrevArray, int32_t * pNextArray,
                      int32_t * pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
void * MultiKeyHash64(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                      int64_t * pIndexArray, int64_t * pRunningCountArray, int64_t * pPrevArray, int64_t * pNextArray,
                      int64_t * pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

//---------------------------------------
// Hash uses
// -------------------------
// Storing the value 0 or null strings, special location
//
// HitCount or without HitCount, useful for findNthActive
// ismember
// unique
//
//---------------------------------------
// For Strings, we want to store an index back to original array?

//----------------------------------------
// T: int8_t,int16_t,int32_t,int64_t, string?
// U: int8_t, int16_t, int32_t or int64_t  (indexing)

template <typename T, typename U>
class CHashLinear
{
    struct HashLocationMK
    {
        U Location;
    };

    struct HashLocation
    {
        T value;
        U Location;
    };

    struct HashEntry
    {
        // Compare to get exact match
        T value;
        U Location;
        U HitCount;
    };

    struct UniqueEntry
    {
        // Compare to get exact match
        const char * Last;
        U UniqueKey;
    };

    struct SingleKeyEntry
    {
        // Compare to get exact match
        T value;
        U UniqueKey;
    };

    struct MultiKeyEntry
    {
        // Compare to get exact match
        U Last;
        U UniqueKey;
    };

    struct MultiKeyEntryRolling
    {
        // Compare to get exact match
        U Last;
        U RunningCount;
        char Key[16]; // upto 16 bytes
    };

    struct MultiKeyEntrySuper
    {
        // Compare to get exact match
        U Last;
        U UniqueKey;
    };

    struct FindNthEntry
    {
        // Compare to get exact match
        T value;
        U HitCount;
    };

public:
    void * pHashTableAny;

    // Max number of unique values possible
    int64_t NumEntries;

    // Actualy unique
    uint64_t NumUnique;

    // Number of collisions
    uint64_t NumCollisions;

    // Hashsize chosen based on input size
    uint64_t HashSize;

    // to determine if hash location has been visited
    uint64_t * pBitFields;

    // Remember the size we allocated for bitfields
    size_t BitAllocSize = 0;
    size_t HashTableAllocSize = 0;

    // how to hash, affects memory allocation
    HASH_MODE HashMode;

    const U BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

    bool Deallocate = true;

public:
    // Parallel hashing does not want memory deallocated so it will set deallocate
    // to false
    CHashLinear(HASH_MODE hashMode = HASH_MODE_PRIME, bool deallocate = true)
    {
        pHashTableAny = NULL;
        pBitFields = NULL;

        NumEntries = 0;
        NumUnique = 0;
        NumCollisions = 0;
        HashSize = 0;
        HashMode = hashMode;
        Deallocate = deallocate;

        // Can set bad index to 0?
        // BAD_INDEX = (U)(1LL << (sizeof(U)*8-1));
    }

    ~CHashLinear()
    {
        FreeMemory(false);
    }

    void FreeMemory(bool forceDeallocate);

    void AllocAndZeroBitFields(uint64_t hashSize);
    char * AllocHashTable(size_t allocSize);

    void * AllocMemory(int64_t numberOfEntries, int64_t sizeofStruct, int64_t sizeofExtraArray, bool isFloat);

    void MakeHash(int64_t numberOfEntries, T * pHashList);

    //-----------------------------------------------
    //----------------------------------------------
    //-----------------------------------------------
    void MakeHashLocationFloat(int64_t arraySize, T * pHashList, int64_t hintSize);

    //----------------------------------------------
    int64_t IsMemberFloatCategorical(int64_t arraySize, T * pHashList, U * pLocationOutput);

    //-----------------------------------------------
    uint64_t GroupByFloat(int64_t totalRows, int64_t totalItemSize, T * pInput, int coreType,
                          // Return values
                          U * pIndexArray, U *& pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

    //-----------------------------------------------
    uint64_t GroupByItemSize(int64_t totalRows, int64_t totalItemSize, T * pInput, int coreType,
                             // Return values
                             U * pIndexArray, U *& pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

    //-----------------------------------------------
    uint64_t GroupBy(int64_t totalRows, int64_t totalItemSize, const char * pInput, int coreType,
                     // Return values
                     U * pIndexArray, U *& pFirstArray, HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

    //-----------------------------------------------
    uint64_t GroupBySuper(int64_t totalRows, int64_t totalItemSize, const char * pInput,

                          int coreType,

                          // Return values
                          U * pIndexArray, U * pNextArray, U * pUniqueArray, U * pUniqueCountArray, HASH_MODE hashMode,
                          int64_t hintSize, bool * pBoolFilter);

    //-----------------------------------------------
    uint64_t Unique(int64_t totalRows, int64_t totalItemSize, const char * pInput,

                    // Return values
                    U * pIndexArray,
                    // Return count values
                    U * pCountArray,

                    HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

    //-----------------------------------------------
    void MultiKeyRolling(int64_t totalRows, int64_t totalItemSize, const char * pInput,

                         // Return values
                         U * pIndexArray, U * pRunningCountArray, HASH_MODE hashMode, int64_t hintSize);

    void MakeHashLocationMultiKey(int64_t totalRows, int64_t totalItemSize, const char * pInput1,

                                  U * pIndexArray, U * pRunningCountArray, U * pPrevArray, U * pNextArray, U * pFirstArray,
                                  HASH_MODE hashMode, int64_t hintSize, bool * pBoolFilter);

    //----------------------------------------------
    void MakeHashLocationString(int64_t arraySize, const char * pHashList, int64_t strWidth, int64_t hintSize, bool isUnicode);

    void InternalSetLocationString(U i, HashLocation * pLocation, const char * strValue, int64_t strWidth, uint64_t hash);

    void InternalSetLocationUnicode(U i, HashLocation * pLocation, const char * strValue, int64_t strWidth, uint64_t hash);

    void InternalGetLocationUnicode(int64_t i, HashLocation * pLocation, int8_t * pBooleanOutput, U * pLocationOutput,
                                    const char * strValue, int64_t strWidth, uint64_t hash);

    void InternalGetLocationString(int64_t i, HashLocation * pLocation, int8_t * pBooleanOutput, U * pLocationOutput,
                                   const char * strValue, int64_t strWidth, uint64_t hash);

    void InternalGetLocationUnicode2(int64_t i, HashLocation * pLocation, int8_t * pBooleanOutput, U * pLocationOutput,
                                     const char * strValue, int64_t strWidth, int64_t strWidth2, uint64_t hash);

    void InternalGetLocationString2(int64_t i, HashLocation * pLocation, int8_t * pBooleanOutput, U * pLocationOutput,
                                    const char * strValue, int64_t strWidth, int64_t strWidth2, uint64_t hash);

    void IsMemberString(int64_t arraySize, int64_t strWidth1, int64_t strWidth2, const char * pHashList, int8_t * pBooleanOutput,
                        U * pLocationOutput, bool isUnicode);

    void InternalGetLocationStringCategorical(int64_t i, HashLocation * pLocation, U * pLocationOutput, const char * strValue,
                                              int64_t strWidth, uint64_t hash, int64_t * missed);

    void InternalGetLocationString2Categorical(int64_t i, HashLocation * pLocation, U * pLocationOutput, const char * strValue,
                                               int64_t strWidth, int64_t strWidth2, uint64_t hash, int64_t * missed);

    //----------------------------------------------
    //----------------------------------------------
    //-----------------------------------------------
    void MakeHashLocationMK(int64_t arraySize, T * pInput, int64_t totalItemSize, int64_t hintSize);

    void MakeHashLocation(int64_t arraySize, T * pHashList, int64_t hintSize);

    void InternalSetLocation(U i, HashLocation * pLocation, T item, uint64_t hash);

    //----------------------------------------------
    void InternalGetLocation(U i, HashLocation * pLocation, int8_t * pBooleanOutput, U * pLocationOutput, T item, uint64_t hash);

    //----------------------------------------------
    void InternalGetLocationCategorical(U i, HashLocation * pLocation, U * pLocationOutput, T item, uint64_t hash,
                                        int64_t * missed);

    //----------------------------------------------
    int64_t IsMemberCategorical(int64_t arraySize, T * pHashList, U * pLocationOutput);

    //-----------------------------------------------

    //-----------------------------------------------
    template <typename V>
    void FindLastMatchMK(int64_t arraySize1, int64_t arraySize2, T * pKey1, T * pKey2, V * pVal1, V * pVal2, U * pLocationOutput,
                         int64_t totalItemSize, bool allowExact);

    template <typename V>
    void FindNextMatchMK(int64_t arraySize1, int64_t arraySize2, T * pKey1, T * pKey2, V * pVal1, V * pVal2, U * pLocationOutput,
                         int64_t totalItemSize, bool allowExact);
    //-----------------------------------------------
    //-----------------------------------------------

    void MakeHashFindNth(int64_t arraySize, T * pHashList,

                         // optional FOR boolEAN OR INDEX MASK
                         int64_t size2, U * pInput2,

                         int8_t * pBooleanOutput, int64_t n);

    void MakeHashFindNthFloat(int64_t arraySize, T * pHashList,

                              // optional FOR boolEAN OR INDEX MASK
                              int64_t size2, U * pInput2,

                              int8_t * pBooleanOutput, int64_t n);

    void InternalSetFindNth(U i, FindNthEntry * pLocation, int8_t * pBooleanOutput, int64_t n, T item, uint64_t hash);

    //-----------------------------------------------
    //-----------------------------------------------

    int64_t GetHashSize(int64_t numberOfEntries);

    //--------------------------------------------
    // Returns 1 if bit set otherwise 0
    inline int IsBitSet(int64_t position)
    {
        uint64_t index = position >> 6;
        if (pBitFields[index] & (1LL << (position & 63)))
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    //--------------------------------------------
    // set the bit to 1
    inline void SetBit(int64_t position)
    {
        int64_t index = position >> 6;
        pBitFields[index] |= (1LL << (position & 63));
    }

    //--------------------------------------------
    // set the bit to 0
    inline void ClearBit(int64_t position)
    {
        int64_t index = position >> 6;
        pBitFields[index] &= ~(1LL << (position & 63));
    }
};

template <typename T, typename U>
static int64_t IsMemberStringCategorical(void * pHashLinearVoid, int64_t arraySize, int64_t strWidth1, int64_t strWidth2,
                                         const char * pHashList, void * pLocationOutputU, bool isUnicode);

//----------------------------------------------
template <typename T, typename U>
static void IsMemberMK(void * pHashLinearVoid, int64_t arraySize, T * pInput, T * pInput2, int8_t * pBooleanOutput,
                       void * pLocationOutput, int64_t totalItemSize);

//----------------------------------------------
template <typename T, typename U>
static void IsMember(void * pHashLinearVoid, int64_t arraySize, void * pHashList, int8_t * pBooleanOutput,
                     void * pLocationOutputU);

//----------------------------------------------
template <typename T, typename U>
static void IsMemberFloat(void * pHashLinearVoid, int64_t arraySize, void * pHashList, int8_t * pBooleanOutput,
                          void * pLocationOutputU);
