#include <assert.h>

PyObject* IsMember32(PyObject *self, PyObject *args);
PyObject* IsMember64(PyObject *self, PyObject *args);
PyObject* IsMemberCategorical(PyObject *self, PyObject *args);

enum HASH_MODE {
   HASH_MODE_PRIME = 1,
   HASH_MODE_MASK = 2
};

struct UINT128 {
   UINT64   _val1;
   UINT64   _val2;
} ;


void* IsMemberHashMK32(
   INT64 size1,
   void* pInput1,
   INT64 size2,
   void* pInput2,
   INT32* pOutput,
   INT8* pBooleanOutput,
   INT64 sizeType,
   INT64 hintSize,
   HASH_MODE hashMode);


INT64 IsMemberHashCategorical(
   INT64 size1,
   void* pInput1,
   INT64 size2,
   void* pInput2,
   INT32* pOutput,
   INT sizeType,
   HASH_MODE hashMode,
   INT64 hintSize);

INT64 IsMemberHashCategorical64(
   INT64 size1,
   void* pInput1,
   INT64 size2,
   void* pInput2,
   INT64* pOutput,
   INT sizeType,
   HASH_MODE hashMode,
   INT64 hintSize);


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
   INT64 hintSize);

void* IsMemberHash64(
   INT64 size1,
   void* pInput1,
   INT64 size2,
   void* pInput2,
   INT64* pOutput,
   INT8* pBooleanOutput,
   INT sizeType,
   HASH_MODE hashMode,
   INT64 hintSize);

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
   BOOL isUnicode);


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
   BOOL isUnicode);

void IsMemberHashMKPre(
   PyArrayObject** indexArray,
   INT64 size1,
   void* pInput1,
   INT64 size2,
   void* pInput2,
   INT8* pBooleanOutput,
   INT64 totalItemSize,
   INT64 hintSize,
   HASH_MODE hashMode);



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
   BOOL isUnicode);


template<typename U, typename V>
void FindLastMatchCategorical(
   INT64    arraySize1,
   INT64    arraySize2,
   U*       pKey1,
   U*       pKey2,
   V*       pVal1,
   V*       pVal2,
   U*       pLocationOutput,
   INT64    totalUniqueSize);


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
   INT dtype);



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
   bool allowExact);

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
   bool allowExact);

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
template<typename INDEX_TYPE>
UINT64 GroupByInternal(
   void** pFirstArrayVoid,
   void** pHashTableAny,
   INT64* hashTableSize,

   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   INDEX_TYPE* pIndexArray,
   HASH_MODE hashMode,
   INT64   hintSize,
   bool* pBoolFilter);

//===================================================================================================
// Declare callback for either INT32 or INT64
typedef  UINT64(*GROUPBYCALL)(
   INT64  partitionLength, // may be 0
   INT64* pCutOffs,        // may be NULL
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   void* pIndexArray,
   PyArrayObject** pFirstArray,
   HASH_MODE hashMode,
   INT64   hintSize,
   bool* pBoolFilter);


UINT64 GroupBy32(
   INT64  partitionLength, // may be 0
   INT64* pCutOffs,        // may be NULL
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   void* pIndexArray,
   PyArrayObject** pFirstArray,
   HASH_MODE hashMode,
   INT64   hintSize,
   bool* pBoolFilter);

UINT64 GroupBy64(
   INT64  partitionLength, // may be 0
   INT64* pCutOffs,        // may be NULL
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,
   int coreType,
   void* pIndexArray,
   PyArrayObject** pFirstArray,
   HASH_MODE hashMode,
   INT64   hintSize,
   bool* pBoolFilter);


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
   INT64   hintSize,
   bool* pBoolFilter);

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
   INT64   hintSize,
   bool* pBoolFilter);


//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
UINT64 Unique32(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,

   INT32* pIndexArray,
   INT32* pCountArray,
   HASH_MODE hashMode,
   INT64   hintSize,
   bool* pBoolFilter);

//-----------------------------------------------------------------------------------------
//  Based on the sizeType it will call different hash functions
UINT64 Unique64(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,

   INT64* pIndexArray,
   INT64* pCountArray,
   HASH_MODE hashMode,
   INT64   hintSize,
   bool* pBoolFilter);

//-----------------------------------------------------------------------------------------
void MultiKeyRollingStep2Delete(
   void* pHashLinearLast);

//-----------------------------------------------------------------------------------------
void* MultiKeyRollingStep2(
   INT64 totalRows,
   INT64 totalItemSize,
   const char* pInput1,

   INT64* pIndexArray,
   INT64* pRunningCountArray,
   HASH_MODE hashMode,
   INT64 hintSize,
   UINT64* numUnique,
   void* pHashLinearLast);

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
   bool* pBoolFilter);

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
   bool* pBoolFilter);

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
// T: INT8,INT16,INT32,INT64, string?
// U: INT8, INT16, INT32 or INT64  (indexing)

template<typename T, typename U>
class CHashLinear {

   struct HashLocationMK
   {
      U        Location;
   };

   struct HashLocation
   {
      T        value;
      U        Location;
   };

   struct HashEntry
   {
      // Compare to get exact match
      T        value;
      U        Location;
      U        HitCount;
   };

   struct UniqueEntry
   {
      // Compare to get exact match
      const char*        Last;
      U        UniqueKey;
   };

   struct SingleKeyEntry
   {
      // Compare to get exact match
      T        value;
      U        UniqueKey;
   };


   struct MultiKeyEntry
   {
      // Compare to get exact match
      U        Last;
      U        UniqueKey;
   };

   struct MultiKeyEntryRolling
   {
      // Compare to get exact match
      U        Last;
      U        RunningCount;
      char     Key[16];  // upto 16 bytes
   };

   struct MultiKeyEntrySuper
   {
      // Compare to get exact match
      U        Last;
      U        UniqueKey;
   };

   struct FindNthEntry
   {
      // Compare to get exact match
      T        value;
      U        HitCount;
   };

public:
   void*       pHashTableAny;

   // Max number of unique values possible
   INT64       NumEntries;

   // Actualy unique
   UINT64      NumUnique;

   // Number of collisions
   UINT64      NumCollisions;

   // Hashsize chosen based on input size
   UINT64      HashSize;

   // to determine if hash location has been visited
   UINT64*     pBitFields;

   // Remember the size we allocated for bitfields
   size_t      BitAllocSize = 0;
   size_t      HashTableAllocSize = 0;

   // how to hash, affects memory allocation
   HASH_MODE   HashMode;

   const U     BAD_INDEX = (U)(1LL << (sizeof(U) * 8 - 1));

   BOOL        Deallocate = TRUE;

public:
   // Parallel hashing does not want memory deallocated so it will set deallocate to FALSE
   CHashLinear(HASH_MODE hashMode = HASH_MODE_PRIME, BOOL deallocate=TRUE) {
      pHashTableAny = NULL;
      pBitFields = NULL;

      NumEntries = 0;
      NumUnique = 0;
      NumCollisions = 0;
      HashSize = 0;
      HashMode = hashMode;
      Deallocate = deallocate;

      // Can set bad index to 0?
      //BAD_INDEX = (U)(1LL << (sizeof(U)*8-1));
   }

   ~CHashLinear() {

      FreeMemory(FALSE);
   }

   void FreeMemory(BOOL forceDeallocate);

   void AllocAndZeroBitFields(UINT64 hashSize);
   char* AllocHashTable(size_t allocSize);

   void* AllocMemory(
      INT64 numberOfEntries,
      INT64 sizeofStruct,
      INT64 sizeofExtraArray,
      BOOL  isFloat);

   void MakeHash(
      INT64 numberOfEntries,
      T* pHashList);


   //-----------------------------------------------
   //----------------------------------------------
   //-----------------------------------------------
   void MakeHashLocationFloat(
      INT64 arraySize,
      T*    pHashList,
      INT64 hintSize);

   //----------------------------------------------
   INT64 IsMemberFloatCategorical(
      INT64 arraySize,
      T*    pHashList,
      U*    pLocationOutput);


   //-----------------------------------------------
   UINT64 GroupByFloat(
      INT64 totalRows,
      INT64 totalItemSize,
      T* pInput,
      int coreType,
      // Return values
      U* pIndexArray,
      U* &pFirstArray,
      HASH_MODE hashMode,
      INT64 hintSize,
      bool* pBoolFilter);


   //-----------------------------------------------
   UINT64 GroupByItemSize(
      INT64 totalRows,
      INT64 totalItemSize,
      T* pInput,
      int coreType,
      // Return values
      U* pIndexArray,
      U* &pFirstArray,
      HASH_MODE hashMode,
      INT64 hintSize,
      bool* pBoolFilter);

   //-----------------------------------------------
   UINT64 GroupBy(
      INT64 totalRows,
      INT64 totalItemSize,
      const char* pInput,
      int coreType,
      // Return values
      U* pIndexArray,
      U* &pFirstArray,
      HASH_MODE hashMode,
      INT64 hintSize,
      bool* pBoolFilter);


   //-----------------------------------------------
   UINT64 GroupBySuper(
      INT64 totalRows,
      INT64 totalItemSize,
      const char* pInput,

      int coreType,

      // Return values
      U* pIndexArray,
      U* pNextArray,
      U* pUniqueArray,
      U* pUniqueCountArray,
      HASH_MODE hashMode,
      INT64 hintSize,
      bool* pBoolFilter);


   //-----------------------------------------------
   UINT64 Unique(
      INT64 totalRows,
      INT64 totalItemSize,
      const char* pInput,

      // Return values
      U* pIndexArray,
      // Return count values
      U* pCountArray,

      HASH_MODE hashMode,
      INT64 hintSize,
      bool* pBoolFilter);

   //-----------------------------------------------
   void MultiKeyRolling(
      INT64 totalRows,
      INT64 totalItemSize,
      const char* pInput,

      // Return values
      U* pIndexArray,
      U* pRunningCountArray,
      HASH_MODE hashMode,
      INT64 hintSize);
      
  void MakeHashLocationMultiKey(
      INT64 totalRows,
      INT64 totalItemSize,
      const char* pInput1,

      U* pIndexArray,
      U* pRunningCountArray,
      U* pPrevArray,
      U* pNextArray,
      U* pFirstArray,
      HASH_MODE hashMode,
      INT64 hintSize,
      bool* pBoolFilter);

   //----------------------------------------------
   void MakeHashLocationString(
      INT64 arraySize,
      const char* pHashList,
      INT64 strWidth,
      INT64 hintSize,
      BOOL  isUnicode);

   void InternalSetLocationString(
      U  i,
      HashLocation* pLocation,
      const char* strValue,
      INT64 strWidth,
      UINT64 hash);

   void InternalSetLocationUnicode(
      U  i,
      HashLocation* pLocation,
      const char* strValue,
      INT64 strWidth,
      UINT64 hash);

   void InternalGetLocationUnicode(
      INT64  i,
      HashLocation* pLocation,
      INT8* pBooleanOutput,
      U*    pLocationOutput,
      const char* strValue,
      INT64 strWidth,
      UINT64 hash);

   void InternalGetLocationString(
      INT64  i,
      HashLocation* pLocation,
      INT8* pBooleanOutput,
      U*    pLocationOutput,
      const char* strValue,
      INT64 strWidth,
      UINT64 hash);

   void InternalGetLocationUnicode2(
      INT64  i,
      HashLocation* pLocation,
      INT8* pBooleanOutput,
      U*    pLocationOutput,
      const char* strValue,
      INT64 strWidth,
      INT64 strWidth2,
      UINT64 hash);

   void InternalGetLocationString2(
      INT64  i,
      HashLocation* pLocation,
      INT8* pBooleanOutput,
      U*    pLocationOutput,
      const char* strValue,
      INT64 strWidth,
      INT64 strWidth2,
      UINT64 hash);

   void IsMemberString(
      INT64 arraySize,
      INT64 strWidth1,
      INT64 strWidth2,
      const char* pHashList,
      INT8* pBooleanOutput,
      U*    pLocationOutput,
      BOOL  isUnicode);


   void InternalGetLocationStringCategorical(
      INT64  i,
      HashLocation* pLocation,
      U*    pLocationOutput,
      const char* strValue,
      INT64 strWidth,
      UINT64 hash,
      INT64* missed);

   void InternalGetLocationString2Categorical(
      INT64  i,
      HashLocation* pLocation,
      U*    pLocationOutput,
      const char* strValue,
      INT64 strWidth,
      INT64 strWidth2,
      UINT64 hash,
      INT64 *missed);

   //----------------------------------------------
   //----------------------------------------------
   //-----------------------------------------------
   void MakeHashLocationMK(
      INT64 arraySize,
      T*    pInput,
      INT64 totalItemSize,
      INT64 hintSize);


   void MakeHashLocation(
      INT64 arraySize,
      T*    pHashList,
      INT64 hintSize);

   void InternalSetLocation(
      U  i,
      HashLocation* pLocation,
      T  item,
      UINT64 hash);

   //----------------------------------------------
   void InternalGetLocation(
      U  i,
      HashLocation* pLocation,
      INT8* pBooleanOutput,
      U* pLocationOutput,
      T  item,
      UINT64 hash);


   //----------------------------------------------
   void InternalGetLocationCategorical(
      U  i,
      HashLocation* pLocation,
      U* pLocationOutput,
      T  item,
      UINT64 hash,
      INT64* missed);


   //----------------------------------------------
   INT64 IsMemberCategorical(
      INT64 arraySize,
      T*    pHashList,
      U*    pLocationOutput);

   //-----------------------------------------------

   //-----------------------------------------------
   template<typename V> void FindLastMatchMK(
      INT64 arraySize1,
      INT64 arraySize2,
      T*    pKey1,
      T*    pKey2,
      V*    pVal1,
      V*    pVal2,
      U*    pLocationOutput,
      INT64 totalItemSize,
      bool allowExact);

   template<typename V> void FindNextMatchMK(
      INT64 arraySize1,
      INT64 arraySize2,
      T*    pKey1,
      T*    pKey2,
      V*    pVal1,
      V*    pVal2,
      U*    pLocationOutput,
      INT64 totalItemSize,
      bool allowExact);
   //-----------------------------------------------
   //-----------------------------------------------


   void MakeHashFindNth(
      INT64 arraySize,
      T*    pHashList,

      // optional FOR BOOLEAN OR INDEX MASK
      INT64 size2,
      U*    pInput2,

      INT8* pBooleanOutput,
      INT64 n);


   void MakeHashFindNthFloat(
      INT64 arraySize,
      T*    pHashList,

      // optional FOR BOOLEAN OR INDEX MASK
      INT64 size2,
      U*    pInput2,

      INT8* pBooleanOutput,
      INT64 n);


   void InternalSetFindNth(
      U  i,
      FindNthEntry* pLocation,
      INT8* pBooleanOutput,
      INT64 n,
      T  item,
      UINT64 hash);

   //-----------------------------------------------
   //-----------------------------------------------

   INT64 GetHashSize(INT64 numberOfEntries);

   //--------------------------------------------
   // Returns 1 if bit set otherwise 0
   inline int IsBitSet(INT64 position) {

      UINT64 index = position >> 6;
      if (pBitFields[index] & (1LL << (position & 63))) {
         return 1;
      }
      else {
         return 0;
      }
   }


   //--------------------------------------------
   // set the bit to 1
   inline void SetBit(INT64 position) {

      INT64 index = position >> 6;
      pBitFields[index] |= (1LL << (position & 63));
   }

   //--------------------------------------------
   // set the bit to 0
   inline void ClearBit(INT64 position) {

      INT64 index = position >> 6;
      pBitFields[index] &= ~(1LL << (position & 63));
   }
};



template<typename T, typename U>
static INT64 IsMemberStringCategorical(
   void* pHashLinearVoid,
   INT64 arraySize,
   INT64 strWidth1,
   INT64 strWidth2,
   const char* pHashList,
   void*    pLocationOutputU,
   BOOL isUnicode);


//----------------------------------------------
template<typename T, typename U>
static void IsMemberMK(
   void* pHashLinearVoid,
   INT64 arraySize,
   T*    pInput,
   T*    pInput2,
   INT8* pBooleanOutput,
   void* pLocationOutput,
   INT64 totalItemSize);

//----------------------------------------------
template<typename T, typename U>
static void IsMember(
   void* pHashLinearVoid,
   INT64 arraySize,
   void* pHashList,
   INT8* pBooleanOutput,
   void* pLocationOutputU);

//----------------------------------------------
template<typename T, typename U>
static void IsMemberFloat(
   void* pHashLinearVoid,
   INT64 arraySize,
   void* pHashList,
   INT8* pBooleanOutput,
   void* pLocationOutputU);


