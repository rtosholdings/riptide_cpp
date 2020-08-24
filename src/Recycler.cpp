#include "RipTide.h"
#include "Recycler.h"
#include "MathWorker.h"
#include "immintrin.h"

// Array to store what we can recycle
stRecycleList  g_stRecycleList[RECYCLE_ENTRIES][RECYCLE_MAXIMUM_TYPE];

INT64 g_DidNotFindRecycled = 0;
INT64 g_FoundRecycled = 0;
INT64 g_TotalHeads = 0;
INT64 g_TotalTails = 0;

// Default garbage collect timespan
// Setting to 0 will stop collection
// Setting to 1 billion will never garbage collect
INT64 g_GarbageCollectTimeSpan = 150;

extern PyTypeObject*  g_FastArrayType;
static const INT64 NANO_BILLION = 1000000000LL;

//#define LOGRECYCLE printf
#define LOGRECYCLE(...)

static UINT64 gLastGarbageCollectTSC = 0;
//-----------------------------------------------
// RECYCLING of BitFields
size_t g_CurrentAllocBitSize = 0;
void* g_pBitFields = NULL;

//-----------------------------------------------
// RECYCLING of HashTable
size_t g_CurrentAllocHashTable = 0;
void*  g_pHashTableAny = NULL;

static INT64 g_TotalAllocs = 0;
static INT64 g_TotalFree = 0;
static INT64 g_TotalMemoryAllocated = 0;
static INT64 g_TotalMemoryFreed = 0;
//-----------------------------------------
static INT64 gRecyleMode = 0;
static int gRecursion = 0;
static int gGarbageCollecting = 0;

#define MAGIC_PAGE_GUARD  0xDEADBEEFDEADBEEF
//-----------------------------------------------
void* FmAlloc(size_t _Size) {
   // make thread safe
   UINT64* pageGuard= (UINT64*)malloc(_Size + 16);
   if (pageGuard) {
      InterlockedIncrement64(&g_TotalAllocs);
      InterlockedAdd64(&g_TotalMemoryAllocated, _Size);
      pageGuard[0] = _Size;
      pageGuard[1] = MAGIC_PAGE_GUARD;

      // Skip past guard
      return &pageGuard[2];
   }
   return NULL;
}

void FmFree(void* _Block) {
   // The C standard requires that free() be a no-op when called with nullptr.
   // FmAlloc can return a nullptr, and since we want this function to behave
   // like free() we also need to handle the nullptr case here.
   if (!_Block) { return; }

   //LOGRECYCLE("Freeing %p\n", _Block);
   InterlockedIncrement64(&g_TotalFree);
   UINT64* pageGuard = (UINT64*)_Block;
   pageGuard--;
   pageGuard--;
   if (pageGuard[1] != MAGIC_PAGE_GUARD) {
      printf("!! User freed bad memory, no page guard %p\n", pageGuard);
   }
   else {
      InterlockedAdd64(&g_TotalMemoryFreed, pageGuard[0]);
      // mark so cannot free again
      pageGuard[1] = 0;
   }

   free(pageGuard);
}

//-----------------------------------------------
void FreeWorkSpaceAllocLarge(void* gpHashTable) {
   LOGRECYCLE("FreeWorkSpaceAllocLarge %p %lld\n", gpHashTable, g_CurrentAllocHashTable);
   // Free global
   if (g_cMathWorker->NoThreading || g_cMathWorker->NoCaching) {
      if (gpHashTable)
         WORKSPACE_FREE(gpHashTable);
   }
   else {
      if (g_pHashTableAny && g_pHashTableAny == gpHashTable) {
         WORKSPACE_FREE(g_pHashTableAny);
         g_pHashTableAny = NULL;
         g_CurrentAllocHashTable = 0;
      }
   }
}


void FreeWorkSpaceAllocSmall(void * gpBits) {
   LOGRECYCLE("FreeWorkSpaceAllocSmall %p %lld\n", gpBits, g_CurrentAllocBitSize);
   if (g_cMathWorker->NoThreading || g_cMathWorker->NoCaching) {
      if (gpBits)
         WORKSPACE_FREE(gpBits);
   }
   else {
      if (g_pBitFields && g_pBitFields == gpBits) {
         WORKSPACE_FREE(g_pBitFields);
         g_pBitFields = NULL;
         g_CurrentAllocBitSize = 0;
      }
   }
}

//=============================================
// Called when hashing.
// Called from AllocHashTable
// If threading is off, we must allocate memory
void* WorkSpaceAllocLarge(size_t HashTableAllocSize) {
   void* pHashTableAny = NULL;
   if (g_cMathWorker->NoThreading || g_cMathWorker->NoCaching) {
      LOGRECYCLE("hashtable threadsafe NOT using recycled %llu\n", HashTableAllocSize);
      pHashTableAny = WORKSPACE_ALLOC(HashTableAllocSize);
   }
   else {
      if (g_pHashTableAny != NULL && HashTableAllocSize <= g_CurrentAllocHashTable) {
         LOGRECYCLE("hashtable using recycled %llu\n", HashTableAllocSize);
         pHashTableAny = g_pHashTableAny;

      }
      else {
         LOGRECYCLE("hashtable NOT using recycled %llu\n", HashTableAllocSize);
         pHashTableAny = WORKSPACE_ALLOC(HashTableAllocSize);

         if (pHashTableAny == NULL) {
            LogError("Out of memory with hash\n");
            return NULL;
         }
         // Free global
         FreeWorkSpaceAllocLarge(g_pHashTableAny);
      }
   }
   LOGRECYCLE("WorkSpaceAllocLarge returning %p %lld\n", pHashTableAny, HashTableAllocSize);
   return pHashTableAny;
}

//=============================================
// Called when hashing.
// Called from AllocAndZeroBitFields
// If threading is off, we must allocate memory
void* WorkSpaceAllocSmall( size_t BitAllocSize) {
   void* pBitFields = NULL;

   if (g_cMathWorker->NoThreading || g_cMathWorker->NoCaching) {
      LOGRECYCLE("hashtable threadsafe NOT using recycled %llu\n", BitAllocSize);
      pBitFields = WORKSPACE_ALLOC(BitAllocSize);
   }
   else {

      if (g_pBitFields != NULL && BitAllocSize <= g_CurrentAllocBitSize) {
         LOGRECYCLE("bitfields using recycled %llu\n", BitAllocSize);
         pBitFields = g_pBitFields;
      }
      else {
         LOGRECYCLE("bitfields NOT using recycled %llu\n", BitAllocSize);
         pBitFields = WORKSPACE_ALLOC(BitAllocSize);

         if (pBitFields == 0) {
            LogError("Out of memory with bitfields\n");
            return NULL;
         }

         // Free global
         FreeWorkSpaceAllocSmall(g_pBitFields);
      }
   }
   LOGRECYCLE("WorkSpaceAllocSmall returning %p\n", pBitFields);
   return pBitFields;
}

void WorkSpaceFreeAllocLarge(void* &pHashTableAny, size_t HashTableAllocSize) {
   LOGRECYCLE("WorkSpaceFreeAllocLarge %p %p %lld %lld\n", pHashTableAny, g_pHashTableAny, g_CurrentAllocHashTable, HashTableAllocSize);

   if (g_cMathWorker->NoThreading || g_cMathWorker->NoCaching) {
      // not using the global memory
      if (pHashTableAny != NULL) {
         WORKSPACE_FREE(pHashTableAny);
         pHashTableAny = NULL;
      }
   } else {
      if (pHashTableAny != NULL) {
         if (g_pHashTableAny == NULL) {
            // recycle this memory
            LOGRECYCLE("WorkSpaceFreeAllocLarge is recycling\n");
            g_CurrentAllocHashTable = HashTableAllocSize;
            g_pHashTableAny = pHashTableAny;
            pHashTableAny = NULL;
         }
         else {
            // Check to see if already using this
            if (g_pHashTableAny != pHashTableAny) {
               if (HashTableAllocSize > g_CurrentAllocHashTable) {
                  LOGRECYCLE("WorkSpaceFreeAllocLarge is replacing\n");
                  WORKSPACE_FREE(g_pHashTableAny);
                  // replace recycler with this memory
                  g_CurrentAllocHashTable = HashTableAllocSize;
                  g_pHashTableAny = pHashTableAny;
                  pHashTableAny = NULL;
               }
               else {
                  // not using the global memory
                  LOGRECYCLE("WorkSpaceFreeAllocLarge not using global mem\n");
                  WORKSPACE_FREE(pHashTableAny);
                  pHashTableAny = NULL;
               }
            }
            else {
               // using the memory -- do nothing
               LOGRECYCLE("WorkSpaceFreeAllocLarge is doing nothing\n");
               pHashTableAny = NULL;
            }
         }
      }
   }
}

void WorkSpaceFreeAllocSmall(void* &pBitFields, size_t BitAllocSize) {
   LOGRECYCLE("WorkSpaceFreeAllocSmall %p %p\n", pBitFields, g_pBitFields);

   if (g_cMathWorker->NoThreading || g_cMathWorker->NoCaching) {

      // not using the global memory
      if (pBitFields != NULL) {
         WORKSPACE_FREE(pBitFields);
         pBitFields = NULL;
      }
   }
   else {
      if (pBitFields != NULL) {
         if (g_pBitFields == NULL) {
            // recycle this memory
            g_CurrentAllocBitSize = BitAllocSize;
            g_pBitFields = pBitFields;
            pBitFields = NULL;
         }
         else {
            // Check to see if already using this
            if (g_pBitFields != pBitFields) {
               if (BitAllocSize > g_CurrentAllocBitSize) {
                  WORKSPACE_FREE(g_pBitFields);
                  // replace recycler this memory
                  g_CurrentAllocBitSize = BitAllocSize;
                  g_pBitFields = pBitFields;
                  pBitFields = NULL;
               }
               else {
                  // not using the global memory
                  WORKSPACE_FREE(pBitFields);
                  pBitFields = NULL;
               }
            }
            else {
               // using the memory -- do nothing
               pBitFields = NULL;
            }
         }
      }
   }
}


//-----------------------------------
// NOTE: This routine does not release the ref count
// It can be called when an array is reused instead of deleted
// Call RefCountNumpyArray instead 
static inline void RemoveFromList(stRecycleList* pItems, INT32 slot) {
   // remove it from list
   // clear the total size to prevent reuse
   //printf("setting slot %d to 0 from %lld\n", slot, pItems->Item[slot].totalSize);
   pItems->Item[slot].totalSize = 0;
   pItems->Item[slot].recycledArray = NULL;

   // Only location tail is incremented
   pItems->Tail++;
   g_TotalTails++;

   //printf("TAIL %lld\n", g_TotalTails);
}


//-----------------------------------
// Will increment or decerement array
// On decrement it will decref array, set the slot to null and inc Tail
// On increment it will incref array, time stamp, and inc Head
// On Entry: the recycledArray must be valid
static void RefCountNumpyArray(stRecycleList* pItems, INT32 lzcount, INT32 type, INT32 slot, BOOL bIncrement) {

   if (pItems->Item[slot].recycledArray == NULL) {
      LOGRECYCLE("!!! Critical error -- recycled array is NULL");
   }

   if (bIncrement) {

      // inc ref count to indicate we want ownership of base array
      //Py_IncRef((PyObject*)(pItems->Item[slot].recycledArray));
      Py_INCREF((PyObject*)(pItems->Item[slot].recycledArray));

      // timestamp
      pItems->Item[slot].tsc = __rdtsc();
      LOGRECYCLE("Adding item to slot %d,%d,%d -- refcnt now is %llu\n", lzcount, type, slot, pItems->Item[slot].recycledArray->ob_base.ob_refcnt);

      // Only location head is incremented
      pItems->Head++;
      g_TotalHeads++;
      //printf("HEAD %lld\n", g_TotalHeads);
   }
   else {
      // This will increment Tail
      PyArrayObject* toDelete = pItems->Item[slot].recycledArray;
      RemoveFromList(pItems, slot);

      // If this is the last decref, python may free memory
      LOGRECYCLE("Removing item in slot %d,%d,%d -- refcnt before is %llu\n", lzcount, type, slot, toDelete->ob_base.ob_refcnt);

      // NOTE: If this is the last decref, it might "free" the memory which might take time
      Py_DecRef((PyObject*)toDelete);
   }
}

//------------------------------------------------------------
// Just does a printf to dump the stats for an item
static void DumpItemStats(stRecycleList* pItems, int k) {

   INT64 delta = (INT64)(__rdtsc() - pItems->Item[k].tsc);
   printf("    delta: %lld   refcount: %d  now: %lld  addr: %p  type: %d \t %p \t %lld\n",
      delta / NANO_BILLION,
      pItems->Item[k].initRefCount,
      pItems->Item[k].recycledArray == NULL ? 0LL : (INT64)(pItems->Item[k].recycledArray->ob_base.ob_refcnt),
      // too dangerous -- might be gone pItems->Item[k].recycledOrigArray == NULL ? 0 : pItems->Item[k].recycledOrigArray->ob_base.ob_refcnt,
      pItems->Item[k].recycledArray == NULL ? NULL: PyArray_BYTES(pItems->Item[k].recycledArray),
      pItems->Item[k].type,
      pItems->Item[k].recycledArray,
      pItems->Item[k].totalSize);
}


//-----------------------------------------------------------------------------------------
// Arg1: input cycle count to delete -- goes by GHZ or cycles
//       input the number 2 or 3 is roughly equivalent to 1 second
//       if your cpu is at 2.5 GHZ then input of 150 is equal to 60 seconds or 1 minute
// Arg2: whether or not to print GC stats and also whether or not to foce a GC
//
// NOTE: on normal operation will only run GC when 1 billion cycles have elapsed
// Returns TotalDeleted
INT64 GarbageCollect(INT64 timespan, bool verbose) {
   UINT64 currentTSC = __rdtsc();
   INT64 totalDeleted = 0;

   if (verbose) {
      // verbose mode will always run GC because assumed user initiated
      printf("--- Garbage Collector start --- timespan: %lld\n", timespan);
   }
   else {
      // non verbose mode will only run GC if enough cycles have expired
      if ((currentTSC - gLastGarbageCollectTSC) < 60*NANO_BILLION) {
         //printf("GC did not happen\n");
         return 0;
      }
   }

   LOGRECYCLE("Checking GC\n");

   // remember last time we garbage collected
   gLastGarbageCollectTSC = currentTSC;

   // multiply by a billion (1GHZ) because timespan is a user value
   timespan = timespan * NANO_BILLION;
   
   gGarbageCollecting++;
   gRecursion ++ ;

   // TODO: loop is larger than needed
   for (int i = 0; i < RECYCLE_ENTRIES; i++) {
      for (int j = 0; j < RECYCLE_MAXIMUM_TYPE; j++) {
         stRecycleList* pItems = &g_stRecycleList[i][j];
         INT32 deltaHead = pItems->Head - pItems->Tail;

         // sanity check head tail
         if (deltaHead < 0 || deltaHead > RECYCLE_MAXIMUM_SEARCH) {
            LogError("!!! critical error with recycler items %d,%d with deltahead %d\n", i, j, deltaHead);
         }
         if (pItems->Head != pItems->Tail) {
            if (verbose) printf("%d:%d  head: %d  tail: %d\n", i, j, pItems->Head, pItems->Tail);
            for (int k = 0; k < RECYCLE_MAXIMUM_SEARCH; k++) {
               INT64 totalSize = pItems->Item[k].totalSize;

               // size must be > 0 for the entry to be valid
               if (totalSize > 0) {
                  INT64 delta = (INT64)(currentTSC - pItems->Item[k].tsc);
                  if (verbose)
                     DumpItemStats(pItems, k);

                  //printf("Comparing %lld to %lld\n", delta, timespan);

                  if (delta > timespan) {
                     if (verbose) printf("    ***GC deleting with size %lld\n", totalSize);
                     
                     // Release ref count and remove entry
                     RefCountNumpyArray(pItems, i,j, k, FALSE);
                     totalDeleted++;
                  }
               }
            }
         }
      }
   }

   gRecursion--;
   gGarbageCollecting--;

   if (verbose) printf("--- Garbage Collector end   --- deleted: %lld\n", totalDeleted);

   return totalDeleted;
}


//-----------------------------------------------------------------------------------------
// Debug routine to list contents of array
// Returns totalSize of items not in use (ararys that are ready to be reused)
PyObject *
RecycleDump(PyObject *self, PyObject *args)
{
   printf("Recycled stats.   Hits: %lld    Misses: %lld    Heads: %lld    Tails: %lld\n",
         g_FoundRecycled, g_DidNotFindRecycled, g_TotalHeads, g_TotalTails);

   UINT64 currentTSC = __rdtsc();

   INT64 totalSize = 0;
   for (int i = 0; i < RECYCLE_ENTRIES; i++) {
      for (int j = 0; j < RECYCLE_MAXIMUM_TYPE; j++) {
         stRecycleList* pItems = &g_stRecycleList[i][j];
         if (pItems->Head != pItems->Tail) {
            printf("%d:%d  head: %d  tail: %d\n", i, j, pItems->Head, pItems->Tail);
            for (int k = 0; k < RECYCLE_MAXIMUM_SEARCH; k++) {
               totalSize += pItems->Item[k].totalSize;
               DumpItemStats(pItems,k);
            }

         }
      }
   }

   printf("Total size recycled memory: %lld MB\n", totalSize/(1024*1024));
   return PyLong_FromLongLong(totalSize);
}


// cache the name
static PyObject* g_namestring=  PyUnicode_FromString("_name");

//-----------------------------------
// Called when we look for recycled array
// Scans the tables to see if a recycled array is available
PyArrayObject* RecycleFindArray(INT32 ndim, INT32 type, INT64 totalSize) {
   if (totalSize >= RECYCLE_MIN_SIZE && type < RECYCLE_MAXIMUM_TYPE && ndim == 1) {
      // Based on size and type, lookup
      INT64 log2 = lzcnt_64(totalSize);

      LOGRECYCLE("totalSize %llu  log2 %llu\n", totalSize, log2);
      
      stRecycleList* pItems = &g_stRecycleList[log2][type];
      for (int i = 0; i < RECYCLE_MAXIMUM_SEARCH; i++) {

         // Search for empty slot
         if (pItems->Item[i].totalSize == totalSize) {

            PyArrayObject* inArr = pItems->Item[i].recycledArray;
            INT64 refCount = inArr->ob_base.ob_refcnt;

            // If the refcnt is one, it must be just us holding on to it
            if (refCount == 1) {
               // We will reuse this item so we do not decrement it here
               RemoveFromList(pItems, i);
               g_FoundRecycled++;

               LOGRECYCLE("Found recycled item %llu  %d %p %p\n", totalSize, type, inArr, PyArray_BYTES(inArr));
               // have to clear the name
               Py_INCREF(Py_None);
               PyObject_SetAttr((PyObject*)inArr, g_namestring, Py_None);
               return inArr;
            }
            else {
               LOGRECYCLE("Rejected recycled item %llu  %d %p %p\n", refCount, type, inArr, PyArray_BYTES(inArr));

            }
         }
      }

      g_DidNotFindRecycled++;

      LOGRECYCLE("Did not find recycled item %llu  %d\n", totalSize, type);
   }
   return NULL;
}


//-----------------------------------
// Called to turn recycling on or off
PyObject *
SetRecycleMode(PyObject *self, PyObject *args)
{
   INT64 mode = 0;

   if (!PyArg_ParseTuple(args, "L", &mode)) return NULL;

   gRecyleMode = mode;

   Py_INCREF(Py_True);
   return Py_True;
}

//-----------------------------------
// Called when an array is deleted
// Returns: TRUE if recycled
// Returns: FALSE if rejected recycling
//
// pFirst == original object without base pointer traversed
// inArr = base array object
static BOOL DeleteNumpyArray(PyArrayObject *inArr) {
   // Make sure this is worth caching
   // For very small sizes, we do not bother to cache
   // For odd types, we also do not bother to cache
   // 1024 = 2^10
   if (gRecyleMode) {
      return FALSE;
   }

   INT64 refCount = inArr->ob_base.ob_refcnt;
   if (refCount != 0) {
      LOGRECYCLE("Rejected recycling because base refCount is %lld\n", refCount);
      return FALSE;
   }

   int flags = PyArray_FLAGS(inArr);
   LOGRECYCLE("del numpy array flags %d, refCount %lld\n", flags, refCount);

   // Check writeable flag on BASE array
   // Cannot recycle readonly (not writeable) or (not owned)
   if ((flags & (NPY_ARRAY_WRITEABLE | NPY_ARRAY_OWNDATA | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_C_CONTIGUOUS)) != (NPY_ARRAY_WRITEABLE | NPY_ARRAY_OWNDATA | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_C_CONTIGUOUS)) {
      LOGRECYCLE("Rejected recycling because object is not writeable or owned or contiguous.\n");
      return FALSE;
   }

   if (gRecursion) {
      //printf("recursion on %lld  %lld\n", refCount, totalSize);
      return FALSE;
   }
   else {
      //printf("non recursion on %lld  %lld\n", refCount, totalSize);

   }

   gRecursion++;

   INT32 ndim = PyArray_NDIM(inArr);
   npy_intp* dims = PyArray_DIMS(inArr);

   INT64 totalSize = CalcArrayLength(ndim, dims);
   INT32  type = PyArray_TYPE(inArr);
   BOOL retVal = FALSE;

   npy_intp* strides= PyArray_STRIDES(inArr);
   npy_intp itemSize = PyArray_ITEMSIZE(inArr);

   // TJD: New code only recycle FastArrays
   //if (g_FastArrayType != NULL && inArr->ob_base.ob_type == g_FastArrayType && g_GarbageCollectTimeSpan > 0) {
   if (g_GarbageCollectTimeSpan > 0) {

      // Multiple checks have to clear before we consider recycling this array
      if (totalSize >= RECYCLE_MIN_SIZE && 
         type < RECYCLE_MAXIMUM_TYPE && 
         // refCount < 3 && 
         ndim == 1 &&
         strides != NULL &&
         itemSize == strides[0]) {

         // Based on size and type, lookup
         INT32 log2 = (INT32)lzcnt_64(totalSize);

         stRecycleList* pItems = &g_stRecycleList[log2][type];

         // Choose default empty slot if everything is full
         void* memAddress = PyArray_BYTES(inArr);

         BOOL abort = FALSE;

         for (int i = 0; i < RECYCLE_MAXIMUM_SEARCH; i++) {
            // Search for same memory address
            if (pItems->Item[i].totalSize != 0 && pItems->Item[i].memoryAddress == memAddress) {
               // Does this happen when wrapped in a FA? does double delete
               //LOGRECYCLE("Rejected recycling due to mem address clash slot:%d  refcnt: %llu   size: %llu   type: %d  %p\n", i, refCount, totalSize, type, memAddress);
               abort = TRUE;
               break;
            }
         }

         if (!abort) {
            BOOL bFoundEmpty = FALSE;
            INT32 slot = -1;

            for (int i = 0; i < RECYCLE_MAXIMUM_SEARCH; i++) {

               // Search for empty slot
               if (pItems->Item[i].totalSize == 0) {

                  // Found empty slot
                  slot = i;
                  break;
               }
               //else {
               //   // TODO: Check current ref count--- if not 1, remove?
               //   INT64 curRefCount = pItems->Item[i].recycledArray->ob_base.ob_refcnt;
               //   if (curRefCount != 1) {
               //      printf("Weird ref count %llu\n", curRefCount);
               //      slot = i;
               //      break;
               //   }
               //}
            }

            if (slot == -1) {
               slot = pItems->Head & RECYCLE_MASK;
               for (int i = 0; i < RECYCLE_MAXIMUM_SEARCH; i++) {

                  // TODO: Check current ref count--- if not 1, remove?
                  INT64 curRefCount = pItems->Item[i].recycledArray->ob_base.ob_refcnt;
                  if (curRefCount != 1) {
                     LOGRECYCLE("Weird ref count %llu\n", curRefCount);
                     slot = i;
                     break;
                  }

               }

               if (pItems->Item[slot].totalSize != 0) {
                  LOGRECYCLE("!! removing existing entry -- slot %d  size: %lld\n", slot, pItems->Item[slot].totalSize);
                  // Let go of old item to make room for new item
                  // NOTE: this can recurse back
                  RefCountNumpyArray(pItems, log2, type, slot, FALSE);
               }
            }

            INT32 deltaHead = pItems->Head - pItems->Tail;

            // sanity check head tail
            if (deltaHead < 0 || deltaHead > RECYCLE_MAXIMUM_SEARCH) {
               LogError("!!! inner critical error with recycler items %d,%d,%d with deltahead %d  totalsize%lld\n",
                  log2,
                  type,
                  slot,
                  deltaHead,
                  pItems->Item[slot].totalSize);
            }

            LOGRECYCLE("-- keeping array with refcnt %llu   head: %d  tail: %d   size: %llu  leadingz:%d   type:%d  %p %p\n", refCount, pItems->Head, pItems->Tail, totalSize, log2, type, inArr, PyArray_BYTES(inArr));
            stRecycleItem* pItemSlot = &pItems->Item[slot];

            pItemSlot->type = (INT16)type;
            pItemSlot->initRefCount = (INT16)refCount;
            pItemSlot->ndim = ndim;

            pItemSlot->totalSize = totalSize;
            pItemSlot->recycledArray = inArr;
            pItemSlot->memoryAddress = memAddress;

            LOGRECYCLE("inner setting slot %d,%d,%d to %lld from %lld\n", log2, type, slot, totalSize, pItemSlot->totalSize);

            // keep it
            RefCountNumpyArray(pItems, log2, type, slot, TRUE);
            retVal = TRUE;

            //if (NOEMPTYSLOT) {
            //   // Remove LAST ITEM
            //   // INSERT NEW OTEMS
            //}
            //printf("-- recycled with refcnt %llu   size: %llu  leadingz:%llu   type:%d\n", refCount, totalSize, leadingZero, type);
            //g_InitialCount[g_recyclecount] = refCount;
            //g_RecycleList[g_recyclecount++] = inArr;
         }
         else {

            LOGRECYCLE("Rejected recycling due to mem address clash refcnt: %llu   size: %llu   type: %d  %p %p\n", refCount, totalSize, type, inArr, PyArray_BYTES(inArr));
         }
      }
      else {
         LOGRECYCLE("Rejected recycling refcnt: %llu   size: %llu   type: %d  %p %p\n", refCount, totalSize, type, inArr, PyArray_BYTES(inArr));
      }

      // Force garbage collection
      GarbageCollect(g_GarbageCollectTimeSpan, FALSE);

   }
   else {
      LOGRECYCLE("Rejected recycling NOT FA refcnt: %llu   size: %llu   type: %d  %p %p\n", refCount, totalSize, type, inArr, PyArray_BYTES(inArr));

   }
   gRecursion--;
   return retVal;
}


//-----------------------------------------------------------------------------------------
// Called when user wants an array
//    First arg: existing  numpy array
//    Second arg: fastType form of dtype
PyObject *
AllocateNumpy(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr = NULL;
   int dtype;

   if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &inArr, &dtype)) return NULL;

   int ndim = PyArray_NDIM(inArr);
   npy_intp* dims = PyArray_DIMS(inArr);

   PyObject* result=(PyObject*)AllocateNumpyArray(ndim, dims, dtype);
   CHECK_MEMORY_ERROR(result);
   return result;
}

//-----------------------------------------------------------------------------------------
// Returns: TRUE if recycled
// Returns: FALSE if rejected recycling
BOOL RecycleNumpyInternal(PyArrayObject *inArr) {

   // Get to the base object
   PyArrayObject* pFirst = inArr;

   if (PyArray_BASE(inArr) == NULL) {
      // make sure we are base object and is a FastArray type (not Categorical or numpy array)
      if (inArr->ob_base.ob_type == g_FastArrayType) {
         return DeleteNumpyArray(inArr);
      }
   }
   return FALSE;
}


//-----------------------------------------------------------------------------------------
// Called when a FastArray is deleted
// Returns: True or False
// False if failed to recycle
PyObject *
RecycleNumpy(PyObject *self, PyObject *args)
{
   PyArrayObject *inArr = NULL;

   if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &inArr)) return NULL;

   BOOL retVal = RecycleNumpyInternal(inArr);

   if (retVal) {
      Py_INCREF(Py_True);
      return Py_True;
   }
   else {
      Py_INCREF(Py_False);
      return Py_False;
   }
}



//-----------------------------------------------------------------------------------------
// Called when a FastArray wants to look for recycled array
// Args:
//    First arg: integer in fast type (which is converted to a numpy type)
//    Second arg: tuple
PyObject *
TryRecycleNumpy(PyObject *self, PyObject *args)
{
   PyObject *tuple = NULL;
   int final_dtype;

   if (!PyArg_ParseTuple(args, "iO", &final_dtype, &tuple)) return NULL;

   LOGRECYCLE("TryRecycleNumpy dtype is %d \n", final_dtype);

   if (PyList_Check(tuple)) {
      Py_ssize_t size = PyList_Size(tuple);

      LOGRECYCLE("tuple has size of %llu\n", size);

      for (INT64 i = 0; i < size; i++) {
         PyObject* item = PyList_GetItem(tuple,i);
         //Py_DecRef(item);

         PyArrayObject* inArr = (PyArrayObject*)item;

         int ndim = PyArray_NDIM(inArr);
         npy_intp* dims = PyArray_DIMS(inArr);

         INT64 len = CalcArrayLength(ndim, dims);

         LOGRECYCLE("item %d is an array of %d dims and size %llu\n", (int)i, ndim, len);

         PyArrayObject* returnObject = RecycleFindArray(ndim, final_dtype, len);

         if (returnObject != NULL) {
            LOGRECYCLE("Found array to return with size %llu\n", len);

            // Flip to FastArray since we store the base object now
            return (SetFastArrayView(returnObject));
         }

      }
   }
   //TryRecycle(inArr);

   Py_INCREF(Py_None);
   return Py_None;

}

//--------------------------------------------
// Arg1: input cycle count to delete -- goes by GHZ or cycles
// Returns previous timespan
PyObject *
RecycleSetGarbageCollectTimeout(PyObject *self, PyObject *args) {
   INT64 timespan;

   if (!PyArg_ParseTuple(args, "L", &timespan)) return NULL;

   INT64 previousTimespan = g_GarbageCollectTimeSpan;
   g_GarbageCollectTimeSpan = timespan;
   return PyLong_FromLongLong(previousTimespan);
}

//-----------------------------------------------------------------------------------------
// Arg1: input cycle count to delete -- goes by GHZ or cycles
//       input the number 2 or 3 is roughly equivalent to 1 second
//       if your cpu is at 2.5 GHZ then input of 150 is equal to 60 seconds or 1 minute
// Will immediately run the garbagecollection with the specified timeout
// Returns TotalDeleted
PyObject *
RecycleGarbageCollectNow(PyObject *self, PyObject *args)
{
   INT64 timespan;

   if (!PyArg_ParseTuple(args, "L", &timespan)) return NULL;

   PyObject* pDict = PyDict_New();

   PyDict_SetItemString(pDict, "Hits", (PyObject*)PyLong_FromLongLong((long long)(g_FoundRecycled)));
   PyDict_SetItemString(pDict, "Misses", (PyObject*)PyLong_FromLongLong((long long)(g_DidNotFindRecycled)));
   PyDict_SetItemString(pDict, "Heads", (PyObject*)PyLong_FromLongLong((long long)(g_TotalHeads)));
   PyDict_SetItemString(pDict, "Tails", (PyObject*)PyLong_FromLongLong((long long)(g_TotalTails)));
   PyDict_SetItemString(pDict, "BitSz", (PyObject*)PyLong_FromLongLong((g_CurrentAllocBitSize)));
   PyDict_SetItemString(pDict, "HashSz", (PyObject*)PyLong_FromLongLong((g_CurrentAllocHashTable)));

   PyDict_SetItemString(pDict, "Alloc", (PyObject*)PyLong_FromLongLong((long long)(g_TotalAllocs)));
   PyDict_SetItemString(pDict, "Free", (PyObject*)PyLong_FromLongLong((long long)(g_TotalFree)));
   PyDict_SetItemString(pDict, "AllocSz", (PyObject*)PyLong_FromLongLong((long long)(g_TotalMemoryAllocated)));
   PyDict_SetItemString(pDict, "FreeSz", (PyObject*)PyLong_FromLongLong((long long)(g_TotalMemoryFreed)));
   PyDict_SetItemString(pDict, "Delta", (PyObject*)PyLong_FromLongLong((long long)g_TotalMemoryAllocated - (long long)g_TotalMemoryFreed));

   FreeWorkSpaceAllocLarge(g_pHashTableAny);
   FreeWorkSpaceAllocSmall(g_pBitFields);

   INT64 totalDeleted = GarbageCollect(timespan, FALSE);

   PyDict_SetItemString(pDict, "TotalDeleted", (PyObject*)PyLong_FromLongLong(totalDeleted));
   //return PyLong_FromLongLong(totalDeleted);
   return pDict;
}

//-----------------------------------------------------------------------------------------
// Called when python inits the module
void InitRecycler() {
   // Clear recycling array
   LOGRECYCLE("**Clearing recycler\n");
   memset(g_stRecycleList, 0, sizeof(stRecycleList) * 64*RECYCLE_MAXIMUM_TYPE);
}


