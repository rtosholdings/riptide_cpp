#pragma once

#include "CommonInc.h"

//==================================================================================

// allocated on 64 byte alignment
struct stRecycleItem {
   INT16                type;          // numpy type -- up to maximum type
   INT16                initRefCount;  // ref count when first put in q
   INT32                ndim;          // number of dims
   INT64                totalSize;     // total size
   UINT64               tsc;           // time stamp counter
   void*                memoryAddress;
   PyArrayObject*       recycledArray;
};


static const INT64 RECYCLE_ENTRIES = 64;
static const INT64 RECYCLE_MAXIMUM_TYPE = 14;
static const INT64 RECYCLE_MAXIMUM_SEARCH = 4;
static const INT64 RECYCLE_MASK = 3;
static const INT64 RECYCLE_MIN_SIZE = 1;

struct stRecycleList {
   // Circular list, when Head==Tail no items

   INT32             Head;
   INT32             Tail;

   stRecycleItem     Item[RECYCLE_MAXIMUM_SEARCH];
};

PyObject* AllocateNumpy(PyObject *self, PyObject *args);
PyObject* RecycleNumpy(PyObject *self, PyObject *args);
PyObject* RecycleGarbageCollectNow(PyObject *self, PyObject *args);
PyObject* RecycleSetGarbageCollectTimeout(PyObject *self, PyObject *args);
PyObject* RecycleDump(PyObject *self, PyObject *args);
PyObject *TryRecycleNumpy(PyObject *self, PyObject *args);

PyObject *SetRecycleMode(PyObject *self, PyObject *args);


PyArrayObject* RecycleFindArray(INT32 ndim, INT32 type, INT64 totalSize);

void InitRecycler();

BOOL RecycleNumpyInternal(PyArrayObject *inArr);

void* WorkSpaceAllocLarge(size_t HashTableAllocSize);
void WorkSpaceFreeAllocLarge(void* &pHashTableAny, size_t HashTableAllocSize);

void* WorkSpaceAllocSmall(size_t BitAllocSize);
void WorkSpaceFreeAllocSmall(void* &pBitFields, size_t BitAllocSize);

INT64 GarbageCollect(INT64 timespan, bool verbose);