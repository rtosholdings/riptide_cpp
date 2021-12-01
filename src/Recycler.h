#pragma once

#include "CommonInc.h"

//==================================================================================

// allocated on 64 byte alignment
struct stRecycleItem
{
    int16_t type;         // numpy type -- up to maximum type
    int16_t initRefCount; // ref count when first put in q
    int32_t ndim;         // number of dims
    int64_t totalSize;    // total size
    uint64_t tsc;         // time stamp counter
    void * memoryAddress;
    PyArrayObject * recycledArray;
};

static const int64_t RECYCLE_ENTRIES = 64;
static const int64_t RECYCLE_MAXIMUM_TYPE = 14;
static const int64_t RECYCLE_MAXIMUM_SEARCH = 4;
static const int64_t RECYCLE_MASK = 3;
static const int64_t RECYCLE_MIN_SIZE = 1;

struct stRecycleList
{
    // Circular list, when Head==Tail no items

    int32_t Head;
    int32_t Tail;

    stRecycleItem Item[RECYCLE_MAXIMUM_SEARCH];
};

PyObject * AllocateNumpy(PyObject * self, PyObject * args);
PyObject * RecycleNumpy(PyObject * self, PyObject * args);
PyObject * RecycleGarbageCollectNow(PyObject * self, PyObject * args);
PyObject * RecycleSetGarbageCollectTimeout(PyObject * self, PyObject * args);
PyObject * RecycleDump(PyObject * self, PyObject * args);
PyObject * TryRecycleNumpy(PyObject * self, PyObject * args);

PyObject * SetRecycleMode(PyObject * self, PyObject * args);

PyArrayObject * RecycleFindArray(int32_t ndim, int32_t type, int64_t totalSize);

void InitRecycler();

bool RecycleNumpyInternal(PyArrayObject * inArr);

void * WorkSpaceAllocLarge(size_t HashTableAllocSize);
void WorkSpaceFreeAllocLarge(void *& pHashTableAny, size_t HashTableAllocSize);

void * WorkSpaceAllocSmall(size_t BitAllocSize);
void WorkSpaceFreeAllocSmall(void *& pBitFields, size_t BitAllocSize);

int64_t GarbageCollect(int64_t timespan, bool verbose);
