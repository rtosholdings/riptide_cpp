#include "RipTide.h"
#include "ndarray.h"

#include "CommonInc.h"
#include "ArenaMemory.h"
#include <pyarena.h>
#include <pymem.h>

#ifdef NEED_THIS_CODE

    #define LOGGING(...)
    #define LOGERROR printf

typedef struct
{
    PyMemAllocatorEx alloc;

    size_t malloc_size;
    size_t calloc_nelem;
    size_t calloc_elsize;
    void * realloc_ptr;
    size_t realloc_new_size;
    void * free_ptr;
    void * ctx;
} alloc_hook_t;

static void * hook_malloc(void * ctx, size_t size)
{
    alloc_hook_t * hook = (alloc_hook_t *)ctx;
    hook->ctx = ctx;
    hook->malloc_size = size;
    return hook->alloc.malloc(hook->alloc.ctx, size);
}

static void * hook_calloc(void * ctx, size_t nelem, size_t elsize)
{
    alloc_hook_t * hook = (alloc_hook_t *)ctx;
    hook->ctx = ctx;
    hook->calloc_nelem = nelem;
    hook->calloc_elsize = elsize;
    return hook->alloc.calloc(hook->alloc.ctx, nelem, elsize);
}

static void * hook_realloc(void * ctx, void * ptr, size_t new_size)
{
    alloc_hook_t * hook = (alloc_hook_t *)ctx;
    hook->ctx = ctx;
    hook->realloc_ptr = ptr;
    hook->realloc_new_size = new_size;
    return hook->alloc.realloc(hook->alloc.ctx, ptr, new_size);
}

static void hook_free(void * ctx, void * ptr)
{
    alloc_hook_t * hook = (alloc_hook_t *)ctx;
    hook->ctx = ctx;
    hook->free_ptr = ptr;
    hook->alloc.free(hook->alloc.ctx, ptr);
}

static PyObject * test_setallocators(PyMemAllocatorDomain domain)
{
    PyObject * res = NULL;
    alloc_hook_t hook;
    PyMemAllocatorEx alloc;
    size_t size;
    // const char *error_msg;
    // size_t size2, nelem, elsize;
    void * ptr;
    // void *ptr2;

    memset(&hook, 0, sizeof(hook));

    alloc.ctx = &hook;
    alloc.malloc = &hook_malloc;
    alloc.calloc = &hook_calloc;
    alloc.realloc = &hook_realloc;
    alloc.free = &hook_free;
    PyMem_GetAllocator(domain, &hook.alloc);
    PyMem_SetAllocator(domain, &alloc);

    /* malloc, realloc, free */
    size = 42;
    hook.ctx = NULL;
    switch (domain)
    {
    case PYMEM_DOMAIN_RAW:
        ptr = PyMem_RawMalloc(size);
        break;
    case PYMEM_DOMAIN_MEM:
        ptr = PyMem_Malloc(size);
        break;
    case PYMEM_DOMAIN_OBJ:
        ptr = PyObject_Malloc(size);
        break;
    default:
        ptr = NULL;
        break;
    }

    // put it back
    PyMem_SetAllocator(domain, &hook.alloc);
    return res;
}

//
//
// static PyMemAllocatorEx allocator = {
//   .ctx = NULL,
//   .malloc = python_malloc,
//   .calloc = python_calloc,
//   .realloc = python_realloc,
//   .free = python_free
//};
//
// void
// kore_python_init(void)
//{
//   PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &allocator);
//   PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &allocator);
//   PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &allocator);
//   PyMem_SetupDebugHooks();
//
//   if (PyImport_AppendInittab("kore", &python_module_init) == -1)
//      fatal("kore_python_init: failed to add new module");
//
//   Py_Initialize();
//}

//
// void
// PyMem_GetAllocator(PyMemAllocatorDomain domain, PyMemAllocatorEx *allocator)
//{
//   switch (domain)
//   {
//   case PYMEM_DOMAIN_RAW: *allocator = _PyMem_Raw; break;
//   case PYMEM_DOMAIN_MEM: *allocator = _PyMem; break;
//   case PYMEM_DOMAIN_OBJ: *allocator = _PyObject; break;
//   default:
//      /* unknown domain: set all attributes to NULL */
//      allocator->ctx = NULL;
//      allocator->malloc = NULL;
//      allocator->calloc = NULL;
//      allocator->realloc = NULL;
//      allocator->free = NULL;
//   }
//}
//
// void
// PyMem_SetAllocator(PyMemAllocatorDomain domain, PyMemAllocatorEx *allocator)
//{
//   switch (domain)
//   {
//   case PYMEM_DOMAIN_RAW: _PyMem_Raw = *allocator; break;
//   case PYMEM_DOMAIN_MEM: _PyMem = *allocator; break;
//   case PYMEM_DOMAIN_OBJ: _PyObject = *allocator; break;
//      /* ignore unknown domain */
//   }
//}
//
// void
// PyObject_GetArenaAllocator(PyObjectArenaAllocator *allocator)
//{
//   *allocator = _PyObject_Arena;
//}
//
// void
// PyObject_SetArenaAllocator(PyObjectArenaAllocator *allocator)
//{
//   _PyObject_Arena = *allocator;
//}

void * InternalAlloc(size_t allocSize)
{
    return malloc(allocSize);
}

void InternalFree(void * memory)
{
    return free(memory);
}

void * CArenaMemory::AllocateSlice(int arenaIndex)
{
    ArenaEntry * pEntry = &pArenaTable[arenaIndex];

    INT64 nodesAllocated = pEntry->nodesAllocated * 2;

    if (nodesAllocated == 0)
    {
        nodesAllocated = arenaMinNodesToAllocate;
    }

    // Calculate the slice size for this bin
    int sliceSize = (arenaLowMask + 1) << arenaIndex;
    sliceSize += sizeof(ArenaNode);

    LOGGING("sliceSize %d   arenaIndex %d\n", sliceSize, arenaIndex);

    INT64 allocSize = sizeof(ArenaSlice) + (nodesAllocated * sliceSize);

    // TODO: Change malloc to numa alloc
    ArenaSlice * pNewSlice = (ArenaSlice *)InternalAlloc(allocSize);

    // No need to zero this
    // RtlZeroMemory(pNewSlice, allocSize);

    pNewSlice->allocSize = allocSize;
    pNewSlice->slices = nodesAllocated;
    pNewSlice->next.Next = 0;

    // Move past header
    char * pSlice = (char *)pNewSlice;
    pSlice += sizeof(ArenaSlice);

    // Queue up all of our slices
    for (int i = 0; i < nodesAllocated; i++)
    {
        ArenaNode * pNode = (ArenaNode *)pSlice;

        pNode->next.Next = NULL;
        pNode->allocSize = sliceSize;
        pNode->magic1 = arenaMagic;
        pNode->arenaIndex = arenaIndex;

        // This will push to front of list
        InterlockedPushEntrySList(&pEntry->SListHead, &pNode->next);

        pSlice += sliceSize;
    }

    //  Book keeping
    pEntry->nodesAllocated += nodesAllocated;

    // place slice on free list
    InterlockedPushEntrySList(&SlicesListHead, &pNewSlice->next);

    return pNewSlice;
}

//-----------------------------------------------
void * CArenaMemory::Allocate(INT64 length)
{
    int sizeIndex = arenaLowIndex;

    if (length > arenaHighMask)
    {
        // Anything above 1 GB is not kept around
        LOGGING("!! error too large an allocation %llu\n", length);
        INT64 sizetoAlloc = sizeof(ArenaNode) + length;

        ArenaNode * pNode = (ArenaNode *)InternalAlloc(sizetoAlloc);

        if (pNode == NULL)
        {
            LOGERROR("Failed to allocate memory %llu\n", sizetoAlloc);
            return NULL;
        }
        pNode->allocSize = sizetoAlloc;
        pNode->arenaIndex = -1;
        pNode->magic1 = arenaMagic;
        pNode->next.Next = 0;

        // Return past the bookkeeping struct
        char * pByteAddress = (char *)pNode;
        pByteAddress += sizeof(ArenaNode);
        return pByteAddress;
        // return NULL;
        // sizeIndex = arenaHighIndex;
    }
    else
    {
        INT64 findMSB = length;

        // initial shift since everything 256 bytes or smaller = same chunk
        findMSB >>= arenaBitShift;

        // Keep shifting until we find proper spot
        while (findMSB > 0)
        {
            sizeIndex++;
            findMSB >>= 1;
        }
    }

    // printf("size %d   arenaIndex %d\n", length, sizeIndex);

    // See if we have that size of memory available
    ArenaEntry * pEntry = &pArenaTable[sizeIndex];

    // Try to atomically pop it off
    ArenaNode * pNode = (ArenaNode *)InterlockedPopEntrySList(&(pEntry->SListHead));

    if (pNode == NULL)
    {
        // Allocate?  Slice up
        AllocateSlice(sizeIndex);
        pNode = (ArenaNode *)InterlockedPopEntrySList(&(pEntry->SListHead));
    }

    if (pNode == NULL)
    {
        LOGERROR("!!! error out of memory when trying to get memory for index %d\n", sizeIndex);
        return pNode;
    }

    assert(pNode->magic1 == arenaMagic);

    // Return past the bookkeeping struct
    char * pByteAddress = (char *)pNode;
    pByteAddress += sizeof(ArenaNode);

    return pByteAddress;
}

//-------------------------------------------------------
// Reference counted buffer
// FAST_BUF zeroed out
// Length and Data valid upon return and can be immediately used
//
FAST_BUF_SHARED * CArenaMemory::AllocateFastBufferInternal(INT64 bufferSize)
{
    // Combine both buffers together
    FAST_BUF_SHARED * pFastBuf = (FAST_BUF_SHARED *)Allocate(sizeof(FAST_BUF_SHARED) + bufferSize);

    if (pFastBuf == NULL)
    {
        printf("Failed to alloc shared memory");
    }

    // Zero front of structure (helpful for OVERLAPPED IO if used)
    RtlZeroMemory(pFastBuf, sizeof(FAST_BUF_SHARED));

    pFastBuf->Length = (UINT32)bufferSize;
    pFastBuf->Data = ((BYTE *)pFastBuf) + sizeof(FAST_BUF_SHARED);

    // This packet can get freed up in two locations
    pFastBuf->ReferenceCount = 1;

    return pFastBuf;
}

void CArenaMemory::FreeFastBufferInternal(FAST_BUF_SHARED * pFastBuf)
{
    INT64 result = InterlockedDecrement64(&pFastBuf->ReferenceCount);

    if (result <= 0)
    {
        if (result < 0)
        {
            LOGERROR(
                "!! reference count below 0.  This might be a shared memory "
                "buffer\n");
        }
        else
        {
            Free(pFastBuf);
        }
    }
}

bool CArenaMemory::Free(void * pBuffer)
{
    // Look backwards in memory to get to our bookkeeping
    char * pByteAddress = (char *)pBuffer;
    pByteAddress -= sizeof(ArenaNode);

    ArenaNode * pNode = (ArenaNode *)pByteAddress;

    // Check for large memory which does not return to list
    if (pNode->arenaIndex == -1)
    {
        InternalFree(pNode);
        return true;
    }

    if (pNode->magic1 != arenaMagic)
    {
        LOGERROR("!! error not freeing memory or corrupted\n");
        return false;
    }

    if (pNode->arenaIndex < 0 || pNode->arenaIndex > arenaHighIndex)
    {
        LOGERROR("!! error not freeing memory or index corrupted\n");
        return false;
    }

    // printf("!!free  %d  %d\n", pNode->allocSize, pNode->arenaIndex);

    // See if we have that size of memory available
    ArenaEntry * pEntry = &pArenaTable[pNode->arenaIndex];

    // This will push to front of list
    InterlockedPushEntrySList(&pEntry->SListHead, &pNode->next);

    return true;
}

// Cannot be called without lock held
bool CArenaMemory::FreeAllSlices()
{
    // Try to atomically pop it off
    ArenaSlice * pSlice;
    pSlice = (ArenaSlice *)InterlockedPopEntrySList(&SlicesListHead);

    while (pSlice != NULL)
    {
        InternalFree(pSlice);
        pSlice = (ArenaSlice *)InterlockedPopEntrySList(&SlicesListHead);
    }

    return true;
}

#endif
