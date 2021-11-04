#pragma once

#ifdef NEED_THIS_CODE

typedef struct SFW_ALIGN(16) _SLIST_ENTRY
{
    struct _SLIST_ENTRY * Next;
} SLIST_ENTRY, *PSLIST_ENTRY;

typedef union SFW_ALIGN(16) _SLIST_HEADER
{
    struct
    { // original struct
        ULONGLONG Alignment;
        ULONGLONG Region;
    } DUMMYSTRUCTNAME;
    struct
    { // x64 16-byte header
        ULONGLONG Depth    :16;
        ULONGLONG Sequence :48;
        ULONGLONG Reserved :4;
        ULONGLONG NextEntry:60; // last 4 bits are always 0's
    } HeaderX64;
} SLIST_HEADER, *PSLIST_HEADER;

extern "C"
{
    PSLIST_ENTRY __stdcall InterlockedPopEntrySList(PSLIST_HEADER ListHead);
    PSLIST_ENTRY __stdcall InterlockedPushEntrySList(PSLIST_HEADER ListHead, PSLIST_ENTRY ListEntry);
    VOID WINAPI InitializeSListHead(PSLIST_HEADER ListHead);
}

    #define LIST_HEADER LF_SLIST
    #define LIST_ENTRY LF_SLIST
    #define LIST_PUSH LockFreePushSList
    #define LIST_POP LockFreePopSList
    #define LIST_INIT_HEAD LockFreeInitSListHead
    #define LIST_FLUSH LockFreeFlushSList
    #define LIST_QUERY_DEPTH LockFreeGetCount

struct LF_COUNTER
{
    UINT32 Insertions;
    UINT32 Deletions;
};

    #pragma warning(push)
    #pragma warning(disable:4324) // structure padded due to align()

// Must be 16 byte aligned due to 128 bit swap
struct SFW_ALIGN(16) LF_SLIST
{
    // Because of CompareExchange128 syntax, we use unions to clarify which part
    // of the struct is (128 bit compare) vs (the high 64 bit compare and the low
    // 64 bit compare)
    union
    {
        struct LF_SLIST * Next;
        INT64 LowPart;
    };

    union
    {
        LF_COUNTER Counter;
        INT64 HighPart;
    };
};

    #pragma warning(pop)

static inline void LockFreeInitSListHead(LF_SLIST * pHead)
{
    _ASSERT(pHead != NULL);

    pHead->Next = NULL;
    pHead->HighPart = 0;
}

// May return NULL if the list is empty
static inline LF_SLIST * LockFreePopSList(LF_SLIST * pHead)
{
    LF_SLIST compare;
    LF_SLIST exchange;
    do
    {
        // the head may keep changing underneath us
        compare = *pHead;

        // we do not want pop the list if there are no entries (the list is
        // circular)
        if (compare.Next == NULL)
        {
            return NULL;
        }

        // Get what the head was pointing to (first item in the list)
        exchange.Next = compare.Next->Next;

        exchange.Counter.Insertions = compare.Counter.Insertions;

        // Keep track of deletions
        exchange.Counter.Deletions = compare.Counter.Deletions + 1;
    }
    while (! InterlockedCompareExchange128((LONGLONG *)pHead, exchange.HighPart, exchange.LowPart, (LONGLONG *)&compare));

    // Assertion assumes less than 4 billion insertions total lifetime
    _ASSERT(exchange.Counter.Insertions >= exchange.Counter.Deletions);

    return compare.Next;
}

// May return NULL if the list is empty
// NOTE: Should return the number of items in the list also
static inline LF_SLIST * LockFreeFlushSList(LF_SLIST * pHead)
{
    LF_SLIST compare;
    LF_SLIST exchange;
    do
    {
        // the head may keep changing underneath us
        compare = *pHead;

        // we do not want pop the list if there are no entries (the list is
        // circular)
        if (compare.Next == NULL)
        {
            return NULL;
        }

        // Point to NULL to indicate nothing in list
        exchange.Next = NULL;

        // The new head will have ZERO items
        exchange.Counter.Insertions = compare.Counter.Insertions;

        // Keep track of deletions -- make same as insertions to indicate ZERO items
        exchange.Counter.Deletions = compare.Counter.Insertions;
    }
    while (! InterlockedCompareExchange128((LONGLONG *)pHead, exchange.HighPart, exchange.LowPart, (LONGLONG *)&compare));

    // Assertion assumes less than 4 billion insertions total lifetime
    _ASSERT(exchange.Counter.Insertions >= exchange.Counter.Deletions);

    // SPECIAL Feature, we update the counter of the entry when a flush occurs
    // Tells us how many items we just popped off all at once
    compare.Next->Counter = compare.Counter;

    return compare.Next;
}

// The head will contain the value in pEntry when completed
// It assumes the pEntry is not in any other list
static inline void LockFreePushSList(LF_SLIST * pHead, LF_SLIST * pEntry)
{
    LF_SLIST compare;
    LF_SLIST exchange;

    do
    {
        // the head may keep changing underneath us
        compare = *pHead;

        // point to what the head was pointing to (we are inserting ourselves)
        pEntry->Next = compare.Next;

        // the head points to the new entry
        exchange.Next = pEntry;

        exchange.Counter.Deletions = compare.Counter.Deletions;

        // Keep track of insertions
        exchange.Counter.Insertions = compare.Counter.Insertions + 1;
    }
    while (! InterlockedCompareExchange128((LONGLONG *)pHead, exchange.HighPart, exchange.LowPart, (LONGLONG *)&compare));

    // Assertion assumes less than 4 billion insertions total lifetime
    _ASSERT(exchange.Counter.Insertions > exchange.Counter.Deletions);
}

// Returns number of elements in Q. It is a snapshot that might change
// immediately.
static inline UINT32 LockFreeGetCount(LF_SLIST * pHead)
{
    LF_COUNTER counter;
    counter = pHead->Counter;
    return (counter.Insertions - counter.Deletions);
}

struct FAST_BUF_SHARED
{
    INT64 Length;
    void * Data;
    INT64 ReferenceCount;
    INT64 Reserved;
};

struct ArenaNode
{
    SLIST_ENTRY next;
    INT64 allocSize;
    INT32 arenaIndex;
    INT32 magic1;
};

struct ArenaSlice
{
    SLIST_ENTRY next;
    INT64 allocSize;
    INT64 slices;
};

//----------------------------------------
struct ArenaEntry
{
    // Front of singly linked list
    SLIST_HEADER SListHead;

    INT64 nodesAllocated;
};

//-----------------------------------------
// Designed to allocate different chunks of memory
// from size 256 to 16 MB fast
// It will reuse the chunks in FIFO manner
class CArenaMemory
{
    static const INT64 arenaLowMask = 0xFF;

    static const INT32 arenaMagic = 0xC5832DE1;

    // 1 GB is the high value
    static const INT64 arenaHighMask = ((1024 * 1024 * 1024) - 1);
    static const INT64 arenaLowIndex = 0;
    static const INT64 arenaHighIndex = 31;
    static const INT64 arenaBitShift = 8;
    static const INT64 arenaMinNodesToAllocate = 4;

    ArenaEntry * pArenaTable;

    // used slices header
    SLIST_HEADER SlicesListHead;

public:
    CArenaMemory(void)
    {
        int allocSize = sizeof(ArenaEntry) * (arenaHighIndex + 1);

        pArenaTable = (ArenaEntry *)malloc(allocSize);
        RtlZeroMemory((void *)pArenaTable, allocSize);

        InitializeSListHead(&SlicesListHead);

        for (int i = 0; i < arenaHighIndex; i++)
        {
            InitializeSListHead(&pArenaTable[i].SListHead);
        }
    }

    void * CArenaMemory::Allocate(INT64 length);
    void * CArenaMemory::AllocateSlice(int arenaIndex);
    FAST_BUF_SHARED * CArenaMemory::AllocateFastBufferInternal(INT64 bufferSize);

    void CArenaMemory::FreeFastBufferInternal(FAST_BUF_SHARED * pFastBuf);

    bool CArenaMemory::Free(void * pBuffer);
    bool CArenaMemory::FreeAllSlices();
};

#endif