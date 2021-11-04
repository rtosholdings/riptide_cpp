#pragma once

#include "CommonInc.h"

#if defined(__unix__) || defined(__unix) || defined(__APPLE__)
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/syscall.h>
    #include <sys/time.h>
    #include <sys/wait.h>
    #include <unistd.h>

    #if defined(__linux__)

        #include <linux/futex.h>

static int futex(int * uaddr, int futex_op, int val, const struct timespec * timeout, int * uaddr2, int val3)
{
    return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr, val3);
}

    #elif defined(__APPLE__)

// temp remove warnings
// #warning MathThreads does not yet support Darwin/macOS.
extern pthread_cond_t g_WakeupCond;

    #endif // defined(__linux__)

#endif // defined(__unix__) || defined(__unix) || defined(__APPLE__)

#define THREADLOGGING(...)
//#define THREADLOGGING printf

//--------------------------------------------------------------------
// bool
// WINAPI
// WaitOnAddress(
//   _In_reads_bytes_(AddressSize) volatile void * Address,
//   _In_reads_bytes_(AddressSize) void* CompareAddress,
//   _In_ size_t AddressSize,
//   _In_opt_ DWORD dwMilliseconds
//);
//
//
// void
// WINAPI
// WakeByAddressSingle(
//   _In_ void* Address
//);
//
//
// void
// WINAPI
// WakeByAddressAll(
//   _In_ void* Address
//);

//-------------------------------------------------------------------
//
// global scope
typedef void(WINAPI * WakeSingleAddress)(void *);
typedef void(WINAPI * WakeAllAddress)(void *);
typedef bool(WINAPI * WaitAddress)(volatile void *, void *, size_t, uint32_t);

extern WakeSingleAddress g_WakeSingleAddress;
extern WakeAllAddress g_WakeAllAddress;
extern WaitAddress g_WaitAddress;

// Forward declaration
extern FUNCTION_LIST * g_FunctionListArray[];

// Callback routine from worker thread
typedef bool (*DOWORK_CALLBACK)(struct stMATH_WORKER_ITEM * pstWorkerItem, int32_t core, int64_t workIndex);

// Callback routine from multithreaded worker thread (items just count up from
// 0,1,2,...)
typedef bool (*MTWORK_CALLBACK)(void * callbackArg, int32_t core, int64_t workIndex);

// Callback routine from multithreaded chunk thread (0, 65536, 130000, etc.)
typedef bool (*MTCHUNK_CALLBACK)(void * callbackArg, int32_t core, int64_t start, int64_t length);

// For auto binning we need to divide bins up amongst multiple thread
struct stBinCount
{
    // Valid if ... > BinLow && <= BinHigh
    int64_t BinLow;
    int64_t BinHigh;
    int64_t BinNum;
    void * pUserMemory;
};

struct OLD_CALLBACK
{
    FUNCTION_LIST * FunctionList;

    // Args to call
    union
    {
        void * pDataInBase1;
        void * pValues;
    };

    union
    {
        void * pDataInBase2;
        void * pIndex;
        void * pToSort;
    };

    void * pDataInBase3;

    //-------------------------------------------------
    union
    {
        void * pDataOutBase1;
        void * pWorkSpace;
    };

    // Total number of array elements
    union
    {
        // int64_t             TotalElements;
        int64_t IndexSize;

        // strlen is for sorting strings
        int64_t StrLen;
    };

    union
    {
        int32_t ScalarMode;
        int32_t MergeBlocks;
    };

    union
    {
        int64_t TotalElements2;
        int64_t ValSize;
    };

    // Default value to fill
    void * pDefault;

    void * pThreadWorkSpace;
};

//-----------------------------------------------------------
//
struct stMATH_WORKER_ITEM
{
    // -----------------------------------
    // Tested with addition
    // %timeit global a; a+= 5

    // a=arange(100_000)
    // operation 0x4000  0x8000   0x10000   0x20000
    // -------   ------  -------  -------   -------
    // a+=5        51      52 us     27 us   27
    // a+=5 nowait 49
    // a+b         49      50 us     46 us   46
    // sqrt:        83     104 us    209 us   209
    // sum:        54                26 us   26
    //
    // arange(1_000_000)
    // operation 0x4000  0x8000   0x10000   0x20000
    // -------   ------  -------  -------   -------
    // a+=5        114     120 us    118 us  133
    // a+b          91     121 us    128 us   46
    // sqrt:        252     293 us    293 us  209
    // sum:         50      51        52 us   68

    // a=arange(100_000.0)
    // operation 0x4000  0x8000   0x10000   0x20000
    // -------   ------  -------  -------   -------
    // a+b         69      xx      137 us    xx
    // sqrt:       85              209
    // sum:        52      xx      30 us     xx

    // Items larger than this might be worked on in parallel
    static const int64_t WORK_ITEM_CHUNK = 0x4000;
    static const int64_t WORK_ITEM_BIG = (WORK_ITEM_CHUNK * 2);
    static const int64_t WORK_ITEM_MASK = (WORK_ITEM_CHUNK - 1);

    // The callback to the thread routine that does work
    // with the argument to pass
    DOWORK_CALLBACK DoWorkCallback;
    void * WorkCallbackArg;

    // How many threads to wake up (atomic decrement)
    int64_t ThreadWakeup;

    // Used when calling MultiThreadedWork
    union
    {
        MTWORK_CALLBACK MTWorkCallback;
        MTCHUNK_CALLBACK MTChunkCallback;
    };

    // TotalElements is used on asymmetric last block
    int64_t TotalElements;

    // How many elements per block to work on
    int64_t BlockSize;

    // The last block to work on
    volatile int64_t BlockLast;

    //-------------------------------------------------
    // The next block (atomic)
    // Incremented
    // If BlockNext > BlockLast -- no work to be done
    volatile int64_t BlockNext;

    //-----------------------------------------------
    // Atomic access
    // When BlocksCompleted == BlockLast , the job is completed
    int64_t BlocksCompleted;

    // Get rid of this
    OLD_CALLBACK OldCallback;

    //==============================================================
    FORCE_INLINE int64_t GetWorkBlock()
    {
        int64_t val = InterlockedIncrement64(&BlockNext);
        return val - 1;
    }

    //==============================================================
    FORCE_INLINE void CompleteWorkBlock()
    {
        // Indicate we completed a block
        InterlockedIncrement64(&BlocksCompleted);
    }

    //=============================================================
    // Called by routines that work by index
    // returns 0 on failure
    // else returns length of workblock
    FORCE_INLINE int64_t GetNextWorkIndex(int64_t * workBlock)
    {
        int64_t wBlock = *workBlock = GetWorkBlock();

        // printf("working on block %llu\n", wBlock);

        // Make sure something to work on
        if (wBlock < BlockLast)
        {
            return wBlock;
        }

        return 0;
    }

    //=============================================================
    // Called by routines that work on chunks/blocks of memory
    // returns 0 on failure
    // else returns length of workblock
    FORCE_INLINE int64_t GetNextWorkBlock(int64_t * workBlock)
    {
        int64_t wBlock = *workBlock = GetWorkBlock();

        // printf("working on block %llu\n", wBlock);

        // Make sure something to work on
        if (wBlock < BlockLast)
        {
            int64_t lenWorkBlock;
            lenWorkBlock = BlockSize;

            // Check if this is the last workblock
            if ((wBlock + 1) == BlockLast)
            {
                // check if ends on perfect boundary
                if ((TotalElements & WORK_ITEM_MASK) != 0)
                {
                    // This is the last block and may have an odd number of data to
                    // process
                    lenWorkBlock = TotalElements & WORK_ITEM_MASK;
                    // printf("last workblock %llu  %llu  MASK  %llu\n", lenWorkBlock,
                    // TotalElements, WORK_ITEM_MASK);
                }
            }
            return lenWorkBlock;
        }
        return 0;
    }

    //------------------------------------------------------------------------------
    // Call this to do work until no work left to do
    // Returns TRUE if it did some work
    // Returns FALSE if it did no work
    // If core is -1, it is the main thread
    FORCE_INLINE bool DoWork(int core, int64_t workIndex)
    {
        return DoWorkCallback(this, core, workIndex);
    }
};

//-----------------------------------------------------------
// allocated on 64 byte alignment
struct stWorkerRing
{
    static const int64_t RING_BUFFER_SIZE = 1024;
    static const int64_t RING_BUFFER_MASK = 1023;

    volatile int64_t WorkIndex;
    volatile int64_t WorkIndexCompleted;

    // incremented when worker thread start
    volatile int64_t WorkThread;
    int32_t Reserved32;
    int32_t SleepTime;

    int32_t NumaNode;
    int32_t Cancelled;

    // Change this value to wake up less workers
    int32_t FutexWakeCount;

    stMATH_WORKER_ITEM WorkerQueue[RING_BUFFER_SIZE];

    void Init()
    {
        WorkIndex = 0;
        WorkIndexCompleted = 0;
        WorkThread = 0;
        NumaNode = 0;
        Cancelled = 0;
        SleepTime = 1;
        // how many threads to wake up on Linux
        FutexWakeCount = FUTEX_WAKE_DEFAULT;
    }

    FORCE_INLINE void Cancel()
    {
        Cancelled = 1;
    }

    FORCE_INLINE stMATH_WORKER_ITEM * GetWorkItem()
    {
        return &WorkerQueue[WorkIndex & RING_BUFFER_MASK];
    }

    FORCE_INLINE stMATH_WORKER_ITEM * GetExistingWorkItem()
    {
        return &WorkerQueue[(WorkIndex - 1) & RING_BUFFER_MASK];
    }

    FORCE_INLINE void SetWorkItem(int32_t maxThreadsToWake)
    {
        // This routine will wakup threads on Windows and Linux
        // Once we increment other threads will notice
        InterlockedIncrement64(&WorkIndex);

#if defined(_WIN32)
        // Are we allowed to wake threads?
        if (g_WakeAllAddress != NULL)
        {
            if (maxThreadsToWake < 5)
            {
                // In windows faster to wake single if just a few threads
                for (int i = 0; i < maxThreadsToWake; i++)
                {
                    g_WakeSingleAddress((void *)&WorkIndex);
                }
            }
            else
            {
                // In windows the more threads we wake up, the longer it takes to return
                // from this OS call
                g_WakeAllAddress((void *)&WorkIndex);
            }
        }

#elif defined(__linux__)
        // Linux thread wakeup
        int s = futex((int *)&WorkIndex, FUTEX_WAKE, maxThreadsToWake, NULL, NULL, 0);
        if (s == -1)
            printf("***error futex-FUTEX_WAKE\n"); // TODO: Change to use
                                                   // fprintf(stderr, msg) instead

#elif defined(__APPLE__)
        // temp remove warning
        //#warning MathThreads does not yet support Darwin/macOS.
        pthread_cond_broadcast(&g_WakeupCond);
#else
    #error riptide MathThreads support not implemented for this platform.

#endif
    }

    FORCE_INLINE void CompleteWorkItem()
    {
        InterlockedIncrement64(&WorkIndexCompleted);
    }
};

WakeSingleAddress InitWakeCalls();
// DWORD WINAPI WorkerThreadFunction(void* lpParam);
