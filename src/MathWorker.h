#pragma once
#include "MathThreads.h"

//#define MATHLOGGING printf
#define MATHLOGGING(...)

#ifndef DECLARE_HANDLE
    #define DECLARE_HANDLE(name) \
        struct name##__ \
        { \
            int unused; \
        }; \
        typedef struct name##__ * name
DECLARE_HANDLE(HINSTANCE);
typedef HINSTANCE HMODULE;
#endif

typedef wchar_t WCHAR; // wc,   16-bit UNICODE character

extern "C"
{
    typedef INT_PTR(WINAPI * FARPROC)();
    typedef DWORD(WINAPI * PTHREAD_START_ROUTINE)(LPVOID lpThreadParameter);
    typedef PTHREAD_START_ROUTINE LPTHREAD_START_ROUTINE;

    extern void PrintCPUInfo();

#if defined(_WIN32)

    typedef HANDLE THANDLE;
    extern int GetProcCount();

    // VOID WINAPI Sleep(DWORD dwMilliseconds);
    // bool WINAPI CloseHandle(HANDLE hObject);
    // HANDLE WINAPI GetCurrentThread(VOID);
    // uint64_t WINAPI SetThreadAffinityMask(HANDLE hThread, uint64_t
    // dwThreadAffinityMask); bool WINAPI GetProcessAffinityMask(HANDLE hProcess,
    // uint64_t* lpProcessAffinityMask, uint64_t* lpSystemAffinityMask); HANDLE
    // WINAPI GetCurrentProcess(VOID); DWORD WINAPI GetLastError(VOID);

    // HANDLE WINAPI CreateThread(VOID* lpThreadAttributes, SIZE_T dwStackSize,
    // LPTHREAD_START_ROUTINE lpStartAddress, LPVOID lpParameter, DWORD
    // dwCreationFlags, LPDWORD lpThreadId); HMODULE WINAPI LoadLibraryW(const WCHAR*
    // lpLibFileName); FARPROC WINAPI GetProcAddress(HMODULE hModule, const char*
    // lpProcName);

#else
    typedef pthread_t THANDLE;

    int GetProcCount();
    void Sleep(unsigned int dwMilliseconds);
    bool CloseHandle(THANDLE hObject);

    uint64_t SetThreadAffinityMask(pid_t hThread, uint64_t dwThreadAffinityMask);

    bool GetProcessAffinityMask(HANDLE hProcess, uint64_t * lpProcessAffinityMask, uint64_t * lpSystemAffinityMask);
    pid_t GetCurrentThread();

    HANDLE GetCurrentProcess();

    unsigned int GetLastError();

    HANDLE CreateThread(void * lpThreadAttributes, size_t dwStackSize, LPTHREAD_START_ROUTINE lpStartAddress, void * lpParameter,
                        unsigned int dwCreationFlags, unsigned int * lpThreadId);

    HMODULE LoadLibraryW(const wchar_t * lpLibFileName);
    FARPROC GetProcAddress(HMODULE hModule, const char * lpProcName);

#endif
};

THANDLE StartThread(stWorkerRing * pWorkerRing);

// Move to reduce
#define MINF(x, y) x < y ? x : y
#define MAXF(x, y) x > y ? x : y

class CMathWorker
{
public:
    static const int64_t WORK_ITEM_CHUNK = stMATH_WORKER_ITEM::WORK_ITEM_CHUNK;
    static const int64_t WORK_ITEM_BIG = stMATH_WORKER_ITEM::WORK_ITEM_BIG;
    static const int64_t WORK_ITEM_MASK = stMATH_WORKER_ITEM::WORK_ITEM_MASK;
    static const int32_t MAX_WORKER_HANDLES = 64;

    int32_t WorkerThreadCount;

    // Set to true to stop threading
    bool NoThreading;

    // Set to true to stop allocating from a cache
    bool NoCaching;

    //------------------------------------------------------------------------------
    // Data Members
    stWorkerRing * pWorkerRing;

    THANDLE WorkerThreadHandles[MAX_WORKER_HANDLES];

    //------------------------------------------------------------------------------
    // Data Members
    CMathWorker()
    {
        PrintCPUInfo();

        WorkerThreadCount = GetProcCount();
        NoThreading = false;
        NoCaching = false;

        pWorkerRing = (stWorkerRing *)ALIGNED_ALLOC(sizeof(stWorkerRing), 64);
        memset(pWorkerRing, 0, sizeof(stWorkerRing));
        pWorkerRing->Init();

        for (int i = 0; i < WorkerThreadCount; i++)
        {
            WorkerThreadHandles[i] = 0;
        }
    };

    ~CMathWorker()
    {
        pWorkerRing->Cancel();
        Sleep(100);
        KillWorkerThreads();
        // DO NOT DEALLOCATE DO TO threads not exiting
        // ALIGNED_FREE(pWorkerRing);
    };

    //------------------------------------------------------------------------------
    // Returns number of worker threads + main thread
    int32_t GetNumCores()
    {
        // include main python thread
        return WorkerThreadCount + 1;
    }

    //---------------------------------
    // Changes how many threads wake up in Linux
    int32_t SetFutexWakeup(int howManyToWake)
    {
        if (howManyToWake < 1)
        {
            // On Windows seem to need at least 1
            howManyToWake = 1;
        }

        if (howManyToWake > FUTEX_WAKE_MAX)
        {
            // see linux man page on futex
            howManyToWake = FUTEX_WAKE_MAX;
        }

        int previousVal = pWorkerRing->FutexWakeCount;

        pWorkerRing->FutexWakeCount = howManyToWake;
        return previousVal;
    }

    int32_t GetFutexWakeup()
    {
        return pWorkerRing->FutexWakeCount;
    }

    //------------------------------------------------------------------------------
    //
    void StartWorkerThreads(int numaNode)
    {
        MATHLOGGING("Start worker threads\n");
        for (int i = 0; i < WorkerThreadCount; i++)
        {
            WorkerThreadHandles[i] = StartThread(pWorkerRing);
        }

        // Pin the main thread to a numa node?
        // TODO: work
        // uint64_t mask = ((uint64_t)1 << WorkerThreadCount);//core number starts
        // from 0
        // uint64_t ret = SetThreadAffinityMask(GetCurrentThread(),
        // (uint64_t)0xFFFF); SetThreadAffinityMask(GetCurrentThread(),
        // (uint64_t)0xFFFFFFFF);
    }

    //------------------------------------------------------------------------------
    //
    void KillWorkerThreads()
    {
        for (int i = 0; i < WorkerThreadCount; i++)
        {
            CloseHandle(WorkerThreadHandles[i]);
        }
    }

    //------------------------------------------------------------------------------
    //  Concurrent callback from multiple threads
    static bool MultiThreadedCounterCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
    {
        // -1 is the first core
        core = core + 1;
        bool didSomeWork = false;

        int64_t index;
        int64_t workBlock;

        // As long as there is work to do
        while ((index = pstWorkerItem->GetNextWorkIndex(&workBlock)) > 0)
        {
            // First index is 1 so we subtract
            index--;

            pstWorkerItem->MTWorkCallback(pstWorkerItem->WorkCallbackArg, core, index);

            didSomeWork = true;
            // tell others we completed this work block
            pstWorkerItem->CompleteWorkBlock();
        }
        return didSomeWork;
    }

    //-----------------------------------------------------------
    // Automatically handles threading vs no threading
    // Uses counters that start at 0 and go up from 1
    void DoMultiThreadedWork(int numItems, MTWORK_CALLBACK doMTWorkCallback, void * workCallbackArg, int32_t threadWakeup = 0)
    {
        // See if we get a work item (threading might be off)
        stMATH_WORKER_ITEM * pWorkItem = GetWorkItemCount(numItems);

        if (pWorkItem)
        {
            //
            // Each thread will call this routine with the callbackArg
            //
            pWorkItem->DoWorkCallback = MultiThreadedCounterCallback;
            pWorkItem->WorkCallbackArg = workCallbackArg;
            pWorkItem->MTWorkCallback = doMTWorkCallback;

            MATHLOGGING("before compress threaded\n");

            // This will notify the worker threads of a new work item
            WorkMain(pWorkItem, numItems, threadWakeup, 1, false);
            MATHLOGGING("after compress threaded\n");
        }
        else
        {
            // Just assume core 0 does all the work
            for (int t = 0; t < numItems; t++)
            {
                doMTWorkCallback(workCallbackArg, 0, t);
            }
        }
    }

    //------------------------------------------------------------------------------
    //  Concurrent callback from multiple threads
    //  Based on chunk size, each workIndex gets (0, 65536, 130000, etc.)
    // callback sig: typedef bool(*MTCHUNK_CALLBACK)(void* callbackArg, int core,
    // int64_t start, int64_t length);

    static bool MultiThreadedChunkCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
    {
        // -1 is the first core
        core = core + 1;
        bool didSomeWork = false;

        int64_t lenX;
        int64_t workBlock;

        // As long as there is work to do
        while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
        {
            int64_t start = pstWorkerItem->BlockSize * workBlock;

            pstWorkerItem->MTChunkCallback(pstWorkerItem->WorkCallbackArg, core, start, lenX);

            didSomeWork = true;
            // tell others we completed this work block
            pstWorkerItem->CompleteWorkBlock();
        }

        return didSomeWork;
    }

    //-----------------------------------------------------------
    // Automatically handles threading vs no threading
    // Used to divide up single array of data into chunks or sections
    // Returns true if actually did multithreaded work, otherwise false
    bool DoMultiThreadedChunkWork(int64_t lengthData, MTCHUNK_CALLBACK doMTChunkCallback, void * workCallbackArg,
                                  int32_t threadWakeup = 0)
    {
        // See if we get a work item (threading might be off)
        stMATH_WORKER_ITEM * pWorkItem = GetWorkItem(lengthData);

        if (pWorkItem)
        {
            //
            //
            // Each thread will call this routine with the callbackArg
            pWorkItem->DoWorkCallback = MultiThreadedChunkCallback;
            pWorkItem->WorkCallbackArg = workCallbackArg;
            pWorkItem->MTChunkCallback = doMTChunkCallback;

            // This will notify the worker threads of a new work item
            WorkMain(pWorkItem, lengthData, threadWakeup);
            return true;
        }
        else
        {
            // Just assume core 0 does all the work
            doMTChunkCallback(workCallbackArg, 0, 0, lengthData);
            return false;
        }
    }

    //--------------------------------------------
    // Caller must free return pointer to ppstBinCount
    // Memory is allocated in this routine
    //
    // Used for Reduce routines that work on a section
    // of bins based on the unique count
    //
    // Returns
    // -------
    // CORES actually used to be passed to DoMultiThreadedWork
    // pointer to stBinCount* to be freed with WORKSPACE_FREE
    // maxCores is the maximum cores allowed
    // pUser will be copied in
    //
    // numCores often passed to DoMultiThreadedWork(numCores,...)
    ///
    int64_t SegmentBins(int64_t bins, int64_t maxCores, stBinCount ** ppstBinCount)
    {
        // TODO: general purpose routine for this
        int32_t numCores = GetFutexWakeup();
        int64_t cores = numCores;

        // Check if we are clamping the core count
        if (maxCores > 0 && cores > maxCores)
        {
            cores = maxCores;
        }

        // shrink cores if we have too many
        if (bins < cores)
            cores = bins;

        // Allocate the struct to be freed later
        stBinCount * pstBinCount = (stBinCount *)WORKSPACE_ALLOC(cores * sizeof(stBinCount));

        if (cores > 0)
        {
            int64_t dividend = bins / cores;
            int64_t remainder = bins % cores;

            int64_t low = 0;
            int64_t high = 0;

            for (int64_t i = 0; i < cores; i++)
            {
                // Calculate band range
                high = low + dividend;

                // add in any remainder until nothing left
                if (remainder > 0)
                {
                    high++;
                    remainder--;
                }

                pstBinCount[i].BinLow = low;
                pstBinCount[i].BinHigh = high;
                pstBinCount[i].BinNum = i;
                pstBinCount[i].pUserMemory = NULL;

                // next low bin is the previous high bin
                low = high;
            }
        }
        *ppstBinCount = pstBinCount;
        return cores;
    }

    //------------------------------------------------------------------------------
    // Returns NULL if work item is too small or threading turned off
    // Otherwise returns a work item
    stMATH_WORKER_ITEM * GetWorkItemCount(int64_t len)
    {
        // If it is a small work item, process it immediately
        if (NoThreading)
        {
            return NULL;
        }

        // Otherwise allow parallel processing
        stMATH_WORKER_ITEM * pWorkItem = pWorkerRing->GetWorkItem();
        return pWorkItem;
    }

    //------------------------------------------------------------------------------
    // Returns NULL if work item is too small or threading turned off
    // Otherwise returns a work item
    stMATH_WORKER_ITEM * GetWorkItem(int64_t len)
    {
        // If it is a small work item, process it immediately
        if (len < WORK_ITEM_BIG || NoThreading)
        {
            return NULL;
        }

        // Otherwise allow parallel processing
        stMATH_WORKER_ITEM * pWorkItem = pWorkerRing->GetWorkItem();
        return pWorkItem;
    }

    //------------------------------------------------------------------------------
    // Called from main thread
    void WorkMain(stMATH_WORKER_ITEM * pWorkItem, int64_t len, int32_t threadWakeup, int64_t BlockSize = WORK_ITEM_CHUNK,
                  bool bGenericMode = true)
    {
        pWorkItem->TotalElements = len;

        const int32_t maxWakeup = GetFutexWakeup();

        MATHLOGGING("wakeup max:%d  requested:%d\n", maxWakeup, threadWakeup);
        // Only windows uses ThreadWakup
        // Linux uses the futex to wakup more threads
        // If the number of threads to wakeup is not specified, we use the default
        if (threadWakeup <= 0)
        {
            // use default number of threads
            threadWakeup = maxWakeup;
        }
        else
        {
            // use lower number to wake up threads
            threadWakeup = threadWakeup < maxWakeup ? threadWakeup : maxWakeup;
        }

        // only windows uses this for now
        pWorkItem->ThreadWakeup = threadWakeup;

        if (bGenericMode)
        {
            // WORK_ITEM_CHUNK at a time
            pWorkItem->BlockLast = (len + (BlockSize - 1)) / BlockSize;
        }
        else
        {
            // custom mode (called from groupby)
            // also can be called from parmerge
            pWorkItem->BlockLast = len + 1;
        }

        pWorkItem->BlocksCompleted = 0;
        pWorkItem->BlockNext = 0;
        pWorkItem->BlockSize = BlockSize;

        // Tell all worker threads about this new work item (futex or wakeall)
        // TODO: Consider waking a different number of threads based on complexity
        uint64_t currentTSC = __rdtsc();
        pWorkerRing->SetWorkItem(threadWakeup);

        MATHLOGGING("Took %lld cycles to wakeup\n", __rdtsc() - currentTSC);

        // Also do work
        pWorkItem->DoWork(-1, 0);

        if (bGenericMode)
        {
            // Check if all workers have completed
            while (pWorkItem->BlocksCompleted < pWorkItem->BlockLast)
            {
                MATHLOGGING("Waiting %llu  %llu \n", pWorkItem->BlocksCompleted, pWorkItem->BlockLast);
                YieldProcessor();
                // Sleep(0);
            }
        }
        else
        {
            // Check if all workers have completed
            while (pWorkItem->BlocksCompleted < len)
            {
                MATHLOGGING("Waiting %llu  %llu \n", pWorkItem->BlocksCompleted, pWorkItem->BlockLast);
                YieldProcessor();
                // Sleep(0);
            }
        }

        // Mark this as completed
        pWorkerRing->CompleteWorkItem();
    }

    //=================================================================================================================

    static bool AnyScatterGather(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
    {
        bool didSomeWork = false;
        OLD_CALLBACK * OldCallback = &pstWorkerItem->OldCallback;

        int64_t typeSizeIn = OldCallback->FunctionList->InputItemSize;
        char * pDataInX = (char *)OldCallback->pDataInBase1;
        int64_t lenX;
        int64_t workBlock;

        // Get the workspace calculation for this column
        stScatterGatherFunc * pstScatterGatherFunc = &((stScatterGatherFunc *)(OldCallback->pThreadWorkSpace))[core + 1];

        THREADLOGGING("[%d] DoWork start loop\n", core);

        // As long as there is work to do
        while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
        {
            // workBlock is length of work
            THREADLOGGING("[%d][%llu] Zero started working on %lld\n", core, workIndex, workBlock);

            int64_t offsetAdj = pstWorkerItem->BlockSize * workBlock * typeSizeIn;

            OldCallback->FunctionList->AnyScatterGatherCall(pDataInX + offsetAdj, lenX, pstScatterGatherFunc);

            // Indicate we completed a block
            didSomeWork = true;

            // tell others we completed this work block
            pstWorkerItem->CompleteWorkBlock();

            THREADLOGGING("[%d][%llu] Zero completed working on %lld\n", core, workIndex, workBlock);
        }
        return didSomeWork;
    }

    static bool AnyGroupby(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
    {
        bool didSomeWork = false;
        GROUPBY_FUNC groupByCall = (GROUPBY_FUNC)(pstWorkerItem->WorkCallbackArg);

        int64_t index;
        int64_t workBlock;

        THREADLOGGING("[%d] DoWork start loop\n", core);

        // As long as there is work to do
        while ((index = pstWorkerItem->GetNextWorkIndex(&workBlock)) > 0)
        {
            THREADLOGGING("[%d][%llu] Groupby started working on %lld\n", core, workIndex, workBlock - 1);

            groupByCall(pstWorkerItem->OldCallback.pDataInBase1, workBlock - 1);

            // Indicate we completed a block
            didSomeWork = true;

            // tell others we completed this work block
            pstWorkerItem->CompleteWorkBlock();

            THREADLOGGING("[%d][%llu] Groupby completed working on %lld\n", core, workIndex, workBlock - 1);
        }

        THREADLOGGING("[%d] Work item complete %lld\n", core, index);

        return didSomeWork;
    }

    static bool AnyTwoCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
    {
        bool didSomeWork = false;
        OLD_CALLBACK * OldCallback = &pstWorkerItem->OldCallback;

        int64_t strideSizeIn = OldCallback->FunctionList->InputItemSize;
        int64_t strideSizeOut = OldCallback->FunctionList->OutputItemSize;

        char * pDataInX = (char *)OldCallback->pDataInBase1;
        char * pDataInX2 = (char *)OldCallback->pDataInBase2;
        char * pDataOutX = (char *)OldCallback->pDataOutBase1;
        int64_t lenX;
        int64_t workBlock;

        // As long as there is work to do
        while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
        {
            // Calculate how much to adjust the pointers to get to the data for this
            // work block
            int64_t offsetAdj = pstWorkerItem->BlockSize * workBlock * strideSizeIn;
            int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * strideSizeOut;
            // int64_t outputAdj = offsetAdj;

            // Check if the outputtype is different
            // if (FunctionList->NumpyOutputType == NPY_BOOL) {
            //   assert(strideSizeOut == 1);
            //   outputAdj = BlockSize * workBlock * 1;
            //}

            // printf("workblock %llu   len=%llu  offset=%llu  strideSize %d\n",
            // workBlock, lenX, offsetAdj, strideSize);

            switch (OldCallback->FunctionList->TypeOfFunctionCall)
            {
            case ANY_TWO:
                {
                    switch (OldCallback->ScalarMode)
                    {
                    case NO_SCALARS:
                        // Process this block of work
                        OldCallback->FunctionList->AnyTwoStubCall(pDataInX + offsetAdj, pDataInX2 + offsetAdj,
                                                                  pDataOutX + outputAdj, lenX, OldCallback->ScalarMode);
                        break;

                    case FIRST_ARG_SCALAR:
                        // Process this block of work
                        OldCallback->FunctionList->AnyTwoStubCall(pDataInX, pDataInX2 + offsetAdj, pDataOutX + outputAdj, lenX,
                                                                  OldCallback->ScalarMode);
                        break;

                    case SECOND_ARG_SCALAR:
                        // Process this block of work
                        OldCallback->FunctionList->AnyTwoStubCall(pDataInX + offsetAdj, pDataInX2, pDataOutX + outputAdj, lenX,
                                                                  OldCallback->ScalarMode);
                        break;

                    case BOTH_SCALAR:
                        printf("** bug both are scalar!\n");
                        // Process this block of work
                        // FunctionList->AnyTwoStubCall(pDataInX, pDataInX2, pDataOutX +
                        // outputAdj, lenX, ScalarMode);
                        break;
                    }
                }
                break;
            case ANY_ONE:
                // Process this block of work
                OldCallback->FunctionList->AnyOneStubCall(pDataInX + offsetAdj, pDataOutX + outputAdj, lenX, strideSizeIn,
                                                          strideSizeOut);
                break;
            default:
                printf("unknown worker function\n");
                break;
            }

            // Indicate we completed a block
            didSomeWork = true;

            // tell others we completed this work block
            pstWorkerItem->CompleteWorkBlock();
            // printf("|%d %d", core, (int)workBlock);
        }

        return didSomeWork;
    }

    //=================================================================================================================

    //------------------------------------------------------------------------------
    //
    void WorkGroupByCall(GROUPBY_FUNC groupByCall, void * pstData, int64_t tupleSize)
    {
        // If it is a small work item, process it immediately
        if (tupleSize < 2 || NoThreading)
        {
            for (int i = 0; i < tupleSize; i++)
            {
                groupByCall(pstData, i);
            }
            return;
        }

        stMATH_WORKER_ITEM * pWorkItem = pWorkerRing->GetWorkItem();
        pWorkItem->DoWorkCallback = AnyGroupby;
        pWorkItem->WorkCallbackArg = (void *)groupByCall;

        // The only item that needs to be filled in for AnyGroupby
        pWorkItem->OldCallback.pDataInBase1 = pstData;

        WorkMain(pWorkItem, tupleSize, 0, 1, false);
    }

    //------------------------------------------------------------------------------
    // Designed to scatter gather
    void WorkScatterGatherCall(FUNCTION_LIST * anyScatterGatherCall, void * pDataIn, int64_t len, int64_t func,
                               stScatterGatherFunc * pstScatterGatherFunc)
    {
        // If it is a small work item, process it immediately
        if (len < WORK_ITEM_BIG || NoThreading)
        {
            anyScatterGatherCall->AnyScatterGatherCall(pDataIn, len, pstScatterGatherFunc);
            return;
        }

        // If we take this path, we are attempting to parallelize the operation
        stMATH_WORKER_ITEM * pWorkItem = pWorkerRing->GetWorkItem();
        pWorkItem->DoWorkCallback = AnyScatterGather;

        int32_t numCores = WorkerThreadCount + 1;
        int64_t sizeToAlloc = numCores * sizeof(stScatterGatherFunc);
        void * pWorkSpace = WORKSPACE_ALLOC(sizeToAlloc);

        if (pWorkSpace)
        {
            // Insert a work item
            pWorkItem->OldCallback.FunctionList = anyScatterGatherCall;
            pWorkItem->OldCallback.pDataInBase1 = pDataIn;
            pWorkItem->OldCallback.pDataInBase2 = NULL;
            pWorkItem->OldCallback.pDataOutBase1 = NULL;
            pWorkItem->OldCallback.pThreadWorkSpace = pWorkSpace;

            // Zero all the workspace values
            memset(pWorkSpace, 0, sizeToAlloc);

            stScatterGatherFunc * pZeroArray = (stScatterGatherFunc *)pWorkSpace;

            // Scatter the work amongst threads
            for (int i = 0; i < numCores; i++)
            {
                pZeroArray[i].inputType = pstScatterGatherFunc->inputType;
                pZeroArray[i].meanCalculation = pstScatterGatherFunc->meanCalculation;

                // the main thread is assigned core -1
                pZeroArray[i].core = i - 1;
            }

            // pWorkItem->TotalElements = len;
            // Kick off the worker threads for calculation
            WorkMain(pWorkItem, len, 0, WORK_ITEM_CHUNK, true);

            // Gather the results from all cores
            if (func == REDUCE_MIN || func == REDUCE_NANMIN || func == REDUCE_MAX || func == REDUCE_NANMAX)
            {
                int32_t calcs = 0;
                // Collect all the results...
                for (int i = 0; i < numCores; i++)
                {
                    pstScatterGatherFunc->lenOut += pZeroArray[i].lenOut;

                    // did we calc something?
                    if (pZeroArray[i].lenOut)
                    {
                        if (calcs == 0)
                        {
                            // We must accept the very first calculation
                            calcs++;
                            pstScatterGatherFunc->resultOut = pZeroArray[i].resultOut;
                            pstScatterGatherFunc->resultOutInt64 = pZeroArray[i].resultOutInt64;
                        }
                        else
                        {
                            if (func == REDUCE_MIN || func == REDUCE_NANMIN)
                            {
                                pstScatterGatherFunc->resultOut = MINF(pstScatterGatherFunc->resultOut, pZeroArray[i].resultOut);
                                pstScatterGatherFunc->resultOutInt64 =
                                    MINF(pstScatterGatherFunc->resultOutInt64, pZeroArray[i].resultOutInt64);
                            }
                            else
                            {
                                pstScatterGatherFunc->resultOut = MAXF(pstScatterGatherFunc->resultOut, pZeroArray[i].resultOut);
                                pstScatterGatherFunc->resultOutInt64 =
                                    MAXF(pstScatterGatherFunc->resultOutInt64, pZeroArray[i].resultOutInt64);
                            }
                        }
                    }
                }
            }
            else
            {
                // Gather for SUM plus other calculations
                MATHLOGGING("Gathering from %d cores\n", numCores);

                // Collect all the results...
                for (int i = 0; i < numCores; i++)
                {
                    pstScatterGatherFunc->lenOut += pZeroArray[i].lenOut;
                    pstScatterGatherFunc->resultOut += pZeroArray[i].resultOut;
                    pstScatterGatherFunc->resultOutInt64 += pZeroArray[i].resultOutInt64;
                }
            }
            WORKSPACE_FREE(pWorkSpace);
        }
    }

    ////------------------------------------------------------------------------------
    ////
    // void WorkOneStubCall(FUNCTION_LIST* anyOneStubCall, void* pDataIn, void*
    // pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    //   // If it is a small work item, process it immediately
    //   if (len < WORK_ITEM_BIG || NoThreading) {
    //      anyOneStubCall->AnyOneStubCall(pDataIn, pDataOut, len, strideIn,
    //      strideOut); return;
    //   }

    //   stMATH_WORKER_ITEM* pWorkItem = pWorkerRing->GetWorkItem();
    //   pWorkItem->DoWorkCallback = AnyTwoCallback;

    //   //Insert a work item
    //   pWorkItem->OldCallback.FunctionList = anyOneStubCall;
    //   pWorkItem->OldCallback.pDataInBase1 = pDataIn;
    //   pWorkItem->OldCallback.pDataInBase2 = NULL;
    //   pWorkItem->OldCallback.pDataOutBase1 = pDataOut;
    //   //pWorkItem->TotalElements = len;

    //   WorkMain(pWorkItem, len, 0, WORK_ITEM_CHUNK, true);
    //}
};

//------------------------------------------------------------
// declare the global math worker
extern CMathWorker * g_cMathWorker;
