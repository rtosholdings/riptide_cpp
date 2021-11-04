#include "MathThreads.h"
#include "MathWorker.h"
#include "platform_detect.h"

// to debug thread wakeup allow LOGGING to printf
//#define LOGGING printf
#define LOGGING(...)

#if defined(RT_OS_DARWIN)
/* For MacOS use a conditional wakeup */
pthread_cond_t g_WakeupCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t g_WakeupMutex = PTHREAD_MUTEX_INITIALIZER;
#endif

#if defined(RT_OS_WINDOWS)
WakeSingleAddress g_WakeSingleAddress = InitWakeCalls();
WakeAllAddress g_WakeAllAddress;
WaitAddress g_WaitAddress;

//-----------------------------------------------------------------
// Not every version of Windows has this useful API so we have to check for it
// dynamically
WakeSingleAddress InitWakeCalls()
{
    FARPROC fp;

    HMODULE hModule = LoadLibraryW(L"kernelbase.dll");

    if (hModule != NULL)
    {
        fp = GetProcAddress(hModule, "WakeByAddressSingle");
        if (fp != NULL)
        {
            // LogInform("**System supports WakeByAddressSingle ...\n");
            g_WakeSingleAddress = (VOID(WINAPI *)(PVOID))fp;

            fp = GetProcAddress(hModule, "WakeByAddressAll");
            g_WakeAllAddress = (WakeAllAddress)fp;

            fp = GetProcAddress(hModule, "WaitOnAddress");
            g_WaitAddress = (WaitAddress)fp;
        }
        else
        {
            LogInform("**System does NOT support WakeByAddressSingle ...\n");
            g_WakeSingleAddress = NULL;
            g_WakeAllAddress = NULL;
            g_WaitAddress = NULL;
        }
    }

    return g_WakeSingleAddress;
}

#else
WakeSingleAddress g_WakeSingleAddress = NULL;
WakeAllAddress g_WakeAllAddress = NULL;
WaitAddress g_WaitAddress = NULL;
#endif

//-----------------------------------------------------------
// Main thread loop
// Threads will wait on an address then wake up when there is work
// Linux uses a futex to control how many threads wakeup
// Windows uses a counter
// Darwin (macOS) does not support futexes or WaitOnAddress, so it will need to
// use one of:
//   * POSIX condition variables
//   * C++11 condition variables from <atomic>
//   * libdispatch (GCD), using dispatch_semaphore_t (via
//   dispatch_semaphore_create()) to control concurrency; include
//   <dispatch/semaphore.h>
//   * BSD syscalls like __psynch_cvwait (and other __psynch functions). These
//   are not externally documented -- need to look in
//   github.com/apple/darwin-libpthread to see how things work.
//
#if defined(RT_OS_WINDOWS)
DWORD WINAPI WorkerThreadFunction(LPVOID lpParam)
#else
void * WorkerThreadFunction(void * lpParam)
#endif
{
    stWorkerRing * pWorkerRing = (stWorkerRing *)lpParam;

    uint32_t core = (uint32_t)(InterlockedIncrement64(&pWorkerRing->WorkThread));
    core = core - 1;

    // if (core > 3) core += 16;
    // core += 16;

    LOGGING("Thread created with parameter: %d   %p\n", core, g_WaitAddress);

    // On windows we set the thread affinity mask
    if (g_WaitAddress != NULL)
    {
        uint64_t mask = (uint64_t)(1) << core; // core number starts from 0
        uint64_t ret = SetThreadAffinityMask(GetCurrentThread(), mask);
        // UINT64 ret = SetThreadAffinityMask(GetCurrentThread(), 0xFFFFFFFF);
    }

    int64_t lastWorkItemCompleted = -1;

    //
    // Setting Cancelled will stop all worker threads
    //
    while (pWorkerRing->Cancelled == 0)
    {
        int64_t workIndexCompleted;
        int64_t workIndex;

        workIndex = pWorkerRing->WorkIndex;
        workIndexCompleted = pWorkerRing->WorkIndexCompleted;

        bool didSomeWork = false;

        // See if work to do
        if (workIndex > workIndexCompleted)
        {
            stMATH_WORKER_ITEM * pWorkItem = pWorkerRing->GetExistingWorkItem();

#if defined(RT_OS_WINDOWS)
            // Windows we check if the work was for our thread
            int64_t wakeup = InterlockedDecrement64(&pWorkItem->ThreadWakeup);
            if (wakeup >= 0)
            {
                didSomeWork = pWorkItem->DoWork(core, workIndex);
            }
            else
            {
                // printf("[%d] not doing work %lld.  %lld  %lld\n", core, wakeup,
                // workIndex, workIndexCompleted); workIndex++;
            }
#else
            didSomeWork = pWorkItem->DoWork(core, workIndex);

#endif
        }

        if (! didSomeWork)
        {
            workIndexCompleted = workIndex;

#if defined(RT_OS_WINDOWS)
            // printf("Sleeping %d", core);
            if (g_WaitAddress == NULL)
            {
                // For Windows 7 we just sleep
                Sleep(pWorkerRing->SleepTime);
            }
            else
            {
                if (! didSomeWork)
                {
                    // workIndexCompleted++;
                }

                LOGGING("[%d] WaitAddress %llu  %llu  %d\n", core, workIndexCompleted, pWorkerRing->WorkIndex, (int)didSomeWork);

                // Otherwise wake up using conditional variable
                g_WaitAddress(&pWorkerRing->WorkIndex, (PVOID)&workIndexCompleted,
                              8, // The size of the value being waited on (i.e. the number of
                                 // bytes to read from the two pointers then compare).
                              1000000L);
            }
#elif defined(RT_OS_LINUX)

            LOGGING("[%d] WaitAddress %llu  %llu  %d\n", core, workIndexCompleted, pWorkerRing->WorkIndex, (int)didSomeWork);

            // int futex(int *uaddr, int futex_op, int val,
            //   const struct timespec *timeout,   /* or: uint32_t val2 */
            //   int *uaddr2, int val3);
            futex((int *)&pWorkerRing->WorkIndex, FUTEX_WAIT, (int)workIndexCompleted, NULL, NULL, 0);

#elif defined(RT_OS_DARWIN)
            LOGGING("[%lu] WaitAddress %llu  %llu  %d\n", core, workIndexCompleted, pWorkerRing->WorkIndex, (int)didSomeWork);

            pthread_mutex_lock(&g_WakeupMutex);
            pthread_cond_wait(&g_WakeupCond, &g_WakeupMutex);
            pthread_mutex_unlock(&g_WakeupMutex);

#else
    #error riptide MathThreads support needs to be implemented for this platform.

#endif

            // printf("Waking %d", core);

            // YieldProcessor();
        }
        // YieldProcessor();
    }

    printf("Thread %d exiting!!!\n", (int)core);
#if defined(RT_OS_WINDOWS)
    return 0;
#else
    return NULL;
#endif
}

#if defined(RT_OS_WINDOWS)

//-----------------------------------------------------------
//
THANDLE StartThread(stWorkerRing * pWorkerRing)
{
    DWORD dwThreadId;
    THANDLE hThread;

    hThread = CreateThread(NULL,                 // default security attributes
                           0,                    // use default stack size
                           WorkerThreadFunction, // thread function
                           pWorkerRing,          // argument to thread function
                           0,                    // use default creation flags
                           &dwThreadId);         // returns the thread identifier

    // printf("The thread ID: %d.\n", dwThreadId);

    // Check the return value for success. If something wrong...
    if (hThread == NULL)
    {
        printf("CreateThread() failed, error: %d.\n", GetLastError());
        return NULL;
    }

    return hThread;
}

#else

//-----------------------------------------------------------
//
THANDLE StartThread(stWorkerRing * pWorkerRing)
{
    int err;
    THANDLE hThread;

    err = pthread_create(&hThread, NULL, &WorkerThreadFunction, pWorkerRing);

    if (err != 0)
    {
        printf("*** Cannot create thread :[%s]\n", strerror(err));
    }

    return hThread;
}
#endif
