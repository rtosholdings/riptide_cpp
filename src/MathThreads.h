#pragma once

#include "CommonInc.h"

#if defined(__unix__) || defined(__unix) || defined(__APPLE__)
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <fcntl.h>

#if defined(__linux__)

#include <linux/futex.h>

static int
futex(int *uaddr, int futex_op, int val,
   const struct timespec *timeout, int *uaddr2, int val3)
{
   return syscall(SYS_futex, uaddr, futex_op, val,
      timeout, uaddr, val3);
}

#elif defined(__APPLE__)

// temp remove warnings
// #warning MathThreads does not yet support Darwin/macOS.
extern pthread_cond_t  g_WakeupCond;

#endif  // defined(__linux__)

#endif  // defined(__unix__) || defined(__unix) || defined(__APPLE__)

#define THREADLOGGING(...)
//#define THREADLOGGING printf


//--------------------------------------------------------------------
//BOOL
//WINAPI
//WaitOnAddress(
//   _In_reads_bytes_(AddressSize) volatile VOID * Address,
//   _In_reads_bytes_(AddressSize) PVOID CompareAddress,
//   _In_ SIZE_T AddressSize,
//   _In_opt_ DWORD dwMilliseconds
//);
//
//
//VOID
//WINAPI
//WakeByAddressSingle(
//   _In_ PVOID Address
//);
//
//
//VOID
//WINAPI
//WakeByAddressAll(
//   _In_ PVOID Address
//);

//-------------------------------------------------------------------
//
// global scope
typedef VOID(WINAPI *WakeSingleAddress)(PVOID);
typedef VOID(WINAPI *WakeAllAddress)(PVOID);
typedef BOOL(WINAPI *WaitAddress)(volatile VOID*, PVOID, SIZE_T, DWORD);

extern WakeSingleAddress g_WakeSingleAddress;
extern WakeAllAddress g_WakeAllAddress;
extern WaitAddress g_WaitAddress;

// Forward declaration
extern FUNCTION_LIST*   g_FunctionListArray[];

// Callback routine from worker thread
typedef BOOL(*DOWORK_CALLBACK)(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex);

// Callback routine from multithreaded worker thread (items just count up from 0,1,2,...)
typedef BOOL(*MTWORK_CALLBACK)(void* callbackArg, int core, INT64 workIndex);

// Callback routine from multithreaded chunk thread (0, 65536, 130000, etc.)
typedef BOOL(*MTCHUNK_CALLBACK)(void* callbackArg, int core, INT64 start, INT64 length);

// For auto binning we need to divide bins up amongst multiple thread
struct stBinCount {
   // Valid if ... > BinLow && <= BinHigh
   INT64 BinLow;
   INT64 BinHigh;
   INT64 BinNum;
   void* pUserMemory;
};


struct OLD_CALLBACK {
   FUNCTION_LIST*    FunctionList;

   // Args to call
   union {
      VOID*             pDataInBase1;
      VOID*             pValues;
   };

   union {
      VOID*             pDataInBase2;
      VOID*             pIndex;
      VOID*             pToSort;
   };

   VOID*             pDataInBase3;

   //-------------------------------------------------
   union {
      VOID*             pDataOutBase1;
      VOID*             pWorkSpace;
   };

   // Total number of array elements
   union {
      //INT64             TotalElements;
      INT64             IndexSize;

      // strlen is for sorting strings
      INT64             StrLen;
   };

   union {
      INT32             ScalarMode;
      INT32             MergeBlocks;
   };

   union {
      INT64             TotalElements2;
      INT64             ValSize;
   };

   // Default value to fill
   void*             pDefault;


   void*             pThreadWorkSpace;


};

//-----------------------------------------------------------
//
struct stMATH_WORKER_ITEM {

   // -----------------------------------
   // Tested with addition
   // %timeit global a; a+= 5

   // a=arange(100_000)
   // operation 0x4000  0x8000   0x10000   0x20000
   // -------   ------  -------  -------   -------
   // a+=5        51      52 us     27 us   27
   // a+=5 nowait 49
   // a+b         49      50 us     46 us   46
   //sqrt:        83     104 us    209 us   209
   // sum:        54                26 us   26
   //
   // arange(1_000_000)
   // operation 0x4000  0x8000   0x10000   0x20000 
   // -------   ------  -------  -------   -------
   // a+=5        114     120 us    118 us  133
   // a+b          91     121 us    128 us   46
   //sqrt:        252     293 us    293 us  209
   // sum:         50      51        52 us   68

   // a=arange(100_000.0)
   // operation 0x4000  0x8000   0x10000   0x20000
   // -------   ------  -------  -------   -------
   // a+b         69      xx      137 us    xx
   // sqrt:       85              209
   // sum:        52      xx      30 us     xx

   // Items larger than this might be worked on in parallel
   static const INT64 WORK_ITEM_CHUNK = 0x4000;
   static const INT64 WORK_ITEM_BIG = (WORK_ITEM_CHUNK * 2);
   static const INT64 WORK_ITEM_MASK = (WORK_ITEM_CHUNK - 1);

   // The callback to the thread routine that does work
   // with the argument to pass
   DOWORK_CALLBACK   DoWorkCallback;
   void*             WorkCallbackArg;

   // How many threads to wake up (atomic decrement)
   INT64             ThreadWakeup;

   // Used when calling MultiThreadedWork
   union {
      MTWORK_CALLBACK   MTWorkCallback;
      MTCHUNK_CALLBACK  MTChunkCallback;
   };

   // TotalElements is used on asymmetric last block
   INT64             TotalElements;

   // How many elements per block to work on
   INT64             BlockSize;


   // The last block to work on
   volatile INT64    BlockLast;

   //-------------------------------------------------
   // The next block (atomic)
   // Incremented
   // If BlockNext > BlockLast -- no work to be done
   volatile INT64    BlockNext;


   //-----------------------------------------------
   // Atomic access
   // When BlocksCompleted == BlockLast , the job is completed
   INT64             BlocksCompleted;


   // Get rid of this
   OLD_CALLBACK      OldCallback;

   //==============================================================
   FORCE_INLINE INT64 GetWorkBlock() {
      INT64 val=  InterlockedIncrement64(&BlockNext);
      return val - 1;
   }

   //==============================================================
   FORCE_INLINE void CompleteWorkBlock() {
      // Indicate we completed a block
      InterlockedIncrement64(&BlocksCompleted);
   }

   //=============================================================
   // Called by routines that work by index 
   // returns 0 on failure
   // else returns length of workblock
   FORCE_INLINE INT64 GetNextWorkIndex(INT64* workBlock) {
      INT64 wBlock = *workBlock = GetWorkBlock();

      //printf("working on block %llu\n", wBlock);

      // Make sure something to work on
      if (wBlock < BlockLast) {
         return wBlock;
      }

      return 0;
   }

   //=============================================================
   // Called by routines that work on chunks/blocks of memory
   // returns 0 on failure
   // else returns length of workblock
   FORCE_INLINE INT64 GetNextWorkBlock(INT64* workBlock) {

      INT64 wBlock = *workBlock = GetWorkBlock();

      //printf("working on block %llu\n", wBlock);

      // Make sure something to work on
      if (wBlock < BlockLast) {
         INT64  lenWorkBlock ;
         lenWorkBlock = BlockSize;

         // Check if this is the last workblock
         if ((wBlock + 1) == BlockLast) {
            
            // check if ends on perfect boundary
            if ((TotalElements & WORK_ITEM_MASK) != 0) {

               // This is the last block and may have an odd number of data to process
               lenWorkBlock = TotalElements & WORK_ITEM_MASK;
               //printf("last workblock %llu  %llu  MASK  %llu\n", lenWorkBlock, TotalElements, WORK_ITEM_MASK);
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
   FORCE_INLINE BOOL DoWork(int core, INT64 workIndex) {

      return DoWorkCallback(this, core, workIndex);
   }

};


//-----------------------------------------------------------
// allocated on 64 byte alignment
struct stWorkerRing {
   static const INT64   RING_BUFFER_SIZE = 1024;
   static const INT64   RING_BUFFER_MASK = 1023;

   volatile INT64       WorkIndex ;
   volatile INT64       WorkIndexCompleted ;

   // incremented when worker thread start
   volatile INT64       WorkThread ;
   INT32                Reserved32;
   INT32                SleepTime ;

   INT32                NumaNode;
   INT32                Cancelled;

   // Change this value to wake up less workers
   INT32                FutexWakeCount;

   stMATH_WORKER_ITEM   WorkerQueue[RING_BUFFER_SIZE];

   void Init() {
      WorkIndex = 0;
      WorkIndexCompleted = 0;
      WorkThread = 0;
      NumaNode = 0;
      Cancelled = 0;
      SleepTime = 1;
      // how many threads to wake up on Linux
      FutexWakeCount = FUTEX_WAKE_DEFAULT;
   }

   FORCE_INLINE void Cancel() {
      Cancelled = 1;
   }

   FORCE_INLINE stMATH_WORKER_ITEM* GetWorkItem() {
      return  &WorkerQueue[WorkIndex & RING_BUFFER_MASK];
   }

   FORCE_INLINE stMATH_WORKER_ITEM* GetExistingWorkItem() {
      return  &WorkerQueue[(WorkIndex - 1) & RING_BUFFER_MASK];
   }

   FORCE_INLINE void SetWorkItem(INT32 maxThreadsToWake) {
      // This routine will wakup threads on Windows and Linux
      // Once we increment other threads will notice
      InterlockedIncrement64(&WorkIndex);

#if defined(_WIN32)
      // Are we allowed to wake threads?
      if (g_WakeAllAddress != NULL) {

         if (maxThreadsToWake < 5) {
            // In windows faster to wake single if just a few threads
            for (int i = 0; i < maxThreadsToWake; i++) {
               g_WakeSingleAddress((PVOID)&WorkIndex);
            }
         }
         else {
            // In windows the more threads we wake up, the longer it takes to return from this OS call
            g_WakeAllAddress((PVOID)&WorkIndex);
         }
      }

#elif defined(__linux__)
      // Linux thread wakeup
      int s = futex((int*)&WorkIndex, FUTEX_WAKE, maxThreadsToWake, NULL, NULL, 0);
      if (s == -1)
         printf("***error futex-FUTEX_WAKE\n");     // TODO: Change to use fprintf(stderr, msg) instead
         
#elif defined(__APPLE__)
// temp remove warning
//#warning MathThreads does not yet support Darwin/macOS.
      pthread_cond_broadcast(&g_WakeupCond);
#else
#error riptide MathThreads support not implemented for this platform.

#endif

   }

   FORCE_INLINE void CompleteWorkItem() {
      InterlockedIncrement64(&WorkIndexCompleted);
   }
};

WakeSingleAddress InitWakeCalls();
//DWORD WINAPI WorkerThreadFunction(LPVOID lpParam);



