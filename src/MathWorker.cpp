#include "MathWorker.h"
#include "platform_detect.h"

/*
MathWorker

This file includes the interface and implementation of the CMathWorker class,
which provides the threading + work dispatch mechanism used by the rest of
riptide.

Implementation notes
====================
Darwin / macOS (OSX) / XNU kernel
---------------------------------
* XNU doesn't support ``pthread_getaffinity_np()``,
``pthread_setaffinity_np()``, ``cpu_set_t`` as supported by Linux and FreeBSD.
  In fact, it explicitly does not provide the ability to bind threads to
specific processor cores. However, it does provide a thread affinity API which
can be used to provide affinity *hints* for application threads to the OS
scheduler.
  * https://developer.apple.com/library/archive/releasenotes/Performance/RN-AffinityAPI/#//apple_ref/doc/uid/TP40006635-CH1-DontLinkElementID_2
  * https://yyshen.github.io/2015/01/18/binding_threads_to_cores_osx.html

* macOS libpthread implementation; useful to see which threading APIs macOS
supports -- particularly any _np (non-POSIX) APIs:
  https://github.com/apple/darwin-libpthread/blob/master/src/pthread.c

* Details on getting the current thread id in macOS / OSX:
  https://elliotth.blogspot.com/2012/04/gettid-on-mac-os.html

TODO
====
* Consider linking in the hwloc library here and using it to detect system
topology and set thread affinity. It's portable, available through the libhwloc
conda package, and already handles more-complex cases around NUMA and systems
with >64 logical processors. For example, we'd use the
``hwloc_set_proc_cpubind()`` and ``hwloc_set_thread_cpubind()`` functions from
the library, as documented here:
  * https://www.open-mpi.org/projects/hwloc/doc/v2.2.0/a00089_source.php
  * https://www.open-mpi.org/projects/hwloc/doc/v2.2.0/a00151.php#ga296db8a3c6d49b51fb83d6f3e45c02a6

  conda package: https://anaconda.org/conda-forge/libhwloc
*/

#if defined(__GNUC__)
    #define MEM_STATIC static __inline __attribute__((unused))
#elif defined(__cplusplus) || (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */)
    #define MEM_STATIC static inline
#elif defined(_MSC_VER)
    #define MEM_STATIC static __inline
#else
    #define MEM_STATIC \
        static /* this version may generate warnings for unused static functions; \
                  disable the relevant warning */
#endif

typedef struct
{
    uint32_t f1c;
    uint32_t f1d;
    uint32_t f7b;
    uint32_t f7c;
} ZSTD_cpuid_t;

MEM_STATIC ZSTD_cpuid_t ZSTD_cpuid(void)
{
    uint32_t f1c = 0;
    uint32_t f1d = 0;
    uint32_t f7b = 0;
    uint32_t f7c = 0;
#ifdef _MSC_VER
    int reg[4];
    __cpuid((int *)reg, 0);
    {
        int const n = reg[0];
        if (n >= 1)
        {
            __cpuid((int *)reg, 1);
            f1c = (uint32_t)reg[2];
            f1d = (uint32_t)reg[3];
        }
        if (n >= 7)
        {
            __cpuidex((int *)reg, 7, 0);
            f7b = (uint32_t)reg[1];
            f7c = (uint32_t)reg[2];
        }
    }
#elif defined(__i386__) && defined(__PIC__) && ! defined(__clang__) && defined(__GNUC__)
    /* The following block like the normal cpuid branch below, but gcc
     * reserves ebx for use of its pic register so we must specially
     * handle the save and restore to avoid clobbering the register
     */
    uint32_t n;
    __asm__(
        "pushl %%ebx\n\t"
        "cpuid\n\t"
        "popl %%ebx\n\t"
        : "=a"(n)
        : "a"(0)
        : "ecx", "edx");
    if (n >= 1)
    {
        uint32_t f1a;
        __asm__(
            "pushl %%ebx\n\t"
            "cpuid\n\t"
            "popl %%ebx\n\t"
            : "=a"(f1a), "=c"(f1c), "=d"(f1d)
            : "a"(1));
    }
    if (n >= 7)
    {
        __asm__(
            "pushl %%ebx\n\t"
            "cpuid\n\t"
            "movl %%ebx, %%eax\n\r"
            "popl %%ebx"
            : "=a"(f7b), "=c"(f7c)
            : "a"(7), "c"(0)
            : "edx");
    }
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    uint32_t n;
    __asm__("cpuid" : "=a"(n) : "a"(0) : "ebx", "ecx", "edx");
    if (n >= 1)
    {
        uint32_t f1a;
        __asm__("cpuid" : "=a"(f1a), "=c"(f1c), "=d"(f1d) : "a"(1) : "ebx");
    }
    if (n >= 7)
    {
        uint32_t f7a;
        __asm__("cpuid" : "=a"(f7a), "=b"(f7b), "=c"(f7c) : "a"(7), "c"(0) : "edx");
    }
#endif
    {
        ZSTD_cpuid_t cpuid;
        cpuid.f1c = f1c;
        cpuid.f1d = f1d;
        cpuid.f7b = f7b;
        cpuid.f7c = f7c;
        return cpuid;
    }
}

#define X(name, r, bit) \
    MEM_STATIC int ZSTD_cpuid_##name(ZSTD_cpuid_t const cpuid) \
    { \
        return ((cpuid.r) & (1U << bit)) != 0; \
    }

/* cpuid(1): Processor Info and Feature Bits. */
#define C(name, bit) X(name, f1c, bit)
C(sse3, 0)
C(pclmuldq, 1)
C(dtes64, 2)
C(monitor, 3)
C(dscpl, 4)
C(vmx, 5)
C(smx, 6)
C(eist, 7)
C(tm2, 8)
C(ssse3, 9)
C(cnxtid, 10)
C(fma, 12)
C(cx16, 13)
C(xtpr, 14)
C(pdcm, 15)
C(pcid, 17)
C(dca, 18)
C(sse41, 19)
C(sse42, 20)
C(x2apic, 21)
C(movbe, 22)
C(popcnt, 23)
C(tscdeadline, 24)
C(aes, 25)
C(xsave, 26)
C(osxsave, 27)
C(avx, 28)
C(f16c, 29)
C(rdrand, 30)
#undef C
#define D(name, bit) X(name, f1d, bit)
D(fpu, 0)
D(vme, 1)
D(de, 2)
D(pse, 3)
D(tsc, 4)
D(msr, 5)
D(pae, 6)
D(mce, 7)
D(cx8, 8)
D(apic, 9)
D(sep, 11)
D(mtrr, 12)
D(pge, 13)
D(mca, 14)
D(cmov, 15)
D(pat, 16)
D(pse36, 17)
D(psn, 18)
D(clfsh, 19)
D(ds, 21)
D(acpi, 22)
D(mmx, 23)
D(fxsr, 24)
D(sse, 25)
D(sse2, 26)
D(ss, 27)
D(htt, 28)
D(tm, 29)
D(pbe, 31)
#undef D

/* cpuid(7): Extended Features. */
#define B(name, bit) X(name, f7b, bit)
B(bmi1, 3)
B(hle, 4)
B(avx2, 5)
B(smep, 7)
B(bmi2, 8)
B(erms, 9)
B(invpcid, 10)
B(rtm, 11)
B(mpx, 14)
B(avx512f, 16)
B(avx512dq, 17)
B(rdseed, 18)
B(adx, 19)
B(smap, 20)
B(avx512ifma, 21)
B(pcommit, 22)
B(clflushopt, 23)
B(clwb, 24)
B(avx512pf, 26)
B(avx512er, 27)
B(avx512cd, 28)
B(sha, 29)
B(avx512bw, 30)
B(avx512vl, 31)
#undef B
#define C(name, bit) X(name, f7c, bit)
C(prefetchwt1, 0)
C(avx512vbmi, 1)
#undef C

#undef X

int g_bmi2 = 0;
int g_avx2 = 0;

#if defined(RT_OS_WINDOWS)

void PrintCPUInfo()
{
    int CPUInfo[4] = { -1 };
    unsigned nExIds, i = 0;
    char CPUBrandString[0x40];
    // Get the information associated with each extended ID.
    __cpuid(CPUInfo, 0x80000000);
    nExIds = CPUInfo[0];
    for (i = 0x80000000; i <= nExIds; ++i)
    {
        __cpuid(CPUInfo, i);
        // Interpret CPU brand string
        if (i == 0x80000002)
            memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }
    // NEW CODE
    g_bmi2 = ZSTD_cpuid_bmi2(ZSTD_cpuid());
    g_avx2 = ZSTD_cpuid_avx2(ZSTD_cpuid());

    // printf("**CPU: %s  AVX2:%d  BMI2:%d\n", CPUBrandString, g_avx2, g_bmi2);
    if (g_bmi2 == 0 || g_avx2 == 0)
    {
        printf(
            "!!!NOTE: this system does not support AVX2 or BMI2 instructions, "
            "and will not work!\n");
    }
}

#else
extern "C"
{
    #include <pthread.h>
    #include <sched.h>
    #include <sys/types.h>

    #include <sys/syscall.h>
    #include <unistd.h>

    #ifdef RT_OS_FREEBSD
        #include <sys/thr.h> // Use thr_self() syscall under FreeBSD to get thread id
    #endif                   // RT_OS_FREEBSD

    pid_t gettid(void)
    {
    #if defined(RT_OS_LINUX)
        return syscall(SYS_gettid);

    #elif defined(RT_OS_DARWIN)
        uint64_t thread_id;
        return pthread_threadid_np(NULL, &thread_id) ? 0 : (pid_t)thread_id;

    #elif defined(RT_OS_FREEBSD)
        // https://www.freebsd.org/cgi/man.cgi?query=thr_self
        long thread_id;
        return thr_self(&thread_id) ? 0 : (pid_t)thread_id;

    #else
        #error Cannot determine how to get the identifier for the current thread on this platform.
    #endif // defined(RT_OS_LINUX)
    }

    void Sleep(unsigned int dwMilliseconds)
    {
        usleep(dwMilliseconds * 1000);
    }

    bool CloseHandle(THANDLE hObject)
    {
        return true;
    }

    pid_t GetCurrentThread()
    {
        return gettid();
    }

    uint64_t SetThreadAffinityMask(pid_t hThread, uint64_t dwThreadAffinityMask)
    {
    #if defined(RT_OS_LINUX) || defined(RT_OS_FREEBSD)
        cpu_set_t cpuset;

        uint64_t bitpos = 1;
        int count = 0;

        while (! (bitpos & dwThreadAffinityMask))
        {
            bitpos <<= 1;
            count++;
            if (count > 63)
            {
                break;
            }
        }

        // printf("**linux setting affinity %d\n", count);

        if (count <= 63)
        {
            CPU_ZERO(&cpuset);
            CPU_SET(count, &cpuset);
            // dwThreadAffinityMask
            sched_setaffinity(GetCurrentThread(), sizeof(cpuset), &cpuset);
        }

    #else
        #warning No thread-affinity support implemented for this OS. This does not prevent riptide from running but overall performance may be reduced.
    #endif // defined(RT_OS_LINUX) || defined(RT_OS_FREEBSD)

        return 0;
    }

    bool GetProcessAffinityMask(HANDLE hProcess, uint64_t * lpProcessAffinityMask, uint64_t * lpSystemAffinityMask)
    {
    #if defined(RT_OS_LINUX) || defined(RT_OS_FREEBSD)
        cpu_set_t cpuset;
        sched_getaffinity(getpid(), sizeof(cpuset), &cpuset);

        *lpProcessAffinityMask = 0;
        *lpSystemAffinityMask = 0;

        uint64_t bitpos = 1;
        for (int i = 0; i < 63; i++)
        {
            if (CPU_ISSET(i, &cpuset))
            {
                *lpProcessAffinityMask |= bitpos;
                *lpSystemAffinityMask |= bitpos;
            }
            bitpos <<= 1;
        }

        if (*lpProcessAffinityMask == 0)
        {
            *lpSystemAffinityMask = 0xFF;
            *lpSystemAffinityMask = 0xFF;
        }

        // CPU_ISSET = 0xFF;
        return true;

    #else
        #warning No thread-affinity support implemented for this OS. This does not prevent riptide from running but overall performance may be reduced.
        return false;

    #endif // defined(RT_OS_LINUX) || defined(RT_OS_FREEBSD)
    }

    HANDLE GetCurrentProcess()
    {
        return NULL;
    }

    unsigned int GetLastError()
    {
        return 0;
    }

    HANDLE CreateThread(void * lpThreadAttributes, size_t dwStackSize, LPTHREAD_START_ROUTINE lpStartAddress, void * lpParameter,
                        unsigned int dwCreationFlags, uint32_t * lpThreadId)
    {
        return NULL;
    }

    HMODULE LoadLibraryW(const wchar_t * lpLibFileName)
    {
        return NULL;
    }

    FARPROC GetProcAddress(HMODULE hModule, const char * lpProcName)
    {
        return NULL;
    }
}

    #include <cpuid.h>

void PrintCPUInfo()
{
    char CPUBrandString[0x40];
    unsigned int CPUInfo[4] = { 0, 0, 0, 0 };

    __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    unsigned int nExIds = CPUInfo[0];

    memset(CPUBrandString, 0, sizeof(CPUBrandString));

    for (unsigned int i = 0x80000000; i <= nExIds; ++i)
    {
        __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);

        if (i == 0x80000002)
            memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }
    // printf("**CPU: %s\n", CPUBrandString);

    // NEW CODE
    g_bmi2 = ZSTD_cpuid_bmi2(ZSTD_cpuid());
    g_avx2 = ZSTD_cpuid_avx2(ZSTD_cpuid());

    // printf("**CPU: %s  AVX2:%d  BMI2:%d\n", CPUBrandString, g_avx2, g_bmi2);
    if (g_bmi2 == 0 || g_avx2 == 0)
    {
        printf(
            "!!!NOTE: this system does not support AVX2 or BMI2 instructions, "
            "and will not work!\n");
    }
}

#endif

int GetProcCount()
{
    HANDLE proc = GetCurrentProcess();

    uint64_t mask1;
    uint64_t mask2;
    int32_t count;

    count = 0;
    GetProcessAffinityMask(proc, &mask1, &mask2);

    while (mask1 != 0)
    {
        if (mask1 & 1)
            count++;
        mask1 = mask1 >> 1;
    }

    // printf("**Process count: %d   riptide_cpp build date and time: %s %s\n",
    // count, __DATE__, __TIME__);

    if (count == 0)
        count = MAX_THREADS_WHEN_CANNOT_DETECT;

    if (count > MAX_THREADS_ALLOWED)
        count = MAX_THREADS_ALLOWED;

    return count;
}
