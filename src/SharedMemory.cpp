#include "SharedMemory.h"
#include "CommonInc.h"
#include "platform_detect.h"

#if defined(RT_OS_WINDOWS) && (defined(RT_COMPILER_MSVC) || defined(RT_COMPILER_CLANG))
    #pragma comment(lib, "advapi32.lib")
#else
    #include <unistd.h>
#endif

#define LOGGING(...)
//#define LOGGING printf

#if defined(RT_OS_WINDOWS)

extern CHAR g_errmsg[512];

const CHAR * GetLastErrorMessage(CHAR * errmsg, DWORD last_error);

void LogWarningLE(HRESULT error)
{
    printf("%s\n", GetLastErrorMessage(g_errmsg, error));
}

//------------------------------------------------------------------------------------------
// Checks Windows policy settings if the current process has
// the privilege enabled.
//
bool CheckWindowsPrivilege(const char * pPrivilegeName)
{
    LUID luid;
    PRIVILEGE_SET privilegeSet;
    HANDLE hCurrentProccess;
    HANDLE hProcessToken = NULL;
    HRESULT hResult;
    BOOL bResult;

    hResult = S_OK;

    // This cannot fail, a pseudo handle that does not need to be closed
    hCurrentProccess = GetCurrentProcess();

    if (! OpenProcessToken(hCurrentProccess, TOKEN_QUERY, &hProcessToken))
    {
        auto e = GetLastError();
        printf("OpenProcessToken error %u\n", e);
        hResult = HRESULT_FROM_WIN32(e);
        LogWarningLE(hResult);
        return false;
    }

    if (! LookupPrivilegeValue(NULL, pPrivilegeName, &luid))
    {
        CloseHandle(hProcessToken);

        auto e = GetLastError();
        printf("LookupPrivilegeValue error %u\n", e);
        hResult = HRESULT_FROM_WIN32(e);
        LogWarningLE(hResult);
        return false;
    }

    privilegeSet.PrivilegeCount = 1;
    privilegeSet.Control = PRIVILEGE_SET_ALL_NECESSARY;
    privilegeSet.Privilege[0].Luid = luid;
    privilegeSet.Privilege[0].Attributes = SE_PRIVILEGE_ENABLED;

    // Do we want to check return code?
    PrivilegeCheck(hProcessToken, &privilegeSet, &bResult);

    CloseHandle(hProcessToken);

    return bResult;
}

//------------------------------------------------------------------------------------------
bool SetPrivilege(HANDLE hToken,         // access token handle
                  LPCTSTR lpszPrivilege, // name of privilege to enable/disable
                  bool bEnablePrivilege  // to enable or disable privilege
)
{
    TOKEN_PRIVILEGES tp;
    LUID luid;

    if (! LookupPrivilegeValue(NULL,          // lookup privilege on local system
                               lpszPrivilege, // privilege to lookup
                               &luid))        // receives LUID of privilege
    {
        printf("LookupPrivilegeValue error: %u\n", GetLastError());
        return false;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    if (bEnablePrivilege)
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    else
        tp.Privileges[0].Attributes = 0;

    // Enable the privilege or disable all privileges.

    if (! AdjustTokenPrivileges(hToken, false, &tp, sizeof(TOKEN_PRIVILEGES), (PTOKEN_PRIVILEGES)NULL, (PDWORD)NULL))
    {
        printf("AdjustTokenPrivileges error: %u\n", GetLastError());
        return false;
    }

    if (GetLastError() == ERROR_NOT_ALL_ASSIGNED)

    {
        printf("The token does not have the specified privilege. \n");
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------------------
//
bool CheckWindowsSharedMemoryPrerequisites(const char * pMappingName)
{
    const char * pPrivilegeName = "SeCreateGlobalPrivilege";
    if (strstr(pMappingName, "Global") && ! CheckWindowsPrivilege(pPrivilegeName))
    {
        printf(
            "CheckWindowsSharedMemoryPrerequisites: privilege %s needs to be "
            "enabled for read / write to global file %s\n",
            pPrivilegeName, pMappingName);
        HANDLE currentToken = NULL;

        // More work to do here
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ALL_ACCESS, &currentToken))
        {
            bool newResult = SetPrivilege(currentToken, pPrivilegeName, true);
            CloseHandle(currentToken);
            return newResult;
        }
        return false;
    }
    return true;
}

//------------------------------------------------------------------------------------------
//
// Allocates shared memory
// if the name starts with Global\ it will be a global name
// Simply memory maps a file and returns the struct holding
// some handles to the memory mapping.
//
HRESULT
UtilSharedMemoryBegin(const char * pMappingName, INT64 Size, PMAPPED_VIEW_STRUCT * pReturnStruct)
{
    PMAPPED_VIEW_STRUCT pMappedViewStruct;
    HRESULT hResult = S_OK;

    //
    // NULL indicates failure - default to that.
    //
    *pReturnStruct = NULL;

    if (! CheckWindowsSharedMemoryPrerequisites(pMappingName))
    {
        hResult = S_FALSE;
        return -(hResult);
    }

    //
    // Allocate fixed, zero inited memory
    //
    pMappedViewStruct = static_cast<PMAPPED_VIEW_STRUCT>(WORKSPACE_ALLOC(sizeof(MAPPED_VIEW_STRUCT)));

    if (pMappedViewStruct == NULL)
    {
        hResult = E_OUTOFMEMORY;
        LogWarningLE(hResult);
        return (hResult);
    }

    DWORD HiWord = (DWORD)((UINT64)Size >> 32);
    DWORD LowWord = (DWORD)((UINT64)Size & 0xFFFFFFFF);
    //
    // We create a file mapping in order to share the memory
    //
    pMappedViewStruct->MapHandle = CreateFileMapping(INVALID_HANDLE_VALUE,
                                                     NULL, // default security
                                                     PAGE_READWRITE, HiWord, LowWord, pMappingName);

    //
    // Check for errors in mapping
    //
    if (pMappedViewStruct->MapHandle == NULL)
    {
        WORKSPACE_FREE(pMappedViewStruct);
        hResult = HRESULT_FROM_WIN32(GetLastError());
        LogWarningLE(hResult);
        return (hResult);
    }

    pMappedViewStruct->BaseAddress =
        static_cast<PVOID>(MapViewOfFile(pMappedViewStruct->MapHandle, FILE_MAP_ALL_ACCESS, 0, 0, Size));

    pMappedViewStruct->pSharedMemoryHeader = pMappedViewStruct->BaseAddress;

    //
    // Check for errors again
    //
    if (pMappedViewStruct->BaseAddress == NULL)
    {
        CloseHandle(pMappedViewStruct->MapHandle);
        WORKSPACE_FREE(pMappedViewStruct);
        hResult = HRESULT_FROM_WIN32(GetLastError());
        LogWarningLE(hResult);
        return (hResult);
    }

    // Success at this point
    pMappedViewStruct->FileSize = pMappedViewStruct->RealFileSize = Size;

    //// The size of the memory is the first 8 bytes
    // SHARED_MEMORY_HEADER* pMemory =
    // (SHARED_MEMORY_HEADER*)(pMappedViewStruct->pSharedMemoryHeader);
    // pMemory->Initialize(Size);
    // LogInform("Creating memory with size %llu -- base address at %p\n",
    // pMemory->MappingSize, pMemory);

    //
    // Set return value to success if we made it this far
    // Also point to structure
    //
    *pReturnStruct = pMappedViewStruct;

    return (hResult);
}

//------------------------------------------------------------------------------------------
//
// Generic file/memory mapping utility.
// Simply memory maps a file and returns the struct holding
// some handles to the memory mapping.
//
HRESULT
UtilSharedNumaMemoryBegin(const char * pMappingName, INT64 Size,
                          DWORD nndPreferred, // preferred numa node
                          LPVOID lpBaseAddress, PMAPPED_VIEW_STRUCT * pReturnStruct)
{
    PMAPPED_VIEW_STRUCT pMappedViewStruct;
    HRESULT hResult = S_OK;

    //
    // NULL indicates failure - default to that.
    //
    *pReturnStruct = NULL;

    if (! CheckWindowsSharedMemoryPrerequisites(pMappingName))
    {
        hResult = S_FALSE;
        return -(hResult);
    }

    //
    // Allocate fixed, zero inited memory
    //
    pMappedViewStruct = static_cast<PMAPPED_VIEW_STRUCT>(WORKSPACE_ALLOC(sizeof(MAPPED_VIEW_STRUCT)));

    if (pMappedViewStruct == NULL)
    {
        hResult = E_OUTOFMEMORY;
        LogWarningLE(hResult);
        return (hResult);
    }

    DWORD HiWord = (DWORD)((UINT64)Size >> 32);
    DWORD LowWord = (DWORD)((UINT64)Size & 0xFFFFFFFF);
    //
    // We create a file mapping inorder to share the memory
    //
    pMappedViewStruct->MapHandle =
        CreateFileMappingNuma(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, HiWord, LowWord, pMappingName, nndPreferred);

    //
    // Check for errors in mapping
    //
    if (pMappedViewStruct->MapHandle == NULL)
    {
        WORKSPACE_FREE(pMappedViewStruct);
        hResult = HRESULT_FROM_WIN32(GetLastError());
        LogWarningLE(hResult);
        return (hResult);
    }

    // Let the OS pick the base address for us
    pMappedViewStruct->BaseAddress = static_cast<PVOID>(
        MapViewOfFileExNuma(pMappedViewStruct->MapHandle, FILE_MAP_ALL_ACCESS, 0, 0, Size, lpBaseAddress, nndPreferred));

    pMappedViewStruct->pSharedMemoryHeader = pMappedViewStruct->BaseAddress;

    //
    // Check for errors again
    //
    if (pMappedViewStruct->BaseAddress == NULL)
    {
        CloseHandle(pMappedViewStruct->MapHandle);
        WORKSPACE_FREE(pMappedViewStruct);
        hResult = HRESULT_FROM_WIN32(GetLastError());
        LogWarningLE(hResult);
        return (hResult);
    }

    // Success at this point
    pMappedViewStruct->FileSize = pMappedViewStruct->RealFileSize = Size;

    //
    // Set return value to success if we made it this far
    // Also point to structure
    //
    *pReturnStruct = pMappedViewStruct;

    return (hResult);
}

//------------------------------------------------------------------------------------------
// The pReturnStruct will be valid if the call succeeded
// if bTest = true, it will not complain if it cannot find shared memory
// if returns S_OK you are mapped
// returns S_FALSE if it does not exist yet
HRESULT
UtilSharedMemoryCopy(const char * pMappingName, PMAPPED_VIEW_STRUCT * pReturnStruct, BOOL bTest)
{
    HRESULT hResult;
    PMAPPED_VIEW_STRUCT pMappedViewStruct;

    //
    // NULL indicates failure - default to that.
    //
    *pReturnStruct = NULL;

    //
    // Allocate fixed, zero inited memory
    //
    pMappedViewStruct = static_cast<PMAPPED_VIEW_STRUCT>(WORKSPACE_ALLOC(sizeof(MAPPED_VIEW_STRUCT)));

    if (pMappedViewStruct == NULL)
    {
        LogWarningLE(E_OUTOFMEMORY);
        return (E_OUTOFMEMORY);
    }

    //
    // We create a file mapping inorder to map the file
    //
    pMappedViewStruct->MapHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, pMappingName);

    // printf("Result from OpenFileMaApping %s -- %p\n", pMappingName,
    // pMappedViewStruct->MapHandle);

    //
    // Check for errors in mapping
    //
    if (pMappedViewStruct->MapHandle == NULL)
    {
        WORKSPACE_FREE(pMappedViewStruct);
        hResult = HRESULT_FROM_WIN32(GetLastError());
        return (hResult);
    }

    // Do we want same address?
    pMappedViewStruct->BaseAddress = static_cast<PVOID>(MapViewOfFile(pMappedViewStruct->MapHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0));

    //
    // Check for errors again
    //
    if (pMappedViewStruct->BaseAddress == NULL)
    {
        CloseHandle(pMappedViewStruct->MapHandle);
        WORKSPACE_FREE(pMappedViewStruct);
        HRESULT hResult = HRESULT_FROM_WIN32(GetLastError());
        LogWarningLE(hResult);
        return (hResult);
    }

    //
    // Set return value to success if we made it this far
    // Also point to structure
    //
    *pReturnStruct = pMappedViewStruct;

    return (S_OK);
}

//------------------------------------------------------------------------------------------
//
HRESULT
UtilSharedMemoryEnd(PMAPPED_VIEW_STRUCT pMappedViewStruct)
{
    HRESULT hResult;

    if (pMappedViewStruct == NULL)
    {
        return (E_POINTER);
    }

    if (UnmapViewOfFile(pMappedViewStruct->BaseAddress) == false)
    {
        hResult = HRESULT_FROM_WIN32(GetLastError());
        return (hResult);
    }

    if (CloseHandle(pMappedViewStruct->MapHandle) == false)
    {
        hResult = HRESULT_FROM_WIN32(GetLastError());
        return (hResult);
    }

    WORKSPACE_FREE(pMappedViewStruct);
    return (S_OK);
}

//------------------------------------------------------------------------------------------
//
// Generic file mapping utility.
// Simply memory maps a file and returns the struct holding
// some handles to the memory mapping.
//
HRESULT
UtilMappedViewReadBegin(const char * pszFilename, PMAPPED_VIEW_STRUCT * pReturnStruct)
{
    PULONG pulFile;
    PMAPPED_VIEW_STRUCT pMappedViewStruct;
    // TODO: const char*              pMappingName;

    //
    // NULL indicates failure - default to that.
    //
    *pReturnStruct = NULL;

    // TODO: if (!CheckWindowsSharedMemoryPrerequisites(pMappingName))
    // TODO: {
    // TODO:    return -(S_FALSE);
    // TODO: }

    //
    // Allocate fixed, zero inited memory
    //
    pMappedViewStruct = static_cast<PMAPPED_VIEW_STRUCT>(WORKSPACE_ALLOC(sizeof(MAPPED_VIEW_STRUCT)));

    if (pMappedViewStruct == NULL)
    {
        return (E_OUTOFMEMORY);
    }

    //
    // Can we successfullly open the file?
    //
    pMappedViewStruct->FileHandle = CreateFile(pszFilename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);

    if (pMappedViewStruct->FileHandle == (void *)HFILE_ERROR)
    {
        //
        // The "AutoCAD has the file as well" bug
        //
        pMappedViewStruct->FileHandle = CreateFile(pszFilename, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    }

    if (pMappedViewStruct->FileHandle == (void *)HFILE_ERROR)
    {
        WORKSPACE_FREE(pMappedViewStruct);
        return (HRESULT_FROM_WIN32(GetLastError()));
    }

    // TODO:   //
    // TODO:   // Break off any \ or : or / in the file
    // TODO:   //
    // TODO:   // Search for the last one
    // TODO:   //
    // TODO:   {
    // TODO:      const char* pTemp = pszFilename;
    // TODO:      pMappingName = pTemp;
    // TODO:
    // TODO:      while (*pTemp != 0) {
    // TODO:
    // TODO:         if (*pTemp == '\\' ||
    // TODO:            *pTemp == ':' ||
    // TODO:            *pTemp == '/') {
    // TODO:
    // TODO:            pMappingName = pTemp + 1;
    // TODO:         }
    // TODO:
    // TODO:         pTemp++;
    // TODO:      }
    // TODO:   }

    //
    // We create a file mapping in order to map the file
    //
    pMappedViewStruct->MapHandle = CreateFileMapping((HANDLE)pMappedViewStruct->FileHandle, NULL, PAGE_READONLY, 0, 0,
                                                     0); // pMappingName);

    //
    // Check for errors in mapping
    //
    if (pMappedViewStruct->MapHandle == NULL)
    {
        CloseHandle(pMappedViewStruct->FileHandle);
        WORKSPACE_FREE(pMappedViewStruct);
        return (HRESULT_FROM_WIN32(GetLastError()));
    }

    pMappedViewStruct->BaseAddress = pulFile =
        static_cast<PULONG>(MapViewOfFile(pMappedViewStruct->MapHandle, FILE_MAP_READ, 0, 0, 0));

    //
    // Check for errors again
    //
    if (pulFile == NULL)
    {
        CloseHandle(pMappedViewStruct->MapHandle);
        CloseHandle(pMappedViewStruct->FileHandle);
        WORKSPACE_FREE(pMappedViewStruct);
        return (HRESULT_FROM_WIN32(GetLastError()));
    }

    //
    // Get File Size
    //
    pMappedViewStruct->FileSize = pMappedViewStruct->RealFileSize = GetFileSize(pMappedViewStruct->FileHandle, NULL);

    //
    // Set return value to success if we made it this far
    // Also point to structure
    //
    *pReturnStruct = pMappedViewStruct;

    return (S_OK);
}

//------------------------------------------------------------------------------------------
//
HRESULT
UtilMappedViewReadEnd(PMAPPED_VIEW_STRUCT pMappedViewStruct)
{
    HRESULT hResult;

    if (pMappedViewStruct == NULL)
    {
        return (E_POINTER);
    }

    if (UnmapViewOfFile(pMappedViewStruct->BaseAddress) == false)
    {
        hResult = HRESULT_FROM_WIN32(GetLastError());
        return (hResult);
    }

    if (CloseHandle(pMappedViewStruct->MapHandle) == false)
    {
        hResult = HRESULT_FROM_WIN32(GetLastError());
        return (hResult);
    }

    if (CloseHandle(pMappedViewStruct->FileHandle) == HFILE_ERROR)
    {
        hResult = HRESULT_FROM_WIN32(GetLastError());
        return (hResult);
    }

    WORKSPACE_FREE(pMappedViewStruct);

    return (S_OK);
}

HRESULT
UtilMappedViewWriteEnd(PMAPPED_VIEW_STRUCT pMappedViewStruct)
{
    return UtilMappedViewReadEnd(pMappedViewStruct);
}

//------------------------------------------------------------------------------------------
//
HRESULT
UtilMappedViewWriteBegin(const char * pszFilename, PMAPPED_VIEW_STRUCT * pReturnStruct, DWORD dwMaxSizeHigh, DWORD dwMaxSizeLow)
{
    PULONG pulFile;
    PMAPPED_VIEW_STRUCT pMappedViewStruct;
    // TODO: const char*              pMappingName;

    //
    // NULL indicates failure - default to that.
    //
    *pReturnStruct = NULL;

    // TODO: if (!CheckWindowsSharedMemoryPrerequisites(pMappingName))
    // TODO: {
    // TODO:    return -(S_FALSE);
    // TODO: }

    //
    // Allocate fixed, zero inited memory
    //
    pMappedViewStruct = static_cast<PMAPPED_VIEW_STRUCT>(WORKSPACE_ALLOC(sizeof(MAPPED_VIEW_STRUCT)));

    if (pMappedViewStruct == NULL)
    {
        return (E_OUTOFMEMORY);
    }

    //
    // Can we successfullly open the file?
    //
    pMappedViewStruct->FileHandle = CreateFile(pszFilename, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);

    if (pMappedViewStruct->FileHandle == (void *)HFILE_ERROR)
    {
        //
        // The "AutoCAD has the file as well" bug
        //
        pMappedViewStruct->FileHandle = CreateFile(pszFilename, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                                   NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    }

    if (pMappedViewStruct->FileHandle == (void *)HFILE_ERROR)
    {
        WORKSPACE_FREE(pMappedViewStruct);
        return (HRESULT_FROM_WIN32(GetLastError()));
    }

    // TODO:   //
    // TODO:   // Break off any \ or : or / in the file
    // TODO:   //
    // TODO:   // Search for the last one
    // TODO:   //
    // TODO:   {
    // TODO:      const char* pTemp = pszFilename;
    // TODO:      pMappingName = pTemp;
    // TODO:
    // TODO:      while (*pTemp != 0) {
    // TODO:
    // TODO:         if (*pTemp == '\\' ||
    // TODO:            *pTemp == ':' ||
    // TODO:            *pTemp == '/') {
    // TODO:
    // TODO:            pMappingName = pTemp + 1;
    // TODO:         }
    // TODO:
    // TODO:         pTemp++;
    // TODO:      }
    // TODO:   }

    //
    // We create a file mapping inorder to map the file
    //
    pMappedViewStruct->MapHandle =
        CreateFileMapping((HANDLE)pMappedViewStruct->FileHandle, NULL, PAGE_READWRITE, dwMaxSizeHigh, dwMaxSizeLow,
                          0); // pMappingName);

    //
    // Check for errors in mapping
    //
    if (pMappedViewStruct->MapHandle == NULL)
    {
        CloseHandle(pMappedViewStruct->FileHandle);
        WORKSPACE_FREE(pMappedViewStruct);
        return (HRESULT_FROM_WIN32(GetLastError()));
    }

    pMappedViewStruct->BaseAddress = pulFile =
        static_cast<PULONG>(MapViewOfFile(pMappedViewStruct->MapHandle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0));

    //
    // Check for errors again
    //
    if (pMappedViewStruct->BaseAddress == NULL)
    {
        CloseHandle(pMappedViewStruct->MapHandle);
        CloseHandle(pMappedViewStruct->FileHandle);
        WORKSPACE_FREE(pMappedViewStruct);
        return (HRESULT_FROM_WIN32(GetLastError()));
    }

    //
    // Get File Size
    //
    pMappedViewStruct->FileSize = pMappedViewStruct->RealFileSize = GetFileSize(pMappedViewStruct->FileHandle, NULL);

    //
    // Set return value to success if we made it this far
    // Also point to structure
    //
    *pReturnStruct = pMappedViewStruct;

    return (S_OK);
}

#else

    #include <fcntl.h> /* For O_* constants */
    #include <sys/mman.h>
    #include <sys/stat.h> /* For mode constants */

//-------------------------------------------------
// When the existing sharedname does not exist and must be created
HRESULT
UtilSharedMemoryBegin(const char * pMappingName, int64_t size, PMAPPED_VIEW_STRUCT * pReturnStruct)
{
    // NULL indicates failure - default to that.
    //
    *pReturnStruct = NULL;
    errno = 0;

    // no need for execute
    // for non ANONYMOUS, the write permissions will be automatically removed
    LOGGING("Trying to open and create shared memory:%s.\n", pMappingName);

    // TO FORCE A NAME pMappingName="/example.sds";
    // NOTE: We use O_EXCL because the file might already exist
    // Should we delete it?
    int fd = shm_open(pMappingName, O_RDWR | O_CREAT | O_EXCL, 0666);

    if (fd <= 0)
    {
        // Try to delete link and re-open
        shm_unlink(pMappingName);
        fd = shm_open(pMappingName, O_RDWR | O_CREAT | O_EXCL, 0666);
    }
    if (fd <= 0)
    {
        // todo use errno
        // this error can be expected if the shared memory does not exist yet
        printf("Error memory copy: %s\n", strerror(errno));
        return -1;
    }

    // Our memory buffer will be readable and writable:
    int protection = PROT_READ | PROT_WRITE;

    // MAP_HUGETLB(since Linux 2.6.32)
    //   Allocate the mapping using "huge pages."  See the Linux kernel
    //   source file Documentation / vm / hugetlbpage.txt for further
    //   information, as well as NOTES, below.
    //
    // MAP_PRIVATE
    //   Create a private copy - on - write mapping.Updates to the
    //   mapping are not visible to other processes mapping the same
    //   file, and are not carried through to the underlying file.It
    //   is unspecified whether changes made to the file after the
    //   mmap() call are visible in the mapped region.
    //
    // MAP_ANONYMOUS
    //   The mapping is not backed by any file; its contents are
    //   initialized to zero.The fd argument is ignored; however,
    //   some implementations require fd to be - 1 if MAP_ANONYMOUS(or
    //      MAP_ANON) is specified, and portable applications should
    //   ensure this.The offset argument should be zero.The use of
    //   MAP_ANONYMOUS in conjunction with MAP_SHARED is supported on
    //   Linux only since kernel 2.4

    // all can see
    int visibility = MAP_SHARED; // | MAP_ANONYMOUS;

    // Needed when non MAP_ANONYMOUS
    int result = ftruncate(fd, size);
    if (result && errno < 0)
    {
        printf("Error UtilSharedMemoryCopy ftruncate: %s\n", strerror(errno));
        return -1;
    }

    // TJD: Note if do not use MAP_ANONYMOUS believe have to use fallocate or
    // similar to actually reserve the space
    void * memaddress = mmap(0, size, protection, visibility, fd, 0);

    if (memaddress == (void *)-1)
    {
        printf("Error UtilSharedMemoryBegin mmap: %s\n", strerror(errno));
        return -1;
    }

    LOGGING("Linux: shared memory %s at %p  with size:%lld\n", pMappingName, memaddress, (long long)size);

    //
    // Allocate memory
    //
    PMAPPED_VIEW_STRUCT pMappedViewStruct = static_cast<PMAPPED_VIEW_STRUCT>(WORKSPACE_ALLOC(sizeof(MAPPED_VIEW_STRUCT)));

    // The remaining parameters to `mmap()` are not important for this use case,
    // but the manpage for `mmap` explains their purpose.
    pMappedViewStruct->FileHandle = fd;
    pMappedViewStruct->FileSize = size;
    pMappedViewStruct->BaseAddress = memaddress;

    *pReturnStruct = pMappedViewStruct;

    return 0;
}

//-------------------------------------------------
// When the existing sharedname already exists
HRESULT
UtilSharedMemoryCopy(const char * pMappingName, PMAPPED_VIEW_STRUCT * pReturnStruct, int bTest)
{
    // NULL indicates failure - default to that.
    *pReturnStruct = NULL;
    errno = 0;
    bool bCanOnlyRead = false;

    LOGGING("Trying to open shared memory %s.\n", pMappingName);
    int fd = shm_open(pMappingName, O_RDWR, 0666);

    if (fd <= 0)
    {
        bCanOnlyRead = true;
        fd = shm_open(pMappingName, O_RDONLY, 0666);
    }

    if (fd <= 0)
    {
        // todo use errno
        printf("UtilSharedMemoryCopy: %s. Error memory copy: %s\n", pMappingName, strerror(errno));
        return -1;
    }

    // Our memory buffer will be readable and writable:
    int protection = PROT_READ;

    if (! bCanOnlyRead)
    {
        protection |= PROT_WRITE;
    }

    // all can see
    int visibility = MAP_SHARED; // | MAP_ANONYMOUS;

    // Get the size of existing shared memory
    struct stat sb;
    fstat(fd, &sb);
    off_t size = sb.st_size;
    if (errno < 0)
    {
        printf("Error UtilSharedMemoryCopy fstat: %s\n", strerror(errno));
        return -1;
    }

    void * memaddress = mmap(NULL, size, protection, visibility, fd, 0);

    if (memaddress == (void *)-1)
    {
        printf("Error UtilSharedMemoryCopy mmap: %s  %lld\n", strerror(errno), (long long)size);
        return -1;
    }

    //
    // Allocate memory
    //
    PMAPPED_VIEW_STRUCT pMappedViewStruct;

    pMappedViewStruct = static_cast<PMAPPED_VIEW_STRUCT>(WORKSPACE_ALLOC(sizeof(MAPPED_VIEW_STRUCT)));

    // The remaining parameters to `mmap()` are not important for this use case,
    // but the manpage for `mmap` explains their purpose.
    pMappedViewStruct->FileHandle = fd;
    pMappedViewStruct->FileSize = size;
    pMappedViewStruct->BaseAddress = memaddress;

    *pReturnStruct = pMappedViewStruct;
    return 0;
}

HRESULT
UtilSharedMemoryEnd(PMAPPED_VIEW_STRUCT pMappedViewStruct)
{
    LOGGING("Closing linux shared memory\n");
    munmap(pMappedViewStruct->BaseAddress, pMappedViewStruct->FileSize);
    close(pMappedViewStruct->FileHandle);

    WORKSPACE_FREE(pMappedViewStruct);
    return 0;
}

#endif
