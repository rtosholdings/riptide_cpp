// dllmain.cpp : Defines the entry point for the DLL application.
#include <stdlib.h>
#include "../../src/CommonInc.h"
#include "../../src/MathWorker.h"
#include "../../src/SDSFile.h"

//#define LOGGING printf
#define LOGGING(...)

//----------------------------------------------------------------------------------
CMathWorker * g_cMathWorker = new CMathWorker();
bool g_bStarted = false;

static int64_t g_TotalAllocs = 0;
static int64_t g_TotalFree = 0;

//-----------------------------------------------
void * FmAlloc(size_t _Size)
{
    // make thread safe
    InterlockedIncrement64(&g_TotalAllocs);
    return malloc(_Size);
}

void FmFree(void * _Block)
{
    InterlockedIncrement64(&g_TotalFree);
    free(_Block);
}

int64_t SumBooleanMask(const int8_t * pData, int64_t length)
{
    // Basic input validation.
    if (! pData)
    {
        return 0;
    }
    else if (length < 0)
    {
        return 0;
    }
    int64_t result = 0;
    for (int64_t i = 0; i < length; i++)
    {
        if (pData[i])
        {
            result++;
        }
    }

    return result;
}

#if defined(_WIN32)

// global scope
typedef VOID(WINAPI * FuncGetSystemTime)(LPFILETIME);
FuncGetSystemTime g_GetSystemTime;
FILETIME g_TimeStart;
static bool g_IsPreciseTime = false;

//------------------------------------
// Returns windows time in Nanos
__inline static uint64_t GetWindowsTime()
{
    FILETIME timeNow;
    g_GetSystemTime(&timeNow);
    return (*(uint64_t *)&timeNow * 100) - 11644473600000000000L;
}

//-------------------------------------------------------------------
//
class CTimeStamp
{
public:
    CTimeStamp()
    {
        FARPROC fp;

        g_GetSystemTime = GetSystemTimeAsFileTime;

        HMODULE hModule = LoadLibraryW(L"kernel32.dll");

        // Use printf instead of logging because logging is probably not up yet
        // Logging uses the timestamping, so timestamping loads first
        if (hModule != NULL)
        {
            fp = GetProcAddress(hModule, "GetSystemTimePreciseAsFileTime");
            if (fp != NULL)
            {
                g_IsPreciseTime = true;
                // printf("Using precise GetSystemTimePreciseAsFileTime time...\n");
                g_GetSystemTime = (VOID(WINAPI *)(LPFILETIME))fp;
            }
            else
            {
                LOGGING("**Using imprecise GetSystemTimeAsFileTime...\n");
            }
        }
        else
        {
            printf("!! error load kernel32\n");
        }
    }
};

static CTimeStamp * g_TimeStamp = new CTimeStamp();
#else

    #define OUT /**/

    #include <sys/time.h>
    #include <time.h>
    #include <unistd.h>

uint64_t GetTimeStamp()
{
    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    // return tv.tv_sec*(uint64_t)1000000 + tv.tv_usec;

    struct timespec x;
    clock_gettime(CLOCK_REALTIME, &x);
    return x.tv_sec * 1000000000L + x.tv_nsec;
}

#endif

//---------------------------------------------------------
// Returns nanoseconds since utc epoch
uint64_t GetUTCNanos()
{
#if defined(_WIN32)
    return GetWindowsTime();
#else
    return GetTimeStamp();
#endif
}

#if defined(_WIN32)
bool APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:

        if (! g_bStarted)
        {
            g_bStarted = true;

            // default to numa node 0
            g_cMathWorker->StartWorkerThreads(0);
        }
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return true;
}
#endif

#include <vector>
typedef std::vector<const char *> SDS_STRING_LIST;

//----------------------------------------
// CALLBACK2 - can wrap more than one file
// finalCount is how many info sections to return
// if there are sections inside a single file, the finalCount > 1
void * ReadFinal(SDS_FINAL_CALLBACK * pSDSFinalCallback, int64_t finalCount)
{
    // printf("End ReadFinal %p.  finalCount %lld\n", pSDSFinalCallback, finalCount);
    return NULL;
}

//----------------------------------------
// CALLBACK2 - all files were stacked into one column
void * ReadFinalStack(SDS_STACK_CALLBACK * pSDSFinalCallback, int64_t finalCount, SDS_STACK_CALLBACK_FILES * pSDSFileInfo,
                      SDS_FILTER * pSDSFilter, int64_t fileCount)
{
    // PyObject* returnArrayTuple =
    //     ReadFinalStackArrays(
    //         pSDSFinalCallback,
    //         finalCount,
    //         pSDSFileInfo,
    //         pSDSFilter,
    //         fileCount);

    // returnItem = returnArrayTuple;

    // printf("End ReadFinalStack %p\n", pSDSFinalCallback);
    return pSDSFinalCallback;
}

//--------------------------------------------------
//
void * BeginAllowThreads()
{
    return NULL;
}

//--------------------------------------------------
//
void EndAllowThreads(void * saveObject) {}

//--------------------------------------------------
//
static char * GetMemoryOffset(char * BaseAddress, int64_t offset)
{
    return (BaseAddress + offset);
}

//--------------------------------------------------
//
static SDS_ARRAY_BLOCK * GetArrayBlock(char * baseOffset, int64_t arrayNum)
{
    SDS_ARRAY_BLOCK * pArrayBlock =
        (SDS_ARRAY_BLOCK *)GetMemoryOffset(baseOffset, ((SDS_FILE_HEADER *)baseOffset)->ArrayBlockOffset);
    return &pArrayBlock[arrayNum];
}

//---------------------------------------------------------
// Linux: long = 64 bits
// Windows: long = 32 bits
// static int FixupDType(int dtype, int64_t itemsize) {
//
//    if (dtype == NPY_LONG) {
//        // types 7 and 8 are ambiguous because of different compilers
//        if (itemsize == 4) {
//            dtype = NPY_INT;
//        }
//        else {
//            dtype = NPY_LONGLONG;
//        }
//    }
//
//    if (dtype == NPY_ULONG) {
//        // types 7 and 8 are ambiguous
//        if (itemsize == 4) {
//            dtype = NPY_UINT;
//        }
//        else {
//            dtype = NPY_ULONGLONG;
//        }
//    }
//    return dtype;
//}

struct stArray
{
    const char * pstrArrayName;
    char * pData;
    int ndims;
    int dtype;
    int enumValue;
    int64_t dimensions[5];
    int64_t itemsize;
    std::string strArrayName;
};

struct stReadSharedMemory
{
    int64_t NumArrays;
    int64_t MaxRow;
    int16_t FileType;
    int16_t StackType; // see SDS_STACK_TYPE
    int32_t AuthorId;  // see SDS_AUTHOR_ID
    const char * pstrMeta;
    stArray * pArrays;
    int64_t sizeofArray;
    void * pSDSDecompressFile;
    std::string strMeta; // holds string that deletes
};

//--------------------------------------------
// Returns a list of string (Column names)
//
// Entry:
// arg1: pointer to string, null terminated, followed by uint8_t enum value
// arg2: how many names
// arg3: the size of pArrayNames (all of the names)

void * MakeListNames(const char * pArrayNames, int64_t nameBlockCount, int64_t nameSize, stArray * pArrays)
{
    const char * nameData = pArrayNames;

    int64_t curPos = 0;
    // for every name
    while (nameBlockCount)
    {
        nameBlockCount--;
        const char * pStart = pArrayNames;

        // skip to end (search for 0 terminating char)
        while (*pArrayNames++)
            ;

        // get the enum
        uint8_t value = *pArrayNames++;

        LOGGING("makelist file name is %s, %lld\n", pStart, nameBlockCount);

        pArrays[curPos].strArrayName = pStart;
        pArrays[curPos].pstrArrayName = pArrays[curPos].strArrayName.c_str();
        pArrays[curPos].enumValue = value;

        curPos++;

        // If we ran too far, break
        if ((pArrayNames - nameData) >= nameSize)
            break;
    }
    return NULL;
}

//----------------------------------------------------
// Input: sharedmemory struct we are reading from
// Output: stReadSharedMemory struct
//
// TOOD: pass in handle to shared memory object so it can be deleted
void * ReadMemoryCallback(SDS_SHARED_MEMORY_CALLBACK * pSMCB)
{
    SDS_FILE_HEADER * pFileHeader = pSMCB->pFileHeader;
    char * baseOffset = pSMCB->baseOffset;
    int mode = pSMCB->mode;

    LOGGING("Reading from shared memory\n");
    int64_t arrayCount = pFileHeader->ArraysWritten;

    stReadSharedMemory * pReturnBack = new stReadSharedMemory;
    stArray * pArrays = new stArray[arrayCount];
    pReturnBack->NumArrays = arrayCount;
    pReturnBack->pArrays = pArrays;
    pReturnBack->MaxRow = 0;

    // Have to free this later
    pReturnBack->pSDSDecompressFile = pSMCB->pSDSDecompressFile;
    pReturnBack->FileType = pFileHeader->FileType;
    pReturnBack->StackType = pFileHeader->StackType;
    pReturnBack->AuthorId = pFileHeader->AuthorId;
    pReturnBack->sizeofArray = sizeof(stArray);

    //----------- LOAD ARRAY NAMES -------------------------
    int64_t nameSize = pFileHeader->NameBlockSize;
    if (nameSize)
    {
        char * nameData = GetMemoryOffset(baseOffset, pFileHeader->NameBlockOffset);
        MakeListNames(nameData, pFileHeader->NameBlockCount, nameSize, pArrays);
    }

    //------------- META DATA -------------------------------
    pReturnBack->strMeta =
        std::string((const char *)GetMemoryOffset(baseOffset, pFileHeader->MetaBlockOffset), pFileHeader->MetaBlockSize);
    pReturnBack->pstrMeta = pReturnBack->strMeta.c_str();

    //--------------- LOAD ARRAYS ---------------------------
    LOGGING("Number of arrays %lld\n", arrayCount);

    // Insert all the arrays
    for (int64_t i = 0; i < arrayCount; i++)
    {
        SDS_ARRAY_BLOCK * pArrayBlock = GetArrayBlock(baseOffset, i);

        // scalars
        // if (pArrayBlock->Dimensions ==0)

        char * data = GetMemoryOffset(baseOffset, pArrayBlock->ArrayDataOffset);

        // TODO: dtype fixup for Windows vs Linux
        int dtype = pArrayBlock->DType; // pArrayBlock->ItemSize);

        pArrays[i].pData = data;
        pArrays[i].dtype = dtype;
        pArrays[i].itemsize = pArrayBlock->ItemSize;
        pArrays[i].dimensions[0] = pArrayBlock->Dimensions[0];
        pArrays[i].dimensions[1] = pArrayBlock->Dimensions[1];
        pArrays[i].dimensions[2] = pArrayBlock->Dimensions[2];
        pArrays[i].dimensions[3] = pArrayBlock->Dimensions[3];
        pArrays[i].dimensions[4] = pArrayBlock->Dimensions[4];
        pArrays[i].ndims = pArrayBlock->NDim;

        // First column dictates size (only works for Dataset)
        if (i == 0)
        {
            pReturnBack->MaxRow = pArrayBlock->Dimensions[0];
        }
    }

    LOGGING("EndSharedMemory\n");
    return pReturnBack;
}

//==================================
// Called back when reading in data
void AllocateArrayCallback(SDS_ALLOCATE_ARRAY * pAllocateArray)
{
    SDSArrayInfo * pDestInfo = pAllocateArray->pDestInfo;
    int ndim = pAllocateArray->ndim;
    const char * pArrayName = pAllocateArray->pArrayName;

    LOGGING("Allocate array name: %s\n", pArrayName);
}

// Export DLL section
#if defined(_WIN32) && ! defined(__GNUC__)
    #define DllExport __declspec(dllexport)
#else
    #define DllExport
#endif

extern "C"
{
    //--------------------------------------------------
    // Input:
    //    fileName: make sure to append .SDS at end
    //    shareName: the sharename the user provided
    // Returns:
    //    pointer to stReadSharedMemory
    DllExport stReadSharedMemory * ReadFromSharedMemory(const char * fileName, const char * shareName)
    {
        // uint64_t fileNameSize;
        // uint64_t shareNameSize = 0;

        SDS_STRING_LIST * folderName = NULL;
        SDS_STRING_LIST * sectionsName = NULL;

        int32_t mode = COMPRESSION_MODE_DECOMPRESS_FILE;

        // fileNameSize = strlen(fileName);
        // shareNameSize = strlen(shareName);

        //==============================================
        // Build callback table
        SDS_READ_CALLBACKS sdsRCB;

        sdsRCB.ReadFinalCallback = ReadFinal;
        sdsRCB.StackFinalCallback = ReadFinalStack;
        sdsRCB.ReadMemoryCallback = ReadMemoryCallback;
        sdsRCB.AllocateArrayCallback = AllocateArrayCallback;
        sdsRCB.BeginAllowThreads = BeginAllowThreads;
        sdsRCB.EndAllowThreads = EndAllowThreads;
        sdsRCB.pInclusionList = NULL; // NO LONGER SUPPORTED includeDict;
        sdsRCB.pExclusionList = NULL;
        sdsRCB.pFolderInclusionList = NULL;

        SDS_READ_INFO sdsRI;

        sdsRI.mode = mode;

        //==============================================
        stReadSharedMemory * result =
            (stReadSharedMemory *)SDSReadFile(fileName, shareName, folderName, sectionsName, &sdsRI, &sdsRCB);

        return result;
    }

    struct stSimpleArray
    {
        int64_t itemsize;
        int dtype;
        int flags;
    };

    void ParseArrayString(const char * pString, OUT stSimpleArray * pSimpleArray)
    {
        if (! pString || *pString == 0)
        {
            pSimpleArray->itemsize = 1;
            pSimpleArray->dtype = SDS_BOOL;
            pSimpleArray->flags = 0;
            return;
        }

        // default to bool
        int dtype = SDS_BOOL;
        int64_t itemsize = 0;
        char firstLetter = *pString++;

        // calc itemsize
        while (*pString)
        {
            char num = *pString++;
            if (num >= '0' && num <= '9')
            {
                itemsize = itemsize * 10 + (num - '0');
            }
            else
            {
                break;
            }
        }

        if (itemsize == 0)
            itemsize = 1;
        if (firstLetter >= 'A' && firstLetter <= 'z')
        {
            switch (firstLetter)
            {
            case 'b':
                dtype = SDS_BOOL;
                break;
            case 'i':
                if (itemsize == 1)
                {
                    dtype = SDS_INT8;
                }
                if (itemsize == 2)
                {
                    dtype = SDS_INT16;
                }
                if (itemsize == 4)
                {
                    dtype = SDS_INT32;
                }
                if (itemsize == 8)
                {
                    dtype = SDS_INT64;
                }
                break;
            case 'u':
                if (itemsize == 1)
                {
                    dtype = SDS_UINT8;
                }
                if (itemsize == 2)
                {
                    dtype = SDS_UINT16;
                }
                if (itemsize == 4)
                {
                    dtype = SDS_UINT32;
                }
                if (itemsize == 8)
                {
                    dtype = SDS_UINT64;
                }
                break;
            case 'f':
                if (itemsize == 4)
                {
                    dtype = SDS_FLOAT;
                }
                if (itemsize == 8)
                {
                    dtype = SDS_DOUBLE;
                }
                if (itemsize == 16)
                {
                    dtype = SDS_LONGDOUBLE;
                }
                break;
            case 'S':
                dtype = SDS_STRING;
                break;
            case 'U':
                dtype = SDS_UNICODE;
                break;
            }
        }
        pSimpleArray->itemsize = itemsize;
        pSimpleArray->dtype = dtype;
        pSimpleArray->flags = 0;
    }

    //--------------------------------------------------
    // Input:
    int64_t CountCommas(const char * inListNames)
    {
        int64_t totalCount = 0;
        if (! *inListNames)
            return 0;

        while (*inListNames != 0)
        {
            if (*inListNames == ',')
            {
                totalCount++;
            }
            inListNames++;
        }
        totalCount++;
        return totalCount;
    }

    //--------------------------------------------------
    // Input:
    // Symbol:S4,BidSize:i4,
    // Output:
    //    pListNames
    //    pstSimpleArray
    //    returns size of pListNames
    int64_t BuildListInfoCpp(const char * inListNames, OUT char * pListNames, OUT stSimpleArray * pstSimpleArray)
    {
        int64_t listNameCount = CountCommas(inListNames);
        char * pStart = pListNames;

        for (int i = 0; i < listNameCount; i++)
        {
            while (*inListNames && *inListNames != ':')
            {
                // copy column name
                *pListNames++ = *inListNames++;
            }
            if (*inListNames == ':')
            {
                // null terminate.
                *pListNames++ = 0;
                // Store the 1 byte enum type
                *pListNames++ = (uint8_t)SDS_FLAGS_ORIGINAL_CONTAINER;

                // skip over colon
                inListNames++;
                ParseArrayString(inListNames, &pstSimpleArray[i]);
            }
            else
            {
                // default to bool
                pstSimpleArray[listNameCount].dtype = SDS_BOOL;
                pstSimpleArray[listNameCount].itemsize = 1;
                pstSimpleArray[listNameCount].flags = 0;
            }

            while (*inListNames && *inListNames != ',')
            {
                inListNames++;
            }
            if (*inListNames == ',')
            {
                inListNames++;
            }
            // skip white space
            while (*inListNames == ' ')
            {
                inListNames++;
            }
        }

        return pListNames - pStart;
    }

    //--------------------------------------------------
    // Input:
    //    pSharedMemory
    // Output:
    //    pListNames
    //    returns size of pListNames
    int64_t BuildListInfoFromShared(stReadSharedMemory * pSharedMemory, OUT char * pListNames)
    {
        int64_t listNameCount = pSharedMemory->NumArrays;
        char * pStart = pListNames;

        for (int i = 0; i < listNameCount; i++)
        {
            const char * inListNames = pSharedMemory->pArrays[i].pstrArrayName;
            while (*inListNames)
            {
                // copy column name
                *pListNames++ = *inListNames++;
            }
            // null terminate.
            *pListNames++ = 0;
            // Store the 1 byte enum type
            *pListNames++ = (uint8_t)SDS_FLAGS_ORIGINAL_CONTAINER;
        }

        return pListNames - pStart;
    }

    //--------------------------------------------------
    // Input:
    //    fileName: make sure to append .SDS at end
    //    shareName: the sharename the user provided, may be NULL
    // Returns:
    //    true/false
    bool CreateSDSFileInternal(const char * fileName, const char * shareName, const char * metaData,
                               const char * inListNames, // use commas to separate
                               int64_t totalRows, int64_t bandSize)
    {
        int32_t mode = COMPRESSION_MODE_COMPRESS_FILE;
        int32_t compType = COMPRESSION_TYPE_NONE; // COMPRESSION_TYPE_ZSTD;
        int32_t level = ZSTD_CLEVEL_DEFAULT;
        int32_t fileType = SDS_FILE_TYPE_DATASET;

        LOGGING("In CompressFile %s  Share: %s   TotalRows: %lld\n", fileName, shareName, totalRows);

        // Handle list of names ------------------------------------------------------
        int64_t listNameCount = CountCommas(inListNames);

        LOGGING("Name count is %d\n", (int)listNameCount);

        // alloc worst case scenario
        char * pListNames = (char *)WORKSPACE_ALLOC((SDS_MAX_FILENAME * listNameCount) + 8);
        if (! pListNames)
        {
            return NULL;
        }

        stSimpleArray * pstSimpleArray = (stSimpleArray *)WORKSPACE_ALLOC((sizeof(stSimpleArray) * listNameCount) + 8);
        if (! pstSimpleArray)
        {
            return NULL;
        }

        // Process list of names tuples
        int64_t listNameSize = BuildListInfoCpp(inListNames, pListNames, pstSimpleArray);

        // Handle list of numpy arrays -----------------------------------
        int64_t arrayCount = listNameCount;

        SDS_WRITE_CALLBACKS SDSWriteCallbacks;

        SDSWriteCallbacks.BeginAllowThreads = BeginAllowThreads;
        SDSWriteCallbacks.EndAllowThreads = EndAllowThreads;

        SDS_WRITE_INFO SDSWriteInfo;
        SDSWriteInfo.aInfo = (SDSArrayInfo *)WORKSPACE_ALLOC(sizeof(SDSArrayInfo) * arrayCount);

        //============================================
        // Convert from ArrayInfo* to SDSArrayInfo*
        //
        SDSArrayInfo * pDest = SDSWriteInfo.aInfo;

        for (int64_t i = 0; i < arrayCount; i++)
        {
            pDest->ArrayLength = totalRows;
            pDest->ItemSize = (int32_t)(pstSimpleArray[i].itemsize);
            pDest->pArrayObject = NULL;
            pDest->NumBytes = pstSimpleArray[i].itemsize * totalRows;
            pDest->NumpyDType = pstSimpleArray[i].dtype;

            // This will allocate and zero an array as opposed to consuming an existing array
            // or allocating an array and NOT zeroing it out
            pDest->pData = (char *)WORKSPACE_ALLOC(pDest->NumBytes);
            if (pDest->pData)
            {
                RtlZeroMemory(pDest->pData, pDest->NumBytes);
            }

            int32_t ndim = 1;
            // if (ndim > SDS_MAX_DIMS) {
            //    printf("!!!SDS: array dimensions too high: %d\n", ndim);
            //    ndim = SDS_MAX_DIMS;
            // }
            // if (ndim < 1) {
            //    printf("!!!SDS: array dimensions too low: %d\n", ndim);
            //    ndim = 1;
            // }
            pDest->NDim = ndim;

            for (int dim_idx = 0; dim_idx < SDS_MAX_DIMS; dim_idx++)
            {
                pDest->Dimensions[dim_idx] = 0;
                pDest->Strides[dim_idx] = 0;
            }

            pDest->Dimensions[0] = totalRows;
            pDest->Strides[0] = pDest->ItemSize;
            pDest->Flags = pstSimpleArray[i].flags;

            pDest++;
        }

        SDSWriteInfo.arrayCount = arrayCount;

        // meta information
        SDSWriteInfo.metaData = metaData;
        SDSWriteInfo.metaDataSize = metaData ? (uint32_t)strlen(metaData) : 0;

        // names of arrays information
        SDSWriteInfo.pListNames = pListNames;
        SDSWriteInfo.listNameSize = listNameSize;
        SDSWriteInfo.listNameCount = listNameCount;

        // compressed or uncompressed
        SDSWriteInfo.mode = mode;
        SDSWriteInfo.compType = compType;
        SDSWriteInfo.level = level;

        // NEED TO SEND in
        SDSWriteInfo.sdsFileType = fileType;
        SDSWriteInfo.sdsAuthorId = SDS_AUTHOR_ID_CSHARP;

        // section and append information
        SDSWriteInfo.appendFileHeadersMode = false;
        SDSWriteInfo.appendRowsMode = false;
        SDSWriteInfo.appendColumnsMode = false;
        SDSWriteInfo.bandSize = bandSize;

        SDSWriteInfo.sectionName = NULL;
        SDSWriteInfo.sectionNameSize = 0;

        bool result = SDSWriteFile(fileName,
                                   shareName, // can be NULL
                                   NULL, &SDSWriteInfo, &SDSWriteCallbacks);

        // Free all the arrays we allocated then zeroed out
        pDest = SDSWriteInfo.aInfo;
        for (int64_t i = 0; i < arrayCount; i++)
        {
            if (pDest[i].pData)
                WORKSPACE_FREE(pDest[i].pData);
        }

        // FREE workspace allocations
        WORKSPACE_FREE(SDSWriteInfo.aInfo);

        WORKSPACE_FREE(pListNames);
        WORKSPACE_FREE(pstSimpleArray);

        return result;
    }

    DllExport bool CreateSDSFile(const char * fileName, const char * shareName, const char * metaData,
                                 const char * inListNames, // use commas to separate
                                 int64_t totalRows, int64_t bandSize = 0)
    {
        bool result = CreateSDSFileInternal(fileName, shareName, metaData,
                                            inListNames, // use commas to separate
                                            totalRows, bandSize);

        return result;
    }

    //--------------------------------------------------
    // Input:
    //    outFileName: full ASCIIZ path of file on disk
    //    shareFileName: make sure to append .SDS at end
    //    shareName: the sharename the user provided, may be NULL
    // Returns:
    //    true/false
    DllExport bool AppendSDSFile(const char * outFileName,

                                 const char * shareFileName, const char * shareName, int64_t totalRows, int64_t bandSize = 0)
    {
        stReadSharedMemory * pSharedMemory = (stReadSharedMemory *)ReadFromSharedMemory(shareFileName, shareName);
        if (pSharedMemory)
        {
            SDS_WRITE_CALLBACKS SDSWriteCallbacks;

            SDSWriteCallbacks.BeginAllowThreads = BeginAllowThreads;
            SDSWriteCallbacks.EndAllowThreads = EndAllowThreads;

            SDS_WRITE_INFO SDSWriteInfo;
            SDSWriteInfo.aInfo = (SDSArrayInfo *)WORKSPACE_ALLOC(sizeof(SDSArrayInfo) * pSharedMemory->NumArrays);

            if (totalRows > pSharedMemory->MaxRow)
            {
                totalRows = pSharedMemory->MaxRow;
            }

            //============================================
            // Convert from ArrayInfo* to SDSArrayInfo*
            //
            SDSArrayInfo * pDest = SDSWriteInfo.aInfo;

            for (int64_t i = 0; i < pSharedMemory->NumArrays; i++)
            {
                stArray * pstArray = &pSharedMemory->pArrays[i];
                pDest->ArrayLength = totalRows;
                pDest->ItemSize = (int32_t)(pstArray->itemsize);
                pDest->pArrayObject = NULL;
                pDest->NumBytes = pstArray->itemsize * totalRows;
                pDest->NumpyDType = pstArray->dtype;
                pDest->pData = pstArray->pData;

                int32_t ndim = 1;
                pDest->NDim = ndim;

                for (int dim_idx = 0; dim_idx < SDS_MAX_DIMS; dim_idx++)
                {
                    pDest->Dimensions[dim_idx] = 0;
                    pDest->Strides[dim_idx] = 0;
                }

                pDest->Dimensions[0] = totalRows;
                pDest->Strides[0] = pDest->ItemSize;
                pDest->Flags = 0;

                pDest++;
            }

            SDSWriteInfo.arrayCount = pSharedMemory->NumArrays;

            // meta information
            SDSWriteInfo.metaData = pSharedMemory->pstrMeta;
            SDSWriteInfo.metaDataSize = SDSWriteInfo.metaData ? (uint32_t)strlen(SDSWriteInfo.metaData) : 0;

            int64_t listNameCount = pSharedMemory->NumArrays;

            // alloc worst case scenario
            char * pListNames = (char *)WORKSPACE_ALLOC((SDS_MAX_FILENAME * listNameCount) + 8);
            if (! pListNames)
            {
                return NULL;
            }

            // Process list of names tuples
            int64_t listNameSize = BuildListInfoFromShared(pSharedMemory, pListNames);

            // names of arrays information
            SDSWriteInfo.pListNames = pListNames;
            SDSWriteInfo.listNameSize = listNameSize;
            SDSWriteInfo.listNameCount = listNameCount;

            // compressed or uncompressed
            SDSWriteInfo.mode = COMPRESSION_MODE_COMPRESS_FILE;
            SDSWriteInfo.compType = COMPRESSION_TYPE_ZSTD;
            SDSWriteInfo.level = ZSTD_CLEVEL_DEFAULT;

            // NEED TO SEND in
            SDSWriteInfo.sdsFileType = SDS_FILE_TYPE_DATASET;
            SDSWriteInfo.sdsAuthorId = SDS_AUTHOR_ID_CSHARP;

            // section and append information
            SDSWriteInfo.appendFileHeadersMode = true;
            SDSWriteInfo.appendRowsMode = true;
            SDSWriteInfo.appendColumnsMode = false;
            SDSWriteInfo.bandSize = bandSize;

            SDSWriteInfo.sectionName = NULL;
            SDSWriteInfo.sectionNameSize = 0;

            // TODO: come up with a section name
            const char * pSectionName = "0";
            if (pSectionName)
            {
                SDSWriteInfo.appendRowsMode = true;
                SDSWriteInfo.sectionName = pSectionName;
                SDSWriteInfo.sectionNameSize = strlen(pSectionName);
            }

            bool result = SDSWriteFile(outFileName,
                                       NULL, // no sharename this time
                                       NULL, &SDSWriteInfo, &SDSWriteCallbacks);

            if (! result)
            {
                printf("Failed to write to %s\n", outFileName);
            }

            return result;
        }

        printf("Failed to open shared memory %s\n", shareFileName);

        return false;
    }
}
