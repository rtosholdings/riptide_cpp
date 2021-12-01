#ifndef RIPTIDE_CPP_SHAREDMEMORY_H
#define RIPTIDE_CPP_SHAREDMEMORY_H

#include "CommonInc.h"

#if defined(_WIN32) && ! defined(__GNUC__)

typedef struct MAPPED_VIEW_STRUCT MAPPED_VIEW_STRUCT, *PMAPPED_VIEW_STRUCT;
struct MAPPED_VIEW_STRUCT
{
    void * BaseAddress; // Past RCF Header
    void * MapHandle;
    void * FileHandle;
    int64_t FileSize;
    void * pSharedMemoryHeader;
    int64_t RealFileSize; // Includes RCF Header size
    int64_t RefCount;
};

HRESULT
UtilSharedMemoryBegin(const char * pMappingName, int64_t Size, PMAPPED_VIEW_STRUCT * pReturnStruct);

HRESULT
UtilSharedNumaMemoryBegin(const char * pMappingName, int64_t Size,
                          uint32_t nndPreferred, // preferred numa node
                          void * lpBaseAddress, PMAPPED_VIEW_STRUCT * pReturnStruct);

HRESULT
UtilSharedMemoryCopy(const char * pMappingName, PMAPPED_VIEW_STRUCT * pReturnStruct, int bTest);

HRESULT
UtilSharedMemoryEnd(PMAPPED_VIEW_STRUCT pMappedViewStruct);

HRESULT
UtilMappedViewReadBegin(const char * pszFilename, PMAPPED_VIEW_STRUCT * pReturnStruct);

HRESULT
UtilMappedViewReadEnd(PMAPPED_VIEW_STRUCT pMappedViewStruct);

HRESULT
UtilMappedViewWriteEnd(PMAPPED_VIEW_STRUCT pMappedViewStruct);

HRESULT
UtilMappedViewWriteBegin(const char * pszFilename, PMAPPED_VIEW_STRUCT * pReturnStruct, uint32_t dwMaxSizeHigh,
                         uint32_t dwMaxSizeLow);

#else

typedef int HRESULT;

typedef struct MAPPED_VIEW_STRUCT MAPPED_VIEW_STRUCT, *PMAPPED_VIEW_STRUCT;
struct MAPPED_VIEW_STRUCT
{
    void * BaseAddress;
    int FileHandle;
    int64_t FileSize;
    int64_t RefCount;
};

HRESULT
UtilSharedMemoryBegin(const char * pMappingName, int64_t Size, PMAPPED_VIEW_STRUCT * pReturnStruct);

HRESULT
UtilSharedMemoryCopy(const char * pMappingName, PMAPPED_VIEW_STRUCT * pReturnStruct, int bTest);

HRESULT
UtilSharedMemoryEnd(PMAPPED_VIEW_STRUCT pMappedViewStruct);

#endif
#endif
