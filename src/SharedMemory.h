
#if defined(_WIN32) && !defined(__GNUC__)


typedef struct MAPPED_VIEW_STRUCT MAPPED_VIEW_STRUCT, *PMAPPED_VIEW_STRUCT;
struct MAPPED_VIEW_STRUCT
{

   void*    BaseAddress;         // Past RCF Header
   void*    MapHandle;
   void*    FileHandle;
   INT64    FileSize;
   void*    pSharedMemoryHeader;
   INT64    RealFileSize;        // Includes RCF Header size
   INT64    RefCount;
};


HRESULT
UtilSharedMemoryBegin(
   const char*          pMappingName,
   INT64                Size,
   PMAPPED_VIEW_STRUCT *pReturnStruct);


HRESULT
UtilSharedNumaMemoryBegin(
   const char*          pMappingName,
   INT64                Size,
   DWORD                nndPreferred,        // preferred numa node
   LPVOID               lpBaseAddress,
   PMAPPED_VIEW_STRUCT *pReturnStruct);

HRESULT
UtilSharedMemoryCopy(
   const char*          pMappingName,
   PMAPPED_VIEW_STRUCT *pReturnStruct,
   BOOL                 bTest);

HRESULT
UtilSharedMemoryEnd(
   PMAPPED_VIEW_STRUCT  pMappedViewStruct);

HRESULT
UtilMappedViewReadBegin(
   const char*          pszFilename,
   PMAPPED_VIEW_STRUCT *pReturnStruct);

HRESULT
UtilMappedViewReadEnd(
   PMAPPED_VIEW_STRUCT pMappedViewStruct);

HRESULT
UtilMappedViewWriteEnd(
   PMAPPED_VIEW_STRUCT pMappedViewStruct);

HRESULT
UtilMappedViewWriteBegin(
   const char*          pszFilename,
   PMAPPED_VIEW_STRUCT *pReturnStruct,
   DWORD                dwMaxSizeHigh,
   DWORD                dwMaxSizeLow);


#else

typedef int HRESULT;

typedef struct MAPPED_VIEW_STRUCT MAPPED_VIEW_STRUCT, *PMAPPED_VIEW_STRUCT;
struct MAPPED_VIEW_STRUCT
{

   void*    BaseAddress;         
   int      FileHandle;
   INT64    FileSize;
   INT64    RefCount;

};

HRESULT
UtilSharedMemoryBegin(
   const char*           pMappingName,
   INT64                Size,
   PMAPPED_VIEW_STRUCT *pReturnStruct);

HRESULT
UtilSharedMemoryCopy(
   const char*          pMappingName,
   PMAPPED_VIEW_STRUCT *pReturnStruct,
   BOOL                 bTest);

HRESULT
UtilSharedMemoryEnd(
   PMAPPED_VIEW_STRUCT  pMappedViewStruct);


#endif


