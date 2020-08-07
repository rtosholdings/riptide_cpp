#pragma once
#pragma once
#include <vector>


/*-=====  Pre-defined compression levels  =====-*/
#ifndef ZSTD_CLEVEL_DEFAULT
#define ZSTD_CLEVEL_DEFAULT 3
#endif

#ifndef ZSTD_MAX_CLEVEL
#define ZSTD_MAX_CLEVEL     22
#endif

#ifndef ZSTD_MIN_CLEVEL
#define ZSTD_MIN_CLEVEL     -5
#endif

// note: should an enum
#define COMPRESSION_TYPE_NONE    0
#define COMPRESSION_TYPE_ZSTD    1

// note: should an enum
#define COMPRESSION_MODE_COMPRESS 0
#define COMPRESSION_MODE_DECOMPRESS 1
#define COMPRESSION_MODE_COMPRESS_FILE 2
#define COMPRESSION_MODE_DECOMPRESS_FILE 3
#define COMPRESSION_MODE_SHAREDMEMORY 4
#define COMPRESSION_MODE_INFO 5
#define COMPRESSION_MODE_COMPRESS_APPEND_FILE 6

// note: should an enum
#define SDS_MULTI_MODE_ERROR 0
#define SDS_MULTI_MODE_READ_MANY  1
#define SDS_MULTI_MODE_READ_MANY_INFO  2
#define SDS_MULTI_MODE_STACK_MANY  3
#define SDS_MULTI_MODE_STACK_MANY_INFO  4
#define SDS_MULTI_MODE_CONCAT_MANY  5
// when not sure if stacking or reading horizontal
#define SDS_MULTI_MODE_STACK_OR_READMANY  6

#define COMPRESSION_MAGIC  253

#define SDS_VERSION_HIGH 4

// TJD added version low = 2 May  2019 for "folders"
// TJD added version low = 3 July 2019 for "bandsize"
// TJD added version low = 4 August 2019 for "section"
// TJD added version low = 5 March 2020 for timestamps and better section, boolean bitmasks
#define SDS_VERSION_LOW  5

#define SDS_VALUE_ERROR 1

#define SDS_MAGIC 0x20534453
#define SDS_PADSIZE  512
#define SDS_PAD_NUMBER(_NUM_) ((_NUM_+ 511) & ~511)

#if defined(_WIN32) && !defined(__GNUC__)

#define SDS_FILE_HANDLE void*
#define SDS_EVENT_HANDLE void*
#define BAD_SDS_HANDLE NULL

#else
#define SDS_FILE_HANDLE int
#define SDS_EVENT_HANDLE void*
#define BAD_SDS_HANDLE 0

#endif

extern std::vector<std::string>   g_gatewaylist;
extern char g_errorbuffer[512];
extern int  g_lastexception;

typedef std::vector<const char*> SDS_STRING_LIST;

#define SDS_MAX_FILENAME 300
#define SDS_MAX_SECTIONNAME 64

//----------------------------------------------------
// NOTE:
// Filename rule "!" may be used to separate nested structs
// array name rule "!" may be used for sub arrays
// sam!col_0/col_1  <- categoricals
//
// 8 bits in the enum -- the LSB 2 bits are defined
#define SDS_FLAGS_ORIGINAL_CONTAINER 1    // Represents a column name in Dataset
#define SDS_FLAGS_STACKABLE          2    // Often used to mark a categorical side car data (can stack but not part of orig container)
#define SDS_FLAGS_SCALAR             4
#define SDS_FLAGS_NESTED             8    // This is set when there is a hierarchy but NO DATA
#define SDS_FLAGS_META              16    // This array is string or unicode that contains meta data (from onefile)

// IN HEADER
#define SDS_FILE_TYPE_UNKNOWN       0
#define SDS_FILE_TYPE_STRUCT        1
#define SDS_FILE_TYPE_DATASET       2
#define SDS_FILE_TYPE_TABLE         3
#define SDS_FILE_TYPE_ARRAY         4
#define SDS_FILE_TYPE_ONEFILE       5

#define SDS_STACK_TYPE_NONE         0
#define SDS_STACK_TYPE_CONCAT       1   // When sds_concat is called and reader wants to stack

#define SDS_AUTHOR_ID_UNKNOWN       0
#define SDS_AUTHOR_ID_PYTHON        1
#define SDS_AUTHOR_ID_MATLAB        2
#define SDS_AUTHOR_ID_CSHARP        3

//===========================================
// taken from numpy
enum SDS_TYPES {
   SDS_BOOL = 0,
   SDS_BYTE, SDS_UBYTE,
   SDS_SHORT, SDS_USHORT,
   SDS_INT, SDS_UINT,
   SDS_LONG, SDS_ULONG,
   SDS_LONGLONG, SDS_ULONGLONG,
   SDS_FLOAT, SDS_DOUBLE, SDS_LONGDOUBLE,
   SDS_CFLOAT, SDS_CDOUBLE, SDS_CLONGDOUBLE,
   SDS_OBJECT = 17,
   SDS_STRING, SDS_UNICODE,
   SDS_VOID
};

#define SDS_INT8 SDS_BYTE
#define SDS_UINT8 SDS_UBYTE
#define SDS_INT16 SDS_SHORT
#define SDS_UINT16 SDS_USHORT
#define SDS_INT32 SDS_INT
#define SDS_UINT32 SDS_UINT
#define SDS_INT64 SDS_LONGLONG
#define SDS_UINT64 SDS_ULONGLONG

/*
* Means c-style contiguous (last index varies the fastest). The data
* elements right after each other.
*
* This flag may be requested in constructor functions.
* This flag may be tested for in PyArray_FLAGS(arr).
*/
#define SDS_ARRAY_C_CONTIGUOUS    0x0001

/*
* Set if array is a contiguous Fortran array: the first index varies
* the fastest in memory (strides array is reverse of C-contiguous
* array)
*
* This flag may be requested in constructor functions.
* This flag may be tested for in PyArray_FLAGS(arr).
*/
#define SDS_ARRAY_F_CONTIGUOUS    0x0002

/*
* Note: all 0-d arrays are C_CONTIGUOUS and F_CONTIGUOUS. If a
* 1-d array is C_CONTIGUOUS it is also F_CONTIGUOUS. Arrays with
* more then one dimension can be C_CONTIGUOUS and F_CONTIGUOUS
* at the same time if they have either zero or one element.
* If SDS_RELAXED_STRIDES_CHECKING is set, a higher dimensional
* array is always C_CONTIGUOUS and F_CONTIGUOUS if it has zero elements
* and the array is contiguous if ndarray.squeeze() is contiguous.
* I.e. dimensions for which `ndarray.shape[dimension] == 1` are
* ignored.
*/

/*
* If set, the array owns the data: it will be free'd when the array
* is deleted.
*
* This flag may be tested for in PyArray_FLAGS(arr).
*/
#define SDS_ARRAY_OWNDATA         0x0004

/*
* An array never has the next four set; they're only used as parameter
* flags to the various FromAny functions
*
* This flag may be requested in constructor functions.
*/

/* Cause a cast to occur regardless of whether or not it is safe. */
#define SDS_ARRAY_FORCECAST       0x0010

/*
* Always copy the array. Returned arrays are always CONTIGUOUS,
* ALIGNED, and WRITEABLE.
*
* This flag may be requested in constructor functions.
*/
#define SDS_ARRAY_ENSURECOPY      0x0020

/*
* Make sure the returned array is a base-class ndarray
*
* This flag may be requested in constructor functions.
*/
#define SDS_ARRAY_ENSUREARRAY     0x0040

/*
* Make sure that the strides are in units of the element size Needed
* for some operations with record-arrays.
*
* This flag may be requested in constructor functions.
*/
#define SDS_ARRAY_ELEMENTSTRIDES  0x0080

/*
* Array data is aligned on the appropriate memory address for the type
* stored according to how the compiler would align things (e.g., an
* array of integers (4 bytes each) starts on a memory address that's
* a multiple of 4)
*
* This flag may be requested in constructor functions.
* This flag may be tested for in PyArray_FLAGS(arr).
*/
#define SDS_ARRAY_ALIGNED         0x0100

/*
* Array data has the native endianness
*
* This flag may be requested in constructor functions.
*/
#define SDS_ARRAY_NOTSWAPPED      0x0200

/*
* Array data is writeable
*
* This flag may be requested in constructor functions.
* This flag may be tested for in PyArray_FLAGS(arr).
*/
#define SDS_ARRAY_WRITEABLE       0x0400

/*
* If this flag is set, then base contains a pointer to an array of
* the same size that should be updated with the current contents of
* this array when this array is deallocated
*
* This flag may be requested in constructor functions.
* This flag may be tested for in PyArray_FLAGS(arr).
*/
#define SDS_ARRAY_UPDATEIFCOPY    0x1000

// TJD New for ver 4.3
// If this flag is set, the allocation of the array has been filtered
#define SDS_ARRAY_FILTERED        0x8000

// The max dimensions we can store, if > 5, the remaining dims are multiplied together
#define SDS_MAX_DIMS 5

// Max number of worker threads allowed
#define SDS_MAX_THREADS 64

// File format layout for one array block header
struct SDS_ARRAY_BLOCK {

   //----- offset 0 -----
   INT16       HeaderTag;
   INT16       HeaderLength;

   UINT8       Magic;  // See compression magic
   INT8        CompressionType;
   INT8        DType;
   INT8        NDim;

   //----- offset 8 -----
   INT32       ItemSize;
   INT32       Flags;

   //----- offset 16 -----
   // no more than 5 dims
   INT64       Dimensions[SDS_MAX_DIMS];
   INT64       Strides[SDS_MAX_DIMS];

   INT64       ArrayDataOffset;     // Start of data in file                          
                                    // If banding, then (UncompressedSize + (BandSize-1)) / BandSize => # of bands
   INT64       ArrayCompressedSize;
   INT64       ArrayUncompressedSize;
   INT32       ArrayBandCount;      // 0=no banding
   INT32       ArrayBandSize;       // 0=no banding

};

//----------------------------------
// At offset 0 of the file is this header
struct SDS_FILE_HEADER {

   UINT32      SDSHeaderMagic;
   INT16       VersionHigh;
   INT16       VersionLow;

   INT16       CompMode;
   INT16       CompType;
   INT32       CompLevel;

   //----- offset 16 -----
   INT64       NameBlockSize;
   INT64       NameBlockOffset;
   INT64       NameBlockCount;

   INT16       FileType;  // see SDS_FILE_TYPE
   INT16       StackType;  // see SDS_STACK_TYPE
   INT32       AuthorId;  // see SDS_AUTHOR_ID

   //----- offset 48 -----
   INT64       MetaBlockSize;
   INT64       MetaBlockOffset;

   //----- offset 64 -----
   INT64       TotalMetaCompressedSize;
   INT64       TotalMetaUncompressedSize;

   //----- offset 80 -----
   INT64       ArrayBlockSize;
   INT64       ArrayBlockOffset;

   //----- offset 96 -----
   INT64       ArraysWritten;
   INT64       ArrayFirstOffset;

   //----- offset 112 -----
   INT64       TotalArrayCompressedSize;        // Includes the SDS_PADDING, next relative offset to write to
   INT64       TotalArrayUncompressedSize;

   //----- offset 128 -----
   //------------- End of Version 4.1 ---------
   //------------- Version 4.3 starts here ----
   INT64       BandBlockSize;       // 00 if nothing
   INT64       BandBlockOffset;     // 00
   INT64       BandBlockCount;      // Number of names
   INT64       BandSize;            // How many elements before creating another band

   //----- offset 160 -----
   //------------- End of Version 4.3 ---------
   //------------- Version 4.4 starts here ----
   INT64       SectionBlockSize;    // how many bytes valid data
   INT64       SectionBlockOffset;  // points to section directory if it exists (often NULL if file was never appended)
                                    // if exists was often the LastFileOffset when an append operation was done
                                    // Then can read that offset to get a list of
                                    // NAMES\0\NAME2\0
   INT64       SectionBlockCount;   // number of names (number of sections total) (often 0 if never appended)
   INT64       SectionBlockReservedSize;   // total size of what we reserved for this block (so we can append names)

   //----- offset 192 -----
   INT64       FileOffset;          // this file offset within the file
   UINT64      TimeStampUTCNanos;   // time stamp when file was last written (0s indicate no time stamp)

   //----- offset 208 -----
   char        Reserved[512 - 208];

   //-----------------------------
   // function to multithreaded safe calculate the next array offset to write to
   // returns the relative file offset (has added pFileHeader->ArrayFirstOffset)
   INT64       AddArrayCompressedSize(INT64 cSize) {
      INT64 padSize = SDS_PAD_NUMBER(cSize);
      INT64 fileOffset = InterlockedAdd64(&TotalArrayCompressedSize, padSize) - padSize;

      // Calculate file offset location where arrays are stored
      fileOffset += ArrayFirstOffset;

      return fileOffset;
   }

   // only valid after file header has been filled in
   INT64    GetEndOfFileOffset() {
      return TotalArrayCompressedSize + ArrayFirstOffset;
   }
};

//---------------------------------------
class SDSSectionName {
public:
   // Section data created when user appends to a file with section name
   char*          pSectionData = NULL;
   const char**   pSectionNames = NULL;
   INT64*         pSectionOffsets = NULL;
   INT64          SectionCount = 0;
   INT64          SectionOffset = 0;

   // For creating one when it was missing (first time appended)
   const char* g_firstsectioname = "0";
   INT64       g_firstsectionoffset = 0;
   char        g_firstsectiondata[10] = { '0',0,0,0,0,0,0,0,0,0 };

   //-----------------------------------------------------------
   // Add one more name and offset
   // Returns sizeof new section
   // Returns pointer in *pListNames
   // NOTE: caller must WORKSPACE_FREE *pListNames
   INT64
      BuildSectionNamesAndOffsets(
         char** pListNames,                // Returned
         const char*  pNewSectionName,
         INT64  newSectionOffset           // 0 Allowed
      );
   void AllocateSectionData(INT64 sectionBlockCount, INT64 sectionSize);

   void DeleteSectionData();
   void MakeListSections(const INT64 sectionBlockCount, const INT64 sectionByteSize);
   char* MakeFirstSectionName();
   char* ReadListSections(SDS_FILE_HANDLE SDSFile, SDS_FILE_HEADER *pFileHeader);
   ~SDSSectionName();
};

//------------ PAD TO 512 ----------------------
// 4 of these (4*128) can fit in 512
// TJD: NOT USED YET
struct SDS_FOLDER_HEADER {
   INT64       FolderHeaderOffset;     // points to SDS_FILE_HEADER
   INT64       FolderCreateTimeUTC;
   INT8        FolderType;             // 0 for now SDS_FILE_TYPE_UNKNOWN, etc.
   INT8        FolderNameLength;       // up to 110 chars
   char        FolderName[128 - 16 -2];
};

//---------------------------------------------------------------------
// NOTE: See ArrayInfo and keep similar
struct SDSArrayInfo {

   // Numpy/Matlab/3rd party object  ??? is this needed anymore since pData exists?
   void*       pArrayObject;

   // First bytes of array data
   char*       pData;

   // total number of items
   INT64       ArrayLength;

   // total number of items * itemsize
   INT64       NumBytes;

   // see SDS_TYPES which follow numpy dtypes
   INT32       NumpyDType;

   // Number of dimensions (not to exceed 3 currently)
   INT32       NDim;

   // Width in bytes of one row
   INT32       ItemSize;

   // See SDS_FLAGS which are same as numpy flags
   INT32       Flags;

   // no more than SDS_MAX_DIMS dims
   INT64       Dimensions[SDS_MAX_DIMS];
   INT64       Strides[SDS_MAX_DIMS];

};

//---------------------------------------------------------------------
// NOTE: filter or mask must be passed
struct SDSFilterInfo {
   INT64    TrueCount;  // If zero, dont bother reading in data
   INT64    RowOffset;  // sum of all previous True Count
};

struct SDS_FILTER {
   //INT32*                           pFancyMask;
   //INT64                            FancyLength;         // length of the index array as well as final array

   // if BoolMaskLength == BoolMaskTrueCount and BoolMaskLength < 100
   // we assume this is looking at the header (first 5, 10, 100 rows)
   bool*                            pBoolMask;
   INT64                            BoolMaskLength;      // length of bool mask (but not final array)
   INT64                            BoolMaskTrueCount;   // length of final array
   SDSFilterInfo*                   pFilterInfo;         // allocated on the fly when stacking
};


//==================================================
// Internal structure used to decompress or read SDS files
//
struct SDS_WRITE_COMPRESS_ARRAYS {
   SDSArrayInfo*        pArrayInfo;
   SDS_ARRAY_BLOCK*     pBlockInfo;

   // used when decompressing
   SDS_FILE_HANDLE      fileHandle;
   SDS_FILE_HEADER*     pFileHeader;

   // Used when going to shared memory
   class SharedMemory*  pMemoryIO;

   INT64                totalHeaders;
   INT16                compMode;
   INT16                compType;
   INT32                compLevel;

   // Per core allocations
   void*                pCoreMemory[SDS_MAX_THREADS];
   INT64                pCoreMemorySize[SDS_MAX_THREADS];
   SDS_EVENT_HANDLE     eventHandles[SDS_MAX_THREADS];

   // See: compMode value -- COMPRESSIOM_MODE_SHAREDMEMORY
   // used when compressing to file

   // This is the callback from SDS_ALLOCATE_ARRAYS
   // The bottom end of this structure is allocated based on how many arrays
   //??SDSArrayInfo          ArrayInfo[1];
};



//==================================================
// Internal structure used to decompress or read SDS files
//
struct SDS_READ_DECOMPRESS_ARRAYS {
   struct SDS_READ_CALLBACKS*  pReadCallbacks;
   //SDSArrayInfo*        pArrayInfo;
   SDS_ARRAY_BLOCK*     pBlockInfo;

   // used when decompressing
   SDS_FILE_HANDLE      fileHandle;
   SDS_FILE_HEADER*     pFileHeader;

   // New filtering
   SDS_FILTER*          pFilter;

   // Used when going to shared memory
   class SharedMemory*  pMemoryIO;

   INT64                totalHeaders;
   INT16                compMode;
   INT16                compType;
   INT32                compLevel;

   // Per core allocations
   void*                pCoreMemory[SDS_MAX_THREADS];
   INT64                pCoreMemorySize[SDS_MAX_THREADS];
   SDS_EVENT_HANDLE     eventHandles[SDS_MAX_THREADS];

   // See: compMode value -- COMPRESSIOM_MODE_SHAREDMEMORY
   // used when compressing to file

   // This is the callback from SDS_ALLOCATE_ARRAYS
   // The bottom end of this structure is allocated based on how many arrays
   SDSArrayInfo          ArrayInfo[1];
};


struct SDS_STACK_CALLBACK_FILES {
   const char*       Filename;
   const char*       MetaData;
   INT64             MetaDataSize;
   SDS_FILE_HEADER*  pFileHeader;
};

//-------------------------------------------------
// For reading...
struct SDS_STACK_CALLBACK {

   const char*       ArrayName;
   void *            pArrayObject;
   INT64*            pArrayOffsets;
   SDS_ARRAY_BLOCK*  pArrayBlock;
   INT32             ArrayEnum;
};



//-------------------------------------------------
// For reading...
struct SDS_FINAL_CALLBACK {
   SDS_FILE_HEADER*  pFileHeader;

   INT32             mode;                      // info or decompress
   INT32             reserved1;

   INT64             arraysWritten;
   SDS_ARRAY_BLOCK*  pArrayBlocks;
   SDSArrayInfo*     pArrayInfo;

   //SDS_READ_DECOMPRESS_ARRAYS*  pstCompressArrays;

   char*             metaData;
   INT64             metaSize;

   char*             nameData;                  // array name data
   SDSSectionName*   pSectionName;              // if the file has section, else NULL
   const char*       strFileName;               // the filename read in
};


//-------------------------------------------------
struct SDS_SHARED_MEMORY_CALLBACK {
   SDS_FILE_HEADER*  pFileHeader;
   char*             baseOffset;

   INT32             mode;
   INT32             reserved1;
   void*             pSDSDecompressFile;

   // used to close memory/file handles
   void*             pMapStruct;
};

//---------------------------------------
// Callee must fill in pDestInfo
struct SDS_ALLOCATE_ARRAY {
   SDSArrayInfo*  pDestInfo;
   //void*          pInclusionList;
   //void*          pExclusionList;

   // Tells caller how to allocate
   int            ndim;
   INT64*         dims;
   INT32          numpyType;
   INT64          itemsize;
   char*          data;
   INT32          numpyFlags;
   INT64*         strides;
   const char*    pArrayName;
   INT32          sdsFlags;     //  see SDS_FLAGS_ORIGINAL_CONTAINER, etc.
};


//-------------------------------------------------
// Callbacks

// Called at the end of reading a file
typedef void*(*SDS_READ_FINAL_CALLBACK)(SDS_FINAL_CALLBACK* pSDSFinalCallback, INT64 finalCount);

// Called at the end of reading a stacked file
typedef void*(*SDS_STACK_FINAL_CALLBACK)(
   SDS_STACK_CALLBACK* pSDSFinalCallback, 
   INT64 finalCount, 
   SDS_STACK_CALLBACK_FILES* pSDSFileInfo, 
   SDS_FILTER*    pSDSFilter,
   INT64 fileCount);

// Only called when reading from shared memory
typedef void*(*SDS_READ_SHARED_MEMORY_CALLBACK) (SDS_SHARED_MEMORY_CALLBACK* pSDSMemoryCallback);

//examples PyArrayObject* AllocateNumpyArray(int ndim, npy_intp* dims, INT32 numpyType, INT64 itemsize, char* data, INT32 numpyFlags, npy_intp* strides) {
// Copy all information into pDestInfo
// Called at the beginning of reading a file
// if pData return as NULL, data is assume to be EXCLUDED
typedef void(*SDS_ALLOCATE_ARRAY_CALLBACK)(SDS_ALLOCATE_ARRAY* pSDSAllocateArray);

// Pass arrayobject returned from SDS_ALLOCATE_ARRAY_CALLBACK and return pointer to first location of array
// example: return PyArray_BYTES(pArrayObject, 0);
//typedef void*(*SDS_GET_ARRAY_POINTER_CALLBACK)(void * pArrayObject);

// example: void BeginAllowThreads() {
// return PyEval_SaveThread();
//   }
typedef void*(*SDS_BEGIN_ALLOW_THREADS)();

//example: void EndAllowThreads() {
//   PyEval_RestoreThread(_save);
//}
typedef void(*SDS_END_ALLOW_THREADS)(void*);

typedef void(*SDS_WARNING)(const char *format, ...);

struct SDS_READ_CALLBACKS {
   SDS_READ_FINAL_CALLBACK          ReadFinalCallback;
   SDS_STACK_FINAL_CALLBACK         StackFinalCallback;
   SDS_READ_SHARED_MEMORY_CALLBACK  ReadMemoryCallback;        // Only called for shared memory
   SDS_ALLOCATE_ARRAY_CALLBACK      AllocateArrayCallback;     // NULL now allowed?  needs testing

   SDS_BEGIN_ALLOW_THREADS          BeginAllowThreads;         // Must be set even if does nothing
   SDS_END_ALLOW_THREADS            EndAllowThreads;           // Must be set even if does nothing

   SDS_WARNING                      WarningCallback;           // To send warnings, may be null

   SDS_STRING_LIST*                 pInclusionList; // Set to NULL if nothing included
   SDS_STRING_LIST*                 pExclusionList; // Set to NULL if nothing excluded
   SDS_STRING_LIST*                 pFolderInclusionList; // Set to NULL if nothing excluded

   double                           ReserveSpace;           // From 0.0 to 1.0 -- percent to reserve

   bool                             MustExist;
   bool                             Dummy1;
   INT16                            Dummy2;
   // new for filtering
   SDS_FILTER                       Filter;

   // the output when concat
   const char*                      strOutputFilename;

};

struct SDS_READ_INFO {
   INT32 mode;       // = COMPRESSION_MODE_COMPRESS_FILE,
};


struct SDS_WRITE_CALLBACKS {
   SDS_BEGIN_ALLOW_THREADS          BeginAllowThreads;
   SDS_END_ALLOW_THREADS            EndAllowThreads;
};


struct SDS_WRITE_INFO {
   // arrays to save information
   SDSArrayInfo* aInfo;
   INT64 arrayCount;

   // meta information
   const char *metaData;
   UINT32 metaDataSize;

   // names of arrays information
   char* pListNames;
   INT64 listNameSize;    // total byte size (store in memory)
   INT64 listNameCount;   // number of names

                          // compressed or uncompressed
   INT32 mode;       // = COMPRESSION_MODE_COMPRESS_FILE,
   INT32 compType;   // = COMPRESSION_TYPE_ZSTD,
   INT32 level;      // = ZSTD_CLEVEL_DEFAULT;

   INT32 sdsFileType; // = SDS_FILE_TYPE_STRUCT
   INT32 sdsAuthorId; // = SDS_AUTHOR_IS

   BOOL  appendFileHeadersMode;  // True if appending to existing file
   BOOL  appendRowsMode;         // Appending rows only
   BOOL  appendColumnsMode;      // Appending columns only (code not written yet)
   BOOL  appendReserved1;

   INT64 bandSize;    // banding divides a column into chunks (set to 0 for no banding)

   const char* sectionName;
   INT64       sectionNameSize;
};

struct SDS_MULTI_READ {
   const char*          pFileName;
   SDS_STRING_LIST*     pFolderName;
   INT64                Index;
   SDS_FINAL_CALLBACK   FinalCallback;
};


// Export DLL section
#if defined(_WIN32) && !defined(__GNUC__)

#define DllExport __declspec(dllexport)


#else 

#define DllExport

#endif


extern "C" {
   //-------------------------------------------------
   // Main API to write SDS file
   // File only
   DllExport BOOL SDSWriteFile(
      const char *fileName,
      const char *shareName,  // can be NULL
      SDS_STRING_LIST *folderName,

                              // arrays to save information
      SDS_WRITE_INFO*         pWriteInfo,

      SDS_WRITE_CALLBACKS*    pWriteCallbacks);

   //---------------------------------------------
   // Main API to read SDS file
   // Can read or get infomration
   DllExport void* SDSReadFile(
      const char *fileName,
      const char *shareName,
      SDS_STRING_LIST *pFolderName,
      SDS_STRING_LIST *pSectionsName,    // can be NULL
      SDS_READ_INFO* pReadInfo,  // has mode= COMPRESSION_MODE_DECOMPRESS_FILE OR INFO MODE
      SDS_READ_CALLBACKS* pReadCallbacks);

   //---------------------------------------------
   DllExport void* SDSReadManyFiles(
      SDS_MULTI_READ*      pMultiRead,
      SDS_STRING_LIST*     pInclusionList,      // may be set to NULL
      SDS_STRING_LIST*     pFolderList,      // may be set to NULL
      SDS_STRING_LIST*     pSectionsName,    // can be NULL
      INT64                fileCount,
      int                  multiMode,           // see SDS_MULTI
      SDS_READ_CALLBACKS*  pReadCallbacks);


   DllExport char* SDSGetLastError();

   DllExport BOOL CloseSharedMemory(void* pMapStruct);

   DllExport BOOL CloseDecompressFile(void* pSDSDecompressFile);

   // no longer supported
   //DllExport void SDSClearBuffers();

   typedef BOOL(*SDS_WRITE_FILE)(const char*, const char*, SDS_WRITE_INFO*, SDS_WRITE_CALLBACKS*);
   typedef void*(*SDS_READ_FILE)(const char*, const char*, SDS_READ_INFO*, SDS_READ_CALLBACKS*);
   typedef void*(*SDS_READ_MANY_FILES)(SDS_MULTI_READ*, SDS_STRING_LIST*, INT64, int, SDS_READ_CALLBACKS*);
   typedef char*(*SDS_GET_LAST_ERROR)();
   typedef void(*SDS_CLEAR_BUFFERS)();
}

#ifdef SDS_SPECIAL_INCLUDE

//====================================================================
// Windows only special include
// TODO: Make this a function table that is loaded once
SDS_READ_FILE           g_fpReadFile = NULL;
SDS_READ_MANY_FILES     g_fpReadManyFiles = NULL;
SDS_WRITE_FILE          g_fpWriteFile = NULL;
SDS_GET_LAST_ERROR      g_fpGetLastError = NULL;
SDS_CLEAR_BUFFERS       g_fpClearBuffers = NULL;
HMODULE                 g_hModule = NULL;

void _LazyLoad(const char* dllLoadPath) {
   // If they pass in a path, try to load that first
   if (g_hModule == NULL && dllLoadPath != NULL) {
      g_hModule = LoadLibraryA(dllLoadPath);
   }
   if (g_hModule == NULL) {
      g_hModule = LoadLibraryW(L"SDSFile.dll");
   }
   if (g_hModule == NULL) {
      printf("Failed to load DLL file SDSFile.dll\n");
   }
}

//----------------------------------------------------
// For matlab which does not statically link to the DLL
// Load DLL on demand
// dllLoadPath may be null, it will try to load from normal paths
BOOL _SDSWriteFile(
   const char* dllLoadPath,
   const char *fileName,
   const char *shareName,  // can be NULL

                           // arrays to save information
   SDS_WRITE_INFO*         pWriteInfo,
   SDS_WRITE_CALLBACKS*    pWriteCallbacks) {

   // Lazy DLL load
   _LazyLoad(dllLoadPath);

   if (g_hModule != NULL) {

      if (g_fpWriteFile == NULL) {
         g_fpWriteFile = (SDS_WRITE_FILE)GetProcAddress(g_hModule, "SDSWriteFile");
      }

      if (g_fpWriteFile != NULL) {
         return g_fpWriteFile(fileName, shareName, pWriteInfo, pWriteCallbacks);
      }
   }
   return FALSE;
}

//===================================================================
void* _SDSReadFile(
   const char* dllLoadPath,
   const char *fileName,
   const char *shareName,
   SDS_READ_INFO*   pReadInfo,  // = COMPRESSION_MODE_DECOMPRESS_FILE OR INFO MODE
   SDS_READ_CALLBACKS* pReadCallbacks) {

   // Lazy DLL load
   _LazyLoad(dllLoadPath);

   if (g_hModule != NULL) {

      if (g_fpReadFile == NULL) {
         g_fpReadFile = (SDS_READ_FILE)GetProcAddress(g_hModule, "SDSReadFile");
      }

      if (g_fpReadFile != NULL) {
         return g_fpReadFile(fileName, shareName, pReadInfo, pReadCallbacks);
      }
   }
   return NULL;

}


//===================================================================
void* _SDSReadManyFiles(
   const char* dllLoadPath,
   SDS_MULTI_READ*      pMultiRead,
   SDS_STRING_LIST*     pInclusionList,      // may be set to NULL
   INT64                fileCount,
   int                  multiMode,           // see SDS_MULTI
   SDS_READ_CALLBACKS*  pReadCallbacks) {

   // Lazy DLL load
   _LazyLoad(dllLoadPath);

   if (g_hModule != NULL) {

      if (g_fpReadManyFiles == NULL) {
         g_fpReadManyFiles = (SDS_READ_MANY_FILES)GetProcAddress(g_hModule, "SDSReadManyFiles");
      }

      if (g_fpReadManyFiles != NULL) {
         return g_fpReadManyFiles(pMultiRead, pInclusionList, fileCount, multiMode, pReadCallbacks);
      }
   }
   return NULL;

}

char* _SDSGetLastError(
   const char* dllLoadPath) {

   // Lazy DLL load
   _LazyLoad(dllLoadPath);

   if (g_hModule != NULL) {

      if (g_fpGetLastError == NULL) {
         g_fpGetLastError = (SDS_GET_LAST_ERROR)GetProcAddress(g_hModule, "SDSGetLastError");
      }

      if (g_fpGetLastError != NULL) {
         return g_fpGetLastError();
      }
   }
   return NULL;
}

void _SDSClearBuffers(const char* dllLoadPath) {
   // Lazy DLL load
   _LazyLoad(dllLoadPath);

   if (g_hModule != NULL) {

      if (g_fpClearBuffers == NULL) {
         g_fpClearBuffers = (SDS_CLEAR_BUFFERS)GetProcAddress(g_hModule, "SDSClearBuffers");
      }

      if (g_fpClearBuffers != NULL) {
         return g_fpClearBuffers();
      }
   }

}
#endif