#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include "CommonInc.h"

#include "zstd.h"
//#include "Compress.h"
#include "SDSFile.h"
#include "SharedMemory.h"
#include "FileReadWrite.h"

#include "MathWorker.h"
#include <stdarg.h>
#include <vector>
#include <set>
#include <unordered_map>
#include <string.h>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wempty-body"
#endif

//found in riptide.h
extern uint64_t GetUTCNanos();
extern int64_t SumBooleanMask(const int8_t* pIn, int64_t length);

//#define MATLAB_MODE 1

//-----------------------------------------------
// SDS File Format
// First 512 bytes - SDS_FILE_HEADER
// 
// --- SDS_FILE_HEADER
//   - NameBlock ptr    (list of array names and types) (variable lengths)
//   - MetaBlock ptr    (meta data json string often compressed) (one string block often compressed)
//   - ArrayBlock ptr   (up to 5 dimensional numpy array with strides/itemsize/dtype) (fixed SIZE)
//   - BandBlock ptr (optional)  --> if BandSize is set (compression bands) (variable length based on how many bands)
//   - SectionBlock ptr (optional - used when appending)   (list of appendname/SDS_FILE_HEADER offset)
//     If a section was appended there is a DIRECTORY of sectionnames/offset to SDS_FILE_HEADERs
//     this acts like multiple files in one file
// TIMESTAMP added in ver 4.5 written when file is done
//

#define SDS_MAX_CORES 65

#define LOGGING(...)
//#define LOGGING printf

#define LOG_THREAD(...)
//#define LOG_THREAD printf

//===========================================
// Globally keep track of gateway list
std::vector<std::string>   g_gatewaylist;

//ZSTD_DCtx* g_DecompressContext[SDS_MAX_CORES] = {
//   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
//   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
//   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
//   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
//   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
//   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
//   NULL, NULL, NULL, NULL, NULL };
//

#ifdef MATLAB_MODE
//-------------------------------------------------------------------------
// CHANGE TABLE FOR MATLAB
//int64_t default1 = -9223372036854775808L;
static int64_t  gDefaultInt64 = 0;
static int32_t  gDefaultInt32 = 0;
static uint16_t gDefaultInt16 = 0;
static uint8_t  gDefaultInt8 = 0;

static uint64_t gDefaultUInt64 = 0;
static uint32_t gDefaultUInt32 = 0;
static uint16_t gDefaultUInt16 = 0;
static uint8_t  gDefaultUInt8 = 0;

static float  gDefaultFloat = std::numeric_limits<float>::quiet_NaN();
static double gDefaultDouble = std::numeric_limits<double>::quiet_NaN();
static long double gDefaultLongDouble = std::numeric_limits<long double>::quiet_NaN();
static int8_t   gDefaultBool = 0;
static char   gString[1024] = { 0,0,0,0 };
#else
static int64_t  gDefaultInt64 = 0x8000000000000000;
static int32_t  gDefaultInt32 = 0x80000000;
static uint16_t gDefaultInt16 = 0x8000;
static uint8_t  gDefaultInt8 = 0x80;

static uint64_t gDefaultUInt64 = 0xFFFFFFFFFFFFFFFF;
static uint32_t gDefaultUInt32 = 0xFFFFFFFF;
static uint16_t gDefaultUInt16 = 0xFFFF;
static uint8_t  gDefaultUInt8 = 0xFF;

static float  gDefaultFloat = std::numeric_limits<float>::quiet_NaN();
static double gDefaultDouble = std::numeric_limits<double>::quiet_NaN();
static long double gDefaultLongDouble = std::numeric_limits<long double>::quiet_NaN();
static int8_t   gDefaultBool = 0;
static char   gString[1024] = { 0,0,0,0 };

#endif

//----------------------------------------------------
// returns pointer to a data type (of same size in memory) that holds the invalid value for the type
// does not yet handle strings
void* SDSGetDefaultType(int32_t numpyInType) {
   void* pgDefault = &gDefaultInt64;

   switch (numpyInType) {
   case SDS_FLOAT:  pgDefault = &gDefaultFloat;
      break;
   case SDS_DOUBLE: pgDefault = &gDefaultDouble;
      break;
   case SDS_LONGDOUBLE: pgDefault = &gDefaultLongDouble;
      break;
      // bool should not really have a default type
   case SDS_BOOL:   pgDefault = &gDefaultBool;
      break;
   case SDS_BYTE:   pgDefault = &gDefaultInt8;
      break;
   case SDS_SHORT:  pgDefault = &gDefaultInt16;
      break;
   case SDS_INT:       pgDefault = &gDefaultInt32;
      break;
   case SDS_LONG:      pgDefault = &gDefaultInt32;  // ambiguous
      break;
   case SDS_LONGLONG:  pgDefault = &gDefaultInt64;
      break;
   case SDS_UBYTE:     pgDefault = &gDefaultUInt8;
      break;
   case SDS_USHORT:    pgDefault = &gDefaultUInt16;
      break;
   case SDS_UINT:      pgDefault = &gDefaultUInt32;
      break;
   case SDS_ULONG:     pgDefault = &gDefaultUInt32;  // ambiguous
      break;
   case SDS_ULONGLONG: pgDefault = &gDefaultUInt64;
      break;
   case SDS_STRING:    pgDefault = &gString;
      break;
   case SDS_UNICODE:   pgDefault = &gString;
      break;
   default:
      printf("!!! likely problem in SDSGetDefaultType\n");
   }

   return pgDefault;
}

//===========================================
// Buffer filled in when there is an error
char g_errorbuffer[512] = { 0 };
int32_t  g_lastexception = 0;

//-----------------------------------------------------
// platform independent error storage
void SetErr_Format(
   int32_t   exception,
   const char *format,
   ...
) {
   // NOTE: What about stacking multiple errors?
   g_lastexception = exception;
   va_list args;
   va_start(args, format);

   vsnprintf(g_errorbuffer, sizeof(g_errorbuffer), format, args);
   va_end(args);

   LOGGING(g_errorbuffer);
   LOGGING("\n");
}

// Call to clear any previous errors
static void ClearErrors() {
   LOGGING("Clearing errors\n");
   g_lastexception = 0;
   g_errorbuffer[0] = 0;
}

static void PrintIfErrors() {
   if (g_errorbuffer[0] != 0) {
      printf("Existing error: %s\n", g_errorbuffer);
   }
}

//------------------------------------------------------------
static size_t CompressGetBound(int32_t compMode, size_t srcSize) {
   if (compMode == COMPRESSION_TYPE_ZSTD) {
      return ZSTD_compressBound(srcSize);
   }

   printf("!!internal CompressBound error\n");
   return srcSize;
}


//------------------------------------------------------------
static size_t CompressData(int32_t compMode, void* dst, size_t dstCapacity,
   const void* src, size_t srcSize,
   int32_t compressionLevel) {

   if (compMode == COMPRESSION_TYPE_ZSTD) {
      return ZSTD_compress(dst, dstCapacity, src, srcSize, compressionLevel);
   }

   printf("!!internal CompressData error\n");
   return srcSize;

}

size_t ZSTD_decompress_stackmode(void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
   ///* stack mode */
   //ZSTD_DCtx dctx;
   //ZSTD_initDCtx_internal(&dctx);
   //return ZSTD_decompressDCtx(&dctx, dst, dstCapacity, src, srcSize);

   size_t regenSize;
   ZSTD_DCtx* const dctx = ZSTD_createDCtx();
   if (dctx == NULL) return -64; // ERROR(memory_allocation);
   regenSize = ZSTD_decompressDCtx(dctx, dst, dstCapacity, src, srcSize);
   ZSTD_freeDCtx(dctx);
   return regenSize;

}


//------------------------------------------------------------
static size_t DecompressData(ZSTD_DCtx* pDecompContext, int32_t compMode, void* dst, size_t dstCapacity, const void* src, size_t srcSize) {

   if (pDecompContext) {
      return ZSTD_decompressDCtx(pDecompContext, dst, dstCapacity, src, srcSize);
   }
   else {
      return ZSTD_decompress(dst, dstCapacity, src, srcSize);

   }

   //if (core >= 0 && core < SDS_MAX_CORES) {

   //   if (g_DecompressContext[core] == NULL) {
   //      g_DecompressContext[core] = ZSTD_createDCtx();
   //   }
   //   size_t regenSize = ZSTD_decompressDCtx(g_DecompressContext[core], dst, dstCapacity, src, srcSize);
   //   return regenSize;
   //}
   //return -1;
}


//------------------------------------------------------------
// Returns <0 for error
// else return bytes left
static size_t DecompressDataPartial(int32_t core, int32_t compMode, void* dst, size_t dstCapacity, const void* src, size_t srcSize) {

   if (core >= 0 && core < SDS_MAX_CORES) {
      ZSTD_DCtx* const dctx = ZSTD_createDCtx();
      size_t dstPos = 0;
      size_t srcPos = 0;
      ZSTD_outBuffer output = { dst, dstCapacity, dstPos };
      ZSTD_inBuffer  input = { src, srcSize, srcPos };
      /* ZSTD_compress_generic() will check validity of dstPos and srcPos */

      size_t cErr = 0;

      do {
         cErr = ZSTD_decompressStream(dctx, &output, &input);
         if (cErr < 0) break;
      } while (input.pos < input.size && output.pos < output.size);

      if (cErr >= 0) {
         cErr = output.pos;
      }

      ZSTD_freeDCtx(dctx);
      return cErr;


      //ZSTD_DStream* decomp=  ZSTD_createDStream();
      //ZSTD_CStream* comp = ZSTD_createCStream();

      //ZSTD_DCtx* const dctx = ZSTD_createDCtx();

      //size_t regenSize = ZSTD_decompressDCtx(g_DecompressContext[core], dst, dstCapacity, src, srcSize);

      //ZSTD_freeCStream(comp);
      //ZSTD_freeDStream(decomp);

      //return regenSize;
   }
   return -1;
}


//------------------------------------------------------------
static bool CompressIsError(int32_t compMode, size_t code) {
   if (ZSTD_isError(code)) {
      SetErr_Format(SDS_VALUE_ERROR, "Decompression error: %s", ZSTD_getErrorName(code));
      return TRUE;
   }
   else {
      return FALSE;
   }

}

// check to see if any errors were recorded
// Returns TRUE if there was an error
//static bool CheckErrors() {
//
//   if (g_lastexception) {
//      return TRUE;
//   }
//   return FALSE;
//}



#define MAX_READSIZE_ALLOWED  2000000000

//-----------------------------------------------------

#if defined(_WIN32)

char g_errmsg[512];

//----------------------------------------------------------
//
const char* GetLastErrorMessage(
   char* errmsg = NULL,
   DWORD last_error = 0)
{
   if (errmsg == NULL) {
      errmsg = g_errmsg;
   }

   if (last_error == 0) {
      last_error = GetLastError();
   }

   bool stripTrailingLineFeed = true;

   if (!FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      0,
      last_error,
      0,
      errmsg,
      511,
      NULL))
   {
      // if we fail, call ourself to find out why and return that error

      const DWORD thisError = ::GetLastError();

      if (thisError != last_error)
      {
         return GetLastErrorMessage(errmsg, thisError);
      }
      else
      {
         // But don't get into an infinite loop...

         return "Failed to obtain error string";
      }
   }

   if (stripTrailingLineFeed)
   {
      const size_t length = strlen(errmsg);

      if (errmsg[length - 1] == '\n')
      {
         errmsg[length - 1] = 0;

         if (errmsg[length - 2] == '\r')
         {
            errmsg[length - 2] = 0;
         }
      }
   }

   return errmsg;
}



SDS_EVENT_HANDLE SDS_CREATE_EVENTHANDLE() {
   return CreateEvent(NULL, TRUE, FALSE, NULL);
}
void SDS_DESTROY_EVENTHANDLE(SDS_EVENT_HANDLE handle) {
   CloseHandle(handle);
}


//---------------------------------------------------
// returns < 0 if file does not exist or error
int64_t SDSFileSize(const char* fileName) {
   WIN32_FILE_ATTRIBUTE_DATA  fileInfo;

   if (GetFileAttributesEx(fileName, GetFileExInfoStandard, &fileInfo)) {
      LARGE_INTEGER li;
      
      li.LowPart = fileInfo.nFileSizeLow;
      li.HighPart = fileInfo.nFileSizeHigh;
      return li.QuadPart;
   }
   return -1;
}

//---------------------------------------------------
// Returns NULL on failure otherwise valid handle
SDS_FILE_HANDLE SDSFileOpen(const char* fileName, bool writeOption, bool overlapped, bool directIO, bool appendOption)
{
   bool WriteOption = writeOption;
   bool Overlapped = true; // overlapped;
   bool DirectIO = directIO;

   int32_t filemode = OPEN_EXISTING;

   if (WriteOption) {
      if (appendOption) {
         filemode = OPEN_ALWAYS;
      }
      else {
         // will overwrite existing file
         filemode = CREATE_ALWAYS;
      }
   }

   // open the existing file for reading       
   SDS_FILE_HANDLE Handle = CreateFile
   (
      fileName,
      GENERIC_READ | (WriteOption ? GENERIC_WRITE : 0),
      FILE_SHARE_READ | FILE_SHARE_WRITE,
      0,
      filemode,

      FILE_ATTRIBUTE_NORMAL |
      FILE_FLAG_SEQUENTIAL_SCAN |
      (Overlapped ? FILE_FLAG_OVERLAPPED : 0) |
      (DirectIO ? FILE_FLAG_NO_BUFFERING : 0),

      0
   );

   if (Handle != INVALID_HANDLE_VALUE)
   {
      if (Handle != NULL)
      {
         // good
         return Handle;
      }
      else
      {
         Handle = (void*)0;
         return Handle;
      }
   }
   Handle = (void*)0;
   return Handle;
}

//---------------------------------------------------
void SDSFileSeek(SDS_FILE_HANDLE handle, int64_t pos) {

   int64_t result = 0;

   LARGE_INTEGER temp;
   temp.QuadPart = pos;

   bool bResult =
      SetFilePointerEx(handle, temp, (PLARGE_INTEGER)&result, SEEK_SET);

   if (!bResult)
   {
      LogError("!!! Seek current to %llu failed!", pos);
   }

}


//---------------------------------------------------
// Returns bytes read
int64_t SDSFileReadChunk(SDS_EVENT_HANDLE eventHandle, SDS_FILE_HANDLE Handle, void* buffer, int64_t bufferSize, int64_t BufferPos) {
   //LogError("!! Suspicious code path for async read %s %d", FileName, LastError);
   OVERLAPPED OverlappedIO;

   OverlappedIO.hEvent = eventHandle;
   OverlappedIO.InternalHigh = 0;
   OverlappedIO.Internal = 0;
   OverlappedIO.OffsetHigh = (uint32_t)(BufferPos >> 32);
   OverlappedIO.Offset = (uint32_t)BufferPos;

   bool bReadDone;

   OVERLAPPED* pos = &OverlappedIO;
   DWORD n;

   if (bufferSize > MAX_READSIZE_ALLOWED) {
      //printf("!!!read buffer size too large %lld\n", bufferSize);
      // break it down

      int64_t totalRead = 0;
      char* cbuffer = (char*)buffer;

      while (bufferSize > MAX_READSIZE_ALLOWED) {
         totalRead += SDSFileReadChunk(eventHandle, Handle, cbuffer, MAX_READSIZE_ALLOWED, BufferPos);

         cbuffer += MAX_READSIZE_ALLOWED;
         BufferPos += MAX_READSIZE_ALLOWED;
         bufferSize -= MAX_READSIZE_ALLOWED;
      }

      if (bufferSize) {
         totalRead += SDSFileReadChunk(eventHandle, Handle, cbuffer, bufferSize, BufferPos);
      }

      return totalRead;
   }
   else {

      DWORD count = (DWORD)bufferSize;
      bReadDone = ReadFile(Handle, buffer, count, &n, pos);

      DWORD LastError = GetLastError();
      //if (!bReadDone) {
      //   printf("read not done %d\n", LastError);
      //}

      if (!bReadDone && LastError == ERROR_IO_PENDING)
      {
         // Wait for IO to complete
         bReadDone = GetOverlappedResult(Handle, pos, &n, true);

         if (!bReadDone)
         {
            LastError = GetLastError();
            printf("!!Read failed in getoverlapped %p %p %p %d\n", (void*)Handle, (void*)eventHandle, buffer, LastError);
            return 0;
         }
         else {
            bool extraCheck = HasOverlappedIoCompleted(pos);
            if (!extraCheck) {
               printf("!! internal error reading... complete but not really\n");
            }
         }
      }
   }

   if (!bReadDone)
   {
      DWORD LastError = GetLastError();
      printf("!!Read failed at end %p %p %p %d\n", (void*)Handle, (void*)eventHandle, buffer, LastError);
      return 0;
   }
   //printf("Read %lld bytes\n",(long long) n);
   return n;
}


//---------------------------------------------------
// Returns bytes read
int64_t SDSFileWriteChunk(SDS_EVENT_HANDLE eventHandle, SDS_FILE_HANDLE Handle, void* buffer, int64_t bufferSize, int64_t BufferPos) {
   //LogError("!! Suspicious code path for async read %s %d", FileName, LastError);
   OVERLAPPED OverlappedIO;

   OverlappedIO.hEvent = eventHandle;
   OverlappedIO.InternalHigh = 0;
   OverlappedIO.Internal = 0;
   OverlappedIO.OffsetHigh = (uint32_t)(BufferPos >> 32);
   OverlappedIO.Offset = (uint32_t)BufferPos;

   OVERLAPPED* pos = &OverlappedIO;
   DWORD n;

   if (bufferSize > MAX_READSIZE_ALLOWED) {
      int64_t totalWritten = 0;
      char* cbuffer = (char*)buffer;

      while (bufferSize > MAX_READSIZE_ALLOWED) {
         totalWritten += SDSFileWriteChunk(eventHandle, Handle, cbuffer, MAX_READSIZE_ALLOWED, BufferPos);

         cbuffer += MAX_READSIZE_ALLOWED;
         BufferPos += MAX_READSIZE_ALLOWED;
         bufferSize -= MAX_READSIZE_ALLOWED;
      }

      if (bufferSize) {
         totalWritten += SDSFileWriteChunk(eventHandle, Handle, cbuffer, bufferSize, BufferPos);
      }

      return totalWritten;
   }

   DWORD count = (DWORD)bufferSize;
   bool bWriteDone = WriteFile(Handle, buffer, count, &n, pos);

   DWORD LastError = GetLastError();
   if (!bWriteDone && LastError == ERROR_IO_PENDING)
   {
      // Wait for IO to complete
      bWriteDone = GetOverlappedResult(Handle, pos, &n, true);

      if (!bWriteDone)
      {
         LastError = GetLastError();
         printf("!!Write failed ovr buff:%p  size:%lld  pos:%lld  error:%d\n", buffer, bufferSize, BufferPos, LastError);
         return 0;
      }
   }

   if (!bWriteDone)
   {
      LastError = GetLastError();
      printf("!!Write failed done  buff:%p  size:%lld  pos:%lld  error:%d\n", buffer, bufferSize, BufferPos, LastError);
      return 0;
   }

   if (n != count) {
      LastError = GetLastError();
      printf("write chunk error  buff:%p  size:%lld  pos:%lld  error:%d\n", buffer, bufferSize, BufferPos, LastError);
   }
   // TODO: n not always filled in... due to delayed write?
   return n;
}

//---------------------------------------------------
void SDSFileClose(SDS_FILE_HANDLE handle) {
   LOGGING("File closed %p\n", handle);
   CloseHandle(handle);
}

#else

#include <errno.h>

const char* GetLastErrorMessage() {
   return (const char*)strerror(errno);
}


SDS_EVENT_HANDLE SDS_CREATE_EVENTHANDLE() {
   // todo: store information for ftruncate
   return NULL;
}
void SDS_DESTROY_EVENTHANDLE(SDS_EVENT_HANDLE handle) {
}

#include <sys/stat.h>

// returns < 0 if file does not exist or error
int64_t SDSFileSize(const char* fileName) {
   struct stat statbuf;
   if (stat(fileName, &statbuf) < 0) {
      return -1;
   }
   return statbuf.st_size;
}

//---------------------------------------------------
SDS_FILE_HANDLE SDSFileOpen(const char* fileName, bool writeOption, bool overlapped, bool directIO, bool appendOption)
{

   errno = 0;
   SDS_FILE_HANDLE filehandle = 0;

   int32_t createFlags = 0;
   if (writeOption) {
      if (appendOption) {
         // NOTE: possibly try first without O_CREAT
         createFlags = O_RDWR;
         filehandle = open(fileName, createFlags, 0666);

         if (filehandle < 0) {
            LOGGING("openning with CREAT\n");
            createFlags = O_RDWR | O_CREAT;
            filehandle = open(fileName, createFlags, 0666);
         }
      }
      else {
         createFlags = O_WRONLY | O_CREAT;
         filehandle = open(fileName, createFlags, 0666);
      }
   }
   else {
      createFlags = O_RDONLY;
      filehandle = open(fileName, createFlags);
   }

   LOGGING("linux handle open\n");
   if (filehandle < 0) {
      printf("error opening file %s -- error %s\n", fileName, strerror(errno));
      return BAD_SDS_HANDLE;
   }
   return filehandle;
}

//---------------------------------------------------
void SDSFileSeek(SDS_FILE_HANDLE handle, int64_t pos) {

   // not used
   return;
}

//---------------------------------------------------
// Returns bytes read
int64_t SDSFileReadChunk(SDS_EVENT_HANDLE eventHandle, SDS_FILE_HANDLE fileHandle, void* buffer, int64_t bufferSize, int64_t bufferPos) {

   if (bufferSize > MAX_READSIZE_ALLOWED) {
      //printf("!!!read buffer size too large %lld\n", bufferSize);
      // break it down

      int64_t totalRead = 0;
      int64_t origSize = bufferSize;
      char* cbuffer = (char*)buffer;

      while (bufferSize > MAX_READSIZE_ALLOWED) {
         totalRead += SDSFileReadChunk(eventHandle, fileHandle, cbuffer, MAX_READSIZE_ALLOWED, bufferPos);

         cbuffer += MAX_READSIZE_ALLOWED;
         bufferPos += MAX_READSIZE_ALLOWED;
         bufferSize -= MAX_READSIZE_ALLOWED;
      }

      if (bufferSize) {
         totalRead += SDSFileReadChunk(eventHandle, fileHandle, cbuffer, bufferSize, bufferPos);
      }

      if (totalRead != origSize) {
         printf("!!readchunk failed for fd %d -- %lld vs %lld (errno %d)\n", fileHandle, origSize, totalRead, errno);
         return 0;
      }
      return totalRead;
   }
   else {
      ssize_t bytes_read = pread(fileHandle, buffer, (size_t)bufferSize, bufferPos);

      // pread() returns -1 on error; make sure the read actually succeeded.
      if (bytes_read == -1)
      {
         printf("!!readchunk failed for fd %d -- %lld vs %lld (errno %d)\n", fileHandle, bufferSize, bytes_read, errno);
         return 0;
      }

      if (bytes_read != bufferSize) {
         printf("!!readchunk failed for fd %d -- %lld vs %lld (errno %d)\n", fileHandle, bufferSize, bytes_read, errno);
         return 0;
      }
      return bytes_read;
   }
}

//---------------------------------------------------
// Returns bytes read
int64_t SDSFileWriteChunk(SDS_EVENT_HANDLE eventHandle, SDS_FILE_HANDLE fileHandle, void* buffer, int64_t bufferSize, int64_t bufferPos) {

   if (bufferSize > MAX_READSIZE_ALLOWED) {
      int64_t totalWritten = 0;
      int64_t origSize = bufferSize;
      char* cbuffer = (char*)buffer;

      while (bufferSize > MAX_READSIZE_ALLOWED) {
         totalWritten += SDSFileWriteChunk(eventHandle, fileHandle, cbuffer, MAX_READSIZE_ALLOWED, bufferPos);

         cbuffer += MAX_READSIZE_ALLOWED;
         bufferPos += MAX_READSIZE_ALLOWED;
         bufferSize -= MAX_READSIZE_ALLOWED;
      }

      if (bufferSize) {
         totalWritten += SDSFileWriteChunk(eventHandle, fileHandle, cbuffer, bufferSize, bufferPos);
      }

      if (totalWritten != origSize) {
         printf("write chunk error  buff:%p  size:%lld  pos:%lld  errno:%d\n", buffer, bufferSize, bufferPos, errno);
         return 0;
      }

      return totalWritten;
   }
   else {
      ssize_t bytes_written = pwrite(fileHandle, buffer, (size_t)bufferSize, bufferPos);

      // pwrite() returns -1 on error; make sure the write actually succeeded.
      if (bytes_written == -1)
      {
         printf("!!Write failed done  buff:%p  size:%lld  pos:%lld  errno:%d\n", buffer, bufferSize, bufferPos, errno);
         return 0;
      }

      if (bytes_written != bufferSize) {
         printf("!!Write failed small buff:%p  size:%lld  pos:%lld  errno:%d\n", buffer, bufferSize, bufferPos, errno);
         return 0;
      }

      return bytes_written;
   }
}

//---------------------------------------------------
void SDSFileClose(SDS_FILE_HANDLE handle) {
   errno = 0;
   int32_t result = close(handle);
   if (result < 0) {
      printf("Error closing file %s\n", strerror(errno));
   }
}

#endif


//==========================================================================================
// In windows when shared memory is created, we need to keep track of all the ref counts
// In linux also?
typedef std::unordered_map<std::string, PMAPPED_VIEW_STRUCT> SHARED_MEMORY_STDMAP;
static SHARED_MEMORY_STDMAP g_SMMap;


//------------------------------------------------------------------------------------------
//
void AddSharedMemory(const char* name, PMAPPED_VIEW_STRUCT pmvs, void* pointer) {
   //printf("in add shared memory\n");

   std::string sname = std::string(name);
   if (g_SMMap.find(sname) == g_SMMap.end()) {
      // not found
      g_SMMap[sname] = pmvs;

      // init ref count
      pmvs->RefCount = 1;
   }
   else {
      // increment reference
      g_SMMap[sname]->RefCount++;
   }
}


//------------------------------------------------------------------------------------------
//
void DelSharedMemory(void* pBase, int64_t length) {
   auto it = g_SMMap.begin();

   while (it != g_SMMap.end())
   {
      PMAPPED_VIEW_STRUCT pmvs = it->second;
      // TODO: check range
      it++;
   }

   // when found g_SMMap.erase(name)
}

//==========================================================================================


typedef SDS_EVENT_HANDLE(*SDS_CreateEventHandle)();
typedef void(*SDS_DestroyEventHandle)(SDS_EVENT_HANDLE handle);
typedef int64_t(*SDS_FileSize)(const char* fileName);
typedef SDS_FILE_HANDLE(*SDS_FileOpen)(const char* fileName, bool writeOption, bool overlapped, bool directIO, bool appendOption);
typedef int64_t(*SDS_FileReadChunk)(SDS_EVENT_HANDLE eventHandle, SDS_FILE_HANDLE Handle, void* buffer, int64_t bufferSize, int64_t BufferPos);
typedef int64_t(*SDS_FileWriteChunk)(SDS_EVENT_HANDLE eventHandle, SDS_FILE_HANDLE fileHandle, void* buffer, int64_t bufferSize, int64_t bufferPos);
typedef void(*SDS_FileClose)(SDS_FILE_HANDLE handle);

class  SDSFileIO {
public:
   SDS_CreateEventHandle    CreateEventHandle = SDS_CREATE_EVENTHANDLE;
   SDS_DestroyEventHandle   DestroyEventHandle = SDS_DESTROY_EVENTHANDLE;
   SDS_FileSize             FileSize = SDSFileSize;
   SDS_FileOpen             FileOpen = SDSFileOpen;
   SDS_FileReadChunk        FileReadChunk = SDSFileReadChunk;
   SDS_FileWriteChunk       FileWriteChunk = SDSFileWriteChunk;
   SDS_FileClose            FileClose = SDSFileClose;
} DefaultFileIO;

typedef HRESULT(*SDS_SharedMemoryBegin)(
   const char*              pMappingName,
   int64_t                Size,
   PMAPPED_VIEW_STRUCT *pReturnStruct);

typedef HRESULT(*SDS_SharedMemoryEnd)(
   PMAPPED_VIEW_STRUCT  pMappedViewStruct);

typedef HRESULT(*SDS_SharedMemoryCopy)(
   const char*          pMappingName,
   PMAPPED_VIEW_STRUCT *pReturnStruct,
   bool                 bTest);

class SharedMemory {

public:
   const SDS_SharedMemoryBegin   SharedMemoryBegin = UtilSharedMemoryBegin;
   const SDS_SharedMemoryEnd     SharedMemoryEnd = UtilSharedMemoryEnd;
   const SDS_SharedMemoryCopy    SharedMemoryCopy = UtilSharedMemoryCopy;
   const SDS_FileReadChunk       FileReadChunk = SDSFileReadChunk;

   char                    SharedMemoryName[SDS_MAX_FILENAME] = { 0 };
   PMAPPED_VIEW_STRUCT     pMapStruct = NULL;
   int64_t                   SharedMemorySize = 0;

   //---------------------- METHODS ---------------------------
   //----------------------------------------------------------
   SDS_FILE_HEADER*   GetFileHeader() {
      if (pMapStruct) {
         return (SDS_FILE_HEADER*)pMapStruct->BaseAddress;
      }
      printf("!!internal shared memory error\n");
      return NULL;
   }

   //--------------------------------------------------
   //
   char*   GetMemoryOffset(int64_t offset) {
      if (pMapStruct) {
         return ((char*)pMapStruct->BaseAddress) + offset;
      }
      printf("!!internal shared memory GetMemoryOffset error\n");
      return NULL;
   }

   //--------------------------------------------------
   //
   SDS_ARRAY_BLOCK*  GetArrayBlock(int64_t arrayNum) {
      SDS_ARRAY_BLOCK* pArrayBlock = (SDS_ARRAY_BLOCK*)GetMemoryOffset(GetFileHeader()->ArrayBlockOffset);
      return &pArrayBlock[arrayNum];
   }

   //--------------------------------------------------
   //
   HRESULT Begin(
      int64_t                Size) {

      LOGGING("Allocating mem share %s with size %lld\n", SharedMemoryName, Size);

      HRESULT hr =
         SharedMemoryBegin(SharedMemoryName, Size, &pMapStruct);

      if (hr < 0) {
         printf("!!!Failed to allocate shared memory share %s with size %lld\n", SharedMemoryName, Size);
         pMapStruct = NULL;
      }
      else {
         AddSharedMemory(SharedMemoryName, pMapStruct, NULL);
      }

      SharedMemorySize = Size;
      return hr;
   }

   //----------------------------------------------
   // hr < 0 on failure
   HRESULT MapExisting() {
      HRESULT hr =
         SharedMemoryCopy(SharedMemoryName, &pMapStruct, TRUE);

      if (hr > 0) {
         AddSharedMemory(SharedMemoryName, pMapStruct, NULL);
      }

      return hr;
   }

   void Destroy() {
      SharedMemoryName[0] = 0;
      if (pMapStruct) {
         SharedMemoryEnd(pMapStruct);
      }
      pMapStruct = NULL;
   }

   //-----------------------------------------------
   // Derives a sharename from a filename
   // NOTE: Often called first before creating memory share
   // On return SharedMemoryName is set
   void MakeShareName(const char* pszFilename, const char* shareName) {
      //
      // Break off any \ or : or / in the file
      //
      // Search for the last one
      //
      {
         const char* pTemp = pszFilename;
         const char* pMappingName = pTemp;

         while (*pTemp != 0) {

            if (*pTemp == '\\' ||
               *pTemp == ':' ||
               *pTemp == '/') {

               pMappingName = pTemp + 1;
            }

            pTemp++;
         }

         LOGGING("Orig filename:%s -- mapping: %s\n", pszFilename, pMappingName);

         char* pStart = SharedMemoryName;

#if defined(_WIN32)
         const char* pPrefix = "Global\\";
#else
         const char* pPrefix = "Global_";
#endif
         // copy over prefix
         while ((*pStart++ = *pPrefix++));

         // back over zero
         pStart--;

         // copy over shareName
         while ((*pStart++ = *shareName++));

         // back over zero and add bang
         pStart--;
         *pStart++ = '!';

         // copy over filename part
         while ((*pStart++ = *pMappingName++));
         LOGGING("Shared memory name is %s\n", SharedMemoryName);
      }

   }


} DefaultMemoryIO;




class SDSIncludeExclude {
   // keep track of inclusion/exclusion list
   std::unordered_map<std::string, int>     InclusionList;
   const char BANG_CHAR = '!';
   char SEP_CHAR = 0;
public:
   //-------------------------------------------------------------
   // Tries to find the '!' char in the string
   // if it finds the ! it returns the location
   // 
   const char* FindSep(const char *pString, char SEPARATOR) {
      while (*pString) {
         if (*pString == SEPARATOR) return pString;
         pString++;
      }
      return NULL;
   }

   const char* FindBackwardSep(const char *pString, char SEPARATOR) {
      const char* pStart = pString;
      while (*pString) {
         pString++;
      }
      while (pString > pStart) {
         pString--;
         if (*pString == SEPARATOR) return pString;
      }
      return NULL;
   }

   void ClearLists() {
      InclusionList.clear();
   }

   bool IsEmpty() {
      return InclusionList.empty();
   }

   void AddItem(const char* stritem) {
      std::string item = std::string(stritem);
      if (InclusionList.find(item) == InclusionList.end())  InclusionList.emplace(item, 1);
   }

   void AddItems(std::vector<const char*>* pItems, const char sep) {

      SEP_CHAR = sep;
      // loop over all included items
      for (const char* includeItem : *pItems) {
         LOGGING("Including item: %s\n", includeItem);
         AddItem(includeItem);
      }
   }

   // Returns 1 if item included
   int32_t IsIncluded(const char* stritem) {

      // If we have no inclusion list, then every item is accepted
      if (InclusionList.empty()) return 1;

      // if this is a column check
      if (SEP_CHAR == 0) {

         // Now check for '/' from onefile (?? should we check for onefile in fileheader first?)
         char* pHasSep = (char*)FindBackwardSep(stritem, '/');
         if (pHasSep && pHasSep != stritem) {
            // if we find a separator, trim the name
            stritem = pHasSep + 1;
         }
      }

      std::string item = std::string(stritem);

      // Check if we matched
      if (InclusionList.find(item) == InclusionList.end()) {

         // Failed to match, but maybe the first part matches
         if (SEP_CHAR == 0) {

            // The pArrayName might be a categorical column
            // If so, it will be in the format categoricalname!col0
            char* pHasBang = (char*)FindSep(stritem, BANG_CHAR);

            if (pHasBang) {
               LOGGING("Failed to match, has bang %s\n", item.c_str());

               // temp remove bang
               *pHasBang = 0;

               std::string newitem = std::string(stritem);

               // replace bang
               *pHasBang = BANG_CHAR;

               // if we matched return TRUE
               if (InclusionList.find(newitem) != InclusionList.end()) {
                  return 1;
               }

            }

            // Now check for '/' from onefile (?? should we check for onefile in fileheader first?)
            //char* pHasSep = (char*)FindBackwardSep(stritem, '/');
            //if (pHasSep && pHasSep != stritem) {
            //   LOGGING("Failed to match, has slash %s  %s\n", item.c_str(), pHasSep);
            //   // skip over '/' and check again
            //   pHasSep++;
            //   if (InclusionList.find(pHasSep) != InclusionList.end()) {
            //      return 1;
            //   }
            //}
            //else {
            //   LOGGING("NO MATCH %s  %s\n", item.c_str(), pHasSep);
            //}

            return 0;
         }
         else {
            // This is a folder check
            // The pArrayName might be from onefile and flattened
            // If so, it will be in the format foldername/col0
            char* pHasSep = (char*)FindBackwardSep(stritem, SEP_CHAR);

            if (pHasSep) {
               // temp remove char after sep
               char tempchar = pHasSep[1];
               pHasSep[1] = 0;

               std::string newitem = std::string(stritem);

               // replace char we just flipped
               pHasSep[1] = tempchar;

               LOGGING("Failed to match, has sep %s   folder: %s\n", item.c_str(), newitem.c_str());

               // if we matched return TRUE
               if (InclusionList.find(newitem) != InclusionList.end()) {
                  return 1;
               }
            }
            return 0;

         }
      }
      else {
         LOGGING("Simple match %s\n", item.c_str());

         return 1;
      }
   }
};

//-------------------------------------------------------
// Returns bytesPerRow 
int64_t
GetBytesPerRow(SDS_ARRAY_BLOCK* pBlockInfo) {
   // calculate how many rows
   // rows based on the first dimension
   int64_t bytesPerRow = 0;
   int64_t dim0 = pBlockInfo->Dimensions[0];
   int64_t arrayLength = dim0;
   for (int32_t i = 1; i < pBlockInfo->NDim; i++) {
      arrayLength *= pBlockInfo->Dimensions[i];
   }

   if (dim0 > 0) {
      bytesPerRow = (arrayLength / dim0);
      bytesPerRow *= pBlockInfo->ItemSize;

   }
   return bytesPerRow;
}

//-------------------------------------------------------
// Returns both bytesPerBand and changes the bandCount
// May pass in NULL for bandCount
// ArrayLength is all the dimensions multiplied together
int64_t
GetBytesPerBand(SDSArrayInfo* pArrayInfo, int64_t bandSize, int64_t* bandCount=NULL) {

   // calculate how many bands
   // band based on the first dimension
   int64_t bytesPerBand = 0;
   int64_t dim0 = pArrayInfo->Dimensions[0];

   if (dim0 > 0) {
      bytesPerBand = (pArrayInfo->ArrayLength / dim0) * bandSize;
      bytesPerBand *= pArrayInfo->ItemSize;

      if (bandCount)
         *bandCount = (dim0 + (bandSize - 1)) / bandSize;
   }
   else {
      // turn off banding, no dims
      if (bandCount)
         *bandCount = 0;
   }
   return bytesPerBand;
}


//------------------------------------
//
static size_t 
DecompressWithFilter(
   int64_t    compressedSize,      // pBlockInfo->ArrayCompressedSize
   int64_t    uncompressedSize,
   int64_t    arrayDataOffset,     // pBlockInfo->ArrayDataOffset
   int64_t    bytesPerRow,
   SDS_EVENT_HANDLE eventHandle,
   SDS_FILE_HANDLE fileHandle,
   void* tempBuffer,             // used to decompress
   void* destBuffer,             // the array buffer (final destination of data)
   SDS_FILTER* pFilter, 
   int64_t rowOffset,              // the original row offset
   int64_t stackIndex,             // when stacking, the stack #
   int32_t core,                     // thread # we are on
   int32_t compMode
) {
   int64_t result = -1;

   int64_t lastRow = pFilter->BoolMaskLength;
   int64_t lastPossibleRow = (uncompressedSize - bytesPerRow) / bytesPerRow;
   int64_t lastData = (bytesPerRow * lastRow);

   // Check if user just wants the very first bytes
   // All set to TRUE in mask
   if (rowOffset ==0 && lastRow <= lastPossibleRow && lastRow == pFilter->BoolMaskTrueCount && lastData <= uncompressedSize) {
      int64_t firstBand = 0;
      int64_t firstData = bytesPerRow * firstBand;

      LOGGING("special read: %lld  lastData:%lld\n", lastRow, lastData);

      if (compressedSize == uncompressedSize) {
         // Read compressed chunk directly into our destBuffer
         result = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, destBuffer, lastData, arrayDataOffset);
      }
      else {

         // Data is not banded (but is compressed)
         // TJD: More work to do here.  We read extra but we do not always have to.
         // Instead we could keep calling the stream decompressor, and if it needed more input, we could then read again
         // tested with 65536, but it was not enough
         int64_t worstCase = CompressGetBound(compMode, lastData) + (2 * 65536);

         LOGGING("[%d][%lld] worst case %lld v %lld  <-- DecompressWithFilter\n", core, stackIndex, worstCase, compressedSize);
         if (worstCase < compressedSize) {
            // can do this with ZSTD
            compressedSize = worstCase;
         }

         // Read compressed chunk directly into our tempBuffer
         result = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, tempBuffer, compressedSize, arrayDataOffset);

         if (result == compressedSize) {

            if (lastData >= uncompressedSize) {
               lastData = uncompressedSize;
            }
            else {
               // TJD limited stream decompression
               // reduce the uncompressedSize since we do not need all the data
               uncompressedSize = lastData;
            }
            int64_t dcSize = DecompressDataPartial(core, compMode, destBuffer, uncompressedSize, tempBuffer, compressedSize);

            if (dcSize != uncompressedSize) {
               printf("[%d][%lld] MTDecompression band error direct size %lld vs %lld\n", core, stackIndex, dcSize, uncompressedSize);
               result = -1;
            }
         }
      }
   } else {
      //===================================================
      // bool MASK
      //TODO: check for out of bounds lastBand and if found reduce masklength
      int64_t lastRow = pFilter->BoolMaskLength;

      if (rowOffset > lastRow) {
         // nothing to do
         LOGGING("nothing to read all filtered out!\n");
         return result;
      }
      LOGGING("Boolmask bpr:%lld  ai:%lld  rowOffset:%lld  fixup:%lld  masklength:%lld\n", bytesPerRow, stackIndex, rowOffset, 0LL, pFilter->BoolMaskLength);

      // Make sure something to read
      //if (pFilter->pFilterInfo && pFilter->pFilterInfo[stackIndex].TrueCount > 0) {
      if (pFilter->BoolMaskTrueCount > 0) {
         bool uncompressedRead = FALSE;

         if (compressedSize == uncompressedSize) {
            // this data was saved uncompressed
            uncompressedRead = TRUE;
            result = compressedSize;
         }
         else {

            // Read compressed chunk directly into our tempBuffer
            result = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, tempBuffer, compressedSize, arrayDataOffset);
         }

         if (result == compressedSize) {
            // Copy all TRUE rows
            bool*    pMask = pFilter->pBoolMask + rowOffset;

            //printf("allocating temp buffer of %lld bytes", uncompressedSize);
            char* pTempBuffer = (char*)WORKSPACE_ALLOC(uncompressedSize);

            if (pTempBuffer) {
               // Read uncompressed chunk directly into our destination
               int64_t dcSize = 0;

               if (uncompressedRead) {
                  dcSize = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, pTempBuffer, uncompressedSize, arrayDataOffset);
               }
               else {
                  dcSize = DecompressDataPartial(core, compMode, pTempBuffer, uncompressedSize, tempBuffer, compressedSize);
               }

               if (dcSize == uncompressedSize) {
                  char* pDest = (char*)destBuffer;
                  int64_t sectionLength = uncompressedSize / bytesPerRow;

                  // User may have clipped data
                  if ((rowOffset + sectionLength) > pFilter->BoolMaskLength) {
                     sectionLength = pFilter->BoolMaskLength - rowOffset;
                  }

                  LOGGING("sifting through %lld bytes with fixup %lld  bpr:%lld  rowOffset: %lld  to dest:%p\n", sectionLength, 0LL, bytesPerRow, rowOffset, pDest);

                  switch (bytesPerRow) {
                  case 1:
                     for (int64_t i = 0; i < sectionLength; i++) {
                        if (pMask[i]) {
                           *pDest = pTempBuffer[i];
                           pDest++;
                        }
                     }
                     break;
                  case 2:
                     {
                        int16_t* pOut = (int16_t*)pDest;
                        int16_t* pIn = (int16_t*)pTempBuffer;
                        for (int64_t i = 0; i < sectionLength; i++) {
                           if (pMask[i]) {
                              *pOut++ = pIn[i];
                           }
                        }
                     }
                     break;
                  case 4:
                  {
                     int32_t* pOut = (int32_t*)pDest;
                     int32_t* pIn = (int32_t*)pTempBuffer;
                     for (int64_t i = 0; i < sectionLength; i++) {
                        if (pMask[i]) {
                           *pOut++ = pIn[i];
                        }
                     }
                  }
                  break;
                  case 8:
                  {
                     int64_t* pOut = (int64_t*)pDest;
                     int64_t* pIn = (int64_t*)pTempBuffer;
                     for (int64_t i = 0; i < sectionLength; i++) {
                        if (pMask[i]) {
                           *pOut++ = pIn[i];
                        }
                     }
                  }
                  break;

                  default:
                     for (int64_t i = 0; i < sectionLength; i++) {
                        if (pMask[i]) {
                           memcpy(pDest, pTempBuffer + (i * bytesPerRow), bytesPerRow);
                           pDest += bytesPerRow;
                        }
                     }

                  }


               }
               WORKSPACE_FREE(pTempBuffer);
            }
         }
      }
   }
   return result;
}

//-----------------------------------------------------
// Called when a filter is passed in to read
// The data is banded.
int64_t
ReadAndDecompressBandWithFilter(
   SDS_ARRAY_BLOCK *pBlockInfo,  // may contain banding information
   SDS_EVENT_HANDLE eventHandle,
   SDS_FILE_HANDLE   fileHandle,
   void*             tempBuffer, // used to decompress
   void*          destBuffer,    // the array buffer (final destination of data)
   SDS_FILTER*    pFilter,
   int64_t          rowOffset,             // the array # we are on
   int64_t          stackIndex,
   int32_t            core,                  // thread # we are on
   int32_t            compMode,

   int64_t          bandDataSize,
   int64_t*         pBands
) {

   // Check if all filtered out
   if (pFilter->pFilterInfo && pFilter->pFilterInfo[stackIndex].TrueCount == 0) return 0;

   int64_t arrayDataOffset = pBlockInfo->ArrayDataOffset + bandDataSize;
   char* destBandBuffer = (char*)destBuffer;

   int64_t bytesPerRow = GetBytesPerRow(pBlockInfo);

   int64_t previousSize = 0;

   // Is it a fancy mask filter?
//   int32_t firstBand = rowOffset;
   int64_t boolLength = pFilter->BoolMaskLength;
   int64_t bandStart = 0;
   int64_t bandIndex = 0;
   int64_t result = -1;
   int64_t runningTrueCount = 0;

   LOGGING("[%lld] seek to %lld  compsz:%lld  uncompsz:%lld  stackIndex:%lld  bytesPerRow:%lld  <-- ReadAndDecompressBandWithFilter\n", rowOffset, pBlockInfo->ArrayDataOffset, pBlockInfo->ArrayCompressedSize, pBlockInfo->ArrayUncompressedSize, stackIndex, bytesPerRow);
   
   for (int32_t i = 0; i < pBlockInfo->ArrayBandCount; i++) {
      int64_t compressedSize = pBands[i] - previousSize;
      previousSize = pBands[i];

      int64_t bandSize = pBlockInfo->ArrayBandSize;
      int64_t uncompressedSize = pBlockInfo->ArrayBandSize * bytesPerRow;

      // check for last band
      if ((i + 1) == pBlockInfo->ArrayBandCount) {
         uncompressedSize = pBlockInfo->ArrayUncompressedSize - (pBlockInfo->ArrayBandSize * bytesPerRow *i);
         bandSize = uncompressedSize / bytesPerRow;
      }
      int64_t bandEnd = bandStart + bandSize;

      int64_t sectionLength = uncompressedSize / bytesPerRow;

      // If the rest of the data is masked out, break
      if (rowOffset >= boolLength) {
         break;
      }

      // User may have clipped data
      if ((rowOffset + sectionLength) > boolLength) {
         sectionLength = boolLength - rowOffset;
      }

      // Copy all TRUE rows
      bool*    pMask = pFilter->pBoolMask + rowOffset;
      int64_t trueCount = SumBooleanMask((int8_t*)pMask, sectionLength);

      if (trueCount) {

         SDSFilterInfo sfi;
         sfi.TrueCount = trueCount;  // If zero, dont bother reading in data

         if (pFilter->pFilterInfo) {
            sfi.RowOffset = pFilter->pFilterInfo[stackIndex].RowOffset + runningTrueCount;  // sum of all previous True Count
         }
         else {
            sfi.RowOffset = runningTrueCount;  // sum of all previous True Count
         }

         SDS_FILTER tempFilter;
         tempFilter.BoolMaskLength = pFilter->BoolMaskLength;
         tempFilter.BoolMaskTrueCount = pFilter->BoolMaskTrueCount;
         tempFilter.pBoolMask = pFilter->pBoolMask;
         tempFilter.pFilterInfo = &sfi;

         LOGGING("want band start:%lld  masklength:%lld  compsize:%lld  decomp:%lld\n", rowOffset, trueCount, compressedSize, uncompressedSize);

         result = DecompressWithFilter(
            compressedSize,
            uncompressedSize,
            arrayDataOffset,
            bytesPerRow,
            eventHandle,
            fileHandle,
            tempBuffer,
            destBandBuffer,
            &tempFilter,
            rowOffset,
            0,             // fake to 0 to read our filterinfo
            core,
            compMode);

         // TODO change based on how many TRUE values in the mask
         destBandBuffer += (trueCount * bytesPerRow);

      }
      rowOffset += sectionLength;
      runningTrueCount += trueCount;

      arrayDataOffset += compressedSize;
      bandStart = bandEnd;
   }

   return result;
};

//------------------------------------------------------------
// New routine for ver 4.3 (July 2019) to read in data from possibly a band with a mask
//
// On Entry tempBuffer is guaranteed to be at least the compressedSize
// The mask must always be a boolean mask
// If a mask exists, data 
// return size of bytes read or -1 on failure
static size_t
ReadAndDecompressArrayBlockWithFilter(
   SDS_ARRAY_BLOCK *pBlockInfo,  // may contain banding information
   SDS_EVENT_HANDLE eventHandle,
   SDS_FILE_HANDLE fileHandle,
   void* tempBuffer,             // used to decompress
   void* destBuffer,             // the array buffer (final destination of data)
   int64_t rowOffset,              // when stacking the orig row offset
   SDS_FILTER* pFilter,
   int64_t stackIndex,             // the stack # we are on
   int32_t core,                     // thread # we are on
   int32_t compMode
) {

   int64_t result = -1;
   bool didAlloc = FALSE;

   if (!tempBuffer) {
      LOGGING("Allocating tempbuffer of %lld\n", pBlockInfo->ArrayCompressedSize);
      tempBuffer = WORKSPACE_ALLOC(pBlockInfo->ArrayCompressedSize);
      didAlloc = TRUE;
   }

   // All boolean masks come here
   if (tempBuffer) {

      if (pBlockInfo->ArrayBandCount > 0) {
         // we have bands
         LOGGING("bandcount:%d  bandsize:%d  uncomp size: %lld\n", pBlockInfo->ArrayBandCount, pBlockInfo->ArrayBandSize, pBlockInfo->ArrayUncompressedSize);

         // read in band header
         // allocate memory on the stack
         int64_t  bandDataSize = pBlockInfo->ArrayBandCount * sizeof(int64_t);
         int64_t* pBands = (int64_t*)alloca(bandDataSize);
         result = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, pBands, bandDataSize, pBlockInfo->ArrayDataOffset);

         if (result == bandDataSize) {
            // read from banded decompressed into array
            result =
               ReadAndDecompressBandWithFilter(
                  pBlockInfo,
                  eventHandle,
                  fileHandle,
                  tempBuffer,
                  destBuffer,
                  pFilter,
                  rowOffset,             // the rowOffset # we are on
                  stackIndex,
                  core,                  // thread # we are on
                  compMode,
                  bandDataSize,
                  pBands);

         }
         else {
            printf("!!!Error reading in band header\n");
         }

      }
      else {
         // No bands, but compressed and a filter exists
         // read from decompressed into array
         result = DecompressWithFilter(
            pBlockInfo->ArrayCompressedSize,
            pBlockInfo->ArrayUncompressedSize,
            pBlockInfo->ArrayDataOffset,
            GetBytesPerRow(pBlockInfo),
            eventHandle,
            fileHandle,
            tempBuffer, 
            destBuffer, 
            pFilter,
            rowOffset,
            stackIndex,
            core,
            compMode);
      }
   }
   else {
      printf("out of mem no temp buffer  bandcount:%d  bandsize:%d  uncomp size: %lld\n", pBlockInfo->ArrayBandCount, pBlockInfo->ArrayBandSize, pBlockInfo->ArrayUncompressedSize);
   }
   if (didAlloc && tempBuffer) {
      WORKSPACE_FREE(tempBuffer);
   }
      
      
   return result;
}



//------------------------------------------------------------
// New routine for ver 4.3 (July 2019) to read in data from possibly a band with a mask
//
// On Entry tempBuffer is guaranteed to be at least the compressedSize
// The mask must always be a boolean mask
// If a mask exists, data 
// return size of bytes read or -1 on failure
static size_t
ReadAndDecompressArrayBlock(
   SDS_ARRAY_BLOCK *pBlockInfo,  // may contain banding information
   SDS_EVENT_HANDLE eventHandle,
   SDS_FILE_HANDLE fileHandle, 
   void* tempBuffer,             // used to decompress
   void* destBuffer,             // the array buffer (final destination of data)
   int64_t arrayIndex,             // the array # we are on
   int32_t core,                     // thread # we are on
   int32_t compMode
) {


   int64_t result = -1;
   int64_t compressedSize = pBlockInfo->ArrayCompressedSize;

   LOGGING("[%lld] seek to %lld  sz: %lld  <-- ReadAndDecompressArrayBlock\n", arrayIndex, pBlockInfo->ArrayDataOffset, pBlockInfo->ArrayCompressedSize);

   // Check if uncompressed
   if (compressedSize == pBlockInfo->ArrayUncompressedSize) {

      // Read uncompressed chunk directly into our destination
      result = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, destBuffer, compressedSize, pBlockInfo->ArrayDataOffset);

      if (result != compressedSize) {
         printf("[%d][%lld] error while reading into uncompressed  sz: %lld\n", core, arrayIndex, compressedSize);
         result = -1;
      }

   } else
   
   if (tempBuffer) {

      if (pBlockInfo->ArrayBandCount > 0) {
         // we have bands
         //printf("bandcount:%d  bandsize:%d  uncomp size: %lld\n", pBlockInfo->ArrayBandCount, pBlockInfo->ArrayBandSize, pBlockInfo->ArrayUncompressedSize);

         // read in band
         // allocate memory on the stack
         int64_t  bandDataSize = pBlockInfo->ArrayBandCount * sizeof(int64_t);
         int64_t* pBands = (int64_t*)alloca(bandDataSize);
         result = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, pBands, bandDataSize, pBlockInfo->ArrayDataOffset);

         int64_t arrayDataOffset = pBlockInfo->ArrayDataOffset + bandDataSize;
         char* destBandBuffer = (char*)destBuffer;

         int64_t bytesPerRow = GetBytesPerRow(pBlockInfo);

         if (result == bandDataSize) {
            int64_t previousSize = 0;

            for (int32_t i = 0; i < pBlockInfo->ArrayBandCount; i++) {
               int64_t compressedSize = pBands[i] - previousSize;
               previousSize = pBands[i];

               int64_t uncompressedSize = pBlockInfo->ArrayBandSize;
               uncompressedSize *= bytesPerRow;

               // check for last band
               if ((i + 1) == pBlockInfo->ArrayBandCount) {
                  uncompressedSize = pBlockInfo->ArrayUncompressedSize - (destBandBuffer - (char*)destBuffer);
               }

               if (compressedSize == uncompressedSize) {
                  result = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, destBandBuffer, uncompressedSize, arrayDataOffset);
                  if (result != uncompressedSize) {
                     printf("[%d][%lld][%d] MTDecompression (uncompressed) band error size %lld vs %lld\n", core, arrayIndex, i, uncompressedSize, compressedSize);
                     result = -1;
                     break;
                  }
               }
               else {

                  result = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, tempBuffer, compressedSize, arrayDataOffset);
                  //printf("[%d] decompressing  %d  size %lld  uncomp: %lld\n", core, i, compressedSize, uncompressedSize);
                  int64_t dcSize = DecompressData(NULL, compMode, destBandBuffer, uncompressedSize, tempBuffer, compressedSize);
                  if (dcSize != uncompressedSize) {
                     printf("[%d][%lld][%d] MTDecompression band error size %lld vs %lld vs %lld\n", core, arrayIndex, i, dcSize, uncompressedSize, compressedSize);
                     result = -1;
                     break;
                  }
               }

               arrayDataOffset += compressedSize;
               destBandBuffer += uncompressedSize;

            }

         }
         else {
            printf("!!!Error reading in band header\n");
         }

      }
      else {

         // Read compressed chunk into temporary buffer
         result = DefaultFileIO.FileReadChunk(eventHandle, fileHandle, tempBuffer, compressedSize, pBlockInfo->ArrayDataOffset);

         if (result != compressedSize) {
            printf("[%d][%lld] error while reading into decompressed  sz: %lld\n", core, arrayIndex, compressedSize);
            result = -1;
         }
         else {

            int64_t uncompressedSize = pBlockInfo->ArrayUncompressedSize;

            int64_t dcSize = DecompressData(NULL, compMode, destBuffer, uncompressedSize, tempBuffer, compressedSize);

            if (CompressIsError(compMode, dcSize)) {
               printf("[%d][%lld] MTDecompression error\n", core, arrayIndex);
               result = -1;
            }
            else if (dcSize != uncompressedSize) {
               printf("[%d][%lld] MTDecompression error size\n", core, arrayIndex);
               result = -1;
            }
            else {
               LOGGING("[%d][%lld] decomp success\n", core, arrayIndex);
            }
         }
      }
   }
   return result;
}


//----------------------------
// Called when starting a file
void FillFileHeader(
   SDS_FILE_HEADER *pFileHeader,
   int64_t fileOffset,
   int16_t compMode,
   int16_t compType,
   int32_t compLevel,
   int16_t fileType,
   int16_t stackType,
   int32_t authorId,
   int64_t listNameLength,
   int64_t listNameCount,
   int64_t metaBlockSize,
   int64_t arrayCount,
   int64_t bandSize) {

   // To help detect old versions or corrupt files
   pFileHeader->SDSHeaderMagic = SDS_MAGIC;

   pFileHeader->VersionHigh = SDS_VERSION_HIGH;
   pFileHeader->VersionLow = SDS_VERSION_LOW;

   pFileHeader->CompMode = compMode;
   pFileHeader->CompType = compType;
   pFileHeader->CompLevel = compLevel;
   pFileHeader->FileType = fileType;
   pFileHeader->StackType = stackType;
   pFileHeader->AuthorId = authorId;

   pFileHeader->NameBlockCount = listNameCount;
   pFileHeader->NameBlockSize = listNameLength;
   pFileHeader->NameBlockOffset = sizeof(SDS_FILE_HEADER) + fileOffset;

   pFileHeader->MetaBlockSize = metaBlockSize;
   pFileHeader->MetaBlockOffset = pFileHeader->NameBlockOffset + SDS_PAD_NUMBER(listNameLength);

   // assume uncompressed
   pFileHeader->TotalMetaCompressedSize = metaBlockSize;
   pFileHeader->TotalMetaUncompressedSize = metaBlockSize;

   // count determined by dividing
   pFileHeader->ArrayBlockSize = arrayCount * sizeof(SDS_ARRAY_BLOCK);
   pFileHeader->ArrayBlockOffset = pFileHeader->MetaBlockOffset + SDS_PAD_NUMBER(pFileHeader->MetaBlockSize);

   // As we write or read arrays, this value should be incremented
   pFileHeader->ArraysWritten = 0;
   pFileHeader->ArrayFirstOffset = pFileHeader->ArrayBlockOffset + SDS_PAD_NUMBER(pFileHeader->ArrayBlockSize);

   LOGGING("main offsets %lld  %lld  %lld\n", pFileHeader->MetaBlockOffset, pFileHeader->ArrayBlockOffset, pFileHeader->ArrayFirstOffset);

   pFileHeader->TotalArrayCompressedSize = 0;
   pFileHeader->TotalArrayUncompressedSize = 0;

   // For version 4.3
   pFileHeader->BandBlockOffset = 0; // pFileHeader->ArrayFirstOffset;
   pFileHeader->BandBlockSize = 0;
   pFileHeader->BandBlockCount = 0;
   pFileHeader->BandSize = bandSize;

   // For version 4.4
   pFileHeader->SectionBlockSize =0;
   pFileHeader->SectionBlockOffset =0;  // points to section directory if it exists
   pFileHeader->SectionBlockCount = 0;  
   pFileHeader->SectionBlockReservedSize = 0;
   pFileHeader->FileOffset = fileOffset;
   pFileHeader->TimeStampUTCNanos = 0;

   for (uint64_t i = 0; i < sizeof(pFileHeader->Reserved); i++) {
      pFileHeader->Reserved[i] = 0;  
   }
}

//----------------------------------------
//
//void FillFileHeaderExtra(
//   SDS_FILE_HEADER *pFileHeader,
//   const char* sectionName,
//   int64_t sectionNameLength) {
//
//   pFileHeader->SectionBlockSize = 0;
//   pFileHeader->SectionBlockOffset = 0;  // points to section directory if it exists
//   pFileHeader->SectionBlockCount = 0;
//   pFileHeader->SectionBlockReservedSize = 0;
//}

//----------------------------------------
// Returns: -1 file will be closed
// Returns: 0 file is ok
int64_t ReadFileHeader(SDS_FILE_HANDLE fileHandle, SDS_FILE_HEADER* pFileHeader, int64_t fileOffset, const char* fileName) {
   size_t bytesRead =
      DefaultFileIO.FileReadChunk(NULL, fileHandle, pFileHeader, sizeof(SDS_FILE_HEADER), fileOffset);

   if (bytesRead != sizeof(SDS_FILE_HEADER)) {
      SetErr_Format(SDS_VALUE_ERROR, "Decompression error cannot read header for file: %s.  Error: %s", fileName, GetLastErrorMessage());
      DefaultFileIO.FileClose(fileHandle);
      return -1;
   }

   if (pFileHeader->SDSHeaderMagic != SDS_MAGIC || pFileHeader->VersionHigh != SDS_VERSION_HIGH) {
      if (pFileHeader->SDSHeaderMagic == SDS_MAGIC && pFileHeader->VersionHigh != SDS_VERSION_HIGH) {

         SetErr_Format(SDS_VALUE_ERROR, "SDS Version number not understood (may need newer version): %s  %d  arrays: %lld", fileName, (int)(pFileHeader->VersionHigh), pFileHeader->ArraysWritten);
      }
      else {
         SetErr_Format(SDS_VALUE_ERROR, "Decompression error cannot understand header for file (corrupt or different filetype): %s  %d  arrays: %lld  fileoffset: %lld", fileName, (int)(pFileHeader->SDSHeaderMagic), pFileHeader->ArraysWritten, fileOffset);
      }
      DefaultFileIO.FileClose(fileHandle);
      return -1;
   }

   return 0;
}

//----------------------------------------
//
//-----------------------------------------------------------
// Add one more name and offset
// Returns sizeof new section
// Returns pointer in *pListNames
// NOTE: caller must WORKSPACE_FREE *pListNames
int64_t
   SDSSectionName::BuildSectionNamesAndOffsets(
      char** pListNames,                // Returned
      const char*  pNewSectionName,
      int64_t  newSectionOffset           // 0 Allowed
) {
   // alloc worst case scenario
   int64_t allocSize = SDS_PAD_NUMBER((((SDS_MAX_SECTIONNAME + 8) * (SectionCount + 1)) + 1024));

   *pListNames = (char*)WORKSPACE_ALLOC(allocSize);

   if (!(*pListNames)) {
      return 0;
   }

   char* pStart = *pListNames;
   char* pDest = *pListNames;

   // For all the section write out the section name and the fileoffset to the SDS_FILE_HEADER for that section
   for (int32_t i = 0; i < SectionCount; i++) {
      const char* pName = pSectionNames[i];

      // strcpy the name with a 0 char termination
      while ((*pDest++ = *pName++));

      // after writing the name, write the new fileheader offset
      // Store the 64 bit offset
      *(int64_t*)pDest = pSectionOffsets[i];
      pDest += sizeof(int64_t);
   }

   // strcpy
   while ((*pDest++ = *pNewSectionName++));

   // Store the 64 bit offset
   *(int64_t*)pDest = newSectionOffset;
   pDest += sizeof(int64_t);

   // return the size used
   return pDest - pStart;
}

//--------------------------------------------
// Also zero out pArrayNames, zero out pArrayEnums
//
void SDSSectionName::AllocateSectionData(int64_t sectionBlockCount, int64_t sectionSize) {
   SectionCount = sectionBlockCount;
   if (pSectionData != NULL) {
      printf("Double Allocation sectionData!!\n");
   }
   pSectionData = (char*)WORKSPACE_ALLOC(sectionSize);

   // ZERO OUT
   pSectionNames = (const char**)WORKSPACE_ALLOC(SectionCount * sizeof(void*));
   for (int32_t i = 0; i < SectionCount; i++) {
      pSectionNames[i] = NULL;
   }
   pSectionOffsets = (int64_t*)WORKSPACE_ALLOC(SectionCount * sizeof(int64_t));
   for (int32_t i = 0; i < SectionCount; i++) {
      pSectionOffsets[i] = 0;
   }
}


//----------------------------------------
// Delete only if allocated
void SDSSectionName::DeleteSectionData() {
   if (pSectionData) {
      WORKSPACE_FREE(pSectionData);
      pSectionData = NULL;
   }
   if (pSectionNames) {
      WORKSPACE_FREE(pSectionNames);
      pSectionNames = NULL;
   }
   if (pSectionOffsets) {
      WORKSPACE_FREE(pSectionOffsets);
      pSectionOffsets = NULL;
   }
}


//--------------------------------------------------------
// Returns the section names and offsets of sections written
//
// Called by DecompressFileIntenral
void SDSSectionName::MakeListSections(const int64_t sectionBlockCount, const int64_t sectionByteSize) {
   const char* startSectionData = pSectionData;
   const char* pSections = pSectionData;

   // for every section
   for (int32_t i = 0; i < sectionBlockCount; i++) {
      const char* pStart = pSections;

      // skip to end (search for 0 terminating char)
      while (*pSections++);

      // get the offset
      int64_t value = *(int64_t*)pSections;
      pSections += sizeof(int64_t);

      LOGGING("makelist section is %s, %d, offset at %lld\n", pStart, i, value);

      // The appended named
      pSectionNames[i] = pStart;

      // Offset within file to SDS_FILE_OFFSET
      pSectionOffsets[i] = value;

      if ((pSections - startSectionData) >= sectionByteSize) break;
   }
}

//===============================================================
// User can append with a section to a file which has not sections
// So we create a dummy section nameed '0'
char* SDSSectionName::MakeFirstSectionName() {
   LOGGING("**First time appending  %p\n", pSectionNames);
   if (SectionCount == 0) {
      // Will allocate pSectionData for us
      AllocateSectionData(1, sizeof(g_firstsectiondata));
      pSectionNames[0] = g_firstsectioname;
      pSectionOffsets[0] = g_firstsectionoffset;
      SectionCount = 1;
      memcpy(pSectionData, g_firstsectiondata, sizeof(g_firstsectiondata));
      return pSectionData;
   }
   printf("This code path should not be hit\n");
   return pSectionData;
}

//===============================================================
//-------------------------------------------------------
// Input: file already opened
// NOTE: Called from DecompressFileInternal
// returns NULL on error
// return list of strings (section names)
// on success pSectionData is valid
char* SDSSectionName::ReadListSections(SDS_FILE_HANDLE SDSFile, SDS_FILE_HEADER *pFileHeader) {
   int64_t sectionSize = pFileHeader->SectionBlockSize;

   if (sectionSize) {
      LOGGING("Section Block Count %lld,  sectionSize %lld, reserved %lld\n", pFileHeader->SectionBlockCount, sectionSize, pFileHeader->SectionBlockReservedSize);

      AllocateSectionData(pFileHeader->SectionBlockCount, sectionSize);

      if (!pSectionData) {
         SetErr_Format(SDS_VALUE_ERROR, "Decompression error in sectionSize: %lld", sectionSize);
         return NULL;
      }

      int64_t bytesRead =
         DefaultFileIO.FileReadChunk(NULL, SDSFile, pSectionData, sectionSize, pFileHeader->SectionBlockOffset);

      if (bytesRead != sectionSize) {
         SetErr_Format(SDS_VALUE_ERROR, "Decompression error in bytesRead: %lld", sectionSize);
         return NULL;
      }

      // Run through Section list and setup pointers
      MakeListSections(SectionCount, sectionSize);
      LOGGING("Returning sections: %s\n", pSectionData);
      return pSectionData;
   }
   return NULL;
}

SDSSectionName::~SDSSectionName() {
   DeleteSectionData();
};



//---------------------------------------------------------------
// Opens the file
// Compresses the meta data
// Write the list of names
// Writes the compressed matadata
//
// Returns: filehandle or BAD_SDS_HANDLE on failure
// pFileHeader is filled in
//
SDS_FILE_HANDLE StartCompressedFile(
   const char* fileName,
   SDS_FILE_HEADER* pFileHeader,

   int16_t compType,
   int32_t compLevel,
   int16_t fileType,
   int16_t stackType,
   int32_t authorId,

   const char* listNames,
   int64_t listNameLength,
   int64_t listNameCount,

   const char* strMeta,
   int64_t strMetaLength,
   int64_t arrayCount,
   int32_t mode,       // COMPRESSION_MODE_COMPRESS_APPEND_FILE
   SDS_STRING_LIST* pFolderName,
   int64_t bandSize,
   SDS_WRITE_INFO* pWriteInfo) {

   int64_t    fileOffset=0;

   // TODO: have an override mode for appending?
   if (mode == COMPRESSION_MODE_COMPRESS_APPEND_FILE) {
      // Check for existing file
   }

   // Check if the user is appending to an existing file
   if (pWriteInfo->sectionName) {
      // Check for existing file
      fileOffset = DefaultFileIO.FileSize(fileName);
      if (fileOffset > 0) {

         LOGGING("File already existed with append section %s, filesize: %lld\n", pWriteInfo->sectionName, fileOffset);
         fileOffset = SDS_PAD_NUMBER(fileOffset);
         mode = COMPRESSION_MODE_COMPRESS_APPEND_FILE;
      }
      else {
         fileOffset = 0;
      }
   }

   SDS_FILE_HANDLE  fileHandle =
      DefaultFileIO.FileOpen(fileName, true, true, false, mode == COMPRESSION_MODE_COMPRESS_APPEND_FILE);

   if (!fileHandle) {
      SetErr_Format(SDS_VALUE_ERROR, "Compression error cannot create/open file: %s.  Error: %s", fileName, GetLastErrorMessage());
      return BAD_SDS_HANDLE;
   }

   if (mode == COMPRESSION_MODE_COMPRESS_APPEND_FILE) {
      // Check to make sure the fileheader is valid before appending
      int64_t result =
         ReadFileHeader(fileHandle, pFileHeader, 0, fileName);

      if (result != 0) {
         return BAD_SDS_HANDLE;
      }
   }


   size_t dest_size = strMetaLength;
   void* dest = (void*)strMeta;
   int64_t cSize = strMetaLength;

   // Check if we compress metadata
   if (compType == COMPRESSION_TYPE_ZSTD) {

      dest_size = CompressGetBound(compType, strMetaLength);
      dest = (char*)WORKSPACE_ALLOC(dest_size);

      if (!dest) {
         DefaultFileIO.FileClose(fileHandle);
         return BAD_SDS_HANDLE;
      }

      cSize = CompressData(compType, dest, dest_size, strMeta, strMetaLength, compLevel);

      if (cSize >= strMetaLength) {
         // store uncompressed
         WORKSPACE_FREE(dest);
         dest = (void*)strMeta;
         cSize = strMetaLength;
      }
      else {
         if (CompressIsError(compType, cSize)) {
            DefaultFileIO.FileClose(fileHandle);
            WORKSPACE_FREE(dest);
            return BAD_SDS_HANDLE;
         }
      }
   }

   // If they have a foldername and they are appending..
   // Check if folder already exists
   // If folder does not exist, seek to end of file and get offset
   // PAD To 512
   // Fill FileHeader with
   LOGGING("Using fileOffset %lld\n", fileOffset);

   FillFileHeader(pFileHeader, fileOffset, COMPRESSION_MODE_COMPRESS_FILE, 
      compType, compLevel,
      fileType, stackType, 
      authorId, 
      listNameLength, listNameCount,
      cSize, arrayCount, bandSize);

   pFileHeader->TotalMetaCompressedSize = cSize;
   pFileHeader->TotalMetaUncompressedSize = strMetaLength;
   DefaultFileIO.FileWriteChunk(NULL, fileHandle, pFileHeader, sizeof(SDS_FILE_HEADER), fileOffset);

   LOGGING("meta compressed to %llu vs %lld  %llu\n", cSize, strMetaLength, sizeof(SDS_FILE_HEADER));

   if (listNameLength) {
      DefaultFileIO.FileWriteChunk(NULL, fileHandle, (void*)listNames, listNameLength, pFileHeader->NameBlockOffset);
   }

   if (cSize) {
      DefaultFileIO.FileWriteChunk(NULL, fileHandle, dest, cSize, pFileHeader->MetaBlockOffset);
   }

   //// PAD THE REST OUT
   //char* filler = (char*)WORKSPACE_ALLOC(SDS_PADSIZE);
   //memset(filler, 0, SDS_PADSIZE);
   //int64_t diff = SDS_PAD_NUMBER(cSize);
   //diff = diff - cSize;
   //if (diff > 0) {
   //   fwrite(filler, diff, 1, fileHandle);
   //}   
   //WORKSPACE_FREE(filler);

   // Check if we allocated when compressing meta
   if (compType == COMPRESSION_TYPE_ZSTD && cSize < strMetaLength) {
      WORKSPACE_FREE(dest);
   }

   return fileHandle;
}


//-------------------------------------------------------
// NOTE: Should sdsFileHandle be passed as a pointer?  are we closing files twice?
// Will close file
// Will free any memory allocated for the arrayblocks and set pointer to NULL
//-------------------------------------------------------
void EndCompressedFile(
   SDS_FILE_HANDLE sdsFile,
   SDS_FILE_HEADER* pFileHeader,
   SDS_ARRAY_BLOCK* pArrayBlocks,
   SDS_WRITE_INFO* pWriteInfo) {

   // close it out

   LOGGING("SDS: Array first offset --- %lld   Total comp size %lld\n",  pFileHeader->ArrayFirstOffset, pFileHeader->TotalArrayCompressedSize);

   int64_t LastFileOffset = pFileHeader->GetEndOfFileOffset();

   // Check if the user is appending to an existing file
   if (pWriteInfo->sectionName) {
      SDSSectionName cSDSSectionName;

      // This is the first section name       
      char*    pListNames = NULL;

      // Are we the first entry?
      if (pFileHeader->FileOffset == 0) {

         LOGGING("SDS: Writing first section %s at %lld\n", pWriteInfo->sectionName, LastFileOffset);

         int64_t sectionSize =
            cSDSSectionName.BuildSectionNamesAndOffsets(
               &pListNames,
               pWriteInfo->sectionName,
               pFileHeader->FileOffset);

         // At the end of the file, write out the section names and the file offset to find them
         // Update the first file header (current file header)
         pFileHeader->SectionBlockCount = 1;
         pFileHeader->SectionBlockOffset = LastFileOffset;
         pFileHeader->SectionBlockSize = sectionSize;
         pFileHeader->SectionBlockReservedSize = SDS_PAD_NUMBER(sectionSize);

         // write first section header
         int64_t result = DefaultFileIO.FileWriteChunk(NULL, sdsFile, pListNames, pFileHeader->SectionBlockReservedSize, pFileHeader->SectionBlockOffset);

         if (result != pFileHeader->SectionBlockReservedSize) {
            SetErr_Format(SDS_VALUE_ERROR, "Compression error cannot append section %lld at %lld", pFileHeader->SectionBlockReservedSize, pFileHeader->SectionBlockOffset);
         }
      }
      else {
         // old fileheader
         SDS_FILE_HEADER   fileHeader;

         // read in the first file header to get section information since we are appending another section
         int64_t result =
            ReadFileHeader(sdsFile, &fileHeader, 0, "reread");

         if (result == 0) {

            char* pSectionNames = cSDSSectionName.ReadListSections(sdsFile, &fileHeader);

            // If the user is appending a section for the first time but the file already exists.
            // In this case the blockcount is really 1 but the first section has no name
            if (!pSectionNames) {
               pSectionNames = cSDSSectionName.MakeFirstSectionName();
               fileHeader.SectionBlockCount = 1;
            }

            int64_t blockCount = fileHeader.SectionBlockCount;

            LOGGING("SDS: Writing section %s with blockcount:%lld at %lld (%lld) at %lld\n", pWriteInfo->sectionName, blockCount, pFileHeader->FileOffset, fileHeader.FileOffset, LastFileOffset);
            int64_t sectionSize =
               cSDSSectionName.BuildSectionNamesAndOffsets(
                  &pListNames,
                  pWriteInfo->sectionName,
                  pFileHeader->FileOffset);

            // Mark this section as older
            fileHeader.SectionBlockCount = blockCount + 1;
            fileHeader.SectionBlockSize = sectionSize;

            LOGGING("SDS: new section size %lld vs %lld\n", sectionSize, fileHeader.SectionBlockReservedSize);

            if (fileHeader.SectionBlockReservedSize < sectionSize) {
               // new section with new block offset
               fileHeader.SectionBlockOffset = LastFileOffset;
               fileHeader.SectionBlockReservedSize = SDS_PAD_NUMBER(sectionSize);
            }
         }

         // update or rewrite section header
         result = DefaultFileIO.FileWriteChunk(NULL, sdsFile, pListNames, fileHeader.SectionBlockReservedSize, fileHeader.SectionBlockOffset);

         if (result != fileHeader.SectionBlockReservedSize) {
            SetErr_Format(SDS_VALUE_ERROR, "Compression error cannot append section %lld at %lld", fileHeader.SectionBlockReservedSize, fileHeader.SectionBlockOffset);
         }

         // update first file header
         result = DefaultFileIO.FileWriteChunk(NULL, sdsFile, &fileHeader, sizeof(SDS_FILE_HEADER), 0);
         if (result != sizeof(SDS_FILE_HEADER)) {
            SetErr_Format(SDS_VALUE_ERROR, "Compression error cannot rewrite fileheader\n");
         }
      }

      if (pListNames)  WORKSPACE_FREE(pListNames);
      
   }

   //Timestamp
   pFileHeader->TimeStampUTCNanos = GetUTCNanos();

   int64_t result = DefaultFileIO.FileWriteChunk(NULL, sdsFile, pFileHeader, sizeof(SDS_FILE_HEADER), pFileHeader->FileOffset);
   if (result != sizeof(SDS_FILE_HEADER)) {
      SetErr_Format(SDS_VALUE_ERROR, "Compression error cannot write fileheader at offset %lld\n", pFileHeader->FileOffset);
   }
   LOGGING("Total arrays written %lld  --- %llu %lld\n", pFileHeader->ArraysWritten, sizeof(SDS_FILE_HEADER), result);

   result = DefaultFileIO.FileWriteChunk(NULL, sdsFile, pArrayBlocks, pFileHeader->ArrayBlockSize, pFileHeader->ArrayBlockOffset);
   LOGGING("array block offset --- %lld %lld  %lld\n", pFileHeader->ArrayBlockOffset, pFileHeader->ArrayBlockSize, pArrayBlocks[0].ArrayDataOffset);
   if (result != pFileHeader->ArrayBlockSize) {
      printf("!!!Internal error closing compressed file\n");
   }

   DefaultFileIO.FileClose(sdsFile);
}


//----------------------------------------------------------
//Decompress -- called from multiple threads -- READING data
//
// t is the array index (0 to total number of arrays, or array sections to read)
// pstCompressArrays must have pBlockInfo set
// It will read from a file using: pstCompressArrays->eventHandles[core], sdsFile
// for normal read: read into pstCompressArrays->ArrayInfo[t].pData
// size of the READ: pBlockInfo->ArrayCompressedSize
//
bool DecompressFileArray(void* pstCompressArraysV, int32_t core, int64_t t) {

   LOGGING("[%lld] Start of decompress array: core %d   compress: %p\n", t, core, pstCompressArraysV);
   SDS_READ_DECOMPRESS_ARRAYS* pstCompressArrays = (SDS_READ_DECOMPRESS_ARRAYS *)pstCompressArraysV;
   SDS_FILE_HANDLE sdsFile = pstCompressArrays->fileHandle;

   // point32_t to blocks
   SDS_ARRAY_BLOCK* pBlockInfo = &pstCompressArrays->pBlockInfo[t];
   //LOGGING("[%lld] Step 2 of decompress array: core %d  blockinfo %p\n", t, core, pBlockInfo);

   int64_t source_size = pBlockInfo->ArrayCompressedSize;
   void* destBuffer = NULL;

   // Check if we are reading into memory or reading into a preallocated numpy array
   if (pstCompressArrays->compMode == COMPRESSION_MODE_SHAREDMEMORY) {
      SDS_ARRAY_BLOCK* pArrayBlock = pstCompressArrays->pMemoryIO->GetArrayBlock(t);
      destBuffer = pstCompressArrays->pMemoryIO->GetMemoryOffset(pArrayBlock->ArrayDataOffset);
      LOGGING("[%lld] decompressing shared memory %p\n", t, destBuffer);
   }
   else {

      // Use callback to get to array buffer
      destBuffer = pstCompressArrays->ArrayInfo[t].pData;
      LOGGING("[%lld] decompressing into %p\n", t, destBuffer);
   }

   // Make sure we have a valid buffer
   if (destBuffer) {

      // Check if uncompressed
      // check if our temporary buffer is large enough to hold decompression data
      if (source_size != pBlockInfo->ArrayUncompressedSize && (pstCompressArrays->pCoreMemorySize[core] < source_size)) {

         //  Not large enough, allocate more
         if (pstCompressArrays->pCoreMemory[core]) {
            // free old one if there
            WORKSPACE_FREE(pstCompressArrays->pCoreMemory[core]);
         }

         // Reallocate larger memory
         pstCompressArrays->pCoreMemorySize[core] = source_size;
         pstCompressArrays->pCoreMemory[core] = WORKSPACE_ALLOC(source_size);

         // Log that we were forced to reallocate
         LOGGING("-");
      }

      void* tempBuffer = pstCompressArrays->pCoreMemory[core];

      if (pBlockInfo->Flags & SDS_ARRAY_FILTERED) {
         // No stacking but filtering
         int64_t result =
            ReadAndDecompressArrayBlockWithFilter(
               pBlockInfo,
               pstCompressArrays->eventHandles[core],
               sdsFile,
               tempBuffer,
               destBuffer,
               0,
               pstCompressArrays->pFilter,
               t,
               core,
               COMPRESSION_TYPE_ZSTD
            );

      }
      else {
         // No stacking, no filtering
         int64_t result =
            ReadAndDecompressArrayBlock(
               pBlockInfo,
               pstCompressArrays->eventHandles[core],
               sdsFile,
               tempBuffer,
               destBuffer,
               t,
               core,
               COMPRESSION_TYPE_ZSTD
            );
      }
   }

   return TRUE;
}



//-------------------------------------------------------
//Compress -- called from multiple threads - WRITING
//
// pstCompressArrays must be set
// pstCompressArrays->pArrayInfo must be set
// pstCompressArrays->pCoreMemory must be set
// pBlockInfo must be set
bool CompressFileArray(void* pstCompressArraysV, int32_t core, int64_t t) {

   SDS_WRITE_COMPRESS_ARRAYS* pstCompressArrays = (SDS_WRITE_COMPRESS_ARRAYS *)pstCompressArraysV;
   SDS_FILE_HEADER* pFileHeader = pstCompressArrays->pFileHeader;

   SDSArrayInfo* pArrayInfo = &pstCompressArrays->pArrayInfo[t];

   int64_t bandSize = pFileHeader->BandSize;
   int64_t bandCount = 0;
   int64_t bytesPerBand = 0;

   int64_t source_size = pArrayInfo->ArrayLength * pArrayInfo->ItemSize;

   // Calculate how much to allocate
   int64_t dest_size = source_size;
   int64_t wantedSize = source_size;
   
   if (pstCompressArrays->compType == COMPRESSION_TYPE_ZSTD) {

      // TODO: subroutine 
      if (bandSize > 0) {
         // calculate how many bands
         bytesPerBand = GetBytesPerBand(pArrayInfo, bandSize, &bandCount);

         if (bytesPerBand==0) {
            // turn off banding, no dims
            bandSize = 0;
         }
      }

      wantedSize =
      dest_size = CompressGetBound(COMPRESSION_TYPE_ZSTD, source_size);

      // If banding is on, we store the last offset upfront
      wantedSize += bandCount * sizeof(int64_t);
   }

   if (pstCompressArrays->pCoreMemorySize[core] < wantedSize) {

      if (pstCompressArrays->pCoreMemory[core]) {
         // free old one if there
         WORKSPACE_FREE(pstCompressArrays->pCoreMemory[core]);
      }

      pstCompressArrays->pCoreMemorySize[core] = wantedSize;
      pstCompressArrays->pCoreMemory[core] = WORKSPACE_ALLOC(wantedSize);

      LOGGING("-");
   }
   else {
      LOGGING("+");
   }

   void* pTempMemory = pstCompressArrays->pCoreMemory[core];

   // Make sure we were able to alloc memory
   if (pTempMemory) {

      LOGGING("[%d] started %lld %p\n", (int)t, source_size, pTempMemory);

      // data to compress is after the header
      size_t cSize = source_size;

      // Check if we compress this array or can compress this array
      if (pstCompressArrays->compType == COMPRESSION_TYPE_ZSTD) {
         // If banding is on, have to do this in steps

         if (bandCount > 0) {
            LOGGING("[%d] banding bytesperband:%lld   bandcount:%lld\n", (int)t, bytesPerBand, bandCount);

            int64_t* pBandOffsets = (int64_t*)pTempMemory;
            char* pWriteMemory = (char*)(pBandOffsets + bandCount);
            char* pStartWriteMemory = pWriteMemory;
            int64_t bytesAvailable = dest_size;
            int64_t bytesRemaining = source_size;
            char* pReadMemory = pArrayInfo->pData;

            for (int64_t i = 0; i < bandCount;) {

               i++;

               // check for last band
               if (i == bandCount) {
                  bytesPerBand = bytesRemaining;
               }

               //printf("[%lld] band %lld   %lld  avail: %lld\n", t, i, bytesPerBand, bytesAvailable);

               cSize = CompressData(COMPRESSION_TYPE_ZSTD, pWriteMemory, bytesAvailable, pReadMemory, bytesPerBand, pstCompressArrays->compLevel);

               if (cSize >= (size_t)bytesPerBand) {
                  // USE UNCOMPRESSED
                  memcpy(pWriteMemory, pReadMemory, bytesPerBand);
                  cSize = bytesPerBand;
               }
               //printf("[%lld] cband %lld   %lld \n", t, i, cSize);

               pWriteMemory += cSize;

               // The offset is to the END not the start
               *pBandOffsets++ = (pWriteMemory - pStartWriteMemory);
               bytesAvailable -= cSize;

               pReadMemory += bytesPerBand;
               bytesRemaining -= bytesPerBand;
            }

            // calculate how much we wrote
            cSize = (char*)pWriteMemory - (char*)pTempMemory;
            //printf("[%lld] ORIGINAL SIZE: %lld  COMPRESSED SIZE: %lld  remaining: %lld\n", t, source_size, cSize, bytesRemaining);
         }
         else {

            cSize = CompressData(COMPRESSION_TYPE_ZSTD, pTempMemory, dest_size, pArrayInfo->pData, source_size, pstCompressArrays->compLevel);
            // Failed to compress
            if (cSize >= (size_t)source_size) {

               // Use original uncompressed memory and size since that is smaller
               cSize = source_size;
               pTempMemory = pArrayInfo->pData;
            }
         }
      }
      else {
         // not compressing
         pTempMemory = pArrayInfo->pData;
      }


      // Get the array to compress (race condition so use interlock)
      InterlockedAdd64(&pFileHeader->ArraysWritten, 1);

      // Also race condition for addition total size
      InterlockedAdd64(&pFileHeader->TotalArrayUncompressedSize, source_size);

      int64_t arrayNumber = t;
      int64_t fileOffset = pFileHeader->AddArrayCompressedSize(cSize);

      //==============================================
      // FILL IN ARRAY BLOCK
      SDS_ARRAY_BLOCK* pArrayBlock = &pstCompressArrays->pBlockInfo[arrayNumber];

      // Record information so we can rebuild this array
      pArrayBlock->ArrayUncompressedSize = source_size;
      pArrayBlock->ArrayCompressedSize = cSize;
      pArrayBlock->ArrayDataOffset = fileOffset;
      pArrayBlock->CompressionType = COMPRESSION_TYPE_ZSTD;

      // New version 4.3
      pArrayBlock->ArrayBandCount = (int32_t)bandCount;
      pArrayBlock->ArrayBandSize = (int32_t)bandSize;

      // record array dimensions
      int32_t ndim = pArrayInfo->NDim;
      if (ndim > SDS_MAX_DIMS) {
         printf("!!!SDS: array dimensions too high: %d\n", ndim);
         ndim = SDS_MAX_DIMS;
      }

      pArrayBlock->NDim = (int8_t)ndim;

      for (int32_t i = 0; i < SDS_MAX_DIMS; i++) {
         pArrayBlock->Dimensions[i] = 0;
         pArrayBlock->Strides[i] = 0;
      }
      for (int32_t i = 0; i < ndim; i++) {
         pArrayBlock->Dimensions[i] = pArrayInfo->Dimensions[i];
         pArrayBlock->Strides[i] = pArrayInfo->Strides[i];
      }

      pArrayBlock->Flags = pArrayInfo->Flags;

      pArrayBlock->DType = pArrayInfo->NumpyDType;
      pArrayBlock->ItemSize = (int32_t)pArrayInfo->ItemSize;
      pArrayBlock->HeaderLength = sizeof(SDS_ARRAY_BLOCK);
      pArrayBlock->Magic = COMPRESSION_MAGIC;

      LOGGING("[%lld][%lld] seek to fileOffset %lld  sz: %lld\n", t, arrayNumber, fileOffset, pArrayBlock->ArrayCompressedSize);

      //===========================
      // Write compressed chunk 
      int64_t result =
         DefaultFileIO.FileWriteChunk(
            pstCompressArrays->eventHandles[core],
            pstCompressArrays->fileHandle,
            pTempMemory,
            cSize,
            fileOffset);

      if ((size_t)result != cSize) {
         printf("[%lld] error while writing into compressed  offset:%lld  sz: %zu  vs %lld\n", t, fileOffset, cSize, result);
      }
   }

   return TRUE;
}



//------------------------------------------------------------
// Return amount of memory needed to allocate
//
int64_t CalculateSharedMemorySize(SDS_FILE_HEADER* pFileHeader, SDS_ARRAY_BLOCK* pArrayBlocks) {

   // Calculate size of shared memory
   int64_t    totalSize = 0;
   totalSize += sizeof(SDS_FILE_HEADER);
   totalSize += SDS_PAD_NUMBER(pFileHeader->NameBlockSize);
   totalSize += SDS_PAD_NUMBER(pFileHeader->TotalMetaUncompressedSize);
   totalSize += SDS_PAD_NUMBER(pFileHeader->ArrayBlockSize);

   int64_t arrayCount = pFileHeader->ArraysWritten;

   if (pArrayBlocks) {
      // Add all the pNumpyArray pointers
      for (int32_t t = 0; t < arrayCount; t++) {
         totalSize += SDS_PAD_NUMBER(pArrayBlocks[t].ArrayUncompressedSize);
      }

      int64_t arraycount = pFileHeader->ArrayBlockSize / sizeof(SDS_ARRAY_BLOCK);

      if (arraycount != pFileHeader->ArraysWritten) {
         printf("possibly incomplete file %lld %lld\n", arraycount, pFileHeader->ArraysWritten);
      }
   }
   return totalSize;
}

#if defined(_WIN32)
// For windows consider MS _try to catch more exceptions
#else
#include <setjmp.h>
#include <signal.h>

jmp_buf sigbus_jmp;

/* sighandler_t is a GNU extension; if it's not available, just define it with a typedef. */
#ifndef sighandler_t
typedef void (*sighandler_t)(int);
#endif  // !defined(sighandler_t)

sighandler_t sigbus_orighandler;

void sigbus_termination_handler(int32_t signum) {
   // try to recover
   // NOTE: consider changing to siglongjmp
   longjmp(sigbus_jmp, 1);
   //printf("BUS Error while writing to linux shared memory.\n");
}
#endif

//=====================================================
// Write an SDS File (platform free -- python free)
//
// INPUT:
// fileName -- name of file to write to
//
// Arrays to write
// aInfo
// arrayCount - number of arrays
// totalItemSize -??
//
// metaData -- block of bytes to store as metadata
// metaDataSize -- 
//
bool SDSWriteFileInternal(
   const char *            fileName,
   const char *            shareName,  // can be NULL
   SDS_STRING_LIST*        pFolderName,  // can be NULL

                           // arrays to save information
   SDS_WRITE_INFO*         pWriteInfo,
   SDS_WRITE_CALLBACKS*    pWriteCallbacks)
{
   // Ferry over information

   // arrays to save information
   SDSArrayInfo* aInfo = pWriteInfo->aInfo;
   int64_t arrayCount = pWriteInfo->arrayCount;

   // meta information
   const char *metaData = pWriteInfo->metaData;
   uint32_t metaDataSize = pWriteInfo->metaDataSize;

   // names of arrays information
   char* pListNames = pWriteInfo->pListNames;
   int64_t listNameSize = pWriteInfo->listNameSize;    // total byte size (store in memory)
   int64_t listNameCount = pWriteInfo->listNameCount;   // number of names

                                                      // compressed or uncompressed
   int32_t mode = pWriteInfo->mode;            // = COMPRESSION_MODE_COMPRESS_FILE, COMPRESSION_MODE_COMPRESS_APPEND_FILE
   int32_t compType = pWriteInfo->compType;    // = COMPRESSION_TYPE_ZSTD,
   int32_t level = pWriteInfo->level;          // = ZSTD_CLEVEL_DEFAULT;

   int32_t sdsFileType = pWriteInfo->sdsFileType;
   int32_t authorId = pWriteInfo->sdsAuthorId;

   bool bAppendHeader = pWriteInfo->appendRowsMode;

   int64_t bandSize = pWriteInfo->bandSize;

   // clamp bandsize to 10K min
   if (bandSize < 10000 && bandSize > 0) bandSize = 10000;

   // TODO: put in class
   SDS_FILE_HEADER   FileHeader;
   SDS_FILE_HEADER*  pFileHeader = &FileHeader;

   // We can write with ZERO arrays
   if (aInfo || arrayCount == 0) {

      if (shareName) {
         LOGGING("Trying to store in shared memory\n");

         // Fill in fake so that we can calculate size
         FillFileHeader(pFileHeader, 0, COMPRESSION_MODE_SHAREDMEMORY, COMPRESSION_TYPE_NONE, 0, sdsFileType, 0, authorId, listNameSize, listNameCount, metaDataSize, arrayCount, bandSize);
         // NOTE: does shared memory have sections?  not for now

         pFileHeader->ArraysWritten = arrayCount;

         LOGGING("Making sharedname %s\n", shareName);

         // Try to allocate sharename
         DefaultMemoryIO.MakeShareName(fileName, shareName);

         // Calculate size of shared memory
         int64_t totalSize = CalculateSharedMemorySize(pFileHeader, NULL);

         // Add all the pNumpyArray pointers
         for (int32_t t = 0; t < arrayCount; t++) {
            totalSize += SDS_PAD_NUMBER(aInfo[t].ArrayLength * aInfo[t].ItemSize);
         }

         LOGGING("trying to allocate %lld\n", totalSize);

         // Make sure we can allocate it
         HRESULT hr = DefaultMemoryIO.Begin(totalSize);

         if (hr >= 0) {

#if defined(_WIN32)
            // For windows consider MS _try to catch more exceptions
#else
            if (setjmp(sigbus_jmp)==0) {
            // for linux handle SIGBUS, other errors
            sigbus_orighandler = signal(SIGBUS, sigbus_termination_handler);
#endif
            try {
               // Reach here, then shared memory allocated
               SDS_FILE_HEADER* pMemoryFileHeader = DefaultMemoryIO.GetFileHeader();

               LOGGING("Writing header to %p from %p\n", pMemoryFileHeader, pFileHeader);
               // Step 1: copy fileheader
               memcpy(pMemoryFileHeader, pFileHeader, sizeof(SDS_FILE_HEADER));

               LOGGING("Step 2\n");

               // Step 2: copy names
               memcpy(
                  DefaultMemoryIO.GetMemoryOffset(pMemoryFileHeader->NameBlockOffset),
                  pListNames,
                  listNameSize);

               LOGGING("Copying metadata  size:%lld \n", (long long)metaDataSize);

               // Step 3: copy metadata
               memcpy(
                  DefaultMemoryIO.GetMemoryOffset(pMemoryFileHeader->MetaBlockOffset),
                  metaData,
                  metaDataSize);

               // Get array block into shared memory
               // Step 4: Fill in this array block
               SDS_ARRAY_BLOCK* pDestArrayBlock = DefaultMemoryIO.GetArrayBlock(0);

               // offset to first location of array data
               int64_t startOffset = pMemoryFileHeader->ArrayFirstOffset;

               // Step 4: have to fill in arrayblocks
               for (int32_t arrayNumber = 0; arrayNumber < arrayCount; arrayNumber++) {

                  LOGGING("start offset %d %lld\n", arrayNumber, startOffset);

                  // same size since uncompressed
                  pDestArrayBlock[arrayNumber].ArrayCompressedSize =
                     pDestArrayBlock[arrayNumber].ArrayUncompressedSize = aInfo[arrayNumber].ArrayLength * aInfo[arrayNumber].ItemSize;

                  pDestArrayBlock[arrayNumber].ArrayDataOffset = startOffset;
                  pDestArrayBlock[arrayNumber].CompressionType = COMPRESSION_TYPE_NONE;

                  // record array dimensions
                  int32_t ndim = aInfo[arrayNumber].NDim;
                  if (ndim > SDS_MAX_DIMS) ndim = SDS_MAX_DIMS;
                  if (ndim < 1) ndim = 1;

                  pDestArrayBlock[arrayNumber].NDim = (int8_t)ndim;

                  for (int32_t j = 0; j < SDS_MAX_DIMS; j++) {
                     pDestArrayBlock[arrayNumber].Dimensions[j] = 0;
                     pDestArrayBlock[arrayNumber].Strides[j] = 0;
                  }
                  for (int32_t j = 0; j < ndim; j++) {
                     pDestArrayBlock[arrayNumber].Dimensions[j] = aInfo[arrayNumber].Dimensions[j];
                     pDestArrayBlock[arrayNumber].Strides[j] = aInfo[arrayNumber].Strides[j];
                  }
                  pDestArrayBlock[arrayNumber].Flags = aInfo[arrayNumber].Flags;
                  pDestArrayBlock[arrayNumber].DType = aInfo[arrayNumber].NumpyDType;
                  pDestArrayBlock[arrayNumber].ItemSize = (int32_t)aInfo[arrayNumber].ItemSize;
                  pDestArrayBlock[arrayNumber].HeaderLength = sizeof(SDS_ARRAY_BLOCK);
                  pDestArrayBlock[arrayNumber].Magic = COMPRESSION_MAGIC;

                  // Keep tallying
                  startOffset += SDS_PAD_NUMBER(pDestArrayBlock[arrayNumber].ArrayUncompressedSize);
               }

               LOGGING("Copying array data  %lld\n", arrayCount);

               // Step 5 (can make multithreaded) -- copy array blocks
               for (int32_t arrayNumber = 0; arrayNumber < arrayCount; arrayNumber++) {
                  //LOGGING("Writing to offset %lld, length %lld, memptr %p\n", pDestArrayBlock[arrayNumber].ArrayDataOffset, pDestArrayBlock[arrayNumber].ArrayUncompressedSize, DefaultMemoryIO.GetMemoryOffset(pDestArrayBlock[arrayNumber].ArrayDataOffset));

                  memcpy(
                     DefaultMemoryIO.GetMemoryOffset(pDestArrayBlock[arrayNumber].ArrayDataOffset),
                     aInfo[arrayNumber].pData,
                     pDestArrayBlock[arrayNumber].ArrayUncompressedSize);
               }

               LOGGING("Succeeded in copying memory\n");
               // Keep handle open?
            }
            catch (...) {
               // NOTE this does not catch bus errors which occur when out of disk space
               printf("!!!Failed to write all of shared memory.  May be out of shared memory.\n");
            }
#if defined(_WIN32)
               // For windows we cannot close it since that will free the memory back
#else
         } else {  // the else is hit when the SIGBUS signal was sent
               printf("BUS Error while writing to shared memory, check swap space.\n");
         }
         // For linux restore signal handler
         signal(SIGBUS, sigbus_orighandler);
         // For linux ok to close it since /dev/shm has it
         DefaultMemoryIO.Destroy();
#endif
         }
         else {
            printf("!!Failed to create shared memory\n");
         }
      }
      else {
         //----------------------------------------------
         // NO sharename specified
         //----------------------------------------------
         // -- Try to open file that we will compress into

         // Are we APPENDING a section?

         SDS_FILE_HANDLE sdsFile =
            StartCompressedFile(
               fileName,
               pFileHeader,
               (int16_t)compType,
               level,
               sdsFileType,
               0,                   // stackType
               authorId,
               pListNames,
               listNameSize,
               listNameCount,
               metaData,
               metaDataSize,
               arrayCount,
               mode,
               pFolderName,
               bandSize,
               pWriteInfo
               );

         LOGGING("Current fileoffset is %lld\n", pFileHeader->FileOffset);
         if (sdsFile) {

            // Allocate an arrayblock and zero it out
            SDS_ARRAY_BLOCK* pArrayBlocks = (SDS_ARRAY_BLOCK*)WORKSPACE_ALLOC(pFileHeader->ArrayBlockSize);
            memset(pArrayBlocks, 0, pFileHeader->ArrayBlockSize);

            //------------------ Allocate struct to hold information during processing
            //SDS_WRITE_COMPRESS_ARRAYS* pstCompressArrays = (SDS_WRITE_COMPRESS_ARRAYS*)WORKSPACE_ALLOC(sizeof(SDS_WRITE_COMPRESS_ARRAYS) + (arrayCount * sizeof(SDSArrayInfo)));
            SDS_WRITE_COMPRESS_ARRAYS* pstCompressArrays = (SDS_WRITE_COMPRESS_ARRAYS*)WORKSPACE_ALLOC(sizeof(SDS_WRITE_COMPRESS_ARRAYS));
            pstCompressArrays->totalHeaders = arrayCount;

            //---------------------
            if (level <= 0) level = ZSTD_CLEVEL_DEFAULT;
            if (level > ZSTD_MAX_CLEVEL) level = ZSTD_MAX_CLEVEL;

            pstCompressArrays->compLevel = level;
            pstCompressArrays->compType = compType;
            pstCompressArrays->compMode = (int16_t)mode;

            pstCompressArrays->pArrayInfo = aInfo;
            pstCompressArrays->fileHandle = sdsFile;
            pstCompressArrays->pFileHeader = pFileHeader;
            pstCompressArrays->pBlockInfo = pArrayBlocks;

            // Make sure there are arrays to write
            if (arrayCount > 0) {
               int32_t numCores = g_cMathWorker->GetNumCores();

               for (int32_t j = 0; j < numCores; j++) {
                  pstCompressArrays->pCoreMemory[j] = NULL;
                  pstCompressArrays->pCoreMemorySize[j] = 0;
                  pstCompressArrays->eventHandles[j] = DefaultFileIO.CreateEventHandle();
               }

               void* saveState = pWriteCallbacks->BeginAllowThreads();
               // This will kick off the workerthread and call CompressFileArray passing pstCompressArrays as argument with counter and core
               g_cMathWorker->DoMultiThreadedWork((int)arrayCount, CompressFileArray, pstCompressArrays);
               pWriteCallbacks->EndAllowThreads(saveState);

               LOGGING("End of compressing\n");

               for (int32_t j = 0; j < numCores; j++) {
                  if (pstCompressArrays->pCoreMemory[j]) {
                     WORKSPACE_FREE(pstCompressArrays->pCoreMemory[j]);
                  }
                  DefaultFileIO.DestroyEventHandle(pstCompressArrays->eventHandles[j]);
               }
            }

            // TODO: Have class that holds sdsdFile, fileheader
            EndCompressedFile(sdsFile, pFileHeader, pArrayBlocks, pWriteInfo);

            WORKSPACE_FREE(pArrayBlocks);
            WORKSPACE_FREE(pstCompressArrays);
         }
         else {
            LOGGING("Failure to start compressed file %s\n", fileName);
         }
      }
   }

   return TRUE;
}





//---------------------------------------------------------
// Linux: long = 64 bits
// Windows: long = 32 bits
static int32_t FixupDType(int32_t dtype, int64_t itemsize) {

   if (dtype == SDS_LONG) {
      // types 7 and 8 are ambiguous because of different compilers
      if (itemsize == 4) {
         dtype = SDS_INT;
      }
      else {
         dtype = SDS_LONGLONG;
      }
   }

   if (dtype == SDS_ULONG) {
      // types 7 and 8 are ambiguous
      if (itemsize == 4) {
         dtype = SDS_UINT;
      }
      else {
         dtype = SDS_ULONGLONG;
      }
   }
   return dtype;
}


//-------------------------------------------------------
//
void CopyFromBlockToInfo(SDS_ARRAY_BLOCK*  pBlock, SDSArrayInfo* pDestInfo) {

   int32_t dtype = FixupDType(pBlock->DType, pBlock->ItemSize);

   // IS THIS EVER USED??
   pDestInfo->NDim = pBlock->NDim;
   pDestInfo->NumpyDType = dtype;
   pDestInfo->ItemSize = pBlock->ItemSize;
   pDestInfo->Flags = pBlock->Flags;

   int32_t ndim = pBlock->NDim;
   if (ndim > SDS_MAX_DIMS) { ndim = SDS_MAX_DIMS; }

   pDestInfo->ArrayLength = 1;

   // Fill in strides, dims, and calc arraylength
   for (int32_t i = 0; i < ndim; i++) {
      pDestInfo->ArrayLength *= pBlock->Dimensions[i];
      pDestInfo->Dimensions[i] = pBlock->Dimensions[i];
      pDestInfo->Strides[i] = pBlock->Strides[i];
   }

   pDestInfo->NumBytes = pDestInfo->ArrayLength * pDestInfo->ItemSize;

   // TODO: check?  vs pBlock->ArrayUncompressedSize;
   //if (pDestInfo->NumBytes != pBlock->ArrayUncompressedSize) {
   //   printf("!!!Error CopyFromBlockToInfo  %lld  vs %lld\n", pDestInfo->NumBytes, pBlock->ArrayUncompressedSize);
   //}

}



struct SDS_COMPATIBLE {
   int8_t                 IsCompatible;  // set to FALSE if conversion required
   int8_t                 NeedsStringFixup;  // set to 1 if conversion required, or in 2 for mtlab conversion
   int8_t                 NeedsConversion;  // if flag set, dtype conversion called
   int8_t                 NeedsRotation;
};

//=============================================================
// MultIO struct (multiple file handles)
//
//-----------------------------------------------------------
struct SDS_IO_PACKET {
   SDS_READ_CALLBACKS*  pReadCallbacks;
   SDS_ARRAY_BLOCK*     pBlockInfo;
   SDS_ARRAY_BLOCK*     pMasterBlock;

   // used when decompressing
   SDS_FILE_HANDLE      FileHandle;
   SDS_FILE_HEADER*     pFileHeader;

   // Used when going to shared memory
   class SharedMemory*  pMemoryIO;

   int16_t                CompMode;
   int8_t                 ReservedMode1;
   int8_t                 ReservedMode2;
   SDS_COMPATIBLE       Compatible;

   // Used in matlab string rotation
   int64_t                ArrayOffset;
   int64_t                OriginalArrayOffset;
   int64_t                StackPosition;
   const char*          ColName;
};

struct SDS_MULTI_IO_PACKETS {
   // Per core allocations
   void*                pCoreMemory[SDS_MAX_CORES];
   int64_t                pCoreMemorySize[SDS_MAX_CORES];
   SDS_EVENT_HANDLE     eventHandles[SDS_MAX_CORES];

   SDSArrayInfo*        pDestInfo;

   // New Filtering
   SDS_FILTER*          pFilter;

   // See: compMode value -- COMPRESSIOM_MODE_SHAREDMEMORY
   // used when compressing to file

   // This is the callback from SDS_ALLOCATE_ARRAYS
   // The bottom end of this structure is allocated based on how many arrays
   SDS_IO_PACKET        pIOPacket[1];


   //--------------------------------------
   // Allocates pMultiIOPackets->pDestInfo
   // Allocates event handles
   static SDS_MULTI_IO_PACKETS* Allocate(int64_t tupleSize, SDS_FILTER*  pFilter) {
      int64_t allocSize = sizeof(SDS_MULTI_IO_PACKETS) + ((sizeof(SDS_IO_PACKET) * tupleSize));
      SDS_MULTI_IO_PACKETS* pMultiIOPackets = (SDS_MULTI_IO_PACKETS*)WORKSPACE_ALLOC(allocSize);
      memset(pMultiIOPackets, 0, allocSize);

      pMultiIOPackets->pFilter = pFilter;

      int64_t allocSizeDestInfo = tupleSize * sizeof(SDSArrayInfo);
      pMultiIOPackets->pDestInfo = (SDSArrayInfo*)WORKSPACE_ALLOC(allocSizeDestInfo);
      memset(pMultiIOPackets->pDestInfo, 0, allocSizeDestInfo);

      // Set all the cores working memory to zero
      int32_t numCores = g_cMathWorker->WorkerThreadCount + 1;
      for (int32_t j = 0; j < numCores; j++) {
         pMultiIOPackets->eventHandles[j] = DefaultFileIO.CreateEventHandle();
      }
      return pMultiIOPackets;
   }

   //------------------------------------
   // Free what was allocated earlier
   static void Free(SDS_MULTI_IO_PACKETS* pMultiIOPackets) {
      //---------- CLEAN UP MEMORY AND HANDLES ---------
      // check if any cores allocated any memory
      // Set all the cores working memory to zero
      int32_t numCores = g_cMathWorker->WorkerThreadCount + 1;
      for (int32_t j = 0; j < numCores; j++) {
         if (pMultiIOPackets->pCoreMemory[j]) {
            WORKSPACE_FREE(pMultiIOPackets->pCoreMemory[j]);
            pMultiIOPackets->pCoreMemory[j] = NULL;
         }
         DefaultFileIO.DestroyEventHandle(pMultiIOPackets->eventHandles[j]);
      }
      WORKSPACE_FREE(pMultiIOPackets->pDestInfo);
      pMultiIOPackets->pDestInfo = NULL;

      WORKSPACE_FREE(pMultiIOPackets);
   }
};


//-----------------------------------------------------------
// Check both folder and column names
bool IsNameIncluded(
   SDSIncludeExclude*    pInclude,
   SDSIncludeExclude*    pFolderName,
   const char*           nameToCheck,
   bool                  isOneFile) {

   if (isOneFile) {
      // Only OneFile can have folders
      // If Folders have been specified but not names -- get the foldername and check
      // All folder names are assumed to end in /
      //
      // If both folders and NAMES -- both the folder names and the column name must match (or just pure path)
      // If just names - problem because the foldername has meta data, and that foldername has to be loaded
      // If just names, see if a pure folder name?
      if (!pInclude || pInclude->IsEmpty()) {
         // we have no column name included
         if (!pFolderName || pFolderName->IsEmpty()) {
            return TRUE;
         }

         // we have a folder name but no column name (any match works)
         return pFolderName->IsIncluded(nameToCheck);
      }

      if (pFolderName && !pFolderName->IsEmpty()) {
         // both must match OR a pure folder match
         // Check for pure folder match here
         // TODO: code to write to see if the name is a folder.
         const char *pStart = nameToCheck;
         while (*pStart) pStart++;
         pStart--;
         if (pStart >= nameToCheck) {
            if (*pStart == '/') {
               // pure folder name -- if it matches, it contains meta that we need
               if (pFolderName->IsIncluded(nameToCheck)) return TRUE;
            }
         }
         
      }
      else {
         // just a name -- no folder -- probably not allowed
         //return FALSE;
      }

      if ((!pInclude || pInclude->IsIncluded(nameToCheck)) &&
         (!pFolderName || pFolderName->IsIncluded(nameToCheck))) {
         return TRUE;
      }
   }
   else {
      if (!pInclude || pInclude->IsIncluded(nameToCheck)) {
         return TRUE;
      }

   }
   return FALSE;
}


//===================================
// Returns True if array was shrunk or had a mask
bool PossiblyShrinkArray(
   SDS_ALLOCATE_ARRAY*  pArrayCallback,
   SDS_READ_CALLBACKS*  pReadCallbacks,
   bool                 isStackable
) {
   bool wasFiltered = FALSE;

   // Categoricals will not be in original container
   int32_t mask = SDS_FLAGS_ORIGINAL_CONTAINER;
   if ((pArrayCallback->sdsFlags & (SDS_FLAGS_SCALAR | SDS_FLAGS_META | SDS_FLAGS_NESTED)) == 0) {
      if (pArrayCallback->sdsFlags & mask) {
         // Did they pass a fancy index or a bool index?
         if (pReadCallbacks->Filter.pBoolMask && pReadCallbacks->Filter.BoolMaskTrueCount >= 0 && isStackable) {
            wasFiltered = TRUE;
            int64_t dim0Length = pArrayCallback->dims[0];
            int64_t newLength = pReadCallbacks->Filter.BoolMaskTrueCount;

            // If the array allocation is too large, we reduce it
            if (newLength > dim0Length) {
               newLength = dim0Length;
            }

            LOGGING("-->Allocation reduction for %s filter: %lld  vs %lld  bool:%lld  flags:%d\n", pArrayCallback->pArrayName, newLength, dim0Length, pReadCallbacks->Filter.BoolMaskTrueCount, pArrayCallback->sdsFlags);
            pArrayCallback->dims[0] = newLength;
         }
      }
      else {
         // Could be a categorical
         //int64_t dim0Length = pArrayCallback->dims[0];
         //if (pReadCallbacks->Filter.pBoolMask) {
         //   LOGGING("-->Allocation no reduction for %s  len:%lld  fancy:%lld  bool:%lld  flags:%d\n", pArrayCallback->pArrayName, dim0Length, pReadCallbacks->Filter.FancyLength, pReadCallbacks->Filter.BoolMaskTrueCount, pArrayCallback->sdsFlags);
         //}
      }

   }
   else {
      // Meta or scalar - nothing to do
   }
   return wasFiltered;
}


//=============================================================
//
//
class SDSDecompressFile {

public:
   // These three are required
   const char*      FileName;       // Fully qualified file path
   const char*      ShareName;      // May be NULL
   int64_t            InstanceIndex;  // 0 if just one instance (for multiday)
   int32_t            Mode;
   int64_t            FileSize;       // only valid when concat

   SDSIncludeExclude*    pInclude;    // Inclusion list
   SDSIncludeExclude*    pFolderName; // Folder inclusion
   SDSIncludeExclude*    pSectionsName; // sections inclusion

                                      // Additional
   SDS_FILE_HEADER  FileHeader;
   SDS_FILE_HEADER* pFileHeader = &FileHeader;

   // Compress arrays
   SDS_READ_DECOMPRESS_ARRAYS* pstCompressArrays = NULL;

   // Array Block
   SDS_FILE_HANDLE  SDSFile = BAD_SDS_HANDLE;
   SDS_ARRAY_BLOCK* pArrayBlocks = NULL;

   // Meta data
   int64_t MetaSize = 0;
   char* MetaData = NULL;

   // Name data
   char*          pNameData = NULL;
   const char**   pArrayNames = NULL;
   int32_t*         pArrayEnums = NULL;
   int64_t          NameCount = 0;

   // Section data
   SDSSectionName cSectionName;

   // Set to true when file header is read (does not work for shared memory file)
   bool           IsFileValid = FALSE;
   bool           IsFileValidAndNotFilteredOut = FALSE;

   //------------------------------------------------
   // constructor
   // pInclude = NULL is allowed
   // shareName NULL is allowed
   SDSDecompressFile(
      const char*          fileName, 
      SDSIncludeExclude*   pInclude = NULL,
      int64_t                instanceIndex=0, 
      const char*          shareName=NULL,
      SDSIncludeExclude*   pFolderName=NULL,
      SDSIncludeExclude*   pSectionsName = NULL,
      int32_t                mode = COMPRESSION_MODE_INFO) {

      this->FileName = fileName;
      this->pInclude = pInclude;
      this->InstanceIndex = instanceIndex;
      this->ShareName = shareName;
      this->pFolderName = pFolderName;
      this->pSectionsName = pSectionsName;
      this->Mode = mode;

   }

   //-----------------------------------------------------------
   // Caller must fill in pDestInfo->pData because memory will be read into
   //// returns TRUE if array allocated
   //bool CallAllocateArray(SDS_READ_CALLBACKS* pReadCallbacks, SDS_ALLOCATE_ARRAY *pAllocateArray) {

   //   if (pReadCallbacks->AllocateArrayCallback) {
   //      pReadCallbacks->AllocateArrayCallback(pAllocateArray);
   //      return TRUE;
   //   }
   //   return FALSE;
   //}


   //===================================
   // Prepare an array callback structure
   // Calls into the user specified callback
   // colPos is used to lookup pArrayBlocks
   //
   // Callee must return:
   // pDestInfo filled in with pData (location of first element in array)
   void AllocateOneArray(
      int32_t                  colPos,          // which array index (which column in the file)
      SDS_READ_CALLBACKS*  pReadCallbacks,
      SDSArrayInfo*        pDestInfo,
      bool                 isSharedMemory,
      bool                 isOneFile,
      bool                 isStackable) {

      SDS_ARRAY_BLOCK*  pBlock = &pArrayBlocks[colPos];

      // Allocate all the arrays before multithreading
      // NOTE: do we care about flags -- what if Fortran mode when saved?
      int32_t dtype = FixupDType(pBlock->DType, pBlock->ItemSize);

      // Build array callback block
      SDS_ALLOCATE_ARRAY sdsArrayCallback;
      sdsArrayCallback.pDestInfo = pDestInfo;
      sdsArrayCallback.ndim = pBlock->NDim;
      sdsArrayCallback.dims = pBlock->Dimensions;
      sdsArrayCallback.numpyType = dtype;
      sdsArrayCallback.itemsize = pBlock->ItemSize;
      sdsArrayCallback.data = NULL;  // for shared memory this is set

      sdsArrayCallback.numpyFlags = pBlock->Flags;
      sdsArrayCallback.strides = pBlock->Strides;

      // Handle error cases
      sdsArrayCallback.pArrayName = pArrayNames ? pArrayNames[colPos]: NULL;
      sdsArrayCallback.sdsFlags = pArrayEnums ? pArrayEnums[colPos]: 0;
      //sdsArrayCallback.pInclusionList = pReadCallbacks->pInclusionList;
      //sdsArrayCallback.pExclusionList = pReadCallbacks->pExclusionList;

      if (isSharedMemory) {
         pDestInfo->pData =
            sdsArrayCallback.data = DefaultMemoryIO.GetMemoryOffset(pBlock->ArrayDataOffset);
         LOGGING("Shared memory set to %p  %lld  memoryoffset: %p\n", sdsArrayCallback.data, pBlock->ArrayDataOffset, DefaultMemoryIO.GetMemoryOffset(0));
      }

      //-----------------------------

      // set to NULL incase excluded or failed to allocate
      pDestInfo->pArrayObject = NULL;
      pDestInfo->pData = NULL;

      bool wasFiltered = FALSE;

      //LOGGING("Checking to allocate name %s\n", pAllocateArray->pArrayName);
      // Include exclude check
      if (!pArrayNames || IsNameIncluded(pInclude, pFolderName, sdsArrayCallback.pArrayName, isOneFile))
      {

         //-----------------------------------
         // Caller will fill info pArrayObject and pData
         // NOTE: this routine can filter names (will return FALSE if filtered out)
         // pData is valid for shared memory
         if (pReadCallbacks->AllocateArrayCallback) {

            // FILTERING check...
            wasFiltered |=
            PossiblyShrinkArray(&sdsArrayCallback, pReadCallbacks, isStackable);

            pReadCallbacks->AllocateArrayCallback(&sdsArrayCallback);

            LOGGING("Allocating dims:%d -- %lld %lld %lld   flags:%d  itemsize:%d  wasFiltered: %d\n", (int)pBlock->NDim, pBlock->Dimensions[0], pBlock->Dimensions[1], pBlock->Dimensions[2], pBlock->Flags, pBlock->ItemSize, wasFiltered);
            LOGGING("Strides  %lld %lld %lld  \n", pBlock->Strides[0], pBlock->Strides[1], pBlock->Strides[2]);
         }
      }

      // Fill in destination information
      CopyFromBlockToInfo(pBlock, pDestInfo);
      if (wasFiltered) {
         pDestInfo->Flags |= SDS_ARRAY_FILTERED;
         pBlock->Flags |= SDS_ARRAY_FILTERED;
      }
   }

   //-----------------------------------------------------------
   // Fills in multiple io packets (tupleSize items)
   //
   void AllocMultiArrays(
      SDS_IO_PACKET*       pIOPacket,
      SDSArrayInfo*        pDestInfo,
      SDS_READ_CALLBACKS*  pReadCallbacks,
      int64_t                tupleSize,
      bool                 isSharedMemory
   ) {

      bool oneFile = (pFileHeader->FileType == SDS_FILE_TYPE_ONEFILE);

      // Init all the pNumpyArray pointers
      for (int32_t t = 0; t < tupleSize; t++) {

         pIOPacket->pReadCallbacks = pReadCallbacks;
         pIOPacket->pBlockInfo = &pArrayBlocks[t];
         pIOPacket->pMasterBlock = NULL;
         pIOPacket->Compatible = { TRUE, FALSE, FALSE, FALSE };
         pIOPacket->ArrayOffset = 0;
         pIOPacket->OriginalArrayOffset = 0;
         pIOPacket->StackPosition = 0;
         pIOPacket->ColName = NULL;

         if (isSharedMemory) {
            pIOPacket->FileHandle = SDSFile;
            pIOPacket->pFileHeader = pFileHeader;
            pIOPacket->pMemoryIO = &DefaultMemoryIO; // WHAT TO DO??
            pIOPacket->CompMode = COMPRESSION_MODE_SHAREDMEMORY;
         }
         else {
            pIOPacket->FileHandle = SDSFile;
            pIOPacket->pFileHeader = pFileHeader;
            pIOPacket->pMemoryIO = &DefaultMemoryIO; // WHAT TO DO??
            pIOPacket->CompMode = COMPRESSION_MODE_DECOMPRESS;
         }

         int32_t fileType = pFileHeader->FileType;
         int32_t fileTypeStackable = (fileType == SDS_FILE_TYPE_DATASET ||
            fileType == SDS_FILE_TYPE_TABLE ||
            fileType == SDS_FILE_TYPE_ARRAY);

         // callback into python or matlab to allocate memory
         AllocateOneArray(t, pReadCallbacks, &pDestInfo[t], FALSE, oneFile, fileTypeStackable);

         // next io packet to build
         pIOPacket++;
      }

   }

   //-----------------------------------------------------------
   // Called before multithreading due to the GIL to allocate the numpy arrays
   // READING data
   //
   // To allocate arrays, it will callback tupleSize times
   // The callback will fill in an array of SDSArrayInfo  (pData is the most important location)
   //
   // Allocates pstCompressArrays
   // Allocates numpy arrays
   // NOTE: These must be freed later
   // RETURNS SDS_READ_DECOMPRESS_ARRAYS + SDSArrayInfo*tupleSize
   SDS_READ_DECOMPRESS_ARRAYS* AllocDecompressArrays(
      SDS_READ_CALLBACKS*  pReadCallbacks,
      int64_t                tupleSize,
      bool                 isSharedMemory,
      bool                 isOneFile,
      bool                 isStackable) {

      // TJD
      // NOTE: TODO, should check isincluded up front

      SDS_READ_DECOMPRESS_ARRAYS* pstDecompressArrays = (SDS_READ_DECOMPRESS_ARRAYS*)WORKSPACE_ALLOC(sizeof(SDS_READ_DECOMPRESS_ARRAYS) + (tupleSize * sizeof(SDSArrayInfo)));
      pstDecompressArrays->totalHeaders = tupleSize;

      // Init all the pNumpyArray pointers
      for (int32_t t = 0; t < tupleSize; t++) {

         SDSArrayInfo*     pDestInfo = &pstDecompressArrays->ArrayInfo[t];

         AllocateOneArray(
            t,
            pReadCallbacks,
            pDestInfo,
            isSharedMemory,
            isOneFile,
            isStackable);
      }

      pstDecompressArrays->pReadCallbacks = pReadCallbacks;
      pstDecompressArrays->pBlockInfo = pArrayBlocks;
      pstDecompressArrays->fileHandle = BAD_SDS_HANDLE;
      pstDecompressArrays->pFileHeader = NULL;

      pstDecompressArrays->pMemoryIO = NULL;
      pstDecompressArrays->compMode = (int16_t)COMPRESSION_MODE_DECOMPRESS;

      return pstDecompressArrays;
   }


   //========================
   // returns pArrayBlocks on success which is not NULL
   // sets pArrayBlocks
   //
   SDS_ARRAY_BLOCK* AllocateArrayBlocks() {
      if (pArrayBlocks != NULL) {
         printf("Double Allocation array blocks!!\n");
      }
      pArrayBlocks = (SDS_ARRAY_BLOCK*)WORKSPACE_ALLOC(pFileHeader->ArrayBlockSize);

      if (pArrayBlocks == NULL) {
         SetErr_Format(SDS_VALUE_ERROR, "Allocation of size %lld for arrayblocks failed.\n", pFileHeader->ArrayBlockSize);
      }
      return pArrayBlocks;
   }

   //----------------------------------------
   // Delete only if allocated
   void DeleteArrayBlocks() {
      if (pArrayBlocks != NULL) {
         WORKSPACE_FREE(pArrayBlocks);
         pArrayBlocks = NULL;
      }
   }

   //--------------------------------------------
   //
   char* AllocateMetaData(int64_t size) {
      if (MetaData != NULL) {
         printf("Double Allocation meta data!!\n");
      }
      MetaData = (char*)WORKSPACE_ALLOC(size);
      if (MetaData) {
         MetaSize = size;
      }
      return MetaData;
   }

   //----------------------------------------
   // Delete only if allocated
   void DeleteMetaData() {
      if (MetaData) {
         WORKSPACE_FREE(MetaData);
         MetaData = NULL;
         MetaSize = 0;
      }
   }

   //--------------------------------------------
   // Also zero out pArrayNames, zero out pArrayEnums
   //
   void AllocateNameData(int64_t nameBlockCount, int64_t nameSize) {
      NameCount = nameBlockCount;
      if (pNameData != NULL) {
         printf("Double Allocation nameData!!\n");
      }
      pNameData = (char*)WORKSPACE_ALLOC(nameSize);

      // ZERO OUT
      pArrayNames = (const char**)WORKSPACE_ALLOC(NameCount * sizeof(void*));
      for (int32_t i = 0; i < NameCount; i++) {
         pArrayNames[i] = NULL;
      }
      pArrayEnums = (int32_t*)WORKSPACE_ALLOC(NameCount * sizeof(int32_t));
      for (int32_t i = 0; i < NameCount; i++) {
         pArrayEnums[i] = 0;
      }
   }

   //----------------------------------------
   // Delete only if allocated
   void DeleteNameData() {
      if (pNameData) {
         WORKSPACE_FREE(pNameData);
         pNameData = NULL;
      }
      if (pArrayNames) {
         WORKSPACE_FREE(pArrayNames);
         pArrayNames = NULL;
      }
      if (pArrayEnums) {
         WORKSPACE_FREE(pArrayEnums);
         pArrayEnums = NULL;
      }
   }

   //------------------------------------------
   // Cleanup
   // Cleanup memory
   // Close file handle
   void EndDecompressedFile() {
      LOGGING("End decompressed file\n");

      if (SDSFile != BAD_SDS_HANDLE) {
         DefaultFileIO.FileClose(SDSFile);
         SDSFile = BAD_SDS_HANDLE;
      }

      DeleteArrayBlocks();
      DeleteMetaData();
      DeleteNameData();

      if (pstCompressArrays) {
         WORKSPACE_FREE(pstCompressArrays);
         pstCompressArrays = NULL;
      }
   }

   //------------------------------------------
   // Understand how to get to sections
   int64_t GetTotalArraysWritten() {
      // todo:
      LOGGING("Section offsets %p   psectiondata %p   sectioncount: %lld\n", cSectionName.pSectionOffsets, cSectionName.pSectionData, cSectionName.SectionCount);

      return pFileHeader->ArraysWritten;
   }


   //===============================
   // Close file handles and release memory
   ~SDSDecompressFile() {
      EndDecompressedFile();
   }


   //---------------------------------------------------------------
   // Open files, make sure file is ok
   // Input: File to read
   //        SDS_FILE_HEADER to read into
   //
   // Output: reads into pFileHeader and returns a good file handle
   //         or returns BAD_SDS_HANDLE on failure
   SDS_FILE_HANDLE
   StartDecompressedFile(const char* fileName, int64_t fileOffset) {

      SDS_FILE_HANDLE  fileHandle =
         DefaultFileIO.FileOpen(fileName, false, true, false, false);

      if (!fileHandle) {
         SetErr_Format(SDS_VALUE_ERROR, "Decompression error cannot create/open file: %s.  Error: %s", fileName, GetLastErrorMessage());
         return BAD_SDS_HANDLE;
      }

      int64_t result =
         ReadFileHeader(fileHandle, pFileHeader, fileOffset, fileName);

      if (result != 0) {
         return BAD_SDS_HANDLE;
      }

      // read in meta data!
      return fileHandle;
   }



   //-------------------------------------------------------
   // Decompress into a STRING object or a location in memory
   // Input:
   //   FileHeader and filehandle
   //   NULL OR metaDataUncompressed but not both
   //   if metaDataUncompressed is not NULL, then coming from CopyIntoBuffer
   //
   //  sdsFile (current file handle)
   //  ppArrayBlock (currently allocated)
   //
   // Returns:
   //   TRUE or FALSE.  if TRUE and NULL passed in then can call GetMetaData()
   //   
   //
   bool DecompressMetaData(
      SDS_FILE_HEADER*  pFileHeader,
      char*             metaDataUncompressed,
      int32_t               core) {

      int64_t metaCompressedSize = pFileHeader->TotalMetaCompressedSize;
      char* metaData = NULL;

      // Read in metadata from disk/network
      int64_t bytesRead = 0;

      LOGGING("in decompress meta %lld vs %lld  offset:%lld  handle:%p\n", metaCompressedSize, pFileHeader->TotalMetaUncompressedSize, pFileHeader->MetaBlockOffset, SDSFile);

      // Did we compress the meta data?
      // If the decompressed size is the same as uncompressed, then it was never compressed
      if (metaCompressedSize && metaCompressedSize != pFileHeader->TotalMetaUncompressedSize) {

         // META DATA IS COMPRESSED ----------------------------------
         char* metaDataCompressed = (char*)WORKSPACE_ALLOC(metaCompressedSize);

         if (!metaDataCompressed) {
            SetErr_Format(SDS_VALUE_ERROR, "Decompression error in metaDataCompressedsize: %lld", metaCompressedSize);
            return FALSE;
         }

         // Read in compressed data into our buffer: metaDataCompressed
         bytesRead = DefaultFileIO.FileReadChunk(NULL, SDSFile, metaDataCompressed, metaCompressedSize, pFileHeader->MetaBlockOffset);

         if (bytesRead != metaCompressedSize) {
            WORKSPACE_FREE(metaDataCompressed);
            SetErr_Format(SDS_VALUE_ERROR, "Decompression error in bytesRead: %lld", metaCompressedSize);
            return FALSE;
         }

         int64_t metaUncompressedSize = pFileHeader->TotalMetaUncompressedSize;

         // check if user passed in buffer to copy into
         if (!metaDataUncompressed) {

            // decompress into our buffer
            metaDataUncompressed = AllocateMetaData(metaUncompressedSize);

            // allocate a string object
            if (!metaDataUncompressed) {
               SetErr_Format(SDS_VALUE_ERROR, "Decompression error meta: could not allocate meta string %lld", metaUncompressedSize);
               return FALSE;
            }

            // get the memory of the pystring
            // metaDataUncompressed = PyBytes_AS_STRING(pystring);
         }

         // decompress meta data into metaDataUncompressed
         int64_t cSize = DecompressData(NULL, COMPRESSION_TYPE_ZSTD, metaDataUncompressed, pFileHeader->TotalMetaUncompressedSize, metaDataCompressed, pFileHeader->TotalMetaCompressedSize);

         WORKSPACE_FREE(metaDataCompressed);
         metaDataCompressed = NULL;

         if (CompressIsError(COMPRESSION_TYPE_ZSTD, cSize) || cSize != metaUncompressedSize) {
            SetErr_Format(SDS_VALUE_ERROR, "Decompression error meta: length mismatch -> decomp %llu != %llu [header]", (uint64_t)cSize, metaUncompressedSize);
            return FALSE;
         }

      }
      else {
         int64_t metaUncompressedSize = metaCompressedSize;

         // meta was never compressed
         LOGGING("meta was not compressed\n");

         // user supplying buffer?
         if (!metaDataUncompressed) {

            // decompress into our internal buffer (not user's bufer)
            metaDataUncompressed = AllocateMetaData(metaUncompressedSize);

            // allocate a string object
            if (!metaDataUncompressed) {
               SetErr_Format(SDS_VALUE_ERROR, "Decompression error meta: could not allocate meta string %lld", metaUncompressedSize);
               return FALSE;
            }
         }

         // read uncompressed data in
         bytesRead = DefaultFileIO.FileReadChunk(NULL, SDSFile, metaDataUncompressed, metaUncompressedSize, pFileHeader->MetaBlockOffset);
      }

      LOGGING("out decompress meta\n");
      return TRUE;
   }

   //------------------------------------------------
   // Returns FALSE on failure
   // Returns TRUE on success
   // 
   // Reads from File and decompresses into shared memory
   // Call EndDecompressedFile() when done
   bool
   CopyIntoSharedMemoryInternal(SDS_READ_CALLBACKS* pReadCallbacks, int32_t core) {

      if (SDSFile) {

         int64_t nameSize = pFileHeader->NameBlockSize;
         int64_t metaSize = pFileHeader->TotalMetaCompressedSize;

         // This will allocate pArrayBlocks
         if (!AllocateArrayBlocks()) {
            SetErr_Format(SDS_VALUE_ERROR, "Decompression error in pArrayBlock: %lld", pFileHeader->ArrayBlockSize);
            return FALSE;
         }

         int64_t bytesRead = DefaultFileIO.FileReadChunk(NULL, SDSFile, pArrayBlocks, pFileHeader->ArrayBlockSize, pFileHeader->ArrayBlockOffset);
         if (bytesRead != pFileHeader->ArrayBlockSize) {
            SetErr_Format(SDS_VALUE_ERROR, "Decompression error in ArrayBlockSize: %lld", pFileHeader->ArrayBlockSize);
            return FALSE;
         }

         // Calculate size of shared memory
         int64_t totalSize = CalculateSharedMemorySize(pFileHeader, pArrayBlocks);
         int64_t arrayCount = pFileHeader->ArraysWritten;

         LOGGING("CopyIntoSharedMem: trying to allocate %lld\n", totalSize);

         // Make sure we can allocate it
         HRESULT hr = DefaultMemoryIO.Begin(totalSize);

         if (hr >= 0) {
            // 
            // Fileheader for shared memory
            SDS_FILE_HEADER* pMemoryFileHeader = DefaultMemoryIO.GetFileHeader();

            LOGGING("Fill file header   namecount:%lld   arrayswritten:%lld  address %p\n", pFileHeader->NameBlockCount, pFileHeader->ArraysWritten, pMemoryFileHeader);

            // The memory header
            // Shared memory is stored uncompressed
            FillFileHeader(
               pMemoryFileHeader,
               0,
               COMPRESSION_MODE_SHAREDMEMORY,
               COMPRESSION_TYPE_NONE,
               0, // comp level
               pFileHeader->FileType,
               0, // stack Type
               pFileHeader->AuthorId,
               pFileHeader->NameBlockSize,
               pFileHeader->NameBlockCount,
               pFileHeader->TotalMetaUncompressedSize,
               pFileHeader->ArraysWritten,
               0);

            LOGGING("Fill name block\n");

            // Read name block
            DefaultFileIO.FileReadChunk(NULL, SDSFile, DefaultMemoryIO.GetMemoryOffset(pMemoryFileHeader->NameBlockOffset), pFileHeader->NameBlockSize, pFileHeader->NameBlockOffset);


            //------------- LIST NAMES -------------------------------
            // NO LIST NAMES ARE ALLOWED (root file?)
            if (pMemoryFileHeader->NameBlockSize && ReadListNames() == NULL) {
               //goto EXIT_DECOMPRESS;
            }

            LOGGING("Fill meta block\n");

            // Read from file and write to shared memory
            bool bResult =
               DecompressMetaData(pFileHeader, DefaultMemoryIO.GetMemoryOffset(pMemoryFileHeader->MetaBlockOffset), core);

            if (bResult) {
               LOGGING("Meta string is %s\n", DefaultMemoryIO.GetMemoryOffset(pMemoryFileHeader->MetaBlockOffset));

               SDS_ARRAY_BLOCK* pDestArrayBlock = DefaultMemoryIO.GetArrayBlock(0);

               // Read array block
               DefaultFileIO.FileReadChunk(NULL, SDSFile, pDestArrayBlock, pFileHeader->ArrayBlockSize, pFileHeader->ArrayBlockOffset);

               // offset to first location of array data
               int64_t startOffset = pMemoryFileHeader->ArrayFirstOffset;

               // NOTE: have to fixup arrayblocks
               for (int32_t i = 0; i < arrayCount; i++) {

                  LOGGING("start offset %d %lld\n", i, startOffset);

                  // same size since uncompressed
                  pDestArrayBlock[i].ArrayCompressedSize = pDestArrayBlock[i].ArrayUncompressedSize;
                  pDestArrayBlock[i].ArrayDataOffset = startOffset;

                  // TJD ADDDED CODE
                  // Not sure if shared memroy needs the offset
                  //pDestArrayBlock[i].Reserved1 = (int64_t)DefaultMemoryIO.GetMemoryOffset(startOffset);
                  pDestArrayBlock[i].ArrayBandCount = 0;
                  pDestArrayBlock[i].ArrayBandSize = 0;

                  pDestArrayBlock[i].CompressionType = COMPRESSION_TYPE_NONE;

                  // Keep tallying
                  startOffset += SDS_PAD_NUMBER(pDestArrayBlock[i].ArrayUncompressedSize);
               }

               LOGGING("last offset %lld\n", startOffset);

               bool oneFile = (pFileHeader->FileType == SDS_FILE_TYPE_ONEFILE);
               int32_t fileType = pFileHeader->FileType;
               bool fileTypeStackable = (fileType == SDS_FILE_TYPE_DATASET ||
                  fileType == SDS_FILE_TYPE_TABLE ||
                  fileType == SDS_FILE_TYPE_ARRAY);

               // MORE WORK TO DO 
               SDS_READ_DECOMPRESS_ARRAYS* pstCompressArrays =
                  AllocDecompressArrays(pReadCallbacks, arrayCount, TRUE, oneFile, fileTypeStackable);

               // Set all the cores working memory to zero
               int32_t numCores = g_cMathWorker->WorkerThreadCount + 1;
               for (int32_t j = 0; j < numCores; j++) {
                  pstCompressArrays->pCoreMemory[j] = NULL;
                  pstCompressArrays->pCoreMemorySize[j] = 0;
                  pstCompressArrays->eventHandles[j] = DefaultFileIO.CreateEventHandle();
               }

               pstCompressArrays->fileHandle = SDSFile;
               pstCompressArrays->pFileHeader = pFileHeader;

               pstCompressArrays->pMemoryIO = &DefaultMemoryIO; // WHAT TO DO??
               pstCompressArrays->compMode = COMPRESSION_MODE_SHAREDMEMORY;

               pstCompressArrays->pFilter = &pReadCallbacks->Filter;

               // TODO: 
               void* saveState = pReadCallbacks->BeginAllowThreads();
               g_cMathWorker->DoMultiThreadedWork((int)arrayCount, DecompressFileArray, pstCompressArrays);
               pReadCallbacks->EndAllowThreads(saveState);

               //---------- CLEAN UP MEMORY AND HANDLES ---------
               // check if any cores allocated any memory
               for (int32_t j = 0; j < numCores; j++) {
                  if (pstCompressArrays->pCoreMemory[j]) {
                     WORKSPACE_FREE(pstCompressArrays->pCoreMemory[j]);
                  }
                  DefaultFileIO.DestroyEventHandle(pstCompressArrays->eventHandles[j]);
               }

               pMemoryFileHeader->ArraysWritten = arrayCount;
               LOGGING("End fill array block %lld\n", arrayCount);;
               return TRUE;
            }
         }
      }
      else {
         SetErr_Format(SDS_VALUE_ERROR, "SDS file was null when copying into shared memory\n");
      }
      return FALSE;
   }


   //------------------------------------------------
   // Returns FALSE on failure
   // Returns TRUE on success
   // 
   // Calls EndDecompressedFile() when done
   bool CopyIntoSharedMemory(SDS_READ_CALLBACKS* pReadCallbacks, const char* fileName, SDSIncludeExclude* pFolderName, int32_t core) {

      // open normal file first
      SDSFile = StartDecompressedFile(fileName, 0);

      if (SDSFile) {
         bool bResult = CopyIntoSharedMemoryInternal(pReadCallbacks, core);

         // Shut down the normal file
         EndDecompressedFile();
         return bResult;
      }
      return FALSE;
   }


   //--------------------------------------------------------
   // Returns the names and enum flags of everything written
   // Called by DecompressFileIntenral
   void MakeListNames(const int64_t nameBlockCount, const int64_t nameByteSize) {
      const char* startNameData = pNameData;
      const char* pNames = pNameData;

      // for every name
      for (int32_t i = 0; i < nameBlockCount; i++) {
         const char* pStart = pNames;

         // skip to end (search for 0 terminating char)
         while (*pNames++);

         // get the enum
         uint8_t value = *pNames++;

         //LOGGING("makelist name is %s, %d\n", pStart, i);

         pArrayNames[i] = pStart;
         pArrayEnums[i] = value;

         if ((pNames - startNameData) >= nameByteSize) break;
      }
   }


   //===============================================================
   //-------------------------------------------------------
   // Input: file already opened
   // NOTE: Called from DecompressFileInternal
   // returns NULL on error
   // return list of strings (column names)
   // on success pNameData is valid
   char* ReadListNames() {
      int64_t nameSize = pFileHeader->NameBlockSize;

      if (nameSize) {
         LOGGING("Name Block Count %lld,  namesize %lld\n", pFileHeader->NameBlockCount, nameSize);

         AllocateNameData(pFileHeader->NameBlockCount, nameSize);

         if (!pNameData) {
            SetErr_Format(SDS_VALUE_ERROR, "Decompression error in nameSize: %lld", nameSize);
            return NULL;
         }

         int64_t bytesRead =
            DefaultFileIO.FileReadChunk(NULL, SDSFile, pNameData, nameSize, pFileHeader->NameBlockOffset);

         if (bytesRead != nameSize) {
            SetErr_Format(SDS_VALUE_ERROR, "Decompression error in bytesRead: %lld", nameSize);
            return NULL;
         }

         // Run through name list and setup pointers
         MakeListNames(NameCount, nameSize);
         return pNameData;
      }
      return NULL;
   }


   //===============================================
   // returns TRUE/FALSE
   // stops early if Mode == COMPRESSION_MODE_INFO
   // new ver 4.3: handles filter=
   bool DecompressFileInternal(SDS_READ_CALLBACKS* pReadCallbacks, int32_t core, int64_t startOffset) {

      LOGGING("Reading filename %s normally.  Offset: %lld\n", FileName, startOffset);

      // read into pFileHeader
      // Fills in pFileHeader
      SDSFile = StartDecompressedFile(FileName, startOffset);

      if (SDSFile) {

         //------------- LIST NAMES -------------------------------
         // NO LIST NAMES ARE ALLOWED (root file?)
         if (pFileHeader->NameBlockSize && ReadListNames() == NULL) {
            goto EXIT_DECOMPRESS;
         }

         //------------- META DATA -------------------------------        
         DecompressMetaData(pFileHeader, NULL, core);

         //------------- SECTION DATA -------------------------------    
         // only offset ==0 (first fileheader) is responsible for reading the sections
         if (startOffset == 0) {
            if (pFileHeader->SectionBlockCount && cSectionName.ReadListSections(SDSFile, pFileHeader) == NULL) {
               goto EXIT_DECOMPRESS;
            }
         }

         // Tag which section of the file we are loading
         cSectionName.SectionOffset = startOffset;

         //--------- READ ARRAY BLOCK ---
         //------------- ALLOCATE ARRAY BLOCKS  -------------------------------
         // This will get freed up in EndCompressedFile
         AllocateArrayBlocks();

         if (!pArrayBlocks) {
            SetErr_Format(SDS_VALUE_ERROR, "Decompression error in pArrayBlock: %lld", pFileHeader->ArrayBlockSize);
            goto EXIT_DECOMPRESS;
         }

         LOGGING("Reading array block at offset:%lld   size: %lld \n", pFileHeader->ArrayBlockOffset, pFileHeader->ArrayBlockSize);
         int64_t bytesRead = DefaultFileIO.FileReadChunk(NULL, SDSFile, pArrayBlocks, pFileHeader->ArrayBlockSize, pFileHeader->ArrayBlockOffset);
         if (bytesRead != pFileHeader->ArrayBlockSize) {
            SetErr_Format(SDS_VALUE_ERROR, "Decompression error in ArrayBlockSize: %lld", pFileHeader->ArrayBlockSize);
            goto EXIT_DECOMPRESS;
         }

         int64_t tupleSize = pFileHeader->ArraysWritten;
         LOGGING("Arrays to read %lld  %p\n", tupleSize, pArrayBlocks);

         // If we got this far, the file is good
         IsFileValid = TRUE;
         IsFileValidAndNotFilteredOut = TRUE;

         // -- STOP EARLY IF THE USER JUST WANTS THE INFORMATION -----------------
         // Leave the files open in case they want to read arrays later
         if (Mode == COMPRESSION_MODE_INFO) {
            return TRUE;
         }

         int32_t fileType = pFileHeader->FileType;
         bool fileTypeStackable = (fileType == SDS_FILE_TYPE_DATASET ||
            fileType == SDS_FILE_TYPE_TABLE ||
            fileType == SDS_FILE_TYPE_ARRAY);

         //--------- ALLOCATE COMPRESS ARRAYS ---
         pstCompressArrays = AllocDecompressArrays(
            pReadCallbacks, 
            tupleSize, 
            FALSE, 
            pFileHeader->FileType == SDS_FILE_TYPE_ONEFILE,
            fileTypeStackable
         );

         pstCompressArrays->fileHandle = SDSFile;
         pstCompressArrays->pFileHeader = pFileHeader;

         LOGGING("Done allocating arrays %lld\n", tupleSize);

         // ---- ALLOCATE EVENT HANDLES
         int32_t numCores = g_cMathWorker->GetNumCores();

         for (int32_t j = 0; j < numCores; j++) {
            pstCompressArrays->pCoreMemory[j] = NULL;
            pstCompressArrays->pCoreMemorySize[j] = 0;
            pstCompressArrays->eventHandles[j] = DefaultFileIO.CreateEventHandle();
         }

         // ----- FILTERING --------------
         pstCompressArrays->pFilter = &pReadCallbacks->Filter;

         //---------  DECOMPRESS ARRAYS -------------

         // Multithreaded work and we tell caller when we started/stopped
         void* saveState = pReadCallbacks->BeginAllowThreads();
         g_cMathWorker->DoMultiThreadedWork((int)tupleSize, DecompressFileArray, pstCompressArrays);
         pReadCallbacks->EndAllowThreads(saveState);

         //---------- CLEAN UP MEMORY AND HANDLES ---------
         // check if any cores allocated any memory
         for (int32_t j = 0; j < numCores; j++) {
            if (pstCompressArrays->pCoreMemory[j]) {
               WORKSPACE_FREE(pstCompressArrays->pCoreMemory[j]);
            }
            DefaultFileIO.DestroyEventHandle(pstCompressArrays->eventHandles[j]);
         }

         return TRUE;
      }
      else {
         //  This overwrites real error
         if (g_errorbuffer[0] == 0) {
            SetErr_Format(SDS_VALUE_ERROR, "Unknown error when opening file %s\n", FileName);
         }
      }

   EXIT_DECOMPRESS:
      EndDecompressedFile();
      if (g_errorbuffer[0] == 0) {
         SetErr_Format(SDS_VALUE_ERROR, "ExitDecompress called for error when opening file %s\n", FileName);
      }
      return FALSE;
   }


   //======================================================
   // Caller must provide TWO callbacks
   // The callback will accept a void* return which will be finally returned
   // The callback is nec so that memory and handle cleanup can occur
   //
   // Arg1: SDS_READ_FINAL_CALLBACK
   // Arg2: SDS_READ_SHARED_MEMORY_CALLBACK -- only called when data is in shared memory
   //
   // Returns NULL or result of what user returned in ReadFinalCallback
   //
   void* DecompressFile(SDS_READ_CALLBACKS* pReadCallbacks, int32_t core, int64_t fileOffset) {

      LOGGING("Start of DCF %s [%lld]  %s\n", FileName, InstanceIndex, ShareName);

      //------------------------------------------------------
      // CHECK FOR SHARED MEMORY
      //------------------------------------------------------
      if (ShareName) {
         // Place GLOBAL in front or other variations to make a sharename
         DefaultMemoryIO.MakeShareName(FileName, ShareName);

         // Check for previous existence first
         HRESULT result = DefaultMemoryIO.MapExisting();
         bool bResult = TRUE;

         if (result < 0) {
            LOGGING("Failed to find existing share name %s\n", DefaultMemoryIO.SharedMemoryName);

            if (Mode == COMPRESSION_MODE_INFO) {
               // When the user just want the file information, we do not need to copy the file into shared memory yet
               bResult = FALSE;
            }
            else {
               bResult = CopyIntoSharedMemory(pReadCallbacks, FileName, pFolderName, core);

               if (!bResult) {
                  LOGGING("Failed to alloc shared memory for %s\n", FileName);
                  DefaultMemoryIO.Destroy();
               }
            }
         }

         if (bResult) {
            LOGGING("Success with find existing share... read from there %s\n", DefaultMemoryIO.SharedMemoryName);

            SDS_SHARED_MEMORY_CALLBACK SMCB;
            SMCB.pFileHeader = DefaultMemoryIO.GetFileHeader();
            SMCB.baseOffset = DefaultMemoryIO.GetMemoryOffset(0);
            SMCB.mode = Mode;
            SMCB.pSDSDecompressFile = this;
            SMCB.pMapStruct = DefaultMemoryIO.pMapStruct;

            void* retObject = pReadCallbacks->ReadMemoryCallback(&SMCB);
            // Leave shared memory open DefaultMemoryIO.Destroy();
            return retObject;
            // NOTE: On Linux if file persists in /dev/shm/Global then when is it ok to close it?
         }
         else {
            LOGGING("Failed to find existing share: %s... going to try file %s\n", DefaultMemoryIO.SharedMemoryName, FileName);

         }

         // Drop into normal file reading
      }

      bool bResult;
      bResult = DecompressFileInternal(pReadCallbacks, core, fileOffset);

      // Make sure final callback is ok to call as well
      if (bResult && pReadCallbacks->ReadFinalCallback) {

         // Ferry data to callback routine
         SDS_FINAL_CALLBACK   FinalCallback;

         // Copy over the important data from read class
         // The metadata is temporary and cannot be held onto (copy into your own buffer)
         // Arrays have been allocated based on what caller wanted
         FinalCallback.pFileHeader = pFileHeader;
         FinalCallback.mode = Mode;
         FinalCallback.arraysWritten = pFileHeader->ArraysWritten;
         FinalCallback.pArrayBlocks = pArrayBlocks;
         if (pstCompressArrays) {
            FinalCallback.pArrayInfo = pstCompressArrays->ArrayInfo;
         }
         else {
            FinalCallback.pArrayInfo = NULL;
         }

         FinalCallback.metaData = MetaData;
         FinalCallback.metaSize = MetaSize;
         FinalCallback.nameData = pNameData;

         // callback -------------------------
         // just one file was read
         return pReadCallbacks->ReadFinalCallback(&FinalCallback, 1);
      }

      return NULL;
   }
};


//enum SDS_TYPES {
//   SDS_BOOL = 0,
//   SDS_BYTE, SDS_UBYTE,
//   SDS_SHORT, SDS_USHORT,
//   SDS_INT, SDS_UINT,
//   SDS_LONG, SDS_ULONG,
//   SDS_LONGLONG, SDS_ULONGLONG,
//   SDS_FLOAT, SDS_DOUBLE, SDS_LONGDOUBLE,
//   SDS_CFLOAT, SDS_CDOUBLE, SDS_CLONGDOUBLE,
//   SDS_OBJECT = 17,
//   SDS_STRING, SDS_UNICODE,
//   SDS_VOID
//};

static int32_t SDS_ITEMSIZE_TABLE[22] = {
   1,         // bool
   1,   1,    // BYTE
   2,   2,    // SHORT
   4,   4,    // SDS_INT
   4,   4,    // AMBIGUOUS
   8,   8,    // SDS_LONGLONG
   4,   8,  16, // float, double, longdouble
   8,   16,  32, // float, double, longdouble
   0,
   0,          // string
   0,          // unicode
   0,          // SDS_VOID
   0
};


void UpgradeType(SDS_ARRAY_BLOCK*  pMasterArrayBlock, int32_t newdtype, int64_t itemsize) {

   int32_t isize = 4;

   // avoid the ambiguous type
   //newdtype = FixupDType(newdtype, itemsize);

   if (newdtype >= 0 && newdtype < 22) {
      isize = SDS_ITEMSIZE_TABLE[newdtype];
      if (isize == 0) {
         isize = (int)itemsize;
      }
   }

   LOGGING("Upgrading master dtype to %d  from %d  size:%d\n", newdtype, pMasterArrayBlock->DType, isize);
   pMasterArrayBlock->DType = newdtype;
   pMasterArrayBlock->ItemSize = isize;

   // Check dims and fortran vs cstyle
   if (pMasterArrayBlock->NDim > 1 && pMasterArrayBlock->Flags  &  SDS_ARRAY_F_CONTIGUOUS) {
      printf("!!!likely internal error with fortran array > dim 1 and upgrading dtype!\n");
      pMasterArrayBlock->Strides[pMasterArrayBlock->NDim - 1] = isize;
   }
   else {
      pMasterArrayBlock->Strides[0] = isize;
   }

}

//==========================================================================
//-------------------------------------------------------------------
// T = data type as input
// U = data type as output
// thus <float, int32> converts a float to an int32

typedef void(*CONVERT_INPLACE)(
   void* pDataIn,
   void* pDataOut,
   int64_t dataOutSize,
   int32_t dtypein,
   int32_t dtypeout);


template<typename T, typename U>
static void ConvertInplace(void* pDataIn, void* pDataOut, int64_t dataInSize, int32_t dtypein, int32_t dtypeout) {

   LOGGING("**conversion %p %p   inLenSize: %lld   inItemSize:%lld  outItemSize:%lld\n", pDataIn, pDataOut, dataInSize, sizeof(T), sizeof(U));
   T* pIn = (T*)pDataIn;
   U* pOut = (U*)pDataOut;
   T pBadValueIn = *(T*)SDSGetDefaultType(dtypein);
   U pBadValueOut = *(U*)SDSGetDefaultType(dtypeout);

   int64_t len = dataInSize / sizeof(T);
   int64_t dataOutSize = len * sizeof(U);

   if (dataInSize > dataOutSize) {
      printf("!! internal error in convertinplace\n");
      return;
   }

   pIn += len;
   pOut += len;
   pIn--;
   pOut--;

   // NAN converts to MINint32_t (for float --> int32_t conversion)
   // then the reverse, MIINint32_t converts to NAN (for int32_t --> float conversion)
   // convert from int32_t --> float
   for (int64_t i = 0; i < len; i++) {
      //printf("[%lld]converted %lf -> %lf\n", i, (double)*pIn, (double)*pOut);
      if (*pIn != pBadValueIn) {
         U temp = (U)*pIn;
         *pOut = temp;
      }
      else {
         *pOut = pBadValueOut;
      }
      //printf("[%lld]converted %lf -> %lf\n", i, (double)*pIn, (double)*pOut);
      pIn--;
      pOut--;
   }

}

//----------------------------------------
// Assumes input is a float/double
template<typename T, typename U>
static void ConvertInplaceFloat(void* pDataIn, void* pDataOut, int64_t dataInSize, int32_t dtypein, int32_t dtypeout) {

   LOGGING("**conversionf %p %p   inLenSize: %lld   inItemSize:%lld  outItemSize:%lld\n", pDataIn, pDataOut, dataInSize, sizeof(T), sizeof(U));
   T* pIn = (T*)pDataIn;
   U* pOut = (U*)pDataOut;
   U pBadValueOut = *(U*)SDSGetDefaultType(dtypeout);

   int64_t len = dataInSize / sizeof(T);
   int64_t dataOutSize = len * sizeof(U);

   if (dataInSize > dataOutSize) {
      printf("!! internal error in convertinplace\n");
      return;
   }

   pIn += len;
   pOut += len;
   pIn--;
   pOut--;

   // NAN converts to MINint32_t (for float --> int32_t conversion)
   // then the reverse, MIINint32_t converts to NAN (for int32_t --> float conversion)
   // convert from int32_t --> float
   for (int64_t i = 0; i < len; i++) {
      if (*pIn == *pIn) {
         *pOut = (U)*pIn;
      }
      else {
         *pOut = pBadValueOut;
      }
      //printf("[%lld]convertedf %lf -> %lf\n", i, (double)*pIn, (double)*pOut);
      pIn--;
      pOut--;
   }

}


//----------------------------------------
// Assumes input is uint8 out is uint32
template<typename T, typename U>
static void ConvertInplaceString(void* pDataIn, void* pDataOut, int64_t dataInSize, int32_t dtypein, int32_t dtypeout) {

   LOGGING("**conversions %p %p   inLenSize: %lld   inItemSize:%lld  outItemSize:%lld\n", pDataIn, pDataOut, dataInSize, sizeof(T), sizeof(U));
   T* pIn = (T*)pDataIn;
   U* pOut = (U*)pDataOut;

   int64_t len = dataInSize / sizeof(T);
   int64_t dataOutSize = len * sizeof(U);

   if (dataInSize > dataOutSize) {
      printf("!! internal error in convertinplace\n");
      return;
   }

   pIn += len;
   pOut += len;
   pIn--;
   pOut--;

   // convert from utf8 to utf32
   for (int64_t i = 0; i < len; i++) {
      *pOut = (U)*pIn;
      pIn--;
      pOut--;
   }

}


template<typename T>
static CONVERT_INPLACE GetInplaceConversionStep2Float(int32_t outputType) {
   switch (outputType) {
      //case SDS_BOOL:     return ConvertInplace<T, bool>;
   case SDS_FLOAT:    return ConvertInplaceFloat<T, float>;
   case SDS_DOUBLE:   return ConvertInplaceFloat<T, double>;
   case SDS_LONGDOUBLE: return ConvertInplaceFloat<T, long double>;
   case SDS_BYTE:     return ConvertInplaceFloat<T, int8_t>;
   case SDS_SHORT:    return ConvertInplaceFloat<T, int16_t>;
   case SDS_INT:      return ConvertInplaceFloat<T, int32_t>;
   case SDS_LONGLONG: return ConvertInplaceFloat<T, int64_t>;
   case SDS_UBYTE:    return ConvertInplaceFloat<T, uint8_t>;
   case SDS_USHORT:   return ConvertInplaceFloat<T, uint16_t>;
   case SDS_UINT:     return ConvertInplaceFloat<T, uint32_t>;
   case SDS_ULONGLONG:return ConvertInplaceFloat<T, uint64_t>;
   }
   return NULL;

}


template<typename T>
static CONVERT_INPLACE GetInplaceConversionStep2(int32_t outputType) {
   switch (outputType) {
      //case SDS_BOOL:     return ConvertInplace<T, bool>;
   case SDS_FLOAT:    return ConvertInplace<T, float>;
   case SDS_DOUBLE:   return ConvertInplace<T, double>;
   case SDS_LONGDOUBLE: return ConvertInplace<T, long double>;
   case SDS_BYTE:     return ConvertInplace<T, int8_t>;
   case SDS_SHORT:    return ConvertInplace<T, int16_t>;
   case SDS_INT:      return ConvertInplace<T, int32_t>;
   case SDS_LONGLONG: return ConvertInplace<T, int64_t>;
   case SDS_UBYTE:    return ConvertInplace<T, uint8_t>;
   case SDS_USHORT:   return ConvertInplace<T, uint16_t>;
   case SDS_UINT:     return ConvertInplace<T, uint32_t>;
   case SDS_ULONGLONG:return ConvertInplace<T, uint64_t>;
   }
   return NULL;

}

static CONVERT_INPLACE GetInplaceConversionFunction(int32_t inputType, int32_t outputType) {

   switch (inputType) {
   case SDS_BOOL:      return GetInplaceConversionStep2<bool>(outputType);
   case SDS_FLOAT:     return GetInplaceConversionStep2Float<float>(outputType);
   case SDS_DOUBLE:    return GetInplaceConversionStep2Float<double>(outputType);
   case SDS_LONGDOUBLE: return GetInplaceConversionStep2Float<long double>(outputType);
   case SDS_BYTE:      return GetInplaceConversionStep2<int8_t>(outputType);
   case SDS_SHORT:     return GetInplaceConversionStep2<int16_t>(outputType);
   case SDS_INT:       return GetInplaceConversionStep2<int32_t>(outputType);
   case SDS_LONGLONG:  return GetInplaceConversionStep2<int64_t>(outputType);
   case SDS_UBYTE:     return GetInplaceConversionStep2<uint8_t>(outputType);
   case SDS_USHORT:    return GetInplaceConversionStep2<uint16_t>(outputType);
   case SDS_UINT:      return GetInplaceConversionStep2<uint32_t>(outputType);
   case SDS_ULONGLONG: return GetInplaceConversionStep2<uint64_t>(outputType);
   case SDS_STRING: if (outputType == SDS_UNICODE) return ConvertInplaceString<uint8_t, uint32_t>; else break;
   }
   return NULL;
}


//----------------------------------------------------------
// Invalid fill when a column exist in one file but not the other
void GapFill(void* destBuffer, SDSArrayInfo* pDestInfo) {
   LOGGING(">>> gap fill in decompress multi array  length:%lld   dtype:%d  %p\n", pDestInfo->ArrayLength, pDestInfo->NumpyDType, destBuffer);
   LOGGING(">>> more gap dims: %d  itemsz:%d  dim0:%lld  strides0:%lld\n", pDestInfo->NDim, pDestInfo->ItemSize, pDestInfo->Dimensions[0], pDestInfo->Strides[0]);

   int64_t oneRowSize = pDestInfo->ItemSize;
   // Calc oneRowSize for multidimensional
   for (int32_t j = 1; j < pDestInfo->NDim; j++) {
      oneRowSize *= pDestInfo->Dimensions[j];
   }

   // Get the invalid fill type
   void* pDefaultType = SDSGetDefaultType(pDestInfo->NumpyDType);
   int64_t bytesToFill = oneRowSize * pDestInfo->ArrayLength;

   switch (pDestInfo->ItemSize) {
   case 1:
   {
      int8_t* pDestBuffer = (int8_t*)destBuffer;
      int8_t invalid = *(int8_t*)pDefaultType;
      for (int32_t j = 0; j < bytesToFill; j++) {
         pDestBuffer[j] = invalid;
      }
   }
   break;
   case 2:
   {
      int16_t* pDestBuffer = (int16_t*)destBuffer;
      int16_t invalid = *(int16_t*)pDefaultType;
      bytesToFill /= 2;
      for (int32_t j = 0; j < bytesToFill; j++) {
         pDestBuffer[j] = invalid;
      }
   }
   break;
   case 4:
   {
      int32_t* pDestBuffer = (int32_t*)destBuffer;
      int32_t invalid = *(int32_t*)pDefaultType;
      bytesToFill /= 4;
      for (int32_t j = 0; j < bytesToFill; j++) {
         pDestBuffer[j] = invalid;
      }
   }
   break;
   case 8:
   {
      int64_t* pDestBuffer = (int64_t*)destBuffer;
      int64_t invalid = *(int64_t*)pDefaultType;
      bytesToFill /= 8;
      for (int32_t j = 0; j < bytesToFill; j++) {
         pDestBuffer[j] = invalid;
      }
   }
   break;
   default:
      // probably a string
      // does matlab want ' ' spaces instead?
      memset(destBuffer, 0, bytesToFill);
      break;
   }

}


//---------------------------------------------------------
//
struct _SDS_WHICH_DTYPE {
   int8_t isFloat;
   int8_t isInt;
   int8_t isUInt;
   int8_t isString;

   int8_t isVoid;
   int8_t isObject;
   int8_t isUnknown;
   int8_t reserved;
};

struct SDS_WHICH_DTYPE {
   union {
      _SDS_WHICH_DTYPE  w;
      int64_t             whole;
   };
};

//----------------------------------------------------------
// returns with one of the dtype flags set
SDS_WHICH_DTYPE WhichDType(int32_t dtype)
{

   SDS_WHICH_DTYPE retVal;
   retVal.whole = 0;

   if (dtype <= SDS_LONGDOUBLE) {
      if (dtype >= SDS_FLOAT) {
         retVal.w.isFloat = TRUE;
      }
      else {
         // Odd dtypes below floats are ints
         if (dtype & 1 || dtype == 0) {
            retVal.w.isInt = TRUE;
         }
         else {
            retVal.w.isUInt = TRUE;
         }
      }
   }
   else {
      if (dtype == SDS_STRING || dtype == SDS_UNICODE) {
         retVal.w.isString = TRUE;
      }
      else {
         retVal.w.isUnknown = TRUE;
      }
   }
   return retVal;
}


//----------------------------------------------------------
// Multithreaded call when stacking
// int32_t to float, int16 to int32, uint8 to int32 types of conversion
// the master is the correct dtype
// the slave must convert and the slave's itemsize is the <= master's itemsize
// destSize MUST be recalculated if filtering
void  ConvertDType(
   char* destBuffer,
   SDS_ARRAY_BLOCK*  pMasterBlock,
   SDS_ARRAY_BLOCK*  pSlaveBlock,
   int64_t slaveRows,
   int64_t slaveItemSize)
{

   int32_t sdtype = FixupDType(pSlaveBlock->DType, pSlaveBlock->ItemSize);
   int32_t mdtype = FixupDType(pMasterBlock->DType, pMasterBlock->ItemSize);

   LOGGING("Convert dtype from:%d  to: %d  buffer:%p  size: %lld   %d vs %d\n", sdtype, mdtype, destBuffer, slaveRows * slaveItemSize, pSlaveBlock->NDim, pMasterBlock->NDim);
   CONVERT_INPLACE pConvert = GetInplaceConversionFunction(
      sdtype,
      mdtype);

   // Code below to be deleted when bugs in filtering while loading are gone
   //// Calculate length
   //int64_t dataInLen = pSlaveBlock->Dimensions[0] * pSlaveBlock->ItemSize;
   //for (int32_t j = 1; j < pSlaveBlock->NDim; j++) {
   //   dataInLen *= pSlaveBlock->Dimensions[j];
   //}
   //
   //// filtering can change size
   ////printf("Calculate destSize: %lld  vs  %lld\n", dataInLen, destSize);
   //destSize = dataInLen;

   if (pConvert) {
      pConvert(destBuffer, destBuffer, slaveRows * slaveItemSize, sdtype, mdtype);
   }
   else {
      printf("!!Internal error cannot convert %d to %d\n", sdtype, mdtype);
   }

}


//-----------------------------------------
typedef void(*SDS_COPY_FORTRAN)(
   void* pDest,        // (upper left corner of array)
   void* pSrc,         // the value to fill each element with
   int64_t totalRowsize, // all the rows of all the files  
   int64_t arrayOffset,  // start row for the current file
   int64_t arrayRowsize, // rowLength for the current file
   int64_t colSize);

//-----------------------------------------
//---------------------------------------------------------------
// pDest = upper left corner of matlab char array
//     |
// a d | g j m    0  1  2  3  4
// b e | h k n    5  6  7  8  9
// c f | i l o   10 11 12 13 14
// <b1>|<block2>
// example above... totalRowSize: 5
//                  arrayOffset:  2
//-----------------------------------------
// Used to stack fortran arrays
// both ararys must be fotran order
template<typename T>
void CopyFortran(
   void* pDestT,        // (upper left corner of array)
   void* pSrcT,         // the value to fill each element with
   int64_t totalRowsize, // all the rows of all the files  
   int64_t arrayOffset,  // start row for the current file
   int64_t arrayRowsize, // rowLength for the current file
   int64_t colSize) {

   T* pDest = (T*)pDestT;
   T* pOriginalDest = pDest;
   T* pSrc = (T*)pSrcT;

   // change all 0s to spaces
   for (int64_t col = 0; col < colSize; col++) {
      // Advance to correct slot
      pDest = pOriginalDest + arrayOffset + (col*totalRowsize);
      for (int64_t i = 0; i < arrayRowsize; i++) {
         *pDest++ = *pSrc++;
      }
   }
}


//-----------------------------------------
SDS_COPY_FORTRAN GetCopyFortran(int32_t dtype) {
   switch (dtype) {
   case SDS_BOOL:      return CopyFortran<int8_t>;
   case SDS_FLOAT:     return CopyFortran<int32_t>;
   case SDS_DOUBLE:    return CopyFortran<int64_t>;
   case SDS_LONGDOUBLE: return CopyFortran<long double>;
   case SDS_BYTE:      return CopyFortran<int8_t>;
   case SDS_SHORT:     return CopyFortran<int16_t>;
   case SDS_INT:       return CopyFortran<int32_t>;
   case SDS_LONGLONG:  return CopyFortran<int64_t>;
   case SDS_UBYTE:     return CopyFortran<int8_t>;
   case SDS_USHORT:    return CopyFortran<int16_t>;
   case SDS_UINT:      return CopyFortran<int32_t>;
   case SDS_ULONGLONG: return CopyFortran<int64_t>;
   }
   return NULL;
}


//-----------------------------------------
typedef void(*SDS_GAP_FILL_SPECIAL)(
   void* pDest,        // (upper left corner of array)
   int64_t totalRowsize, // all the rows of all the files  
   int64_t arrayOffset,  // start row for the current file
   int64_t arrayRowsize, // rowLength for the current file
   int64_t colSize);

//-----------------------------------------
// Used to fill 2 dim rotated arrays
// Arrays must be row major order
// TJD: not tested
template<typename T>
void GapFillSpecial(
   void* pDestT,        // (upper left corner of array)
   int64_t totalRowsize, // all the rows of all the files  
   int64_t arrayOffset,  // start row for the current file
   int64_t arrayRowsize, // rowLength for the current file
   int64_t colSize) {

   T* pDest = (T*)pDestT;
   T* pOriginalDest = pDest;
   T fill = 0;
   fill = GET_INVALID(fill);

   LOG_THREAD("In gap fill special sizeof:%lld   %lld %lld %lld %lld  %lf\n", sizeof(T), totalRowsize, arrayOffset, arrayRowsize, colSize, (double)fill);

   // Read horizontally (contiguous)
   // Write vertical (skip around)
   for (int64_t i = 0; i < arrayRowsize; i++) {

      // Advance to correct slot
      pDest = pOriginalDest + arrayOffset + i;

      // change all 0s to spaces
      for (int64_t col = 0; col < colSize; col++) {
         // replace 0 with space for matlab
         *pDest = fill;

         // skip horizontally
         pDest += totalRowsize;
      }
   }
}



//-----------------------------------------
// Used to fill 2 dim rotated arrays
// Arrays must be Fortran order
template<typename T>
void GapFillSpecialRowMajor(
   void* pDestT,        // (upper left corner of array)
   int64_t totalRowsize, // all the rows of all the files  
   int64_t arrayOffset,  // start row for the current file
   int64_t arrayRowsize, // rowLength for the current file
   int64_t colSize) {

   T* pDest = (T*)pDestT;
   T* pOriginalDest = pDest;
   T fill = 0;
   fill = GET_INVALID(fill);

   LOG_THREAD("In gap fill special sizeof:%lld   %lld %lld %lld %lld  %lf\n", sizeof(T), totalRowsize, arrayOffset, arrayRowsize, colSize, (double)fill);
   
   pDest = pDest + (arrayOffset * colSize);

   // Read horizontally (contiguous)
   // Write vertical (skip around)
   for (int64_t i = 0; i < (arrayRowsize * colSize); i++) {
      *pDest++ = fill;
   }
}



//---------------------------------------------------------------
// pDest = upper left corner of matlab char array
//
// a  b  c     0  1  2
// d  e  f     3  4  5
// g  h  i     6  7  8  <--- pretend second block loads here
// j  k  l     9 10 11
// m  n  o    12 13 14
//
// a d g j m   0  1  2  3  4
// b e h k n   5  6  7  8  9
// c f i l o  10 11 12 13 14
//
// example above... totalRowSize: 5
//                  arrayOffset:  2
//                  arrayRowSize: per file (3)
//                  itemSize:     3
// T is char for string
// T is uint32_t for UNICODE
//
template<typename T>
void MatlabStringFill(
   uint16_t* pDest,      // 2 byte unicode (see note on location)
   int64_t totalRowsize, // all the rows of all the files  
   int64_t arrayOffset,  // start row for the current file
   int64_t arrayRowsize, // rowLength for the current file
   int64_t itemSizeMaster) {

   // NOTE:
   // the pDest is not top left corner.. rather it is 
   // the top left corner + itemsize * row
   //
   // NOTE what happens for a gap fill for UNICODE and matlab is the loader?
   char* pCurrentDest = (char*)pDest;
   pCurrentDest = pCurrentDest - (arrayOffset*itemSizeMaster);

   uint16_t* pOriginalDest = (uint16_t*)pCurrentDest;

   pOriginalDest += arrayOffset;
   // the itemsize is correct as save as 1byte char then matlab wants 2byte char
   //itemSizeMaster = itemSizeMaster / sizeof(T);

   LOG_THREAD("Matlab string fill  %p  %lld %lld %lld %lld\n", pDest, totalRowsize, arrayOffset, arrayRowsize, itemSizeMaster);

   // Read horizontally (contiguous)
   // Write vertical (skip around)
   for (int64_t i = 0; i < arrayRowsize; i++) {

      // Advance to correct slot
      pDest = pOriginalDest + i;

      // matlab wants gapfill with spaces
      for (int64_t col = 0; col < itemSizeMaster; col++) {
         *pDest = 32;
         pDest += totalRowsize;
      }
   }
}

template<typename T>
void MatlabStringFillFromUnicode(
   uint16_t* pDest,      // 2 byte unicode (see note on location)
   int64_t totalRowsize, // all the rows of all the files  
   int64_t arrayOffset,  // start row for the current file
   int64_t arrayRowsize, // rowLength for the current file
   int64_t itemSizeMaster) {

   // NOTE:
   // the pDest is not top left corner.. rather it is 
   // the top left corner + itemsize * row
   //
   // NOTE what happens for a gap fill for UNICODE and matlab is the loader?
   char* pCurrentDest = (char*)pDest;
   // the itemsize is 4byte char then matlab wants 2byte char
   pCurrentDest = pCurrentDest - (arrayOffset*itemSizeMaster);
   itemSizeMaster = itemSizeMaster / 4;

   uint16_t* pOriginalDest = (uint16_t*)pCurrentDest;

   pOriginalDest += arrayOffset;

   LOG_THREAD("Matlab string fill  %p  %lld %lld %lld %lld\n", pDest, totalRowsize, arrayOffset, arrayRowsize, itemSizeMaster);

   // Read horizontally (contiguous)
   // Write vertical (skip around)
   for (int64_t i = 0; i < arrayRowsize; i++) {

      // Advance to correct slot
      pDest = pOriginalDest + i;

      // matlab wants gapfill with spaces
      for (int64_t col = 0; col < itemSizeMaster; col++) {
         *pDest = 32;
         pDest += totalRowsize;
      }
   }
}



//-----------------------------------------
SDS_GAP_FILL_SPECIAL GetGapFillSpecial(int32_t dtype) {
   switch (dtype) {
   case SDS_BOOL:      return GapFillSpecial<bool>;
   case SDS_FLOAT:     return GapFillSpecial<float>;
   case SDS_DOUBLE:    return GapFillSpecial<double>;
   case SDS_LONGDOUBLE: return GapFillSpecial<long double>;
   case SDS_BYTE:      return GapFillSpecial<int8_t>;
   case SDS_SHORT:     return GapFillSpecial<int16_t>;
   case SDS_INT:       return GapFillSpecial<int32_t>;
   case SDS_LONGLONG:  return GapFillSpecial<int64_t>;
   case SDS_UBYTE:     return GapFillSpecial<uint8_t>;
   case SDS_USHORT:    return GapFillSpecial<uint16_t>;
   case SDS_UINT:      return GapFillSpecial<uint32_t>;
   case SDS_ULONGLONG: return GapFillSpecial<uint64_t>;
   }
   return NULL;
}


//---------------------------------------------------------
// Called from multithreaded when a gap exists
//
void GapFillAny(
   SDSArrayInfo*  pDestInfo,
   void*          destBuffer,
   SDS_IO_PACKET* pIOPacket) {

   SDS_ARRAY_BLOCK* pMasterBlock = pIOPacket->pMasterBlock;

   LOG_THREAD("gapfill any %p  ndim:%d  flags:%d  destBuffer:%p  arrayoffset:%lld  itemsize:%d  trowlength:%lld\n", pDestInfo, pMasterBlock->NDim, pMasterBlock->Flags, destBuffer, pIOPacket->ArrayOffset, pMasterBlock->ItemSize, pMasterBlock->Dimensions[0]);

   // TODO: matlab string fill is different
   // Fortran >= 2d arrays gap fill is different
   if (pMasterBlock->NDim >= 2 && pMasterBlock->Flags & SDS_ARRAY_F_CONTIGUOUS) {
      int64_t totalRowLength = pMasterBlock->Dimensions[0];

      // rows for this file only
      int64_t rowLength = pDestInfo->Dimensions[0];

      int64_t colSize = 1;
      for (int32_t j = 1; j < pMasterBlock->NDim; j++) {
         colSize *= pMasterBlock->Dimensions[j];
      }

      //printf("Multidim gapfill dims: %d   rowlen:%lld  colsize:%lld  startrow:%lld   dtype: %d\n", (int)pMasterBlock->NDim, rowLength, colSize, pIOPacket->ArrayOffset, (int)pMasterBlock->DType);

      SDS_GAP_FILL_SPECIAL pGapFill = GetGapFillSpecial(pMasterBlock->DType);

      pGapFill(
         destBuffer,             // upper left corner of src
         totalRowLength,
         pIOPacket->ArrayOffset, // array offset (start row)
         rowLength,              // rows for this file only
         colSize);

   }
   else {

#ifdef MATLAB_MODE
      if (pMasterBlock->DType == SDS_UNICODE || pMasterBlock->DType == SDS_STRING) {
         int64_t totalRowLength = pMasterBlock->Dimensions[0];
         int64_t rowLength = pDestInfo->Dimensions[0];

         // fill with spaces for matlab
         if (pMasterBlock->DType == SDS_STRING) {
            // fill with spaces for matlab
            MatlabStringFill<uint16_t>(
               (uint16_t*)destBuffer,    // upper left corner of src
               totalRowLength,
               pIOPacket->ArrayOffset, // array offset (start row)
               rowLength,              // rows for this file only
               pMasterBlock->ItemSize);
         }
         else {
            // fill with spaces for matlab
            MatlabStringFillFromUnicode<uint16_t>(
               (uint16_t*)destBuffer,    // upper left corner of src
               totalRowLength,
               pIOPacket->ArrayOffset, // array offset (start row)
               rowLength,              // rows for this file only
               pMasterBlock->ItemSize);
         }
      }
      else {
         GapFill(destBuffer, pDestInfo);
      }
#else
      GapFill(destBuffer, pDestInfo);
#endif
   }
}


//---------------------------------------------------------------
// type U pDest = upper left corner of large array
// type T pSrc = upper left corner of smaller array
//
// This routine could convert on the fly also (but does not)
// This routine could widen on the fly (but does not)
//
// row major layout:
//   A1, B1, C1, D1, E1  <-- first file
//   A2, B2, C2, D2, E2  <-- second file (row length:2, col:5) (arrayOffset =1 )
//   A3, B3, C3, D3, E3  
//
// Fortran layout (total row length 3, col length:5)
//  A1, *A2, *A3
//  B1, *B2, *B3
//  C1, *C2, *C3
//  D1, *D2, *D3
//  E1, *E2, *E3

void RotationalFixup(
   char*      pDest,      // 
   char*      pSrc,       // 
   int64_t totalRowsize, // all the rows of all the files  
   int64_t arrayOffset,  // start row for the current file
   int64_t arrayRowsize, // rowLength for the current file
   int64_t arrayColsize, // colSize of the current file
   int64_t itemSize) {

   LOG_THREAD("Rotational2 fix off:%lld  totlrow:%lld  filerow:%lld  col:%lld  itemsz:%lld  dest:%p  %p\n", arrayOffset, totalRowsize, arrayRowsize, arrayColsize, itemSize, pDest, pSrc);
   char* pOriginalDest = pDest + (arrayOffset * itemSize);

   for (int64_t i = 0; i < arrayColsize; i++) {

      // Advance to correct slot
      pDest = pOriginalDest + (i*totalRowsize*itemSize);
      memcpy(pDest, pSrc, arrayRowsize * itemSize);
      pSrc = pSrc + (arrayRowsize * itemSize);
   }
}


//---------------------------------------------------------------
// pDest = upper left corner of matlab char array
// pSrc = often python 1byte string array
//
// a  b  c     0  1  2
// d  e  f     3  4  5
// g  h  i     6  7  8  <--- pretend second block loads here
// j  k  l     9 10 11
// m  n  o    12 13 14
//
// a d g j m   0  1  2  3  4
// b e h k n   5  6  7  8  9
// c f i l o  10 11 12 13 14
//
// example above... totalRowSize: 5
//                  arrayOffset:  2
//                  arrayRowSize: per file (3)
//                  itemSize:     3
// T is char for string
// T is uint32_t for UNICODE
//
template<typename T>
void MatlabStringFixup(
   uint16_t* pDest,      // 2 byte unicode (upper left corner of array)
   T*      pSrc,       // 1 or 4 byte unicode (file upper left corner)
   int64_t totalRowsize, // all the rows of all the files  
   int64_t arrayOffset,  // start row for the current file
   int64_t arrayRowsize, // rowLength for the current file
   int64_t itemSize,
   int64_t itemSizeMaster) {

   uint16_t* pOriginalDest = pDest + arrayOffset;
   itemSize = itemSize / sizeof(T);
   itemSizeMaster = itemSizeMaster / sizeof(T);
   int64_t itemSizeDelta = itemSizeMaster - itemSize;

   if (itemSizeDelta < 0) {
      // internal bug
      return;
   }
   LOG_THREAD("Matlab string fixup  %p %p  %lld %lld %lld %lld\n", pDest, pSrc, totalRowsize, arrayOffset, arrayRowsize, itemSize);

   // Read horizontally (contiguous)
   // Write vertical (skip around)
   for (int64_t i = 0; i < arrayRowsize; i++) {

      // Advance to correct slot
      pDest = pOriginalDest + i;

      // change all 0s to spaces
      for (int64_t col = 0; col < itemSize; col++) {
         uint16_t c = *pSrc++;
         if (c == 0) {
            // replace 0 with space for matlab
            *pDest = ' ';
         }
         else {
            *pDest = c;
         }
         // skip horizontally
         pDest += totalRowsize;
      }
      for (int64_t col = 0; col < itemSizeDelta; col++) {
         *pDest = ' ';
         pDest += totalRowsize;
      }
   }
}

//----------------------------------------------------------
// Multithreaded call when stacking
// String width fixup up
// The master is always the widest
// The slave string is widened inplace
// The slave buffer must be smaller than required
// Each line is stretched inplace starting at the back so we do not overwrite
void  StringFixup(
   char* destBuffer,
   SDS_ARRAY_BLOCK*  pMasterBlock,
   SDS_ARRAY_BLOCK*  pSlaveBlock,
   int64_t slaveRowLength,
   int64_t slaveItemSize)
{

   // Only the itemsize for master is correct
   // The other dimensions are correct in the pSlaveBlock
   int64_t oneRowSize = pMasterBlock->ItemSize;
   for (int32_t j = 1; j < pSlaveBlock->NDim; j++) {
      oneRowSize *= pSlaveBlock->Dimensions[j];
   }

   int64_t gap = oneRowSize - slaveItemSize;

   // Code below to be deleted when ugs in filtering gone
   //int64_t rows = pSlaveBlock->Dimensions[0];
   
   // new code for filtering..
   int64_t rows = slaveRowLength;

   if (gap > 0 && rows > 0) {
      if (pMasterBlock->DType == SDS_STRING) {
         LOGGING("string gap: %lld  rows: %lld  smallerSize:%lld  oneRowSize:%lld\n", gap, rows, slaveItemSize, oneRowSize);

         char* pEndString1 = destBuffer + (slaveItemSize * rows);
         char* pEndString2 = destBuffer + (oneRowSize * rows);
         pEndString1--;
         pEndString2--;

         for (int32_t i = 0; i < rows; i++) {
            char* pFront = pEndString2 - gap;
            while (pEndString2 > pFront) {
               // zero pad backwards
               //printf("z%c", *pEndString2);
               *pEndString2-- = 0;
            }
            //then copy the string
            pFront = pEndString2 - slaveItemSize;
            while (pEndString2 > pFront) {
               // copy string backwards
               *pEndString2-- = *pEndString1--;
            }
         }

      }
      else {
         LOGGING("unicode gap: %lld  rows: %lld  smallerSize:%lld  oneRowSize:%lld\n", gap, rows, slaveItemSize, oneRowSize);
         // Unicode loop
         oneRowSize /= 4;
         slaveItemSize /= 4;
         gap /= 4;

         uint32_t* pEndString1 = (uint32_t*)destBuffer + (slaveItemSize * rows);
         uint32_t* pEndString2 = (uint32_t*)destBuffer + (oneRowSize * rows);
         pEndString1--;
         pEndString2--;

         for (int32_t i = 0; i < rows; i++) {
            // first copy the gap
            for (int32_t j = 0; j < gap; j++) {
               // zero pad backwards
               *pEndString2-- = 0;
            }
            //then copy the string
            for (int32_t j = 0; j < slaveItemSize; j++) {
               // zero pad backwards
               *pEndString2-- = *pEndString1--;
            }
         }
      }
   }
}


//-----------------------------------------------------
// Called when stacking to determine how to homogenize arrays
//
SDS_COMPATIBLE IsArrayCompatible(
   const char* colName,
   SDS_ARRAY_BLOCK* pMasterArrayBlock,
   SDS_ARRAY_BLOCK* pArrayBlock,
   bool     doFixup) {

   // Fixup for int32 linux vs windows
   int32_t mdtype = FixupDType(pMasterArrayBlock->DType, pMasterArrayBlock->ItemSize);
   int32_t odtype = FixupDType(pArrayBlock->DType, pArrayBlock->ItemSize);

   SDS_COMPATIBLE  c;
   c.IsCompatible = TRUE;
   c.NeedsConversion = FALSE;
   c.NeedsRotation = FALSE;
   c.NeedsStringFixup = 0;

   if (mdtype != odtype) {
      // check conversion rules

      SDS_WHICH_DTYPE wmdtype = WhichDType(mdtype);
      SDS_WHICH_DTYPE wodtype = WhichDType(odtype);

      // If they are in the same class.. we can convert
      if (wmdtype.whole == wodtype.whole) {
         // String to UNICODE happens here
         c.NeedsConversion = TRUE;
         LOGGING("step1 conversion %s  %d  %d  masteritemsize:%d vs  %d  length: %lld %lld\n", colName, odtype, mdtype, pMasterArrayBlock->ItemSize, pArrayBlock->ItemSize, pMasterArrayBlock->Dimensions[0], pArrayBlock->Dimensions[0]);
         // pick the higher type
         if (odtype > mdtype) {
            if (doFixup) {
               // upgrade the dtype
               LOGGING("step2 conversion %d\n", pArrayBlock->ItemSize);
               UpgradeType(pMasterArrayBlock, odtype, pArrayBlock->ItemSize);
            }
         }
         else {
            // Special case when unicode to string
            if (mdtype == SDS_UNICODE && odtype == SDS_STRING) {
               LOGGING("step2 special conversion %d\n", pArrayBlock->ItemSize);
               if ((pArrayBlock->ItemSize * 4) > pMasterArrayBlock->ItemSize) {
                  UpgradeType(pMasterArrayBlock, mdtype, pArrayBlock->ItemSize * 4);
               }
            }
         }
      }
      else {
         // Some issue here... float vs int32_t or similar
         // int32_t vs uint
         SDS_WHICH_DTYPE cwhole;
         cwhole.whole = (wmdtype.whole | wodtype.whole);

         // Float to int32_t or uint?
         if (cwhole.w.isFloat && (cwhole.w.isInt || cwhole.w.isUInt)) {
            c.NeedsConversion = TRUE;
            LOGGING("Conversion: possible upgrade to float32/64 from int/uint32_t for col %s  %d to %d\n", colName, mdtype, odtype);

            if (doFixup) {
               // does the master have the float?
               if (mdtype > odtype) {
                  if (odtype >= SDS_int32_t && mdtype < SDS_DOUBLE) {
                     // MUST BE ATLEAST FLOAT64!! (auto upgrade)
                     // upgrade the dtype
                     UpgradeType(pMasterArrayBlock, SDS_DOUBLE, 0);
                  }
                  else {
                     // we are ok here...
                     LOGGING("Conversion: did nothing\n");
                  }
               }
               else {

                  if (mdtype >= SDS_int32_t && odtype < SDS_DOUBLE) {
                     // MUST BE ATLEAST FLOAT64!! (auto upgrade)
                     // upgrade the dtype
                     UpgradeType(pMasterArrayBlock, SDS_DOUBLE, 0);
                  }
                  else {

                     // other has the float, upgrade to other
                     UpgradeType(pMasterArrayBlock, odtype, pArrayBlock->ItemSize);
                  }
               }
            }

         }
         else
            if (cwhole.w.isUInt && cwhole.w.isInt) {
               bool handleInt64ToUInt64 = TRUE;
               c.NeedsConversion = TRUE;
               LOGGING("Conversion: int32_t /  uint32_t for col %s  %d to %d  fixup: %d\n", colName, mdtype, odtype, doFixup);

               if (doFixup) {
                  int32_t hightype = mdtype;
                  int32_t lowtype = odtype;
                  if (lowtype > hightype) {
                     lowtype = mdtype;
                     hightype = odtype;
                  }

                  if (hightype == SDS_ULONGLONG && lowtype == SDS_LONGLONG) {
                     // Due to bug with filtering - instead of convert will give warning here... precision loss
                     // This should be an option 
                     if (handleInt64ToUInt64) {
                        printf("Warning: ignoring sign loss going to from int/uint64 for col: %s\n", colName);
                        // flip it back off
                        c.NeedsConversion = FALSE;
                     }
                     else {
                        // This is the old code
                        printf("Warning: precision loss going to float64 from int/uint64 for col: %s\n", colName);
                        UpgradeType(pMasterArrayBlock, SDS_DOUBLE, 8);
                     }
                  }
                  else {
                     // newtype must be an int, so or 1
                     int32_t newtype = hightype | 1;

                     // if we bumped up to the ambiguous type, (going from 6 to 7)
                     if (newtype == SDS_LONG) newtype = SDS_LONGLONG;
                     if (newtype > SDS_LONGLONG) {
                        // could switch to double
                        newtype = SDS_LONGLONG;
                     }

                     if (mdtype != newtype) {
                        UpgradeType(pMasterArrayBlock, newtype, 0);
                     }
                  }
               }
            }
            else {
               // NOTE possible string to unicode conversion here.. unicode
               LOGGING("!!!Incompat due to dtypes %d %d\n", mdtype, odtype);
               c.IsCompatible = FALSE;
            }
      }

   }
   else {
      if (pMasterArrayBlock->DType == SDS_STRING || pMasterArrayBlock->DType == SDS_UNICODE) {
         if (pMasterArrayBlock->ItemSize != pArrayBlock->ItemSize) {
            LOGGING("Conversion: String width mismatch on col: %s   master itemsize:%d  vs  itemsize: %d\n", colName, pMasterArrayBlock->ItemSize, pArrayBlock->ItemSize);
            if (doFixup) {
               if (pMasterArrayBlock->ItemSize < pArrayBlock->ItemSize) {
                  // auto expand string
                  UpgradeType(pMasterArrayBlock, pMasterArrayBlock->DType, pArrayBlock->ItemSize);
               }
            }
            c.NeedsStringFixup = 1;
         }
#ifdef MATLAB_MODE
         // Matlab needs rotational fixing
         c.NeedsStringFixup |= 2;
#endif
      }
      else
         // dtypes MATCH, and are NOT STRINGS, but itemsize does not match (is this a void)?
         if (pMasterArrayBlock->ItemSize != pArrayBlock->ItemSize) {
            LOGGING("!!!Incompat due to itemsize\n");
            c.IsCompatible = FALSE;
         }
   }
   if (pMasterArrayBlock->NDim != pArrayBlock->NDim) {
      LOGGING("!!!Incompat due to ndim %d not macthing\n", pMasterArrayBlock->NDim);
      c.IsCompatible = FALSE;
   }
   if (pMasterArrayBlock->NDim > 1) {
      for (int32_t i = 1; i < pMasterArrayBlock->NDim; i++) {
         if (pMasterArrayBlock->Dimensions[i] != pArrayBlock->Dimensions[i]) {
            LOGGING("!!!Incompat due to dim %d not macthing\n", i);
            c.IsCompatible = FALSE;
         }
      }
      int32_t mflag = (pMasterArrayBlock->Flags & SDS_ARRAY_F_CONTIGUOUS);
      int32_t oflag = (pArrayBlock->Flags & SDS_ARRAY_F_CONTIGUOUS);

      if (mflag != oflag) {
         // possibly incompatible
         c.NeedsRotation = TRUE;
      }
   }
   return c;
}


//----------------------------------------------------------
// Called from multiple threads
// Returns NULL if nothing to do
// Otherwise returns the extra buffer
//
void* CheckRotationalFixup(SDS_IO_PACKET* pIOPacket) {
   // Master block only exists when stacking
   SDS_ARRAY_BLOCK*        pMasterBlock = pIOPacket->pMasterBlock;

   // no master block for normal reading (only stacking)
   if (pMasterBlock) {
      SDS_ARRAY_BLOCK*        pBlockInfo = pIOPacket->pBlockInfo;

      if (pMasterBlock->NDim >= 2 && pMasterBlock->Flags & SDS_ARRAY_F_CONTIGUOUS) {
         if (pMasterBlock->NDim > 2) {
            printf("!!! error cannot rotate above two dimensions\n");
         }
         int64_t allocSize = pBlockInfo->ItemSize;
         for (int32_t j = 0; j < pBlockInfo->NDim; j++) {
            allocSize *= pBlockInfo->Dimensions[j];
         }
         return WORKSPACE_ALLOC(allocSize);

      }
      else {
         if (pIOPacket->Compatible.NeedsStringFixup & 2) {
            int64_t allocSize = pBlockInfo->ItemSize * pBlockInfo->Dimensions[0];
            // pick up more string dimensions
            if (pBlockInfo->NDim > 1) {
               printf("!!! error cannot handle multid strings\n");
            }
            // Allocate a temporary buffer to load 1 or 4 byte string into
            // This buffer will have to be copied into
            return WORKSPACE_ALLOC(allocSize);
         }
      }
   }

   // no extra buffer required, nothing to rotate
   return NULL;
}

//----------------------------------------------------------
// Called from multiple threads
// Frees the buffer allocated from CheckRotationalFixup
bool FinishRotationalFixup(
   SDS_IO_PACKET* pIOPacket,
   void*          origBuffer,
   void*          tempBuffer) {

   if (!tempBuffer || !origBuffer) {
      return FALSE;
   }

   SDS_ARRAY_BLOCK*        pMasterBlock = pIOPacket->pMasterBlock;
   SDS_ARRAY_BLOCK*        pBlockInfo = pIOPacket->pBlockInfo;

   if (pMasterBlock->NDim >= 2 && pMasterBlock->Flags & SDS_ARRAY_F_CONTIGUOUS) {
      int64_t totalRowLength = pMasterBlock->Dimensions[0];
      int64_t rowLength = pBlockInfo->Dimensions[0];
      int64_t colLength = pMasterBlock->Dimensions[1];

      RotationalFixup(
         (char*)origBuffer,  // original buffer (upper left of final destination)
         (char*)tempBuffer,  // upper left corner of src
         totalRowLength,
         pIOPacket->ArrayOffset, // array offset (start row)
         rowLength,              // rows for this file only
         colLength,
         pMasterBlock->ItemSize);

   }
   else
      if (pIOPacket->Compatible.NeedsStringFixup & 2) {
         int64_t totalRowLength = pMasterBlock->Dimensions[0];
         int64_t rowLength = pBlockInfo->Dimensions[0];

         if (pMasterBlock->DType == SDS_UNICODE) {
            //rotate and convert data from 1 byte to 2 bytes
            MatlabStringFixup<uint32_t>(
               (uint16_t*)origBuffer,  // original buffer (upper left of final destination)
               (uint32_t*)tempBuffer,  // upper left corner of src
               totalRowLength,
               pIOPacket->ArrayOffset, // array offset (start row)
               rowLength,              // rows for this file only
               pBlockInfo->ItemSize,
               pMasterBlock->ItemSize);
         }
         else {

            MatlabStringFixup<char>(
               (uint16_t*)origBuffer,  // original buffer (upper left of final destination)
               (char*)tempBuffer,    // upper left corner of src
               totalRowLength,
               pIOPacket->ArrayOffset, // array offset (start row)
               rowLength,              // rows for this file only
               pBlockInfo->ItemSize,
               pMasterBlock->ItemSize);
         }
      }
   WORKSPACE_FREE(tempBuffer);
   return TRUE;

}



//----------------------------------------------------------
//Decompress -- called from multiple threads -- READING data
//
// t is the array index (iopacket index)
// pstCompressArrays must have pBlockInfo set (except for stacked gaps?)
// It will read from a file using: pstCompressArrays->eventHandles[core], sdsFile
// for normal read: read into pstCompressArrays->ArrayInfo[t].pData
// size of the READ: pBlockInfo->ArrayCompressedSize
//
bool DecompressMultiArray(
   void* pstCompressArraysV,
   int32_t   core,    // which thread running on
   int64_t t)       // t=task count from 0 - # of iopackets
{
   LOG_THREAD("[%lld] Start of decompress multi array: core %d   compress: %p\n", t, core, pstCompressArraysV);

   SDS_MULTI_IO_PACKETS*   pMultiIOPackets = (SDS_MULTI_IO_PACKETS *)pstCompressArraysV;
   SDS_IO_PACKET*          pIOPacket = &pMultiIOPackets->pIOPacket[t];
   SDS_FILE_HANDLE         sdsFile = pIOPacket->FileHandle;

   // point32_t to block (different from single)
   SDS_ARRAY_BLOCK*        pBlockInfo = pIOPacket->pBlockInfo;

   // Master block only exists when stacking
   SDS_ARRAY_BLOCK*        pMasterBlock = pIOPacket->pMasterBlock;
   void*                   destBuffer = NULL;

   LOG_THREAD("[%lld] Step 2 of decompress array: core %d  blockinfo %p\n", t, core, pBlockInfo);

   // Check if we are reading into memory or reading into a preallocated numpy array
   if (pIOPacket->CompMode == COMPRESSION_MODE_SHAREDMEMORY) {
      SDS_ARRAY_BLOCK* pArrayBlock = pIOPacket->pMemoryIO->GetArrayBlock(t);
      destBuffer = pIOPacket->pMemoryIO->GetMemoryOffset(pArrayBlock->ArrayDataOffset);
      LOG_THREAD("[%lld] multi decompressing shared memory %p\n", t, destBuffer);
   }
   else {

      // Use callback to get to array buffer
      destBuffer = pMultiIOPackets->pDestInfo[t].pData;
      LOG_THREAD("[%lld] multi decompressing into %p\n", t, destBuffer);
   }

   SDSArrayInfo* pDestInfo = &pMultiIOPackets->pDestInfo[t];

   // If the file has a gap, there is no BlockInfo
   // early exit GAP FILL (no need to read file chunk because it does not exist)
   // if the column is not compatible such as string to float, we also gap fill
   if (pBlockInfo == NULL || !pIOPacket->Compatible.IsCompatible) {

      GapFillAny(pDestInfo, destBuffer, pIOPacket);
      return TRUE;
   }

   // Make sure we have a valid buffer
   if (destBuffer) {
      int64_t source_size = pBlockInfo->ArrayCompressedSize;

      // Check for matlab strings which are 2 byte
      // If so, allocate another buffer
      // Often strings come in 1 byte --> Matlab wants it rotated and 2 bytes
      void* origBuffer = NULL;
      void* rotateBuffer = CheckRotationalFixup(pIOPacket);

      // rotateBuffer exists for rotational fixups like fortran 2d or matlab strings
      if (rotateBuffer) {
         // Allocate a temporary buffer to load 1 or 4 byte string into
         // This buffer will have to be copied into
         origBuffer = destBuffer;
         destBuffer = rotateBuffer;
      }

      // Check if uncompressed
      // check if our temporary buffer is large enough to hold decompression data
      if ((source_size != pBlockInfo->ArrayUncompressedSize) && (pMultiIOPackets->pCoreMemorySize[core] < source_size)) {

         if (pMultiIOPackets->pCoreMemory[core]) {
            // free old one if there
            WORKSPACE_FREE(pMultiIOPackets->pCoreMemory[core]);
         }

         // Reallocate larger memory
         pMultiIOPackets->pCoreMemorySize[core] = source_size;
         pMultiIOPackets->pCoreMemory[core] = WORKSPACE_ALLOC(source_size);
         ;
         // Log that we were forced to reallocate
         LOG_THREAD("-");
      }

      LOG_THREAD("ALLOC %p %d %p  stackpos: %lld  bpr: %lld\n", pMultiIOPackets, core, pMultiIOPackets->pCoreMemory[core], pIOPacket->StackPosition, GetBytesPerRow(pBlockInfo));
      void* tempFileBuffer = pMultiIOPackets->pCoreMemory[core];

      // NEW ROUTINE...
      int64_t result = 0;
      if (pIOPacket->StackPosition >= 0) {

         // If there is no filtering, the array length in bytes is the uncompressed size
         int64_t slaveRowLength = 0;
         int64_t slaveItemSize = GetBytesPerRow(pBlockInfo);

         //  If no bytes to read..
         if (slaveItemSize == 0) return TRUE;

         if ((pMasterBlock && pMasterBlock->Flags & SDS_ARRAY_FILTERED) || (pBlockInfo->Flags & SDS_ARRAY_FILTERED)) {
            //printf("filtering on for %lld\n", t);
            //if (pMultiIOPackets->pFilter->pFancyMask) {
            //   printf("Fancy mask filtering not supported for %lld\n", t);
            //}
            // Stacking plus filtering
            //printf("**step2 %p\n", pMultiIOPackets->pFilter->pFilterInfo);
            LOGGING("***stacking  plus filter  stack position: %lld  buffer: %p  TrueCount: %lld\n", pIOPacket->StackPosition, destBuffer, pMultiIOPackets->pFilter->pFilterInfo ? pMultiIOPackets->pFilter->pFilterInfo[pIOPacket->StackPosition].TrueCount: 0);
            result =
               ReadAndDecompressArrayBlockWithFilter(
                  pBlockInfo,
                  pMultiIOPackets->eventHandles[core],
                  sdsFile,
                  tempFileBuffer,
                  destBuffer,
                  pIOPacket->OriginalArrayOffset, // array offset (start row)

                  pMultiIOPackets->pFilter,
                  pIOPacket->StackPosition,
                  core,
                  COMPRESSION_TYPE_ZSTD);

            // When filtered the arrayLength is how many true bools * the bytes per row
            if (pMultiIOPackets->pFilter->pFilterInfo) {
               slaveRowLength = pMultiIOPackets->pFilter->pFilterInfo[pIOPacket->StackPosition].TrueCount;
            }
            else {
               // must be SDS file appended and called with stack=False
               slaveRowLength = pMultiIOPackets->pFilter->BoolMaskTrueCount;
            }

         }
         else {
            LOGGING("[%lld]***stacking no filter  stack position: %lld  buffer: %p  rowlength: %lld  colname: %s\n", t, pIOPacket->StackPosition, destBuffer, pBlockInfo->Dimensions[0], pIOPacket->ColName);

            result =
               ReadAndDecompressArrayBlock(
                  pBlockInfo,
                  pMultiIOPackets->eventHandles[core],
                  sdsFile,
                  tempFileBuffer,
                  destBuffer,
                  pIOPacket->StackPosition,
                  core,
                  COMPRESSION_TYPE_ZSTD);


            slaveRowLength = pBlockInfo->ArrayUncompressedSize / slaveItemSize;
         }

         if (result >= 0) {
            // Check if we had to allocate an extra buffer for rotation
            if (origBuffer) {
               FinishRotationalFixup(pIOPacket, origBuffer, rotateBuffer);
               destBuffer = origBuffer;
            }
            else {
               // Python string fixup path
               if (pIOPacket->Compatible.NeedsStringFixup == 1) {
                  // This is when column have strings, but they are different widths
                  // Find the wide string and expand everyone else
                  StringFixup((char*)destBuffer, pMasterBlock, pBlockInfo, slaveRowLength, slaveItemSize);
               }
               if (pIOPacket->Compatible.NeedsConversion) {
                  // done inplace (no need for extra buffer)
                  LOGGING("Needs conversion  mb dtype:%d  mb itemsize: %d    dim0: %lld   slaveRowLength:%lld  needs stringfixup: %d  filterflags: %d\n", pIOPacket->pMasterBlock->DType, pIOPacket->pMasterBlock->ItemSize, pIOPacket->pMasterBlock->Dimensions[0], slaveRowLength, pIOPacket->Compatible.NeedsStringFixup, pMasterBlock->Flags);
                  ConvertDType((char*)destBuffer, pMasterBlock, pBlockInfo, slaveRowLength, slaveItemSize);
               }
            }
         }
      }
   }
   
   else {
      LOG_THREAD("!!bad destBuffer\n");
   }
   return TRUE;
}


//=================================================
// main class used to read many files at once
//=================================================
// main class used to read many files at once
// The array pSDSDecompressFile has length fileCount
//
class SDSDecompressManyFiles {
public:

   // pointer to list of files
   SDSDecompressFile**  pSDSDecompressFile = NULL;
   SDSIncludeExclude*   pIncludeList = NULL;
   SDSIncludeExclude*   pFolderList = NULL;
   SDSIncludeExclude*   pSectionsList = NULL;

   // how many files (include both valid and invalid)
   int64_t                FileCount = 0;

   // user callbacks
   SDS_READ_CALLBACKS*  pReadCallbacks = NULL;
   SDS_FINAL_CALLBACK** pReadManyFinalCallbacks = NULL;

   // Length of all valid filenames
   // Uee for stacking only
   struct SDS_DATASET {
      int64_t                Length;
      // pointer to bands
      // which bands are valid?
   };

   SDS_DATASET*         pDatasets = NULL;

   // holds unique columns
   // hash map for columns
   // second paramter is NxM -- upper 32 => filecount
   //                           lower 32 => column index
   struct SDS_COLUMN_KING {
      const char*       ColName;
      int64_t             ColPos;     // matches the ColumnVector

      int32_t             FileRow;    // first file that had this column
      int32_t             ArrayEnum;  // enum of first file that had this column

      SDS_ARRAY_BLOCK   KingBlock;          // the king block (master block that determines array dtype)
      SDS_ARRAY_BLOCK** ppArrayBlocks;      // allocate array of these
      int64_t*            pArrayOffsets;      // must be deleted separately 0:5:10:15:20
      int64_t*            pOriginalArrayOffsets;      // must be deleted separately 0:5:10:15:20
      int64_t*            pOriginalLengths;   // must be deleted separately 5:5:5:5:5 -- 0 if needs invalid filling

      int64_t             TotalRowLength;      // valid when done tallying
   };

   // check for existence (store column position which can be used in SDS_COLUMN_ORDER)
   typedef std::unordered_map<std::string, int64_t>  SDS_COLUMN_HASH;

   // insert in order.  TotalUniqueColumns is how many vectors.
   typedef std::vector<SDS_COLUMN_KING>   SDS_COLUMN_ORDER;

   // warning: not concurrent multithread safe
   SDS_COLUMN_HASH      ColumnExists;
   SDS_COLUMN_ORDER     ColumnVector;
   int32_t                  TotalUniqueColumns = 0;
   int32_t                  TotalStringFixups = 0;
   int32_t                  TotalConversions = 0;
   int32_t                  TotalDimensionProblems = 0;
   int32_t                  TotalColumnGaps = 0;
   int32_t                  TotalFirstColumns = 0;
   int32_t                  LastRow = -1;


   //------------------------------------------------
   // input validCount (number of files and thus number of Datasets)
   void AllocateDatasetLengths(
      int64_t validCount) {

      int64_t allocSize = sizeof(SDS_DATASET)*(validCount);
      pDatasets = (SDS_DATASET*)WORKSPACE_ALLOC(allocSize);
      memset(pDatasets, 0, allocSize);
   }

   //------------------------------------------------
   // input validCount
   //
   // returns pArrayOffsets
   // returns pArrayLengths
   // returns ppArrayBlocks
   void AllocateVectorList(
      int64_t validCount,
      int64_t* &pArrayOffsets,
      int64_t* &pOriginalArrayOffsets,
      int64_t* &pOriginalLengths,
      SDS_ARRAY_BLOCK** &ppArrayBlocks) {

      // for only valid files
      // arrayoffsets has onemore entry for the final total
      int64_t allocSize = sizeof(int64_t)*(validCount + 1);
      pArrayOffsets = (int64_t*)WORKSPACE_ALLOC(allocSize);
      //memset(pArrayOffsets, 0, allocSize);

      pOriginalArrayOffsets = (int64_t*)WORKSPACE_ALLOC(allocSize);
      //memset(pOriginalArrayOffsets, 0, allocSize);

      pOriginalLengths = (int64_t*)WORKSPACE_ALLOC(allocSize);
      memset(pOriginalLengths, 0, allocSize);

      allocSize = sizeof(SDS_ARRAY_BLOCK*)*(validCount);
      ppArrayBlocks = (SDS_ARRAY_BLOCK**)WORKSPACE_ALLOC(allocSize);
      memset(ppArrayBlocks, 0, allocSize);
   }

   //-------------------------------------------------------------------
   // Cleans up the additional allocation that go with each unique column
   void ClearVectorList() {
      // TODO delete this better
      for (int32_t i = 0; i < TotalUniqueColumns; i++) {
         WORKSPACE_FREE(ColumnVector[i].ppArrayBlocks);
         WORKSPACE_FREE(ColumnVector[i].pArrayOffsets);
         WORKSPACE_FREE(ColumnVector[i].pOriginalArrayOffsets);
         WORKSPACE_FREE(ColumnVector[i].pOriginalLengths);
      }

      TotalUniqueColumns = 0;

      ColumnVector.clear();

      if (pDatasets) {
         WORKSPACE_FREE(pDatasets);
         pDatasets = NULL;
      }
   }

   //------------------------------------------------
   // Also clears the vector list
   void ClearColumnList() {
      ColumnExists.clear();
      ClearVectorList();
      TotalUniqueColumns = 0;
      TotalStringFixups = 0;
      TotalConversions = 0;
      TotalDimensionProblems = 0;
      TotalColumnGaps = 0;
      TotalFirstColumns = 0;
      LastRow = -1;
   }



   //------------------------------------------------
   // input - new or existing column
   // if new, a new ColumnVector will be created
   //
   // 
   // if the name is not found, the unique count goes up
   void AddColumnList(
      int64_t validPos,
      int64_t validCount,
      const char* columnName,
      int32_t arrayEnum,  // DATASET, OTHER
      int32_t fileRow,
      int64_t column,
      SDS_ARRAY_BLOCK* pArrayBlock) {

      SDS_ARRAY_BLOCK**  ppArrayBlocks = NULL;
      int64_t* pArrayOffsets = NULL;
      int64_t* pOriginalArrayOffsets = NULL;
      int64_t* pOriginalLengths = NULL;

      std::string item = std::string(columnName);
      auto columnFind = ColumnExists.find(item);

      // Is this a new column name?
      if (columnFind == ColumnExists.end()) {

         // check if the new entry is for the first row
         if (fileRow > 0) {
            TotalColumnGaps++;
         }

         LOGGING("%s ** %lld   enum: %d\n", columnName, validCount, arrayEnum);
         ColumnExists.emplace(item, TotalUniqueColumns);

         // Brand new column
         // Allocate...pArrayOffsets, pOriginalLengths, ppArrayBlocks
         AllocateVectorList(validCount, pArrayOffsets, pOriginalArrayOffsets, pOriginalLengths, ppArrayBlocks);

         // Add to the end (in this column order) the new column
         ColumnVector.push_back(
            { columnName,
            TotalUniqueColumns,
            fileRow,
            arrayEnum,
            *pArrayBlock,      // copy of the king (determines shape of column)
            ppArrayBlocks,
            pArrayOffsets,
            pOriginalArrayOffsets,
            pOriginalLengths,
            0 });

         if (validPos == 0) {
            // do we care anymore
            TotalFirstColumns++;
         }

         // important count of unique column that we pushed back
         TotalUniqueColumns++;
      }
      else {

         // Get the master column
         int64_t colPos = columnFind->second;
         ppArrayBlocks = ColumnVector[colPos].ppArrayBlocks;
         pArrayOffsets = ColumnVector[colPos].pArrayOffsets;
         pOriginalArrayOffsets = ColumnVector[colPos].pOriginalArrayOffsets;
         pOriginalLengths = ColumnVector[colPos].pOriginalLengths;

         // For strings, if the width increased, we update the king block
         SDS_COMPATIBLE compat =
            IsArrayCompatible(columnName, &ColumnVector[colPos].KingBlock, pArrayBlock, TRUE);

         if (compat.NeedsRotation) {
            // cannot handle
            printf("Column '%s' needs rotation from col or row major and currently this is not allowed\n", columnName);
            TotalDimensionProblems++;
         }
         if (!compat.IsCompatible) {
            printf("Warning: Column '%s' has both string and unicode. Support for this is experimental\n", columnName);
            //++;
         }
         if (compat.NeedsStringFixup) {
            TotalStringFixups++;
         }
         if (compat.NeedsConversion) {
            TotalConversions++;
         }
      }

      // Update our array block with the new entry
      ppArrayBlocks[validPos] = pArrayBlock;
      pOriginalLengths[validPos] = 0;

      // If we have 1 or more dimensions, the length is the first dimension
      if (pArrayBlock->NDim > 0) {
         int64_t rowLength = pArrayBlock->Dimensions[0];
         pOriginalLengths[validPos] = rowLength;
      }

      // track last valid row?
      LastRow = (INT)fileRow;
   }


   //------------------------------------------------
   // constructor
   // shareName NULL is allowed
   SDSDecompressManyFiles(
      SDSDecompressFile**  pSDSDecompressFile,
      SDSIncludeExclude*   pIncludeList,
      SDSIncludeExclude*   pFolderList,
      SDSIncludeExclude*   pSectionsList,
      int64_t                fileCount,
      SDS_READ_CALLBACKS*  pReadCallbacks) {

      this->pSDSDecompressFile = pSDSDecompressFile;
      this->pIncludeList = pIncludeList;
      this->pFolderList = pFolderList;
      this->pSectionsList = pSectionsList;
      this->FileCount = fileCount;
      this->pReadCallbacks = pReadCallbacks;
   }



   //=================================================
   // Multithreaded callback
   // passes a pointer to itself since static function
   static bool DecompressManyFiles(void* pstManyV, int32_t core, int64_t t) {

      SDSDecompressManyFiles* pSDSDecompressMany = (SDSDecompressManyFiles*)pstManyV;
      SDSDecompressFile* pSDSDecompressFile = pSDSDecompressMany->pSDSDecompressFile[t];

      // isFileValid will be set if the file is good
      pSDSDecompressFile->DecompressFileInternal(pSDSDecompressMany->pReadCallbacks, core, 0);

      if (pSDSDecompressFile->IsFileValid && pSDSDecompressFile->Mode == SDS_MULTI_MODE_CONCAT_MANY) {
         pSDSDecompressFile->FileSize = DefaultFileIO.FileSize(pSDSDecompressFile->FileName);
      }
      return TRUE;
   }

   //===========================================
   // multithreaded kick off  (NOTE: do not use non-thread safe calls here)
   void GetFileInfo(int32_t multiMode) {
      //printf("Calling... %lld %p\n", i, pSDSDecompressFile[i]);
      // Multithreaded work and we tell caller when we started/stopped
      void* saveState = pReadCallbacks->BeginAllowThreads();
      g_cMathWorker->DoMultiThreadedWork((int)FileCount, &DecompressManyFiles, this);
      pReadCallbacks->EndAllowThreads(saveState);
   }



   //===========================================
   // singlethreaded allocation, followed by multithreaded reads
   //
   void* ReadIOPackets(SDS_FINAL_CALLBACK* pReadFinalCallback, SDS_READ_CALLBACKS* pReadCallbacks) {
      // Build IO LIST
      //Allocate all the arrays
      int64_t totalIOPackets = 0;

      // First pass, calculate what we need
      for (int64_t f = 0; f < FileCount; f++) {
         SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[f];
         if (pSDSDecompress->IsFileValid) {
            totalIOPackets += pSDSDecompress->GetTotalArraysWritten();
         }
      }

      LOGGING("Total IO Packet %lld\n", totalIOPackets);

      // Allocate MultiIO PACKETS!!
      SDS_MULTI_IO_PACKETS* pMultiIOPackets = SDS_MULTI_IO_PACKETS::Allocate(totalIOPackets, &pReadCallbacks->Filter);
      int64_t currentPos = 0;

      // Fill in all the IOPACKETs
      // Skip over invalid files
      for (int64_t f = 0; f < FileCount; f++) {
         SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[f];
         if (pSDSDecompress->IsFileValid) {
            int64_t tupleSize = pSDSDecompress->pFileHeader->ArraysWritten;

            LOGGING("Allocating %lld arrays\n", tupleSize);

            SDS_IO_PACKET* pIOPacket = &pMultiIOPackets->pIOPacket[currentPos];
            SDSArrayInfo* pArrayInfo = &pMultiIOPackets->pDestInfo[currentPos];

            //--------- ALLOCATE COMPRESS ARRAYS INTO all the IO PACKETS
            pSDSDecompress->AllocMultiArrays(
               pIOPacket,
               pArrayInfo,
               pReadCallbacks,
               tupleSize,
               FALSE);

            pReadFinalCallback[f].pArrayInfo = pArrayInfo;
            currentPos += tupleSize;
         }
      }

      if (totalIOPackets) {
         //--------- ALLOCATE COMPRESS ARRAYS ---
         //---------  DECOMPRESS ARRAYS -------------
         // Multithreaded work and we tell caller when we started/stopped
         void* saveState = pReadCallbacks->BeginAllowThreads();
         g_cMathWorker->DoMultiThreadedWork((int)totalIOPackets, DecompressMultiArray, pMultiIOPackets);
         pReadCallbacks->EndAllowThreads(saveState);
      }

      void* result = pReadCallbacks->ReadFinalCallback(pReadFinalCallback, FileCount);

      //---------- CLEAN UP MEMORY AND HANDLES ---------
      SDS_MULTI_IO_PACKETS::Free(pMultiIOPackets);

      return result;
   }


   static void UpdateSectionData(
      char* pSectionData,
      int64_t currentSection,
      int64_t currentOffset) {

      // again very hackish
      char* pEntry = pSectionData + (currentSection * 10);
      pEntry[0] = '0';
      pEntry[1] = 0;
      *(int64_t*)(pEntry + 2) = currentOffset;

   }
   //=====================================================
   // fileHandle is the output file
   // pSDSDecompress is the input file
   // fileOffset is the offset in the output file to start writing at
   // fileSize is the size of the input file
   // set localOffset to non-zero to indicate a section copy
   static int64_t AppendToFile(
      SDS_FILE_HANDLE  outFileHandle, 
      SDSDecompressFile* pSDSDecompress,
      int64_t fileOffset, 
      int64_t fileSize,
      char* pSectionData,
      int64_t &currentSection) {

      LOGGING("files %s has size %lld\n", pSDSDecompress->FileName, fileSize);
      SDS_FILE_HEADER *pFileHeader = &pSDSDecompress->FileHeader;
      int64_t currentOffset = fileOffset;

      int64_t origArrayBlockOffset = pFileHeader->ArrayBlockOffset;
      SDS_FILE_HANDLE inFile = pSDSDecompress->SDSFile;
      bool hasSections = FALSE;
      int64_t localOffset = 0;

      if (pFileHeader->SectionBlockOffset) {
         LOGGING("!!warning file %s has section within section when concat\n", pSDSDecompress->FileName);
         hasSections = TRUE;
      }

      // end of file might be larger due to padding... when it is, cap to filesize
      //int64_t calculatedSize = pFileHeader->GetEndOfFileOffset();
      //
      //if (calculatedSize < fileSize) {
      //   printf("reducing size %lld!\n", fileSize);
      //   fileSize = calculatedSize;
      //}

      LOGGING("sds_concat %lld  vs  %lld   fileoffset:%lld\n", fileSize, pFileHeader->GetEndOfFileOffset(), fileOffset);

      // Fixup header offsets
      if (pFileHeader->NameBlockOffset) pFileHeader->NameBlockOffset += fileOffset;
      if (pFileHeader->MetaBlockOffset) pFileHeader->MetaBlockOffset += fileOffset;
      if (pFileHeader->ArrayBlockOffset) pFileHeader->ArrayBlockOffset += fileOffset;
      if (pFileHeader->ArrayFirstOffset) pFileHeader->ArrayFirstOffset += fileOffset;
      if (pFileHeader->BandBlockOffset) pFileHeader->BandBlockOffset += fileOffset;

      pFileHeader->SectionBlockOffset = 0;
      pFileHeader->FileOffset += fileOffset;

      // append the file header to the output file at fileOffset
      int64_t bytesXfer = DefaultFileIO.FileWriteChunk(NULL, outFileHandle, pFileHeader, sizeof(SDS_FILE_HEADER), fileOffset);
      if (bytesXfer != sizeof(SDS_FILE_HEADER)) {
         LOGGING("!!warning file %s failed to write header at offset %lld\n", pSDSDecompress->FileName, fileOffset);
      }

      UpdateSectionData(pSectionData, currentSection++, currentOffset);

      // copy the rest of the file
      fileSize -= sizeof(SDS_FILE_HEADER);
      currentOffset += sizeof(SDS_FILE_HEADER);
      localOffset += sizeof(SDS_FILE_HEADER);

      // Use a 1 MB buffer
      const int64_t BUFFER_SIZE = 1024 * 1024;
      char* pBuffer = (char*)WORKSPACE_ALLOC(BUFFER_SIZE);
      if (!pBuffer) return 0;

      // Read from source at localoffset and copy to output at currentOffset (which starts at fileOffset)
      while (fileSize > 0) {
         int64_t copySize = fileSize;
         if (fileSize > BUFFER_SIZE) {
            copySize = BUFFER_SIZE;
         }
         // read and write
         int64_t sizeRead = DefaultFileIO.FileReadChunk(NULL, inFile, pBuffer, copySize, localOffset);
         int64_t sizeWritten = DefaultFileIO.FileWriteChunk(NULL, outFileHandle, pBuffer, copySize, currentOffset);
         if (sizeRead != sizeWritten || sizeRead != copySize) {
            printf("!!Failed to copy file %s at offset %lld and %lld\n", pSDSDecompress->FileName, currentOffset, localOffset);
         }
         currentOffset += copySize;
         localOffset += copySize;
         fileSize -= copySize;
      }
      WORKSPACE_FREE(pBuffer);

      //-----------------------------------------
      // Read array block and fix that up
      SDS_ARRAY_BLOCK* pDestArrayBlock = (SDS_ARRAY_BLOCK*)WORKSPACE_ALLOC(pFileHeader->ArrayBlockSize);

      bytesXfer =
      DefaultFileIO.FileReadChunk(NULL, inFile, pDestArrayBlock, pFileHeader->ArrayBlockSize, origArrayBlockOffset);
      if (bytesXfer != pFileHeader->ArrayBlockSize) {
         printf("!!warning file %s failed to read array block at offset %lld\n", pSDSDecompress->FileName, origArrayBlockOffset);
      }

      // fixup arrayblocks
      for (int32_t i = 0; i < pFileHeader->ArraysWritten; i++) {
         //printf("start offset %d %lld\n", i, pDestArrayBlock[i].ArrayDataOffset);
         pDestArrayBlock[i].ArrayDataOffset += fileOffset;
      }
      // Write the array block with offsets fixed up
      DefaultFileIO.FileWriteChunk(NULL, outFileHandle, pDestArrayBlock, pFileHeader->ArrayBlockSize, fileOffset + origArrayBlockOffset);

      WORKSPACE_FREE(pDestArrayBlock);

      //------------
      // Check if sections.. if so read back in each section and fix it up
      if (hasSections) {
         int64_t sections = pSDSDecompress->cSectionName.SectionCount;

         for (int64_t section = 1; section < sections; section++) {
            int64_t sectionOffset = pSDSDecompress->cSectionName.pSectionOffsets[section];
            // section within a section
            // read in a new fileheader
            SDS_FILE_HEADER tempFileHeader;
            int64_t bytesRead = DefaultFileIO.FileReadChunk(NULL, inFile, &tempFileHeader, sizeof(SDS_FILE_HEADER), sectionOffset);

            LOGGING("concat: reading section at %lld for output fileoffset %lld\n", sectionOffset, fileOffset);

            if (bytesRead != sizeof(SDS_FILE_HEADER)) {
               printf("!!warning file %s failed to read section header at offset %lld\n", pSDSDecompress->FileName, sectionOffset);
               return 0;
            }

            origArrayBlockOffset = tempFileHeader.ArrayBlockOffset;

            LOGGING("concat: Some offsets %lld %lld %lld  sbo:%lld  fo:%lld\n", tempFileHeader.NameBlockOffset, tempFileHeader.MetaBlockOffset, tempFileHeader.ArrayBlockOffset, tempFileHeader.SectionBlockOffset, tempFileHeader.FileOffset);
            // Fixup header offsets
            if (tempFileHeader.NameBlockOffset) tempFileHeader.NameBlockOffset += fileOffset;
            if (tempFileHeader.MetaBlockOffset) tempFileHeader.MetaBlockOffset += fileOffset;
            if (tempFileHeader.ArrayBlockOffset) tempFileHeader.ArrayBlockOffset += fileOffset;
            if (tempFileHeader.ArrayFirstOffset) tempFileHeader.ArrayFirstOffset += fileOffset;
            if (tempFileHeader.BandBlockOffset) tempFileHeader.BandBlockOffset += fileOffset;

            tempFileHeader.SectionBlockOffset = 0;
            tempFileHeader.FileOffset += fileOffset;

            int64_t newOffset = fileOffset + sectionOffset;

            LOGGING("concat: newoffset: %lld   %lld + %lld\n", newOffset, fileOffset, sectionOffset);
            UpdateSectionData(pSectionData, currentSection++, newOffset);

            int64_t sizeWritten = DefaultFileIO.FileWriteChunk(NULL, outFileHandle, &tempFileHeader, sizeof(SDS_FILE_HEADER), newOffset);
            LOGGING("Wrote subsect fileheader %lld bytes at offset %lld\n", sizeof(SDS_FILE_HEADER), newOffset);
            if (sizeof(SDS_FILE_HEADER) != sizeWritten) {
               printf("!!Failed to copy file %s at offset %lld and %lld\n", pSDSDecompress->FileName, newOffset, sectionOffset);
            }

            // NEW CODE
            // fixup arrayblocks
            //-----------------------------------------
            // Read array block and fix that up
            SDS_ARRAY_BLOCK* pDestArrayBlock2 = (SDS_ARRAY_BLOCK*)WORKSPACE_ALLOC(tempFileHeader.ArrayBlockSize);

            bytesXfer =
                DefaultFileIO.FileReadChunk(NULL, inFile, pDestArrayBlock2, tempFileHeader.ArrayBlockSize, origArrayBlockOffset);
            if (bytesXfer != tempFileHeader.ArrayBlockSize) {
                printf("!!warning file %s failed to read array block at offset %lld\n", pSDSDecompress->FileName, origArrayBlockOffset);
            }

            // fixup arrayblocks
            for (int32_t i = 0; i < tempFileHeader.ArraysWritten; i++) {
                //printf("start offset %d %lld\n", i, pDestArrayBlock2[i].ArrayDataOffset);
                pDestArrayBlock2[i].ArrayDataOffset += fileOffset;
            }
            // Write the array block with offsets fixed up
            DefaultFileIO.FileWriteChunk(NULL, outFileHandle, pDestArrayBlock2, tempFileHeader.ArrayBlockSize, fileOffset + origArrayBlockOffset);
            LOGGING("Wrote arrayblock %lld bytes at offset %lld from orig: %lld\n", tempFileHeader.ArrayBlockSize, fileOffset + origArrayBlockOffset, origArrayBlockOffset);

            WORKSPACE_FREE(pDestArrayBlock2);

         }

      }

      // return current offset in destination file
      //return pSDSDecompress->FileSize;
      return currentOffset;
   }

   //=====================================================
   // In: strOutputFilename: full path filename to create
   // Uses pSDSDecompressFile
   //
   void* SDSConcatFiles(const char* strOutputFilename, int64_t validCount) {
      LOGGING("concat mode!  found %lld files\n", FileCount);

      if (validCount == 0) {
         SetErr_Format(SDS_VALUE_ERROR, "Concat error cannot find any valid files to concat to file: %s.  Error: %s", strOutputFilename, "None");
         return BAD_SDS_HANDLE;
      }

      SDS_FILE_HANDLE  fileHandle =
         DefaultFileIO.FileOpen(strOutputFilename, true, true, false, false);

      if (!fileHandle) {
         SetErr_Format(SDS_VALUE_ERROR, "Concat error cannot create/open file: %s.  Error: %s", strOutputFilename, "none");
         return BAD_SDS_HANDLE;
      }

      // The very first valid file is copied as is
      // Keep track of section offsets (section offset rewritten)
      //
      int64_t fileOffset = 0;
      SDS_FILE_HEADER*  pFileHeader = NULL;
      int64_t sectionCount = 0;

      // Pass 1 count sections
      for (int64_t t = 0; t < FileCount; t++) {
         SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[t];
         if (pSDSDecompress->IsFileValid) {
            if (!pFileHeader) pFileHeader = &pSDSDecompress->FileHeader;

            // Check for sections
            if (pSDSDecompress->FileHeader.SectionBlockOffset) {
               sectionCount += pSDSDecompress->cSectionName.SectionCount;
            }
            else {
               sectionCount += 1;
            }
         }
      }

      // Allocate section offset
      // write to end of file
      if (pFileHeader) {

         // Allocate section data
         int64_t sectionSize = sectionCount * 10;
         int64_t sectionTotalSize = SDS_PAD_NUMBER(sectionSize);
         int64_t currentSection = 0;
         char* pSectionData = (char*)WORKSPACE_ALLOC(sectionTotalSize);

         // Pass 2 append to file
         for (int64_t t = 0; t < FileCount; t++) {
            SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[t];
            if (pSDSDecompress->IsFileValid) {

               int64_t nextOffset =
                  AppendToFile(fileHandle, pSDSDecompress, fileOffset, pSDSDecompress->FileSize, pSectionData, currentSection);

               int64_t padFileSize = SDS_PAD_NUMBER(nextOffset);
               fileOffset = padFileSize;
            }
         }

         // write section header
         int64_t result = DefaultFileIO.FileWriteChunk(NULL, fileHandle, pSectionData, sectionTotalSize, fileOffset);

         if (result != sectionTotalSize) {
            SetErr_Format(SDS_VALUE_ERROR, "Compression error cannot append section %lld at %lld", pFileHeader->SectionBlockReservedSize, pFileHeader->SectionBlockOffset);
         }

         WORKSPACE_FREE(pSectionData);

         // At the end of the file, write out the section names and the file offset to find them
         // Update the first file header (current file header)
         pFileHeader->SectionBlockCount = sectionCount;
         pFileHeader->SectionBlockOffset = fileOffset;
         pFileHeader->SectionBlockSize = sectionSize;
         pFileHeader->SectionBlockReservedSize = sectionTotalSize;
         pFileHeader->StackType = 1;
         DefaultFileIO.FileWriteChunk(NULL, fileHandle, pFileHeader, sizeof(SDS_FILE_HEADER), 0);

      }

      return NULL;
   }

   //========================================
   // main routine for reading in multiple files
   // this routine does NOT stack
   // may return NULL if not all files exist
   // 
   void* ReadManyFiles(SDS_READ_CALLBACKS*  pReadCallbacks, int32_t multiMode) {
      // Open what might be 100+ files
      GetFileInfo(multiMode);

      // If any of the files have sections, we have to grow the list of files
      SDSDecompressFile** pSDSDecompressFileExtra = NULL;
      
      // Check if concat mode
      if (multiMode != SDS_MULTI_MODE_CONCAT_MANY) {
         pSDSDecompressFileExtra = ScanForSections();
      }

      int64_t validCount = 0;
      void* result = NULL;
      int32_t missingfile = -1;

      // Get valid count
      for (int32_t t = 0; t < FileCount; t++) {
         if (pSDSDecompressFile[t]->IsFileValid) validCount++;
         else missingfile = t;
      }

      if (pReadCallbacks->MustExist && missingfile >= 0) {
         // Find first missing file
         SetErr_Format(SDS_VALUE_ERROR, "Not all files found : Expected %lld files.  Missing %s\n", FileCount, pSDSDecompressFile[missingfile]->FileName);
         printf("ReadManyFiles failed!  FileCount %lld. valid %lld.\n", FileCount, validCount);
         return NULL;
      }

      LOGGING("GetInfo ReadManyFiles complete.  FileCount %lld. valid %lld.  mode:%d \n", FileCount, validCount, multiMode);

      // Check if concat mode
      if (multiMode == SDS_MULTI_MODE_CONCAT_MANY) {
         result =SDSConcatFiles(pReadCallbacks->strOutputFilename, validCount);
      }
      else {
         // TJD new code for ver 4.4... check for sections
         // To be completed when we support nested structs within the same file
         //int64_t   FileWithSectionsCount = 0;
         //for (int64_t t = 0; t < FileCount; t++) {
         //   SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[t];

         //   if (pSDSDecompress->IsFileValid && pSDSDecompress->pFileHeader->SectionBlockCount > 1) {
         //      FileWithSectionsCount += pSDSDecompress->pFileHeader->SectionBlockCount;
         //   }
         //   else {
         //      FileWithSectionsCount += 1;
         //   }
         //}
         //if (FileWithSectionsCount > FileCount) {
         //   // Reallocate
         //}

         //if (pReadCallbacks->MustExist) {
         //   // check if all files valid
         //   for (int64_t t = 0; t < FileCount; t++) {
         //      SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[t];

         //      if (!pSDSDecompress->IsFileValid) {
         //         SetErr_Format(SDS_VALUE_ERROR, "Not all files found : Expected %lld files.  Missing %s\n", FileCount, pSDSDecompress->FileName);
         //         printf("Not all files found: Expected %lld files.\n", FileCount);
         //         return NULL;
         //      }
         //   }
         //}

         // ALLOCATE all the Final Callbacks
         SDS_FINAL_CALLBACK* pReadFinalCallback = (SDS_FINAL_CALLBACK*)WORKSPACE_ALLOC(sizeof(SDS_FINAL_CALLBACK) * FileCount);

         // Now build a hash of all the valid filenames
         // Also.. how homogenous are the files... all datasets?  all structs?
         for (int64_t t = 0; t < FileCount; t++) {
            SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[t];

            // 
            pReadFinalCallback[t].strFileName = pSDSDecompress->FileName;

            if (pSDSDecompress->IsFileValid) {

               // Ferry data to callback routine
               //SDS_FINAL_CALLBACK   FinalCallback;

               // Copy over the important data from read class
               // The metadata is temporary and cannot be held onto (copy into your own buffer)
               // Arrays have been allocated based on what caller wanted
               pReadFinalCallback[t].pFileHeader = pSDSDecompress->pFileHeader;
               pReadFinalCallback[t].arraysWritten = pSDSDecompress->pFileHeader->ArraysWritten;
               pReadFinalCallback[t].pArrayBlocks = pSDSDecompress->pArrayBlocks;

               // Only fill in when read data
               pReadFinalCallback[t].pArrayInfo = NULL;
               pReadFinalCallback[t].metaData = pSDSDecompress->MetaData;
               pReadFinalCallback[t].metaSize = pSDSDecompress->MetaSize;
               pReadFinalCallback[t].nameData = pSDSDecompress->pNameData;

               // There may be no sections
               pReadFinalCallback[t].pSectionName = &pSDSDecompress->cSectionName;

            }
            else {
               // invalid file path
               // zero out data
               LOGGING("[%lld] Zeroing bad file\n", t);
               //memset(&pReadFinalCallback[t], 0, sizeof(SDS_READ_FINAL_CALLBACK));
               pReadFinalCallback[t].pFileHeader = NULL;
               pReadFinalCallback[t].arraysWritten = 0;
               pReadFinalCallback[t].pArrayBlocks = NULL;

               // Only fill in when read data
               pReadFinalCallback[t].pArrayInfo = NULL;
               pReadFinalCallback[t].nameData = 0;
               pReadFinalCallback[t].metaData = 0;
               pReadFinalCallback[t].metaSize = 0;
               pReadFinalCallback[t].pSectionName = NULL;
            }

            // Always fill in valid mode
            if (multiMode == SDS_MULTI_MODE_READ_MANY_INFO) {
               pReadFinalCallback[t].mode = COMPRESSION_MODE_INFO;
            }
            else {
               pReadFinalCallback[t].mode = COMPRESSION_MODE_DECOMPRESS_FILE;
            }

         }


         if (multiMode != SDS_MULTI_MODE_READ_MANY_INFO) {
            // read in array data
            result = ReadIOPackets(pReadFinalCallback, pReadCallbacks);
         }
         else {

            // They just want info
            result = pReadCallbacks->ReadFinalCallback(pReadFinalCallback, FileCount);
         }

         WORKSPACE_FREE(pReadFinalCallback);
      }
     

      // Tear it down
      ClearColumnList();

      // Check if we expanded becaue we found extra sections
      if (pSDSDecompressFileExtra) {
         for (int64_t i = 0; i < FileCount; i++) {
            delete pSDSDecompressFileExtra[i];
         }
         WORKSPACE_FREE(pSDSDecompressFileExtra);
      }

      return result;
   }

   //========================================
   SDSDecompressFile* CopyDecompressFileFrom(SDSDecompressFile* pSDSDecompress, int64_t instance) {
      return new SDSDecompressFile(
         pSDSDecompress->FileName,
         pSDSDecompress->pInclude,
         instance,               // instanceindex
         pSDSDecompress->ShareName, 
         pSDSDecompress->pFolderName,
         pSDSDecompress->pSectionsName,
         pSDSDecompress->Mode);
   }

   //--------------------------------
   // Returns NULL if no sections found
   // Otherwise updates FileCount and pSDSDecompressFile with sections
   SDSDecompressFile**  ScanForSections() {
      SDSDecompressFile** pSDSDecompressFileExtra = NULL;

      int64_t   FileWithSectionsCount = 0;

      // First pass, count up how many sections total
      for (int64_t t = 0; t < FileCount; t++) {
         SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[t];

         if (pSDSDecompress->IsFileValid && pSDSDecompress->pFileHeader->SectionBlockCount > 0) {
            // This file has sections that need to be read
            FileWithSectionsCount += pSDSDecompress->pFileHeader->SectionBlockCount;
         }
         else {
            FileWithSectionsCount += 1;
         }
      }

      LOGGING("In scan for sections: %lld vs  %lld\n", FileWithSectionsCount, FileWithSectionsCount);
      // If we found sections then we have to re-expand
      if (FileWithSectionsCount > FileCount) {
         // Reallocate
         // Allocate an array of all the files we need to open
         pSDSDecompressFileExtra = (SDSDecompressFile**)WORKSPACE_ALLOC(sizeof(void*) * FileWithSectionsCount);

         // count up again the same way
         FileWithSectionsCount = 0;

         // TODO: multithread since we know at least one file has a section
         for (int64_t t = 0; t < FileCount; t++) {
            SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[t];

            if (pSDSDecompress->IsFileValid && pSDSDecompress->pFileHeader->SectionBlockCount > 1) {

               // blockcount is how many sections (files within a file) this file has
               int64_t blockcount = pSDSDecompress->pFileHeader->SectionBlockCount;

               // Read in all the sections (new for ver 4.4)
               for (int64_t section = 0; section < blockcount; section++) {
                  int64_t instance = FileWithSectionsCount + section;
                  SDSDecompressFile* pSDSDecompressExtra = CopyDecompressFileFrom(pSDSDecompress, instance);

                  pSDSDecompressFileExtra[instance] = pSDSDecompressExtra;
                  
                  // read in header for section (note if this is the first header (i.e. section ==0), we are reading it again)
                  // TODO: optimization here
                  int64_t fileoffset = pSDSDecompress->cSectionName.pSectionOffsets[section];
                  pSDSDecompressExtra->DecompressFileInternal(pReadCallbacks, 0, fileoffset);
               }
               FileWithSectionsCount += blockcount;
            }
            else {
               pSDSDecompressFileExtra[FileWithSectionsCount] = CopyDecompressFileFrom(pSDSDecompress, FileWithSectionsCount);
               
               // TJD: Check to make sure no memory leaks, also, future optimization - we do not have to read it again
               pSDSDecompressFileExtra[FileWithSectionsCount]->DecompressFileInternal(pReadCallbacks, 0, 0);
               FileWithSectionsCount += 1;
            }
         }

         LOGGING("Found more sections. %lld vs %lld\n", FileCount, FileWithSectionsCount);

         // Swap in the expanded decompressfile with the sections
         pSDSDecompressFile = pSDSDecompressFileExtra;
         FileCount = FileWithSectionsCount;
         return pSDSDecompressFileExtra;
      }
      LOGGING("Did not find more sections. %lld\n", FileCount);
      return NULL;
   }

   //========================================
   // main routine for reading in and stacking multiple files
   // more complex routine that makes multiple passes
   // note: pSDSDecompressFile allocated up to FileCount
   // may return NULL when all files do not exist
   void* ReadAndStackFiles(SDS_READ_CALLBACKS*  pReadCallbacks, int32_t multiMode) {
      // Open what might be 100+ files
      GetFileInfo(multiMode);

      SDSDecompressFile** pSDSDecompressFileExtra = ScanForSections();

      void* result = NULL;
      int64_t validPos = 0;
      int64_t validCount = 0;

      // Capture and clip percent of space to reserve
      double reserveSpace = pReadCallbacks->ReserveSpace;
      if (reserveSpace < 0.0) reserveSpace = 0.0;
      if (reserveSpace > 1.0) reserveSpace = 1.0;

      // NOTE: back into single threaded mode here...
      //ClearColumnList();
      int32_t missingfile = -1;

      // Get valid count
      for (int32_t t = 0; t < FileCount; t++) {
         if (pSDSDecompressFile[t]->IsFileValid) validCount++;
         else missingfile = t;
      }

      if (pReadCallbacks->MustExist && missingfile >=0) {
         // Find first missing file
         SetErr_Format(SDS_VALUE_ERROR, "Not all files found : Expected %lld files.  Missing %s\n", FileCount, pSDSDecompressFile[missingfile]->FileName);
         printf("ReadAndStackFiles failed!  FileCount %lld. valid %lld.\n", FileCount, validCount);
         return NULL;
      }

      LOGGING("GetInfo ReadAndStackFiles complete.  FileCount %lld. valid %lld. reserve space set to %lf\n", FileCount, validCount, reserveSpace);

      // Allocate dataset row length for blank rows (gaps)
      AllocateDatasetLengths(validCount);

      // Now build a hash of all the valid filenames
      // Also.. how homogenous are the files... all datasets?  all structs?
      for (int32_t t = 0; t < FileCount; t++) {
         bool isStruct = FALSE;
         bool isDataset = FALSE;
         bool isArray = FALSE;
         bool isOneFile = FALSE;

         SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[t];
         if (pSDSDecompress->IsFileValid) {
            auto fileType = pSDSDecompress->FileHeader.FileType;
            // Make sure we have homogneous filetypes
            isDataset |= (fileType == SDS_FILE_TYPE_TABLE);
            isDataset |= (fileType == SDS_FILE_TYPE_DATASET);
            isStruct  |= (fileType == SDS_FILE_TYPE_STRUCT);
            isArray   |= (fileType == SDS_FILE_TYPE_ARRAY);
            isOneFile |= (fileType == SDS_FILE_TYPE_ONEFILE);

            // Try to find out how many valid columns
            for (int64_t c = 0; c < pSDSDecompress->NameCount; c++) {
               const char* columnName = pSDSDecompress->pArrayNames[c];

               // Include exclude check
               // If ONEFILE and NO FOLDERS
               if (IsNameIncluded(pIncludeList, pFolderList, columnName, isOneFile)) {
               //if ((!pIncludeList || pIncludeList->IsIncluded(columnName)) &&
               //   (!pFolderList || pFolderList->IsIncluded(columnName))
               //   )
               //{

                  // if this column is not in the hash, it will be added
                  // The very first column has the proper attributes
                  LOGGING("[%d][%lld] %s", t, c, columnName);

                  SDS_ARRAY_BLOCK *pArrayBlock = &pSDSDecompress->pArrayBlocks[c];
                  int32_t arrayEnum = pSDSDecompress->pArrayEnums[c];

                  // If the column is new, SDS_COLUMN_KING will be allocated
                  AddColumnList(
                     validPos,
                     validCount,
                     columnName,
                     arrayEnum,
                     t,  // file row (for all files, valid or not)
                     c,  // column pos
                     pArrayBlock);

                  // See if we need to fixup default dataset length
                  if (isDataset) {

                     // Check if we have a stackable dataset
                     int32_t mask = SDS_FLAGS_ORIGINAL_CONTAINER;
                     if ((arrayEnum & mask) == mask) {
                        if (pArrayBlock && pArrayBlock->NDim > 0) {

                           int64_t dlength = pArrayBlock->Dimensions[0];

                           // Get the dataset size
                           if (pDatasets[validPos].Length != 0) {
                              if (pDatasets[validPos].Length != dlength) {
                                 printf("WARNING: datasets not same length %lld v. %lld for column %s\n",
                                    pDatasets[validPos].Length,
                                    dlength,
                                    columnName);
                              }
                           }

                           if (validPos > validCount) {
                              printf("!! internal error on validPos\n");
                           }

                           // Remember the length of this dataset
                           pDatasets[validPos].Length = dlength;
                           LOGGING("Thomas band %lld %lld\n", pSDSDecompress->FileHeader.BandBlockCount, pSDSDecompress->FileHeader.BandSize);
                        }
                     }
                  }
               }
            }
            // increment for every valid file
            validPos++;

            //pSDSDecompress->pNameData;
         }
      }

      LOGGING("\nTotal column stats  uniq:%d, first:%d, conv:%d, strfix:%d, dim:%d, colgaps:%d, valid: %lld\n",
         TotalUniqueColumns,
         TotalFirstColumns,
         TotalConversions,
         TotalStringFixups,
         TotalDimensionProblems,
         TotalColumnGaps,
         validCount);

      const char* pFirstFileName = "<no valid file>";
      if (FileCount > 0) {
         SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[0];
         pFirstFileName = pSDSDecompress->FileName;
      }

      //===========================================================================
      // final tally
      // array offset has one more entry for the last entry
      int64_t rowLengthToAlloc = 0;


      //if ((isStruct + isDataset + isArray) != 1) {
      //   SetErr_Format(SDS_VALUE_ERROR, "MultiDecompress error -- all the filetypes must be the same type (struct or dataset or array)\nFilename: %s\n", pFirstFileName);
      //   goto EXIT_EARLY;
      //}
      bool isGood = FALSE;

      if (validCount == 0) {
         SetErr_Format(SDS_VALUE_ERROR, "MultiDecompress error -- none of the files were valid: %s\n", pFirstFileName);
      }
      else

      //if (TotalConversions != 0) {
      //   SetErr_Format(SDS_VALUE_ERROR, "MultiDecompress error -- detected a conversion problem %s\n", pFirstFileName);
      //} else

      if (TotalDimensionProblems != 0) {
         SetErr_Format(SDS_VALUE_ERROR, "MultiDecompress error -- detected a dimension fixup problem %s\n", pFirstFileName);
      }
      else {
         isGood = TRUE;
      }

      if (isGood) {

         if (TotalStringFixups != 0) {
            // ? nothing to do yet, fixup when copying
         }

         //===========================================================
         // Go ahead and read..
         // Clear any previous file errors
         PrintIfErrors();
         ClearErrors();

         int64_t totalIOPackets = TotalUniqueColumns * validCount;

         // Allocate MultiIO PACKETS -- worst case assuming every col * rows
         SDS_MULTI_IO_PACKETS* pMultiIOPackets = SDS_MULTI_IO_PACKETS::Allocate(totalIOPackets, &pReadCallbacks->Filter);

         // Also for each column we create SDSArrayInfo
         SDSArrayInfo*         pManyDestInfo = (SDSArrayInfo*)WORKSPACE_ALLOC(sizeof(SDSArrayInfo)*TotalUniqueColumns);
         SDSFilterInfo*        pFilterInfo = NULL;
         
         // Check if we are filtering the data with a boolean mask
         if (pReadCallbacks->Filter.pBoolMask) {
            // Worst case allocation (only valid files calculated)
            LOGGING("~ALLOCATING for filtering\n");
            pFilterInfo = (SDSFilterInfo*)WORKSPACE_ALLOC(sizeof(SDSFilterInfo)*FileCount);
            pReadCallbacks->Filter.pFilterInfo = pFilterInfo;
         }

         // position of totalIOPackets
         validPos = 0;

         // for calculating boolean filter mask
         int32_t     hasFilter = 0;
         int64_t   filterTrueCount = 0;
         int64_t   boolMaskLength = pReadCallbacks->Filter.BoolMaskLength;
         int32_t     fileTypeStackable = 0;

         // The first valid file determines the filetype when stacking
         for (int32_t f = 0; f < FileCount; f++) {
            SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[f];
            if (pSDSDecompress->IsFileValid) {
               int32_t fileType = pSDSDecompress->FileHeader.FileType;
               fileTypeStackable = (fileType == SDS_FILE_TYPE_DATASET ||
                  fileType == SDS_FILE_TYPE_TABLE ||
                  fileType == SDS_FILE_TYPE_ARRAY);
               break;
            }
         }

         LOGGING("Filter info: boolmasklength:%lld  stackable: %d \n", boolMaskLength, fileTypeStackable);

         // Fill in all the IOPACKETs
         // Skip over invalid files, loop over all valid filenames
         for (int64_t col = 0; col < TotalUniqueColumns; col++) {

            // Only one king
            SDS_COLUMN_KING*  pKing = &ColumnVector[col];
            // Appears to be a save bug
            // When saving a Dataset, the categories have STACKABLE set but the bin array on has ORIGINAL_CONTAINER
            // When saving a Struct, the categories do not have STACKABLE set
            //int32_t mask = SDS_FLAGS_ORIGINAL_CONTAINER;
            LOGGING("[%lld]Col: %s -- ColPos:%lld  FileRow:%d pBlock:%p Enum:%d  ftype:%d\n", col, pKing->ColName, pKing->ColPos, pKing->FileRow, pKing->ppArrayBlocks, pKing->ArrayEnum, fileTypeStackable);

            int64_t    currentOffset = 0;
            int64_t    currentUnfilteredOffset = 0;
            int64_t    row = 0;
            int64_t    filteredOut = 0;
            int64_t*   pArrayOffsets = pKing->pArrayOffsets;
            int64_t*   pOriginalArrayOffsets = pKing->pOriginalArrayOffsets;
            int64_t*   pOriginalLengths = pKing->pOriginalLengths;

            SDS_ARRAY_BLOCK* pMasterBlock = &pKing->KingBlock;
            //LOGGING("master %p ", pMasterBlock);

            int32_t mask = SDS_FLAGS_ORIGINAL_CONTAINER;
            bool  isFilterable = FALSE;           

            if (pFilterInfo && 
               (pKing->ArrayEnum & mask) == mask &&
               (pKing->ArrayEnum & (SDS_FLAGS_SCALAR | SDS_FLAGS_META | SDS_FLAGS_NESTED)) == 0 &&
               fileTypeStackable) {

               // only the first column that is stackable is used to calculate
               isFilterable = TRUE;
            }

            // PASS #1...
            // Check for any new columns that might pop up or disappear
            // These create gaps that are often filled in with invalids
            // For every valid file there is SDS_ARRAY_BLOCK
            // If the filetype is 2 and the enum is 1 (SDS_FLAGS_STACKABLE)
            for (int32_t f = 0; f < FileCount; f++) {
               SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[f];
               if (pSDSDecompress->IsFileValid) {

                  // Store the offset per stack position used later when read in data
                  pArrayOffsets[row] = currentOffset;
                  pOriginalArrayOffsets[row] = currentUnfilteredOffset;

                  SDS_ARRAY_BLOCK* pArrayBlock = pKing->ppArrayBlocks[row];

                  //LOGGING("{%d} {%p} ", f, pArrayBlock);
                  // TODO -- CHECK MASTER BLOCK??
                  SDS_COMPATIBLE isCompatible = { TRUE, FALSE, FALSE, FALSE };

                  // The array might be missing in another file
                  // This happens when users add new rows
                  if (pArrayBlock) {
                     isCompatible = IsArrayCompatible(pKing->ColName, pMasterBlock, pArrayBlock, FALSE);
                  }

                  int64_t calcLength = 0;
                  if (!pArrayBlock || pOriginalLengths[row] == 0) {
                     // NO IO PACKET (GAP)
                     LOGGING(">>> gap fill for row: %lld   col: %lld  name: %s\n", row, col, pKing->ColName);
                     calcLength = pDatasets[row].Length;
                  }
                  else {
                     //LOGGING(">>> normal fill for row: %lld   col: %lld  name: %s  length: %lld  enum:%d  ftype: %d\n", row, col, pKing->ColName, pOriginalLengths[row], pKing->ArrayEnum, pSDSDecompress->FileHeader.FileType);
                     // arrays might be incompatible, but fixup later
                     calcLength = pOriginalLengths[row];
                  }

                  // If this is the first pass for the filter on a Dataset, calculate boolean mask info
                  //if (hasFilter == 1) {
                  if (isFilterable) {
                     filterTrueCount = 0;
                     if (currentUnfilteredOffset < boolMaskLength) {
                        int64_t bLength = calcLength;
                        if (currentUnfilteredOffset + calcLength > boolMaskLength)
                           bLength = boolMaskLength - currentUnfilteredOffset;

                        if (hasFilter) {
                           // this code trusts that original container in a Dataset is truthful
                           filterTrueCount = pFilterInfo[row].TrueCount;
                        }
                        else {
                           // First pass calculate how many true values in bool mask
                           filterTrueCount =
                              SumBooleanMask((int8_t*)pReadCallbacks->Filter.pBoolMask + currentUnfilteredOffset, bLength);
                        }
                     }
                     pFilterInfo[row].TrueCount = filterTrueCount;
                     // similar to a cumsum
                     pFilterInfo[row].RowOffset = row == 0 ? 0 : pFilterInfo[row - 1].RowOffset + pFilterInfo[row - 1].TrueCount;

                     // Keep track of how many completely filtered out
                     if (filterTrueCount == 0) {
                        pSDSDecompress->IsFileValidAndNotFilteredOut = FALSE;
                        filteredOut++;
                     }                     

                     // The length has been shortened by the filter
                     currentUnfilteredOffset += calcLength;
                     calcLength = filterTrueCount;

                     LOGGING("ROW %lld has true: %lld  ufo: %lld  for offset: %lld vs %lld\n", row, pFilterInfo[row].TrueCount, currentUnfilteredOffset, currentOffset, pFilterInfo[row].RowOffset);
                  }
                  else {
                     // check if completely filtered out??
                     if (hasFilter) {
                        if (pFilterInfo[row].TrueCount == 0) {
                           //printf("REMOVING ENTIRE row: %lld   col : %lld  name : %s\n", row, col, pKing->ColName);
                           calcLength = 0;
                           pOriginalLengths[row] = 0;
                        }
                     }
                     currentUnfilteredOffset += calcLength;
                  }

                  currentOffset += calcLength;

                  // the valid row count
                  row++;
               }
            }

            if (isFilterable && hasFilter == 0) {
               // Readjust TrueCount based on what we calculated
               pReadCallbacks->Filter.BoolMaskTrueCount = row == 0 ? 0 : pFilterInfo[row - 1].RowOffset + pFilterInfo[row - 1].TrueCount;
               LOGGING("Final TrueCount %lld  masterRow: %lld\n", pReadCallbacks->Filter.BoolMaskTrueCount, row);
               hasFilter++;
            }


            // Calculate how many rows to allocate based on all the stacking
            rowLengthToAlloc = currentOffset;
            if (reserveSpace > 0.0) {
               rowLengthToAlloc += (int64_t)(currentOffset * reserveSpace);
            }

            //CreateFilter(row, pReadCallbacks->Filter);

            // Need to see if we are filtering -- if we are certain files might not be loaded at all since they are clipped
            // Fixup last row + 1
            pArrayOffsets[row] = rowLengthToAlloc;
            pOriginalArrayOffsets[row] = currentUnfilteredOffset;
            pKing->TotalRowLength = rowLengthToAlloc;

            LOGGING("stack: %d   totalplusreseve:%lld  totalrowlength: %lld  rows: %lld\n", isFilterable, rowLengthToAlloc, currentOffset, row);
            // TODO: -- allocate one big array and fixup IO PACKETS

            SDS_ALLOCATE_ARRAY   sdsArrayCallback;
            int64_t                dimensions[SDS_MAX_DIMS];
            int64_t                strides[SDS_MAX_DIMS];
            int64_t                oneRowSize = pMasterBlock->ItemSize;

            // Use a destination info we allocated
            sdsArrayCallback.pDestInfo = &pManyDestInfo[col];
            sdsArrayCallback.ndim = pMasterBlock->NDim;

            for (int32_t j = 0; j < SDS_MAX_DIMS; j++) {
               dimensions[j] = pMasterBlock->Dimensions[j];
               strides[j] = pMasterBlock->Strides[j];
            }
            sdsArrayCallback.dims = dimensions;
            sdsArrayCallback.strides = strides;

            sdsArrayCallback.numpyType = FixupDType(pMasterBlock->DType, pMasterBlock->ItemSize);
            sdsArrayCallback.itemsize = pMasterBlock->ItemSize;
            sdsArrayCallback.data = NULL;  // for shared memory this is set

            sdsArrayCallback.numpyFlags = pMasterBlock->Flags;
            sdsArrayCallback.pArrayName = pKing->ColName;
            sdsArrayCallback.sdsFlags = pKing->ArrayEnum;

            sdsArrayCallback.pDestInfo->pData = NULL;
            sdsArrayCallback.pDestInfo->pArrayObject = NULL;

            // TODO: oneRowSize not correct for multidimensional
            for (int32_t j = 1; j < pMasterBlock->NDim; j++) {
               oneRowSize *= dimensions[j];
            }

            bool wasFiltered = FALSE;

            // Caller will fill info pArrayObject and pData
            // pData is valid for shared memory
            if (pReadCallbacks->AllocateArrayCallback) {
               dimensions[0] = rowLengthToAlloc;

               // Check dims and fortran vs cstyle
               // NOTE: use absolute value of last stride vs first stride to determine C or F
               if (pMasterBlock->NDim > 1 && pMasterBlock->Flags  &  SDS_ARRAY_F_CONTIGUOUS) {
                  //printf("!!!likely internal error with fortran array > dim 1 and upgrading dtype!\n");
                  int64_t isize = pMasterBlock->ItemSize;
                  strides[0] = isize;
                  for (int32_t j = 1; j < pMasterBlock->NDim; j++) {
                     strides[j] = dimensions[j - 1] * strides[j - 1];
                  }
               }

               // FILTERING check...
               wasFiltered |= PossiblyShrinkArray(&sdsArrayCallback, pReadCallbacks, fileTypeStackable);

               LOGGING("Allocating col: %s  lengthplusr: %lld  length: %lld   itemsize: %lld  strides: %lld  dims:%d  dtype:%d  wasFiltered:%d\n", sdsArrayCallback.pArrayName, rowLengthToAlloc, currentOffset, sdsArrayCallback.itemsize, strides[0], (int)sdsArrayCallback.ndim, sdsArrayCallback.numpyType, wasFiltered);
               // callback into python or matlab to allocate memory
               // this will fill in pData and pArrayObject
               pReadCallbacks->AllocateArrayCallback(&sdsArrayCallback);
               LOGGING("Array allocated at %p for object %p\n", sdsArrayCallback.pDestInfo->pData, sdsArrayCallback.pDestInfo->pArrayObject);
            }

            // Fill in destination information
            CopyFromBlockToInfo(pMasterBlock, sdsArrayCallback.pDestInfo);
            if (wasFiltered) {
               pMasterBlock->Flags |= SDS_ARRAY_FILTERED;
               sdsArrayCallback.pDestInfo->Flags |= SDS_ARRAY_FILTERED;
            }

            // update master block to total length
            pMasterBlock->Dimensions[0] = rowLengthToAlloc;

            // reset for every column
            row = 0;

            // PASS #2... build IOPackets
            // Loop over all files
            for (int32_t f = 0; f < FileCount; f++) {
               SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[f];
               if (pSDSDecompress->IsFileValid) {

                  // Get next valid IO packet and ArrayInfo
                  SDS_IO_PACKET*    pIOPacket = &pMultiIOPackets->pIOPacket[validPos];
                  SDSArrayInfo*     pDestInfo = &pMultiIOPackets->pDestInfo[validPos];

                  // Used for offset in file
                  // Used for compressed/decompressed size
                  SDS_ARRAY_BLOCK*  pArrayBlock = pKing->ppArrayBlocks[row];

                  // Build IO Packets
                  pIOPacket->pReadCallbacks = pReadCallbacks;

                  // NOTE: will be NULL for gap fills
                  pIOPacket->pBlockInfo = pArrayBlock;
                  pIOPacket->pMasterBlock = &pKing->KingBlock;

                  pIOPacket->FileHandle = pSDSDecompress->SDSFile;
                  pIOPacket->pFileHeader = pSDSDecompress->pFileHeader;
                  pIOPacket->pMemoryIO = &DefaultMemoryIO; // WHAT TO DO??
                  pIOPacket->CompMode = COMPRESSION_MODE_DECOMPRESS;

                  // TODO -- CHECK MASTER BLOCK??
                  pIOPacket->Compatible = { TRUE, FALSE, FALSE, FALSE };
                  pIOPacket->ArrayOffset = pArrayOffsets[row];
                  pIOPacket->OriginalArrayOffset = pOriginalArrayOffsets[row];

                  // only when stacking is there a colname
                  pIOPacket->ColName = pKing->ColName;
                  pIOPacket->StackPosition = row;
                  if (pOriginalLengths[row] == 0) {
                     pIOPacket->StackPosition = -1;
                  }

                  //printf("Stack position %lld  %s\n", row, pKing->ColName);
                  // The array might be missing in another file
                  // This happens when they add new rows
                  if (pArrayBlock) {
                     pIOPacket->Compatible = IsArrayCompatible(pKing->ColName, pMasterBlock, pArrayBlock, FALSE);
                  }

                  //printf("array offset %lld\n", pArrayOffsets[row]);

                  int64_t gapLength = pArrayOffsets[row + 1] - pArrayOffsets[row];

                  // Check for gap
                  if (!pArrayBlock || pOriginalLengths[row] == 0) {
                     SDS_ARRAY_BLOCK* pMasterBlock = &pKing->KingBlock;

                     // NO IO PACKET (GAP)
                     LOGGING(">>> gap fill %lld  for row: %lld   col: %lld   orig:%lld  ds:%lld  mdtype:%d\n", gapLength, row, col, pOriginalLengths[row], pDatasets[row].Length, pMasterBlock->DType);
                     CopyFromBlockToInfo(pMasterBlock, pDestInfo);
                  }

                  // possibly change length to missing length
                  pDestInfo->ArrayLength = gapLength;

                  // needed for convertinplace?
                  // possibly change first dimension also
                  pDestInfo->Dimensions[0] = gapLength;

                  pDestInfo->pArrayObject = sdsArrayCallback.pDestInfo->pArrayObject;

                  if ((pIOPacket->pMasterBlock->NDim >= 2 && pIOPacket->pMasterBlock->Flags & SDS_ARRAY_F_CONTIGUOUS) ||
                     (pIOPacket->Compatible.NeedsStringFixup & 2)) {
                     // TODO: filter two dimensions
                     //pReadCallbacks->WarningCallback("**matlab!  arrayOffset:%lld  totalrows:%lld  rows:%lld\n", pIOPacket->ArrayOffset, pKing->TotalRowLength, pArrayBlock->Dimensions[0]);
                     pDestInfo->pData = sdsArrayCallback.pDestInfo->pData;
                  }
                  else {
                     // This is the important one, this is where the IO will be copied into
                     pDestInfo->pData = sdsArrayCallback.pDestInfo->pData + (oneRowSize * pArrayOffsets[row]);
                     LOGGING("Fixing up pData for stackposition: %lld  itemoffset:%lld  onerowsize:%lld  mbitemsize:%d  dest:%p\n", row, pArrayOffsets[row], oneRowSize, pMasterBlock->ItemSize, pDestInfo->pData);
                  }

                  if (validPos > totalIOPackets) {
                     printf("!!!internal error validPos vs totalIOPackets\n");
                  }

                  validPos++;

                  // this is the valid row count
                  row++;
               }
            }
         }

         //=========================================
         // Read all our IOpackets if we have any
         if (validPos) {
            //--------- ALLOCATE COMPRESS ARRAYS ---
            //---------  DECOMPRESS ARRAYS -------------
            // Multithreaded work and we tell caller when we started/stopped
            void* saveState = pReadCallbacks->BeginAllowThreads();
            g_cMathWorker->DoMultiThreadedWork((int)validPos, DecompressMultiArray, pMultiIOPackets);
            pReadCallbacks->EndAllowThreads(saveState);
         }

         LOGGING("Multistack done reading -- returning %d cols\n", TotalUniqueColumns);

         // ALLOCATE all the Final Callbacks
         SDS_STACK_CALLBACK*        pMultiFinalCallback =
            (SDS_STACK_CALLBACK*)WORKSPACE_ALLOC(sizeof(SDS_STACK_CALLBACK) * TotalUniqueColumns);

         SDS_STACK_CALLBACK_FILES*  pMultiCallbackFileInfo =
            (SDS_STACK_CALLBACK_FILES*)WORKSPACE_ALLOC(sizeof(SDS_STACK_CALLBACK_FILES) * validCount);

         // Now build a hash of all the valid filenames
         // Also.. how homogenous are the files... all datasets?  all structs?
         for (int64_t col = 0; col < TotalUniqueColumns; col++) {
            // Only one king, get the first file that had this column
            SDS_COLUMN_KING*     pKing = &ColumnVector[col];
            SDSDecompressFile*   pSDSDecompress = pSDSDecompressFile[pKing->FileRow];
            SDSArrayInfo*        pDestInfo = &pManyDestInfo[col];
            int64_t*               pArrayOffsets = pKing->pArrayOffsets;

            if (pSDSDecompress == NULL) {
               printf("!!!internal error in final multistack loop\n");

            }
            else {

               // Ferry data to callback routine
               //SDS_FINAL_CALLBACK   FinalCallback;

               // Copy over the important data from read class
               // The metadata is temporary and cannot be held onto (copy into your own buffer)
               // Arrays have been allocated based on what caller wanted

               pMultiFinalCallback[col].pArrayObject = pDestInfo->pArrayObject;

               // alocate int64 from this to make array slices for partitioned data
               pMultiFinalCallback[col].pArrayOffsets = pArrayOffsets;
               pMultiFinalCallback[col].ArrayName = pKing->ColName;
               pMultiFinalCallback[col].ArrayEnum = pKing->ArrayEnum;
               pMultiFinalCallback[col].pArrayBlock = &pKing->KingBlock;
            }
         }

         int64_t vrow = 0;
         for (int32_t f = 0; f < FileCount; f++) {
            SDSDecompressFile* pSDSDecompress = pSDSDecompressFile[f];
            if (pSDSDecompress->IsFileValid) {
               pMultiCallbackFileInfo[vrow].Filename = pSDSDecompress->FileName;
               pMultiCallbackFileInfo[vrow].MetaData = pSDSDecompress->MetaData;
               pMultiCallbackFileInfo[vrow].MetaDataSize = pSDSDecompress->MetaSize;
               pMultiCallbackFileInfo[vrow].pFileHeader = pSDSDecompress->pFileHeader;
               vrow++;
            }
         }

         result = pReadCallbacks->StackFinalCallback(pMultiFinalCallback, TotalUniqueColumns, pMultiCallbackFileInfo, &pReadCallbacks->Filter, validCount);

         WORKSPACE_FREE(pMultiCallbackFileInfo);
         WORKSPACE_FREE(pMultiFinalCallback);
         WORKSPACE_FREE(pManyDestInfo);
         WORKSPACE_FREE(pFilterInfo);

         // Tear it down
         ClearColumnList();

         //---------- CLEAN UP MEMORY AND HANDLES ---------
         SDS_MULTI_IO_PACKETS::Free(pMultiIOPackets);

         // Check if we expanded becaue we found extra sections
         if (pSDSDecompressFileExtra) {
            for (int64_t i = 0; i < FileCount; i++) {
               delete pSDSDecompressFileExtra[i];
            }
            WORKSPACE_FREE(pSDSDecompressFileExtra);
         }
      }

      // Reduce list to valids
      LOGGING("ReadAndStackFiles complete %p\n", result);
      return result;
   }

};

extern "C" {

   //=====================================================
   // Write an SDS File (platform free -- python free)
   //
   // INPUT:
   // fileName -- name of file to write to
   //
   // Arrays to write
   // aInfo
   // arrayCount - number of arrays
   // totalItemSize -?? notused
   //
   // metaData -- block of bytes to store as metadata
   // metaDataSize -- 
   //
   DllExport int32_t SDSWriteFile(
      const char *fileName,
      const char *shareName,   // can be NULL
      SDS_STRING_LIST *folderName,  // can be NULL

                              // arrays to save information
      SDS_WRITE_INFO*         pWriteInfo,
      SDS_WRITE_CALLBACKS*    pWriteCallbacks)
   {
      ClearErrors();

      return SDSWriteFileInternal(fileName, shareName, folderName, pWriteInfo, pWriteCallbacks);
   }



   //=====================================================
   //  filename must be provided
   //  ReadFinal must be provided
   //  mode  can be to get information or decompress file
   // ---
   //  sharedMemory <optional>  if provided will check shared memory first
   //  ReadFromSharedMemory   must be provided
   // 
   // Returns what the user specified in ReadFinal
   DllExport void* SDSReadFile(
      const char *fileName,
      const char *shareName,     // can be NULL
      SDS_STRING_LIST *folderName,    // can be NULL
      SDS_STRING_LIST *sectionsName,    // can be NULL
      SDS_READ_INFO* pReadInfo,  // Default to COMPRESSION_MODE_DECOMPRESS_FILE
      SDS_READ_CALLBACKS* pReadCallbacks)
   {
      ClearErrors();

      SDSIncludeExclude includeList;
      SDSIncludeExclude folderList;
      SDSIncludeExclude sectionsList;

      SDS_STRING_LIST*     pInclusionList = pReadCallbacks->pInclusionList;

      if (pInclusionList) {
         includeList.AddItems(pInclusionList, 0);
      }

      if (folderName) {
         folderList.AddItems(folderName, '/');
      }

      if (sectionsName) {
         sectionsList.AddItems(sectionsName, 0);
      }

      // Place class object on the stack so it self cleans up
      SDSDecompressFile decompress(fileName, &includeList, 0, shareName, &folderList, &sectionsList, pReadInfo->mode);

      return decompress.DecompressFile(pReadCallbacks, 0, 0);
   }


   //=====================================================
   //  filename must be provided
   //  ReadFinal must be provided
   //  mode  can be to get information or decompress file
   // ---
   //  sharedMemory <optional>  if provided will check shared memory first
   //  ReadFromSharedMemory   must be provided
   // 
   // Returns what the user specified in ReadFinal
   DllExport void* SDSReadManyFiles(
      SDS_MULTI_READ*      pMultiRead,
      SDS_STRING_LIST*     pInclusionList,      // may be set to NULL
      SDS_STRING_LIST*     pFolderInclusionList,      // may be set to NULL
      SDS_STRING_LIST*     pSectionInclusionList,      // may be set to NULL
      int64_t                fileCount,
      int32_t                  multiMode,
      SDS_READ_CALLBACKS*  pReadCallbacks)
   {
      void* result = NULL;
      ClearErrors();

      SDSIncludeExclude includeList;
      SDSIncludeExclude folderList;
      SDSIncludeExclude sectionsList;

      if (pInclusionList) {
         includeList.AddItems(pInclusionList, 0);
      }

      if (pFolderInclusionList) {
         folderList.AddItems(pFolderInclusionList, '/');
      }

      if (pSectionInclusionList) {
         sectionsList.AddItems(pSectionInclusionList, 0);
      }

      // when mode =SDS_MULTI_MODE_STACK_OR_READMANY the stack mode is ambiguous
      if (multiMode == SDS_MULTI_MODE_STACK_OR_READMANY) {
         // Read in the header... to determine type of file
         if (fileCount > 0) {
            SDS_FILE_HANDLE  fileHandle =
               DefaultFileIO.FileOpen(pMultiRead[0].pFileName, false, true, false, false);

            if (!fileHandle) {
               SetErr_Format(SDS_VALUE_ERROR, "Decompression error cannot create/open file: %s.  Error: %s", pMultiRead[0].pFileName, GetLastErrorMessage());
               multiMode = SDS_MULTI_MODE_ERROR;
            }
            else {
               // Read file header
               SDS_FILE_HEADER tempFileHeader;
               int64_t result =
                  ReadFileHeader(fileHandle, &tempFileHeader, 0, pMultiRead[0].pFileName);

               if (result != 0) {
                  multiMode = SDS_MULTI_MODE_ERROR;
               }
               else {
                  if (tempFileHeader.StackType == 1) {
                     multiMode = SDS_MULTI_MODE_STACK_MANY;
                  }
                  else {
                     multiMode = SDS_MULTI_MODE_READ_MANY;
                  }
               }

               DefaultFileIO.FileClose(fileHandle);
            }
         }
      }

      if (multiMode != SDS_MULTI_MODE_ERROR) {

         // Allocate an array of all the files we need to open
         SDSDecompressFile** pSDSDecompressFile = (SDSDecompressFile**)WORKSPACE_ALLOC(sizeof(void*) * fileCount);

         for (int64_t i = 0; i < fileCount; i++) {
            LOGGING("...%s\n", pMultiRead[i].pFileName);
            pSDSDecompressFile[i] = new SDSDecompressFile(pMultiRead[i].pFileName, &includeList, i, NULL, &folderList, &sectionsList, multiMode == SDS_MULTI_MODE_CONCAT_MANY ? SDS_MULTI_MODE_CONCAT_MANY : COMPRESSION_MODE_INFO);
         }

         SDSDecompressManyFiles  manyFiles(pSDSDecompressFile, &includeList, &folderList, &sectionsList, fileCount, pReadCallbacks);

         if (multiMode == SDS_MULTI_MODE_READ_MANY || multiMode == SDS_MULTI_MODE_READ_MANY_INFO || multiMode == SDS_MULTI_MODE_CONCAT_MANY) {
            result = manyFiles.ReadManyFiles(pReadCallbacks, multiMode);
         }
         else {
            result = manyFiles.ReadAndStackFiles(pReadCallbacks, multiMode);
         }

         // Shut it down
         for (int64_t i = 0; i < fileCount; i++) {
            delete pSDSDecompressFile[i];
         }

         // cleanup memory
         WORKSPACE_FREE(pSDSDecompressFile);
      }

      // result may be NULL if not all files existed
      return result;
   }

   DllExport char* SDSGetLastError() {
      return g_errorbuffer;
   }

   DllExport int32_t CloseSharedMemory(void* pMapStruct) {
      PMAPPED_VIEW_STRUCT pMappedStruct = (PMAPPED_VIEW_STRUCT)pMapStruct;
      if (pMappedStruct) {
         UtilSharedMemoryEnd(pMappedStruct);
         return TRUE;
      }
      return FALSE;
   }

   DllExport int32_t CloseDecompressFile(void* pInput) {      
      SDSDecompressFile* pSDSDecompressFile = (SDSDecompressFile * )pInput;
      delete pSDSDecompressFile;
      return TRUE;
   }

   //DllExport void SDSClearBuffers() {
   //   for (int32_t i = 0; i < SDS_MAX_CORES; i++) {
   //      if (g_DecompressContext[i] != NULL) {
   //         ZSTD_freeDCtx(g_DecompressContext[i]);
   //         g_DecompressContext[i] = NULL;
   //      }
   //   }
   //}

}

