
#if defined(_WIN32) && !defined(__GNUC__)

//#include "winbase.h"

typedef BYTE BOOLEAN;

class CFileReadWrite
{
public:
   CFileReadWrite();
   ~CFileReadWrite();

   BOOLEAN Open(const char* fileName, BOOLEAN writeOption = false, BOOLEAN overlapped = false, BOOLEAN directIO = false);

   static BOOLEAN FlushCache(CHAR DriveLetter);

   BOOLEAN CancelIO();

   DWORD ReadChunk(void* buffer, UINT32 count);

   DWORD ReadChunkAsync(
      void* buffer,
      UINT32 count,
      DWORD* lastError,
      OVERLAPPED* pOverlapped);

   BOOLEAN WaitForLastRead(DWORD* lastError, OVERLAPPED* pos);

   BOOLEAN WaitIoComplete(OVERLAPPED* pos);

   DWORD WriteChunk(void* buffer, UINT32 count);

   // File existence check
   static BOOLEAN FileExists(const char* filename) {

      if (INVALID_FILE_ATTRIBUTES == GetFileAttributes(filename) && GetLastError() == ERROR_FILE_NOT_FOUND)
      {
         //File not found
         return false;
      }
      return true;
   }

   DWORD WriteChunkAsync(
      void* buffer,
      UINT32 count,
      DWORD* lastError,
      OVERLAPPED* pOverlapped,
      BOOLEAN bWaitOnIO = false);

   // Seek from current location
   INT64 SeekCurrentEx(INT64 pos);

   // Seek from beginning of file
   INT64 SeekBeginEx(INT64 pos);

   BOOLEAN Close();

   //-----------------------------------
   HANDLE Handle = INVALID_HANDLE_VALUE;
   TCHAR  FileName[_MAX_FNAME] = { 0 };

   BOOLEAN WriteOption = FALSE;
   BOOLEAN Overlapped = FALSE;
   BOOLEAN DirectIO = FALSE;

   INT64 BufferPos = 0;
   OVERLAPPED OverlappedIO;
   OVERLAPPED OverlappedIO2;

   DWORD LastError = 0;
   DWORD LastError2 = 0;

};

#else
// LINUX
class CFileReadWrite
{
public:
   CFileReadWrite();
   ~CFileReadWrite();

   BOOLEAN Open(const char* fileName, BOOLEAN writeOption = false, BOOLEAN overlapped = false, BOOLEAN directIO = false);

   FILE*    Handle;
};


#endif