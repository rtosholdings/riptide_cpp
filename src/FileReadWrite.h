#pragma once

#if defined(_WIN32) && ! defined(__GNUC__)

//#include "winbase.h"

class CFileReadWrite
{
public:
    CFileReadWrite();
    ~CFileReadWrite();

    bool Open(const char * fileName, bool writeOption = false, bool overlapped = false, bool directIO = false);

    static bool FlushCache(CHAR DriveLetter);

    bool CancelIO();

    DWORD ReadChunk(void * buffer, UINT32 count);

    DWORD ReadChunkAsync(void * buffer, UINT32 count, DWORD * lastError, OVERLAPPED * pOverlapped);

    bool WaitForLastRead(DWORD * lastError, OVERLAPPED * pos);

    bool WaitIoComplete(OVERLAPPED * pos);

    DWORD WriteChunk(void * buffer, UINT32 count);

    // File existence check
    static bool FileExists(const char * filename)
    {
        if (INVALID_FILE_ATTRIBUTES == GetFileAttributes(filename) && GetLastError() == ERROR_FILE_NOT_FOUND)
        {
            // File not found
            return false;
        }
        return true;
    }

    DWORD WriteChunkAsync(void * buffer, UINT32 count, DWORD * lastError, OVERLAPPED * pOverlapped, bool bWaitOnIO = false);

    // Seek from current location
    INT64 SeekCurrentEx(INT64 pos);

    // Seek from beginning of file
    INT64 SeekBeginEx(INT64 pos);

    bool Close();

    //-----------------------------------
    HANDLE Handle = INVALID_HANDLE_VALUE;
    TCHAR FileName[_MAX_FNAME] = { 0 };

    bool WriteOption = false;
    bool Overlapped = false;
    bool DirectIO = false;

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

    bool Open(const char * fileName, bool writeOption = false, bool overlapped = false, bool directIO = false);

    FILE * Handle;
};

#endif
