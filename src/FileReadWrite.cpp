#include "FileReadWrite.h"
#include "CommonInc.h"

#if defined(_WIN32) && ! defined(__GNUC__)

    #define LogVerbose(...)

void LogLevel(int64_t level, LPCTSTR szString, ...)
{
    printf("%lld %s\n", level, szString);
}

// defined in SDSFile
const char * GetLastErrorMessage(char * errmsg, DWORD last_error);

    #define LogErrorLE(lastError) \
        { \
            CHAR errmsg[512]; \
            GetLastErrorMessage(errmsg, lastError); \
            LogLevel(1, "File: %s  Line: %d  Function: %s  Error: %s\n", __FILE__, __LINE__, __FUNCTION__, errmsg); \
        }
    #define LogWarningLE(lastError) \
        { \
            CHAR errmsg[512]; \
            GetLastErrorMessage(errmsg, lastError); \
            LogLevel(1, "File: %s  Line: %d  Function: %s  Error: %s\n", __FILE__, __LINE__, __FUNCTION__, errmsg); \
        }
    #define LogInformLE(lastError) \
        { \
            CHAR errmsg[512]; \
            GetLastErrorMessage(errmsg, lastError); \
            LogLevel(1, "File: %s  Line: %d  Function: %s  Error: %s\n", __FILE__, __LINE__, __FUNCTION__, errmsg); \
        }
    #define LogVerboseLE(lastError) \
        { \
            CHAR errmsg[512]; \
            GetLastErrorMessage(errmsg, lastError); \
            LogLevel(1, "File: %s  Line: %d  Function: %s  Error: %s\n", __FILE__, __LINE__, __FUNCTION__, errmsg); \
        }

    #define LogErrorX(...) \
        LogLevel(1, "File: %s  Line: %d  Function: %s\n", __FILE__, __LINE__, __FUNCTION__); \
        LogLevel(1, __VA_ARGS__)
    #define LogWarningX(...) \
        LogLevel(2, "File: %s  Line: %d  Function: %s\n", __FILE__, __LINE__, __FUNCTION__); \
        LogLevel(2, __VA_ARGS__)
    #define LogInformX(...) \
        LogLevel(3, "File: %s  Line: %d  Function: %s\n", __FILE__, __LINE__, __FUNCTION__); \
        LogLevel(3, __VA_ARGS__)
    #define LogVerboseX(...) \
        LogLevel(4, "File: %s  Line: %d  Function: %s\n", __FILE__, __LINE__, __FUNCTION__); \
        LogLevel(4, __VA_ARGS__)

CFileReadWrite::CFileReadWrite()
{
    RtlZeroMemory(&OverlappedIO, sizeof(OVERLAPPED));
    RtlZeroMemory(&OverlappedIO2, sizeof(OVERLAPPED));
}

CFileReadWrite::~CFileReadWrite() {}

// Returns true if cache was successfully flushed
bool CFileReadWrite::FlushCache(CHAR driveLetter)
{
    bool result = false;
    char szVolumeName[32] = "\\\\.\\X:";

    szVolumeName[4] = driveLetter;

    LogVerbose("Flushing drive letter %s", szVolumeName);

    // open the existing file for reading
    HANDLE tempHandle = CreateFile(szVolumeName, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_WRITE | FILE_SHARE_READ, 0,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, 0);

    if (tempHandle != INVALID_HANDLE_VALUE)
    {
        result = FlushFileBuffers(tempHandle);

        if (! result)
        {
            LogErrorLE(GetLastError());
        }

        CloseHandle(tempHandle);
    }

    return result;
}

bool CFileReadWrite::Open(const char * fileName, bool writeOption, bool overlapped, bool directIO)
{
    strcpy_s(FileName, sizeof(FileName), fileName);
    WriteOption = writeOption;
    Overlapped = overlapped;
    DirectIO = directIO;

    //
    // if they try to open without closing
    //
    if (Handle != (void *)0)
    {
        CloseHandle(Handle);
        Handle = (void *)0;
    }

    // open the existing file for reading
    Handle = CreateFile(FileName, GENERIC_READ | (WriteOption ? GENERIC_WRITE : 0), FILE_SHARE_READ | FILE_SHARE_WRITE, 0,
                        WriteOption ? CREATE_ALWAYS : OPEN_EXISTING,

                        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN | (Overlapped ? FILE_FLAG_OVERLAPPED : 0) |
                            (DirectIO ? FILE_FLAG_NO_BUFFERING : 0),

                        0);

    if (Handle != INVALID_HANDLE_VALUE)
    {
        if (Handle != NULL)
        {
            return true;
        }
        else
        {
            Handle = (void *)0;
            return false;
        }
    }
    Handle = (void *)0;
    return false;
}

// Cancels all IO for the Handle
bool CFileReadWrite::CancelIO()
{
    return CancelIo(Handle);
}

//---------------------------------------------------------
// Standard read ahead
DWORD CFileReadWrite::ReadChunk(void * buffer, uint32_t count)
{
    DWORD n = 0;
    if (! Overlapped)
    {
        if (! ReadFile(Handle, buffer, count, &n, NULL))
        {
            LogErrorX("!!Read failed %s", FileName);
            return 0;
        }
        BufferPos = BufferPos + n;
        LastError = 0;
        return n;
    }
    else
    {
        // LogError("!! Suspicious code path for async read %s %d", FileName,
        // LastError);
        OverlappedIO.hEvent = NULL;
        OverlappedIO.InternalHigh = 0;
        OverlappedIO.Internal = 0;
        OverlappedIO.OffsetHigh = (uint32_t)(BufferPos >> 32);
        OverlappedIO.Offset = (uint32_t)BufferPos;

        bool bReadDone;

        OVERLAPPED * pos = &OverlappedIO;
        {
            bReadDone = ReadFile(Handle, buffer, count, &n, pos);

            LastError = GetLastError();
            if (! bReadDone && LastError == ERROR_IO_PENDING)
            {
                // Wait for IO to complete
                bReadDone = GetOverlappedResult(Handle, pos, &n, true);

                if (! bReadDone)
                {
                    LastError = GetLastError();
                    LogErrorX("!!Read failed %s %d", FileName, LastError);
                    return 0;
                }
            }

            if (! bReadDone)
            {
                LastError = GetLastError();
                LogErrorX("!!Read failed %s %d", FileName, LastError);
                return 0;
            }
        }
        BufferPos = BufferPos + n;
        LastError = 0;
        return n;
    }
}

//----------------------------------------------------------------------------------------------
// MAIN Read routine for Async IO
// The API may return and data will be pending
DWORD CFileReadWrite::ReadChunkAsync(void * buffer, uint32_t count, DWORD * lastError, OVERLAPPED * pOverlapped)
{
    DWORD n = 0;
    if (! Overlapped)
    {
        if (! ReadFile(Handle, buffer, count, &n, NULL))
        {
            LogErrorX("!!Read failed %s", FileName);
            return 0;
        }
        BufferPos = BufferPos + n;
        *lastError = 0;
        return n;
    }
    else
    {
        pOverlapped->InternalHigh = 0;
        pOverlapped->Internal = 0;
        pOverlapped->OffsetHigh = (uint32_t)(BufferPos >> 32);
        pOverlapped->Offset = (uint32_t)BufferPos;

        bool bReadDone;

        // Async version of READ
        bReadDone = ReadFile(Handle, buffer, count, &n, pOverlapped);

        if (! bReadDone)
        {
            *lastError = GetLastError();
            if (*lastError == ERROR_IO_PENDING)
            {
                // OPERATION IS PENDING (this is expected)
                // Go ahead and pretend we read all of it so that
                // the BufferPos counter is correct for the next async read
                BufferPos = BufferPos + count;
                return 0;
            }
            else
            {
                LogError("!!Read failed %s %d", FileName, *lastError);
                return 0;
            }
        }

        LogError("!! no need to wait for data");
        *lastError = 0;
        BufferPos = BufferPos + n;
        return n;
    }
}

//----------------------------------------------------------------------------
// ONLY if the last call was to ReadAsync can this be called
bool CFileReadWrite::WaitForLastRead(DWORD * lastError, OVERLAPPED * pos)
{
    // Check if last time operation was pending
    if (*lastError == ERROR_IO_PENDING)
    {
        DWORD n = 0;
        // Wait for this to complete
        bool bReadDone = GetOverlappedResult(Handle, pos, &n, true);

        if (! bReadDone)
        {
            *lastError = GetLastError();
            if (*lastError == ERROR_INVALID_PARAMETER)
            {
                LogError("!! Read Invalid Param %s %d", FileName, *lastError);
            }
            else if (*lastError == ERROR_HANDLE_EOF)
            {
                LogError("!! Read EOF %s %d", FileName, *lastError);
            }
            else if (*lastError == 998)
            {
                LogError("!! Error 998, invalid access to memory location %s %d", FileName, *lastError);
            }
            else
            {
                LogError("!!Read failed %s %d", FileName, *lastError);
            }
            return false;
        }
    }
    else
    {
        LogError("!!! Not calling GetOverlapped because data available? %d", *lastError);
    }
    return true;
}

bool CFileReadWrite::WaitIoComplete(OVERLAPPED * pos)
{
    DWORD n = 0;
    bool bWriteDone;
    // Wait for IO to complete
    bWriteDone = GetOverlappedResult(Handle, pos, &n, true);

    if (! bWriteDone)
    {
        LastError = GetLastError();
        LogError("!!Write failed %s %d", FileName, LastError);
        return false;
    }

    return true;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Will write for chunk to be written )non-Async version
// Returns bytes written
DWORD CFileReadWrite::WriteChunk(void * buffer, uint32_t count)
{
    DWORD n = 0;
    if (! Overlapped)
    {
        if (! WriteFile(Handle, buffer, count, &n, 0))
        {
            LogError("!!Write failed %s", FileName);
            return 0;
        }
        LogVerbose("RawStream wrote %d", n);
        BufferPos = BufferPos + n;
        LastError = 0;
        return n;
    }
    else
    {
        // LogError("!! Suspicious code path for async read %s %d", FileName,
        // LastError);
        OverlappedIO.hEvent = 0;
        OverlappedIO.InternalHigh = 0;
        OverlappedIO.Internal = 0;
        OverlappedIO.OffsetHigh = (uint32_t)(BufferPos >> 32);
        OverlappedIO.Offset = (uint32_t)BufferPos;

        bool bWriteDone;

        OVERLAPPED * pos = &OverlappedIO;
        {
            bWriteDone = WriteFile(Handle, buffer, count, &n, pos);

            LastError = GetLastError();
            if (! bWriteDone && LastError == ERROR_IO_PENDING)
            {
                // Wait for IO to complete
                bWriteDone = GetOverlappedResult(Handle, pos, &n, true);

                if (! bWriteDone)
                {
                    LastError = GetLastError();
                    LogError("!!Write failed %s %d", FileName, LastError);
                    return 0;
                }
            }

            if (! bWriteDone)
            {
                LastError = GetLastError();
                LogError("!!Write failed %s %d", FileName, LastError);
                return 0;
            }
        }
        LogVerbose("RawStream A wrote %d", n);
        BufferPos = BufferPos + n;
        LastError = 0;
        return n;
    }
}

//----------------------------------------------------------------------------------------------
// MAIN Read routine for Async IO
// The API may return and data will be pending
// Returns: number of bytes written
// Returns: error result if any, in lastError
DWORD CFileReadWrite::WriteChunkAsync(void * buffer, uint32_t count, DWORD * lastError, OVERLAPPED * pOverlapped, bool bWaitOnIO)
{
    DWORD n = 0;
    if (! Overlapped)
    {
        if (! WriteFile(Handle, buffer, count, &n, NULL))
        {
            LogError("!!Write failed %s", FileName);
            return 0;
        }
        LogVerbose("RawStream ca wrote %d", count);
        BufferPos = BufferPos + n;
        *lastError = 0;
        return n;
    }
    else
    {
        pOverlapped->InternalHigh = 0;
        pOverlapped->Internal = 0;
        pOverlapped->OffsetHigh = (uint32_t)(BufferPos >> 32);
        pOverlapped->Offset = (uint32_t)BufferPos;

        bool bWriteDone;

        // Async version of WRITE
        bWriteDone = WriteFile(Handle, buffer, count, &n, pOverlapped);

        if (! bWriteDone)
        {
            *lastError = GetLastError();
            if (*lastError == ERROR_IO_PENDING)
            {
                // Wait for IO to complete
                if (bWaitOnIO)
                {
                    GetOverlappedResult(Handle, pOverlapped, &n, true);
                }
                // OPERATION IS PENDING (this is expected)
                // Go ahead and pretend we wrote all of it so that
                // the BufferPos counter is correct for the next async write
                BufferPos = BufferPos + count;

                LogVerbose("RawStream ca wrote %d", count);
                // Assume this will complete
                return count;
            }
            else
            {
                LogError("!!Write failed %s %d", FileName, *lastError);
                return 0;
            }
        }

        // LogError("!! no need to wait for data");
        *lastError = 0;
        LogVerbose("RawStream ca wrote %d", n);
        BufferPos = BufferPos + n;
        return n;
    }
}

//--------------------------------------------------------------------
// Seek from current position
// Returns offset
int64_t CFileReadWrite::SeekCurrentEx(int64_t pos)
{
    int64_t result = 0;

    LARGE_INTEGER temp;
    temp.QuadPart = pos;

    bool bResult = SetFilePointerEx(Handle, temp, (PLARGE_INTEGER)&result, SEEK_CUR);

    if (! bResult)
    {
        LogError("!!! Seek current to %llu failed!", pos);
    }

    // Have to keep track of position for overlapped IO
    if (Overlapped)
    {
        BufferPos += pos;

        // In async mode, the file pointer may not yet be updated
        return BufferPos;
    }

    BufferPos = result;
    return result;
}

//--------------------------------------------------------------------
// Seek from start of file position
// Move method = 0 = FILE_BEGIN
int64_t CFileReadWrite::SeekBeginEx(int64_t pos)
{
    int64_t result = 0;

    LARGE_INTEGER temp;
    temp.QuadPart = pos;

    bool bResult = SetFilePointerEx(Handle, temp, (PLARGE_INTEGER)&result, SEEK_SET);

    if (! bResult)
    {
        LogError("!!! Seek begin to %llu failed!", pos);
    }

    // Have to keep track of position for overlapped IO
    if (Overlapped)
    {
        BufferPos = pos;
        return BufferPos;
    }

    BufferPos = result;
    return result;
}

//--------------------------------------------------------------------
bool CFileReadWrite::Close()
{
    if (Handle != NULL)
    {
        bool Result = CloseHandle(Handle);
        Handle = NULL;
        return Result;
    }
    return false;
}

extern "C"
{
    CFileReadWrite * ReadWriteOpen(const char * fullFileName, bool writeOption = false, bool overlapped = true,
                                   bool directIO = false)
    {
        CFileReadWrite * pReadWrite = new CFileReadWrite();

        if (pReadWrite)
        {
            pReadWrite->Open(fullFileName, writeOption, overlapped, directIO);
        }

        return pReadWrite;
    }

    DWORD ReadChunk(CFileReadWrite * pReadWrite, void * buffer, uint32_t count)
    {
        return pReadWrite->ReadChunk(buffer, count);
    }

    DWORD ReadChunkAsync(CFileReadWrite * pReadWrite, void * buffer, uint32_t count, DWORD * lastError, OVERLAPPED * pOverlapped)
    {
        return pReadWrite->ReadChunkAsync(buffer, count, lastError, pOverlapped);
    }

    bool ReadWriteClose(CFileReadWrite * pReadWrite)
    {
        return pReadWrite->Close();
    }

    bool FlushCache(CHAR driveLetter)
    {
        return CFileReadWrite::FlushCache(driveLetter);
    }
};

#else

#endif
