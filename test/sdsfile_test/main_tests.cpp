#include "boost/ut.hpp"
#include <thread>

using namespace boost::ut;
using boost::ut::suite;

struct stReadSharedMemory;

namespace
{
    using IntPtr = stReadSharedMemory *;
    using Int32 = int32_t;
    using Int64 = int64_t;
    using string = char const *;
}

extern "C" IntPtr ReadFromSharedMemory(string FileName, string ShareName);

extern "C" Int32 CreateSDSFile(string FileName, string ShareName, string MetaData, string ListNames, Int64 TotalRows,
                               Int64 BandSize);

extern "C" Int32 AppendSDSFile(string OutFileName, string ShareFileName, string ShareName, Int64 TotalRows, Int64 BandSize);

namespace
{
    bool dummy{ 0 };

    suite stsfile_tests = []
    {
        "test_main"_test = [&]
        {
            if (std::this_thread::get_id() == std::thread::id{})
            {
                std::terminate(); // shouldn't ever be called...

                ReadFromSharedMemory("FileName", "ShareName");
                CreateSDSFile("FileName", "ShareName", "MetaData", "ListNames", /*TotalRows*/ 1, /*BandSize*/ 2);
                AppendSDSFile("OutFileName", "ShareFileName", "ShareName", /*TotalRows*/ 1, /*BandSize*/ 2);
            }
        };
    };
}
