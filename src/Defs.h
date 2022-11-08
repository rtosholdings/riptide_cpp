#pragma once

// Basic cross-platform definitions.

// Export DLL section. For the optimization reason behind
// this (and why you don't need DllExport unless it's data)
// see https://docs.microsoft.com/en-us/cpp/build/importing-function-calls-using-declspec-dllimport?view=msvc-170
#if defined(_WIN32) && ! defined(__GNUC__)
    #ifdef BUILDING_RIPTIDE_CPP
        #define DllExport __declspec(dllexport)
    #else
        #define DllExport __declspec(dllimport)
    #endif
#else
    #define DllExport
#endif
