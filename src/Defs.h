#pragma once

// Basic cross-platform definitions.

// Export DLL section. For the optimization reason behind
// this (and why you don't need RT_DLLEXPORT unless it's data)
// see https://docs.microsoft.com/en-us/cpp/build/importing-function-calls-using-declspec-dllimport?view=msvc-170
#if defined(_MSC_VER) && ! defined(__GNUC__)
    #ifdef BUILDING_RIPTIDE_CPP
        #define RT_DLLEXPORT __declspec(dllexport)
    #else
        #define RT_DLLEXPORT __declspec(dllimport)
    #endif
#else
    #define RT_DLLEXPORT
#endif

#if defined(_MSC_VER) && ! defined(__clang__)
    #define RT_FORCEINLINE __forceinline
#else
    #define RT_FORCEINLINE inline __attribute__((always_inline))
#endif // defined(_MSC_VER) && !defined(__clang__)
