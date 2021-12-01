#pragma once

/*
Macro symbol definitions to simplify conditional code compilation within
riptide.

References:
* https://sourceforge.net/p/predef/wiki/Compilers/

*/

/*
Platform/OS detection
*/

#if defined(_WIN32)
    // Target OS is Windows
    #define RT_OS_WINDOWS 1

#elif defined(__linux__)
    // Target OS is Linux
    #define RT_OS_LINUX 1

    // Target OS is UNIX-like
    #define RT_OS_FAMILY_UNIX 1

#elif defined(__APPLE__)
    // Target OS is macOS or iOS
    #define RT_OS_DARWIN 1

    // Target OS is UNIX-like
    #define RT_OS_FAMILY_UNIX 1

    // Target OS is BSD-like
    #define RT_OS_FAMILY_BSD 1

#elif __FreeBSD__
    // Target OS is FreeBSD
    #define RT_OS_FREEBSD 1

    // Target OS is UNIX-like
    #define RT_OS_FAMILY_UNIX 1

    // Target OS is BSD-like
    #define RT_OS_FAMILY_BSD 1

#else
    // If we can't detect the OS, make it a compiler error; compilation is likely to
    // fail anyway due to not having any working implementations of some functions,
    // so at least we can make it obvious why the compilation is failing.
    #error Unable to detect/classify the target OS.

#endif /* Platform/OS detection */

/*
Compiler detection.
The order these detection checks operate in is IMPORTANT -- use CAUTION if
changing or reordering them!
*/

#if defined(__clang__)
    // Compiler is Clang/LLVM.
    #define RT_COMPILER_CLANG 1

#elif defined(__GNUC__)
    // Compiler is GCC/g++.
    #define RT_COMPILER_GCC 1

#elif defined(__INTEL_COMPILER) || defined(_ICC)
    // Compiler is the Intel C/C++ compiler.
    #define RT_COMPILER_INTEL 1

#elif defined(_MSC_VER)
    /*
    This check needs to be towards the end; a number of compilers (e.g. clang, Intel
    C/C++) define the _MSC_VER symbol when running on Windows, so putting this check
    last means we should have caught any of those already and this should be
    bona-fide MSVC.
    */
    // Compiler is the Microsoft C/C++ compiler.
    #define RT_COMPILER_MSVC 1

#else
    // Couldn't detect the compiler.
    // We could allow compilation to proceed anyway, but the compiler/platform
    // behavior detection below won't pass and it's important for correctness so
    // this is an error.
    #error Unable to detect/classify the compiler being used.

#endif /* compiler detection */

/*
Compiler behavior detection.
For conciseness/correctness in riptide code, we define some additional symbols
here specifying certain compiler behaviors. This way any code depending on these
behaviors expresses it in terms of the behavior rather than whether it's being
compiled under a specific compiler(s) and/or platforms; this in turn makes it
easier to support new compilers and platforms just by adding the necessary
defines here.
*/

#if ! defined(RT_COMPILER_MSVC)
    // Indicates whether the targeted compiler/platform defaults to emitting vector
    // load/store operations requiring an aligned pointer when a vector pointer is
    // dereferenced (so any such pointers must be aligned to prevent segfaults).
    // When zero/false, the targeted compiler/platform emits unaligned vector
    // load/store instructions by default.
    #define RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED 1
#else
    // Indicates whether the targeted compiler/platform defaults to emitting vector
    // load/store operations requiring an aligned pointer when a vector pointer is
    // dereferenced (so any such pointers must be aligned to prevent segfaults).
    // When zero/false, the targeted compiler/platform emits unaligned vector
    // load/store instructions by default.
    #define RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED 0
#endif /* RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED */
