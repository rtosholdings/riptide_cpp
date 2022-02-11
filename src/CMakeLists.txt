set(TARGET_NAME riptide_cpp)

if(PROJ_COMPILER_FRONTEND STREQUAL "MSVC")
    add_compile_options(-FC -Zc:__cplusplus)
    add_compile_options("$<$<CONFIG:Debug>:-Od;-Z7>")
    add_compile_options("$<$<CONFIG:Release>:-Ox;-Ob2;-Oi;-Ot>")
    add_compile_options(/W3 /WX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(-permissive- -d2FH4- -Zc:strictStrings-)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-Wno-unused-variable -Wno-unused-function)
    endif()
elseif(PROJ_COMPILER_FRONTEND STREQUAL "GNU")
    add_compile_options("$<$<CONFIG:Debug>:-O0>")
    add_compile_options("$<$<CONFIG:Release>:-O2>")
    add_compile_options(-g -x c++ -mavx2 -mbmi2 -fpermissive -pthread)
    add_compile_options(-falign-functions=32 -fno-strict-aliasing)
    #add_compile_options(-falign-loops=32)
    add_compile_options(-Wall -Werror -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-parameter -Wno-ignored-attributes)
    add_compile_options(-Wno-format) # TODO: Remove this and fix all the printf format mismatches
elseif(PROJ_COMPILER_FRONTEND STREQUAL "LLVM")
    add_compile_options("$<$<CONFIG:Debug>:-O0>")
    add_compile_options("$<$<CONFIG:Release>:-O2>")
    add_compile_options(-g -x c++ -mavx2 -mbmi2 -fpermissive -pthread)
    add_compile_options(-falign-functions=32 -fno-strict-aliasing)
    #add_compile_options(-falign-loops=32)
    add_compile_options(-Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-parameter -Wno-ignored-attributes)
    add_compile_options(-Wno-format) # TODO: Remove this and fix all the printf format mismatches
else()
     message(FATAL_ERROR "Unexpected proj compiler front-end, ${PROJ_COMPILER_FRONTEND}")
endif()

set(HEADERS
    #ArenaMemory.h
    Array.h
    BasicMath.h
    Bins.h
    BitCount.h
    CommonInc.h
    Compare.h
    Compress.h
    Convert.h
    DateTime.h
    Ema.h
    #FileReadWrite.h
    GroupBy.h
    HashFunctions.h
    HashLinear.h
    Hook.h
    MathThreads.h
    MathWorker.h
    Merge.h
    #missing_values.h
    MultiKey.h
    #ndarray.h
    #numpy_traits.h
    one_input.h
    #platform_detect.h
    Recycler.h
    Reduce.h
    RipTide.h
    SDSFile.h
    SDSFilePython.h
    SharedMemory.h
    Sort.h
    #stdafx.h
    TileRepeat.h
    TimeWindow.h
    #TypeSystem.h
    UnaryOps.h)

set(SOURCES
    #ArenaMemory.cpp
    Array.cpp
    BasicMath.cpp
    Bins.cpp
    BitCount.cpp
    Compare.cpp
    Compress.cpp
    Convert.cpp
    CRC32.cpp
    DateTime.cpp
    Ema.cpp
    #FileReadWrite.cpp
    GroupBy.cpp
    HashFunctions.cpp
    HashLinear.cpp
    Hook.cpp
    MathThreads.cpp
    MathWorker.cpp
    #mat_interp.cpp
    Merge.cpp
    MultiKey.cpp
    one_input.cpp
    Recycler.cpp
    Reduce.cpp
    RipTide.cpp
    SDSFile.cpp
    SDSFilePython.cpp
    SharedMemory.cpp
    Sort.cpp
    strptime5.cpp
    TileRepeat.cpp
    TimeWindow.cpp
    #TypeSystem.cpp
    UnaryOps.cpp)

add_library(${TARGET_NAME} SHARED
    ${HEADERS}
    ${SOURCES})

get_target_property(RT_SOURCE_DIR riptide_cpp SOURCE_DIR)

target_include_directories(${TARGET_NAME} PRIVATE
    ${EXTERNAL_ZSTD_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS} )

target_link_libraries(${TARGET_NAME}
    Python3::Python
    libzstd_static
    $<$<PLATFORM_ID:Linux>:rt>)

# Configure the library name to identify it as a Python extension module.
set_target_properties(${TARGET_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ${PYTHON_EXT_LIB_SUFFIX})

if(WIN32)
    set(RUNTIME_SPECIFIER RUNTIME)
elseif(UNIX)
    set(RUNTIME_SPECIFIER LIBRARY)
else()
    message(FATAL_ERROR "Cannot specify runtime component")
endif()

install(TARGETS riptide_cpp ${RUNTIME_SPECIFIER} DESTINATION .)
