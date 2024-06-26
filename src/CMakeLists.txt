set(TARGET_NAME riptide_cpp)

set(HEADERS
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
    FileReadWrite.h
    GroupBy.h
    HashFunctions.h
    HashLinear.h
    Hook.h
    interrupt.h
    is_member_tg.h
    Logger.h
    logging/logging.h
    MathThreads.h
    MathWorker.h
    Merge.h
    missing_values.h
    MultiKey.h
    one_input.h
    operations.h
    Recycler.h
    Reduce.h
    RipTide.h
    SDSFile.h
    SDSFilePython.h
    simd.h
    SharedMemory.h
    Sort.h
    TileRepeat.h
    TimeWindow.h
    TypeSystem.h
    UnaryOps.h
    ZstdCompress.h)

set(SOURCES
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
    GroupBy.cpp
    HashFunctions.cpp
    HashLinear.cpp
    Hook.cpp
    interrupt.cpp
    is_member_tg.cpp
    Logger.cpp
    logging/logging.cpp
    MathThreads.cpp
    MathWorker.cpp
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
    TypeSystem.cpp
    UnaryOps.cpp
    ZstdCompress.cpp)

add_library(${TARGET_NAME} SHARED
    ${HEADERS}
    ${SOURCES})

target_compile_definitions(${TARGET_NAME} PRIVATE BUILDING_RIPTIDE_CPP)

target_include_directories(${TARGET_NAME} PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

# Match MSVC debug compile settings to those used in external release libs.
target_compile_definitions(${TARGET_NAME} PUBLIC
    $<$<AND:$<CONFIG:Debug>,$<STREQUAL:${CMAKE_CXX_COMPILER_ID},MSVC>>:_ITERATOR_DEBUG_LEVEL=0>)
target_compile_options(${TARGET_NAME} PUBLIC
    $<$<AND:$<CONFIG:Debug>,$<STREQUAL:${CMAKE_CXX_COMPILER_ID},MSVC>>:-MD>)

target_link_libraries(${TARGET_NAME}
    rt_common_settings
    Python3::Python
    zstd::libzstd_shared
    TBB::tbb
    TBB::tbbmalloc
    $<$<PLATFORM_ID:Linux>:rt>
)

# Configure the library name to identify it as a Python extension module.
set_target_properties(${TARGET_NAME} PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXT_LIB_SUFFIX}"
    )

if(WIN32)
    set(RUNTIME_SPECIFIER RUNTIME)
elseif(UNIX)
    set(RUNTIME_SPECIFIER LIBRARY)
else()
    message(FATAL_ERROR "Cannot specify runtime component")
endif()

install(TARGETS ${TARGET_NAME} ${RUNTIME_SPECIFIER} DESTINATION .)
