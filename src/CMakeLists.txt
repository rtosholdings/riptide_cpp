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
    is_member_tg.h
    MathThreads.h
    MathWorker.h
    Merge.h
    #missing_values.h
    MultiKey.h
    one_input.h
    Recycler.h
    Reduce.h
    RipTide.h
    SDSFile.h
    SDSFilePython.h
    SharedMemory.h
    Sort.h
    TileRepeat.h
    TimeWindow.h
    TypeSystem.h
    UnaryOps.h)

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
    is_member_tg.cpp
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
    UnaryOps.cpp)

add_library(${TARGET_NAME} SHARED
    ${HEADERS}
    ${SOURCES})

target_compile_definitions(${TARGET_NAME} PRIVATE BUILDING_RIPTIDE_CPP)

get_target_property(RT_SOURCE_DIR riptide_cpp SOURCE_DIR)

target_include_directories(${TARGET_NAME} PRIVATE
    ${EXTERNAL_ZSTD_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    )

target_compile_options(${TARGET_NAME} PRIVATE
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},MSVC>:-bigobj>)

target_link_libraries(${TARGET_NAME}
    rt_common_settings
    Python3::Python
    libzstd_static
    TBB::tbb
    TBB::tbbmalloc
    absl::base
    absl::flat_hash_map
    $<$<PLATFORM_ID:Linux>:rt>
    )

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
