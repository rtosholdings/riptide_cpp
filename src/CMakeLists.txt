set(TARGET_NAME riptide_cpp)

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
    ${RT_SOURCE_DIR}/../external/zstd/lib
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${EXTERNAL_ZSTD_INCLUDE_DIR})

target_link_libraries(${TARGET_NAME}
    Python3::Python
    libzstd_static)

# Configure the library name to identify it as a Python extension module.
set_target_properties(${TARGET_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ${PYTHON_EXT_LIB_SUFFIX})
