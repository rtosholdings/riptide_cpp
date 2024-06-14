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

# Setting the target properties doesn't cause the corresponding .pdb to be renamed to
# match, which then causes the .pdb not to be considered part of the output by the 'install' target.
# Fix this so the .pdb is available in the installed Python egg.
if (WIN32)
  get_filename_component(_PYTHON_EXTENSION_SUFFIX_NOEXT ${_PYTHON_EXTENSION_SUFFIX} NAME_WLE)

  # Create the PDB filename by taking the "normal" output name of the library and appending
  # the CPython-extension suffix (without filename extension).
  #get_target_property(RIPTIDE_OUTPUT_NAME riptide_cpp LIBRARY_OUTPUT_NAME) # TODO: Use this instead to ensure consistency; this needs to be fixed though, it gives RIPTIDE_OUTPUT_NAME-NOTFOUND
  set(RIPTIDE_OUTPUT_NAME "riptide_cpp")
  string(CONCAT _PYTHON_EXTENSION_PDB_NAME ${RIPTIDE_OUTPUT_NAME} ${_PYTHON_EXTENSION_SUFFIX_NOEXT})

  set_target_properties(riptide_cpp PROPERTIES PDB_NAME ${_PYTHON_EXTENSION_PDB_NAME})
endif()

################################################################################
# Set some additional optimization options (compiler-specific, but that's handled by CMake).
################################################################################
# Optional IPO. Do not use IPO if it's not supported by compiler.
# If compiling with MSVC, this is meant to enable the use of Link-Time Code Generation (LTCG).
check_ipo_supported(RESULT ipo_is_supported OUTPUT error_text_if_any)
if(ipo_is_supported)
  set_property(TARGET riptide_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(WARNING "IPO is not supported: ${error_text_if_any}")
endif()

################################################################################
# Preserve debugging information.
################################################################################
# The RelWithDebInfo configuration passes compiler flags to preserve/emit debugging info
# but also uses /O2 (MSVC) or -O2 (gcc) rather than /Ox (MSVC) or -O3 (gcc).
# We want to keep the max. optimization flags but turn on debugging info so we get better
# symbols / stacktraces in case there's a crash. Do this by adding the compiler-specific flag
# for this regardless of build configuration.
if (MSVC)
  target_compile_options(riptide_cpp PUBLIC /Zi)
  target_link_options(riptide_cpp PUBLIC /DEBUG)
else() # MSVC
  # Assuming gcc/clang here
  target_compile_options(riptide_cpp PUBLIC -g)
endif()

if(WIN32)
    set(RUNTIME_SPECIFIER RUNTIME)
elseif(UNIX)
    set(RUNTIME_SPECIFIER LIBRARY)
else()
    message(FATAL_ERROR "Cannot specify runtime component")
endif()

install(TARGETS ${TARGET_NAME} ${RUNTIME_SPECIFIER} DESTINATION .)
