set(TARGET_NAME riptide_bench)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES main.cpp
    bench_one_input.cpp
    hash_linear_bench.cpp
    memcmp_bench.cpp)

add_executable(${TARGET_NAME} ${HEADERS} ${SOURCES})

target_compile_definitions(${TARGET_NAME} PRIVATE BENCHMARK_STATIC_DEFINE)

target_include_directories(${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:riptide_cpp,SOURCE_DIR> ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

target_link_directories(${TARGET_NAME} PRIVATE
    ${Python3_LIBRARY_DIRS}
)

target_link_libraries(${TARGET_NAME} PRIVATE
    rt_common_settings
    riptide_cpp
    TBB::tbb
    ${Python3_Libraries}
    benchmark::benchmark
    $<$<PLATFORM_ID:Linux>:pthread>
    $<$<PLATFORM_ID:Linux>:rt>)

if(WIN32)
    set(_TARGET_DIR $<TARGET_FILE_DIR:${TARGET_NAME}>)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMENT "Copying runtime dependencies from riptide_cpp to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:riptide_cpp> ${_TARGET_DIR}

        COMMENT "Copying Python dependencies to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Python3_RUNTIME_LIBRARY_DIRS}/python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll ${_TARGET_DIR}

        COMMENT "Copying TBB dependencies to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:TBB::tbb> ${_TARGET_DIR}

        COMMENT "Copying TBBmalloc dependencies to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:TBB::tbbmalloc> ${_TARGET_DIR}

        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:zstd::libzstd_shared> ${_TARGET_DIR}
    )
endif()
