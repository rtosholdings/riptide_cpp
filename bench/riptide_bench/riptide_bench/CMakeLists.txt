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

proj_copy_runtime_deps(TARGET ${TARGET_NAME})