set(TARGET_NAME riptide_bench)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES main.cpp bench_one_input.cpp)

add_executable(${TARGET_NAME} ${HEADERS} ${SOURCES})

get_target_property(RT_SOURCE_DIR riptide_cpp SOURCE_DIR)

target_include_directories(${TARGET_NAME} PRIVATE ${RT_SOURCE_DIR} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

target_link_directories(${TARGET_NAME} PRIVATE ${Python3_LIBRARY_DIRS})

target_link_libraries(${TARGET_NAME} PRIVATE riptide_cpp ${Python3_Libraries} benchmark::benchmark $<$<PLATFORM_ID:Linux>:pthread> $<$<PLATFORM_ID:Linux>:rt>)
