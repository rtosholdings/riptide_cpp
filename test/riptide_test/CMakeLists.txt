set(TARGET_NAME riptide_test)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES
    buffer_tests.cpp
    invalid_tests.cpp
    is_member_tests.cpp
    logging_test.cpp
    main.cpp
    math_tests.cpp
    math_worker_tests.cpp
    numpy_traits_tests.cpp
    test_one_input.cpp)

add_executable(${TARGET_NAME}
    ${HEADERS}
    ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    $<TARGET_PROPERTY:riptide_cpp,SOURCE_DIR>
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(${TARGET_NAME}
    rt_common_settings
    riptide_cpp
    test_utility
    TBB::tbb
    ut
    $<$<PLATFORM_ID:Linux>:pthread>
    $<$<PLATFORM_ID:Linux>:rt>
)

proj_copy_runtime_deps(TARGET ${TARGET_NAME})

gtest_discover_tests(${TARGET_NAME}
    COMMAND ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DISCOVERY_TIMEOUT 10)
