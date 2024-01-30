set(TARGET_NAME riptide_python_test)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES
    basic_math_tests.cpp
    bins_bsearch_tests.cpp
    bins_sorted_tests.cpp
    combine_accum_tests.cpp
    ema_test.cpp
    groupby_tests.cpp
    hash_linear_tests.cpp
    lexsort_tests.cpp
    main.cpp
    mkalign_tests.cpp
    mbget_tests.cpp
    tests.cpp
    reduce_tests.cpp
    utility_functions.cpp)

add_executable(${TARGET_NAME}
    ${HEADERS}
    ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    $<TARGET_PROPERTY:riptide_cpp,SOURCE_DIR>
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS})

target_link_directories(${TARGET_NAME} PRIVATE
    ${Python3_LIBRARY_DIRS})

target_link_libraries(${TARGET_NAME}
    rt_common_settings
    riptide_cpp
    test_utility
    ${Python3_LIBRARIES})

proj_copy_runtime_deps(TARGET ${TARGET_NAME})

target_link_libraries(${TARGET_NAME}
    Python3::Python
    riptide_cpp
    ut)

gtest_discover_tests(${TARGET_NAME}
    COMMAND ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DISCOVERY_TIMEOUT 10
    PROPERTIES ENVIRONMENT "PYTHONHOME=${Python3_LIBRARY_DIRS}/..")