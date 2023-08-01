set(TARGET_NAME riptide_python_test)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES
    basic_math_tests.cpp
    ema_test.cpp
    groupby_tests.cpp
    hash_linear_tests.cpp
    main.cpp
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

add_test(NAME test_riptide_cpp_python
    COMMAND ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set_tests_properties(test_riptide_cpp_python PROPERTIES
    ENVIRONMENT "PYTHONHOME=${Python3_RUNTIME_LIBRARY_DIRS}")

proj_copy_runtime_deps(TARGET ${TARGET_NAME})

target_link_libraries(${TARGET_NAME}
    Python3::Python
    riptide_cpp
    ut)
