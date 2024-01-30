set(TARGET_NAME sdsfile_test)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES
    main.cpp
    main_tests.cpp)

add_executable(${TARGET_NAME}
    ${HEADERS}
    ${SOURCES})

#target_include_directories(${TARGET_NAME} PRIVATE)

target_link_libraries(${TARGET_NAME}
    rt_common_settings
    sdsfile
    test_utility
    ut)

proj_copy_runtime_deps(TARGET ${TARGET_NAME})

gtest_discover_tests(${TARGET_NAME}
    COMMAND ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
