set(TARGET_NAME riptide_cpp_tests)

set(SOURCES
    main.cpp
    riptide_tests.cpp
    sds_tests.cpp)

add_executable(${TARGET_NAME}
    ${HEADERS}
    ${SOURCES})

get_target_property(RT_SOURCE_DIR riptide_cpp SOURCE_DIR)

target_include_directories(${TARGET_NAME} PRIVATE
    ${RT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS})

target_link_directories(${TARGET_NAME} PRIVATE
    ${Python3_LIBRARY_DIRS})

target_link_libraries(${TARGET_NAME}
    riptide_cpp
    ${Python3_LIBRARIES}
    gtest
    $<$<PLATFORM_ID:Linux>:pthread>
    $<$<PLATFORM_ID:Linux>:rt>)

add_test(NAME test_riptide_cpp
    COMMAND ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set_tests_properties(test_riptide_cpp PROPERTIES
    ENVIRONMENT "PYTHONHOME=${Python3_RUNTIME_LIBRARY_DIRS};PYTHONPATH=${Python3_STDLIB}")

if(WIN32)
    set(_TARGET_DIR $<TARGET_FILE_DIR:${TARGET_NAME}>)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMENT "Copying runtime dependencies from riptide_cpp to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:riptide_cpp> ${_TARGET_DIR}

        COMMENT "Copying Python dependencies to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Python3_RUNTIME_LIBRARY_DIRS}/python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll ${_TARGET_DIR})
endif()
