set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME riptide_python_test)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES
    basic_math_tests.cpp
    ema_test.cpp
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
    absl::base
    absl::flat_hash_map
    ${Python3_LIBRARIES})

add_test(NAME test_riptide_cpp_python
    COMMAND ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set_tests_properties(test_riptide_cpp_python PROPERTIES
    ENVIRONMENT "PYTHONHOME=${Python3_RUNTIME_LIBRARY_DIRS}")

if(WIN32)
    set(_TARGET_DIR $<TARGET_FILE_DIR:${TARGET_NAME}>)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMENT "Copying runtime dependencies from riptide_cpp to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:riptide_cpp> ${_TARGET_DIR}

        COMMENT "Copying Python dependencies to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Python3_RUNTIME_LIBRARY_DIRS}/python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll ${_TARGET_DIR}

        COMMENT "Copying TBB dependencies to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:TBB::tbb> ${_TARGET_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:TBB::tbbmalloc> ${_TARGET_DIR}

        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:zstd::libzstd_shared> ${_TARGET_DIR}
    )
endif()

target_link_libraries(${TARGET_NAME}
    Python3::Python
    riptide_cpp
    ut)
