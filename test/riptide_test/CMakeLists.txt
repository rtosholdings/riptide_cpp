set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME riptide_test)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES
    invalid_tests.cpp
    is_member_tests.cpp
    main.cpp
    math_worker_tests.cpp
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
    TBB::tbb
    ut
    absl::base
    absl::flat_hash_map
    $<$<PLATFORM_ID:Linux>:pthread>
    $<$<PLATFORM_ID:Linux>:rt>
)

if(WIN32)
    set(_TARGET_DIR $<TARGET_FILE_DIR:${TARGET_NAME}>)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMENT "Copying runtime dependencies to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:riptide_cpp> ${_TARGET_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Python3_RUNTIME_LIBRARY_DIRS}/python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll ${_TARGET_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:TBB::tbb> ${_TARGET_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:zstd::libzstd_shared> ${_TARGET_DIR}
    )
endif()
