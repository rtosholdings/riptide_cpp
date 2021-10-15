set(TARGET_NAME riptide_cpp_tests)

set(SOURCES
    main.cpp
    riptide_tests.cpp
    sds_tests.cpp)

add_executable(${TARGET_NAME}
    ${HEADERS}
    ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    ${ROOT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS})

target_link_directories(${TARGET_NAME} PRIVATE
    ${Python3_LIBRARY_DIRS})

target_link_libraries(${TARGET_NAME}
    riptide_cpp
    ${Python3_LIBRARIES}
    gtest)

add_test(NAME test_riptide_cpp
    COMMAND ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
    set(_TARGET_DIR $<TARGET_FILE_DIR:${TARGET_NAME}>)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMENT "Copying runtime dependencies from riptide_cpp to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:riptide_cpp> ${_TARGET_DIR})

    # Copy the whole Python env to bin dir.
    # TODO: This is silly and should just copy the embedded component.
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMENT "Copying runtime dependencies from ${PYTHON_EXE_DIR} to ${_TARGET_DIR}"
        COMMAND if not exist ${_TARGET_DIR}/python.exe ${CMAKE_COMMAND} -E copy_directory ${PYTHON_EXE_DIR} ${_TARGET_DIR})
endif()
