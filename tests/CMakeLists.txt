set(TARGET_NAME riptide_cpp_tests)

set(SOURCES
    main.cpp
    test_sample.cpp)

add_executable(${TARGET_NAME}
    ${HEADERS}
    ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    ${SOURCE_LIBS_DIR}
    ${Python3_INCLUDE_DIRS})

target_link_directories(${TARGET_NAME} PRIVATE
    ${Python3_LIBRARY_DIRS})

target_link_libraries(${TARGET_NAME}
    #riptide_cpp
    #python38 #${Python3_LIBRARIES}
    gtest)

add_test(NAME test_riptide_cpp
    COMMAND ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
