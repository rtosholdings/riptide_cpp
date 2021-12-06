set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME riptide_test)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES
    main.cpp
    test_one_input.cpp)

add_executable(${TARGET_NAME}
               ${HEADERS}
			   ${SOURCES})

get_target_property(RT_SOURCE_DIR riptide_cpp SOURCE_DIR)

target_include_directories(${TARGET_NAME} PRIVATE
                             ${RT_SOURCE_DIR}
							 ${Python3_INCLUDE_DIRS}
							 ${Python3_NumPy_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME}
                      riptide_cpp)
