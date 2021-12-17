set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(PROJ_COMPILER_FRONTEND STREQUAL "MSVC")
    add_compile_options(-FC -Zc:__cplusplus)
    add_compile_options("$<$<CONFIG:Debug>:-Od;-Z7>")
    add_compile_options("$<$<CONFIG:Release>:-Ox;-Ob2;-Oi;-Ot>")
    add_compile_options(/W3 /WX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(-permissive- -d2FH4- -Zc:strictStrings-)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-Wno-unused-variable -Wno-unused-function)
    endif()
elseif(PROJ_COMPILER_FRONTEND STREQUAL "GNU")
    add_compile_options(-x c++ -mavx2 -mbmi2 -fpermissive -pthread)
    add_compile_options(-falign-functions=32 -fno-strict-aliasing)
    #add_compile_options(-falign-loops=32)
    add_compile_options(-Wall -Werror -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-parameter -Wno-ignored-attributes)
    add_compile_options(-Wno-format) # TODO: Remove this and fix all the printf format mismatches
else()
     message(FATAL_ERROR "Unexpected proj compiler front-end, ${PROJ_COMPILER_FRONTEND}")
endif()

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