set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(PROJ_COMPILER_FRONTEND STREQUAL "MSVC")
    add_compile_options(-FC -Zc:__cplusplus)
    add_compile_options("$<$<CONFIG:Debug>:-Od;-Z7>")
    add_compile_options("$<$<CONFIG:Release>:-Ox;-Ob2;-Oi;-Ot>")
    add_compile_options(/W3 /WX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(-permissive- -d2FH4- -Zc:strictStrings- -D_CRT_SECURE_NO_WARNINGS)
		add_link_options(-NoDefaultLib:tbb12_debug -DefaultLib:tbb12)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-Wno-unused-variable -Wno-unused-function)
    endif()
elseif(PROJ_COMPILER_FRONTEND STREQUAL "GNU")
    add_compile_options("$<$<CONFIG:Debug>:-O0>")
    add_compile_options("$<$<CONFIG:Release>:-O2>")
    add_compile_options(-x c++ -mavx2 -mbmi2 -fpermissive -pthread)
    add_compile_options(-falign-functions=32 -fno-strict-aliasing)
    #add_compile_options(-falign-loops=32)
    add_compile_options(-Wall -Werror -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-parameter -Wno-ignored-attributes)
    add_compile_options(-Wno-format -Wno-error=cast-qual -Wno-error=useless-cast -Wno-error=duplicated-branches -Wno-error=old-style-cast -Wno-error=redundant-decls -Wno-error=double-promotion -Wno-error=zero-as-null-pointer-constant -Wno-error=unused-but-set-parameter) # TODO: Remove this and fix all the printf format mismatches
elseif(PROJ_COMPILER_FRONTEND STREQUAL "LLVM")
    add_compile_options("$<$<CONFIG:Debug>:-O0>")
    add_compile_options("$<$<CONFIG:Release>:-O2>")
    add_compile_options(-x c++ -mavx2 -mbmi2 -fpermissive -pthread)
    add_compile_options(-falign-functions=32 -fno-strict-aliasing)
    #add_compile_options(-falign-loops=32)
    add_compile_options(-Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-parameter -Wno-ignored-attributes)
    add_compile_options(-Wno-format) # TODO: Remove this and fix all the printf format mismatches
else()
     message(FATAL_ERROR "Unexpected proj compiler front-end, ${PROJ_COMPILER_FRONTEND}")
endif()

set(TARGET_NAME riptide_test)

set(CMAKE_VERBOSE_MAKEFILE on)

set(SOURCES
	flat_hash_map_tests.cpp
    main.cpp
    test_one_input.cpp)

add_executable(${TARGET_NAME}
               ${HEADERS}
			   ${SOURCES})

get_target_property(RT_SOURCE_DIR riptide_cpp SOURCE_DIR)

target_include_directories(${TARGET_NAME} PRIVATE
                             ${RT_SOURCE_DIR}
							 ${Python3_INCLUDE_DIRS}
							 ${Python3_NumPy_INCLUDE_DIRS}
							 ${ABSL_COMMON_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME}
                      riptide_cpp
					  TBB::tbb
					  ut)

if(WIN32)
    set(_TARGET_DIR $<TARGET_FILE_DIR:${TARGET_NAME}>)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMENT "Copying runtime dependencies to ${_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:riptide_cpp> ${_TARGET_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Python3_RUNTIME_LIBRARY_DIRS}/python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll ${_TARGET_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:TBB::tbb> ${_TARGET_DIR}
		)
endif()
