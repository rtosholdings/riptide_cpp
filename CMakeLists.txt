cmake_minimum_required(VERSION 3.18)

project(riptide_cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(CheckIPOSupported)
include(AddPythonSupport)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMAKE_CXX_COMPILER_ID - actual compiler used
# CMAKE_CXX_SIMULATE_ID - compiler simulated ABI
# CMAKE_CXX_COMPILER_FRONTEND_VARIANT - compiler simulated front-end
message(STATUS "CMake compiler=${CMAKE_CXX_COMPILER_ID}(${CMAKE_CXX_COMPILER_VERSION}), ABI=${CMAKE_CXX_SIMULATE_ID}(${CMAKE_CXX_SIMULATE_VERSION}), front-end=${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")

# PROJ_COMPILER_ID - compiler ABI (real or simulated)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    set(PROJ_COMPILER_ID "MSVC")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_SIMULATE_ID STREQUAL "GNU")
    set(PROJ_COMPILER_ID "GNU")
else()
    message(FATAL_ERROR "Unknown compiler ID")
endif()

# PROJ_COMPILER_FRONTEND - compiler front-end (real or simulated)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(PROJ_COMPILER_FRONTEND "MSVC")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    set(PROJ_COMPILER_FRONTEND "GNU")
else()
    message(FATAL_ERROR "Unknown compiler front-end")
endif()

# PROJ_COMPILER_TOOLSET - compiler toolset (if available)
if(PROJ_COMPILER_ID STREQUAL "MSVC")
    # Using MSVC_TOOLSET_VERSION is not reliable, so test MSVC_VERSION like auto_link.h does.
    if(MSVC_VERSION)
        if(MSVC_VERSION GREATER_EQUAL "1910" AND MSVC_VERSION LESS "1920")
            set(PROJ_COMPILER_TOOLSET "v141")
        elseif((MSVC_VERSION GREATER_EQUAL "1920" AND MSVC_VERSION LESS "1930"))
            set(PROJ_COMPILER_TOOLSET "v142")
        else()
            message(FATAL_ERROR "Unknown MSVC_VERSION, ${MSVC_VERSION}")
        endif()
    else()
        message(FATAL_ERROR "MSVC_VERSION is not defined")
    endif()
elseif(PROJ_COMPILER_ID STREQUAL "GNU")
    # No toolset variants
else()
    message(FATAL_ERROR "Unexpected proj compiler ID, ${PROJ_COMPILER_ID}")
endif()

message(STATUS "Proj compiler=${PROJ_COMPILER_ID}, toolset=${PROJ_COMPILER_TOOLSET}")

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

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(test)