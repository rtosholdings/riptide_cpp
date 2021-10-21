cmake_minimum_required(VERSION 3.18)

project(riptide_cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include(CTest)
include(CheckIPOSupported)
include(AddPythonSupport)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

message("Setup compiler: ${CMAKE_CXX_COMPILER_ID}, simulates: ${CMAKE_CXX_SIMULATE_ID}, variant: ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
   add_compile_options(-d2FH4- /permissive- -FC -Zc:__cplusplus -Zc:strictStrings-)
   add_compile_options("$<$<CONFIG:Debug>:-Od;-Z7>")
   add_compile_options("$<$<CONFIG:Release>:-Ox;-Ob2;-Oi;-Ot>")
   add_compile_options(/W3 /WX)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
   add_compile_options(-x c++ -mavx2 -mbmi2 -fpermissive -pthread)
   add_compile_options(-falign-functions=32 -fno-strict-aliasing)
   #add_compile_options(-falign-loops=32)
   add_compile_options(-Wall -Werror -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-parameter -Wno-ignored-attributes)
   add_compile_options(-Wno-format) # TODO: Remove this and fix all the printf format mismatches
else()
   message(FATAL_ERROR "Unknown compiler")
endif()

enable_testing()

set(ROOT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src)
add_subdirectory(external)
add_subdirectory(tests)
