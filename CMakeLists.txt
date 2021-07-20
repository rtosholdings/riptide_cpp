cmake_minimum_required(VERSION 3.10)

project(riptide_cpp)

include(CTest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
   #add_compile_options(/W4 /WX /permissive-)
   add_compile_options(/W3 /WX- /permissive-)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_SIMULATE_ID STREQUAL "GNU")
   add_compile_options(-Wall -Wextra -Werror -Wno-unknown-pragmas)
else()
   message(FATAL_ERROR "Unknown compiler")
endif()

enable_testing()

set(SOURCE_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src)
add_subdirectory(external)
add_subdirectory(tests)
