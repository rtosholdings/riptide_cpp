# Hand-crafted library

set(SOURCES
   src/gtest-all.cc)

add_library(gtest STATIC ${SOURCES})

target_include_directories(gtest PUBLIC include ./)