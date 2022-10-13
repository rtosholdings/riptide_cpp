# zstd
#
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
# We only use the static library for zstd, so don't bother building the shared lib.
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
# Turn off multithreading support -- we have our own multithreading in riptide.
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
add_subdirectory(zstd/build/cmake)

set(EXTERNAL_ZSTD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zstd/lib PARENT_SCOPE)

# abseil-cpp
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(abseil-cpp)

# boost::ut
# Create an interface library target that refers to the in-source copy of boost::ut.
# It's not yet available via find_package().
add_library(ut INTERFACE)

target_include_directories(ut INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../test/ut/include)

target_compile_definitions(ut INTERFACE BOOST_UT_DISABLE_MODULE)
target_compile_definitions(ut INTERFACE $<$<STREQUAL:${PROJ_COMPILER_FRONTEND},MSVC>:_SILENCE_CXX20_CISO646_REMOVED_WARNING>)
