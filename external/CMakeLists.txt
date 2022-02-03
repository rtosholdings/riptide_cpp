# googletest
#
add_subdirectory(googletest)

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
