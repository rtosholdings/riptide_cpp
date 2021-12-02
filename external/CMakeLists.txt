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

# In order to retain current source using "include zstd/zstd.h" point include dir at shim dir which supports that.
# TODO: consider switching to "include zstd.h" and using proper public include variables.
set(EXTERNAL_ZSTD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rt-shims PARENT_SCOPE)