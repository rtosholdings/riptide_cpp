# boost::ut
# Create an interface library target that refers to the in-source copy of boost::ut.
# It's not yet available via find_package().
add_library(ut INTERFACE)

set(UT_VERSION c3ed7a21a18a4de3fc807451ae86c9f51e706a11)

# Extract the archive to the build location, so we can reference it from there.
file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_SOURCE_DIR}/ut-${UT_VERSION}.zip
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ut-src)

target_include_directories(ut INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/ut-src/ut-${UT_VERSION}/include)

target_compile_definitions(ut INTERFACE BOOST_UT_DISABLE_MODULE)
target_compile_definitions(ut INTERFACE $<$<STREQUAL:${PROJ_COMPILER_FRONTEND},MSVC>:_SILENCE_CXX20_CISO646_REMOVED_WARNING>)
