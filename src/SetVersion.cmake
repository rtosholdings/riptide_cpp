set(VERSION_D "${CMAKE_CURRENT_SOURCE_DIR}/_version.d")
set(VERSION_TMP_D "${CMAKE_CURRENT_BINARY_DIR}/_version.tmp.d")
set(GET_VERSION_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/get_version.cmake")

# Helper script to obtain the current version via setuptools_scm and write it out.
# We need to use the setuptools_scm CLI to do this, which loads config from
# pyproject.toml to emit the correctly formatted version.
file(WRITE ${GET_VERSION_SCRIPT} [=[
execute_process(
    COMMAND "${_PYTHON_EXE}" -m setuptools_scm
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE VERSION_STR)
message("Determined version: ${VERSION_STR}")
file(WRITE "${_VERSION_PATH}" \"${VERSION_STR}\")
]=])

# Custom target that always creates the tmp version file.
add_custom_target(set_version_temp
    BYPRODUCTS "${VERSION_TMP_D}"
    COMMAND ${CMAKE_COMMAND} -D_PYTHON_EXE=${Python3_EXECUTABLE} -D_VERSION_PATH=${VERSION_TMP_D} -P "${GET_VERSION_SCRIPT}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    VERBATIM)

# Custom command that depends on the tmp version file and copies it to the
# final version file, iff it's newer, to avoid triggering spurious rebuilds.
add_custom_command(OUTPUT "${VERSION_D}"
    DEPENDS "${VERSION_TMP_D}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VERSION_TMP_D}" "${VERSION_D}"
    VERBATIM)
