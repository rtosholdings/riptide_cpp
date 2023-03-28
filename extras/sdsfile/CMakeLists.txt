set(TARGET_NAME sdsfile)

if(WIN32)
    # Convert current timestamp into SemVer: {YYYY}.{MM}{DD}.{HH}{mm}
    # NOTE: this captures the timestamp at cmake generation time, not build time, which is close enough for CI builds.
    string(TIMESTAMP PROJ_SEMVER_MAJOR "%Y")
    string(TIMESTAMP PROJ_SEMVER_MINOR "%m%d")
    string(TIMESTAMP PROJ_SEMVER_PATCH "%H%M")
    string(TIMESTAMP PROJ_SEMVER "${PROJ_SEMVER_MAJOR}.${PROJ_SEMVER_MINOR}.${PROJ_SEMVER_PATCH}")

    set(VER_FILEVERSION "${PROJ_SEMVER_MAJOR},${PROJ_SEMVER_MINOR},${PROJ_SEMVER_PATCH},0")
    set(VER_FILEVERSION_STR "${PROJ_SEMVER_MAJOR}.${PROJ_SEMVER_MINOR}.${PROJ_SEMVER_PATCH}.0\\0")
    set(VER_PRODUCTVERSION  "${PROJ_SEMVER_MAJOR},${PROJ_SEMVER_MINOR},${PROJ_SEMVER_PATCH},0")
    set(VER_PRODUCTVERSION_STR "${PROJ_SEMVER_MAJOR}.${PROJ_SEMVER_MINOR}.${PROJ_SEMVER_PATCH}\\0")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sdsfile.rc.in ${CMAKE_CURRENT_BINARY_DIR}/sdsfile.rc)
endif()

set(HEADERS
    ../../src/MathThreads.h
    ../../src/MathWorker.h
    ../../src/SDSFile.h
    ../../src/SharedMemory.h)

set(SOURCES
    dllmain.cpp
    $<$<PLATFORM_ID:Windows>:${CMAKE_CURRENT_BINARY_DIR}/sdsfile.rc>
    ../../src/MathThreads.cpp
    ../../src/MathWorker.cpp
    ../../src/SDSFile.cpp
    ../../src/SharedMemory.cpp)

add_library(${TARGET_NAME} SHARED
    ${HEADERS}
    ${SOURCES})

target_compile_definitions(${TARGET_NAME} PRIVATE BUILDING_RIPTIDE_CPP)

target_link_libraries(${TARGET_NAME}
    rt_common_settings
    zstd::libzstd_shared
    $<$<PLATFORM_ID:Linux>:rt>)

if(WIN32)
    set(PKG_ID "SDSFile$<$<CONFIG:Debug>:-Debug>")

    find_program(DOTNET_EXE dotnet)

    # This is terribly inefficient as we need to use a dummy magic .csproj, just to pack stuff we already built.
    # Maybe someday this issue will be fixed: https://github.com/NuGet/Home/issues/4254
    add_custom_target(pack-${TARGET_NAME}
        COMMENT "Packing NuGet package for ${TARGET_NAME} version ${PROJ_SEMVER}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/nuget-project.in ${CMAKE_CURRENT_BINARY_DIR}/nuget-project
        COMMAND ${DOTNET_EXE} pack ${CMAKE_CURRENT_BINARY_DIR}/nuget-project/sdsfile.csproj --no-build --no-restore --no-dependencies --output ${CMAKE_BINARY_DIR}/nuget_pkgs_output -p:PKG_ID=${PKG_ID} -p:PKG_VER=${PROJ_SEMVER} -p:PKG_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR} -p:PKG_BINFILE=$<TARGET_FILE:${TARGET_NAME}> -p:PKG_PDBFILE=$<TARGET_PDB_FILE:${TARGET_NAME}>
        VERBATIM)

endif()
