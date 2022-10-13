# These must match expectations in ut/cmake/CPM.cmake
set(CPM_DOWNLOAD_VERSION 0.31.1)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
file(WRITE ${CPM_DOWNLOAD_LOCATION} [=[
# dummy CPM-based packaging stuff for ut...
message("Defining fake CPMAddPackage()...")
function(CPMAddPackage)
    message("NOT doing anything for fake CPMAddPackage()")
endfunction()
message("Defining fake packageProject()...")
function(packageProject)
    message("NOT doing anything for fake packageProject()")
endfunction()
]=])
