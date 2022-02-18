cmake_minimum_required(VERSION 3.19)

find_package(Git REQUIRED)

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
  message(STATUS "Initializing Git submodules")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY)
else()
  message(STATUS "Not initializing Git submodules: no .git directory found")
endif()