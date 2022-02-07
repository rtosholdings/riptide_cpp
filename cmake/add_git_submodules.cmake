cmake_minimum_required(VERSION 3.19)

find_package(Git REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMAND_ERROR_IS_FATAL ANY)
