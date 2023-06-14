function(proj_copy_runtime_deps)
   set(ONEVALUEARGS TARGET)
   cmake_parse_arguments(PROJ "" "${ONEVALUEARGS}" "" ${ARGN})

   set(_TARGET_DIR $<TARGET_FILE_DIR:${PROJ_TARGET}>)

   if(WIN32)
        add_custom_command(TARGET ${PROJ_TARGET} POST_BUILD
           COMMENT "Copying runtime dependencies to ${_TARGET_DIR}"
           COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${PROJ_TARGET}> ${_TARGET_DIR}
           COMMAND_EXPAND_LISTS
           VERBATIM)

        set(_COPY_PDBS_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/_copy_pdbs.cmake")

        file(WRITE ${_COPY_PDBS_SCRIPT} [=[
foreach(PROJ_DLL ${PROJ_DLLS})
    cmake_path(REPLACE_EXTENSION PROJ_DLL LAST_ONLY pdb OUTPUT_VARIABLE PROJ_PDB)
    if(EXISTS "${PROJ_PDB}")
        file(COPY "${PROJ_PDB}" DESTINATION "${PROJ_DIR}")
    endif()
endforeach()]=])

        add_custom_command(TARGET ${PROJ_TARGET} POST_BUILD
           COMMENT "Copying debugging dependencies to ${_TARGET_DIR}"
           COMMAND ${CMAKE_COMMAND} "-DPROJ_DLLS='$<TARGET_RUNTIME_DLLS:${PROJ_TARGET}>'" "-DPROJ_DIR=${_TARGET_DIR}" -P ${_COPY_PDBS_SCRIPT}
           VERBATIM)
   endif()
endfunction()