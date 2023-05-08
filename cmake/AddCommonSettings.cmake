# The rt_common_settings interface library provides the common set of compiler options to use for all
# targets.

add_library(rt_common_settings INTERFACE)

if(PROJ_COMPILER_FRONTEND STREQUAL "MSVC")
    target_compile_options(rt_common_settings INTERFACE -FC -Zc:__cplusplus)
    target_compile_options(rt_common_settings INTERFACE "$<$<CONFIG:Debug>:-Od>")
    target_compile_options(rt_common_settings INTERFACE "$<$<CONFIG:Release>:-Ox;-Ob2;-Oi;-Ot>")
    target_compile_options(rt_common_settings INTERFACE -Z7) # generate symbols for all configs
    target_link_options(rt_common_settings INTERFACE "$<$<CONFIG:Release>:-DEBUG>") # generate PDB for all configs
    target_compile_options(rt_common_settings INTERFACE -W4 -WX)
    target_compile_options(rt_common_settings INTERFACE -wd4100) # unreferenced formal parameter
    target_compile_options(rt_common_settings INTERFACE -wd4127) # conditional expression is constant
    target_compile_options(rt_common_settings INTERFACE -wd4189) # local variable is initialized but not referenced
    target_compile_options(rt_common_settings INTERFACE -wd4244) # 'argument': conversion from 'int32_t' to 'const unsigned char', possible loss of data
    target_compile_options(rt_common_settings INTERFACE -wd4310) # cast truncates constant value
    target_compile_options(rt_common_settings INTERFACE -wd4324) # structure was padded due to alignment specifier
    target_compile_options(rt_common_settings INTERFACE -wd4456) # FIXME: declaration of '...' hides previous local declaration
    target_compile_options(rt_common_settings INTERFACE -wd4457) # FIXME: declaration of '...' hides function parameter
    target_compile_options(rt_common_settings INTERFACE -wd4458) # FIXME: declaration of '...' hides class member
    target_compile_options(rt_common_settings INTERFACE -wd4459) # FIXME: declaration of '...' hides global declaration
    target_compile_options(rt_common_settings INTERFACE -wd4702) # FIXME: unreachable code

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(rt_common_settings INTERFACE -permissive- -d2FH4- -Zc:strictStrings-)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(rt_common_settings INTERFACE -march=haswell)
        target_compile_options(rt_common_settings INTERFACE -Wno-format) # TODO: Remove this and fix all the printf format mismatches
        target_compile_options(rt_common_settings INTERFACE -Wno-ignored-attributes)
        target_compile_options(rt_common_settings INTERFACE -Wno-unknown-pragmas)
        target_compile_options(rt_common_settings INTERFACE -Wno-unused-function)
        target_compile_options(rt_common_settings INTERFACE -Wno-unused-parameter)
        target_compile_options(rt_common_settings INTERFACE -Wno-unused-variable)
        target_compile_options(rt_common_settings INTERFACE -Wno-unused-value)
        target_compile_options(rt_common_settings INTERFACE -Wno-duplicated-branches)
    endif()
elseif(PROJ_COMPILER_FRONTEND STREQUAL "GNU")
    target_compile_options(rt_common_settings INTERFACE "$<$<CONFIG:Debug>:-O0>")
    target_compile_options(rt_common_settings INTERFACE "$<$<CONFIG:Release>:-O2>")
    target_compile_options(rt_common_settings INTERFACE -ggdb -x c++ -mavx2 -mbmi2 -fpermissive -pthread)
    target_compile_options(rt_common_settings INTERFACE -falign-functions=32 -fno-strict-aliasing)
    target_compile_options(rt_common_settings INTERFACE -Wall -Werror)
    target_compile_options(rt_common_settings INTERFACE -Wno-error=cast-qual)
    target_compile_options(rt_common_settings INTERFACE -Wno-error=double-promotion)
    target_compile_options(rt_common_settings INTERFACE -Wno-error=duplicated-branches)
    target_compile_options(rt_common_settings INTERFACE -Wno-error=old-style-cast)
    target_compile_options(rt_common_settings INTERFACE -Wno-error=redundant-decls)
    target_compile_options(rt_common_settings INTERFACE -Wno-error=unused-but-set-parameter)
    target_compile_options(rt_common_settings INTERFACE -Wno-error=useless-cast)
    target_compile_options(rt_common_settings INTERFACE -Wno-error=zero-as-null-pointer-constant)
    target_compile_options(rt_common_settings INTERFACE -Wno-format) # TODO: Remove this and fix all the printf format mismatches
    target_compile_options(rt_common_settings INTERFACE -Wno-ignored-attributes)
    target_compile_options(rt_common_settings INTERFACE -Wno-unknown-pragmas)
    target_compile_options(rt_common_settings INTERFACE -Wno-unused-parameter)
    target_compile_options(rt_common_settings INTERFACE -Wno-unused-variable)
    target_compile_options(rt_common_settings INTERFACE -Wno-unused-value)
    target_compile_options(rt_common_settings INTERFACE -Wno-duplicated-branches)
    target_compile_options(rt_common_settings INTERFACE -Wno-useless-cast)
    target_compile_options(rt_common_settings INTERFACE -Wno-old-style-cast)
elseif(PROJ_COMPILER_FRONTEND STREQUAL "LLVM")
    target_compile_options(rt_common_settings INTERFACE "$<$<CONFIG:Debug>:-O0>")
    target_compile_options(rt_common_settings INTERFACE "$<$<CONFIG:Release>:-O2>")
    target_compile_options(rt_common_settings INTERFACE -ggdb -x c++ -mavx2 -mbmi2 -fpermissive -pthread)
    target_compile_options(rt_common_settings INTERFACE -falign-functions=32 -fno-strict-aliasing)
    target_compile_options(rt_common_settings INTERFACE -Wall -Werror)
    target_compile_options(rt_common_settings INTERFACE -Wno-format) # TODO: Remove this and fix all the printf format mismatches
    target_compile_options(rt_common_settings INTERFACE -Wno-ignored-attributes)
    target_compile_options(rt_common_settings INTERFACE -Wno-unknown-pragmas)
    target_compile_options(rt_common_settings INTERFACE -Wno-unused-parameter)
    target_compile_options(rt_common_settings INTERFACE -Wno-unused-variable)
    target_compile_options(rt_common_settings INTERFACE -Wno-unused-value)
    target_compile_options(rt_common_settings INTERFACE -Wno-duplicated-branches)
    target_compile_options(rt_common_settings INTERFACE -Wno-useless-cast)
    target_compile_options(rt_common_settings INTERFACE -Wno-old-style-cast)
else()
    message(FATAL_ERROR "Unexpected proj compiler front-end, ${PROJ_COMPILER_FRONTEND}")
endif()
