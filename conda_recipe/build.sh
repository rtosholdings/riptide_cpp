cmake -E make_directory "build"
cmake -E chdir "build" cmake -DBENCHMARK_ENABLE_GTEST_TESTS=off -DCMAKE_BUILD_TYPE=Release -DRIPTIDE_PYTHON_VER="3.9.9" ../
cmake --build "build" --copnfig Release

