# Enable testing
enable_testing()

# Find GTest
find_package(GTest REQUIRED)
include_directories(/usr/src)

# Add the test executable
add_executable(runTests test_math.cpp)
target_link_libraries(runTests my_library /usr/src/gtest/lib/libgtest.a /usr/src/gtest/lib/libgtest_main.a pthread)

# Include directories for header files
include_directories(/mnt/c/users/moise/desktop/projects/lab44/nuts_c/src)

# Add the test
add_test(NAME runTests COMMAND runTests)
