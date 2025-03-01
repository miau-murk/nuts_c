# Add the library
add_library(my_library math.cpp)

# Add the executable
add_executable(executable main.cpp)
target_link_libraries(executable my_library)

# Include directories for header files
include_directories(/mnt/c/users/moise/desktop/projects/lab44/nuts_c/src)
