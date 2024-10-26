cmake_minimum_required(VERSION 3.5)
project(GeneticAlgorithmProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

# Source files
set(SOURCES
    main.cpp
    src/GA.cpp
)

# Add executable
add_executable(genetic_algorithm ${SOURCES})

# Link libraries
target_link_libraries(genetic_algorithm Eigen3::Eigen yaml-cpp)

# Enable testing
enable_testing()

# Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Test sources
set(TEST_SOURCES
    tests/GATest.cpp
    src/GA.cpp  # Include source files needed for the tests
)

add_executable(runTests ${TEST_SOURCES})

# Link test executable against GTest and Eigen
target_link_libraries(runTests GTest::gtest_main Eigen3::Eigen pthread)

# Add test to CTest
add_test(NAME GeneticAlgorithmTest COMMAND runTests)
