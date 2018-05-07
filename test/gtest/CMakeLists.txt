include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${GTEST_INCLUDE_DIRS})

FILE(GLOB_RECURSE sources "gtest_*.cpp")

add_executable(gtest_run main.cpp ${sources})
BOLTZMANN_SETUP_DEALII_TARGET(gtest_run)
target_link_libraries(gtest_run GTest::GTest GTest::Main ${SYSTEM_LIBRARIES};${GTEST_BOTH_LIBRARIES};${HDF5_C_LIBRARIES};${HDF5_HL_LIBRARIES})
add_test(AllTests gtest_run)
