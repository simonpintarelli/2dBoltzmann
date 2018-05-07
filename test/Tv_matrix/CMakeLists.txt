include_directories(${PROJECT_SOURCE_DIR}/src)


set(EXE test_Tv_matrix)
set(SOURCES main.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
