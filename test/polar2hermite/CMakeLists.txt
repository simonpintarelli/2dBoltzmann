# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)

set(EXE test_p2h)
set(SOURCES main.cpp)
add_executable(${EXE} ${SOURCES})
target_link_libraries(${EXE} ${MPI_CXX_LIBRARIES};${MPI_C_LIBRARIES};${HDF5_C_LIBRARIES};${HDF5_HL_LIBRARIES})
BOLTZMANN_SETUP_TARGET(${EXE})
