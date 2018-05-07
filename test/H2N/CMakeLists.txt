# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)

set(EXE test_H2N)
set(SOURCES main.cpp)
add_executable(${EXE} ${SOURCES})
#target_link_libraries(${EXE} "${MPI_CXX_LIBRARIES};${MPI_C_LIBRARIES};${HDF5_C_LIBRARIES};${HDF5_HL_LIBRARIES}")
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})

# set(EXE test_H2Ng)
# set(SOURCES main2.cpp)
# add_executable(${EXE} ${SOURCES})
# BOLTZMANN_SETUP_TARGET(${EXE})
