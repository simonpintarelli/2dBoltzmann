# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)

# Kitzler-Schoeberl Basis
set(SOURCES main.cpp )
set(EXE test_export_hermite_shift_matrix)
add_executable(${EXE} ${SOURCES})
target_link_libraries(${EXE} ${MPI_CXX_LIBRARIES};${MPI_C_LIBRARIES};${HDF5_C_LIBRARIES};${HDF5_HL_LIBRARIES})
BOLTZMANN_SETUP_TARGET(${EXE})
