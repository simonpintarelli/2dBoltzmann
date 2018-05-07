# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)

set(EXE create_spectral_basis)
set(SOURCES main.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_TARGET(${EXE})



set(EXE test_filter_spectral_basis)
set(SOURCES main_filter.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_TARGET(${EXE})


set(EXE test_extract_spectral_basis)
set(SOURCES main_test_extract_spectral_basis.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_TARGET(${EXE})
