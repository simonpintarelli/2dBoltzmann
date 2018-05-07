include_directories(${PROJECT_SOURCE_DIR}/src)

set(EXE test_rotate_basis)
set(SOURCES main.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_TARGET(${EXE})