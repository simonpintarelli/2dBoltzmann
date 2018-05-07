include_directories(${PROJECT_SOURCE_DIR}/src)

set(EXE test_function_parser)
set(SOURCES main.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
