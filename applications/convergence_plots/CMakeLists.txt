
include_directories(${PROJECT_SOURCE_DIR}/src)

# set(EXE pp_convergence)
# set(SOURCES main.cpp)
# add_executable(${EXE} ${SOURCES})
# BOLTZMANN_SETUP_DEALII_TARGET(${EXE})

# `pp_convergence2` does exactly the same as `pp_convergence` except that it does
# compute RELATIVE ERRORS.

set(EXE pp_convergence2)
set(SOURCES main_relative_errors.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
INSTALL(TARGETS ${EXE} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")