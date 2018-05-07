# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)

# Maxwellian molecules
set(SOURCES main_compute_ctensor.cpp)
set(EXE compute_ctensor)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
set_target_properties(
  ${EXE}
  PROPERTIES
  COMPILE_FLAGS
  "-DTIMING")

# Variable hard spheres
set(SOURCES main_compute_ctensor_vhs.cpp)
set(EXE compute_ctensor_vhs)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})

set_target_properties(
  ${EXE}
  PROPERTIES
  COMPILE_FLAGS
  "-DTIMING")

set(SOURCES main_timestep_galerkin.cpp)
set(EXE timestep_hom_galerkin)

add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
set_target_properties(
  ${EXE}
  PROPERTIES
  COMPILE_FLAGS
  "")


INSTALL(
  TARGETS compute_ctensor compute_ctensor_vhs timestep_hom_galerkin RUNTIME
  DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
