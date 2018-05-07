# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)



set(EXE test_p2h_shift_h2p)

set(SOURCES main.cpp)

add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})

# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------

set(EXE test_p2h_shift_h2p_extended_precision)
set(SOURCES main.cpp)

add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
set_target_properties(
  ${EXE}
  PROPERTIES
  COMPILE_FLAGS
  "-DEXTENDED_PRECISION"
  )


# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------

set(EXE timings_p2h)
set(SOURCES main_timings.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})


set(EXE timings_p2h_ep)
set(SOURCES main_timings.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})

set_target_properties(
  ${EXE}
  PROPERTIES
  COMPILE_FLAGS
  "-DEXTENDED_PRECISION"
  )
