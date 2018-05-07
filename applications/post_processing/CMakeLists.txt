
include_directories(${PROJECT_SOURCE_DIR}/src)

IF(NOT "${HDF5_CXX_LIBRARIES}" STREQUAL "")
  set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${HDF5_CXX_LIBRARIES}")

  set(EXE pp-clean-parallel-io)
  set(SOURCES main-clean-parallel-io.cpp)
  add_executable(${EXE} ${SOURCES})

  BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
  INSTALL(TARGETS ${EXE} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
ENDIF()
