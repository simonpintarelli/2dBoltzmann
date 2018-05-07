include_directories(${PROJECT_SOURCE_DIR}/src)

if(USE_MPI AND MPI_CXX_FOUND)
  add_definitions(-DUSE_MPI)
  set(EXE mls_otf_bc)
  set(SOURCES main_otf.cpp)
  add_executable(${EXE} ${SOURCES})
  BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
  INSTALL(
    TARGETS ${EXE} RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

  set(EXE ls_otf_semi)
  set(SOURCES main_otf_semi.cpp)
  add_executable(${EXE} ${SOURCES})
  BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
  INSTALL(
    TARGETS ${EXE} RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

  set(EXE ls_periodic)
  set(SOURCES main_periodic.cpp)
  add_executable(${EXE} ${SOURCES})
  BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
  INSTALL(
    TARGETS ${EXE} RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

  set(EXE mls_periodic)
  set(SOURCES main_periodic_x.cpp)
  add_executable(${EXE} ${SOURCES})
  BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
  INSTALL(
    TARGETS ${EXE} RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()
