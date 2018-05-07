list(APPEND CMAKE_PREFIX_PATH "/usr/lib")
find_package(likwid QUIET)

STRING(TOLOWER "${CMAKE_BUILD_TYPE}" _cmake_build_type)
include_directories(${PROJECT_SOURCE_DIR}/src)

if(USE_MPI)
  add_definitions(-DUSE_MPI)
endif()

if(likwid_FOUND)
  add_definitions(-DLIKWID -DLIKWID_PERFMON)
  include_directories(BEFORE SYSTEM ${LIKWID_INCLUDE})
endif()

set(EXE test_ct_vbcrs)
set(SOURCES main_vbcrs.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})

set(EXE test_ct_timing_sparse)
set(SOURCES main_timing_sparse.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
if(likwid_FOUND)
target_link_libraries(${EXE} ${LIKWID_LIBRARIES})
endif()

set(EXE test_ct_timing_sparse_blocked)
set(SOURCES main_timing_sparse_blocked.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
if(likwid_FOUND)
target_link_libraries(${EXE} ${LIKWID_LIBRARIES})
endif()

INSTALL(
  TARGETS
  test_ct_timing_sparse
  test_ct_timing_sparse_blocked
  RUNTIME
  DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
  )

set(EXE test_ct_dense_timing_zlastAM)
set(SOURCES main_timing_zlastAM.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
if(likwid_FOUND)
target_link_libraries(${EXE} ${LIKWID_LIBRARIES})
endif()

set(EXE test_ct_dense_timing_zlastAMEigen)
set(SOURCES main_timing_zlastAMEigen.cpp)
add_executable(${EXE} ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET(${EXE})
if(likwid_FOUND)
target_link_libraries(${EXE} ${LIKWID_LIBRARIES})
endif()

INSTALL(
  TARGETS
  test_ct_dense_timing_zlastAM
  test_ct_dense_timing_zlastAMEigen
  RUNTIME
  DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
  )
