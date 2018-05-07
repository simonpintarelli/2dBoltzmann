cmake_minimum_required (VERSION 2.8.8)

project (boltzmann)

cmake_policy(SET CMP0054 NEW)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake/modules/")
set(USE_MPI          ON CACHE BOOL "use mpi")
set(USE_PYTHON       ON CACHE BOOL "use python")
set(CLANG_COMPLETE   NO CACHE BOOL "IDE mode (do not use for compiling)")
set(BUILD_TESTS      NO CACHE BOOL "build test binaries")

add_compile_options("-Wno-unused-parameter")
add_compile_options("-Wno-unused-but-set-variable")
add_compile_options("-Wno-unused-variable")

if(USE_PYTHON)
  find_package(PythonInterp 3.6 REQUIRED)
  find_package(PythonLibs 3.6 REQUIRED)
  message("PYTHON_VERSION_STRING: ${PYTHON_VERSION_STRING}")
  message("PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
endif()


include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(cmake/checks/CMakeLists.txt)

if(HAVE_AVX)
  add_compile_options("-mavx")
  add_compile_options("-DHAVE_AVX")
endif(HAVE_AVX)

if(HAVE_FMA)
  add_compile_options("-mavx2")
  add_compile_options("-mfma")
  add_compile_options("-DHAVE_FMA")
endif(HAVE_FMA)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-sign-compare -Wno-unused-local-typedefs")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-ftree-vectorize -O3 -DNDEBUG")

message("CXX14 support: ${HAVE_CXX14_RETURN_AUTO}")
if(HAVE_CXX14_RETURN_AUTO)
  add_compile_options("-std=c++14")
  add_compile_options("-DUSE_CXX14")
  message("Using c++14 features")
else()
  add_compile_options("-std=c++11")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release."
    FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
IF( NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF()

find_package(deal.II
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" which contains this path."
    )
ENDIF()

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)
find_package(Trilinos REQUIRED
  COMPONENTS
  Epetra
  EpetraExt)
find_package(YamlCpp REQUIRED)
find_package(Eigen3 3.2.4 REQUIRED)
# TODO: find eigen2
find_package(Boost COMPONENTS regex program_options filesystem serialization system REQUIRED)
set(SYSTEM_LIBRARIES "${Boost_LIBRARIES}")

if(USE_PYTHON)
  if(${PYTHON_VERSION_STRING} GREATER 3.0 )
    message(STATUS "Using Python3")
    find_package(Boost COMPONENTS python3)
  else()
    message(STATUS "Using Python2")
    find_package(Boost COMPONENTS python)
  endif()
endif()
set(Boost_PYTHON_LIB ${Boost_LIBRARIES})

# workaround for cmake 3.11.1 (it wants to link against boost_python for v2 and 3)
set(Boost_PYTHON_LIBw "")
foreach(ii ${Boost_PYTHON_LIB})
  string(REGEX MATCHALL ".*libboost_python.so.*" oo ${ii})
  if(oo STREQUAL "")
    set(Boost_PYTHON_LIBw "${Boost_PYTHON_LIBw};${ii}")
  endif()
endforeach(ii)
set(Boost_PYTHON_LIB ${Boost_PYTHON_LIBw})
message("Boost_PYTHON_LIB: ${Boost_PYTHON_LIB}")


find_package(NumPy QUIET)
find_package(MPFR REQUIRED)
find_package(Metis REQUIRED)
if(USE_PYTHON)
  find_package(BoostNpy)
endif()
find_package(OpenMP REQUIRED)
find_package(GTest)



if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Cblas REQUIRED)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
include_directories(BEFORE "${EIGEN3_INCLUDE_DIR}")
include_directories(BEFORE "${EIGEN3_UNS_INCLUDE_DIR}")
include_directories(BEFORE SYSTEM "${HDF5_INCLUDE_DIRS}")
include_directories(BEFORE SYSTEM "${YAMLCPP_INCLUDE_DIR}")
include_directories(BEFORE SYSTEM "${METIS_INCLUDE_DIR}")
include_directories(BEFORE SYSTEM "${EIGEN3_INCLUDE_DIR}")
include_directories(BEFORE SYSTEM "${Boost_INCLUDE_DIR}")
include_directories(BEFORE SYSTEM "${Trilinos_INCLUDE_DIRS}")
include_directories(BEFORE SYSTEM "${DEAL_II_INCLUDE_DIRS}")
include_directories("${MPFR_INCLUDE_DIR}")
include_directories(BEFORE SYSTEM "${CBLAS_INCLUDE_DIR}")
if(Blaze_found)
  include_directories(BEFORE SYSTEM "${BLAZE_INCLUDE_DIR}")
endif()


## Annoying problem on Euler: HDF5_C_LIBRARIES contains HDF5_sz_NOTFOUND
set(HDF5_C_LIBS "${HDF5_C_LIBRARIES}")
set(HDF5_C_LIBRARIES "")
foreach(ii ${HDF5_C_LIBS})
  string(REGEX MATCHALL "NOTFOUND" oo ${ii})
  if(oo STREQUAL "")
    set(HDF5_C_LIBRARIES "${HDF5_C_LIBRARIES};${ii}")
  endif()
endforeach(ii)


set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${MPFR_LIBRARIES};${YAMLCPP_LIBRARY}")


exec_program(
  "git"
  ${CMAKE_CURRENT_SOURCE_DIR}
  ARGS "describe --always"
  OUTPUT_VARIABLE VERSION
  RETURN_VALUE GIT_QUERY_VERSION)
exec_program(
  "git"
  ${CMAKE_CURRENT_SOURCE_DIR}
  ARGS "describe --all"
  OUTPUT_VARIABLE BRANCHNAME
  RETURN_VALUE GIT_QUERY_BRANCH)
exec_program(
  "git"
  ${CMAKE_CURRENT_SOURCE_DIR}
  ARGS "describe --tags"
  OUTPUT_VARIABLE GIT_DESCRIBE_TAGS
  RETURN_VALUE GIT_QUERY_DT)
exec_program(
  "git diff | wc -l"
  ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GITDIFFC
  )
if(GITDIFFC EQUAL 0)
else()
  message("!!!! WARNING: source contains uncommited changes !!!!")
  set(BRANCHNAME "${BRANCHNAME}@@DIRTY@@")
endif()
if((GIT_QUERY_VERSION EQUAL 0) OR (GIT_QUERY_BRANCH EQUAL 0))
  string( REGEX MATCH "-g.*$" VERSION_SHA1 ${VERSION} )
  if(VERSION_SHA1 STREQUAL "")
    set(VERSION_SHA1 "${VERSION}")
  else()
    string( REGEX REPLACE "[-g]" "" VERSION_SHA1 ${VERSION_SHA1} )
  endif()
  message("GIT version id: ${VERSION_SHA1}")
  message("GIT branch name: ${BRANCHNAME}")
  set(GIT_SHA1 "${VERSION_SHA1}" )
  set(GIT_BNAME "${BRANCHNAME}")
  set(GIT_DT "${GIT_DESCRIBE_TAGS}")
else()
  message("!!!! WARNING: No git repo found! !!!!")
endif()
configure_file (
  "${PROJECT_SOURCE_DIR}/bte_config.h.in"
  "${PROJECT_BINARY_DIR}/bte_config.h"
  )
include_directories(BEFORE SYSTEM "${PROJECT_BINARY_DIR}")

MACRO(BOLTZMANN_SETUP_DEALII_TARGET _target)
  add_dependencies(${_target} quad matrix_assembly collision grid export)
  target_link_libraries(${_target}
    "${CBLAS_LIBRARIES};${matrix_assembly_location};${collision_location};${quad_location};${export_location};${grid_location};${SYSTEM_LIBRARIES};${MPI_CXX_LIBRARIES};${Trilinos_LIBRARIES};${HDF5_C_LIBRARIES};${HDF5_HL_LIBRARIES}")
  ## Note: deal.II always passes -O2 to compile flags, also in debug mode ...
  ## workaround: do not call DEAL_II_SETUP_TARGET when compiling in debug mode
  #SET(_build "RELEASE")
  STRING(TOLOWER "${CMAKE_BUILD_TYPE}" _cmake_build_type)
  IF(${_cmake_build_type} MATCHES "debug")
    SET(_build "DEBUG")
  ELSE()
    SET(_build "RELEASE")
  ENDIF()
  set_property(TARGET ${_target} APPEND PROPERTY
    INCLUDE_DIRECTORIES "${DEAL_II_INCLUDE_DIRS}")

  # not setting silly deal.II compile flags
  # ...
  set_property(TARGET ${_target} APPEND_STRING PROPERTY
    LINK_FLAGS " ${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_${_build}}"
    )

  get_property(_type TARGET ${_target} PROPERTY TYPE)
  IF(NOT ${_type} STREQUAL "OBJECT_LIBRARY")
    #target_link_libraries(${_target} ${DEAL_II_TARGET_${_build}})
    # I usually don't build deal.II with debug flags
    target_link_libraries(${_target} ${DEAL_II_LIBRARIES_RELEASE})
  ENDIF()

ENDMACRO()

MACRO(BOLTZMANN_SETUP_TARGET _target)
  add_dependencies(${_target} quad collision export)
  target_link_libraries(${_target}
    "${collision_location};${quad_location};${export_location};${SYSTEM_LIBRARIES}")
ENDMACRO()


##########################
# add target directories #
##########################
add_subdirectory(src)
add_subdirectory(src/matrix)
add_subdirectory(src/quadrature)
add_subdirectory(src/collision_tensor)
add_subdirectory(src/grid)
add_subdirectory(src/export)

set(matrix_assembly_location $<TARGET_FILE:matrix_assembly>)
set(quad_location $<TARGET_FILE:quad>)
set(collision_location $<TARGET_FILE:collision>)
set(grid_location $<TARGET_FILE:grid>)
set(export_location $<TARGET_FILE:export>)

add_subdirectory(doc)
add_subdirectory(applications/convergence_plots)
add_subdirectory(applications/homogeneous)
add_subdirectory(applications/modified_least_squares)
add_subdirectory(applications/post_processing)
add_subdirectory(helpers)

if(USE_PYTHON)
  add_subdirectory(applications/python_bindings/quadrature)
  add_subdirectory(applications/python_bindings/spectral_tools)
  add_subdirectory(applications/python_bindings/collision_tensor)
  add_subdirectory(python/pyboltz)
endif()

add_subdirectory(example)
if(BUILD_TESTS)
  if(GTEST_FOUND)
    add_subdirectory(test/gtest)
  endif()
  add_subdirectory(test/collision_tensor_dense)
  add_subdirectory(test/collision_tensor_load_from_file)
  add_subdirectory(test/function_parser)
  add_subdirectory(test/gauss_hermite)
  add_subdirectory(test/H2N)
  add_subdirectory(test/hermite_basis)
  add_subdirectory(test/maxwell_quadrature)
  add_subdirectory(test/p2h_shift_h2p)
  add_subdirectory(test/P2N)
  add_subdirectory(test/polar2hermite)
  add_subdirectory(test/rotate_polar_basis)
  add_subdirectory(test/shift_hermite_1d)
  add_subdirectory(test/spectral_basis)
  add_subdirectory(test/Tv_matrix)
endif()
