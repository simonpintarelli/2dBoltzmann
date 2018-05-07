# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)

set(EXE test_hermite_basis)

set(SOURCES main.cpp)

add_executable(${EXE} ${SOURCES})
target_include_directories(
  ${EXE}
  PRIVATE
  ${PROJECT_SRC_DIR}/src/quadrature
  ${PROJECT_SRC_DIR}/src/polar
  )

DEAL_II_SETUP_TARGET(${EXE})
add_dependencies(${EXE} quad matrix_assembly)
target_link_libraries(${EXE} "${quad_location};${matrix_assembly_location};${SYSTEM_LIBRARIES}")
