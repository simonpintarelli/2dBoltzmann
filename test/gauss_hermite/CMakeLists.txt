# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)

##########################################################################################

set(EXE test_gauss_hermite)
set(SOURCES main.cpp )
add_executable(${EXE} ${SOURCES})

add_dependencies(${EXE} quad matrix_assembly collision)


set_target_properties(
  ${EXE}
  PROPERTIES
  COMPILE_FLAGS
  "-DTIMING")

DEAL_II_SETUP_TARGET( ${EXE})
target_link_libraries(${EXE} "${quad_location};${SYSTEM_LIBRARIES}")
