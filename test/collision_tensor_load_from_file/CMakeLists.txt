# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)

set(SOURCES main.cpp )
add_executable(test_tensor_load_from_file ${SOURCES})
BOLTZMANN_SETUP_DEALII_TARGET( test_tensor_load_from_file)

set_target_properties(
  test_tensor_load_from_file
  PROPERTIES
  COMPILE_FLAGS
  "-DTIMING")

target_link_libraries(test_tensor_load_from_file "${matrix_assembly_location};${quad_location};${collision_location};${SYSTEM_LIBRARIES}")
