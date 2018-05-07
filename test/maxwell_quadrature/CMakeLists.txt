# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)

# ##########################################################################################
set(SOURCES main.cpp )
set(EXE test_maxwell_quad)
add_executable(${EXE} ${SOURCES})

BOLTZMANN_SETUP_TARGET(${EXE})

set_target_properties(
  ${EXE}
  PROPERTIES
  COMPILE_FLAGS
  "-DTIMING")




# ##########################################################################################
# set(SOURCES main2.cpp )
# set(EXE test_maxwell_quad2)
# add_executable(${EXE} ${SOURCES})

# add_dependencies(${EXE} quad matrix_assembly collision)
# DEAL_II_SETUP_TARGET( ${EXE})

# set_target_properties(
#   ${EXE}
#   PROPERTIES
#   COMPILE_FLAGS
#   "-DTIMING")

# target_link_libraries(${EXE} "${SYSTEM_LIBRARIES};${quad_location}")

# set(SOURCE main_timing.cpp)
# set(EXE gperf_qmaxwell)
# add_executable(${EXE} ${SOURCE})

# add_dependencies(${EXE} quad)
# #DEAL_II_SETUP_TARGET( ${EXE})

# set_target_properties(
#   ${EXE}
#   PROPERTIES
#   COMPILE_FLAGS
#   "-DTIMING")

# target_link_libraries(${EXE} "${SYSTEM_LIBRARIES};${quad_location};/usr/lib/libprofiler.so")
