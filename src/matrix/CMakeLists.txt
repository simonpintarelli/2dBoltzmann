include_directories(${PROJECT_SOURCE_DIR}/src)

FILE(GLOB_RECURSE sources  "*.cpp")
add_library(matrix_assembly ${sources})
INSTALL(TARGETS matrix_assembly ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
#DEAL_II_SETUP_TARGET(matrix_assembly)
