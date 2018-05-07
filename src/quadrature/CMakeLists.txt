include_directories(${PROJECT_SOURCE_DIR}/src)

FILE(GLOB sources "*.cpp")
add_library(quad ${sources})
INSTALL(TARGETS quad ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
#BOLTZMANN_SETUP_TARGET(quad)
