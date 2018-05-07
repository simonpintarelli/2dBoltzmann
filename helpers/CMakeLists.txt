include_directories(${PROJECT_SOURCE_DIR}/src)

#file(GLOB files RELATIVE ${CMAKE_SOURCE_DIR}/test "main*cpp")
file(GLOB files RELATIVE ${CMAKE_SOURCE_DIR}/helpers "main*cpp")
foreach(file ${files})
  string(REGEX REPLACE "(main_|.cpp)" "" target ${file})
  string(CONCAT target "helper_" ${target})
  add_executable(${target} ${file})
  BOLTZMANN_SETUP_DEALII_TARGET(${target})
  INSTALL(
    TARGETS ${EXE} RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")


endforeach()
