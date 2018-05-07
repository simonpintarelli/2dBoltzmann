include_directories(${PROJECT_SOURCE_DIR}/src)
#FILE(GLOB_RECURSE sources "*.cpp")

set(sources collision_tensor.cpp collision_tensor_galerkin_base.cpp
collision_tensor_galerkin.cpp collision_tensor_operator.cpp)

set(sources_dense dense/collision_tensor_dense_base.cpp dense/collision_tensor_zlastAM_storage.cpp)

add_library(collision ${sources} ${sources_dense})
if(MPI_CXX_FOUND AND USE_MPI)
  add_definitions(-DUSE_MPI)
endif()
INSTALL(TARGETS collision ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
