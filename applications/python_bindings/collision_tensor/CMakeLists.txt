include_directories(${PROJECT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(BEFORE SYSTEM "${PYTHON_INCLUDE_DIRS}")

if(BOOST_NPY_FOUND)
   if(NOT _OFFICIAL_BOOST_NUMPY_FOUND)
    add_compile_options(-DUSE_DEPRECATED_BOOST_NPY)
  endif()
  include_directories(BEFORE SYSTEM "${BOOST_NPY_INCLUDE_DIR}")
  set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES}")

  set(sources CTensor.C)
  set(libname CTensor)

  file(GLOB QUAD_SOURCES "${PROJECT_SOURCE_DIR}/src/quadrature/*.cpp")
  set(TENSOR_SOURCES
    "${PROJECT_SOURCE_DIR}/src/collision_tensor/collision_tensor_galerkin.cpp"
    "${PROJECT_SOURCE_DIR}/src/collision_tensor/collision_tensor_galerkin_base.cpp"
    "${PROJECT_SOURCE_DIR}/src/collision_tensor/dense/collision_tensor_dense_base.cpp"
    "${PROJECT_SOURCE_DIR}/src/collision_tensor/dense/collision_tensor_zlastAM_storage.cpp")

  add_library(${libname} SHARED ${sources};${QUAD_SOURCES};${TENSOR_SOURCES})
  SET_TARGET_PROPERTIES(
    ${libname}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DPYTHON -DUSE_MPI -shared -fPIC"
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
  target_link_libraries(${libname} "${SYSTEM_LIBRARIES};${PYTHON_LIBRARIES};${Boost_PYTHON_LIB};${BOOST_NPY_LIBRARIES};${HDF5_HL_LIBRARIES};${HDF5_C_LIBRARIES};${MPI_CXX_LIBRARIES};${CBLAS_LIBRARIES}")

  INSTALL (TARGETS ${libname} LIBRARY DESTINATION
    ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/ctensor)

  INSTALL (FILES __init__.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/ctensor)
endif()
