include_directories(${PROJECT_SOURCE_DIR}/src)

include_directories(BEFORE SYSTEM "${PYTHON_INCLUDE_DIRS}")
if(BOOST_NPY_FOUND)
   if(NOT _OFFICIAL_BOOST_NUMPY_FOUND)
    add_compile_options(-DUSE_DEPRECATED_BOOST_NPY)
  endif()
  include_directories(BEFORE SYSTEM "${BOOST_NPY_INCLUDE_DIR}")
  set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES}")

  set(sources bquad.C)
  set(libname bquad)

  file(GLOB QUAD_SOURCES "${PROJECT_SOURCE_DIR}/src/quadrature/*.cpp")
  add_library(${libname} SHARED ${sources} ${QUAD_SOURCES})
  SET_TARGET_PROPERTIES(
    ${libname}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DPYTHON -shared -fPIC"
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
  target_link_libraries(${libname} "${SYSTEM_LIBRARIES};${PYTHON_LIBRARIES};${Boost_PYTHON_LIB};${BOOST_NPY_LIBRARIES}")

  INSTALL (TARGETS ${libname} LIBRARY DESTINATION
    ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/bquad)

  INSTALL (FILES __init__.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/bquad)
endif()
