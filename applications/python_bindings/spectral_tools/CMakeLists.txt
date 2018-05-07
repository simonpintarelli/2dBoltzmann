# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(BEFORE SYSTEM "${PYTHON_INCLUDE_DIRS}")

add_compile_options("-DNOHDF5")
if(BOOST_NPY_FOUND)
  if(NOT _OFFICIAL_BOOST_NUMPY_FOUND)
    add_compile_options(-DUSE_DEPRECATED_BOOST_NPY)
  endif()
  include_directories(BEFORE SYSTEM "${BOOST_NPY_INCLUDE_DIR}")
  set(libname SpectralTools)
  set(SOURCES SpectralTools.C)
  file(GLOB QUAD_SOURCES "${PROJECT_SOURCE_DIR}/src/quadrature/*.cpp")
  add_library(${libname} SHARED ${SOURCES} ${QUAD_SOURCES})
  set_target_properties(
    ${libname}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DPYTHON -shared -fPIC "
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    )
  target_link_libraries(${libname} "${SYSTEM_LIBRARIES};${PYTHON_LIBRARIES};${BOOST_NPY_LIBRARIES}")
  INSTALL (TARGETS ${libname}
    LIBRARY DESTINATION  ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/spectral_tools )

  set(libname LaguerreKS)
  set(SOURCES LaguerreKS.C)
  add_library(${libname} SHARED ${SOURCES})
  set_target_properties(
    ${libname}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DPYTHON -shared -fPIC "
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    )
  target_link_libraries(${libname} "${SYSTEM_LIBRARIES};${PYTHON_LIBRARIES};${BOOST_NPY_LIBRARIES}")
  INSTALL (TARGETS ${libname}
    LIBRARY DESTINATION  ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/spectral_tools )


  INSTALL(FILES __init__.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/spectral_tools)

endif()
