FILE(GLOB_RECURSE _src_header_files "(*.hpp|*.h)")
INSTALL(FILES ${_src_header_files} DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
