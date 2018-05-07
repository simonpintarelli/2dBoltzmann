include(FindPackageHandleStandardArgs)


find_path(CBLAS_INCLUDE_DIR
  NAMES
  cblas.h cblas_f77.h
  HINTS ENV CBLAS_INC_DIR
  /usr/include
  PATH_SUFFIXES include
  DOC "The directory containing the CBLAS header files")


find_library(CBLAS_LIBRARIES NAMES cblas openblas
  HINTS ENV CBLAS_LIB_DIR
  /usr/lib
  PATH_SUFFIXES lib
  DOC "Path to the cblas library"
  )

find_package_handle_standard_args(CBLAS "DEFAULT_MSG" CBLAS_LIBRARIES CBLAS_INCLUDE_DIR)
