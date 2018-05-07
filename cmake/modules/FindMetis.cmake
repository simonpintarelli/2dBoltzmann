## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2014 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------

#
# Try to find the HDF5 library
#
# This module exports
#
#   HDF5_LIBRARIES
#   HDF5_INCLUDE_DIRS
#   HDF5_WITH_MPI
#

SET(METIS_DIR "$ENV{METIS_DIR}")


find_path(METIS_INCLUDE_DIR metis.h
  HINTS ${METIS_DIR} ${METIS_INCLUDE_DIR} /usr/include
  PATH_SUFFIXES metis metis/include include/metis include
  )

find_library(METIS_LIBRARY NAMES metis
  HINTS ${METIS_DIR}
  PATH_SUFFIXES metis/lib lib${LIB_SUFFIX} lib64 lib
  )
