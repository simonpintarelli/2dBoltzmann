.. image:: https://zenodo.org/badge/126182914.svg
   :target: https://zenodo.org/badge/latestdoi/126182914

.. contents:: Table of contents
    :depth: 2

#####
Usage
#####

Precomputing the collision tensor
*********************************

Usage information:

.. code-block:: bash

  compute_ctensor --help

Example (for Maxwellian molecules):

.. code-block:: bash

  compute_ctensor -r80 -a100 -i130 -K 60

This computes the tensor entries with 80 Gauss points in radial, 100 quad points in
angular direction and use 130 points to evaluate the inner integral (trapezoidal
rule). The output will be written to ``collision_tensor.h5``. For
portability reasons the tensor is stored in sparse COO format.

The tensor assembly is parallelized via OpenMP. Set the environment variable
``OMP_NUM_THREADS`` to the desired value.

Variable hard spheres (VHS) model: Use the ``compute_ctensor_vhs`` executable,
see ``--help`` for options.


Inhomogeneous Boltzmann solver
******************************

A complete example of how to run the code can be found after installation in
``${CMAKE_INSTALL_PREFIX}/example/inhomogeneous``.  The bash-script ``run-all.sh``
contains the complete procedure.



Homogeneous Case
****************

Have a look at ``timestep_hom --help``.

Example:
::

   timestep_hom --ct=/path/to/collision_tensor.h5 -i init.h5 -n 200 --dt=0.001

This makes 200 time-steps using RK4 with step size 0.001. By default it looks for the
coefficients in the dataset ``coeffs`` in the file specified with ``-i``.  The ``-a``
flag disables high frequency components in angular direction, when the corresponding
coefficients are below a certain threshold, recommended for radially symmetric
solutions.

Coefficients of the initial distribution may be generated with the help of the Python
scripts. An example can be found in ``example/homogeneous``. The file
``gen_init.py`` can be used as a template.

Ouput:
``timestep_hom`` writes the solution coefficients for each time-step in a file called
``coefficients.h5``. This file can be used to visualize the data:

.. code-block:: python

   import h5py
   from numpy import *
   from matplotlib.pyplot import *
   from pyboltz.basis import get_basis, KSBasis
   basis = get_basis() # loads basis from spectral_basis.desc
   x = linspace(-3,3, 100)
   X,Y = meshgrid(x,x)
   # evaluation points as complex numbers
   z = X+1j*Y
   with h5py.File('coefficients.h5', 'r') as fh5:
       for i in range(nsteps):
           C = array(fh5['data/%d' % i])
           # evaluate solution at z
           F = basis.evaluatez(C, z)
           figure()
           imshow(F)

The C++ classes related to the homogeneous Boltzmann equation are completely
exposed to Python (via Boost.Python). This includes the basis transformations
(Polar-Laguerre, Hermite, nodal basis), the collision operator and the
quadrature rules.


Yaml Configuration
==================

The config file must be called ``yaml.config`` and reside in the same directory where
the executable is run.

Example config:

.. code-block:: yaml

   Linear Solvers:
   # Definitions
   - &gmres
     type: gmres
     maxiter: 1000
     tol: 1e-8
     restart: 30
     log result: true
     log history: false

     Boundary Conditions:
    # Definitions
    - &inflow1
      type: inflow
      func: zero

    - &inflow2
      type: inflow
      # inflow function type
      func: maxwellian
      # temperature
      T: 1.2
      # inflow velocity
      v: [2, 0]
      # inflow density
      rho: 1.0

    - &diffusive_reflection1
      type: diffusive reflection
      # temperature
      T: 2
      # tangential velocity
      vt: 1

    - &diffusive_reflection_x_dependent
      type: diffusive reflection x
      # function Tx(x,y)
      Tx: "1-0.5*cos(2*pi*x)"
      vt: 0.03

    Mesh:
      type: extern
      file: gmsh.msh

    SpectralBasis:
      deg: 20

    BoundaryDescriptors:
      # map gmsh ids to boundary conditions
      {0: *inflow1, 1: *inflow2, 2: *diffusive_reflection1}

    Scattering:
      file: collision_tensor.h5
      # Knudsen number
      kn: 1.0

    TimeStepping:
      # delta t
      dt: 1e-3
      # total number of time-steps
      N: 1000
      # write solution vector to disk every n-th time-step
      dump: 10
      # export paraview files every n-th time-step
      export_vtk: 5

    Solver: *gmres
    Preconditioner: ilu



Boundary conditions
===================

Currently implemented boundary conditions are:

- Inflow:

.. code-block:: yaml

     type: inflow
     func: zero

     # or

     type: inflow
     func: maxwellian
     # Temperature
     T: 1
     # velocity
     v: [3, 0]
     # density
     rho: 1.4

- Specular reflection:

.. code-block:: yaml

     type: specular reflection

- Diffusive reflection,

  .. code-block:: yaml

      type: diffusive reflection
      v: [1, 0]

  If ``v`` is not specified, ``v=0`` is assumed.
  Alternatively, the tangential velocity ``vt`` can be defined.

  .. code-block:: yaml

      type: diffusive reflection
      vt: 1 # tangential velocity


############
Installation
############

There is a CMake script.

External dependencies
*********************

Currently, the following versions are known to work.

- Deal.II >= 8.5.1
- Trilinos >= 12.12.1
- HDF5 C library with MPI >= 1.8.12.
- Boost >= 1.63.0 (required: numpy support in Boost.Python)
- Python >= 3.6
- Eigen >= 3.3.1
- Cmake >= 2.8
- MPFR >= 3.1.2
- METIS >= 5.1
- yaml-cpp >= 0.6.1
- HDF5 python (h5py): http://www.h5py.org/
- recent gcc compiler

If the above mentioned libraries reside in custom locations the paths must be
passed to cmake using `-DPACKAGENAME_DIR`, see `ccmake .`.

Trilinos
========

Tested with Trilinos 12.12.1, earlier versions should work as well.

Configure:

.. code-block:: bash

  cmake .. \
      -DTrilinos_ENABLE_ALL_PACKAGES:BOOL=ON \
      -DTrilinos_ENABLE_NOX:BOOL=OFF \
      -DTrilinos_ENABLE_OpenMP:BOOL=ON \
      -DTPL_ENABLE_MPI:BOOL=ON \
      -DTrilinos_ENABLE_CXX11:BOOL=ON \
      -DEpetraExt_USING_HDF5=ON \
  -DTrilinos_ENABLE_HDF5:BOOL=ON \
      -DCMAKE_INSTALL_PREFIX:PATH=/usr \
      -DBUILD_SHARED_LIBS:BOOL=ON \
      $EXTRA_ARGS

EpetraExt_USING_HDF5 and must be set to ON.


Deal.II
=======

Versions 8.5 and 8.4 should work.

Configure:

.. code-block:: bash

    cmake -DCMAKE_BUILD_TYPE=Release \
        -DDEAL_II_WITH_MPI=ON \
        -DDEAL_II_WITH_TRILINOS=ON \
        -DDEAL_II_WITH_CXX11=ON \
        -DDEAL_II_WITH_PETSC=OFF \
        -DDEAL_II_WITH_MUMPS=OFF \
        -DDEAL_II_WITH_SLEPC=OFF \
        -DDEAL_II_WITH_NETCDF=OFF \
        -DDEAL_II_WITH_COMPONENT_DOCUMENTATION=OFF \
        -DDEAL_II_WITH_P4EST=OFF \
        -DDEAL_II_WITH_THREADS=ON \
        -DTRILINOS_DIR=/usr \
        -DHDF5_DIR=/path/tohdf5 \


yaml-cpp
========

Application code in ``applications/modified_least_squares`` uses yaml config files.
Sources can be found here: https://github.com/jbeder/yaml-cpp
Environment variable ``YAMLCPP_DIR`` must point to the install directory.

Optional: some post processing python scripts use pyyaml (http://pyyaml.org)

Boost
=====

Make sure that Boost is built with Python 3 and numpy support.


Paraview
========

Simulation results are exported to the HDF5/xdmf format, which is compatible
with Paraview.


Build
=====

.. code-block:: bash

   git clone git@gitlab.math.ethz.ch:simonpi/2dBoltzmann.git
   mkdir build
   cd build
   cmake -DCMAKE_INSTALL_PREFIX=/path/to/installdir ../code

Note that the ``build`` directory should reside outside the source directory.

``make install`` will install binaries to ``${CMAKE_INSTALL_PREFIX}/bin``.
Python modules will be installed to ``${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages``.
In order to use the Python scripts for pre- and postprocessing the environment variable ``PYTHONPATH``
should be set accordingly.

In case that some of the dependencies are installed in non standard paths, one has to
set environment variables pointing to the correct locations (e.g. METIS_DIR,
Trilinos_DIR, etc...).  The name of these variables can be found in the files
``FindPackage.cmake`` under ``cmake-Modules``, or by calling ``ccmake .`` in the
build directory.
