##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

# mpi_sources includes only the routines that are MPI function entry points
# The code for the MPI operations (e.g., MPI_SUM) is not included in 
# mpi_sources

mpi_core_sources += \
    src/mpi/coll/neighbor_alltoallv/neighbor_alltoallv.c \
    src/mpi/coll/neighbor_alltoallv/neighbor_alltoallv_init.c \
    src/mpi/coll/neighbor_alltoallv/neighbor_alltoallv_allcomm_nb.c
