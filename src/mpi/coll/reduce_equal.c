/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

int MPIR_Reduce_equal(const void *sendbuf, MPI_Aint count, MPI_Datatype datatype, int root,
	int *is_equal, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
/*
	IMPORTANT:
	char, int, and long are verified to be supported.
	for any other types, if you beleive memcmp() gives correct comparision, 
	you may call this func at own risk.
 */
    MPI_Op op = MPIX_EQUAL;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint type_size, is_equal_pos, lb, extent;
    void *local_sendbuf,*local_recvbuf;

    /*Check if only one short, int, or long*/
    if(count == 1)
    {
        if(datatype == MPI_SHORT)
        {
            struct{
                short val;
                int is_equal;
            }send_struct, recv_struct;
            send_struct.val = *(short*)sendbuf;
            send_struct.is_equal = 1;
            mpi_errno = MPIR_Reduce_impl(&send_struct, &recv_struct, 1, MPI_SHORT_INT, MPIX_EQUAL, root, comm_ptr, errflag);

            *is_equal = recv_struct.is_equal;
            return mpi_errno;
        }
        else if(datatype == MPI_INT)
        {
            struct{
                int val;
                int is_equal;
            }send_struct, recv_struct;
            send_struct.val = *(int*)sendbuf;
            send_struct.is_equal = 1;
            mpi_errno = MPIR_Reduce_impl(&send_struct, &recv_struct, 1, MPI_2INT, MPIX_EQUAL, root, comm_ptr, errflag);

            *is_equal = recv_struct.is_equal;
            return mpi_errno;

        }
        else if(datatype == MPI_LONG)
        {
            struct{
                long val;
                int is_equal;
            }send_struct, recv_struct;
            send_struct.val = *(long*)sendbuf;
            send_struct.is_equal = 1;
            mpi_errno = MPIR_Reduce_impl(&send_struct, &recv_struct, 1, MPI_LONG_INT, MPIX_EQUAL, root, comm_ptr, errflag);

            *is_equal = recv_struct.is_equal;
            return mpi_errno;
        }
        else
        {
            /* continue */
        }
    }

    mpi_errno = MPIR_Type_get_extent_impl(datatype, &lb, &extent);
    if(mpi_errno != MPI_SUCCESS || lb != 0)//Not able to handle structs not starting at 0
    	return MPI_ERR_TYPE;

    type_size = extent - lb;

    /* Dealing simple c stuct alignment and padding. */
    if ((type_size % sizeof(int)) == 0) 
    {
    	is_equal_pos = type_size * count;
	} 
	else 
	{
    	is_equal_pos = type_size * count + (sizeof(int) - type_size % sizeof(int));
	}

    int s_count = 2;
    int s_blocklengths[2] = {count, 1};
    MPI_Datatype s_types[2] = {datatype, MPI_INT};
    MPI_Datatype derived_type;  

    MPI_Aint s_displacements[2] = {0, is_equal_pos};

    /* set up type */
    mpi_errno = MPI_Type_create_struct(s_count, s_blocklengths, s_displacements, s_types, &derived_type);
    MPI_Type_commit(&derived_type);

    mpi_errno = MPIR_Type_get_extent_impl(derived_type, &lb, &extent);

    local_sendbuf = MPL_malloc(extent-lb, MPL_MEM_OTHER);
    local_recvbuf = MPL_malloc(extent-lb, MPL_MEM_OTHER);

    MPIR_Localcopy(sendbuf, count, datatype, local_sendbuf, count, datatype);

    *(int*)(local_sendbuf + is_equal_pos) = 1;

    mpi_errno = MPIR_Reduce_impl(local_sendbuf, local_recvbuf, 1, derived_type, MPIX_EQUAL, root, comm_ptr, errflag);

    *is_equal = *(int*)(local_recvbuf + is_equal_pos);

    MPL_free(local_recvbuf);
    MPL_free(local_recvbuf);
    MPI_Type_free(&derived_type);
    return mpi_errno;
}

int MPIR_Allreduce_equal(const void *sendbuf, MPI_Aint count, MPI_Datatype datatype, int *is_equal,
        MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
/*
	IMPORTANT:
	char, int, and long are verified to be supported.
	for any other types, if you beleive memcmp() gives correct comparision, 
	you may call this func at own risk.
 */
    MPI_Op op = MPIX_EQUAL;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint type_size, is_equal_pos, lb, extent;
    void *local_sendbuf,*local_recvbuf;

    /*Check if only one short, int, or long*/
    if(count == 1)
    {
        if(datatype == MPI_SHORT)
        {
            struct{
                short val;
                int is_equal;
            }send_struct, recv_struct;
            send_struct.val = *(short*)sendbuf;
            send_struct.is_equal = 1;
            mpi_errno = MPIR_Allreduce_impl(&send_struct, &recv_struct, 1, MPI_SHORT_INT, MPIX_EQUAL, comm_ptr, errflag);

            *is_equal = recv_struct.is_equal;
            return mpi_errno;
        }
        else if(datatype == MPI_INT)
        {
            struct{
                int val;
                int is_equal;
            }send_struct, recv_struct;
            send_struct.val = *(int*)sendbuf;
            send_struct.is_equal = 1;
            mpi_errno = MPIR_Allreduce_impl(&send_struct, &recv_struct, 1, MPI_2INT, MPIX_EQUAL, comm_ptr, errflag);

            *is_equal = recv_struct.is_equal;
            return mpi_errno;

        }
        else if(datatype == MPI_LONG)
        {
            struct{
                long val;
                int is_equal;
            }send_struct, recv_struct;
            send_struct.val = *(long*)sendbuf;
            send_struct.is_equal = 1;
            mpi_errno = MPIR_Allreduce_impl(&send_struct, &recv_struct, 1, MPI_LONG_INT, MPIX_EQUAL, comm_ptr, errflag);

            *is_equal = recv_struct.is_equal;
            return mpi_errno;
        }
        else
        {
            /* continue */
        }
    }

    mpi_errno = MPIR_Type_get_extent_impl(datatype, &lb, &extent);
    if(mpi_errno != MPI_SUCCESS || lb != 0)//Not able to handle structs not starting at 0
    	return MPI_ERR_TYPE;

    type_size = extent - lb;

    /* Dealing simple c stuct alignment and padding. */
    if ((type_size % sizeof(int)) == 0) 
    {
    	is_equal_pos = type_size * count;
	} 
	else 
	{
    	is_equal_pos = type_size * count + (sizeof(int) - type_size % sizeof(int));
	}

    int s_count = 2;
    int s_blocklengths[2] = {count, 1};
    MPI_Datatype s_types[2] = {datatype, MPI_INT};
    MPI_Datatype derived_type;  

    MPI_Aint s_displacements[2] = {0, is_equal_pos};

    /* set up type */
    mpi_errno = MPI_Type_create_struct(s_count, s_blocklengths, s_displacements, s_types, &derived_type);
    MPI_Type_commit(&derived_type);

    mpi_errno = MPIR_Type_get_extent_impl(derived_type, &lb, &extent);

    local_sendbuf = MPL_malloc(extent-lb, MPL_MEM_OTHER);
    local_recvbuf = MPL_malloc(extent-lb, MPL_MEM_OTHER);

    MPIR_Localcopy(sendbuf, count, datatype, local_sendbuf, count, datatype);

    *(int*)(local_sendbuf + is_equal_pos) = 1;

    mpi_errno = MPIR_Allreduce_impl(local_sendbuf, local_recvbuf, 1, derived_type, MPIX_EQUAL, comm_ptr, errflag);

    *is_equal = *(int*)(local_recvbuf + is_equal_pos);

    MPL_free(local_sendbuf);
    MPL_free(local_recvbuf);
    MPI_Type_free(&derived_type);
    return mpi_errno;
}

