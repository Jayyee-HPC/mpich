/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

#define MPI_DATATYPE_HASH_UNDEFINED 0xFFFFFFFF
#define CHAR_BIT 8

typedef struct Type_Sig {
    unsigned long num_types;
    unsigned long hash_value;   /* used in hash generator */
} Type_Sig;

//Example 1001 << 1 -> 0011
unsigned long MPIR_Circular_Left_Shift(unsigned long val, unsigned long nums_left_shift)
{
    unsigned long return_val;
    nums_left_shift = nums_left_shift % (sizeof(val)*CHAR_BIT);

    return_val = (val << nums_left_shift) | 
    		(val >> (sizeof(val)*CHAR_BIT - nums_left_shift));

    return return_val;
}

//(a, n) op (b,m) = (a XOR (B Circular_Left_Shift n), n+m)
Type_Sig MPIR_Datatype_Sig_Hash_Generator(Type_Sig  type_sig_a, Type_Sig type_sig_b)
{
    Type_Sig  type_sig_to_return;
    unsigned long temp_hash_value;

    type_sig_to_return.num_types = type_sig_a.num_types + type_sig_b.num_types;

    temp_hash_value = MPIR_Circular_Left_Shift(type_sig_b.hash_value, type_sig_a.num_types);

    type_sig_to_return.hash_value = type_sig_b.hash_value ^ temp_hash_value;

    return type_sig_to_return;
}

//Generate the hash info for basic MPI_Datatypes
Type_Sig MPIR_Datatype_Init_Hash_Generator(MPI_Datatype type)
{
	Type_Sig temp_hash;
	temp_hash.num_types = 1;

	HASH_VALUE(&type, sizeof(MPI_Datatype), temp_hash.hash_value);
	
	return temp_hash;
}

int MPIR_Coll_len_check_scatter_1(MPI_Aint count, MPI_Datatype datatype, int root, MPIR_Comm * comm_ptr, 
        MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    Type_Sig sig;
    int is_equal;

    sig = MPIR_Datatype_Init_Hash_Generator(datatype);
    sig.hash_value += root;
    sig.num_types = count;

    mpi_errno = MPIR_Allreduce_equal(&sig, 2, MPI_UNSIGNED_LONG, &is_equal, comm_ptr, errflag);

    if(is_equal)
        return mpi_errno;
    else
    {
        *errflag = MPIR_ERR_OTHER;
        MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
            "**collective_size_mismatch",
            "**collective_size_mismatch %d %d", root, comm_ptr->rank);
        return mpi_errno;
    }
}

int MPIR_Coll_len_check_scatter_2(MPI_Aint count, MPI_Datatype datatype, int root, MPIR_Comm * comm_ptr, 
        MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    Type_Sig sig, rsig;

    sig = MPIR_Datatype_Init_Hash_Generator(datatype);
    sig.hash_value += root;
    sig.num_types = count;

    if(root == comm_ptr->rank){
        rsig.hash_value = sig.hash_value;
        rsig.num_types = sig.num_types;
    }

    mpi_errno = MPIR_Bcast_impl(&rsig, 2, MPI_UNSIGNED_LONG, root, comm_ptr, errflag);

    if(root == comm_ptr->rank && (*errflag == MPIR_ERR_NONE))
        return true;
    else
    {
        if((rsig.hash_value == sig.hash_value ||rsig.num_types == sig.num_types) 
                && (*errflag == MPIR_ERR_NONE))
            return true;
        else
        {
            *errflag = MPIR_ERR_OTHER;
            MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
                "**collective_size_mismatch",
                "**collective_size_mismatch %d %d", root, comm_ptr->rank);
            return mpi_errno;
        }  
    }
}

int MPIR_Coll_len_check_scatterv(MPI_Aint sendcount, MPI_Datatype sendtype, MPI_Aint* recvcount, MPI_Datatype recvtype, 
		int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;
    Type_Sig sig, rsig[comm_ptr->size], temp_sig;
    int i, is_equal;

    sig = MPIR_Datatype_Init_Hash_Generator(sendtype);
    sig.hash_value += root;
    sig.num_types = sendcount;

    mpi_errno = MPIR_Gather_impl(&sig, 2, MPI_UNSIGNED_LONG, rsig, 2, MPI_UNSIGNED_LONG, root, comm, errflag);

    if(root == comm_ptr->rank){
    	is_equal = 1;

    	for(i = 0; i < comm_ptr->size; ++i)
    	{
    		temp_sig = MPIR_Datatype_Init_Hash_Generator(recvtype);
    		temp_sig.num_types = recvcount[i];

    		if(temp_sig.hash_value != rsig[i].hash_value || temp_sig.num_types != rsig[i].num_types)
    			is_equal = 0;
    	}
    }

    mpi_errno = MPIR_Bcast_impl(&is_equal, 1, MPI_INT, root, comm_ptr, errflag);

    if(is_equal && (*errflag == MPIR_ERR_NONE))
        return mpi_errno;
    else
    {
        *errflag = MPIR_ERR_OTHER;
        MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
            "**collective_size_mismatch",
            "**collective_size_mismatch %d %d", root, comm_ptr->rank);
        return mpi_errno;
    }
}

int MPIR_Coll_len_check_allgather(MPI_Aint sendcount, MPI_Datatype sendtype, MPI_Aint recvcount, MPI_Datatype recvtype, 
		MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
	int mpi_errno = MPI_SUCCESS;
    Type_Sig sendsig, recvsig;
    int i, is_equal;
    unsigned long sendbuf[4];

    sendsig = MPIR_Datatype_Init_Hash_Generator(sendtype);
    sendsig.num_types = sendcount;

    recvsig = MPIR_Datatype_Init_Hash_Generator(recvtype);
    recvsig.num_types = recvcount;

    sendbuf[0] = sendsig.hash_value;
    sendbuf[1] = sendsig.num_types;
    sendbuf[2] = recvsig.hash_value;
    sendbuf[3] = recvsig.num_types;

    mpi_errno = MPIR_Allreduce_equal(sendbuf, 4, MPI_UNSIGNED_LONG, &is_equal, comm_ptr, errflag);

    if(is_equal && (*errflag == MPIR_ERR_NONE))
    {
    	/*
    	Please verify: do recvcount = sendcount * numprocs needed in alltoall and allgather?
    	 */
    	if(sendsig.hash_value == recvsig.hash_value || 
    			(sendsig.num_types * comm_ptr->size) == recvsig.num_types)
    	{
    		*errflag = MPIR_ERR_OTHER;
	        MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
	            "**collective_size_mismatch",
	            "**collective_size_mismatch %d", comm_ptr->rank);
	        return mpi_errno;
    	}

        return mpi_errno;
    }
    else
    {
        *errflag = MPIR_ERR_OTHER;
        MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
            "**collective_size_mismatch",
            "**collective_size_mismatch %d", comm_ptr->rank);
        return mpi_errno;
    }
}

int MPIR_Coll_len_check_allgatherv(MPI_Aint sendcount, MPI_Datatype sendtype, MPI_Aint *recvcount, MPI_Datatype recvtype, 
		MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
	struct allgatherv_info{
		MPI_Aint recv_count[comm_ptr->size];
		unsigned long send_hash_value;  
        unsigned long recv_hash_value; 
        unsigned long recv_num_types; 
        int is_send_count_equal;
        int is_equal;
    }sendbuf, recvbuf;

    MPI_Datatype derived_type;

    int s_count = 6, s_blocklengths[6] = {comm_ptr->size, 1, 1, 1, 1, 1};
    MPI_Aint s_displacements[6] = {offsetof(allgatherv_info, recv_count), offsetof(allgatherv_info, send_hash_value), 
        	offsetof(allgatherv_info, recv_hash_value), offsetof(allgatherv_info, recv_num_types), 
        	offsetof(allgatherv_info, is_send_count_equal), offsetof(allgatherv_info, is_equal)};
    MPI_Datatype s_types[6] = {MPI_AINT, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_INT, MPI_INT};

    int mpi_errno = MPI_SUCCESS;
    Type_Sig sendsig, recvsig;
    int i;

    /* set up type */
    mpi_errno = MPI_Type_create_struct(s_count, s_blocklengths, s_displacements, s_types, &derived_type);
    MPI_Type_commit(&derived_type);

    /* construct the struct to compare */
    MPIR_Localcopy(sendbuf.recv_count, comm_ptr->size, MPI_Aint, recvcount, comm_ptr->size, MPI_Aint);

	sendsig = MPIR_Datatype_Init_Hash_Generator(sendtype);
    sendsig.num_types = sendcount;
    recvsig = MPIR_Datatype_Init_Hash_Generator(sendtype);
    recvsig.num_types = 0;

    for(i = 0; i < comm_ptr->size; ++i)
    {
    	recvsig.num_types += recvcount[i];
    }

    sendbuf.send_hash_value = sendsig.hash_value;
    sendbuf.recv_hash_value = recvsig.hash_value;
    sendbuf.recv_num_types = recvsig.num_types;

    if(sendsig.hash_value == recvsig.hash_value &&
    		sendsig.num_types == recvcount[comm_ptr->rank])
    	sendbuf.is_send_count_equal = 1;
    else
    	sendbuf.is_send_count_equal = 0;

    sendbuf.is_equal = 1;

    mpi_errno = MPIR_Allreduce_impl(&sendbuf, &recvbuf, 1, derived_type, MPIX_EQUAL, comm_ptr, errflag);

    MPI_Type_free(&derived_type);

    if(recvbuf.is_equal && (*errflag == MPIR_ERR_NONE))
    {
    	/*
    	Please verify: do recvcount = sendcount * numprocs needed in alltoall and allgather?
    	 */
    	if(!sendbuf.is_send_count_equal)
    	{
    		*errflag = MPIR_ERR_OTHER;
	        MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
	            "**collective_size_mismatch",
	            "**collective_size_mismatch %d", comm_ptr->rank);
	        return mpi_errno;
    	}

        return mpi_errno;
    }
    else
    {
        *errflag = MPIR_ERR_OTHER;
        MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
            "**collective_size_mismatch",
            "**collective_size_mismatch %d", comm_ptr->rank);
        return mpi_errno;
    }
}


int MPIR_Coll_len_check_alltoallv(MPI_Aint sendcount, MPI_Datatype sendtype, MPI_Aint *recvcount, MPI_Datatype recvtype, 
		MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{

	return mpi_errno;
}

int MPIR_Coll_len_check_alltoallw(MPI_Aint sendcount, MPI_Datatype sendtype, MPI_Aint *recvcount, MPI_Datatype recvtype, 
		MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{

	return mpi_errno;
}
