/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

#ifndef MPI_DATATYPE_HASH_UNDEFINED
#define MPI_DATATYPE_HASH_UNDEFINED 0xFFFFFFFF
#endif //ndef MPI_DATATYPE_HASH_UNDEFINED

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif //ndef CHAR_BIT

/* Datatype signature sturct */
typedef struct type_sig {
    uint64_t  num_types;
    uint64_t  hash_value;   
}type_sig;

/* Type_sig linked list node*/
typedef struct type_sig_node {
    MPI_Datatype dtype;
    struct type_sig typesig;
    struct type_sig_node * next;
}type_sig_node;

/* Entry of the list of computed hash values */
/* Only basic types' hash values are stored */
struct type_sig_node coll_dtype_sigs_entry[] =
{
    {MPI_DATATYPE_NULL, {0, MPI_DATATYPE_HASH_UNDEFINED}, NULL }
};

/* Circular left shift */ 
/* Example 1001 << 1 -> 0011 */
static uint64_t  dtype_sig_circular_left_shift(uint64_t  val, uint64_t  nums_left_shift)
{
    uint64_t return_val;
    uint64_t hash_bit = sizeof(val)*CHAR_BIT;

    nums_left_shift = nums_left_shift % hash_bit;

    return_val = (val << nums_left_shift) | 
    		(val >> (hash_bit - nums_left_shift));

    return return_val;
}

/* For debug */
static void  print_datatype_items(MPI_Datatype type,  MPIR_Comm * comm_ptr)
{
    int i, *ints, type_count; 
    int nints, nadds, ntypes, combiner;
    MPI_Aint *aints, *counts;
    MPI_Datatype *types;
    MPIR_Datatype *typeptr;
    MPIR_Datatype_contents *cp;
    char* combiner_name;

    MPIR_Type_get_envelope_impl(type, &nints, &nadds, &ntypes, &combiner);

    if(nints == 0 || nadds == 0 || ntypes == 0)
    {
        printf("Rank %d, 0dt, %d, %d, %d \n", comm_ptr->rank, nints, nadds, ntypes);
        fflush(stdout);
    }

    combiner_name = MPIR_Datatype_combiner_to_string(combiner);

    struct typesig *sig;
    sig = MPIR_type_get_typesig(type);

    printf("\n Rank %d start print %ld datatype, combiner %s\n", comm_ptr->rank, sig->n, combiner_name);
    fflush(stdout);
    for (i = 0; i < sig->n && i < 10; i++) {
        if (i > 0) {
            printf(",");
        }
        printf("%s:%ld", MPIR_Datatype_builtin_to_string(sig->types[i]), (long) sig->counts[i]);
    }
    fflush(stdout);

    for (i = 0; i < sig->n - 1; i++) {
        if(sig->types[i] == sig->types[i+1])
            printf("Same type find at %d, with type %s, count %ld %ld\n", i, MPIR_Datatype_builtin_to_string(sig->types[i]), (long) sig->counts[i], (long) sig->counts[i+1]);
    }
    printf("\n Rank %d end print datatype \n \n", comm_ptr->rank);
    fflush(stdout);

    return;
}

/* Core datatype hash algorithm */
/* (a, n) op (b,m) = (a XOR (B Circular_Left_Shift n), n+m) */
static void dtype_sigs_op(type_sig  type_sig_a, type_sig type_sig_b, type_sig * sig)
{
    uint64_t temp_hash_value;

    sig->num_types = type_sig_a.num_types + type_sig_b.num_types;

    temp_hash_value = dtype_sig_circular_left_shift(type_sig_b.hash_value, type_sig_a.num_types);

    sig->hash_value = type_sig_b.hash_value ^ temp_hash_value;

    return;
}

/* Return a type sig for given MPI_Datatype */
/* Uthash is used */
static void  init_hash_henerator(MPI_Datatype type, type_sig *sig)
{
	sig->num_types = 1ULL;

	HASH_VALUE(&type, sizeof(MPI_Datatype), sig->hash_value);
	
	return;
}

/* Return a type sig for given baisc MPI_Datatype */
/* This type sig is/will stored in a linked list */
static void baisc_dtype_sig_generator(MPI_Datatype type, type_sig *sig)
{
    /* First check if type sig is produced before coll_dtype_sigs_entry*/
    struct type_sig_node *sig_node, *sig_node_pre;
    sig_node = sig_node_pre = coll_dtype_sigs_entry[0].next;

    while(sig_node != NULL)
    {
        if(type == sig_node->dtype)
        {
        	sig->hash_value = sig_node->typesig.hash_value;
    		sig->num_types = sig_node->typesig.num_types;
            return;
        }

        sig_node_pre = sig_node;
        sig_node = sig_node->next;
    }

    /* coll_dtype_sigs_entry is empty*/
    if(sig_node == NULL && sig_node_pre == NULL)
    {
        type_sig temp_sig;
        init_hash_henerator(type, &temp_sig);

        //TODO: does this linked node need to be freed in finalize?
        struct type_sig_node *temp_sig_node = MPL_malloc(sizeof(type_sig_node),MPL_MEM_OTHER);

        temp_sig_node->dtype = type;
        temp_sig_node->typesig = temp_sig;
        temp_sig_node->next = NULL;

        coll_dtype_sigs_entry[0].next = temp_sig_node;

        sig->hash_value = temp_sig.hash_value;
    	sig->num_types = temp_sig.num_types;
        return;
    }

    /* Generate new type_sig*/
    type_sig temp_sig;
    init_hash_henerator(type, &temp_sig);

    //TODO: does this linked node need to be freed in finalize?
    struct type_sig_node *temp_sig_node = MPL_malloc(sizeof(type_sig_node),MPL_MEM_OTHER);

    temp_sig_node->dtype = type;
    temp_sig_node->typesig = temp_sig;
    temp_sig_node->next = NULL;

    /* Push the new type_sig into coll_dtype_sigs_entry */
    sig_node_pre->next = temp_sig_node;

    sig->hash_value = temp_sig.hash_value;
    sig->num_types = temp_sig.num_types;

    return;
}

/* Generate a type sig for a general MPI_Datatype */
/* Return error handle in case MPIR_Datatype_access_contents failed*/
static int dtype_sig_generator(MPI_Datatype type, type_sig *coll_type_sig)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    type_sig temp_sig_to_return, temp_sig;
    struct typesig *sig;

    sig = MPIR_type_get_typesig(type);

    baisc_dtype_sig_generator(sig->types[0], &temp_sig_to_return);
    temp_sig_to_return.num_types = temp_sig_to_return.num_types * sig->counts[0];

    for (i = 1; i < sig->n; i++) {
        baisc_dtype_sig_generator(sig->types[i], &temp_sig);
        temp_sig.num_types = temp_sig.num_types * sig->counts[i];
        dtype_sigs_op(temp_sig_to_return, temp_sig, &temp_sig_to_return);
    }

    coll_type_sig->num_types = temp_sig_to_return.num_types;
    coll_type_sig->hash_value = temp_sig_to_return.hash_value;
    //printf("0x%08x 0x%08x \n", temp_sig_to_return.hash_value, temp_sig_to_return.num_types);
    return mpi_errno;
}

static int dtype_sig_generator_n(MPI_Datatype type, type_sig *coll_type_sig, MPI_Aint count)
{
    int mpi_errno = MPI_SUCCESS;
    int i, j;
    type_sig temp_sig_to_return, temp_sig_0, temp_sig_1;
    struct typesig *sig;

    sig = MPIR_type_get_typesig(type);

    /*Generate type sig for the first basic type*/
    baisc_dtype_sig_generator(sig->types[0], &temp_sig_to_return);
    temp_sig_to_return.num_types = temp_sig_to_return.num_types * sig->counts[0];

    /*Generate type sig for the type*/
    for (i = 1; i < sig->n; i++) {
        temp_sig_0.num_types = temp_sig_to_return.num_types;
        temp_sig_0.hash_value = temp_sig_to_return.hash_value;

        baisc_dtype_sig_generator(sig->types[i], &temp_sig_1);
        temp_sig_1.num_types = temp_sig_1.num_types * sig->counts[i];

        dtype_sigs_op(temp_sig_0, temp_sig_1, &temp_sig_to_return);
    }

    /*Generate type sig for count types*/
    if(MPIR_DATATYPE_IS_PREDEFINED(type) || sig->n == 1)
    {
        /* For types only have one basic type*/
        temp_sig_to_return.num_types = temp_sig_to_return.num_types * count;
    }
    else
    {
        for (i = 1; i < count; ++i)
        {
            for (j = 0; j < sig->n; ++j) 
            {
                /*TODO: for types in ABA format, it may not match types in AB(A2)BA format.
                    A rare case, to be fixed*/
                temp_sig_0.num_types = temp_sig_to_return.num_types;
                temp_sig_0.hash_value = temp_sig_to_return.hash_value;

                baisc_dtype_sig_generator(sig->types[j], &temp_sig_1);
                temp_sig_1.num_types = temp_sig_1.num_types * sig->counts[j];

                dtype_sigs_op(temp_sig_0, temp_sig_1, &temp_sig_to_return);
            }
        } 
    }

    coll_type_sig->num_types = temp_sig_to_return.num_types;
    coll_type_sig->hash_value = temp_sig_to_return.hash_value;

    return mpi_errno;
}


static uint64_t dtype_get_len(MPI_Datatype type, MPI_Aint count)
{
    uint64_t type_len;

    MPIR_Datatype_get_size_macro(type, type_len);
	type_len *= count;

    return type_len;
}

/*
Collective length check for: bcast,reduce,allreduce,reducescatter,scan,exscan,gather,scatter.
Input: count, datatype, root, *comm_ptr, *errflag, *op(NULL if no op used).
Output: error handle
 */
int MPIR_Coll_len_check_scatter(MPI_Aint count, MPI_Datatype datatype, int root, MPI_Op * op, MPIR_Comm * comm_ptr, 
        MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
        /* intracommunicator */
        if (root == MPI_PROC_NULL) 
        {
        /* NULL root process do nothing */
	        mpi_errno = MPI_SUCCESS;
	        return mpi_errno;
	    }

        type_sig sig;
        int is_equal;

        mpi_errno = dtype_sig_generator_n(datatype, &sig, count);
        
        /* Append root info to the hash_value*/
        sig.hash_value += root;

        /* Append op info to the hash_value*/
        if(op != NULL)
            sig.hash_value += (*op);

        mpi_errno = MPIR_Allreduce_equal(&sig, 2, MPI_UINT64_T, &is_equal, comm_ptr, errflag);
       
        if(is_equal && (*errflag == MPIR_ERR_NONE))
            return mpi_errno;
        else
        {
            if(*errflag == MPIR_ERR_NONE){
            	/* Check if MPI_BYTE being used */
				uint64_t type_len;
				int is_byte_type, is_byte_type_recv;

				/* Check if at least one process using MPI_BYTE*/
				is_byte_type = (datatype == MPI_BYTE) ? 1 : 0;
				mpi_errno = MPIR_Allreduce_impl(&is_byte_type, &is_byte_type_recv, 1, MPI_INT, 
						MPI_LOR, comm_ptr, errflag);
				 
				if(is_byte_type_recv)
				{
					/* For MPI_BYTE, only check the numbers of bits */
					type_len = dtype_get_len(datatype, count);  

                    /* Append root info to the type_len*/
                    type_len += root;

                    /* Append op info to the type_len*/
                    if(op != NULL)
                        type_len += (*op);

					mpi_errno = MPIR_Allreduce_equal(&type_len, 1, MPI_UINT64_T, &is_equal, comm_ptr, errflag);

					if(is_equal && (*errflag == MPIR_ERR_NONE))
            			return mpi_errno;
				}
            }

	        *errflag = MPI_ERR_ARG;
	        MPIR_ERR_SET(mpi_errno, MPI_ERR_ARG, "**collective_size_mismatch");
	        return mpi_errno;	
        }
    }
    else
    {
    	type_sig sig;
        int is_equal;

        /* intercommunicator */
        if (root == MPI_PROC_NULL) 
        {
        /* local processes other than root do nothing */
	        return mpi_errno;
	    }

        mpi_errno = dtype_sig_generator_n(datatype, &sig, count);
        
        /* Do not check root info when using inter communicator*/

        /* Append op info to the hash_value*/
        if(op != NULL)
            sig.hash_value += (*op);        
        
        /* Please verify: root mismatch may impact this function, is extra root confirmation needed?*/
        mpi_errno = MPIR_Reduce_equal(&sig, 2, MPI_UINT64_T, root, &is_equal, comm_ptr, errflag);

        mpi_errno = MPIR_Bcast_impl(&is_equal, 1, MPI_INT, root, comm_ptr, errflag);
       
        if(is_equal && (*errflag == MPIR_ERR_NONE))
            return mpi_errno;
        else
        {
            if(*errflag == MPIR_ERR_NONE){
            	/* Check if MPI_BYTE being used */
				uint64_t type_len;
				int is_byte_type, is_byte_type_recv;

				/* Check if at least one process using MPI_BYTE*/
				is_byte_type = (datatype == MPI_BYTE) ? 1 : 0;
				mpi_errno = MPIR_Allreduce_impl(&is_byte_type, &is_byte_type_recv, 1, MPI_INT, 
						MPI_LOR, comm_ptr, errflag);
				 
				if(is_byte_type_recv)
				{
					/* For MPI_BYTE, only check the numbers of bits */
					type_len = dtype_get_len(datatype, count);  

                    /* Do not check root info when using inter communicator*/

                    /* Append op info to the type_len*/
                    if(op != NULL)
                        type_len += (*op);

					mpi_errno = MPIR_Allreduce_equal(&type_len, 1, MPI_UINT64_T, &is_equal, comm_ptr, errflag);

					if(is_equal && (*errflag == MPIR_ERR_NONE))
            			return mpi_errno;
				}
            }

            *errflag = MPI_ERR_ARG;
            MPIR_ERR_SET(mpi_errno, MPI_ERR_ARG, "**collective_size_mismatch");
            return mpi_errno;
        }
    }
}

/*
Collective length check for: gatherv,scatterv.
Input: count, datatype, root, *comm_ptr, *errflag
Output: error handle
 */
int MPIR_Coll_len_check_scatterv(MPI_Aint sendcount, MPI_Datatype sendtype, const MPI_Aint* recvcounts, MPI_Datatype recvtype, 
		int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
        /* intracommunicator */
	    type_sig sig, *rsigs, temp_sig;
        int comm_size;
	    int i, is_equal;
	    uint64_t temp_num_types;
        
        comm_size = comm_ptr->local_size;

        rsigs = (type_sig *)MPL_malloc(sizeof(type_sig) * comm_size, MPL_MEM_OTHER);

        mpi_errno = dtype_sig_generator_n(sendtype, &sig, sendcount);
        
        /* Append root info to the hash_value*/
        sig.hash_value += root;

        /* Please verify: root mismatch may impact this function, is extra root confirmation needed?*/

	    mpi_errno = MPIR_Gather_impl(&sig, 2, MPI_UINT64_T, rsigs, 2, MPI_UINT64_T, root, comm_ptr, errflag);

	    if(root == comm_ptr->rank){
	    	is_equal = 1;

            for(i = 0; i < comm_size; ++i)
            {	 
                mpi_errno = dtype_sig_generator_n(recvtype, &temp_sig, recvcounts[i]); 

                /* Append root info to the hash_value*/
                temp_sig.hash_value += root;

                if(temp_sig.hash_value != rsigs[i].hash_value || temp_sig.num_types != rsigs[i].num_types)
                {
                    is_equal = 0;
                    break;
                }	    			
            }           
	    }

	    mpi_errno = MPIR_Bcast_impl(&is_equal, 1, MPI_INT, root, comm_ptr, errflag);

        if(rsigs != NULL)
            MPL_free(rsigs);

	    if(is_equal && (*errflag == MPIR_ERR_NONE))
	        return mpi_errno;
	    else
	    {
            if(*errflag == MPIR_ERR_NONE){
            	/* Check if MPI_BYTE being used */
				uint64_t type_len, r_type_lens[comm_size], temp_type_len;
				int is_byte_type, is_byte_type_recv;

				/* Check if at least one process using MPI_BYTE*/
				is_byte_type = (sendtype == MPI_BYTE || recvtype == MPI_BYTE) ? 1 : 0;
				mpi_errno = MPIR_Allreduce_impl(&is_byte_type, &is_byte_type_recv, 1, MPI_INT, 
						MPI_LOR, comm_ptr, errflag);
				 
				if(is_byte_type_recv)
				{
					/* For MPI_BYTE, only check the numbers of bits */
					type_len = dtype_get_len(sendtype, sendcount);  

                    /* Append root info to the type length*/
                    type_len += root;

                    mpi_errno = MPIR_Gather_impl(&type_len, 1, MPI_UINT64_T, r_type_lens, 1, MPI_UINT64_T, root, comm_ptr, errflag);

                    if(root == comm_ptr->rank){
                        is_equal = 1;

                        for(i = 0; i < comm_size; ++i)
                        {	        
                            temp_type_len = dtype_get_len(recvtype, recvcounts[i]);

                            if(temp_type_len != r_type_lens[i])
                            {
                                is_equal = 0;
                                break;
                            }	    			
                        }
                    }

                    mpi_errno = MPIR_Bcast_impl(&is_equal, 1, MPI_INT, root, comm_ptr, errflag);

					if(is_equal && (*errflag == MPIR_ERR_NONE))
            			return mpi_errno;
				}
            }

            *errflag = MPI_ERR_ARG;
            MPIR_ERR_SET(mpi_errno, MPI_ERR_ARG, "**collective_size_mismatch");
            return mpi_errno;
	    }

	        }
else
    {
        /* intercommunicator */
        if (root == MPI_PROC_NULL) 
        {
        /* local processes other than root do nothing */
	        mpi_errno = MPI_SUCCESS;
	        return mpi_errno;
	    }

        int comm_size;

        if ((comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) && (root == MPI_ROOT))
        {
            comm_size = comm_ptr->remote_size;

            if(comm_size <= 0)
            {
                /* Remote comm is empty */
	            mpi_errno = MPI_SUCCESS;
	            return mpi_errno;
            }
        }
        else
        {
            comm_size = 0;
        }

	    type_sig sig, temp_sig;
	    int i, is_equal;
	    unsigned long temp_num_types;
        type_sig * rsigs = NULL;
        if (comm_size > 0)
            rsigs = (type_sig *)MPL_malloc(sizeof(type_sig) * comm_size, MPL_MEM_OTHER);

        mpi_errno = dtype_sig_generator_n(sendtype, &sig, sendcount);
        
        /* Do not check root info when using inter communicator*/

	    mpi_errno = MPIR_Gather_impl(&sig, 2, MPI_UINT64_T, rsigs, 2, MPI_UINT64_T, root, comm_ptr, errflag);

	    if(root == MPI_ROOT){
	    	is_equal = 1;
            
	    	for(i = 0; i < comm_size; ++i)
	    	{
	            mpi_errno = dtype_sig_generator_n(recvtype, &temp_sig, recvcounts[i]);

	    		if(temp_sig.hash_value != rsigs[i].hash_value || temp_sig.num_types != rsigs[i].num_types)
	    			is_equal = 0;
	    	}
	    }

	    mpi_errno = MPIR_Bcast_impl(&is_equal, 1, MPI_INT, root, comm_ptr, errflag);

        if(rsigs != NULL)
            MPL_free(rsigs);

        if(is_equal && (*errflag == MPIR_ERR_NONE))
            return mpi_errno;
        else
        {
            if(*errflag == MPIR_ERR_NONE){
            	/* Check if MPI_BYTE being used */
				uint64_t type_len, *r_type_lens, temp_type_len;
				int is_byte_type, is_byte_type_recv;

                r_type_lens = NULL;
                if(comm_size > 0)
                    r_type_lens = (uint64_t *)MPL_malloc(sizeof(uint64_t) * comm_size, MPL_MEM_OTHER);

				/* Check if at least one process using MPI_BYTE*/
				is_byte_type = (sendtype == MPI_BYTE || recvtype == MPI_BYTE) ? 1 : 0;
            
				mpi_errno = MPIR_Allreduce_impl(&is_byte_type, &is_byte_type_recv, 1, MPI_INT, 
						MPI_LOR, comm_ptr, errflag);
				 
				if(is_byte_type_recv)
				{
					/* For MPI_BYTE, only check the numbers of bits */
					type_len = dtype_get_len(sendtype, sendcount);  

                    /* Do not check root info when using inter communicator*/

                    mpi_errno = MPIR_Gather_impl(&type_len, 1, MPI_UINT64_T, r_type_lens, 1, MPI_UINT64_T, root, comm_ptr, errflag);

                    if(root == MPI_ROOT){
                        is_equal = 1;

                        for(i = 0; i < comm_size; ++i)
                        {	        
                            temp_type_len = dtype_get_len(recvtype, recvcounts[i]);

                            if(temp_type_len != r_type_lens[i])
                            {
                                is_equal = 0;
                                break;
                            }	    			
                        }
                    }

                    mpi_errno = MPIR_Bcast_impl(&is_equal, 1, MPI_INT, root, comm_ptr, errflag);

					if(is_equal && (*errflag == MPIR_ERR_NONE))
            			return mpi_errno;
				}

                if(r_type_lens != NULL)
                    MPL_free(r_type_lens);
            }

            *errflag = MPI_ERR_ARG;
            MPIR_ERR_SET(mpi_errno, MPI_ERR_ARG, "**collective_size_mismatch");
            return mpi_errno;
        }
    }
}

/*
Collective length check for: Allgather,Allreduce, Alltoall.
Input: sendcount, sendtype, recvcount, recvtype, *op, *comm_ptr, *errflag
Output: error handle
 */
int MPIR_Coll_len_check_allgather(MPI_Aint sendcount, MPI_Datatype sendtype, MPI_Aint recvcount, MPI_Datatype recvtype, 
		MPI_Op * op, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
	int mpi_errno = MPI_SUCCESS;
    type_sig sendsig, recvsig;
    int i, is_equal;
    uint64_t sig_compare_buf[4];

    mpi_errno = dtype_sig_generator_n(sendtype, &sendsig, sendcount);
    
    /* Append op info to the hash_value*/
    if(op != NULL)
    {
        sendsig.hash_value += (*op);
        recvsig.hash_value += (*op);
    }
        

    mpi_errno = dtype_sig_generator_n(recvtype, &recvsig, recvcount);

    sig_compare_buf[0] = sendsig.hash_value;
    sig_compare_buf[1] = sendsig.num_types;
    sig_compare_buf[2] = recvsig.hash_value;
    sig_compare_buf[3] = recvsig.num_types;

    mpi_errno = MPIR_Allreduce_equal(sig_compare_buf, 4, MPI_UINT64_T, &is_equal, comm_ptr, errflag);

    if(is_equal && (*errflag == MPIR_ERR_NONE))
    {
        /* Check if sendbuf matches recvbuf */
    	if(sendsig.hash_value == recvsig.hash_value && 
    			sendsig.num_types == recvsig.num_types)
        {
            return mpi_errno;
        }       
    }


    /* Check if MPI_BYTE being used */
    uint64_t type_len[2];
    int is_byte_type, is_byte_type_recv;


    /* Check if at least one process using MPI_BYTE*/
    is_byte_type = (sendtype == MPI_BYTE || recvtype == MPI_BYTE) ? 1 : 0;

    mpi_errno = MPIR_Allreduce_impl(&is_byte_type, &is_byte_type_recv, 1, MPI_INT, 
            MPI_LOR, comm_ptr, errflag);
        
    if(is_byte_type_recv)
    {
        /* For MPI_BYTE, only check the numbers of bits */
        type_len[0] = dtype_get_len(sendtype, sendcount);  
        type_len[1] = dtype_get_len(recvtype, recvcount); 

        /* Append op info to the type_len*/
        if(op != NULL)
        {
            type_len[0] += (*op);
            type_len[1] += (*op);
        }
            
        mpi_errno = MPIR_Allreduce_equal(type_len, 2, MPI_UINT64_T, &is_equal, comm_ptr, errflag);

        if(is_equal && (*errflag == MPIR_ERR_NONE) && type_len[0] == type_len[1])
            return mpi_errno;
    }

    *errflag = MPI_ERR_ARG;
    MPIR_ERR_SET(mpi_errno, MPI_ERR_ARG, "**collective_size_mismatch");
    return mpi_errno;

}

//#define OFFSETOF1(_type, _field)    ((unsigned long) &(((_type *) 0)->_field))
/*
Collective length check for: Allgatherv.
Input: sendcount, sendtype, recvcounts, recvtype, *comm_ptr, *errflag
Output: error handle
 */

int MPIR_Coll_len_check_allgatherv(MPI_Aint sendcount, MPI_Datatype sendtype, const MPI_Aint *recvcounts, MPI_Datatype recvtype, 
		MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
	int mpi_errno = MPI_SUCCESS;
    type_sig sendsig, recvsig;
    int i, comm_size;
    uint64_t temp_num_types;
	uint64_t *local_check_arr;
	int is_equal;

	if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
        /* intracommunicator */
        comm_size = MPIR_Comm_size(comm_ptr);
	}
	else
	{
		/* intercommunicator */
		comm_size = comm_ptr->local_size + comm_ptr->remote_size;
	}

	local_check_arr = MPL_malloc(sizeof(uint64_t) * (comm_size + 4),MPL_MEM_OTHER);

    /* construct the struct to compare */
    /* It's MPI_AINT to uint64_t, so do copy one by one instead of MPIR_Localcopy()*/
    for(i = 0; i < comm_size; ++i)
    {
    	local_check_arr[i] = recvcounts[i];
    }

    mpi_errno = dtype_sig_generator(sendtype, &sendsig);
    sendsig.num_types = sendsig.num_types * sendcount;

    mpi_errno = dtype_sig_generator(recvtype, &recvsig);
    temp_num_types = recvsig.num_types;
    recvsig.num_types = 0;

    for(i = 0; i < comm_ptr->local_size; ++i)
    {
    	recvsig.num_types += recvcounts[i];
    }

    recvsig.num_types = recvsig.num_types * temp_num_types;
    
    /* For local_check_arr */ 
    /* store recvcounts on pos 0 to comm_size-1,*/
    /* store sendsig.hash_value on pos comm_size,*/
    /* store recvsig.hash_value on pos comm_size+1,*/
    /* store recvsig.num_types on pos comm_size+2,*/
    /* store comparing result of sendcount to recvcounts[my_rank] on pos comm_size+3.*/
    local_check_arr[comm_size] = sendsig.hash_value;
    local_check_arr[comm_size+1] = recvsig.hash_value;
    local_check_arr[comm_size+2] = recvsig.num_types;

    // TODO: verify what's the rank of a process in a inter communicator?
    // Now it should work correctly for intra communicators.
    if(sendsig.hash_value == recvsig.hash_value &&
    		sendsig.num_types == recvcounts[comm_ptr->rank])
    	local_check_arr[comm_size+3] = 1ULL;
    else
    	local_check_arr[comm_size+3] = 0ULL;

    is_equal = 1;

	mpi_errno = MPIR_Allreduce_equal(local_check_arr, comm_size+4, MPI_UINT64_T, &is_equal, comm_ptr, errflag);

    if(is_equal && (*errflag == MPIR_ERR_NONE))
    {
    	/*
    	Please verify: do recvcount = sendcount * numprocs needed in alltoall and allgather?
    	 */
    	if(!local_check_arr[comm_size+3])
    	{
    		*errflag = MPI_ERR_ARG;
        	MPIR_ERR_SET(mpi_errno, MPI_ERR_ARG, "**collective_size_mismatch");
        	return mpi_errno;
    	}

        return mpi_errno;
    }
    else
    {
        *errflag = MPI_ERR_ARG;
        MPIR_ERR_SET(mpi_errno, MPI_ERR_ARG, "**collective_size_mismatch");
        return mpi_errno;
    }
}


/*
If typecount == 1, called by alltoallv. 
Else, called by alltoallw.
 */
int MPIR_Coll_len_check_alltoallvw(MPI_Aint* sendcount, MPI_Aint *sdispls, MPI_Datatype *sendtype, MPI_Aint *recvcount, 
		MPI_Aint *rdispls, MPI_Datatype *recvtype, MPI_Aint typecount, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
	int mpi_errno = MPI_SUCCESS;
	type_sig *sendsigs, *recvsigs, temp_sig;
	int i, is_sigs_equal, is_equal;

	if(typecount != 1 && typecount != comm_ptr->local_size)
	{
		*errflag = MPIR_ERR_OTHER;
        MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
            "**collective_size_mismatch",
            "**collective_size_mismatch %d %d", typecount, comm_ptr->rank);
        return mpi_errno;
	}

	sendsigs = MPL_malloc(sizeof(type_sig) * comm_ptr->local_size, MPL_MEM_OTHER);
	recvsigs = MPL_malloc(sizeof(type_sig) * comm_ptr->local_size, MPL_MEM_OTHER);

	if(typecount == 1)
	{
        mpi_errno = dtype_sig_generator(sendtype[0], &temp_sig);

		for(i = 0; i < comm_ptr->local_size; ++i)
		{
			sendsigs[i].hash_value = temp_sig.hash_value;
			sendsigs[i].num_types = temp_sig.num_types * sendcount[i];
		}
	}
	else
	{
		for(i = 0; i < comm_ptr->local_size; ++i)
		{
            mpi_errno = dtype_sig_generator(sendtype[i], &sendsigs[i]);
			sendsigs[i].num_types = sendsigs[i].num_types * sendcount[i];
		}
	}


	mpi_errno = MPIR_Alltoall_impl(sendsigs, 2, MPI_UNSIGNED_LONG, recvsigs, 2, MPI_UNSIGNED_LONG, comm_ptr, errflag);

	is_sigs_equal = 1;
	if(typecount == 1)
	{
        mpi_errno = dtype_sig_generator(recvtype[0], &temp_sig);

		for(i = 0; i < comm_ptr->local_size; ++i)
		{
			if(temp_sig.hash_value != recvsigs[i].hash_value || 
					(temp_sig.num_types * recvcount[i]) != recvsigs[i].num_types)
				is_sigs_equal = 0;
		}
	}
	else
	{
		for(i = 0; i < comm_ptr->local_size; ++i)
		{
            mpi_errno = dtype_sig_generator(recvtype[i], &temp_sig);

			if(temp_sig.hash_value != recvsigs[i].hash_value || 
					(temp_sig.num_types * recvcount[i]) != recvsigs[i].num_types)
				is_sigs_equal = 0;
		}

	}

	MPL_free(sendsigs);
	MPL_free(recvsigs);

	mpi_errno = MPIR_Allreduce_equal(&is_sigs_equal, 1, MPI_INT, &is_equal, comm_ptr, errflag);

	if(is_equal && is_sigs_equal && (*errflag == MPIR_ERR_NONE))
    {    
        return mpi_errno;
    }
    else
    {
        *errflag = MPIR_ERR_OTHER;
        MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
            "**collective_size_mismatch",
            "**collective_size_mismatch %d %d", typecount, comm_ptr->rank);
        return mpi_errno;
    }

	return mpi_errno;
}

