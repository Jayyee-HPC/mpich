/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

/* Equal structures */
/* For reusing code, using int type instead of bool*/
typedef struct MPIR_2int_eqltype {
    int value;
    int is_equal;
} MPIR_2int_eqltype;

typedef struct MPIR_longint_eqltype {
    long value;
    int is_equal;
} MPIR_longint_eqltype;

typedef struct MPIR_shortint_eqltype {
    short value;
    int is_equal;
} MPIR_shortint_eqltype;


/* If a child found unequal, its parent sticks to unequal.*/
/* Values of is_equal: 1 equal 0 not equal, init using any value other than 0.*/
#define MPIR_EQUAL_C_CASE_INT(c_type_, _len_) {             \
        c_type_ *a = (c_type_ *)inoutvec;                   \
        c_type_ *b = (c_type_ *)invec;                      \
        for (i = 0; i < _len_; ++i) {                       \
            if (!(b[i].is_equal) || !(a[i].is_equal)){      \
                a[i].is_equal = 0;                          \
            }else if (a[i].value != b[i].value){            \
                a[i].is_equal = 0;                          \
            }else{                                          \
                a[i].is_equal = 1;                          \
            }                                               \
        }                                                   \
    }                                                   


void MPIR_EQUAL_struct(void *invec, void *inoutvec, int *Len, MPI_Datatype * type)
{
    int len = *Len, element_len = 0, type_len = 0;
    int i, j, is_equal, *ints;
    void *invec_i_pos, *invec_i_bool_pos, *inoutvec_i_pos, *inoutvec_i_bool_pos;
    MPI_Aint *aints, *counts, lb, extent, size, element_pos;
    MPI_Datatype *types;
    MPIR_Datatype *typeptr;
    MPIR_Datatype_contents *cp;

    /* decode */
    MPIR_Datatype_get_ptr(*type, typeptr);

    cp = typeptr->contents;

    MPIR_Datatype_access_contents(cp, &ints, &aints, &counts, &types);

    MPIR_Type_get_extent_impl(*type, &lb, &extent);
    type_len = extent - lb;

    /*Compare structs*/
    for (i = 0; i < len; ++i)       
    {
        invec_i_pos = invec + i*type_len;
        invec_i_bool_pos = invec_i_pos + aints[cp->nr_aints-1];
        inoutvec_i_pos = inoutvec + i*type_len;
        inoutvec_i_bool_pos = inoutvec_i_pos + aints[cp->nr_aints-1];

        if(!(*(int*)invec_i_bool_pos) ||
            !(*(int*)inoutvec_i_bool_pos))
        {
        /* compare values of is_equal */
            *(int*)inoutvec_i_bool_pos  = 0;
        }
        else
        {
        /* compare the content of the struct */
            is_equal = 1;

            for(j = 0; j < cp->nr_types-1; ++j)
            {
                MPIR_Type_size_impl(types[j], &size); 
                element_len = ints[j + 1] * size;
                element_pos = aints[j];
                /* compare jth element of ith struct */
                if(memcmp(invec_i_pos+element_pos, inoutvec_i_pos+element_pos, element_len))
                    is_equal = 0;
            }

            *(int*)inoutvec_i_bool_pos  = is_equal;
        }
    }
}

void MPIR_EQUAL(void *invec, void *inoutvec, int *Len, MPI_Datatype * type)
{
    int i, len = *Len;
   
    switch (*type) {
            /* first the C types */
        case MPI_2INT:
            MPIR_EQUAL_C_CASE_INT(MPIR_2int_eqltype, len);
            break;
        case MPI_LONG_INT:
            MPIR_EQUAL_C_CASE_INT(MPIR_longint_eqltype, len);
            break;
        case MPI_SHORT_INT:
            MPIR_EQUAL_C_CASE_INT(MPIR_shortint_eqltype, len);
            break;

        /* default treate the type as a derived datatype */
        default:
            MPIR_EQUAL_struct(invec, inoutvec, Len, type);
            break;
    }

}


static int derived_datatype_check(MPI_Datatype type)
{
    /* Check if the derived type meets the requirements for MPI_EQUAL */
    int mpi_errno = MPI_SUCCESS; 
    int i, combiner, *ints; 
    MPI_Aint *aints, *counts;
    MPI_Datatype *types;
    MPIR_Datatype *typeptr;
    MPIR_Datatype_contents *cp;

    combiner = MPIR_Type_get_combiner(type);
    if(combiner != MPI_COMBINER_STRUCT)
    {
        MPIR_ERR_SET1(mpi_errno, MPI_ERR_OP, "**opundefined", "**opundefined %s", "MPIX_EQUAL");
        return mpi_errno;
    }

    MPIR_Datatype_get_ptr(type, typeptr);
    cp = typeptr->contents;

    MPIR_Datatype_access_contents(cp, &ints, &aints, &counts, &types);

    /* At least 2 elements is required, data, result */
    if(ints == NULL || ints[0] < 2) 
    {
        MPIR_ERR_SET1(mpi_errno, MPI_ERR_OP, "**opundefined", "**opundefined %s", "MPIX_EQUAL");
        return mpi_errno;
    }

    /* addrs is required to avoid impacts of struct alignment */
    if(aints == NULL || cp->nr_aints < 2) 
    {
        MPIR_ERR_SET1(mpi_errno, MPI_ERR_OP, "**opundefined", "**opundefined %s", "MPIX_EQUAL");
        return mpi_errno;
    }

    /* The last element has to be int */
    if(types[cp->nr_types-1] != MPI_INT) 
    {
        MPIR_ERR_SET1(mpi_errno, MPI_ERR_OP, "**opundefined", "**opundefined %s", "MPIX_EQUAL");
        return mpi_errno;
    }

    /* All elements have to be basic datatypes */
    for(i = 0; i < cp->nr_types-1; ++i)
    {
    	if(!HANDLE_IS_BUILTIN(types[i]))
    	{
    		MPIR_ERR_SET1(mpi_errno, MPI_ERR_OP, "**opundefined", "**opundefined %s", "MPIX_EQUAL");
        	return mpi_errno;
    	}
    }

    return mpi_errno;
}


int MPIR_EQUAL_check_dtype(MPI_Datatype type)
{
    /* To support user defined datatypes, no actual type check now. */
    int mpi_errno = MPI_SUCCESS;

    switch (type) {
        /*  C types */
        case MPI_2INT:
        case MPI_LONG_INT:
        case MPI_SHORT_INT:

            break;

        /* default treate the type as a derived datatype*/
        default:
            mpi_errno = derived_datatype_check(type);
            break;
    }

    return mpi_errno;
}
