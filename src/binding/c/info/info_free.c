/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* -- THIS FILE IS AUTO-GENERATED -- */

#include "mpiimpl.h"

/* -- Begin Profiling Symbol Block for routine MPI_Info_free */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Info_free = PMPI_Info_free
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Info_free  MPI_Info_free
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Info_free as PMPI_Info_free
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Info_free(MPI_Info *info)  __attribute__ ((weak, alias("PMPI_Info_free")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Info_free
#define MPI_Info_free PMPI_Info_free

#endif

/*@
   MPI_Info_free - Frees an info object

Input/Output Parameters:
. info - info object (handle)

.N ThreadSafe

.N Fortran

.N Errors
.N MPI_SUCCESS

.N MPI_ERR_ARG
.N MPI_ERR_INFO
@*/

int MPI_Info_free(MPI_Info *info)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Info *info_ptr = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_INFO_FREE);
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_INFO_FREE);

#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_ERRTEST_ARGNULL(info, "info", mpi_errno);
            MPIR_ERRTEST_INFO(*info, mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    MPIR_Info_get_ptr(*info, info_ptr);

#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_Info_valid_ptr(info_ptr, mpi_errno);
            if (mpi_errno) {
                goto fn_fail;
            }
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ... */
    mpi_errno = MPIR_Info_free_impl(info_ptr);
    if (mpi_errno) {
        goto fn_fail;
    }
    *info = MPI_INFO_NULL;
    /* ... end of body of routine ... */

  fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_INFO_FREE);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLINE-- */
#ifdef HAVE_ERROR_CHECKING
    mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, __func__, __LINE__, MPI_ERR_OTHER,
                                     "**mpi_info_free", "**mpi_info_free %p", info);
#endif
    mpi_errno = MPIR_Err_return_comm(0, __func__, mpi_errno);
    /* --END ERROR HANDLING-- */
    goto fn_exit;
}