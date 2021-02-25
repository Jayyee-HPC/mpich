/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* -- THIS FILE IS AUTO-GENERATED -- */

#include "mpiimpl.h"

/* -- Begin Profiling Symbol Block for routine MPI_Is_thread_main */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Is_thread_main = PMPI_Is_thread_main
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Is_thread_main  MPI_Is_thread_main
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Is_thread_main as PMPI_Is_thread_main
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Is_thread_main(int *flag)  __attribute__ ((weak, alias("PMPI_Is_thread_main")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Is_thread_main
#define MPI_Is_thread_main PMPI_Is_thread_main

#endif

/*@
   MPI_Is_thread_main - Returns a flag indicating whether this thread called 'MPI_Init' or 'MPI_Init_thread'

Output Parameters:
. flag - true if calling thread is main thread, false otherwise (logical)

.N ThreadSafe

.N Fortran

.N Errors
.N MPI_SUCCESS

.N MPI_ERR_ARG
@*/

int MPI_Is_thread_main(int *flag)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_IS_THREAD_MAIN);

    MPIR_ERRTEST_INITIALIZED_ORDIE();
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_IS_THREAD_MAIN);

#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_ERRTEST_ARGNULL(flag, "flag", mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ... */
#if MPICH_THREAD_LEVEL <= MPI_THREAD_FUNNELED || ! defined(MPICH_IS_THREADED)
    {
        *flag = TRUE;
    }
    #else
    {
        MPID_Thread_id_t my_thread_id;

        MPID_Thread_self(&my_thread_id);
        MPID_Thread_same(&MPIR_ThreadInfo.main_thread, &my_thread_id, flag);
    }
#endif
    /* ... end of body of routine ... */

  fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_IS_THREAD_MAIN);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLINE-- */
#ifdef HAVE_ERROR_CHECKING
    mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, __func__, __LINE__, MPI_ERR_OTHER,
                                     "**mpi_is_thread_main", "**mpi_is_thread_main %p", flag);
#endif
    mpi_errno = MPIR_Err_return_comm(0, __func__, mpi_errno);
    /* --END ERROR HANDLING-- */
    goto fn_exit;
}