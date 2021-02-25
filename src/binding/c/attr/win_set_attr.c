/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* -- THIS FILE IS AUTO-GENERATED -- */

#include "mpiimpl.h"

/* -- Begin Profiling Symbol Block for routine MPI_Win_set_attr */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Win_set_attr = PMPI_Win_set_attr
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Win_set_attr  MPI_Win_set_attr
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Win_set_attr as PMPI_Win_set_attr
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val)
     __attribute__ ((weak, alias("PMPI_Win_set_attr")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Win_set_attr
#define MPI_Win_set_attr PMPI_Win_set_attr

int MPII_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val, MPIR_Attr_type attr_type)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Win *win_ptr = NULL;
    MPII_Keyval *win_keyval_ptr = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_WIN_SET_ATTR);

    MPIR_ERRTEST_INITIALIZED_ORDIE();

    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_WIN_SET_ATTR);

#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_ERRTEST_WIN(win, mpi_errno);
            MPIR_ERRTEST_KEYVAL(win_keyval, MPIR_WIN, "window", mpi_errno);
            MPIR_ERRTEST_KEYVAL_PERM(win_keyval, mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    MPIR_Win_get_ptr(win, win_ptr);
    MPII_Keyval_get_ptr(win_keyval, win_keyval_ptr);

#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_Win_valid_ptr(win_ptr, mpi_errno);
            if (mpi_errno) {
                goto fn_fail;
            }
            MPII_Keyval_valid_ptr(win_keyval_ptr, mpi_errno);
            if (mpi_errno) {
                goto fn_fail;
            }
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ... */
    mpi_errno = MPIR_Win_set_attr_impl(win_ptr, win_keyval_ptr, attribute_val, attr_type);
    if (mpi_errno) {
        goto fn_fail;
    }
    /* ... end of body of routine ... */

  fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_WIN_SET_ATTR);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLINE-- */
#ifdef HAVE_ERROR_CHECKING
    mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, __func__, __LINE__, MPI_ERR_OTHER,
                                     "**mpi_win_set_attr", "**mpi_win_set_attr %W %K %p", win,
                                     win_keyval, attribute_val);
#endif
    mpi_errno = MPIR_Err_return_win(win_ptr, __func__, mpi_errno);
    /* --END ERROR HANDLING-- */
    goto fn_exit;
}
#endif /* MPICH_MPI_FROM_PMPI */

/*@
   MPI_Win_set_attr - Stores attribute value associated with a key

Input Parameters:
+ win - window to which attribute will be attached (handle)
. win_keyval - key value (integer)
- attribute_val - attribute value (None)

Notes:

The type of the attribute value depends on whether C or Fortran is being used.
In C, an attribute value is a pointer ('void *'); in Fortran, it is an
address-sized integer.

If an attribute is already present, the delete function (specified when the
corresponding keyval was created) will be called.

.N ThreadSafe

.N Fortran

.N Errors
.N MPI_SUCCESS

.N MPI_ERR_KEYVAL
.N MPI_ERR_WIN
@*/

int MPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_WIN_SET_ATTR);
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_WIN_SET_ATTR);

    mpi_errno = MPII_Win_set_attr(win, win_keyval, attribute_val, MPIR_ATTR_PTR);

    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_WIN_SET_ATTR);
    return mpi_errno;
}