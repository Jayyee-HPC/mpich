/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* -- THIS FILE IS AUTO-GENERATED -- */

#include "mpiimpl.h"

/* -- Begin Profiling Symbol Block for routine MPI_T_cvar_get_info */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_T_cvar_get_info = PMPI_T_cvar_get_info
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_T_cvar_get_info  MPI_T_cvar_get_info
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_T_cvar_get_info as PMPI_T_cvar_get_info
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_T_cvar_get_info(int cvar_index, char *name, int *name_len, int *verbosity,
                        MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len,
                        int *bind, int *scope)
                         __attribute__ ((weak, alias("PMPI_T_cvar_get_info")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_T_cvar_get_info
#define MPI_T_cvar_get_info PMPI_T_cvar_get_info

#endif

/*@
   MPI_T_cvar_get_info - Get the information about a control variable

Input/Output Parameters:
+ name_len - length of the string and/or buffer for name (integer)
- desc_len - length of the string and/or buffer for desc (integer)

Input Parameters:
. cvar_index - index of the control variable to be queried, value between $0$ and $num_cvar-1$ (integer)

Output Parameters:
+ name - buffer to return the string containing the name of the control variable (string)
. verbosity - verbosity level of this variable (integer)
. datatype - mpi datatype of the information stored in the control variable (handle)
. enumtype - optional descriptor for enumeration information (handle)
. desc - buffer to return the string containing a description of the control variable (string)
. bind - type of mpi object to which this variable must be bound (integer)
- scope - scope of when changes to this variable are possible (integer)

.N ThreadSafe

.N Errors
.N MPI_SUCCESS

.N MPI_T_ERR_INVALID_INDEX
.N MPI_T_ERR_NOT_INITIALIZED
@*/

int MPI_T_cvar_get_info(int cvar_index, char *name, int *name_len, int *verbosity,
                        MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len,
                        int *bind, int *scope)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_T_CVAR_GET_INFO);

    MPIT_ERRTEST_MPIT_INITIALIZED();

    MPIR_T_THREAD_CS_ENTER();
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_T_CVAR_GET_INFO);

#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIT_ERRTEST_CVAR_INDEX(cvar_index);
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ... */
    cvar_table_entry_t *cvar;
    cvar = (cvar_table_entry_t *) utarray_eltptr(cvar_table, cvar_index);

    MPIR_T_strncpy(name, cvar->name, name_len);
    MPIR_T_strncpy(desc, cvar->desc, desc_len);

    if (verbosity != NULL)
        *verbosity = cvar->verbosity;

    if (datatype != NULL)
        *datatype = cvar->datatype;

    if (enumtype != NULL)
        *enumtype = cvar->enumtype;

    if (bind!= NULL)
        *bind= cvar->bind;

    if (scope != NULL)
        *scope = cvar->scope;
    /* ... end of body of routine ... */

  fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_T_CVAR_GET_INFO);
    MPIR_T_THREAD_CS_EXIT();
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}