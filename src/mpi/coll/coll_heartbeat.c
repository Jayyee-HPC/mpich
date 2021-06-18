/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

static const uint16_t coll_op_get_id(const char *op_name);
static const uint64_t hb_KVS_key_gen(const char *op_name, MPIR_Comm * comm_ptr);
static void heartbeat_t_insert_kvs(uint64_t key, uint64_t value, MPIR_Comm * comm_ptr);
static void heartbeat_t_remove_kvs(uint64_t key, uint64_t value, MPIR_Comm * comm_ptr);

typedef struct coll_op_id {
    const char *op_name;
    const uint16_t op_id;    
} coll_op_id_t;

static coll_op_id_t coll_op_ids[] = {
    {"MPIR_NO_Coll", 0},
    {"MPIR_Allgather", 1},
    {"MPIR_Allgatherv", 2},
    {"MPIR_Allreduce", 3},
    {"MPIR_Alltoall", 4},
    {"MPIR_Alltoallv", 5},
    {"MPIR_Alltoallw", 6},
    {"MPIR_Bcast", 7},
    {"MPIR_Exscan", 8},
    {"MPIR_Gather", 9},
    {"MPIR_Gatherv", 10},
    {"MPIR_Reduce", 11},
    {"MPIR_Scan", 12},
    {"MPIR_Scatter", 13},
    {"MPIR_Scatterv", 14},
    {"MPIR_Barrier", 15}
};

static const uint16_t coll_op_get_id(const char *op_name)
{
    int i;
    
    for (i = 0; i < sizeof(coll_op_ids) / sizeof(coll_op_id_t); ++i) 
    {
        if (strcmp(coll_op_ids[i].op_name, op_name) == 0)
            return coll_op_ids[i].op_id;
    }

    MPIR_Assert(0);//Exit if coll op id not found
    return 0;
}

static const uint64_t hb_KVS_key_gen(const char *op_name, MPIR_Comm * comm_ptr)
{
    int i;
    uint16_t op_id, comm_id, world_rank;
    uint64_t return_value;

    MPIR_Comm *comm_world_ptr = NULL;

    //printf("%.16"PRIx64" %p \n", MPIR_Process.heartbeat_t.key, (void*)(MPIR_Process.heartbeat_t.next));
    MPIR_Comm_get_ptr(MPI_COMM_WORLD, comm_world_ptr);

    op_id = coll_op_get_id(op_name);

    world_rank = (comm_world_ptr->rank) & 0xFFFF;

    comm_id = comm_ptr->context_id;
    /* Assemble KVS value */
    return_value = 0;
    return_value += (((uint64_t)world_rank)<<48);
    return_value += (((uint64_t)op_id)<<32);
    return_value += (((uint64_t)comm_id)<<16);
    return_value += 0x7777; //A simple check that this value is a heartbeat message

    return return_value;
}

static void heartbeat_t_insert_kvs(uint64_t key, uint64_t value, MPIR_Comm * comm_ptr)
{
    struct MPIR_Heartbeat_t * current, *prev;

    prev = &(MPIR_Process.heartbeat_t);
    current = MPIR_Process.heartbeat_t.next;

    if(prev->key == 0 && prev->value == 0 && prev->comm == NULL)
    {
        /* Store & remove in the entry to save one MPL_malloc and MPL_free */
        prev->comm = comm_ptr;
        prev->key = key;
        prev->value = value;
        return;
    }

    struct MPIR_Heartbeat_t *temp_hb = MPL_malloc(sizeof(struct MPIR_Heartbeat_t), MPL_MEM_OTHER);

    temp_hb->next = NULL;
    temp_hb->comm = comm_ptr;
    temp_hb->key = key;
    temp_hb->value = value;

    /* search the table to avoid collision */
    /* PLEASE VERIFY: Is it true that only one collective allowed at same time for one comm? */
    while (NULL != current)
    {
        if (current->comm == comm_ptr)
        {
            printf("WARNING, comm %s appears twice (or more) in the hearbeat table\n", comm_ptr->name);
        }
        prev = current;
        current = current->next;
    }

    prev->next = temp_hb;
}

static void heartbeat_t_remove_kvs(uint64_t key, uint64_t value, MPIR_Comm * comm_ptr)
{
    struct MPIR_Heartbeat_t * current, *prev;
    prev = &(MPIR_Process.heartbeat_t);
    current = MPIR_Process.heartbeat_t.next;

    if(prev->comm == comm_ptr && prev->key == key && prev->value == value)
    {
        /* Store & remove in the entry to save one MPL_malloc and MPL_free */
        prev->comm = NULL;
        prev->key = 0;
        prev->value = 0;
        return;
    }

    while (NULL != current)
    {
        if (current->comm == comm_ptr && current->key == key && current->value == value)
        {
            prev->next = current->next;
            MPL_free(current);
            return;
        }

        prev = current;
        current = current->next;
    }

    printf("Can not find comm %s in the hearbeat table\n", comm_ptr->name);
    assert(0);
}

int MPIR_Coll_heartbeat_put(coll_hb_info * op_info, MPIR_Comm * comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    char kvs_key[MAXKEYLEN];
    
    op_info->key_uint64 = hb_KVS_key_gen(op_info->op_name, comm_ptr);

    sprintf(kvs_key, "%.16"PRIx64"", op_info->key_uint64);
    //send recv rank

    strcpy(op_info->key, kvs_key);
    
    comm_ptr->hb_counter++;
    op_info->value_uint64 = comm_ptr->hb_counter;
    sprintf(op_info->value, "%.16"PRIx64"",  comm_ptr->hb_counter);

    heartbeat_t_insert_kvs(op_info->key_uint64, op_info->value_uint64, comm_ptr);
    //if(comm_ptr->hb_counter % 2)
        //strcpy(op_info->value, "9");
    //    sprintf(op_info->value, "%.08x",  9);
    //else
        //strcpy(op_info->value, "10");
    //    sprintf(op_info->value, "%.08x",  10);
    //MPIR_Comm *comm_world_ptr = NULL;
    //MPIR_Comm_get_ptr(MPI_COMM_WORLD, comm_world_ptr);
    //printf("%d %u %u\n", comm_world_ptr->rank, (unsigned int)comm_world_ptr->context_id, (unsigned int)comm_world_ptr->recvcontext_id);

    //printf("%d %s\n", comm_ptr->rank, op_info->value);
    mpi_errno = PMI_KVS_Put(op_info->kvsname, op_info->key, op_info->value);

    if (mpi_errno != PMI_SUCCESS)
    {
        mpi_errno =
            MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,
                                 MPI_ERR_OTHER, "**pmi_kvs_put", 0);

        return mpi_errno;
    }

    mpi_errno =  PMI_KVS_Commit(op_info->kvsname);

    if (mpi_errno != PMI_SUCCESS)
    {
        mpi_errno =
            MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,
                                 MPI_ERR_OTHER, "**pmi_kvs_commit", 0);

        return mpi_errno;
    }
    return mpi_errno;
}

int MPIR_Coll_heartbeat_get(coll_hb_info * op_info, char key[], char value[], int* bufsize)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = PMI_KVS_Get(op_info->kvsname, key, value, *bufsize );
    
    return mpi_errno;
}

int MPIR_Coll_heartbeat_t_remove(coll_hb_info * op_info, MPIR_Comm * comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    heartbeat_t_remove_kvs(op_info->key_uint64, op_info->value_uint64, comm_ptr);

    return mpi_errno;
}

static int PMI_fd = -1;

int MPIR_Coll_heartbeat_server(void)
{
    int mpi_errno = MPI_SUCCESS;
    char *failed_procs_string;
    int pmi_max_val_size;
    uint16_t key = 0;
    uint16_t value = 0;
    char *p;

    const char *pmi_kvs_name = MPIR_pmi_job_id();

    if (PMI_fd == -1)
    {
        p = getenv("PMI_FD");

        if (p) {
            PMI_fd = atoi(p);
            //return PMI_SUCCESS;
        }
    }

    //if (PMI_fd != -1)
        //HYD_cache_flush_temp(PMI_fd);

    struct MPIR_Heartbeat_t * current;

    current = &(MPIR_Process.heartbeat_t);

    while (NULL != current)
    {
        printf("%s %d %.16"PRIx64" %.16"PRIx64" \n", current->comm->name, current->comm->rank, current->key, current->value);
        current = current->next;
    }


    //assert(key != 0);
    //assert(value != 0);

    pmi_max_val_size = MPIR_pmi_max_val_size();
    

    failed_procs_string = MPL_malloc(pmi_max_val_size, MPL_MEM_OTHER);
    MPIR_Assert(failed_procs_string);

    mpi_errno = PMI_KVS_Get(pmi_kvs_name, "PMI_dead_processes",
                            failed_procs_string, pmi_max_val_size);

    printf("MPIR_Coll_heartbeat_server %s\n", failed_procs_string); 

    if (strcmp(failed_procs_string, "") != 0)
    {
        MPL_free(failed_procs_string);
        printf("Failed procs %s\n", failed_procs_string); 
        return -1;
    }
        
    MPL_free(failed_procs_string);
    return mpi_errno;
}
