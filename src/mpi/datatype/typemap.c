/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "datatype.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int MPIR_type_dump_typemap(MPI_Datatype dt);
int MPIR_type_dump_typesig(MPI_Datatype dt);
struct typemap *MPIR_type_get_typemap(MPI_Datatype dt);
struct typesig *MPIR_type_get_typesig(MPI_Datatype dt);
void MPIR_type_free_typemap(struct typemap *map);
void MPIR_type_free_typesig(struct typesig *sig);

static void typemap_set(struct typemap *map, MPI_Datatype dt, MPI_Aint idx, MPI_Aint offset);
static void typesig_set(struct typesig *sig, MPI_Datatype dt, MPI_Aint * p_idx, MPI_Aint count);
static void type_get_n_elem_extent(MPI_Datatype dt, MPI_Aint * p_n_elem, MPI_Aint * p_extent);
static void typesig_check_space(struct typesig *sig, MPI_Aint n);

int MPIR_type_dump_typemap(MPI_Datatype dt)
{
    MPI_Aint lb;
    MPI_Aint extent;
    MPIR_Type_get_extent_impl(dt, &lb, &extent);
    printf("  %20s: %10ld\n", "lb", (long) lb);
    printf("  %20s: %10ld\n", "ub", (long) (lb + extent));

    struct typemap *map;
    map = MPIR_type_get_typemap(dt);
    for (int i = 0; i < map->n; i++) {
        printf("  %20s: %10ld\n", MPIR_Datatype_builtin_to_string(map->types[i]),
               (long) map->disps[i]);
    }

    MPIR_type_free_typemap(map);
    return MPI_SUCCESS;
}

int MPIR_type_dump_typesig(MPI_Datatype dt)
{
    struct typesig *sig;
    sig = MPIR_type_get_typesig(dt);
    for (int i = 0; i < sig->n; i++) {
        if (i > 0) {
            printf(",");
        }
        printf("%s:%ld", MPIR_Datatype_builtin_to_string(sig->types[i]), (long) sig->counts[i]);
    }
    puts("");

    MPIR_type_free_typesig(sig);
    return MPI_SUCCESS;
}

struct typemap *MPIR_type_get_typemap(MPI_Datatype dt)
{
    struct typemap *map;

    map = (struct typemap *) malloc(sizeof(struct typemap));

    if (HANDLE_IS_BUILTIN(dt)) {
        if (dt == MPI_2INT) {
            map->n = 2;
        } else {
            map->n = 1;
        }
    } else {
        MPIR_Datatype *dt_ptr;
        MPIR_Datatype_get_ptr(dt, dt_ptr);
        MPIR_Assert(dt_ptr != NULL);
        map->n = dt_ptr->n_builtin_elements;
    }

    MPIR_Assert(map->n > 0);
    map->types = MPL_malloc(map->n * sizeof(MPI_Datatype), MPL_MEM_OTHER);
    map->disps = MPL_malloc(map->n * sizeof(MPI_Aint), MPL_MEM_OTHER);
    MPI_Aint n_elem;
    MPI_Aint extent;
    typemap_set(map, dt, 0, 0);
    return map;
}

struct typesig *MPIR_type_get_typesig(MPI_Datatype dt)
{
    struct typesig *sig;

    sig = (struct typesig *) malloc(sizeof(struct typesig));
    sig->n = 1;
    sig->types = MPL_malloc(1 * sizeof(MPI_Datatype), MPL_MEM_OTHER);
    sig->counts = MPL_malloc(1 * sizeof(MPI_Aint), MPL_MEM_OTHER);

    MPI_Aint idx = 0;
    typesig_set(sig, dt, &idx, 1);
    sig->n = idx;
    return sig;
}

void MPIR_type_free_typemap(struct typemap *map)
{
    MPL_free(map->types);
    MPL_free(map->disps);
    MPL_free(map);
}

void MPIR_type_free_typesig(struct typesig *sig)
{
    MPL_free(sig->types);
    MPL_free(sig->counts);
    MPL_free(sig);
}

void typemap_set(struct typemap *map, MPI_Datatype dt, MPI_Aint idx, MPI_Aint offset)
{
    int *p_ints;
    MPI_Aint *p_aints;
    MPI_Aint *p_counts;
    MPI_Datatype *p_types;
    MPI_Aint i;
    MPI_Aint j;

    if (HANDLE_IS_BUILTIN(dt)) {
        if (dt == MPI_2INT) {
            map->types[idx] = MPI_INT;
            map->disps[idx] = offset;
            map->types[idx + 1] = MPI_INT;
            map->disps[idx + 1] = offset + MPIR_Datatype_get_basic_size(MPI_INT);
        } else {
            map->types[idx] = dt;
            map->disps[idx] = offset;
        }
        return;
    } else if (MPIR_DATATYPE_IS_PREDEFINED(dt)) {
        MPIR_Datatype *dt_ptr;
        MPIR_Datatype_get_ptr(dt, dt_ptr);
        MPIR_Assert(dt_ptr != NULL);
        MPI_Aint disp = dt_ptr->true_ub - MPIR_Datatype_get_basic_size(MPI_INT);
        if (dt == MPI_FLOAT_INT) {
            map->types[idx] = MPI_FLOAT;
        }
        if (dt == MPI_DOUBLE_INT) {
            map->types[idx] = MPI_DOUBLE;
        }
        if (dt == MPI_LONG_INT) {
            map->types[idx] = MPI_LONG;
        }
        if (dt == MPI_SHORT_INT) {
            map->types[idx] = MPI_SHORT;
        }
        map->disps[idx] = offset;
        map->types[idx + 1] = MPI_INT;
        map->disps[idx + 1] = offset + disp;
        return;
    } else {
        MPI_Aint n_elem;
        MPI_Aint extent;

        MPIR_Datatype *dt_ptr;
        MPIR_Datatype_get_ptr(dt, dt_ptr);
        MPIR_Assert(dt_ptr != NULL);
        MPIR_Datatype_contents *cp = dt_ptr->contents;
        MPIR_Datatype_access_contents(cp, &p_ints, &p_aints, &p_counts, &p_types);
        if (cp->nr_counts == 0) {
            if (cp->combiner == MPI_COMBINER_DUP) {
                typemap_set(map, p_types[0], idx, offset);
            } else if (cp->combiner == MPI_COMBINER_RESIZED) {
                typemap_set(map, p_types[0], idx, offset + p_aints[0]);
            } else if (cp->combiner == MPI_COMBINER_CONTIGUOUS) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset);
                MPI_Aint idx2 = idx + n_elem;
                for (int i = 1; i < p_ints[0]; i++) {
                    for (int j = 0; j < n_elem; j++) {
                        map->types[idx2] = map->types[idx + j];
                        map->disps[idx2] = map->disps[idx + j] + extent * i;
                        idx2++;
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_VECTOR) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_ints[0]; k++) {
                    off2 = p_ints[2] * k * extent;
                    for (int i = 0; i < p_ints[1]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_HVECTOR) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_ints[0]; k++) {
                    off2 = p_aints[0];
                    for (int i = 0; i < p_ints[1]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_INDEXED_BLOCK) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + p_ints[2 + 0]);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_ints[0]; k++) {
                    off2 = (p_ints[2 + k] - p_ints[2 + 0]) * extent;
                    for (int i = 0; i < p_ints[1]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_HINDEXED_BLOCK) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + p_aints[0]);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_ints[0]; k++) {
                    off2 = p_aints[k] - p_aints[0];
                    for (int i = 0; i < p_ints[1]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_INDEXED) {
                int *p_blkl = p_ints + 1;
                int *p_disp = p_ints + 1 + p_ints[0];
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + p_disp[0]);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_ints[0]; k++) {
                    off2 = (p_disp[k] - p_disp[0]) * extent;
                    for (int i = 0; i < p_blkl[k]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_HINDEXED) {
                int *p_blkl = p_ints + 1;
                MPI_Aint *p_disp = p_aints;
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + p_disp[0]);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_ints[0]; k++) {
                    off2 = p_disp[k] - p_disp[0];
                    for (int i = 0; i < p_blkl[k]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_STRUCT) {
                int *p_blkl = p_ints + 1;
                MPI_Aint *p_disp = p_aints;
                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx;
                for (int k = 0; k < p_ints[0]; k++) {
                    type_get_n_elem_extent(p_types[k], &n_elem, &extent);
                    typemap_set(map, p_types[k], idx2, offset + p_disp[k]);
                    idx2 += n_elem;
                    for (int i = 1; i < p_blkl[k]; i++) {
                        for (int j = 0; j < n_elem; j++) {
                            map->types[idx2] = map->types[idx + j];
                            map->disps[idx2] = map->disps[idx + j] + extent * i;
                            idx2++;
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_SUBARRAY) {
                int order = p_ints[1 + p_ints[0] * 3];
                int *p_sizes = p_ints + 1;
                int *p_subsizes = p_ints + 1 + p_ints[0];
                int *p_starts = p_ints + 1 + p_ints[0] * 2;
                MPI_Aint off0 = 0;
                if (order == MPI_ORDER_C) {
                    for (int i = 0; i < p_ints[0]; i++) {
                        off0 = (off0 * p_sizes[i]) + p_starts[i];
                    }
                } else {
                    for (int i = p_ints[0] - 1; i >= 0; i--) {
                        off0 = (off0 * p_sizes[i]) + p_starts[i];
                    }
                }
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + off0);

                MPI_Aint off2 = off0;
                MPI_Aint idx2 = idx + n_elem;
                MPI_Aint *counters;
                counters = MPL_calloc(p_ints[0], sizeof(MPI_Aint), MPL_MEM_OTHER);
                while (1) {
                    int alldone = 0;
                    MPI_Aint stride = 1;
                    int i;
                    if (order == MPI_ORDER_C) {
                        i = p_ints[0] - 1;
                    } else {
                        i = 0;
                    }
                    while (1) {
                        counters[i]++;
                        off2 += stride;
                        if (counters[i] < p_subsizes[i]) {
                            break;
                        }
                        off2 -= stride * p_subsizes[i];
                        counters[i] = 0;
                        stride *= p_sizes[i];
                        if (order == MPI_ORDER_C) {
                            i--;
                            if (i < 0) {
                                alldone = 1;
                                break;
                            }
                        } else {
                            i++;
                            if (i == p_ints[0]) {
                                alldone = 1;
                                break;
                            }
                        }
                    }
                    if (alldone) {
                        break;
                    }
                    for (int j = 0; j < n_elem; j++) {
                        map->types[idx2] = map->types[idx + j];
                        map->disps[idx2] = map->disps[idx + j] + off2 - off0;
                        idx2++;
                    }
                }
                MPL_free(counters);
            } else if (cp->combiner == MPI_COMBINER_DARRAY) {
                int size = p_ints[0];
                int rank = p_ints[1];
                int n = p_ints[2];
                int *p_gsizes = p_ints + 3;
                int *p_distribs = p_ints + 3 + n;
                int *p_dargs = p_ints + 3 + n * 2;
                int *p_sizes = p_ints + 3 + n * 3;
                int order = p_ints[3 + n * 4];

                int *p_prank;
                p_prank = MPL_malloc(n * sizeof(int), MPL_MEM_OTHER);
                int *p_isblk;
                p_isblk = MPL_malloc(n * sizeof(int), MPL_MEM_OTHER);
                MPI_Aint *p_parg;
                p_parg = MPL_malloc(n * sizeof(MPI_Aint), MPL_MEM_OTHER);
                MPI_Aint *p_starts;
                p_starts = MPL_malloc(n * sizeof(MPI_Aint), MPL_MEM_OTHER);
                for (int i = 0; i < n; i++) {
                    size /= p_sizes[i];
                    p_prank[i] = rank / size;
                    rank = rank % size;
                    if (p_sizes[i] == 1) {
                        p_isblk[i] = 1;
                        p_parg[i] = p_gsizes[i];
                        p_starts[i] = 0;
                    } else if (p_distribs[i] == MPI_DISTRIBUTE_BLOCK &&
                               p_dargs[i] == MPI_DISTRIBUTE_DFLT_DARG) {
                        p_parg[i] = p_gsizes[i] / p_sizes[i];
                        if (p_gsizes[i] % p_sizes[i] == 0) {
                            p_isblk[i] = 1;
                            p_starts[i] = p_parg[i] * p_prank[i];
                        } else {
                            int r = p_gsizes[i] % p_sizes[i];
                            p_isblk[i] = 2;
                            if (p_prank[i] < r) {
                                p_parg[i] += 1;
                                p_starts[i] = p_parg[i] * p_prank[i];
                            } else {
                                p_starts[i] = p_parg[i] * p_prank[i] + r;
                            }
                        }
                    } else if (p_distribs[i] == MPI_DISTRIBUTE_CYCLIC &&
                               p_dargs[i] == MPI_DISTRIBUTE_DFLT_DARG) {
                        p_isblk[i] = 0;
                        p_parg[i] = 1;
                        p_starts[i] = p_prank[i];
                    } else {
                        p_isblk[i] = 0;
                        p_parg[i] = p_dargs[i];
                        p_starts[i] = p_parg[i] * p_prank[i];
                    }
                }

                MPI_Aint off0 = 0;
                if (order == MPI_ORDER_C) {
                    for (int i = 0; i < n; i++) {
                        off0 = (off0 * p_gsizes[i]) + p_starts[i];
                    }
                } else {
                    for (int i = n - 1; i >= 0; i--) {
                        off0 = (off0 * p_gsizes[i]) + p_starts[i];
                    }
                }
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + off0);

                MPI_Aint off2 = off0;
                MPI_Aint idx2 = idx + n_elem;
                MPI_Aint *counters;
                counters = MPL_calloc(n, sizeof(MPI_Aint), MPL_MEM_OTHER);
                if (p_isblk[0] == 0) {
                    counters[0] = p_prank[0] * p_parg[0];
                }
                while (1) {
                    int alldone = 0;
                    MPI_Aint stride = 1;
                    int i;
                    if (order == MPI_ORDER_C) {
                        i = n - 1;
                    } else {
                        i = 0;
                    }
                    while (1) {
                        if (p_isblk[i] == 0) {
                            MPI_Aint old_counter = counters[i];
                            counters[i]++;
                            if (counters[i] % p_parg[i] == 0) {
                                counters[i] += p_parg[i] * (p_sizes[i] - 1);
                            }
                            if (counters[i] < p_gsizes[i]) {
                                off2 += stride * (counters[i] - old_counter);
                                break;
                            }
                            counters[i] = p_starts[i];
                            off2 += stride * (counters[i] - old_counter);
                        } else {
                            counters[i]++;
                            off2 += stride;
                            if (counters[i] < p_parg[i]) {
                                break;
                            }
                            off2 -= stride * p_parg[i];
                            counters[i] = 0;
                        }
                        stride *= p_gsizes[i];
                        if (order == MPI_ORDER_C) {
                            i--;
                            if (i < 0) {
                                alldone = 1;
                                break;
                            }
                        } else {
                            i++;
                            if (i == n) {
                                alldone = 1;
                                break;
                            }
                        }
                    }
                    if (alldone) {
                        break;
                    }
                    for (int j = 0; j < n_elem; j++) {
                        map->types[idx2] = map->types[idx + j];
                        map->disps[idx2] = map->disps[idx + j] + off2 - off0;
                        idx2++;
                    }
                }
                MPL_free(counters);

                MPL_free(p_prank);
                MPL_free(p_isblk);
                MPL_free(p_parg);
                MPL_free(p_starts);
            } else {
                MPIR_Assert(0);
            }

        } else {
            if (cp->combiner == MPI_COMBINER_DUP) {
                typemap_set(map, p_types[0], idx, offset);
            } else if (cp->combiner == MPI_COMBINER_RESIZED) {
                typemap_set(map, p_types[0], idx, offset + p_counts[0]);
            } else if (cp->combiner == MPI_COMBINER_CONTIGUOUS) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset);
                MPI_Aint idx2 = idx + n_elem;
                for (int i = 1; i < p_counts[0]; i++) {
                    for (int j = 0; j < n_elem; j++) {
                        map->types[idx2] = map->types[idx + j];
                        map->disps[idx2] = map->disps[idx + j] + extent * i;
                        idx2++;
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_VECTOR) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_counts[0]; k++) {
                    off2 = p_counts[2] * k * extent;
                    for (int i = 0; i < p_counts[1]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_HVECTOR) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_counts[0]; k++) {
                    off2 = p_counts[2] * k;
                    for (int i = 0; i < p_counts[1]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_INDEXED_BLOCK) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + p_counts[2 + 0]);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_counts[0]; k++) {
                    off2 = (p_counts[2 + k] - p_counts[2 + 0]) * extent;
                    for (int i = 0; i < p_counts[1]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_HINDEXED_BLOCK) {
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + p_counts[2 + 0]);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_counts[0]; k++) {
                    off2 = p_counts[2 + k] - p_counts[2 + 0];
                    for (int i = 0; i < p_counts[1]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_INDEXED) {
                MPI_Aint *p_blkl = p_counts + 1;
                MPI_Aint *p_disp = p_counts + 1 + p_counts[0];
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + p_disp[0]);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_counts[0]; k++) {
                    off2 = (p_disp[k] - p_disp[0]) * extent;
                    for (int i = 0; i < p_blkl[k]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_HINDEXED) {
                MPI_Aint *p_blkl = p_counts + 1;
                MPI_Aint *p_disp = p_counts + 1 + p_counts[0];
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + p_disp[0]);

                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx + n_elem;
                for (int k = 0; k < p_counts[0]; k++) {
                    off2 = p_disp[k] - p_disp[0];
                    for (int i = 0; i < p_blkl[k]; i++) {
                        if (k || i) {
                            for (int j = 0; j < n_elem; j++) {
                                map->types[idx2] = map->types[idx + j];
                                map->disps[idx2] = map->disps[idx + j] + off2 + extent * i;
                                idx2++;
                            }
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_STRUCT) {
                MPI_Aint *p_blkl = p_counts + 1;
                MPI_Aint *p_disp = p_counts + 1 + p_counts[0];
                MPI_Aint k;
                MPI_Aint i;
                MPI_Aint j;
                MPI_Aint off2;
                MPI_Aint idx2 = idx;
                for (int k = 0; k < p_counts[0]; k++) {
                    type_get_n_elem_extent(p_types[k], &n_elem, &extent);
                    typemap_set(map, p_types[k], idx2, offset + p_disp[k]);
                    idx2 += n_elem;
                    for (int i = 1; i < p_blkl[k]; i++) {
                        for (int j = 0; j < n_elem; j++) {
                            map->types[idx2] = map->types[idx + j];
                            map->disps[idx2] = map->disps[idx + j] + extent * i;
                            idx2++;
                        }
                    }
                }
            } else if (cp->combiner == MPI_COMBINER_SUBARRAY) {
                int order = p_ints[2];
                MPI_Aint *p_sizes = p_counts;
                MPI_Aint *p_subsizes = p_counts + p_ints[0];
                MPI_Aint *p_starts = p_counts + p_ints[0] * 2;
                MPI_Aint off0 = 0;
                if (order == MPI_ORDER_C) {
                    for (int i = 0; i < p_ints[0]; i++) {
                        off0 = (off0 * p_sizes[i]) + p_starts[i];
                    }
                } else {
                    for (int i = p_ints[0] - 1; i >= 0; i--) {
                        off0 = (off0 * p_sizes[i]) + p_starts[i];
                    }
                }
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + off0);

                MPI_Aint off2 = off0;
                MPI_Aint idx2 = idx + n_elem;
                MPI_Aint *counters;
                counters = MPL_calloc(p_ints[0], sizeof(MPI_Aint), MPL_MEM_OTHER);
                while (1) {
                    int alldone = 0;
                    MPI_Aint stride = 1;
                    int i;
                    if (order == MPI_ORDER_C) {
                        i = p_ints[0] - 1;
                    } else {
                        i = 0;
                    }
                    while (1) {
                        counters[i]++;
                        off2 += stride;
                        if (counters[i] < p_subsizes[i]) {
                            break;
                        }
                        off2 -= stride * p_subsizes[i];
                        counters[i] = 0;
                        stride *= p_sizes[i];
                        if (order == MPI_ORDER_C) {
                            i--;
                            if (i < 0) {
                                alldone = 1;
                                break;
                            }
                        } else {
                            i++;
                            if (i == p_ints[0]) {
                                alldone = 1;
                                break;
                            }
                        }
                    }
                    if (alldone) {
                        break;
                    }
                    for (int j = 0; j < n_elem; j++) {
                        map->types[idx2] = map->types[idx + j];
                        map->disps[idx2] = map->disps[idx + j] + off2 - off0;
                        idx2++;
                    }
                }
                MPL_free(counters);
            } else if (cp->combiner == MPI_COMBINER_DARRAY) {
                int size = p_ints[0];
                int rank = p_ints[1];
                int n = p_ints[2];
                MPI_Aint *p_gsizes = p_counts;
                int *p_distribs = p_ints + 3;
                int *p_dargs = p_ints + 3 + n;
                int *p_sizes = p_ints + 3 + n * 2;
                int order = p_ints[3 + n * 3];

                int *p_prank;
                p_prank = MPL_malloc(n * sizeof(int), MPL_MEM_OTHER);
                int *p_isblk;
                p_isblk = MPL_malloc(n * sizeof(int), MPL_MEM_OTHER);
                MPI_Aint *p_parg;
                p_parg = MPL_malloc(n * sizeof(MPI_Aint), MPL_MEM_OTHER);
                MPI_Aint *p_starts;
                p_starts = MPL_malloc(n * sizeof(MPI_Aint), MPL_MEM_OTHER);
                for (int i = 0; i < n; i++) {
                    size /= p_sizes[i];
                    p_prank[i] = rank / size;
                    rank = rank % size;
                    if (p_sizes[i] == 1) {
                        p_isblk[i] = 1;
                        p_parg[i] = p_gsizes[i];
                        p_starts[i] = 0;
                    } else if (p_distribs[i] == MPI_DISTRIBUTE_BLOCK &&
                               p_dargs[i] == MPI_DISTRIBUTE_DFLT_DARG) {
                        p_parg[i] = p_gsizes[i] / p_sizes[i];
                        if (p_gsizes[i] % p_sizes[i] == 0) {
                            p_isblk[i] = 1;
                            p_starts[i] = p_parg[i] * p_prank[i];
                        } else {
                            int r = p_gsizes[i] % p_sizes[i];
                            p_isblk[i] = 2;
                            if (p_prank[i] < r) {
                                p_parg[i] += 1;
                                p_starts[i] = p_parg[i] * p_prank[i];
                            } else {
                                p_starts[i] = p_parg[i] * p_prank[i] + r;
                            }
                        }
                    } else if (p_distribs[i] == MPI_DISTRIBUTE_CYCLIC &&
                               p_dargs[i] == MPI_DISTRIBUTE_DFLT_DARG) {
                        p_isblk[i] = 0;
                        p_parg[i] = 1;
                        p_starts[i] = p_prank[i];
                    } else {
                        p_isblk[i] = 0;
                        p_parg[i] = p_dargs[i];
                        p_starts[i] = p_parg[i] * p_prank[i];
                    }
                }

                MPI_Aint off0 = 0;
                if (order == MPI_ORDER_C) {
                    for (int i = 0; i < n; i++) {
                        off0 = (off0 * p_gsizes[i]) + p_starts[i];
                    }
                } else {
                    for (int i = n - 1; i >= 0; i--) {
                        off0 = (off0 * p_gsizes[i]) + p_starts[i];
                    }
                }
                type_get_n_elem_extent(p_types[0], &n_elem, &extent);
                typemap_set(map, p_types[0], idx, offset + off0);

                MPI_Aint off2 = off0;
                MPI_Aint idx2 = idx + n_elem;
                MPI_Aint *counters;
                counters = MPL_calloc(n, sizeof(MPI_Aint), MPL_MEM_OTHER);
                if (p_isblk[0] == 0) {
                    counters[0] = p_prank[0] * p_parg[0];
                }
                while (1) {
                    int alldone = 0;
                    MPI_Aint stride = 1;
                    int i;
                    if (order == MPI_ORDER_C) {
                        i = n - 1;
                    } else {
                        i = 0;
                    }
                    while (1) {
                        if (p_isblk[i] == 0) {
                            MPI_Aint old_counter = counters[i];
                            counters[i]++;
                            if (counters[i] % p_parg[i] == 0) {
                                counters[i] += p_parg[i] * (p_sizes[i] - 1);
                            }
                            if (counters[i] < p_gsizes[i]) {
                                off2 += stride * (counters[i] - old_counter);
                                break;
                            }
                            counters[i] = p_starts[i];
                            off2 += stride * (counters[i] - old_counter);
                        } else {
                            counters[i]++;
                            off2 += stride;
                            if (counters[i] < p_parg[i]) {
                                break;
                            }
                            off2 -= stride * p_parg[i];
                            counters[i] = 0;
                        }
                        stride *= p_gsizes[i];
                        if (order == MPI_ORDER_C) {
                            i--;
                            if (i < 0) {
                                alldone = 1;
                                break;
                            }
                        } else {
                            i++;
                            if (i == n) {
                                alldone = 1;
                                break;
                            }
                        }
                    }
                    if (alldone) {
                        break;
                    }
                    for (int j = 0; j < n_elem; j++) {
                        map->types[idx2] = map->types[idx + j];
                        map->disps[idx2] = map->disps[idx + j] + off2 - off0;
                        idx2++;
                    }
                }
                MPL_free(counters);

                MPL_free(p_prank);
                MPL_free(p_isblk);
                MPL_free(p_parg);
                MPL_free(p_starts);
            } else {
                MPIR_Assert(0);
            }

        }
    }

}

void typesig_set(struct typesig *sig, MPI_Datatype dt, MPI_Aint * p_idx, MPI_Aint count)
{
    MPI_Aint idx = *p_idx;
    int *p_ints;
    MPI_Aint *p_aints;
    MPI_Aint *p_counts;
    MPI_Datatype *p_types;

    if (HANDLE_IS_BUILTIN(dt)) {
        typesig_check_space(sig, idx + 1);
        if (dt == MPI_2INT) {
            sig->types[idx] = MPI_INT;
            sig->counts[idx] = 2 * count;
        } else {
            sig->types[idx] = dt;
            sig->counts[idx] = count;
        }
        *p_idx = idx + 1;
        return;
    } else if (MPIR_DATATYPE_IS_PREDEFINED(dt)) {
        typesig_check_space(sig, idx + 2 * count);
        MPI_Datatype dt_a;
        if (dt == MPI_FLOAT_INT) {
            dt_a = MPI_FLOAT;
        }
        if (dt == MPI_DOUBLE_INT) {
            dt_a = MPI_DOUBLE;
        }
        if (dt == MPI_LONG_INT) {
            dt_a = MPI_LONG;
        }
        if (dt == MPI_SHORT_INT) {
            dt_a = MPI_SHORT;
        }
        for (int i = 0; i < count; i++) {
            sig->types[idx] = dt_a;
            sig->types[idx + 1] = MPI_INT;
            sig->counts[idx] = 1;
            sig->counts[idx + 1] = 1;
            idx += 2;
        }
        *p_idx = idx;
    } else {
        MPIR_Datatype *dt_ptr;
        MPIR_Datatype_get_ptr(dt, dt_ptr);
        MPIR_Assert(dt_ptr != NULL);
        if (dt_ptr->basic_type != MPI_DATATYPE_NULL) {
            typesig_check_space(sig, idx + 1);
            sig->types[idx] = dt_ptr->basic_type;
            sig->counts[idx] = dt_ptr->n_builtin_elements * count;
            *p_idx = idx + 1;
            return;
        } else {
            MPIR_Datatype_contents *cp = dt_ptr->contents;
            MPIR_Datatype_access_contents(cp, &p_ints, &p_aints, &p_counts, &p_types);
            if (cp->nr_counts == 0) {
                if (cp->combiner == MPI_COMBINER_DUP) {
                    typesig_set(sig, p_types[0], p_idx, count);
                } else if (cp->combiner == MPI_COMBINER_RESIZED) {
                    typesig_set(sig, p_types[0], p_idx, count);
                } else if (cp->combiner == MPI_COMBINER_CONTIGUOUS) {
                    typesig_set(sig, p_types[0], p_idx, count * p_ints[0]);
                } else if (cp->combiner == MPI_COMBINER_VECTOR) {
                    typesig_set(sig, p_types[0], p_idx, count * p_ints[0] * p_ints[1]);
                } else if (cp->combiner == MPI_COMBINER_HVECTOR) {
                    typesig_set(sig, p_types[0], p_idx, count * p_ints[0] * p_ints[1]);
                } else if (cp->combiner == MPI_COMBINER_INDEXED_BLOCK) {
                    MPI_Aint blkl_sum = 0;
                    for (int i = 0; i < p_ints[0]; i++) {
                        blkl_sum += p_ints[1];
                    }
                    typesig_set(sig, p_types[0], p_idx, count * blkl_sum);
                } else if (cp->combiner == MPI_COMBINER_HINDEXED_BLOCK) {
                    MPI_Aint blkl_sum = 0;
                    for (int i = 0; i < p_ints[0]; i++) {
                        blkl_sum += p_ints[1];
                    }
                    typesig_set(sig, p_types[0], p_idx, count * blkl_sum);
                } else if (cp->combiner == MPI_COMBINER_INDEXED) {
                    int *p_blkl = p_ints + 1;
                    int *p_disp = p_ints + 1 + p_ints[0];
                    MPI_Aint blkl_sum = 0;
                    for (int i = 0; i < p_ints[0]; i++) {
                        blkl_sum += p_blkl[i];
                    }
                    typesig_set(sig, p_types[0], p_idx, count * blkl_sum);
                } else if (cp->combiner == MPI_COMBINER_HINDEXED) {
                    int *p_blkl = p_ints + 1;
                    MPI_Aint *p_disp = p_aints;
                    MPI_Aint blkl_sum = 0;
                    for (int i = 0; i < p_ints[0]; i++) {
                        blkl_sum += p_blkl[i];
                    }
                    typesig_set(sig, p_types[0], p_idx, count * blkl_sum);
                } else if (cp->combiner == MPI_COMBINER_STRUCT) {
                    int *p_blkl = p_ints + 1;
                    MPI_Aint *p_disp = p_aints;
                    MPI_Aint i;
                    MPI_Aint j;
                    MPI_Aint idx_save = *p_idx;
                    MPI_Aint idx_last = *p_idx;
                    for (int i = 0; i < p_ints[0]; i++) {
                        typesig_set(sig, p_types[i], p_idx, p_blkl[i]);
                        if (idx_last > 0 && sig->types[idx_last - 1] == sig->types[idx_last]) {
                            sig->counts[idx_last - 1] += sig->counts[idx_last];
                            for (int j = idx_last; j < (*p_idx - 1); j++) {
                                sig->types[j] = sig->types[j + 1];
                                sig->counts[j] = sig->counts[j + 1];
                            }
                            (*p_idx)--;
                        }
                        idx_last = *p_idx;
                    }
                    if (count > 1) {
                        MPI_Aint num = *p_idx - idx_save;
                        typesig_check_space(sig, idx_save + count * num);
                        idx = *p_idx;
                        for (int i = 1; i < count; i++) {
                            for (int j = 0; j < num; j++) {
                                sig->types[idx] = sig->types[idx_save + j];
                                sig->counts[idx] = sig->counts[idx_save + j];
                                idx++;
                            }
                        }
                        *p_idx = idx;
                    }
                } else if (cp->combiner == MPI_COMBINER_SUBARRAY) {
                    int order = p_ints[1 + p_ints[0] * 3];
                    int *p_sizes = p_ints + 1;
                    int *p_subsizes = p_ints + 1 + p_ints[0];
                    int *p_starts = p_ints + 1 + p_ints[0] * 2;
                    MPI_Aint num = 1;
                    for (int i = 0; i < p_ints[0]; i++) {
                        num *= p_subsizes[i];
                    }
                    typesig_set(sig, p_types[0], p_idx, num);
                } else if (cp->combiner == MPI_COMBINER_DARRAY) {
                    int size = p_ints[0];
                    int rank = p_ints[1];
                    int n = p_ints[2];
                    int *p_gsizes = p_ints + 3;
                    int *p_distribs = p_ints + 3 + n;
                    int *p_dargs = p_ints + 3 + n * 2;
                    int *p_sizes = p_ints + 3 + n * 3;
                    int order = p_ints[3 + n * 4];

                    int *p_prank;
                    p_prank = MPL_malloc(n * sizeof(int), MPL_MEM_OTHER);
                    int *p_isblk;
                    p_isblk = MPL_malloc(n * sizeof(int), MPL_MEM_OTHER);
                    MPI_Aint *p_parg;
                    p_parg = MPL_malloc(n * sizeof(MPI_Aint), MPL_MEM_OTHER);
                    MPI_Aint *p_starts;
                    p_starts = MPL_malloc(n * sizeof(MPI_Aint), MPL_MEM_OTHER);
                    for (int i = 0; i < n; i++) {
                        size /= p_sizes[i];
                        p_prank[i] = rank / size;
                        rank = rank % size;
                        if (p_sizes[i] == 1) {
                            p_isblk[i] = 1;
                            p_parg[i] = p_gsizes[i];
                            p_starts[i] = 0;
                        } else if (p_distribs[i] == MPI_DISTRIBUTE_BLOCK &&
                                   p_dargs[i] == MPI_DISTRIBUTE_DFLT_DARG) {
                            p_parg[i] = p_gsizes[i] / p_sizes[i];
                            if (p_gsizes[i] % p_sizes[i] == 0) {
                                p_isblk[i] = 1;
                                p_starts[i] = p_parg[i] * p_prank[i];
                            } else {
                                int r = p_gsizes[i] % p_sizes[i];
                                p_isblk[i] = 2;
                                if (p_prank[i] < r) {
                                    p_parg[i] += 1;
                                    p_starts[i] = p_parg[i] * p_prank[i];
                                } else {
                                    p_starts[i] = p_parg[i] * p_prank[i] + r;
                                }
                            }
                        } else if (p_distribs[i] == MPI_DISTRIBUTE_CYCLIC &&
                                   p_dargs[i] == MPI_DISTRIBUTE_DFLT_DARG) {
                            p_isblk[i] = 0;
                            p_parg[i] = 1;
                            p_starts[i] = p_prank[i];
                        } else {
                            p_isblk[i] = 0;
                            p_parg[i] = p_dargs[i];
                            p_starts[i] = p_parg[i] * p_prank[i];
                        }
                    }
                    MPI_Aint num = 1;
                    for (int i = 0; i < n; i++) {
                        if (p_isblk[i] == 0) {
                            MPI_Aint num_this;
                            MPI_Aint n_blks;
                            MPI_Aint n_groups;
                            n_blks = p_gsizes[i] / p_parg[i];
                            n_groups = n_blks / p_sizes[i];
                            num_this = n_groups * p_parg[i];
                            if (p_prank[i] < n_blks % p_sizes[i]) {
                                num_this += p_prank[i];
                            }
                            if (p_prank[i] == n_blks % p_sizes[i]) {
                                num_this += p_gsizes[i] % p_parg[i];
                            }
                            num *= num_this;
                        } else if (p_isblk[i] == 1) {
                            num *= p_gsizes[i] / p_sizes[i];
                        } else {
                            if (p_prank[i] < p_gsizes[i] % p_sizes[i]) {
                                num *= p_gsizes[i] / p_sizes[i] + 1;
                            } else {
                                num *= p_gsizes[i] / p_sizes[i];
                            }
                        }
                    }
                    typesig_set(sig, p_types[0], p_idx, num);
                    MPL_free(p_prank);
                    MPL_free(p_isblk);
                    MPL_free(p_parg);
                    MPL_free(p_starts);
                } else {
                    MPIR_Assert(0);
                }

            } else {
                if (cp->combiner == MPI_COMBINER_DUP) {
                    typesig_set(sig, p_types[0], p_idx, count);
                } else if (cp->combiner == MPI_COMBINER_RESIZED) {
                    typesig_set(sig, p_types[0], p_idx, count);
                } else if (cp->combiner == MPI_COMBINER_CONTIGUOUS) {
                    typesig_set(sig, p_types[0], p_idx, count * p_counts[0]);
                } else if (cp->combiner == MPI_COMBINER_VECTOR) {
                    typesig_set(sig, p_types[0], p_idx, count * p_counts[0] * p_counts[1]);
                } else if (cp->combiner == MPI_COMBINER_HVECTOR) {
                    typesig_set(sig, p_types[0], p_idx, count * p_counts[0] * p_counts[1]);
                } else if (cp->combiner == MPI_COMBINER_INDEXED_BLOCK) {
                    MPI_Aint blkl_sum = 0;
                    for (int i = 0; i < p_counts[0]; i++) {
                        blkl_sum += p_counts[1];
                    }
                    typesig_set(sig, p_types[0], p_idx, count * blkl_sum);
                } else if (cp->combiner == MPI_COMBINER_HINDEXED_BLOCK) {
                    MPI_Aint blkl_sum = 0;
                    for (int i = 0; i < p_counts[0]; i++) {
                        blkl_sum += p_counts[1];
                    }
                    typesig_set(sig, p_types[0], p_idx, count * blkl_sum);
                } else if (cp->combiner == MPI_COMBINER_INDEXED) {
                    MPI_Aint *p_blkl = p_counts + 1;
                    MPI_Aint *p_disp = p_counts + 1 + p_counts[0];
                    MPI_Aint blkl_sum = 0;
                    for (int i = 0; i < p_counts[0]; i++) {
                        blkl_sum += p_blkl[i];
                    }
                    typesig_set(sig, p_types[0], p_idx, count * blkl_sum);
                } else if (cp->combiner == MPI_COMBINER_HINDEXED) {
                    MPI_Aint *p_blkl = p_counts + 1;
                    MPI_Aint *p_disp = p_counts + 1 + p_counts[0];
                    MPI_Aint blkl_sum = 0;
                    for (int i = 0; i < p_counts[0]; i++) {
                        blkl_sum += p_blkl[i];
                    }
                    typesig_set(sig, p_types[0], p_idx, count * blkl_sum);
                } else if (cp->combiner == MPI_COMBINER_STRUCT) {
                    MPI_Aint *p_blkl = p_counts + 1;
                    MPI_Aint *p_disp = p_counts + 1 + p_counts[0];
                    MPI_Aint i;
                    MPI_Aint j;
                    MPI_Aint idx_save = *p_idx;
                    MPI_Aint idx_last = *p_idx;
                    for (int i = 0; i < p_counts[0]; i++) {
                        typesig_set(sig, p_types[i], p_idx, p_blkl[i]);
                        if (idx_last > 0 && sig->types[idx_last - 1] == sig->types[idx_last]) {
                            sig->counts[idx_last - 1] += sig->counts[idx_last];
                            for (int j = idx_last; j < (*p_idx - 1); j++) {
                                sig->types[j] = sig->types[j + 1];
                                sig->counts[j] = sig->counts[j + 1];
                            }
                            (*p_idx)--;
                        }
                        idx_last = *p_idx;
                    }
                    if (count > 1) {
                        MPI_Aint num = *p_idx - idx_save;
                        typesig_check_space(sig, idx_save + count * num);
                        idx = *p_idx;
                        for (int i = 1; i < count; i++) {
                            for (int j = 0; j < num; j++) {
                                sig->types[idx] = sig->types[idx_save + j];
                                sig->counts[idx] = sig->counts[idx_save + j];
                                idx++;
                            }
                        }
                        *p_idx = idx;
                    }
                } else if (cp->combiner == MPI_COMBINER_SUBARRAY) {
                    int order = p_ints[2];
                    MPI_Aint *p_sizes = p_counts;
                    MPI_Aint *p_subsizes = p_counts + p_ints[0];
                    MPI_Aint *p_starts = p_counts + p_ints[0] * 2;
                    MPI_Aint num = 1;
                    for (int i = 0; i < p_ints[0]; i++) {
                        num *= p_subsizes[i];
                    }
                    typesig_set(sig, p_types[0], p_idx, num);
                } else if (cp->combiner == MPI_COMBINER_DARRAY) {
                    int size = p_ints[0];
                    int rank = p_ints[1];
                    int n = p_ints[2];
                    MPI_Aint *p_gsizes = p_counts;
                    int *p_distribs = p_ints + 3;
                    int *p_dargs = p_ints + 3 + n;
                    int *p_sizes = p_ints + 3 + n * 2;
                    int order = p_ints[3 + n * 3];

                    int *p_prank;
                    p_prank = MPL_malloc(n * sizeof(int), MPL_MEM_OTHER);
                    int *p_isblk;
                    p_isblk = MPL_malloc(n * sizeof(int), MPL_MEM_OTHER);
                    MPI_Aint *p_parg;
                    p_parg = MPL_malloc(n * sizeof(MPI_Aint), MPL_MEM_OTHER);
                    MPI_Aint *p_starts;
                    p_starts = MPL_malloc(n * sizeof(MPI_Aint), MPL_MEM_OTHER);
                    for (int i = 0; i < n; i++) {
                        size /= p_sizes[i];
                        p_prank[i] = rank / size;
                        rank = rank % size;
                        if (p_sizes[i] == 1) {
                            p_isblk[i] = 1;
                            p_parg[i] = p_gsizes[i];
                            p_starts[i] = 0;
                        } else if (p_distribs[i] == MPI_DISTRIBUTE_BLOCK &&
                                   p_dargs[i] == MPI_DISTRIBUTE_DFLT_DARG) {
                            p_parg[i] = p_gsizes[i] / p_sizes[i];
                            if (p_gsizes[i] % p_sizes[i] == 0) {
                                p_isblk[i] = 1;
                                p_starts[i] = p_parg[i] * p_prank[i];
                            } else {
                                int r = p_gsizes[i] % p_sizes[i];
                                p_isblk[i] = 2;
                                if (p_prank[i] < r) {
                                    p_parg[i] += 1;
                                    p_starts[i] = p_parg[i] * p_prank[i];
                                } else {
                                    p_starts[i] = p_parg[i] * p_prank[i] + r;
                                }
                            }
                        } else if (p_distribs[i] == MPI_DISTRIBUTE_CYCLIC &&
                                   p_dargs[i] == MPI_DISTRIBUTE_DFLT_DARG) {
                            p_isblk[i] = 0;
                            p_parg[i] = 1;
                            p_starts[i] = p_prank[i];
                        } else {
                            p_isblk[i] = 0;
                            p_parg[i] = p_dargs[i];
                            p_starts[i] = p_parg[i] * p_prank[i];
                        }
                    }
                    MPI_Aint num = 1;
                    for (int i = 0; i < n; i++) {
                        if (p_isblk[i] == 0) {
                            MPI_Aint num_this;
                            MPI_Aint n_blks;
                            MPI_Aint n_groups;
                            n_blks = p_gsizes[i] / p_parg[i];
                            n_groups = n_blks / p_sizes[i];
                            num_this = n_groups * p_parg[i];
                            if (p_prank[i] < n_blks % p_sizes[i]) {
                                num_this += p_prank[i];
                            }
                            if (p_prank[i] == n_blks % p_sizes[i]) {
                                num_this += p_gsizes[i] % p_parg[i];
                            }
                            num *= num_this;
                        } else if (p_isblk[i] == 1) {
                            num *= p_gsizes[i] / p_sizes[i];
                        } else {
                            if (p_prank[i] < p_gsizes[i] % p_sizes[i]) {
                                num *= p_gsizes[i] / p_sizes[i] + 1;
                            } else {
                                num *= p_gsizes[i] / p_sizes[i];
                            }
                        }
                    }
                    typesig_set(sig, p_types[0], p_idx, num);
                    MPL_free(p_prank);
                    MPL_free(p_isblk);
                    MPL_free(p_parg);
                    MPL_free(p_starts);
                } else {
                    MPIR_Assert(0);
                }

            }
        }
    }

}

void type_get_n_elem_extent(MPI_Datatype dt, MPI_Aint * p_n_elem, MPI_Aint * p_extent)
{
    if (HANDLE_IS_BUILTIN(dt)) {
        if (dt == MPI_2INT) {
            *p_n_elem = 2;
            *p_extent = MPIR_Datatype_get_basic_size(MPI_INT);
        } else {
            *p_n_elem = 1;
            *p_extent = MPIR_Datatype_get_basic_size(dt);
        }
        return;
    }
    MPIR_Datatype *dt_ptr;
    MPIR_Datatype_get_ptr(dt, dt_ptr);
    MPIR_Assert(dt_ptr != NULL);
    *p_n_elem = dt_ptr->n_builtin_elements;
    *p_extent = dt_ptr->extent;
}

void typesig_check_space(struct typesig *sig, MPI_Aint n)
{
    if (sig->n < n) {
        sig->n = n * 2;
        sig->types = MPL_realloc(sig->types, sig->n * sizeof(MPI_Datatype), MPL_MEM_OTHER);
        sig->counts = MPL_realloc(sig->counts, sig->n * sizeof(MPI_Aint), MPL_MEM_OTHER);
    }
}
