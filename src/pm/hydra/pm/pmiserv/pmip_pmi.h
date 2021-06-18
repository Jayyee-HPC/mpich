/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef PMIP_PMI_H_INCLUDED
#define PMIP_PMI_H_INCLUDED

#include "hydra.h"
#include "common.h"

/* PMI-1 specific definitions */
extern struct HYD_pmcd_pmip_pmi_handle *HYD_pmcd_pmip_pmi_v1;

/* PMI-2 specific definitions */
extern struct HYD_pmcd_pmip_pmi_handle *HYD_pmcd_pmip_pmi_v2;

struct HYD_pmcd_pmip_pmi_handle {
    const char *cmd;
     HYD_status(*handler) (int fd, char *args[]);
};

extern struct HYD_pmcd_pmip_pmi_handle *HYD_pmcd_pmip_pmi_handle;

void HYD_cache_flush_temp(int fd);

#endif /* PMIP_PMI_H_INCLUDED */
