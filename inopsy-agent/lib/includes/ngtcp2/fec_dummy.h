#pragma once
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fec_callbacks.h"

struct FecDummyCfg {
    uint32_t payload_size;
    uint32_t num_original;
};

int dummy_encode(void* cfg, void* state, uint8_t* original, uint8_t* recovery) {
    return 0;
}

int dummy_decode(void* cfg, void* state, uint8_t* data, uint32_t* lost, uint32_t num_lost) {
    if (num_lost > 0) {
        return -1;
    } else {
        return 0;
    }
}

uint32_t dummy_payload_size(void* cfg) {
    FecDummyCfg* dcfg = (FecDummyCfg*)cfg;
    return dcfg->payload_size;
}

uint32_t dummy_num_original(void* cfg) {
    FecDummyCfg* dcfg = (FecDummyCfg*)cfg;
    return dcfg->num_original;
}

uint32_t dummy_num_recovery(void* cfg) {
    return 0;
}

struct fec_callbacks dummy_callbacks = {
    .init = NULL,
    .deinit = NULL,
    .payload_size = dummy_payload_size,
    .num_original = dummy_num_original, 
    .num_recovery = dummy_num_recovery,
    .encode = dummy_encode,
    .decode = dummy_decode
};