#pragma once
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fec_callbacks.h"

struct FecXORCfg {
    uint32_t payload_size;
    uint32_t num_original;
};

// create xor frame
void xor_frame(uint8_t* frames, uint8_t* xor_frame, size_t frames_cnt, size_t frame_len) {
    for (size_t i = 0; i < frames_cnt; ++i) {
        for (size_t j = 0; j < frame_len; ++j) {
            xor_frame[j] ^= frames[i * frame_len + j];
        }
    }
}

int xor_encode(void* cfg, void* state, uint8_t* original, uint8_t* recovery) {

    FecXORCfg* xcfg = (FecXORCfg*)cfg; 

    memset(recovery, 0, xcfg->payload_size);    
    xor_frame(original, recovery, xcfg->num_original, xcfg->payload_size); // have to be freed after sending

    return 0;
}

// expect lost indices sorted
int xor_decode(void* cfg, void* state, uint8_t* data, uint32_t* lost, uint32_t num_lost) {
    FecXORCfg* xcfg = (FecXORCfg*)cfg; 
    uint32_t k = xcfg->num_original;
    uint32_t frame_len = xcfg->payload_size;

    if (!num_lost) { // nothing lost
        return 0;
    }
    if (num_lost == 1 && lost[0] >= k) { // lost only control packets
        return 0; 
    }
    if (num_lost > 1) { // cannot decode
        return -1;
    }

    memset(data + lost[0] * frame_len, 0, frame_len);

    uint8_t* recovery = (uint8_t*)calloc(frame_len, sizeof(uint8_t));
    xor_frame(data, recovery, k + 1, frame_len);
    memcpy(data + lost[0] * frame_len, recovery, frame_len);
    free(recovery);

    return 0;
}


uint32_t xor_payload_size(void* cfg) {
    FecXORCfg* xcfg = (FecXORCfg*)cfg;
    return xcfg->payload_size;
}

uint32_t xor_num_original(void* cfg) {
    FecXORCfg* xcfg = (FecXORCfg*)cfg;
    return xcfg->num_original;
}


uint32_t xor_num_recovery(void* cfg) {
    return 1;
}

struct fec_callbacks xor_callbacks = {
    .init = NULL,
    .deinit = NULL,
    .payload_size = xor_payload_size,
    .num_original = xor_num_original, 
    .num_recovery = xor_num_recovery,
    .encode = xor_encode,
    .decode = xor_decode
};