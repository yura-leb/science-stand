#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fec_callbacks.h"

struct Fec2PRCfg {
    uint32_t payload_size;
    uint32_t num_original;
};

uint32_t pr2_payload_size(void* cfg) {
    Fec2PRCfg* prcfg = (Fec2PRCfg*)cfg;
    return prcfg->payload_size;
}

uint32_t pr2_num_original(void* cfg) {
    Fec2PRCfg* prcfg = (Fec2PRCfg*)cfg;
    return prcfg->num_original;
}

uint32_t pr2_num_recovery(void* cfg) {
    return 2;
}

void xor_frames(uint8_t* first, uint8_t* second, uint32_t frame_len) {
    for (size_t j = 0; j < frame_len; ++j) {
        first[j] ^= second[j];
    }
}

// create 2 xor frames with even and odd indices
void xor_even_odd_frame(uint8_t* frames, uint8_t* xor_frame, size_t frames_cnt, size_t frame_len, bool is_even) {

    size_t xor_id = 0;
    size_t xorcnt = 0;
    for (size_t i = is_even; i < frames_cnt; i+=2) {
        xor_frames(xor_frame, frames + i * frame_len, frame_len);
    }
}


int pr2_encode(void* cfg, void* state, uint8_t* original, uint8_t* recovery) {

    Fec2PRCfg* prcfg = (Fec2PRCfg*)cfg; 
    uint32_t frame_len = prcfg->payload_size;
    uint32_t k = prcfg->num_original;
    
    if (k == 0 || frame_len == 0 || recovery == NULL) {
        return 1;
    }

    memset(recovery, 0, pr2_num_recovery(cfg) * prcfg->payload_size);
    xor_even_odd_frame(original, recovery, prcfg->num_original, frame_len, 0);
    if (k > 1) {
        xor_even_odd_frame(original, recovery + frame_len, prcfg->num_original, frame_len, 1);
    }

    return 0;
}


// expect lost indices sorted
int pr2_decode(void* cfg, void* state, uint8_t* data, uint32_t* lost, uint32_t num_lost) {

    Fec2PRCfg* prcfg = (Fec2PRCfg*)cfg; 
    uint32_t k = prcfg->num_original;
    uint32_t frame_len = prcfg->payload_size;

    if (!num_lost) { // nothing lost
        return 0;
    }
    if (num_lost <= 2 && lost[0] >= k) { // lost only control packets
        return 0; 
    }
    if (num_lost > 2) { // cannot decode
        return -1;
    }
    if (num_lost == 2 && lost[1] < k && lost[0] % 2 == lost[1] % 2) { // cannot decode, equal parity
        return -1;
    }
    // due to our encode algorithm 1st recovery packet is for even and 2nd for odd
    // recovery packet number parity depends on num_original so we substract it
    if (num_lost == 2 && lost[1] >= k && lost[0] % 2 == (lost[1] - k) % 2) { // cannot decode, equal parity
        return -1;
    }

    for (uint32_t i = 0; i < num_lost; ++i) {
        memset(data + lost[i] * frame_len, 0, frame_len);
    }

    uint8_t* recovered = (uint8_t*)calloc(frame_len, sizeof(uint8_t));
    
    bool is_even = lost[0] % 2; // 0 index is odd, 1 index is even
    xor_even_odd_frame(data, recovered, k, frame_len, is_even);
    xor_frames(recovered, data + (k + is_even) * frame_len, frame_len);
    memcpy(data + lost[0] * frame_len, recovered, frame_len);

    if (num_lost == 2 && lost[1] < k) {
        for (int i = 0; i < frame_len; ++i) {
            recovered[i] = 0;
        }
        is_even = lost[1] % 2; 
        xor_even_odd_frame(data, recovered, k, frame_len, is_even);
        xor_frames(recovered, data + (k + is_even) * frame_len, frame_len);
        memcpy(data + lost[1] * frame_len, recovered, frame_len);
    }

    free(recovered);

    return 0;
}

struct fec_callbacks pr2_callbacks = {
    .init = NULL,
    .deinit = NULL,
    .payload_size = pr2_payload_size,
    .num_original = pr2_num_original, 
    .num_recovery = pr2_num_recovery,
    .encode = pr2_encode,
    .decode = pr2_decode
};