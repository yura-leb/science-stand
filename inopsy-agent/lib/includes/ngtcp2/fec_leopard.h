#include <math.h>
#include <stdlib.h>

#include "fec_callbacks.h"
#include "fec_matrix_utils.h"
#include "leopard/leopard.h"

struct FecLeoCfg {
    // size of one packet payload
    uint32_t payload_size;
    // number of original payloads
    uint32_t num_original;
    // number of recovery payloads
    uint32_t num_recovery;
    // buffer size required by leopard
    // expecting payload_size % buf_size = 0
    uint32_t buf_size;
};

struct FecLeoState {
    uint32_t encode_buf_count;
    uint32_t decode_buf_count;
    uint8_t** encode_work_data_bufs;
    uint8_t** decode_work_data_bufs;
};

// l added to name to avoid name collision
int lleo_init(void* cfg, void** state) 
{
    FecLeoCfg* leo_cfg = (FecLeoCfg*)cfg;
    *state = NULL;

    if (leo_cfg->payload_size % leo_cfg->buf_size) {
        return -1; // cannot form payload from buffers
    }

    FecLeoState* leo_state = (FecLeoState*)calloc(1, sizeof(*leo_state));

    uint32_t num_original_bufs = leo_cfg->num_original * (leo_cfg->payload_size / leo_cfg->buf_size);
    uint32_t num_recovery_bufs = leo_cfg->num_recovery * (leo_cfg->payload_size / leo_cfg->buf_size);
    
    leo_state->encode_buf_count = leo_encode_work_count(num_original_bufs, num_recovery_bufs);
    leo_state->decode_buf_count = leo_decode_work_count(num_original_bufs, num_recovery_bufs);

    // TODO: use leo safe simd allocation?
    // allocating work data once instead of separately allocating each buffer
    // !!!: encode/decode expect work data to be allocated in one chunk
    uint8_t* encode_work_data = (uint8_t*)malloc(leo_state->encode_buf_count * leo_cfg->buf_size);
    uint8_t* decode_work_data = (uint8_t*)malloc(leo_state->decode_buf_count * leo_cfg->buf_size);
    uint8_t** encode_work_data_bufs = (uint8_t**)malloc(leo_state->encode_buf_count * sizeof(uint8_t*));
    uint8_t** decode_work_data_bufs = (uint8_t**)malloc(leo_state->decode_buf_count * sizeof(uint8_t*));
    for (int i = 0; i < leo_state->encode_buf_count; ++i) {
        encode_work_data_bufs[i] = &encode_work_data[i * leo_cfg->buf_size];
    }
    for (int i = 0; i < leo_state->decode_buf_count; ++i) {
        decode_work_data_bufs[i] = &decode_work_data[i * leo_cfg->buf_size];
    }
    leo_state->encode_work_data_bufs = encode_work_data_bufs;
    leo_state->decode_work_data_bufs = decode_work_data_bufs;

    *state = leo_state;

    return leo_init();
}

void lleo_deinit(void* cfg, void** state)
{
    (void)cfg;

    if (*state == nullptr) return;
    FecLeoState* leo_state = (FecLeoState*)(*state);
    free(leo_state->encode_work_data_bufs[0]);
    free(leo_state->decode_work_data_bufs[0]);
    free(leo_state->encode_work_data_bufs);
    free(leo_state->decode_work_data_bufs);
    free(leo_state);
}

int lleo_encode(void* cfg, void* state, uint8_t* original, uint8_t* recovery) {
    FecLeoCfg* leo_cfg = (FecLeoCfg*)cfg; 
    FecLeoState* leo_state = (FecLeoState*)state;

    uint32_t num_bufs_in_payload = leo_cfg->payload_size / leo_cfg->buf_size;
    uint32_t num_original_bufs = leo_cfg->num_original * num_bufs_in_payload;
    uint8_t** original_bufs = (uint8_t**)malloc(num_original_bufs * sizeof(uint8_t*));
    for (int i = 0; i < num_original_bufs; ++i) {
        original_bufs[i] = &original[i * leo_cfg->buf_size];
    }
    
    int_least16_t res = leo_encode(
        leo_cfg->buf_size, 
        num_original_bufs, 
        leo_cfg->num_recovery * num_bufs_in_payload, 
        leo_state->encode_buf_count, 
        (void**)&original_bufs[0], 
        (void**)&leo_state->encode_work_data_bufs[0]
    );
    free(original_bufs);
    if (res != Leopard_Success) {
        printf("Leopard failed: %i\n", res);
        return res;
    }

    // !!!: expect work data to be allocated in one chunk
    memcpy(
        recovery, leo_state->encode_work_data_bufs[0], 
        leo_cfg->num_recovery * num_bufs_in_payload * leo_cfg->buf_size
    );
    return 0;
}

// expect lost indices sorted
// expect lost data to be filled in |data| with some garbage
int lleo_decode(void* cfg, void* state, uint8_t* data, uint32_t* lost, uint32_t num_lost) {
    FecLeoCfg* leo_cfg = (FecLeoCfg*)cfg; 
    FecLeoState* leo_state = (FecLeoState*)state;

    if (num_lost == 0) {
        return 0;
    }
    if (num_lost > leo_cfg->num_recovery) {
        return -1;
    }
    if (num_lost && lost[0] >= leo_cfg->num_original) {
        return 0;
    }
    
    uint32_t num_bufs_in_payload = leo_cfg->payload_size / leo_cfg->buf_size;

    uint32_t num_original_bufs = leo_cfg->num_original * num_bufs_in_payload;
    uint8_t** original_bufs = (uint8_t**)malloc(num_original_bufs * sizeof(uint8_t*));
    for (int i = 0; i < num_original_bufs; ++i) {
        original_bufs[i] = &data[i * leo_cfg->buf_size];
    }

    uint32_t num_recovery_bufs = leo_cfg->num_recovery * num_bufs_in_payload;
    uint8_t** recovery_bufs = (uint8_t**)malloc(num_recovery_bufs * sizeof(uint8_t*));
    for (int i = 0; i < num_recovery_bufs; ++i) {
        recovery_bufs[i] = &data[(i + num_original_bufs) * leo_cfg->buf_size];
    }

    uint32_t num_lost_bufs = num_lost * num_bufs_in_payload;
    for (int i = 0; i < num_lost; ++i) {
        for (int j = 0; j < num_bufs_in_payload; ++j) {
            if (lost[i] < leo_cfg->num_original) {
                original_bufs[lost[i] * num_bufs_in_payload + j] = NULL;
            } else {
                recovery_bufs[(lost[i] - leo_cfg->num_original) * num_bufs_in_payload + j] = NULL;
            }
        }
    }
    
    int_least16_t res = leo_decode(
        leo_cfg->buf_size, 
        num_original_bufs, 
        num_recovery_bufs, 
        leo_state->decode_buf_count, 
        (void**)&original_bufs[0],
        (void**)&recovery_bufs[0], 
        (void**)&leo_state->decode_work_data_bufs[0]
    );
    if (res != Leopard_Success) {
        free(original_bufs);
        free(recovery_bufs);
        return res;
    }

    for (int i = 0; i < num_lost; ++i) {
        // !!!: expect work data to be allocated in one chunk
        memcpy(
            &data[lost[i] * leo_cfg->payload_size],
            leo_state->decode_work_data_bufs[lost[i] * num_bufs_in_payload], 
            leo_cfg->payload_size
        );
    }
    free(original_bufs);
    free(recovery_bufs);

    return 0;
}


uint32_t leo_payload_size(void* cfg) {
    FecLeoCfg* leo_cfg = (FecLeoCfg*)cfg;
    return leo_cfg->payload_size;
}

uint32_t leo_num_original(void* cfg) {
FecLeoCfg* leo_cfg = (FecLeoCfg*)cfg;
    return leo_cfg->num_original;
}

uint32_t leo_num_recovery(void* cfg) {
    FecLeoCfg* leo_cfg = (FecLeoCfg*)cfg;
    return leo_cfg->num_recovery;
}

struct fec_callbacks leopard_callbacks = {
    .init = lleo_init,
    .deinit = lleo_deinit,
    .payload_size = leo_payload_size,
    .num_original = leo_num_original, 
    .num_recovery = leo_num_recovery,
    .encode = lleo_encode,
    .decode = lleo_decode
};