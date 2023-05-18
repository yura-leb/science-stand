#ifndef FEC_CALLBACKS_H
#define FEC_CALLBACKS_H

#include <stdint.h>

typedef int (*fec_init)(void *cfg, void **state);
typedef void (*fec_deinit)(void *cfg, void **state);
typedef uint32_t (*fec_payload_size)(void *cfg);
typedef uint32_t (*fec_num_original)(void *cfg);
typedef uint32_t (*fec_num_recovery)(void *cfg);
typedef int (*fec_encode)(void *cfg, void *state, uint8_t *original, uint8_t *recovery);
typedef int (*fec_decode)(void *cfg, void *state, uint8_t *data, uint32_t *lost_indices, uint32_t num_lost);

typedef struct fec_callbacks {
    fec_init init;
    fec_deinit deinit;
    fec_payload_size payload_size;
    fec_num_original num_original;
    fec_num_recovery num_recovery;
    fec_encode encode;
    fec_decode decode;
} fec_callbacks;

#endif