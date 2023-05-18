#include <math.h>
#include <stdlib.h>

#include "fec_callbacks.h"
#include "fec_matrix_utils.h"

struct FecRCfg {
    // size of one packet payload
    uint32_t payload_size;
    // number of original payloads
    uint32_t num_original;
};

struct FecRState {
    Matrix* utility_mx;
};

uint32_t payload_size(void* cfg) {
    FecRCfg* rcfg = (FecRCfg*)cfg;
    return rcfg->payload_size;
}

uint32_t num_original(void* cfg) {
    FecRCfg* rcfg = (FecRCfg*)cfg;
    return rcfg->num_original;
}

uint32_t num_recovery(void* cfg) {
    FecRCfg* rcfg = (FecRCfg*)cfg;
    return ceil(log2(rcfg->num_original)) + 2;
}


// create utility matrix G
Matrix* makeUtilMatrix(uint32_t t, uint32_t k) {
    Matrix* m = mx_alloc(t + 2, k);
    for (int i = 0; i < k; ++i) {
        mx_set(m, 0, i, 1);
        mx_set(m, t + 1, i, 1);
    }
    for (int col = 1; col < k; ++col) {
        int val = 1;
        for (int row = t; row >= 1; --row) {
            mx_set(m, row, col, (mx_get(m, row, col - 1) + val) % 2);
            if (val != 0 && mx_get(m, row, col) == 1) {
                val = 0;
            }
        }
    }
    return m;
}

int init(void* cfg, void** state) 
{
    FecRCfg* rcfg = (FecRCfg*)cfg; 

    FecRState* r_state = (FecRState*)malloc(sizeof(FecRState));
    
    uint32_t k = rcfg->num_original;   
    uint32_t t = ceil(log2(k));
    r_state->utility_mx = makeUtilMatrix(t, k);
    *state = r_state;

    return 0;
}

void deinit(void* cfg, void** state)
{
    (void)cfg;

    FecRState* r_state = (FecRState*)(*state);
    mx_free(r_state->utility_mx);
    free(r_state);
}

// multiply matrices using xor instead of plus
// TODO: can multiply faster using transposed matrix
void xormul(Matrix* m1, Matrix* m2, Matrix* res)
{ 
    for (auto i = 0; i < m1->num_rows; ++i) {
        for (auto j = 0; j < m2->num_cols; ++j) {
            mx_set(res, i, j, 0);
 
            for (auto k = 0; k < m1->num_cols; ++k) {
                mx_set(res, i, j, (
                    mx_get(res, i, j) ^ 
                    (mx_get(m1, i, k) * mx_get(m2, k, j))
                ));
            }
        }
    }
}

int encode(void* cfg, void* state, uint8_t* original, uint8_t* recovery) {
    FecRCfg* rcfg = (FecRCfg*)cfg; 
    FecRState* rstate = (FecRState*)state;

    memset(recovery, 0, num_recovery(cfg) * rcfg->payload_size);
    
    uint32_t k = rcfg->num_original;   
    uint32_t t = ceil(log2(k));
    Matrix* original_m = mx_from_buf(original, k, rcfg->payload_size);
    Matrix* recovery_m = mx_from_buf(recovery, t + 2, rcfg->payload_size);
    xormul(rstate->utility_mx, original_m, recovery_m);
    free(original_m);
    free(recovery_m);
    return 0;
}

// expect lost indices sorted
// expect lost data to be filled in |data| with some garbage    
int decode(void* cfg, void* state, uint8_t* data, uint32_t* lost, uint32_t num_lost) {
    FecRCfg* rcfg = (FecRCfg*)cfg; 
    FecRState* rstate = (FecRState*)state;

    int k = rcfg->num_original;
    int t = ceil(log2(k));

    if (!num_lost) { // nothing lost
        return 0;
    }
    if (num_lost >= 1 && lost[0] >= k) { // lost only control packets
        return 0; 
    }
    if (num_lost > 2 && lost[0] < k) { // cannot decode
        return -1;
    }

    for (int i = 0; i < num_lost; ++i) {
        memset(&data[lost[i] * rcfg->payload_size], 0, rcfg->payload_size);
    }

    int num_data_rows = k + t + 2;
    int num_data_cols = rcfg->payload_size;
    Matrix* data_mx = mx_from_buf(data, num_data_rows, num_data_cols);
    // page 56 item C.3: one data packet lost
    if (num_lost == 1 || (
        num_lost == 2 && lost[0] < k && lost[1] >= k
    )) {
        int parity = k; // index of used parity packet
        if (num_lost == 2 && lost[1] == k) {
            parity = k + t + 1;
        }
        for (int i = 0; i < num_data_cols; ++i) {
            int vxor = 0;
            for (int j = 0; j < k; ++j) {
                if (j == lost[0]) {
                    continue;
                }
                vxor ^= mx_get(data_mx, j, i);
            }
            mx_set(data_mx, lost[0], i, (mx_get(data_mx, parity, i) ^ vxor));
        }
        return 0;
    }
    // only one option left: 2 data packets lost
    auto i = 1;
    for (; i < t + 1; ++i) {
        if (mx_get(rstate->utility_mx, i, lost[0]) == 0 && 
            mx_get(rstate->utility_mx, i, lost[1]) == 1) {
            break;
        }
    }
    // recover lost[1] (j2 in report)
    for (int col = 0; col < num_data_cols; ++col) {
        int vxor = 0;
        for (int row = 0; row < k; ++row) {
            if (row == lost[1]) {
                continue;
            }
            vxor = vxor ^ (mx_get(data_mx, row, col) * mx_get(rstate->utility_mx, i, row));
        }
        mx_set(data_mx, lost[1], col, (mx_get(data_mx, k + i, col) ^ vxor));
    }
    // recover lost[0] like in 1PR
    for (int col = 0; col < num_data_cols; ++col) {
        int vxor = 0;
        for (int row = 0; row < k; ++row) {
            if (row == lost[0]) {
                continue;
            }
            vxor = vxor ^ mx_get(data_mx, row, col);
        }
        mx_set(data_mx, lost[0], col, (mx_get(data_mx, k, col) ^ vxor));
    }
    free(data_mx);
    return 0;
}

struct fec_callbacks rscheme_callbacks = {
    .init = init,
    .deinit = deinit,
    .payload_size = payload_size,
    .num_original = num_original, 
    .num_recovery = num_recovery,
    .encode = encode,
    .decode = decode
};