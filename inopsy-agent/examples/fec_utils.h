#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct vec {
  uint8_t *base;
  size_t len;
} vec;

/**
 * @brief Cut buffers for |bufs| so that joint length is |len|
 * 
 * @param bufs buffers of original data 
 * @param bufcnt number of buffers
 * @param len joint length of requested buffers
 * @param rescnt number of buffers in result
 * @return vec* requested buffers or null if not enough data
 */
vec* cut_len(vec* bufs, size_t bufcnt, size_t len, size_t* rescnt) {
    size_t tmp_len = 0;
    size_t last_buf_id; // id of last buffer in result
    vec* res = NULL;
    // count buffers to allocate exact size for result
    // could make result have |bufcnt| with some empty data
    for (size_t i = 0; i < bufcnt; ++i) {
        tmp_len += bufs[i].len;
        if (tmp_len >= len) {
            last_buf_id = i;
            break; // stop if data len is enough
        }
    }
    if (tmp_len < len) {
        return res;
    }
    res = (vec*)malloc((last_buf_id + 1) * sizeof(*res));
    for (size_t i = 0; i < last_buf_id; ++i) {
        res[i] = bufs[i];
    }
    res[last_buf_id].base = bufs[last_buf_id].base;
    res[last_buf_id].len = bufs[last_buf_id].len - (tmp_len - len);
    if (rescnt != NULL) {
        *rescnt = last_buf_id + 1;
    }
    return res;
}

/**
 * @brief Produce xor of data in |bufs| seen as |elemcnt| number of |elemlen| elements
 * 
 * @param bufs buffers of original data
 * @param bufcnt number of buffers
 * @param elemlen length of xor'ed element
 * @param elemcnt number of xor'ed elements
 * @return vec xor or empty vec if not enough data
 */
vec naive_xor(vec* bufs, size_t bufcnt, size_t elemlen, size_t elemcnt) {
    vec res;
    if (elemcnt == 0 || elemlen == 0) {
        res.base = NULL;
        res.len = 0;
        return res;
    }
    res.base = (uint8_t*)calloc(elemlen, sizeof(uint8_t));
    res.len = elemlen;
    size_t xor_id = 0;
    size_t xorcnt = 0;
    for (size_t i = 0; i < bufcnt; ++i) {
        for (size_t j = 0; j < bufs[i].len; ++j) {
            res.base[xor_id] ^= bufs[i].base[j];
            ++xor_id;
            if (xor_id == elemlen) {
                xor_id = 0;
                ++xorcnt;
                if (xorcnt == elemcnt) {
                    return res;
                }
            }
        }
    }
    free(res.base);
    res.base = NULL;
    res.len = 0;
    return res;
}

/**
 * @brief Flatten multiple buffers into one
 * 
 * @param bufs buffers of original data 
 * @param bufcnt number of buffers
 * @return vec 
 */
vec flatten_data(vec* bufs, size_t bufcnt) {
    size_t total_len = 0;
    for (size_t i = 0; i < bufcnt; ++i) {
        total_len += bufs[i].len;
    }
    vec res;
    res.base = (uint8_t*)malloc(total_len);
    res.len = total_len;
    size_t offset = 0;
    for (size_t i = 0; i < bufcnt; ++i) {
        memcpy(res.base + offset, bufs[i].base, bufs[i].len);
        offset += bufs[i].len;
    }
    return res;
}
