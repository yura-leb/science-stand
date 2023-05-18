#ifndef FEC_MATRIX16_UTILS_H
#define FEC_MATRIX16_UTILS_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef struct Matrix16 {
    uint32_t num_rows;
    uint32_t num_cols;
    uint16_t* data;
} Matrix16;

Matrix16* mx16_alloc(uint32_t num_rows, uint32_t num_cols) {
    Matrix16* m = (Matrix16*)malloc(sizeof(*m));
    m->num_rows = num_rows;
    m->num_cols = num_cols;
    m->data = (uint16_t*)calloc(num_rows * num_cols, sizeof(*m->data));
    return m;
}

Matrix16* mx16_from_buf(uint16_t* buf, uint32_t rows, uint32_t cols) {
    Matrix16* m = (Matrix16*)malloc(sizeof(*m));
    m->num_rows = rows;
    m->num_cols = cols;
    m->data = buf;
    return m;
}

inline void mx16_set(Matrix16* m, uint32_t row, uint32_t col, uint16_t value) {
    assert(m != NULL && row < m->num_rows && col < m->num_cols);
    m->data[row * m->num_cols + col] = value;
}

inline uint16_t mx16_get(Matrix16* m, uint32_t row, uint32_t col) {
    assert(m != NULL && row < m->num_rows && col < m->num_cols);
    return m->data[row * m->num_cols + col];
}

// check rows and cols from 0 to |rows|/|cols| of matrices |a| and |b| are equal
bool mx16_check_equal(Matrix16* a, Matrix16* b, uint32_t num_rows, uint32_t num_cols) {
    for (auto i = 0; i < num_rows; ++i) {
        for (auto j = 0; j < num_cols; ++j) {
            if (mx16_get(a, i, j) != mx16_get(b, i, j)) {
                return false;
            }
        }
    }
    return true;
}

// erase rows from matrix |mx| with indices from |rows|
void mx16_erase_rows(Matrix16* mx, uint32_t* rows, uint32_t num_rows) {
    for (int i = 0; i < num_rows; ++i) {
        assert(rows[i] < mx->num_rows);
        for (auto j = 0; j < mx->num_cols; ++j) {
            mx16_set(mx, i, j, 0);
        }
    }
}

void mx16_free(Matrix16* m) {
    free(m->data);
    free(m);
}

void mx16_print(Matrix16* m) {
    for (int i = 0; i < m->num_rows; ++i) {
        for (int j = 0; j < m->num_cols; ++j) {
            printf("%u ", mx16_get(m, i, j));
        }
        printf("\n");
    }
}

#endif
