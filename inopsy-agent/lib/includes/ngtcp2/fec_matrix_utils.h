#ifndef FEC_MATRIX_UTILS_H
#define FEC_MATRIX_UTILS_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef struct Matrix {
    uint32_t num_rows;
    uint32_t num_cols;
    uint8_t* data;
} Matrix;

Matrix* mx_alloc(uint32_t num_rows, uint32_t num_cols) {
    Matrix* m = (Matrix*)malloc(sizeof(*m));
    m->num_rows = num_rows;
    m->num_cols = num_cols;
    m->data = (uint8_t*)calloc(num_rows * num_cols, sizeof(uint8_t));
    return m;
}

Matrix* mx_from_buf(uint8_t* buf, uint32_t rows, uint32_t cols) {
    Matrix* m = (Matrix*)malloc(sizeof(*m));
    m->num_rows = rows;
    m->num_cols = cols;
    m->data = buf;
    return m;
}

inline void mx_set(Matrix* m, uint32_t row, uint32_t col, uint8_t value) {
    assert(m != NULL && row < m->num_rows && col < m->num_cols);
    m->data[row * m->num_cols + col] = value;
}

inline uint8_t mx_get(Matrix* m, uint32_t row, uint32_t col) {
    assert(m != NULL && row < m->num_rows && col < m->num_cols);
    return m->data[row * m->num_cols + col];
}

// check rows and cols from 0 to |rows|/|cols| of matrices |a| and |b| are equal
bool mx_check_equal(Matrix* a, Matrix* b, uint32_t num_rows, uint32_t num_cols) {
    for (auto i = 0; i < num_rows; ++i) {
        for (auto j = 0; j < num_cols; ++j) {
            if (mx_get(a, i, j) != mx_get(b, i, j)) {
                return false;
            }
        }
    }
    return true;
}

// erase rows from matrix |mx| with indices from |rows|
void mx_erase_rows(Matrix* mx, uint32_t* rows, uint32_t num_rows) {
    for (int i = 0; i < num_rows; ++i) {
        assert(rows[i] < mx->num_rows);
        for (auto j = 0; j < mx->num_cols; ++j) {
            mx_set(mx, i, j, 0);
        }
    }
}

void mx_free(Matrix* m) {
    free(m->data);
    free(m);
}

void mx_print(Matrix* m) {
    for (int i = 0; i < m->num_rows; ++i) {
        for (int j = 0; j < m->num_cols; ++j) {
            printf("%u ", mx_get(m, i, j));
        }
        printf("\n");
    }
}

#endif