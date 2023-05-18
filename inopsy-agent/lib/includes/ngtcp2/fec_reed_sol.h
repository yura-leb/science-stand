#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "fec_callbacks.h"
#include "fec_matrix16_utils.h"

const int RS_NUM_SYMBOLS = 65535;
const int RS_SYMBOL_SIZE = 2;

struct FecRSCfg {
    uint32_t payload_size;
    uint32_t num_original;
};

struct FecRSState {
    // tables for fast multiplication/division in Galois field
    uint16_t* deg_to_val;
    uint16_t* val_to_deg;
    // utility matrix P
    Matrix16* P;
};

static inline uint16_t rotl16(uint16_t n, unsigned int c)
{
    const uint16_t mask = (CHAR_BIT*sizeof(n) - 1);
    c &= mask;
    return (n<<c) | (n>>( (-c)&mask ));
}

static inline uint16_t rotr16(uint16_t n, unsigned int c)
{
    const uint16_t mask = (CHAR_BIT*sizeof(n) - 1);
    c &= mask;
    return (n>>c) | (n<<( (-c)&mask ));
}

// irreducible polynomial x^16 + x^12 + x^3 + x + 1 can be represented as 69643 (10001000000001011)
const uint32_t IRR_POLY = 69643;
// for primitive element alpha in GF(2^16) alpha^0 == alpha^ZERO_DEGREE_16
const uint16_t ZERO_DEGREE_16 = 65535;

// multiply 2 values in finite field GF(2^16)
uint16_t fin_mul(uint16_t lhs, uint16_t rhs) {
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    uint32_t result = 0; // using 32 bits so that we can shift without overflow
    for (int i = 0; i < 16; ++i) {
        if ((lhs>>i) & 1) { // ith bit from end is 1
            result ^= (rhs << i);
        }
    }
    // replace all x^k where k >= 16
    // starting from higher bits to correctly reduce degree
    for (int j = 15; j >= 0; --j) {
        if ((result >> (16 + j)) & 1) {
            result = result ^ (IRR_POLY << j); // decrease degrees using irreducible polynomial
        }
    }   
    assert(result <= USHRT_MAX);
    return result;
}

void fill_div_tables(
    uint16_t* deg_to_val,
    uint16_t* val_to_deg
) {
    uint16_t prim_elem = 2; 
    uint16_t val = 1;
    deg_to_val[0] = val;
    val_to_deg[val] = 0;
    for (int i = 0; i < USHRT_MAX; ++i) {
        val = fin_mul(val, prim_elem);
        deg_to_val[1 + i] = val;
        val_to_deg[val] = 1 + i;
    }
}

inline uint16_t fin_div(
    uint16_t lhs, uint16_t rhs, 
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    if (lhs == 0 || rhs == 0) {
        return 0; // invalid value for invalid inputs
    }
    int lhs_deg = val_to_deg[lhs];
    int rhs_deg = val_to_deg[rhs];
    int res_deg = (lhs_deg >= rhs_deg) ? lhs_deg - rhs_deg : USHRT_MAX - (rhs_deg - lhs_deg);
    return deg_to_val[res_deg];
}

// faster finite field multiplication using degrees
inline uint16_t fin_fmul(
    uint16_t lhs, uint16_t rhs, 
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    int lhs_deg = val_to_deg[lhs];
    int rhs_deg = val_to_deg[rhs];
    int res_deg = lhs_deg + rhs_deg;
    if (res_deg >= USHRT_MAX) {
        res_deg -= USHRT_MAX;
    }
    return deg_to_val[res_deg];
}

// can be replaced with assert of chosen test framework
void rt_assert(bool condition) {
    if (!condition) {
        printf("test failed\n");
        exit(-1);
    }
}

// multiply |polynomial| (array size |max_degree|) with degree < |max_degree| to x + alpha^k
void mul_polynomial(
    uint16_t* polynomial, uint32_t max_degree, 
    uint32_t k,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    assert(polynomial[max_degree] == 0);
    uint16_t result[max_degree + 1];
    memset(result, 0, sizeof(result));
    result[0] = fin_fmul(deg_to_val[k], polynomial[0], deg_to_val, val_to_deg);
    // TODO: try in another direction to make in place
    for (int i = 1; i <= max_degree; ++i) {
        // each coeff[i] is coeff[i] * alpha^k + coeff[i-1] 
        // with plus/multiplication from finite field
        result[i] = fin_fmul(deg_to_val[k], polynomial[i], deg_to_val, val_to_deg) ^ polynomial[i - 1];
    }
    for (int i = 0; i <= max_degree; ++i) {
        polynomial[i] = result[i];
    }
}

uint16_t calc_polynomial_val(
    uint16_t* polynomial, uint32_t max_degree,
    uint16_t x, 
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    uint16_t res = 0;
    uint32_t x_deg = val_to_deg[x]; // 2^x_deg == x
    for (int i = 0; i <= max_degree; ++i) {
        uint16_t add = polynomial[i];
        uint32_t temp_deg = x_deg * i % ZERO_DEGREE_16;
        add = fin_fmul(add, deg_to_val[temp_deg], deg_to_val, val_to_deg);

        res ^= add;
    }
    return res;
}

void calc_gen_polynomial(
    uint16_t* polynomial, uint32_t max_degree,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    polynomial[0] = 1;
    for (int i = 1; i <= max_degree; ++i) {
        mul_polynomial(polynomial, max_degree, i, deg_to_val, val_to_deg);
    }
}

// fill matrix with polynomial values for RS code
int mx16_fill_poly(Matrix16* m, uint16_t* poly, uint32_t max_degree) {
    // M - poly degree, N*K - matrix size, M + K = N
    if (m->num_cols != m->num_rows + max_degree) {
        return -1; // invalid matrix/polynomial sizes
    }
    for (int i = 0; i < m->num_rows; i++) {
        for (int j = 0; j <= max_degree; ++ j) {
            mx16_set(m, i, i + j, poly[j]);
        }
    }
    return 0;
}

// add |row_to_add|*|row_multiplier| to |dst_row|
void mx16_add_row_lin(
    Matrix16* m, uint32_t dst_row, uint32_t row_to_add, uint16_t row_multiplier,
    uint16_t* deg_to_val, uint16_t* val_to_deg    
) {
    for (int i = 0; i < m->num_cols; ++i) {
        uint16_t dst_val = mx16_get(m, dst_row, i);
        mx16_set(m, dst_row, i,
            dst_val ^ fin_fmul(mx16_get(m, row_to_add, i), row_multiplier, deg_to_val, val_to_deg)
        );
    }
}

// naive implementation is too slow (requires hours for 51000x65535)
// reduce matrix to [I, P] form
void mx16_reduce(
    Matrix16* m,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    assert(m->num_cols >= m->num_rows);
    for (int i = 0; i < m->num_rows; ++i) {
        // diagonal elements of matrix |m| are non-zero
        uint16_t multiplier = fin_div(1, mx16_get(m, i, i), deg_to_val, val_to_deg);
        for (int j = i; j < m->num_cols; ++j) {
            uint16_t val = mx16_get(m, i, j);
            if (val == 0) { // g0 ... gm 0 ... 0 and gi != 0
                break;
            }
            mx16_set(m, i, j, fin_fmul(val, multiplier, deg_to_val, val_to_deg));
        }
        for (int j = i + 1; j < m->num_rows; ++j) {
            multiplier = fin_div(mx16_get(m, i, j), mx16_get(m, j, j), deg_to_val, val_to_deg);
            mx16_add_row_lin(m, i, j, multiplier, deg_to_val, val_to_deg);
        }
    }
}

// this implementation uses fact that all rows in matrix have the same values shifted by 1 column
// to each row i we apply the same changes that we applied to (i+1)th (copy i+1 to i shifting by 1) 
// and (k-1)kth value will no be zero as for (i+1)th row it is in P part of [I, P]
// so we need to zero it using last row (as last row will have 1 only in (k-1)th col)
void mx16_fast_reduce(
    Matrix16* m,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    assert(m->num_cols >= m->num_rows);

    int i = m->num_rows - 1;
    // diagonal elements of matrix |m| are non-zero
    uint16_t multiplier = fin_div(1, mx16_get(m, i, i), deg_to_val, val_to_deg);
    for (int j = i; j < m->num_cols; ++j) {
        uint16_t val = mx16_get(m, i, j);
        if (val == 0) { // g0 ... gm 0 ... 0 and gi != 0
            break;
        }
        mx16_set(m, i, j, fin_fmul(val, multiplier, deg_to_val, val_to_deg));
    }

    for (i = i - 1; i >= 0; --i) {
        for (int j = i; j < m->num_cols - 1; ++j) {
            mx16_set(m, i, j, mx16_get(m, i + 1, j + 1));
        }
        multiplier = mx16_get(m, i, m->num_rows - 1);
        mx16_add_row_lin(m, i, m->num_rows - 1, multiplier, deg_to_val, val_to_deg);
    }
}

void mx16_xor_mul(
    Matrix16* m1, Matrix16* m2, Matrix16* res,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    assert(m1->num_cols == m2->num_rows);
    assert(m1->num_rows == res->num_rows);
    assert(m2->num_cols == res->num_cols);
     
    uint16_t tmp;
    for (int i = 0; i < m1->num_rows; ++i) {
        for (int j = 0; j < m2->num_cols; ++j) {
            tmp = 0;
 
            for (int k = 0; k < m1->num_cols; ++k) {
                tmp ^= fin_fmul(
                    mx16_get(m1, i, k), mx16_get(m2, k, j),
                    deg_to_val, val_to_deg
                );
            }
            mx16_set(res, i, j, tmp);
        }
    }
}

// produce additional symbols, original ones are not changed
int rs_encode(void* cfg, void* state, uint8_t* original, uint8_t* recovery) {
    FecRSCfg* rs_cfg = (FecRSCfg*)cfg;
    FecRSState* rs_state = (FecRSState*)state;

    uint32_t K = rs_cfg->payload_size * rs_cfg->num_original / RS_SYMBOL_SIZE;;
    uint32_t M = RS_NUM_SYMBOLS - K;

    Matrix16* original_mx = mx16_from_buf((uint16_t*)original, 1, K);
    Matrix16* recovery_mx = mx16_from_buf((uint16_t*)recovery, 1, M);
    mx16_xor_mul(original_mx, rs_state->P, recovery_mx, rs_state->deg_to_val, rs_state->val_to_deg);
    free(original_mx);
    free(recovery_mx);
    return 0;
}

// solve linear system Ax = b with square A
int linear_solve(
    Matrix16* A, Matrix16* b, Matrix16* x,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    if (A->num_cols != A->num_rows) {
        return -1; // works only for square matrices
    }
    uint32_t n = A->num_rows - 1;

    Matrix16* D = mx16_alloc(A->num_rows, A->num_cols + 1);
    for (int row = 0; row <= n; ++row) {
        for (int col = 0; col <= n; ++col) {
            mx16_set(D, row, col, mx16_get(A, row, col));
        }
        mx16_set(D, row, n + 1, mx16_get(b, row, 0));
    }
    for (int k = 0; k <= n; ++k) {
        uint16_t r = 0;
        uint16_t lp = 0;   
        for (int l = k; l <= n; ++l) {
            if (r < mx16_get(D, l, k)) { // ???: is it ok for finite field?
                lp = l;
                r = mx16_get(D, l, k);
            }
        }
        if (r == 0) {
            return -1; // zero determinant
        }
        if (k != lp) {
            for (int j = k; j <= n + 1; ++j) {
                r = mx16_get(D, k, j);
                mx16_set(D, k, j, mx16_get(D, lp, j));
                mx16_set(D, lp, j, r);
            }
        }
        for (int j = k + 1; j <= n + 1; ++j) {
            mx16_set(D, k, j, fin_div(
                mx16_get(D, k, j), mx16_get(D, k, k),
                deg_to_val, val_to_deg
            ));
        }

        // !!!: probably error in theory, so iterations are a bit different
        if (k != n) {
            for (int i = k + 1; i <= n; ++i) {
                for (int j = k + 1; j <= n + 1; ++j) {
                    mx16_set(D, i, j, 
                        mx16_get(D, i, j) ^ fin_fmul(
                            mx16_get(D, k, j), mx16_get(D, i, k),
                            deg_to_val, val_to_deg
                        )
                    );
                }
            }
        }
        if (k != 0) {
            for (int i = 0; i <= k - 1; ++i) {
                for (int j = k + 1; j <= n + 1; ++j) {
                    mx16_set(D, i, j, 
                        mx16_get(D, i, j) ^ fin_fmul(
                            mx16_get(D, k, j), mx16_get(D, i, k),
                            deg_to_val, val_to_deg
                        )
                    );
                }
            }
        }
    }
    for (int i = 0; i <= n; ++i) {
        mx16_set(x, i, 0, mx16_get(D, i, n + 1));        
    }
    return 0;
}

// check whether linear system Ax = b solution is correct
bool check_linear_solution(
    Matrix16* A, Matrix16* b, Matrix16* x,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    Matrix16* mul = mx16_alloc(b->num_rows, b->num_cols);
    mx16_xor_mul(A, x, mul, deg_to_val, val_to_deg);
    for (int row = 0; row < x->num_cols; ++x) {
        if (mx16_get(b, row, 0) != mx16_get(mul, row, 0)) {
            return false;
        }
    }
    mx16_free(mul);
    return true;
}

// expect lost positions in data to be filled with zeroes
// FIXME: does not recover data properly if some recovery symbols lost
int rs_decode(void* cfg, void* state, uint8_t* data, uint32_t* lost, uint32_t num_lost) {
    FecRSCfg* rs_cfg = (FecRSCfg*)cfg;
    FecRSState* rs_state = (FecRSState*)state;

    if (num_lost == 0) {
        return 0;
    }

    uint32_t num_symbols_in_payload = rs_cfg->payload_size / RS_SYMBOL_SIZE;
    uint32_t num_lost_symbols = num_lost * num_symbols_in_payload;
    uint32_t* lost_symbols = (uint32_t*)malloc(num_lost_symbols * sizeof(*lost_symbols));
    for (int i = 0; i < num_lost; ++i) {
        for (int j = 0; j < num_symbols_in_payload; ++j) {
            lost_symbols[i * num_symbols_in_payload + j] = lost[i] * num_symbols_in_payload + j;
        }
    }

    uint32_t k = rs_cfg->payload_size * rs_cfg->num_original / RS_SYMBOL_SIZE;
    uint32_t m = RS_NUM_SYMBOLS - k;
    uint16_t* deg_to_val = rs_state->deg_to_val;
    uint16_t* val_to_deg = rs_state->val_to_deg;
    Matrix16* P = rs_state->P;
    if (num_lost_symbols > m) {
        return -1; // cannot recover more than m symbols
    }
    // TODO: shortcut for case when only non-data symbols lost
    // fill A and b matrices for linear system
    Matrix16* A = mx16_alloc(num_lost_symbols, num_lost_symbols);
    Matrix16* b = mx16_alloc(num_lost_symbols, 1);
    for (int i = 0; i < num_lost_symbols; ++i) {
        for (int j = 0; j < num_lost_symbols; ++j) {
            if (lost_symbols[i] < k) { // P part
                mx16_set(A, j, i, mx16_get(P, lost_symbols[i], j));
            } else { // I part
                mx16_set(A, j, i, (lost_symbols[i] == j + k) ? 1 : 0);
            }
        }
    }

    Matrix16* data_mx = mx16_from_buf((uint16_t*)data, 1, k + m);
    for (int i = 0; i < num_lost_symbols; ++i) {
        uint16_t res = 0;
        for (int j = 0; j < data_mx->num_cols; ++j) {
            uint16_t coeff = (j < k) ? mx16_get(P, j, i) : ((j == i + k) ? 1 : 0); 
            res ^= fin_fmul(coeff, mx16_get(data_mx, 0, j), deg_to_val, val_to_deg);
        }
        
        mx16_set(b, i, 0, res);
    }

    // solve linear system
    Matrix16* x = mx16_alloc(num_lost_symbols, 1);
    int ret = linear_solve(A, b, x, deg_to_val, val_to_deg);
    
    // if (!check_linear_solution(A, b, x, deg_to_val, val_to_deg)) {
    //     printf("solution is incorrect\n");
    // }
    
    mx16_free(A);
    mx16_free(b);
    if (ret) {
        printf("failed to solve\n");
        free(data_mx);
        free(lost_symbols);
        mx16_free(x);
        return ret;
    }

    // fill lost data
    for (int i = 0; i < num_lost_symbols; ++i) {
        mx16_set(data_mx, 0, lost_symbols[i], mx16_get(x, i, 0));
    }
    free(data_mx);
    free(lost_symbols);
    mx16_free(x);

    return 0;
}

int rs_init(void* cfg, void** state) 
{
    FecRSCfg* rs_cfg = (FecRSCfg*)cfg; 
    *state = NULL;
    if (RS_NUM_SYMBOLS * RS_SYMBOL_SIZE % rs_cfg->payload_size) {
        return -1; // cannot distribute symbols evenly between payloads
    }
    if (rs_cfg->payload_size % RS_SYMBOL_SIZE) {
        return -1; // cannot form payload from symbols
    }

    FecRSState* rs_state = (FecRSState*)malloc(sizeof(FecRSState));
    
    uint16_t* deg_to_val = (uint16_t*)calloc(USHRT_MAX + 1, sizeof(uint16_t));
    uint16_t* val_to_deg = (uint16_t*)calloc(USHRT_MAX + 1, sizeof(uint16_t));
    fill_div_tables(deg_to_val, val_to_deg);
    rs_state->deg_to_val = deg_to_val;
    rs_state->val_to_deg = val_to_deg;

    uint32_t K = rs_cfg->payload_size * rs_cfg->num_original / RS_SYMBOL_SIZE;;
    uint32_t M = RS_NUM_SYMBOLS - K;
    uint32_t N = M + K;
    uint16_t poly[M + 1];
    memset(poly, 0, sizeof(poly));
    calc_gen_polynomial(poly, M, deg_to_val, val_to_deg);

    Matrix16* G = mx16_alloc(K, N);
    mx16_fill_poly(G, poly, M);
    mx16_fast_reduce(G, deg_to_val, val_to_deg);

    // extract P part of G = [I, P]
    Matrix16* P = mx16_alloc(K, M);
    for (int row = 0; row < K; ++row) {
        for (int col = 0; col < M; ++col) {
            mx16_set(P, row, col, mx16_get(G, row, col + K));
        }
    }
    rs_state->P = P;
    *state = rs_state;

    mx16_free(G);

    return 0;
}

void rs_deinit(void* cfg, void** state)
{
    (void)cfg;

    if (*state == NULL) { return; }

    FecRSState* rs_state = (FecRSState*)(*state);
    mx16_free(rs_state->P);
    free(rs_state->deg_to_val);
    free(rs_state->val_to_deg);
    free(rs_state);
}

uint32_t rs_payload_size(void* cfg) {
    FecRSCfg* rs_cfg = (FecRSCfg*)cfg;
    return rs_cfg->payload_size;
}

uint32_t rs_num_original(void* cfg) {
    FecRSCfg* rs_cfg = (FecRSCfg*)cfg;
    return rs_cfg->num_original;
}

uint32_t rs_num_recovery(void* cfg) {
    FecRSCfg* rs_cfg = (FecRSCfg*)cfg;
    return RS_NUM_SYMBOLS * RS_SYMBOL_SIZE / rs_cfg->payload_size - rs_cfg->num_original;
}

struct fec_callbacks reedsol_callbacks = {
    .init = rs_init,
    .deinit = rs_deinit,
    .payload_size = rs_payload_size,
    .num_original = rs_num_original, 
    .num_recovery = rs_num_recovery,
    .encode = rs_encode,
    .decode = rs_decode
};
