#include <stdint.h>
#include <limits.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

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
    // printf("%u %u\n", lhs, rhs);    
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
        // printf("%i -> %u\n", i + 1, val);
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
    // printf("at 0 %u\n", fin_mul(deg_to_val[k], polynomial[0]));
    result[0] = fin_fmul(deg_to_val[k], polynomial[0], deg_to_val, val_to_deg);
    // TODO: try in another direction to make in place
    for (int i = 1; i <= max_degree; ++i) {
        // each coeff[i] is coeff[i] * alpha^k + coeff[i-1] 
        // with plus/multiplication from finite field
        result[i] = fin_fmul(deg_to_val[k], polynomial[i], deg_to_val, val_to_deg) ^ polynomial[i - 1];
    }
    for (int i = 0; i <= max_degree; ++i) {
        polynomial[i] = result[i];
        // printf("%u", polynomial[i]);
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
    // printf(" = %u\n", res);
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

typedef struct Matrix {
    uint32_t num_rows;
    uint32_t num_cols;
    uint16_t* data;
} Matrix;

Matrix* matrix_alloc(uint32_t num_rows, uint32_t num_cols) {
    Matrix* m = (Matrix*)malloc(sizeof(*m));
    m->num_rows = num_rows;
    m->num_cols = num_cols;
    m->data = (uint16_t*)calloc(num_rows * num_cols, sizeof(uint16_t));
    return m;
}

inline void matrix_set(Matrix* m, uint32_t row, uint32_t col, uint16_t value) {
    assert(m != NULL && row < m->num_rows && col < m->num_cols);
    m->data[row * m->num_cols + col] = value;
}

inline uint16_t matrix_get(Matrix* m, uint32_t row, uint32_t col) {
    assert(m != NULL && row < m->num_rows && col < m->num_cols);
    return m->data[row * m->num_cols + col];
}

void matrix_free(Matrix* m) {
    free(m->data);
    free(m);
}

void matrix_print(Matrix* m) {
    for (int i = 0; i < m->num_rows; ++i) {
        for (int j = 0; j < m->num_cols; ++j) {
            printf("%u ", matrix_get(m, i, j));
        }
        printf("\n");
    }
}

// fill matrix with polynomial values for RS code
int matrix_fill_poly(Matrix* m, uint16_t* poly, uint32_t max_degree) {
    // M - poly degree, N*K - matrix size, M + K = N
    if (m->num_cols != m->num_rows + max_degree) {
        return -1; // invalid matrix/polynomial sizes
    }
    for (int i = 0; i < m->num_rows; i++) {
        for (int j = 0; j <= max_degree; ++ j) {
            matrix_set(m, i, i + j, poly[j]);
        }
    }
    return 0;
}

// add |row_to_add|*|row_multiplier| to |dst_row|
void matrix_add_row_lin(
    Matrix* m, uint32_t dst_row, uint32_t row_to_add, uint16_t row_multiplier,
    uint16_t* deg_to_val, uint16_t* val_to_deg    
) {
    for (int i = 0; i < m->num_cols; ++i) {
        uint16_t dst_val = matrix_get(m, dst_row, i);
        matrix_set(m, dst_row, i,
            dst_val ^ fin_fmul(matrix_get(m, row_to_add, i), row_multiplier, deg_to_val, val_to_deg)
        );
    }
}

// naive implementation is too slow (requires hours for 51000x65535)
// reduce matrix to [I, P] form
void matrix_reduce(
    Matrix* m,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    assert(m->num_cols >= m->num_rows);
    for (int i = 0; i < m->num_rows; ++i) {
        // diagonal elements of matrix |m| are non-zero
        uint16_t multiplier = fin_div(1, matrix_get(m, i, i), deg_to_val, val_to_deg);
        for (int j = i; j < m->num_cols; ++j) {
            uint16_t val = matrix_get(m, i, j);
            if (val == 0) { // g0 ... gm 0 ... 0 and gi != 0
                break;
            }
            matrix_set(m, i, j, fin_fmul(val, multiplier, deg_to_val, val_to_deg));
        }
        for (int j = i + 1; j < m->num_rows; ++j) {
            multiplier = fin_div(matrix_get(m, i, j), matrix_get(m, j, j), deg_to_val, val_to_deg);
            matrix_add_row_lin(m, i, j, multiplier, deg_to_val, val_to_deg);
        }
    }
}

// this implementation uses fact that all rows in matrix have the same values shifted by 1 column
// to each row i we apply the same changes that we applied to (i+1)th (copy i+1 to i shifting by 1) 
// and (k-1)kth value will no be zero as for (i+1)th row it is in P part of [I, P]
// so we need to zero it using last row (as last row will have 1 only in (k-1)th col)
void matrix_fast_reduce(
    Matrix* m,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    assert(m->num_cols >= m->num_rows);

    int i = m->num_rows - 1;
    // diagonal elements of matrix |m| are non-zero
    uint16_t multiplier = fin_div(1, matrix_get(m, i, i), deg_to_val, val_to_deg);
    for (int j = i; j < m->num_cols; ++j) {
        uint16_t val = matrix_get(m, i, j);
        if (val == 0) { // g0 ... gm 0 ... 0 and gi != 0
            break;
        }
        matrix_set(m, i, j, fin_fmul(val, multiplier, deg_to_val, val_to_deg));
    }

    for (i = i - 1; i >= 0; --i) {
        for (int j = i; j < m->num_cols - 1; ++j) {
            matrix_set(m, i, j, matrix_get(m, i + 1, j + 1));
        }
        multiplier = matrix_get(m, i, m->num_rows - 1);
        matrix_add_row_lin(m, i, m->num_rows - 1, multiplier, deg_to_val, val_to_deg);
    }
}

Matrix* matrix_xor_mul(
    Matrix* m1, Matrix* m2,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    if (m1->num_cols != m2->num_rows) {
        return NULL;
    }
    Matrix* res = matrix_alloc(m1->num_rows, m2->num_cols);
 
    uint16_t tmp;
    for (int i = 0; i < m1->num_rows; ++i) {
        for (int j = 0; j < m2->num_cols; ++j) {
            tmp = 0;
 
            for (int k = 0; k < m1->num_cols; ++k) {
                tmp ^= fin_fmul(
                    matrix_get(m1, i, k), matrix_get(m2, k, j),
                    deg_to_val, val_to_deg
                );
            }
            matrix_set(res, i, j, tmp);
        }
    }
    return res;
}

// produce additional symbols, original ones are not changed
int encode(
    Matrix* P,
    Matrix* data,
    Matrix** res,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    *res = matrix_xor_mul(data, P, deg_to_val, val_to_deg);
    return (*res == NULL) ? -1 : 0;
}

// solve linear system Ax = b with square A
int linear_solve(
    Matrix* A, Matrix* b, Matrix* x,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    if (A->num_cols != A->num_rows) {
        return -1; // works only for square matrices
    }
    uint32_t n = A->num_rows - 1;

    Matrix* D = matrix_alloc(A->num_rows, A->num_cols + 1);
    for (int row = 0; row <= n; ++row) {
        for (int col = 0; col <= n; ++col) {
            matrix_set(D, row, col, matrix_get(A, row, col));
        }
        matrix_set(D, row, n + 1, matrix_get(b, row, 0));
    }
    for (int k = 0; k <= n; ++k) {
        uint16_t r = 0;
        uint16_t lp = 0;   
        for (int l = k; l <= n; ++l) {
            if (r < matrix_get(D, l, k)) { // ???: is it ok for finite field?
                lp = l;
                r = matrix_get(D, l, k);
            }
        }
        if (r == 0) {
            return -1; // zero determinant
        }
        if (k != lp) {
            for (int j = k; j <= n + 1; ++j) {
                r = matrix_get(D, k, j);
                matrix_set(D, k, j, matrix_get(D, lp, j));
                matrix_set(D, lp, j, r);
            }
        }
        for (int j = k + 1; j <= n + 1; ++j) {
            matrix_set(D, k, j, fin_div(
                matrix_get(D, k, j), matrix_get(D, k, k),
                deg_to_val, val_to_deg
            ));
        }

        // !!!: probably error in theory, so iterations are a bit different
        if (k != n) {
            for (int i = k + 1; i <= n; ++i) {
                for (int j = k + 1; j <= n + 1; ++j) {
                    matrix_set(D, i, j, 
                        matrix_get(D, i, j) ^ fin_fmul(
                            matrix_get(D, k, j), matrix_get(D, i, k),
                            deg_to_val, val_to_deg
                        )
                    );
                }
            }
        }
        if (k != 0) {
            for (int i = 0; i <= k - 1; ++i) {
                for (int j = k + 1; j <= n + 1; ++j) {
                    matrix_set(D, i, j, 
                        matrix_get(D, i, j) ^ fin_fmul(
                            matrix_get(D, k, j), matrix_get(D, i, k),
                            deg_to_val, val_to_deg
                        )
                    );
                }
            }
        }
    }
    for (int i = 0; i <= n; ++i) {
        matrix_set(x, i, 0, matrix_get(D, i, n + 1));        
    }
    return 0;
}

// check whether linear system Ax = b solution is correct
bool check_linear_solution(
    Matrix* A, Matrix* b, Matrix* x,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    Matrix* mul = matrix_xor_mul(A, x, deg_to_val, val_to_deg);
    if (mul->num_cols != b->num_cols || mul->num_rows != b->num_rows) {
        printf("incorrect matrix size\n");
        return false;
    }
    for (int row = 0; row < x->num_cols; ++x) {
        if (matrix_get(b, row, 0) != matrix_get(mul, row, 0)) {
            return false;
        }
    }
    return true;
}

// expect lost positions in data to be filled with zeroes
int decode(
    Matrix* P, // k*m matrix
    Matrix* data,
    uint32_t* lost, uint32_t num_lost,
    uint16_t* deg_to_val, uint16_t* val_to_deg
) {
    clock_t start = clock();
    uint32_t k = P->num_rows;
    uint32_t m = P->num_cols;
    if (num_lost > m) {
        return -1; // cannot recover more than m symbols
    }
    // TODO: shortcut for case when only non-data symbols lost
    // fill A and b matrices for linear system
    Matrix* A = matrix_alloc(num_lost, num_lost);
    Matrix* b = matrix_alloc(num_lost, 1);
    for (int i = 0; i < num_lost; ++i) {
        for (int j = 0; j < num_lost; ++j) {
            if (lost[i] < k) { // P part
                matrix_set(A, j, i, matrix_get(P, lost[i], j));
            } else { // I part
                matrix_set(A, j, i, (lost[i] == j + k) ? 1 : 0);
            }
        }
    }

    for (int i = 0; i < num_lost; ++i) {
        uint16_t res = 0;
        for (int j = 0; j < data->num_cols; ++j) {
            uint16_t coeff = (j < k) ? matrix_get(P, j, i) : ((j == i + k) ? 1 : 0); 
            res ^= fin_fmul(coeff, matrix_get(data, 0, j), deg_to_val, val_to_deg);
        }
        matrix_set(b, i, 0, res);
    }

    // solve linear system
    Matrix* x = matrix_alloc(num_lost, 1);
    int ret = linear_solve(A, b, x, deg_to_val, val_to_deg);
    if (ret) {
        printf("failed to solve\n");
        return ret;
    }

    if (!check_linear_solution(A, b, x, deg_to_val, val_to_deg)) {
        printf("solution is incorrect\n");
    }

    // fill lost data
    for (int i = 0; i < num_lost; ++i) {
        matrix_set(data, 0, lost[i], matrix_get(x, i, 0));
    }

    return 0;
}

void test() {
    // test some concrete examples
    rt_assert(fin_mul(3, 3) == 5);
    rt_assert(fin_mul(1, 2) == 2);
    // test 2 is primitive element (all non-zero elements are its degrees)
    bool buf[USHRT_MAX + 1];
    memset(buf, 0, sizeof(buf));
    uint16_t prim_elem = 2; 
    uint16_t val = prim_elem;
    buf[val] = true;
    for (int i = 0; i < USHRT_MAX; ++i) {
        val = fin_mul(val, prim_elem);
        buf[val] = true;
    }
    for (int i = 1; i <= USHRT_MAX; ++i) {
        rt_assert(buf[i]);
    }
    // test division tables are filled with all unique degrees and values
    uint16_t deg_to_val[USHRT_MAX + 1];
    uint16_t val_to_deg[USHRT_MAX + 1];
    bool val_buf[USHRT_MAX + 1];
    bool deg_buf[USHRT_MAX + 1];
    memset(deg_to_val, 0, sizeof(deg_to_val));
    memset(val_to_deg, 0, sizeof(val_to_deg));
    memset(val_buf, false, sizeof(val_buf));
    memset(deg_buf, false, sizeof(deg_buf));
    fill_div_tables(deg_to_val, val_to_deg);
    for (int i = 0; i <= USHRT_MAX; ++i) {
        val_buf[deg_to_val[i]] = true;
        deg_buf[val_to_deg[i]] = true;
    }
    rt_assert(!val_buf[0]);
    for (int i = 1; i <= USHRT_MAX; ++i) {
        rt_assert(val_buf[i]);
    }
    for (int i = 0; i <= USHRT_MAX; ++i) {
        rt_assert(deg_buf[i]);
    }
    // test concrete division examples
    rt_assert(fin_div(5, 3, deg_to_val, val_to_deg) == 3);
    rt_assert(fin_div(8214, 4, deg_to_val, val_to_deg) == 32768);
    rt_assert(fin_div(8214, 32768, deg_to_val, val_to_deg) == 4);
    rt_assert(fin_div(1, 2, deg_to_val, val_to_deg) == 34821);
    rt_assert(fin_div(1, 4, deg_to_val, val_to_deg) == 52231);
    // test polynomial multiplication
    uint16_t polynomial_a[3];
    memset(polynomial_a, 0, sizeof(polynomial_a));
    polynomial_a[0] = 2;
    polynomial_a[1] = 1;
    mul_polynomial(polynomial_a, 2, 2, deg_to_val, val_to_deg);
    // (x + 2) * (x + 2^2) == x^2 + 6x + 8
    rt_assert(polynomial_a[0] == 8 && polynomial_a[1] == 6 && polynomial_a[2] == 1);
    // check generating polynomial is zero for all non-zero values of x
    uint32_t M = 14535;
    uint16_t polynomial[M + 1];
    memset(polynomial, 0, sizeof(polynomial));
    calc_gen_polynomial(polynomial, M, deg_to_val, val_to_deg);
    for (int i = 1; i <= M; ++i) {
        rt_assert(!calc_polynomial_val(polynomial, M, deg_to_val[i], deg_to_val, val_to_deg));
    }

    // TODO: we can store only P part of matrix
    // test matrix reduction to [I, P]
    uint16_t poly[4];
    poly[0] = 4;
    poly[1] = 3;
    poly[2] = 9;
    poly[3] = 1;

    Matrix* m1 = matrix_alloc(4, 7);
    matrix_fill_poly(m1, poly, 3);
    matrix_reduce(m1, deg_to_val, val_to_deg);

    Matrix* m2 = matrix_alloc(4, 7);
    matrix_fill_poly(m2, poly, 3);
    matrix_fast_reduce(m2, deg_to_val, val_to_deg);

    // check m1 contains I
    for (int i = 0; i < m1->num_rows; ++i) {
        for (int j = 0; j < m1->num_rows; ++j) {
            rt_assert(matrix_get(m1, i, j) == (i == j ? 1 : 0));
        }
    }

    // check m1 == m2
    rt_assert(m1->num_cols == m2->num_cols && m1->num_rows == m2->num_rows);
    for (int i = 0; i < m1->num_rows; ++i) {
        for (int j = 0; j < m1->num_cols; ++j) {
            rt_assert(matrix_get(m1, i, j) == matrix_get(m2, i, j));
        }
    }

    // check solving linear systems
    Matrix* A = matrix_alloc(3, 3);
    matrix_set(A, 0, 0, 1);
    matrix_set(A, 1, 1, 1);
    matrix_set(A, 2, 0, 1);
    matrix_set(A, 2, 2, 1);
    Matrix* b = matrix_alloc(3, 1);
    matrix_set(b, 0, 0, 1);
    matrix_set(b, 1, 0, 2);
    matrix_set(b, 2, 0, 3);
    Matrix* x = matrix_alloc(3, 1);
    int res = linear_solve(A, b, x, deg_to_val, val_to_deg);
    rt_assert(!res && 
        matrix_get(x, 0, 0) == 1 && 
        matrix_get(x, 1, 0) == 2 && 
        matrix_get(x, 2, 0) == 2
    );
}

int test_rs() {
    // test division tables are filled with all unique degrees and values
    uint16_t deg_to_val[USHRT_MAX + 1];
    uint16_t val_to_deg[USHRT_MAX + 1];
    memset(deg_to_val, 0, sizeof(deg_to_val));
    memset(val_to_deg, 0, sizeof(val_to_deg));
    fill_div_tables(deg_to_val, val_to_deg);
    printf("filled degree tables\n");

    uint32_t M = 14535;
    uint32_t K = 51000;
    uint32_t N = M + K;
    uint16_t poly[M + 1];
    memset(poly, 0, sizeof(poly));
    calc_gen_polynomial(poly, M, deg_to_val, val_to_deg);
    printf("calculated polynomial\n");

    Matrix* G = matrix_alloc(K, N);
    matrix_fill_poly(G, poly, M);
    matrix_fast_reduce(G, deg_to_val, val_to_deg);
    printf("prepared matrix G\n");

    // extract P part of G = [I, P]
    Matrix* P = matrix_alloc(K, M);
    for (int row = 0; row < K; ++row) {
        for (int col = 0; col < M; ++col) {
            matrix_set(P, row, col, matrix_get(G, row, col + K));
        }
    }
    printf("extracted P\n");

    matrix_free(G);

    // generate data
    // TODO: will need conversion from custom buffer to Matrix
    Matrix* data = matrix_alloc(1, K);
    for (int i = 0; i < K; ++i) {
        matrix_set(data, 0, i, rand());
    }
    printf("generated data\n");

    // encode
    Matrix* service_data = NULL;
    clock_t start = clock();
    int ret = encode(P, data, &service_data, deg_to_val, val_to_deg);
    if (ret) {
        printf("failed to encode\n");
        return ret;
    }
    clock_t end = clock();
    printf("encoded data in %f\n", (1.0 * (end - start)) / CLOCKS_PER_SEC);

    // form full data matrix
    Matrix* msg = matrix_alloc(1, N);
    for (int i = 0; i < K; ++i) {
        matrix_set(msg, 0, i, matrix_get(data, 0, i));
    }
    for (int i = K; i < N; ++i) {
        matrix_set(msg, 0, i, matrix_get(service_data, 0, i - K));
    }
    printf("prepared msg\n");

    Matrix* msg_recv = matrix_alloc(1, N);
    for (int i = 0; i < N; ++i) {
        matrix_set(msg_recv, 0, i, matrix_get(msg, 0, i));
    }

    // lose some first elements
    uint32_t num_lost = 2000;
    uint32_t lost[num_lost];
    for (int i = 0; i < num_lost; ++i) {
        lost[i] = i;
        matrix_set(msg_recv, 0, i, 0);
    }
    printf("lost data\n");

    // decode
    start = clock();
    ret = decode(P, msg_recv, lost, num_lost, deg_to_val, val_to_deg);
    if (ret) {
        printf("failed to decode\n");
        return ret;
    }
    end = clock();
    printf("decoded & recovered data in %f\n", (1.0 * (end - start)) / CLOCKS_PER_SEC);
    
    // check data recovered
    for (int i = 0; i < num_lost; ++i) {
        rt_assert(matrix_get(msg, 0, i) == matrix_get(msg_recv, 0, i));
    }
    printf("checked correctness\n");

    return 0;
}

int main()
{
    test();
    test_rs();
    return 0;
}
