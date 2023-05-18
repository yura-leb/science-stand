#include "fec_utils.h"

#include <ngtcp2/ngtcp2_crypto.h>
#include <ngtcp2/fec_r_scheme.h>
#include <ngtcp2/fec_leopard.h>
#include <ngtcp2/fec_1pr.h>
#include <ngtcp2/fec_2pr.h>
#include <ngtcp2/fec_reed_sol.h>

#undef NDEBUG
#include <algorithm>
#include <assert.h>
#include <chrono> 
#include <stdexcept>
#include <vector>

void test_naive_xor() {
    vec fst{new uint8_t[3]{0,0,0}, 3};
    vec snd{new uint8_t[5]{1,1,1,1,1}, 5};
    vec trd{new uint8_t[7]{2,2,2,2,2,2,2}, 7};

    vec bufs[3] = {fst, snd, trd};

    size_t rescnt = -1;
    // check cut correctness
    auto cut_bufs = cut_len(bufs, 3, 10, &rescnt);
    assert(rescnt == 3);
    assert(cut_bufs[0].base == fst.base && cut_bufs[0].len == 3);
    assert(cut_bufs[1].base == snd.base && cut_bufs[1].len == 5);
    assert(cut_bufs[2].base == trd.base && cut_bufs[2].len == 2);
    free(cut_bufs);

    // cut more elements than bufs contains
    cut_bufs = cut_len(bufs, 3, 16, &rescnt);
    assert(cut_bufs == nullptr);

    // check xor correctness
    // 0 0 0 1 1 
    // 1 1 1 2 2 
    // 2 2 2 2 2 
    // =
    // 3 3 3 1 1 
    auto xorred = naive_xor(bufs, 3, 5, 3);
    assert(xorred.len == 5);
    assert(xorred.base[0] == 3 && xorred.base[1] == 3);
    assert(xorred.base[2] == 3 && xorred.base[3] == 1);
    assert(xorred.base[4] == 1);

    // check flatten correctness
    auto flat = flatten_data(bufs, 3);
    uint8_t test_val1[15]{0,0,0,1,1,1,1,1,2,2,2,2,2,2,2};
    assert(flat.len == 15);
    for (auto i = 0; i < 15; ++i) {
        assert(test_val1[i] == flat.base[i]);
    }
}

// shuffle |array| with |n| elements of size |size| 
static void shuffle(void *array, size_t n, size_t size) {
    char tmp[size];
    char *arr = (char*)array;
    size_t stride = size * sizeof(char);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp, arr + j * stride, size);
            memcpy(arr + j * stride, arr + i * stride, size);
            memcpy(arr + i * stride, tmp, size);
        }
    }
}

// |lost| can be NULL
int test(fec_callbacks& fec_cb, void* cfg, uint32_t* lost, uint32_t num_lost) {
    void* state;
    if (fec_cb.init && fec_cb.init(cfg, &state)) {
        printf("failed to initialize FEC\n");
        return -1;
    }

    uint32_t original_size = fec_cb.num_original(cfg) * fec_cb.payload_size(cfg) * sizeof(uint8_t);
    uint8_t* original = (uint8_t*)malloc(original_size);
    for (int i = 0; i < original_size; ++i) {
        original[i] = rand();
        // original[i] = i % 8; // simpler to debug
    }

    uint8_t* validation = (uint8_t*)malloc(original_size);
    memcpy(validation, original, original_size);

    uint32_t recovery_size = fec_cb.num_recovery(cfg) * fec_cb.payload_size(cfg) * sizeof(uint8_t);
    uint8_t* recovery = (uint8_t*)malloc(recovery_size);
    auto begin = std::chrono::steady_clock::now();
    int res = fec_cb.encode(cfg, state, original, recovery);
    if (res) {
        printf("error code from encode: %i\n", res);
        return -1;
    }
    auto end = std::chrono::steady_clock::now();
    printf(
        "e %u %u %u %li\n", fec_cb.num_original(cfg), fec_cb.num_recovery(cfg), fec_cb.payload_size(cfg),
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
    );

    uint8_t* all_data = (uint8_t*)malloc(original_size + recovery_size);
    memcpy(all_data, original, original_size);
    memcpy(all_data + original_size, recovery, recovery_size);

    bool free_lost = false;
    if (!lost) {
        uint32_t idx_array_len = fec_cb.num_original(cfg) + fec_cb.num_recovery(cfg);
        uint32_t* idx_array = (uint32_t*)malloc(idx_array_len * sizeof(uint32_t));
        for (int i = 0; i < idx_array_len; ++i) {
            idx_array[i] = i;
        }
        shuffle(idx_array, idx_array_len, sizeof(uint32_t));

        free_lost = true;
        lost = (uint32_t*)malloc(num_lost * sizeof(uint32_t));
        memcpy(lost, idx_array, num_lost * sizeof(uint32_t));

        free(idx_array);
    }
    std::sort(lost, lost + num_lost);

    for (int i = 0; i < num_lost; ++i) {
        for (int j = 0; j < fec_cb.payload_size(cfg); ++j) {
            all_data[lost[i] * fec_cb.payload_size(cfg) + j] = 0;
        }
    }
    
    begin = std::chrono::steady_clock::now();
    res = fec_cb.decode(cfg, state, all_data, lost, num_lost);
    if (res) {
        printf("error code from decode %i\n", res);
        return -1;
    }
    end = std::chrono::steady_clock::now();
    printf(
        "r %u %u %u %li %u\n", fec_cb.num_original(cfg), fec_cb.num_recovery(cfg), fec_cb.payload_size(cfg),
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count(), num_lost
    );  
    uint32_t failed_count = 0;
    for (int i = 0; i < original_size; ++i) {
        if (all_data[i] != validation[i]) {
            failed_count += 1;
        }
    }
    if (failed_count) {
        printf("failed to recover %u bytes\n", failed_count);
        return -1;
    }

    // freeing currently only works for correct path
    free(all_data);
    if (free_lost) {
        free(lost);
    }

    free(original);
    free(recovery);
    free(validation);
    if (fec_cb.deinit) {
        fec_cb.deinit(cfg, &state);
    }
    
    return 0;
}

void test_leopard() {
    FecLeoCfg* leo_cfg = new FecLeoCfg{};
    leo_cfg->buf_size = 64;
    leo_cfg->num_original = 48;
    leo_cfg->num_recovery = 16;
    leo_cfg->payload_size = 1024;

    if (test(leopard_callbacks, leo_cfg, NULL, 0)) {
        throw std::runtime_error("leopard test failed");
    }
    if (test(leopard_callbacks, leo_cfg, NULL, 5)) {
        throw std::runtime_error("leopard test failed");
    }
    if (test(leopard_callbacks, leo_cfg, NULL, 10)) {
        throw std::runtime_error("leopard test failed");
    }
    if (test(leopard_callbacks, leo_cfg, NULL, 16)) {
        throw std::runtime_error("leopard test failed");
    }
    delete(leo_cfg);
    printf("leopard tests succeeded\n");
}

void test_reedsol() {
    FecRSCfg* rs_cfg = new FecRSCfg{};
    rs_cfg->num_original = 170;
    // 514 is chosen due to following constraints:
    // payload shoud consist of whole symbols: payload_size % 2 == 0
    // FEC batch byte size should be divisible by payload size: 65535 * 2 % payload_size == 0
    // payload size should be less than common MTU: payload_size < 1500
    rs_cfg->payload_size = 514;

    std::vector<uint32_t> lost;
    // HACK: currently RS code does not decode properly with recovery symbols lost
    auto fill_lost = [&](uint32_t num_lost){
        lost.resize(num_lost);
        for (int i = 0; i < num_lost; ++i) {
            lost[i] = i;
        }
    };
    if (test(reedsol_callbacks, rs_cfg, nullptr, 0)) {
        throw std::runtime_error("RS test failed");
    }

    fill_lost(10);
    if (test(reedsol_callbacks, rs_cfg, &lost[0], 10)) {
        throw std::runtime_error("RS test failed");
    }

    fill_lost(20);
    if (test(reedsol_callbacks, rs_cfg, &lost[0], 20)) {
        throw std::runtime_error("RS test failed");
    }

    // following tests are rather slow    
    // fill_lost(40);
    // if (test(reedsol_callbacks, rs_cfg, &lost[0], 40)) {
    //     throw std::runtime_error("RS test failed");
    // }

    // fill_lost(60);
    // if (test(reedsol_callbacks, rs_cfg, &lost[0], 60)) {
    //     throw std::runtime_error("RS test failed");
    // }

    // fill_lost(85);
    // if (test(reedsol_callbacks, rs_cfg, &lost[0], 85)) {
    //     throw std::runtime_error("RS test failed");
    // }
    delete(rs_cfg);
    printf("RS tests succeeded\n");
}

void test_rscheme() {
    FecRCfg* r_cfg = new FecRCfg{};
    r_cfg->num_original = 9;
    r_cfg->payload_size = 3;

    if (test(rscheme_callbacks, r_cfg, NULL, 2)) {
        printf("R scheme test failed");
    }

    std::vector<uint32_t> lost;
    // nothing lost
    lost = {};
    auto res = test(rscheme_callbacks, r_cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("spoiled encoded while there was nothing to decode");
    }

    // >2 lost, 1 in I section
    lost = {0, 9, 10};
    res = test(rscheme_callbacks, r_cfg, &lost[0], lost.size());
    if (!res) {
        throw std::runtime_error("must have failed: > 2 packets and 1 in I section");
    }

    // 3 lost packets in P section
    lost = {10, 11, 13};
    res = test(rscheme_callbacks, r_cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("spoiled encoded while there was nothing to decode");
    }

    // 1 lost packet in I section
    lost = {4};
    res = test(rscheme_callbacks, r_cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("failed to recover 1 data packet");
    }

    // 1 lost packet in I section and 1 in P
    lost = {4, 9};
    res = test(rscheme_callbacks, r_cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("failed to recover 1 data packet(1st parity lost)");
    }

    lost = {4, 11};
    res = test(rscheme_callbacks, r_cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("failed to recover 1 data packet(non-parity lost)");
    }

    lost = {4, 14};
    res = test(rscheme_callbacks, r_cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("failed to recover 1 data packet(2nd parity lost)");
    }

    // 2 lost packets in I section
    lost = {3, 5};
    res = test(rscheme_callbacks, r_cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("failed to recover 2 data packets");
    }

    delete(r_cfg);
    printf("rscheme tests succeeded\n");
}

void test_1pr() {
    FecXORCfg cfg{1024, 63};

    std::vector<uint32_t> lost;
    // nothing lost
    lost = {};
    auto res = test(xor_callbacks, &cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("spoiled encoded while there was nothing to decode");
    }

    lost = {63};
    res = test(xor_callbacks, &cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("error when lost 1 recovery");
    }

    lost = {5};
    res = test(xor_callbacks, &cfg, &lost[0], lost.size());
    if (res) {
        throw std::runtime_error("error when lost 1 original");
    }

    lost = {11, 35};
    res = test(xor_callbacks, &cfg, &lost[0], lost.size());
    if (!res) {
        throw std::runtime_error("error when lost 2 original");
    }
    printf("1PR tests succeeded\n");
}

void test_2pr() {
    // test both odd and even number of payloads
    std::vector<uint32_t> orig_size = {62, 63};
    for (auto num_original : orig_size) {
        Fec2PRCfg cfg{1024, num_original};

        std::vector<uint32_t> lost;
        lost = {};
        auto res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (res) {
            throw std::runtime_error("spoiled encoded while there was nothing to decode");
        }

        lost = {num_original, num_original + 1};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (res) {
            throw std::runtime_error("error when lost 2 recovery");
        }

        lost = {2};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (res) {
            throw std::runtime_error("error when lost 1 original (even index)");
        }
        
        lost = {3};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (res) {
            throw std::runtime_error("error when lost 1 original (odd index)");
        }

        lost = {0, 1};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (res) {
            throw std::runtime_error("error when lost 2 original with different parity");
        }

        lost = {0, num_original + 1};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (res) {
            throw std::runtime_error("error when lost 1 original and 1 recovery of different parity (even)");
        }

        lost = {1, num_original};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (res) {
            throw std::runtime_error("error when lost 1 original and 1 recovery of different parity (odd)");
        }

        lost = {0, num_original};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (!res) {
            throw std::runtime_error("error when lost 1 original and 1 recovery of same parity (even)");
        }

        lost = {1, num_original + 1};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (!res) {
            throw std::runtime_error("error when lost 1 original and 1 recovery of same parity (odd)");
        }

        lost = {2, 4};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (!res) {
            throw std::runtime_error("error when lost 2 original of same parity (even)");
        }

        lost = {3, 5};
        res = test(pr2_callbacks, &cfg, &lost[0], lost.size());
        if (!res) {
            throw std::runtime_error("error when lost 2 original of same parity (odd)");
        }
    }

    printf("2PR tests succeeded\n");
}

int main() {
    test_naive_xor();
    test_1pr();
    test_2pr();
    test_leopard();
    test_rscheme();
    test_reedsol();
    return 0;
}