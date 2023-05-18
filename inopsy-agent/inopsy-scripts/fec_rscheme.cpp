// download https://eigen.tuxfamily.org/index.php?title=Main_Page
// build: g++ -I path/to/eigen/ fec_rscheme.cpp -o fec_rscheme

#include <iostream>
#include <cmath>
#include <optional>
#include <Eigen/Dense>
 
using Eigen::MatrixXi;

// multiply matrices using xor instead of plus
std::optional<MatrixXi> xormul(MatrixXi m1, MatrixXi m2)
{
    if (m1.cols() != m2.rows()) {
        return std::nullopt;
    }
    MatrixXi res{m1.rows(), m2.cols()};
 
    for (auto i = 0; i < m1.rows(); ++i) {
        for (auto j = 0; j < m2.cols(); ++j) {
            res(i, j) = 0;
 
            for (auto k = 0; k < m1.cols(); ++k) {
                res(i, j) ^= m1(i, k) * m2(k, j);
            }
        }
    }
    return res;
}

// k - number of data packets in correspondence to theory

// create utility matrix G
MatrixXi makeUtilMatrix(uint32_t t, uint32_t k) {
    MatrixXi m(t + 2, k);
    m.fill(0);
    for (int i = 0; i < k; ++i) {
        m(0, i) = 1;
        m(t + 1, i) = 1;
    }
    for (int col = 1; col < k; ++col) {
        int val = 1;
        for (int row = t; row >= 1; --row) {
            m(row, col) = (m(row, col - 1) + val) % 2;
            if (val != 0 && m(row, col) == 1) {
                val = 0;
            }
        }
    }
    return m;
}

MatrixXi encode(const MatrixXi& data) {    
    int t = ceil(std::log2(data.rows()));
    auto utilMatrix = makeUtilMatrix(t, data.rows());
    auto s = xormul(utilMatrix, data).value();
    return s;
}

int decode(MatrixXi& data, int k, std::vector<uint32_t>& lost) {
    std::sort(lost.begin(), lost.end());
    if (lost.size() == 0) { // nothing lost
        return 0;
    }
    if (lost.size() >= 1 && lost[0] >= k) { // lost only control packets
        return 0; 
    }
    if (lost.size() > 2 && lost.front() < k) { // cannot decode
        return -1;
    }

    int t = ceil(std::log2(k));
    // page 56 item C.3: one data packet lost
    if (lost.size() == 1 || (
        lost.size() == 2 && lost[0] < k && lost[1] >= k
    )) {
        int parity = k; // index of used parity packet
        if (lost.size() == 2 && lost[1] == k) {
            parity = k + t + 1;
        }
        for (int i = 0; i < data.cols(); ++i) {
            int vxor = 0;
            for (int j = 0; j < k; ++j) {
                if (j == lost[0]) {
                    continue;
                }
                vxor ^= data(j, i);
            }
            data(lost[0], i) = data(parity, i) ^ vxor;
        }
        return 0;
    }
    // only one option left: 2 data packets lost
    auto utilMatrix = makeUtilMatrix(t, k);
    auto i = 1;
    for (; i < t + 1; ++i) {
        if (utilMatrix(i, lost[0]) == 0 && 
            utilMatrix(i, lost[1]) == 1) {
            break;
        }
    }
    // recover lost[1] (j2 in report)
    for (int col = 0; col < data.cols(); ++col) {
        int vxor = 0;
        for (int row = 0; row < k; ++row) {
            if (row == lost[1]) {
                continue;
            }
            vxor = vxor ^ (data(row, col) * utilMatrix(i, row));
        }
        data(lost[1], col) = data(k + i, col) ^ vxor;
    }
    // recover lost[0] like in 1PR
    for (int col = 0; col < data.cols(); ++col) {
        int vxor = 0;
        for (int row = 0; row < k; ++row) {
            if (row == lost[0]) {
                continue;
            }
            vxor = vxor ^ data(row, col);
        }
        data(lost[0], col) = data(k, col) ^ vxor;
    }
    return 0;
}

// check rows and cols from 0 to |rows|/|cols| of matrices |a| and |b| are equal
bool checkEqual(MatrixXi& a, MatrixXi& b, uint32_t rows, uint32_t cols) {
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            if (a(i, j) != b(i, j)) {
                return false;
            }
        }
    }
    return true;
}

// erase rows from matrix |data| with indices from |lost|
void eraseData(MatrixXi& data, const std::vector<uint32_t>& lost) {
    for (auto lost_idx : lost) {
        if (lost_idx >= data.rows()) {
            throw std::runtime_error("invalid idx of lost packet");
        }
        for (auto i = 0; i < data.cols(); ++i) {
            data(lost_idx, i) = 0;
        }
    }
}

int main()
{
    MatrixXi data {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {11, 12, 13},
        {14, 15, 16},
        {17, 18, 19},
        {20, 21, 22},
        {23, 24, 25},
        {26, 27, 28}
    };
    auto control = encode(data);
    MatrixXi encoded(data.rows() + control.rows(), data.cols());
    encoded << data, control;

    int res;
    std::vector<uint32_t> lost;
    
    // nothing lost
    encoded << data, control;
    lost = {};
    res = decode(encoded, data.rows(), lost);
    if (res || !checkEqual(data, encoded, data.rows(), data.cols())) {
        throw std::runtime_error("spoiled encoded while there was nothing to decode");
    }

    // >2 lost, 1 in I section
    encoded << data, control;
    lost = {0, 9, 10};
    eraseData(encoded, lost);
    res = decode(encoded, data.rows(), lost);
    if (!res) {
        throw std::runtime_error("must have failed: > 2 packets and 1 in I section");
    }

    // 3 lost packets in P section
    encoded << data, control;
    lost = {10, 11, 13};
    eraseData(encoded, lost);
    res = decode(encoded, data.rows(), lost);
    if (res || !checkEqual(data, encoded, data.rows(), data.cols())) {
        throw std::runtime_error("spoiled encoded while there was nothing to decode");
    }

    // 1 lost packet in I section
    encoded << data, control;
    lost = {4};
    eraseData(encoded, lost);
    res = decode(encoded, data.rows(), lost);
    if (res || !checkEqual(data, encoded, data.rows(), data.cols())) {
        throw std::runtime_error("failed to recover 1 data packet");
    }

    // 1 lost packet in I section and 1 in P
    encoded << data, control;
    lost = {4, 9};
    eraseData(encoded, lost);
    res = decode(encoded, data.rows(), lost);
    if (res || !checkEqual(data, encoded, data.rows(), data.cols())) {
        throw std::runtime_error("failed to recover 1 data packet(1st parity lost)");
    }

    encoded << data, control;
    lost = {4, 11};
    eraseData(encoded, lost);
    res = decode(encoded, data.rows(), lost);
    if (res || !checkEqual(data, encoded, data.rows(), data.cols())) {
        throw std::runtime_error("failed to recover 1 data packet(non-parity lost)");
    }

    encoded << data, control;
    lost = {4, 14};
    eraseData(encoded, lost);
    res = decode(encoded, data.rows(), lost);
    if (res || !checkEqual(data, encoded, data.rows(), data.cols())) {
        throw std::runtime_error("failed to recover 1 data packet(2nd parity lost)");
    }

    // 2 lost packets in I section
    encoded << data, control;
    lost = {3, 5};
    eraseData(encoded, lost);
    res = decode(encoded, data.rows(), lost);
    if (res || !checkEqual(data, encoded, data.rows(), data.cols())) {
        throw std::runtime_error("failed to recover 2 data packets");
    }
}