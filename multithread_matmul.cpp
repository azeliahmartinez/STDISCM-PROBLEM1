// multithread_matmul.cpp
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

static inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

bool parseMatrixSection(const std::string& section, Matrix& M, std::string& err) {
    std::istringstream in(section);
    std::string line;
    size_t expected_cols = SIZE_MAX;
    M.clear();

    while (std::getline(in, line)) {
        auto t = trim(line);
        if (t.empty()) continue; // skip stray blank lines
        std::istringstream ls(t);
        std::vector<double> row;
        std::string tok;
        while (ls >> tok) {
            try {
                row.push_back(std::stod(tok));
            } catch (...) {
                err = "Non-numeric token in matrix: '" + tok + "'";
                return false;
            }
        }
        if (row.empty()) continue;
        if (expected_cols == SIZE_MAX) expected_cols = row.size();
        if (row.size() != expected_cols) {
            err = "Matrix rows have inconsistent column counts.";
            return false;
        }
        M.push_back(std::move(row));
    }
    if (M.empty()) {
        err = "Matrix section is empty.";
        return false;
    }
    return true;
}

bool readInputMatrices(const std::string& path, Matrix& A, Matrix& B, std::string& err) {
    std::ifstream fin(path);
    if (!fin) { err = "Cannot open input file: " + path; return false; }

    std::ostringstream buf;
    buf << fin.rdbuf();
    std::string all = buf.str();

    // Split by the first blank line (one or more \n with optional spaces)
    // We'll scan line by line to find the first blank line.
    std::istringstream in(all);
    std::ostringstream part1, part2;
    std::string line;
    bool separator_found = false;
    bool last_was_blank = false;

    while (std::getline(in, line)) {
        std::string t = trim(line);
        if (!separator_found) {
            if (t.empty()) {
                // First blank line marks separation
                separator_found = true;
            } else {
                part1 << line << "\n";
            }
        } else {
            part2 << line << "\n";
        }
    }

    if (!separator_found) {
        err = "Expected a blank line separating Matrix A and Matrix B.";
        return false;
    }

    std::string e1, e2;
    if (!parseMatrixSection(part1.str(), A, e1)) { err = "Matrix A: " + e1; return false; }
    if (!parseMatrixSection(part2.str(), B, e2)) { err = "Matrix B: " + e2; return false; }

    // Dimension check: A is m x k, B is k x n
    size_t m = A.size();
    size_t kA = A[0].size();
    size_t kB = B.size();
    if (kA != kB) {
        err = "Dimension mismatch: A is " + std::to_string(m) + "x" + std::to_string(kA) +
              ", B is " + std::to_string(kB) + "x" + std::to_string(B[0].size()) + ".";
        return false;
    }
    return true;
}

Matrix transpose(const Matrix& M) {
    size_t r = M.size(), c = M[0].size();
    Matrix T(c, std::vector<double>(r));
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            T[j][i] = M[i][j];
    return T;
}

void multiplyRange(const Matrix& A, const Matrix& BT, Matrix& C, size_t row_begin, size_t row_end) {
    // BT is B transposed: size n x k
    const size_t n = BT.size();
    const size_t k = BT[0].size(); // equals A[0].size()
    for (size_t i = row_begin; i < row_end; ++i) {
        C[i].assign(n, 0.0);
        for (size_t j = 0; j < n; ++j) {
            // dot( A[i][0..k-1], B[0..k-1][j] ) = dot( A[i], BT[j] )
            const auto& rowA = A[i];
            const auto& rowBT = BT[j];
            double sum = 0.0;
            // Manual unrolling can help; keep it straightforward and safe:
            for (size_t t = 0; t < k; ++t) sum += rowA[t] * rowBT[t];
            C[i][j] = sum;
        }
    }
}

void writeMatrix(const std::string& path, const Matrix& M) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot open output file: " + path);
    }
    for (const auto& row : M) {
        for (size_t j = 0; j < row.size(); ++j) {
            if (j) out << ' ';
            // Use default formatting; adjust as needed
            out << row[j];
        }
        out << '\n';
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage:\n  " << argv[0] << " <input.txt> <output.txt>\n\n"
                  << "Input format:\n"
                  << "- Matrix A lines\n- (at least one) blank line\n- Matrix B lines\n"
                  << "Rows are space-separated numbers; both matrices must be rectangular.\n";
        return 1;
    }

    const std::string inPath = argv[1];
    const std::string outPath = argv[2];

    Matrix A, B;
    std::string err;
    if (!readInputMatrices(inPath, A, B, err)) {
        std::cerr << "Error: " << err << "\n";
        return 1;
    }

    const size_t m = A.size();
    const size_t k = A[0].size();
    const size_t n = B[0].size();

    // Transpose B for cache-friendly dot products
    Matrix BT = transpose(B);

    Matrix C(m, std::vector<double>(n, 0.0));

    // Decide thread count and partition rows
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4; // fallback
    size_t threads = std::min<size_t>(hw, m);
    if (threads == 0) threads = 1;

    std::vector<std::thread> pool;
    pool.reserve(threads);

    size_t rows_per_thread = m / threads;
    size_t extra = m % threads;
    size_t begin = 0;

    for (size_t t = 0; t < threads; ++t) {
        size_t take = rows_per_thread + (t < extra ? 1 : 0);
        size_t end = begin + take;
        if (begin >= end) break; // in case m < threads
        pool.emplace_back(multiplyRange, std::cref(A), std::cref(BT), std::ref(C), begin, end);
        begin = end;
    }

    for (auto& th : pool) th.join();

    try {
        writeMatrix(outPath, C);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
