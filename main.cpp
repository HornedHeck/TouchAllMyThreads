#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std::chrono;
using std::vector;

#define UNIT microseconds

long long scalar(const vector<int> &a, const vector<int> &b) {

    if (a.size() != b.size()) return 0;

    long long sum = 0;

#pragma omp parallel for reduction(+:sum) default(none) shared(a, b)
    for (int i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

long long scalar_sync(vector<int> a, vector<int> b) {

    if (a.size() != b.size()) return 0;

    long long sum = 0;

    for (int i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void test_omp(const vector<int> &a, const vector<int> &b, int threads, long long expected) {
    omp_set_num_threads(threads);
    auto start = system_clock::now();
    auto res = scalar(a, b);
    auto end = system_clock::now();
    std::cout << "OMP-" << threads << ": " << duration_cast<UNIT>(end - start).count() << " ";
    if (res == expected) {
        std::cout << "Correct\n";
    } else {
        std::cout << "Wrong\n";
        exit(1);
    }
}

int main() {
    vector<int> a, b;
    for (int i = 0; i < 1000000; ++i) {
        a.push_back(random() % 1000 - 500);
        b.push_back(random() % 1000 - 500);
    }

    long long res;
    auto start = system_clock::now();
    res = scalar_sync(a, b);
    auto end = system_clock::now();
    std::cout << "Sync: " << duration_cast<UNIT>(end - start).count() << std::endl;

    omp_set_dynamic(false);
    test_omp(a, b, 2, res);
    test_omp(a, b, 4, res);
    test_omp(a, b, 8, res);
    test_omp(a, b, 16, res);
    test_omp(a, b, 32, res);
    test_omp(a, b, 64, res);
    test_omp(a, b, 128, res);

    return 0;
}
