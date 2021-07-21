// -*-c++-*-
// CUDA matrix multiplication

#include <mma.h>

#include <cassert>
// #include <cstdint>
// #include <cstdlib>
// #include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

using namespace nvcuda;
using namespace nvcuda::wmma;

// These sizes are dictated by CUDA
constexpr int m = 8;
constexpr int n = 8;
constexpr int k = 32;

// A[m][k]   must be row major
// B[k][n]   must be column major
// C[m][n]   row major
// C[m][n] += A[m][k] * B[k][n]

__global__ void matmul(int *restrict const Cptr, const unsigned char *restrict const Aptr,
                       const unsigned char *restrict const Bptr) {
  fragment<wmma::accumulator, m, n, k, int> C;
  // fill_fragment(C, 0);
  load_matrix_sync(C, Cptr, n, mem_row_major);

  // A must be row major
  fragment<wmma::matrix_a, m, n, k, experimental::precision::s4, row_major> A;
  load_matrix_sync(A, Aptr, k);

  // B must be column major
  fragment<wmma::matrix_b, m, n, k, experimental::precision::s4, col_major> B;
  load_matrix_sync(B, Bptr, k);

  // Multiply
  mma_sync(C, A, B, C);

  store_matrix_sync(Cptr, C, n, mem_row_major);
}

#define CHECK_RESULT(err) check_result(__FILE__, __LINE__, err)
void check_result(const char *file, int line, cudaError_t err) {
  if (err != cudaSuccess) {
    cerr << file << ":" << line << ": CUDA error " << err << ": " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err)
         << "\n";
    exit(1);
  }
}

int main(int argc, char **argv) {
  cout << "matmul.cuda\n";

  vector<unsigned char> Avec(m * k / 2);
  vector<unsigned char> Bvec(k * n / 2, 0x11);
  vector<int> Cvec(m * n);

  const auto Aget = [&](size_t r, size_t c) {
    assert(r < m && c < k);
    auto val = Avec.at((c + k * r) / 2);
    return c % 2 == 0 ? val & 0x0f : val >> 4;
  };
  const auto Aset = [&](size_t r, size_t c, unsigned char val) {
    assert(r < m && c < k);
    assert(val <= 0x0f);
    auto &ref = Avec.at((c + k * r) / 2);
    if (c % 2 == 0)
      ref = (ref & 0xf0) | val;
    else
      ref = (ref & 0x0f) | (val << 4);
  };

  const auto Bget = [&](size_t r, size_t c) {
    assert(r < k && c < n);
    auto val = Bvec.at((r + k * c) / 2);
    return r % 2 == 0 ? val & 0x0f : val >> 4;
  };
  const auto Bset = [&](size_t r, size_t c, unsigned char val) {
    assert(r < k && c < n);
    assert(val <= 0x0f);
    auto &ref = Bvec.at((r + k * c) / 2);
    if (r % 2 == 0)
      ref = (ref & 0xf0) | val;
    else
      ref = (ref & 0x0f) | (val << 4);
  };

  const auto Cget = [&](size_t r, size_t c) {
    assert(r < m && c < n);
    return Cvec.at(c + n * r);
  };
  const auto Cset = [&](size_t r, size_t c, int val) {
    assert(r < m && c < n);
    Cvec.at(c + n * r) = val;
  };

  Aset(2, 2, 3);
  Bset(2, 1, 2);
  Cset(4, 5, 4);

  cout << "A:\n";
  for (size_t r = 0; r < m; ++r) {
    cout << "  ";
    for (size_t c = 0; c < k; ++c) {
      cout << Aget(r, c) << " ";
    }
    cout << "\n";
  }

  cout << "B:\n";
  for (size_t r = 0; r < k; ++r) {
    cout << "  ";
    for (size_t c = 0; c < n; ++c) {
      cout << Bget(r, c) << " ";
    }
    cout << "\n";
  }

  cout << "C:\n";
  for (size_t r = 0; r < m; ++r) {
    cout << "  ";
    for (size_t c = 0; c < n; ++c) {
      cout << Cget(r, c) << " ";
    }
    cout << "\n";
  }

  unsigned char *Aptr = nullptr;
  cudaMalloc(&Aptr, Avec.size() * sizeof *Avec.data());
  cudaMemcpy(Aptr, Avec.data(), Avec.size() * sizeof *Avec.data(), cudaMemcpyHostToDevice);
  unsigned char *Bptr = nullptr;
  cudaMalloc(&Bptr, Bvec.size() * sizeof *Bvec.data());
  cudaMemcpy(Bptr, Bvec.data(), Bvec.size() * sizeof *Bvec.data(), cudaMemcpyHostToDevice);
  int *Cptr = nullptr;
  cudaMalloc(&Cptr, Cvec.size() * sizeof *Cvec.data());
  cudaMemcpy(Cptr, Cvec.data(), Cvec.size() * sizeof *Cvec.data(), cudaMemcpyHostToDevice);

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const dim3 numBlocks(1);
  const dim3 threadsPerBlock(32);
  matmul<<<numBlocks, threadsPerBlock>>>(Cptr, Aptr, Bptr);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Aptr);
  cudaFree(Bptr);
  cudaMemcpy(Cvec.data(), Cptr, Cvec.size() * sizeof *Cvec.data(), cudaMemcpyDeviceToHost);
  cudaFree(Cptr);

  cout << "C:\n";
  for (size_t r = 0; r < m; ++r) {
    cout << "  ";
    for (size_t c = 0; c < n; ++c) {
      cout << Cget(r, c) << " ";
    }
    cout << "\n";
  }

  cout << "Done.\n";
  return 0;
}
