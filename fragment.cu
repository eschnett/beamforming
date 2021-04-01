// -*-c++-*-
// CUDA fragment layout

#include <mma.h>

#include <cassert>
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

__global__ void matmul(int *restrict const Cptr) {
  fragment<wmma::accumulator, m, n, k, int> C;

  // Set fragment
  for (int t = 0; t < C.num_elements; t++)
    if (t % 2 == 0)
      C.x[t] = 2 * threadIdx.x + 0;
    else
      C.x[t] = 2 * threadIdx.x + 1;

  // Store to global memory
  store_matrix_sync(Cptr, C, n, mem_row_major);

  // // Store to local memory (not supported by wmma)
  // int Ctmp[m * n];
  // store_matrix_sync(Ctmp, C, n, mem_row_major);

  // Cptr[2 * threadIdx.x + 0] = Ctmp[2 * threadIdx.x + 0];
  // Cptr[2 * threadIdx.x + 1] = Ctmp[2 * threadIdx.x + 1];
}

__global__ void matmul2(int *restrict const Cptr) {
  fragment<wmma::accumulator, m, n, k, int> C;
  // Set fragment
  for (int t = 0; t < C.num_elements; t++)
    C.x[t] = 0;

  fragment<wmma::matrix_a, m, n, k, experimental::precision::s4, row_major> A;
  // Set fragment
  static_assert(A.num_elements == 8 * A.num_storage_elements, "");
  for (int t = 0; t < A.num_storage_elements; ++t)
    A.x[t] = (((8 * threadIdx.x + 7) & 0x0f) << 0x1c) |
             (((8 * threadIdx.x + 6) & 0x0f) << 0x18) |
             (((8 * threadIdx.x + 5) & 0x0f) << 0x14) |
             (((8 * threadIdx.x + 4) & 0x0f) << 0x10) |
             (((8 * threadIdx.x + 3) & 0x0f) << 0x0c) |
             (((8 * threadIdx.x + 2) & 0x0f) << 0x08) |
             (((8 * threadIdx.x + 1) & 0x0f) << 0x04) |
             (((8 * threadIdx.x + 0) & 0x0f) << 0x00);

  fragment<wmma::matrix_b, m, n, k, experimental::precision::s4, col_major> B;
  // Set fragment
  for (int t = 0; t < B.num_elements; t++)
    B.x[t] = 1;

  // Store to global memory
  mma_sync(C, A, B, C);
  store_matrix_sync(Cptr, C, n, mem_row_major);
}

#define CHECK_RESULT(err) check_result(__FILE__, __LINE__, err)
void check_result(const char *file, int line, cudaError_t err) {
  if (err != cudaSuccess) {
    cerr << file << ":" << line << ": CUDA error " << err << ": "
         << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << "\n";
    exit(1);
  }
}

int main(int argc, char **argv) {
  cout << "fragment.cuda\n";

  vector<int> Cvec(m * n);

  const auto Cget = [&](size_t r, size_t c) {
    assert(r < m && c < n);
    return Cvec.at(c + n * r);
  };

  int *Cptr = nullptr;
  cudaMalloc(&Cptr, Cvec.size() * sizeof *Cvec.data());
  cudaMemcpy(Cptr, Cvec.data(), Cvec.size() * sizeof *Cvec.data(),
             cudaMemcpyHostToDevice);

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const dim3 numBlocks(1);
  const dim3 threadsPerBlock(32);
  matmul<<<numBlocks, threadsPerBlock>>>(Cptr);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaMemcpy(Cvec.data(), Cptr, Cvec.size() * sizeof *Cvec.data(),
             cudaMemcpyDeviceToHost);
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
