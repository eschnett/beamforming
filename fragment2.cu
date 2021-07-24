// -*-c++-*-
// Examine CUDA fragment layout

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

__global__ void examine_framgent(int *restrict const Fptr, const int *restrict const Cptr) {
  fragment<wmma::accumulator, m, n, k, int> C;

  // Load fragment
  load_matrix_sync(C, Cptr, n, mem_row_major);

  // Decompose fragment
  static_assert(C.num_elements == 2);
  for (int t = 0; t < C.num_elements; ++t) {
    Fptr[C.num_elements * threadIdx.x + t] = C.x[t];
  }
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
  cout << "fragment2.cuda\n";

  vector<int> Cmat(m * n);
  vector<int> frag(m * n);

  const auto elt = [&](vector<int> &A, size_t r, size_t c) -> int & {
    assert(r < m && c < n);
    return A.at(c + n * r);
  };

  for (size_t r = 0; r < m; ++r) {
    for (size_t c = 0; c < n; ++c) {
      elt(Cmat, r, c) = n * r + c;
    }
  }

  int *Cptr = nullptr;
  cudaMalloc(&Cptr, Cmat.size() * sizeof *Cmat.data());
  cudaMemcpy(Cptr, Cmat.data(), Cmat.size() * sizeof *Cmat.data(), cudaMemcpyHostToDevice);

  int *Fptr = nullptr;
  cudaMalloc(&Fptr, Cmat.size() * sizeof *Cmat.data());

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const dim3 numBlocks(1);
  const dim3 threadsPerBlock(32);
  examine_framgent<<<numBlocks, threadsPerBlock>>>(Fptr, Cptr);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Cptr);

  cudaMemcpy(frag.data(), Fptr, Cmat.size() * sizeof *Cmat.data(), cudaMemcpyDeviceToHost);
  cudaFree(Fptr);

  cout << "fragmentC:\n";
  for (size_t r = 0; r < m; ++r) {
    cout << "  ";
    for (size_t c = 0; c < n; ++c) {
      cout << elt(frag, r, c) << " ";
    }
    cout << "\n";
  }

  cout << "Done.\n";
  return 0;
}
