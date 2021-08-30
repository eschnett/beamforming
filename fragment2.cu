// -*-c++-*-
// Examine CUDA fragment layout

#include <mma.h>

#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace std;

using namespace nvcuda;
using namespace nvcuda::wmma;

template <typename T> struct nbits : integral_constant<size_t, 8 * sizeof(T)> {};
template <> struct nbits<wmma::experimental::precision::s4> : integral_constant<size_t, 4> {};
template <> struct nbits<wmma::experimental::precision::u4> : integral_constant<size_t, 4> {};
template <> struct nbits<wmma::experimental::precision::b1> : integral_constant<size_t, 1> {};

// A[m][k]   must be row major
// B[k][n]   must be column major
// C[m][n]   row major
// C[m][n] += A[m][k] * B[k][n]

template <typename T, int m, int n, int k>
__global__ void load_matrix_a_u4(unsigned *restrict const Fptr, const unsigned *restrict const Aptr) {
  fragment<wmma::matrix_a, m, n, k, T, row_major> A;

  // Load fragment
  load_matrix_sync(A, Aptr, k); // mem_row_major

  // Decompose fragment
  static_assert(A.num_storage_elements == 1);
  for (int t = 0; t < A.num_storage_elements; ++t) {
    Fptr[A.num_storage_elements * threadIdx.x + t] = A.x[t];
  }
}

template <typename T, int m, int n, int k, typename layout>
__global__ void load_matrix_a(T *restrict const Fptr, const T *restrict const Aptr) {
  fragment<wmma::matrix_a, m, n, k, T, layout> A;

  // Load fragment
  load_matrix_sync(A, Aptr, is_same_v<layout, row_major> ? k : m);

  // Decompose fragment
  static_assert(A.num_elements == m * k / 32);
  // for (int t = 0; t < A.num_elements; ++t) {
  //   Fptr[A.num_elements * threadIdx.x + t] = A.x[t];
  // }
  const int num_int_elements = A.num_elements * sizeof(A.x[0]) / sizeof(int);
  struct int_elements {
    int x[num_int_elements];
  };
  for (int t = 0; t < num_int_elements; ++t) {
    ((int *)Fptr)[num_int_elements * threadIdx.x + t] = (*(int_elements *)&A).x[t];
  }
}

template <typename T, int m, int n, int k>
__global__ void load_matrix_b_u4(unsigned *restrict const Fptr, const unsigned *restrict const Bptr) {
  fragment<wmma::matrix_b, m, n, k, T, col_major> B;

  // Load fragment
  load_matrix_sync(B, Bptr, k); // mem_col_major

  // Decompose fragment
  static_assert(B.num_storage_elements == 1);
  for (int t = 0; t < B.num_storage_elements; ++t) {
    Fptr[B.num_storage_elements * threadIdx.x + t] = B.x[t];
  }
}

template <typename T, int m, int n, int k, typename layout>
__global__ void load_matrix_b(T *restrict const Fptr, const T *restrict const Bptr) {
  fragment<wmma::matrix_b, m, n, k, T, layout> B;

  // Load fragment
  load_matrix_sync(B, Bptr, is_same_v<layout, row_major> ? n : k);

  // Decompose fragment
  // static_assert(B.num_elements == n * k / 32);
  // for (int t = 0; t < B.num_elements; ++t) {
  //   Fptr[B.num_elements * threadIdx.x + t] = B.x[t];
  // }
  const int num_int_elements = B.num_elements * sizeof(B.x[0]) / sizeof(int);
  struct int_elements {
    int x[num_int_elements];
  };
  for (int t = 0; t < num_int_elements; ++t) {
    ((int *)Fptr)[num_int_elements * threadIdx.x + t] = (*(int_elements *)&B).x[t];
  }
}

template <int m, int n, int k> __global__ void load_accumulator(int *restrict const Fptr, const int *restrict const Cptr) {
  fragment<wmma::accumulator, m, n, k, int> C;

  // Load fragment
  load_matrix_sync(C, Cptr, n, mem_row_major);

  // Decompose fragment
  static_assert(32 * C.num_elements == m * n);
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

template <typename T, int m, int n, int k> void examine_matrix_a_u4() {
  cout << "\n"
       << "matrix_a [bits=" << nbits<T>::value << ",m=" << m << ",n=" << n << ",k=" << k << "]:\n";

  vector<unsigned> Amat(m * k / 8);
  vector<unsigned> frag(m * k / 8);

  const auto elt = [&](vector<unsigned> &A, size_t r, size_t c) -> unsigned & {
    assert(r < m && c < k / 8);
    return A.at(c + k / 8 * r);
  };

  for (size_t r = 0; r < m; ++r) {
    for (size_t c = 0; c < k / 8; ++c) {
      elt(Amat, r, c) = k * r + 8 * c;
    }
  }

  cout << "  Amat:\n";
  for (size_t r = 0; r < m; ++r) {
    cout << "    ";
    for (size_t c = 0; c < k / 8; ++c) {
      cout << setw(2) << setfill('0') << hex << elt(Amat, r, c) << dec << " ";
    }
    cout << "\n";
  }

  unsigned *Aptr = nullptr;
  cudaMalloc(&Aptr, Amat.size() * sizeof *Amat.data());
  cudaMemcpy(Aptr, Amat.data(), Amat.size() * sizeof *Amat.data(), cudaMemcpyHostToDevice);

  unsigned *Fptr = nullptr;
  cudaMalloc(&Fptr, Amat.size() * sizeof *Amat.data());

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const dim3 numBlocks(1);
  const dim3 threadsPerBlock(32);
  load_matrix_a_u4<T, m, n, k><<<numBlocks, threadsPerBlock>>>(Fptr, Aptr);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Aptr);

  cudaMemcpy(frag.data(), Fptr, Amat.size() * sizeof *Amat.data(), cudaMemcpyDeviceToHost);
  cudaFree(Fptr);

  cout << "  fragmentA:\n";
  for (size_t thread = 0; thread < 32; ++thread) {
    cout << "    ";
    for (size_t reg = 0; reg < m * k / 8 / 32; ++reg) {
      cout << setw(2) << setfill('0') << hex << unsigned(frag.at(thread * m * k / 8 / 32 + reg)) << dec << " ";
    }
    cout << "\n";
  }
}

template <typename T, int m, int n, int k, typename layout> void examine_matrix_a() {
  cout << "\n"
       << "matrix_a [bits=" << nbits<T>::value << ",m=" << m << ",n=" << n << ",k=" << k
       << ",layout=" << (is_same_v<layout, row_major> ? "row_major" : "col_major") << "]:\n";

  vector<T> Amat(m * k);
  vector<T> frag(m * k);

  const auto elt = [&](vector<T> &A, size_t r, size_t c) -> T & {
    assert(r < m && c < k);
    if (is_same_v<layout, row_major>) {
      return A.at(c + k * r);
    } else {
      return A.at(r + m * c);
    }
  };

  for (size_t r = 0; r < m; ++r) {
    for (size_t c = 0; c < k; ++c) {
      if (is_same_v<layout, row_major>) {
        elt(Amat, r, c) = k * r + c;
        // elt(Amat, r, c) = (k * r + c) / 2;
      } else {
        elt(Amat, r, c) = m * c + r;
        // elt(Amat, r, c) = (m * c + r) / 2;
      }
    }
  }

  cout << "  Amat:\n";
  for (size_t r = 0; r < m; ++r) {
    cout << "    ";
    for (size_t c = 0; c < k; ++c) {
      cout << setw(2) << setfill('0') << hex << unsigned(elt(Amat, r, c)) << dec << " ";
    }
    cout << "\n";
  }

  T *Aptr = nullptr;
  cudaMalloc(&Aptr, Amat.size() * sizeof *Amat.data());
  cudaMemcpy(Aptr, Amat.data(), Amat.size() * sizeof *Amat.data(), cudaMemcpyHostToDevice);

  T *Fptr = nullptr;
  cudaMalloc(&Fptr, Amat.size() * sizeof *Amat.data());

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const dim3 numBlocks(1);
  const dim3 threadsPerBlock(32);
  load_matrix_a<T, m, n, k, layout><<<numBlocks, threadsPerBlock>>>(Fptr, Aptr);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Aptr);

  cudaMemcpy(frag.data(), Fptr, Amat.size() * sizeof *Amat.data(), cudaMemcpyDeviceToHost);
  cudaFree(Fptr);

  cout << "  fragmentA:\n";
  for (size_t thread = 0; thread < 32; ++thread) {
    cout << "    ";
    for (size_t reg = 0; reg < m * k / 32; ++reg) {
      cout << setw(2) << setfill('0') << hex << unsigned(frag.at(thread * m * k / 32 + reg)) << dec << " ";
    }
    cout << "\n";
  }
}

template <typename T, int m, int n, int k> void examine_matrix_b_u4() {
  cout << "\n"
       << "matrix_b [bits=" << nbits<T>::value << ",m=" << m << ",n=" << n << ",k=" << k << "]:\n";

  vector<unsigned> Bmat(n * k / 8);
  vector<unsigned> frag(n * k / 8);

  const auto elt = [&](vector<unsigned> &B, size_t r, size_t c) -> unsigned & {
    assert(r < k / 8 && c < n);
    return B.at(r + k / 8 * c);
  };

  for (size_t r = 0; r < k / 8; ++r) {
    for (size_t c = 0; c < n; ++c) {
      elt(Bmat, r, c) = r + k * c;
    }
  }

  cout << "  Bmat:\n";
  for (size_t r = 0; r < k / 8; ++r) {
    cout << "    ";
    for (size_t c = 0; c < n; ++c) {
      cout << setw(2) << setfill('0') << hex << elt(Bmat, r, c) << dec << " ";
    }
    cout << "\n";
  }

  unsigned *Bptr = nullptr;
  cudaMalloc(&Bptr, Bmat.size() * sizeof *Bmat.data());
  cudaMemcpy(Bptr, Bmat.data(), Bmat.size() * sizeof *Bmat.data(), cudaMemcpyHostToDevice);

  unsigned *Fptr = nullptr;
  cudaMalloc(&Fptr, Bmat.size() * sizeof *Bmat.data());

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const dim3 numBlocks(1);
  const dim3 threadsPerBlock(32);
  load_matrix_b_u4<T, m, n, k><<<numBlocks, threadsPerBlock>>>(Fptr, Bptr);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Bptr);

  cudaMemcpy(frag.data(), Fptr, Bmat.size() * sizeof *Bmat.data(), cudaMemcpyDeviceToHost);
  cudaFree(Fptr);

  cout << "  fragmentB:\n";
  for (size_t thread = 0; thread < 32; ++thread) {
    cout << "    ";
    for (size_t reg = 0; reg < n * k / 8 / 32; ++reg) {
      cout << setw(2) << setfill('0') << hex << unsigned(frag.at(thread * n * k / 8 / 32 + reg)) << dec << " ";
    }
    cout << "\n";
  }
}

template <typename T, int m, int n, int k, typename layout> void examine_matrix_b() {
  cout << "\n"
       << "matrix_b [bits=" << nbits<T>::value << ",m=" << m << ",n=" << n << ",k=" << k
       << ",layout=" << (is_same_v<layout, row_major> ? "row_major" : "col_major") << "]:\n";

  vector<T> Bmat(k * n);
  vector<T> frag(k * n);

  const auto elt = [&](vector<T> &B, size_t r, size_t c) -> T & {
    assert(r < k && c < n);
    if (is_same_v<layout, row_major>) {
      return B.at(c + n * r);
    } else {
      return B.at(r + k * c);
    }
  };

  for (size_t r = 0; r < k; ++r) {
    for (size_t c = 0; c < n; ++c) {
      if constexpr (is_same_v<layout, row_major>) {
        elt(Bmat, r, c) = n * r + c;
        // elt(Bmat, r, c) = (n * r + c) / 2;
      } else {
        elt(Bmat, r, c) = k * c + r;
        // elt(Bmat, r, c) = (k * c + r) / 2;
      }
    }
  }

  cout << "  Bmat:\n";
  for (size_t r = 0; r < k; ++r) {
    cout << "    ";
    for (size_t c = 0; c < n; ++c) {
      cout << setw(2) << setfill('0') << hex << unsigned(elt(Bmat, r, c)) << dec << " ";
    }
    cout << "\n";
  }

  T *Bptr = nullptr;
  cudaMalloc(&Bptr, Bmat.size() * sizeof *Bmat.data());
  cudaMemcpy(Bptr, Bmat.data(), Bmat.size() * sizeof *Bmat.data(), cudaMemcpyHostToDevice);

  T *Fptr = nullptr;
  cudaMalloc(&Fptr, Bmat.size() * sizeof *Bmat.data());

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const dim3 numBlocks(1);
  const dim3 threadsPerBlock(32);
  load_matrix_b<T, m, n, k, layout><<<numBlocks, threadsPerBlock>>>(Fptr, Bptr);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Bptr);

  cudaMemcpy(frag.data(), Fptr, Bmat.size() * sizeof *Bmat.data(), cudaMemcpyDeviceToHost);
  cudaFree(Fptr);

  cout << "  fragmentB:\n";
  for (size_t thread = 0; thread < 32; ++thread) {
    cout << "    ";
    for (size_t reg = 0; reg < n * k / 32; ++reg) {
      cout << setw(2) << setfill('0') << hex << unsigned(frag.at(thread * n * k / 32 + reg)) << dec << " ";
    }
    cout << "\n";
  }
}

template <int m, int n, int k> void examine_accumulator() {
  cout << "\n"
       << "accumulator [m=" << m << ",n=" << n << ",k=" << k << "]:\n";

  vector<int> Cmat(m * n);
  vector<int> frag(m * n);

  const auto elt = [&](vector<int> &C, size_t r, size_t c) -> int & {
    assert(r < m && c < n);
    return C.at(c + n * r);
  };

  for (size_t r = 0; r < m; ++r) {
    for (size_t c = 0; c < n; ++c) {
      elt(Cmat, r, c) = n * r + c;
    }
  }

  cout << "  Cmat:\n";
  for (size_t r = 0; r < m; ++r) {
    cout << "    ";
    for (size_t c = 0; c < n; ++c) {
      cout << setw(2) << setfill('0') << hex << elt(Cmat, r, c) << dec << " ";
    }
    cout << "\n";
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
  load_accumulator<m, n, k><<<numBlocks, threadsPerBlock>>>(Fptr, Cptr);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Cptr);

  cudaMemcpy(frag.data(), Fptr, Cmat.size() * sizeof *Cmat.data(), cudaMemcpyDeviceToHost);
  cudaFree(Fptr);

  cout << "  fragmentC:\n";
  for (size_t thread = 0; thread < 32; ++thread) {
    cout << "    ";
    for (size_t reg = 0; reg < m * n / 32; ++reg) {
      cout << setw(2) << setfill('0') << hex << frag.at(thread * m * n / 32 + reg) << dec << " ";
    }
    cout << "\n";
  }
}

int main(int argc, char **argv) {
  cout << "fragment2.cuda\n";

  // // s4: m=8, n=8, k=32
  examine_matrix_a_u4<wmma::experimental::precision::u4, 8, 8, 32>();
  examine_matrix_b_u4<wmma::experimental::precision::u4, 8, 8, 32>();
  examine_accumulator<8, 8, 32>();

  // int8_t: m=16, n=16, k=16
  examine_matrix_a<uint8_t, 16, 16, 16, row_major>();
  examine_matrix_a<uint8_t, 16, 16, 16, col_major>();
  examine_matrix_b<uint8_t, 16, 16, 16, row_major>();
  examine_matrix_b<uint8_t, 16, 16, 16, col_major>();
  examine_accumulator<16, 16, 16>();

  // int8_t: m=32, n=8, k=16
  examine_matrix_a<uint8_t, 32, 8, 16, row_major>();
  examine_matrix_a<uint8_t, 32, 8, 16, col_major>();
  examine_matrix_b<uint8_t, 32, 8, 16, row_major>();
  examine_matrix_b<uint8_t, 32, 8, 16, col_major>();
  examine_accumulator<32, 8, 16>();

  // int8_t: m=8, n=32, k=16
  examine_matrix_a<uint8_t, 8, 32, 16, row_major>();
  examine_matrix_a<uint8_t, 8, 32, 16, col_major>();
  examine_matrix_b<uint8_t, 8, 32, 16, row_major>();
  examine_matrix_b<uint8_t, 8, 32, 16, col_major>();
  examine_accumulator<8, 32, 16>();

  cout << "Done.\n";
  return 0;
}
