// -*-c++-*-
// Beamforming with CUDA

#include "adler32.h"
#include "arraysizes.hxx"
#include "icomplex4.hxx"

#include <mma.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

using namespace nvcuda;
using namespace nvcuda::wmma;

// E[time][frequency][dish][polarization][complex]
// A[frequency][beam][dish][complex]
// J[beam][frequency][polarization][time][complex]
// G[frequency][beam][complex]

__device__ int32_t extract_real(const int32_t x0, const int32_t x1) {
  return ((uint32_t)(x0 & 0xf0f0f0f0U) >> 4) | (x1 & 0xf0f0f0f0U);
}

__device__ int32_t extract_imag(const int32_t x0, const int32_t x1) {
  return (x0 & 0x0f0f0f0fU) | ((uint32_t)(x1 & 0x0f0f0f0fU) << 4);
}

__global__ void form_beams(ucomplex4 *restrict const Jarray, const ucomplex4 *restrict const Earray,
                           const ucomplex4 *restrict const Aarray, const float *restrict Garray) {
  // operation:    D = A * B + C
  // matrix sizes: C, D: [m,n]
  //               A:    [m,k]   (row major)
  //               B:    [k,n]   (column major)
  //               D[m,n] = C[m,n] + A[m,k] B[k,n]
  // consequences: m ranges over beams
  //               k ranges over dishes
  //               n ranges over times
  //               A[m,k]: A[beam][dish]
  //               B[k,n]: E[dish][time]
  // Nothing ranges over frequencies since all of E, A, and J depend
  // on frequencies.

  // Handling the polarizations and complex numbers is not
  // straightforward. We calculate the following:
  //
  //   J += A * E
  //
  //   Jre += Are * Ere - Aim * Eim
  //   Jim += Are * Eim + Aim * Ere
  //
  //   J+re += Are * E+re - Aim * E+im
  //   J+im += Are * E+im + Aim * E+re
  //   J-re += Are * E-re - Aim * E-im
  //   J-im += Are * E-im + Aim * E-re

  // These sizes are dictated by CUDA
  constexpr int m = 8;  // beams
  constexpr int n = 8;  // times
  constexpr int k = 32; // dishes
  static_assert(nbeams % m == 0, "");
  static_assert(ndishes % k == 0, "");
  static_assert(ntimes % n == 0, "");

  constexpr size_t J_num_storage_elements = 2;
  static_assert(m % J_num_storage_elements == 0);

  assert(blockDim.x == m / J_num_storage_elements);
  assert(blockDim.y == n);
  assert(nbeams / m % blockDim.z == 0);
  assert(blockDim.z >= 1);

  const size_t f = blockIdx.x;
  for (size_t b0 = 0; b0 < nbeams; b0 += blockDim.z * m) {
    const size_t b = b0 + threadIdx.z * m;
    for (size_t t = 0; t < ntimes; t += n) {

      // rawJ[2]
      // rawJ[2][m][n] = rawJ[polarization][beam][time]
      fragment<wmma::accumulator, m, n, k, int32_t> rawJre[npolarizations], rawJreNeg[npolarizations], rawJim[npolarizations];
      for (size_t p = 0; p < npolarizations; ++p) {
        fill_fragment(rawJre[p], 0);
        fill_fragment(rawJreNeg[p], 0);
        fill_fragment(rawJim[p], 0);
      }

      for (size_t d = 0; d < ndishes; d += k) {

        ////////////////////////////////////////////////////////////////////////////////

        // A[beam][dish]
        // wmma::A[m][k]   (must be row major)
        fragment<wmma::matrix_a, m, n, k, experimental::precision::s4, row_major> A[ncomplex];
        for (size_t c = 0; c < ncomplex; ++c) {
          load_matrix_sync(A[c], &Aarray[Alinear(f, b, d + c * k / 2, 0) / 2], ndishes * ncomplex);
        }

        // A[beam][dish]
        // wmma::A[m][k]   (must be row major)
        fragment<wmma::matrix_a, m, n, k, experimental::precision::s4, row_major> Are, Aim;
        static_assert(Are.num_storage_elements == 1, "");
        for (int i = 0; i < Are.num_storage_elements; ++i) {
          Are.x[i] = extract_real(A[0].x[i], A[1].x[i]) ^ 0x88888888U;
          Aim.x[i] = extract_imag(A[0].x[i], A[1].x[i]) ^ 0x88888888U;
        }

        ////////////////////////////////////////////////////////////////////////////////

        // E[time][dish]
        // wmma::B[k][n]   (must column major)
        fragment<wmma::matrix_b, m, n, k, experimental::precision::s4, col_major> E[ncomplex][npolarizations];
        for (size_t c = 0; c < ncomplex; ++c) {
          for (size_t p = 0; p < npolarizations; ++p) {
            load_matrix_sync(E[c][p], &Earray[Elinear(t, f, d + (2 * c + p) * k / 4, 0, 0) / 2],
                             nfrequencies * ndishes * npolarizations * ncomplex);
          }
        }
        // We have:
        //   [L]E[c][p] = t=L/4, d=10c+8p+2(L%4)+j/2, p=j%2
        // Example:
        //   [0]E[0][p] = t=0, d=8p+0+j/2, p=j%2
        //   [1]E[0][p] = t=0, d=8p+2+j/2, p=j%2
        //   [2]E[0][p] = t=0, d=8p+4+j/2, p=j%2
        //   [3]E[0][p] = t=0, d=8p+6+j/2, p=j%2
        // Example:
        //   [0]E[0][p] = [d=8p+1,p=1   d=8p+1,p=0   d=8p+0,p=1   d=8p+0,p=0]
        //   [1]E[0][p] = [d=8p+3,p=1   d=8p+3,p=0   d=8p+2,p=1   d=8p+2,p=0]
        //   [2]E[0][p] = [d=8p+5,p=1   d=8p+5,p=0   d=8p+4,p=1   d=8p+4,p=0]
        //   [3]E[0][p] = [d=8p+7,p=1   d=8p+7,p=0   d=8p+6,p=1   d=8p+6,p=0]
        // Destination lanes:
        //   [0]E[0][0] = [0 0 0 0]
        //   [0]E[0][1] = [2 2 2 2]
        //   [1]E[0][0] = [0 0 0 0]
        //   [1]E[0][1] = [2 2 2 2]
        //   [2]E[0][0] = [1 1 1 1]
        //   [2]E[0][1] = [3 3 3 3]
        //   [3]E[0][0] = [1 1 1 1]
        //   [3]E[0][1] = [3 3 3 3]

        // We need:
        //   [L]E[c][p] = t=L/4, d=10c+4(L%4)+j, p=p
        // Example:
        //   [0]E[0][p] = t=0, d=0+j, p=p
        //   [1]E[0][p] = t=0, d=4+j, p=p
        //   [2]E[0][p] = t=0, d=8+j, p=p
        //   [3]E[0][p] = t=0, d=c+j, p=p
        // Example:
        //   [0]E[0][p] = [d=3   d=2   d=1   d=0]
        //   [1]E[0][p] = [d=7   d=6   d=5   d=4]
        //   [2]E[0][p] = [d=b   d=a   d=9   d=8]
        //   [3]E[0][p] = [d=f   d=e   d=d   d=c]

        // We need to shuffle data within groups of 4 lanes to separate the
        // polarizations
        fragment<wmma::matrix_b, m, n, k, experimental::precision::s4, col_major> Ep[ncomplex][npolarizations];
        const int32_t L = threadIdx.x & 0x03;
        for (size_t c = 0; c < ncomplex; ++c) {
          for (size_t i = 0; i < E[c][0].num_storage_elements; ++i) {
#if 0
            switch (L) {
            case 0: {
              int32_t E_L0_c0 = E[c][0].x[i];
              int32_t E_L0_c1 = E[c][1].x[i];
              int32_t E_L2_c0 = __shfl_xor_sync(0xffffffffU, E_L0_c1, 0x02);
              int32_t E_L1_c0 = __shfl_xor_sync(0xffffffffU, E_L2_c0, 0x01);
              Ep[c][0].x[i] = __byte_perm(E_L0_c0, E_L1_c0, 0x6420);
              Ep[c][1].x[i] = __byte_perm(E_L0_c0, E_L1_c0, 0x7531);
              break;
            }
            case 1: {
              int32_t E_L1_c0 = E[c][0].x[i];
              int32_t E_L1_c1 = E[c][1].x[i];
              int32_t E_L3_c0 = __shfl_xor_sync(0xffffffffU, E_L1_c1, 0x02);
              int32_t E_L2_c0 = __shfl_xor_sync(0xffffffffU, E_L1_c0, 0x01);
              Ep[c][0].x[i] = __byte_perm(E_L2_c0, E_L3_c0, 0x6420);
              Ep[c][1].x[i] = __byte_perm(E_L2_c0, E_L3_c0, 0x7531);
              break;
            }
            case 2: {
              int32_t E_L2_c0 = E[c][0].x[i];
              int32_t E_L2_c1 = E[c][1].x[i];
              int32_t E_L0_c1 = __shfl_xor_sync(0xffffffffU, E_L2_c0, 0x02);
              int32_t E_L1_c1 = __shfl_xor_sync(0xffffffffU, E_L2_c1, 0x01);
              Ep[c][0].x[i] = __byte_perm(E_L0_c1, E_L1_c1, 0x6420);
              Ep[c][1].x[i] = __byte_perm(E_L0_c1, E_L1_c1, 0x7531);
              break;
            }
            case 3: {
              int32_t E_L3_c0 = E[c][0].x[i];
              int32_t E_L3_c1 = E[c][1].x[i];
              int32_t E_L1_c1 = __shfl_xor_sync(0xffffffffU, E_L3_c0, 0x02);
              int32_t E_L2_c1 = __shfl_xor_sync(0xffffffffU, E_L1_c1, 0x01);
              Ep[c][0].x[i] = __byte_perm(E_L2_c1, E_L3_c1, 0x6420);
              Ep[c][1].x[i] = __byte_perm(E_L2_c1, E_L3_c1, 0x7531);
              break;
            }
            }
#else
            int32_t E0 = E[c][0].x[i];
            int32_t E1 = E[c][1].x[i];
            const auto select4 = [=](int a, int b, int c, int d) { return L == 0 ? a : L == 1 ? b : L == 2 ? c : d; };
            int32_t E2 = __shfl_xor_sync(0xffffffffU, select4(E1, E1, E0, E0), 0x02);
            int32_t E3 = __shfl_xor_sync(0xffffffffU, select4(E2, E0, E1, E2), 0x01);
            Ep[c][0].x[i] = __byte_perm(select4(E0, E3, E2, E3), select4(E3, E2, E3, E1), 0x6420);
            Ep[c][1].x[i] = __byte_perm(select4(E0, E3, E2, E3), select4(E3, E2, E3, E1), 0x7531);
#endif
          }
        }

        // E[time][dish]
        // wmma::B[k][n]   (must be row major)
        fragment<wmma::matrix_b, m, n, k, experimental::precision::s4, col_major> Ere[npolarizations], Eim[npolarizations];
        for (size_t p = 0; p < npolarizations; ++p) {
          static_assert(Ere[p].num_storage_elements == 1, "");
          for (int i = 0; i < Ere[0].num_storage_elements; ++i) {
            Ere[p].x[i] = extract_real(Ep[0][p].x[i], Ep[1][p].x[i]) ^ 0x88888888U;
            Eim[p].x[i] = extract_imag(Ep[0][p].x[i], Ep[1][p].x[i]) ^ 0x88888888U;
          }
        }

        ////////////////////////////////////////////////////////////////////////////////

        // Multiply
        for (size_t p = 0; p < npolarizations; ++p) {
          mma_sync(rawJre[p], Are, Ere[p], rawJre[p]);
          mma_sync(rawJreNeg[p], Aim, Eim[p], rawJreNeg[p]);
          mma_sync(rawJim[p], Are, Eim[p], rawJim[p]);
          mma_sync(rawJim[p], Aim, Ere[p], rawJim[p]);
        }

      } // for dish

      for (size_t p = 0; p < npolarizations; ++p) {
        static_assert(rawJre[p].num_storage_elements == J_num_storage_elements, "");
        for (int i = 0; i < rawJre[0].num_storage_elements; ++i) {
          const size_t t1 = J_num_storage_elements * threadIdx.x + i;
          const size_t b1 = threadIdx.y;
          const float G = Garray[Glinear(f, b + b1)];
          int Jre = max(-7, min(7, int(lrint(G * float(rawJre[p].x[i] - rawJreNeg[p].x[i])))));
          int Jim = max(-7, min(7, int(lrint(G * float(rawJim[p].x[i])))));
          Jarray[Jlinear(b + b1, f, p, t + t1, 0) / 2] = ucomplex4(Jre, Jim);
        }
      }

    } // for time
  }   // for beam
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
  cout << "beamforming.cuda\n";

  vector<ucomplex4> Earray;
  vector<ucomplex4> Aarray;
  vector<float> Garray;
  vector<ucomplex4> Jarray;
  setup(Earray, Aarray, Garray, Jarray);

  cout << "Forming beams...\n";
  ucomplex4 *Earray2 = nullptr;
  cudaMalloc(&Earray2, Earray.size() * sizeof(ucomplex4));
  cudaMemcpy(Earray2, Earray.data(), Earray.size() * sizeof(ucomplex4), cudaMemcpyHostToDevice);
  ucomplex4 *Aarray2 = nullptr;
  cudaMalloc(&Aarray2, Aarray.size() * sizeof(ucomplex4));
  cudaMemcpy(Aarray2, Aarray.data(), Aarray.size() * sizeof(ucomplex4), cudaMemcpyHostToDevice);
  float *Garray2 = nullptr;
  cudaMalloc(&Garray2, Garray.size() * sizeof(float));
  cudaMemcpy(Garray2, Garray.data(), Garray.size() * sizeof(float), cudaMemcpyHostToDevice);
  ucomplex4 *Jarray2 = nullptr;
  cudaMalloc(&Jarray2, Jarray.size() * sizeof(ucomplex4));

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);

  const auto t0 = gettime();

  const int m = 8;
  const int n = 8;
  const dim3 numBlocks(nfrequencies);
  const dim3 threadsPerBlock(m / 2, n, 16); // 16 seems optimal
  form_beams<<<numBlocks, threadsPerBlock>>>(Jarray2, Earray2, Aarray2, Garray2);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);

  const auto t1 = gettime();
  cout << "Elapsed time: " << (t1 - t0) << " seconds\n";

  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Earray2);
  Earray2 = nullptr;
  cudaFree(Aarray2);
  Aarray2 = nullptr;
  cudaFree(Garray2);
  Garray2 = nullptr;
  cudaMemcpy(Jarray.data(), Jarray2, Jarray.size() * sizeof(ucomplex4), cudaMemcpyDeviceToHost);
  cudaFree(Jarray2);
  Jarray2 = nullptr;

  check(Jarray);

  cout << "Done.\n";
  return 0;
}
