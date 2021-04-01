// -*-c++-*-
// Beamforming with CUDA

#include "adler32.h"
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

__global__ void form_beams(ucomplex4 *restrict const Jarray,
                           const ucomplex4 *restrict const Earray,
                           const ucomplex4 *restrict const Aarray,
                           const float *restrict Garray) {
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

  const size_t f = blockIdx.x;
  for (size_t b = 0; b < nbeams; b += n) {
    for (size_t t = 0; t < ntimes; t += m) {

      // rawJ[2]
      // rawJ[2][m][n] = rawJ[polarization][beam][time]
      fragment<wmma::accumulator, m, n, k, int32_t> rawJre[npolarizations],
          rawJreNeg[npolarizations], rawJim[npolarizations];
      for (size_t p = 0; p < npolarizations; ++p) {
        fill_fragment(rawJre[p], 0);
        fill_fragment(rawJreNeg[p], 0);
        fill_fragment(rawJim[p], 0);
      }

      for (size_t d = 0; d < ndishes; d += k) {

        ////////////////////////////////////////////////////////////////////////////////

        __shared__ __align__(64) unsigned char AreArray[m][k / 2],
            AimArray[m][k / 2];
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          for (size_t b1 = 0; b1 < m; ++b1) {
            for (size_t d1 = 0; d1 < k; d1 += 2) {
              ucomplex4 A0 = Aarray[Alinear(f, b + b1, d + d1 + 0, 0) / 2];
              ucomplex4 A1 = Aarray[Alinear(f, b + b1, d + d1 + 1, 0) / 2];
              AreArray[b1][d1 / 2] = icomplex4(A1.real(), A0.real()).data;
              AimArray[b1][d1 / 2] = icomplex4(A1.imag(), A0.imag()).data;
            }
          }
        }

        // a[beam][dish]
        // wmma::a[m][k]   (must be row major)
        fragment<wmma::matrix_a, m, n, k, experimental::precision::s4,
                 row_major>
            Are, Aim;
        load_matrix_sync(Are, &AreArray[0][0], k);
        load_matrix_sync(Aim, &AimArray[0][0], k);

        ////////////////////////////////////////////////////////////////////////////////

        __shared__ __align__(
            64) unsigned char EreArray[npolarizations][n][k / 2],
            EimArray[npolarizations][n][k / 2];
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          for (size_t p = 0; p < npolarizations; ++p) {
            for (size_t t1 = 0; t1 < n; ++t1) {
              for (size_t d1 = 0; d1 < k; d1 += 2) {
                ucomplex4 E0 = Earray[Elinear(t + t1, f, d + d1 + 0, p, 0) / 2];
                ucomplex4 E1 = Earray[Elinear(t + t1, f, d + d1 + 1, p, 0) / 2];
                EreArray[p][t1][d1 / 2] = icomplex4(E1.real(), E0.real()).data;
                EimArray[p][t1][d1 / 2] = icomplex4(E1.imag(), E0.imag()).data;
              }
            }
          }
        }

        // E[time][dish]
        // wmma::B[k][n]   (must be row major)
        fragment<wmma::matrix_b, m, n, k, experimental::precision::s4,
                 col_major>
            Ere[npolarizations], Eim[npolarizations];
        for (size_t p = 0; p < npolarizations; ++p) {
          load_matrix_sync(Ere[p], &EreArray[p][0][0], k);
          load_matrix_sync(Eim[p], &EimArray[p][0][0], k);
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

      __shared__ __align__(64) int rawJreArray[npolarizations][m][n],
          rawJreNegArray[npolarizations][m][n],
          rawJimArray[npolarizations][m][n];
      for (size_t p = 0; p < npolarizations; ++p) {
        store_matrix_sync(&rawJreArray[p][0][0], rawJre[p], n, mem_row_major);
        store_matrix_sync(&rawJreNegArray[p][0][0], rawJreNeg[p], n,
                          mem_row_major);
        store_matrix_sync(&rawJimArray[p][0][0], rawJim[p], n, mem_row_major);
      }

      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        for (size_t b1 = 0; b1 < m; ++b1) {
          const float G = Garray[Glinear(f, b + b1)];
          for (size_t p = 0; p < npolarizations; ++p) {
            for (size_t t1 = 0; t1 < n; ++t1) {
              int Jre = max(
                  -7, min(7, int(lrint(G * float(rawJreArray[p][b1][t1] -
                                                 rawJreNegArray[p][b1][t1])))));
              int Jim = max(
                  -7, min(7, int(lrint(G * float(rawJimArray[p][b1][t1])))));
              Jarray[Jlinear(b + b1, f, p, t + t1, 0) / 2] =
                  ucomplex4(Jre, Jim);
            }
          }
        }
      }

    } // for time
  }   // for beam

#if 0
  constexpr size_t J_num_storage_elements = 2;
  static_assert(m % J_num_storage_elements == 0);

  assert(blockDim.x == n / J_num_storage_elements);
  assert(blockDim.y == m);
  assert(blockDim.z == 1);

  const size_t f = blockIdx.x;
  // TODO: use blockIdx.[yz] for times and beams
  // TODO: re-use A over times
  for (size_t b = 0; b < nbeams; b += n) {
    for (size_t t = 0; t < ntimes; t += m) {

      // rawJ[polarization]
      // rawJ[polarization][m][n] = rawJ[polarization][beam][time]
      fragment<wmma::accumulator, m, n, k, int32_t> rawJre[npolarizations],
          rawJreNeg[npolarizations], rawJim[npolarizations];
      for (size_t p = 0; p < npolarizations; ++p) {
        fill_fragment(rawJre[p], 0);
        fill_fragment(rawJreNeg[p], 0);
        fill_fragment(rawJim[p], 0);
      }

      for (size_t d = 0; d < ndishes; d += k) {

#if 1

        // const ucomplex4 A = Aarray[Alinear(f, b, d, 0) / 2];
        __shared__ unsigned char AreArray[k][n / 2], AimArray[k][n / 2];
        assert(Alinear(f, b + 1, d, 0) - Alinear(f, b, d, 0) ==
               ndishes * ncomplex);
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          for (size_t b1 = 0; b1 < n; ++b1) {
            for (size_t d1 = 0; d1 < k; d1 += 2) {
              ucomplex4 A0 = Aarray[Alinear(f, b + b1, d + d1 + 0, 0) / 2];
              ucomplex4 A1 = Aarray[Alinear(f, b + b1, d + d1 + 1, 0) / 2];
              AreArray[d1 / 2][b1] = icomplex4(A1.real(), A0.real()).data;
              AimArray[d1 / 2][b1] = icomplex4(A1.imag(), A0.imag()).data;
            }
          }
        }

        // A[beam][dish]
        // wmma::B[k][n]   (must be row major)
        fragment<wmma::matrix_b, m, n, k, experimental::precision::s4,
                 col_major>
            Are, Aim;
        load_matrix_sync(Are, &AreArray[0][0], n);
        load_matrix_sync(Aim, &AimArray[0][0], n);

#else

        // A[beam][dish]
        // wmma::B[k][n]   (must be row major)
        fragment<wmma::matrix_b, m, n, k, experimental::precision::s4,
                 col_major>
            A[2];
        // const ucomplex4 A = Aarray[Alinear(f, b, d, 0) / 2];
        assert(Alinear(f, b + 1, d, 0) - Alinear(f, b, d, 0) ==
               ndishes * ncomplex);
        load_matrix_sync(A[0], &Aarray[Alinear(f, b, d + 0 * k / 2, 0) / 2],
                         ndishes * ncomplex);
        load_matrix_sync(A[1], &Aarray[Alinear(f, b, d + 1 * k / 2, 0) / 2],
                         ndishes * ncomplex);

        fragment<wmma::matrix_b, m, n, k, experimental::precision::s4,
                 col_major>
            Are, Aim;
        // This loop handles 8 elements at a time
        for (int i = 0; i < A[0].num_storage_elements; ++i) {
          // Remove bias
          int A0 = A[0].x[i] ^ 0x88888888U;
          int A1 = A[1].x[i] ^ 0x88888888U;

#error "TODO: use shared memory to set this up"

          // Shuffle variables to access data from other lanes
          //   L%4=0: A0  = [03, 02, 01, 00]   needs to go to: [1 1 0 0]
          //   L%4=1: A0  = [07, 06, 05, 04]                   [3 3 2 2]
          //   L%4=2: A0  = [0b, 0a, 09, 08]                   [1 1 0 0]
          //   L%4=3: A0  = [0f, 0e, 0d, 0c]                   [3 3 2 2]
          //   L%4=0: A1  = [13, 12, 11, 10]
          //   L%4=1: A1  = [17, 16, 15, 14]
          //   L%4=2: A1  = [1b, 1a, 19, 18]
          //   L%4=3: A1  = [1f, 1e, 1d, 1c]
          int A0o = __shfl_xor_sync(0xffffffffU, A0, 0x01);
          int A1o = __shfl_xor_sync(0xffffffffU, A1, 0x01);
          //   L%4=0: A0o = [07, 06, 05, 04]
          //   L%4=1: A0o = [03, 02, 01, 00]
          //   L%4=2: A0o = [0f, 0e, 0d, 0c]
          //   L%4=3: A0o = [0b, 0a, 09, 08]
          int A0p =
              __byte_perm(A0, A0o, (threadIdx.x & 1) == 0 ? 0x5410 : 0x7632);
          int A1p =
              __byte_perm(A1, A1o, (threadIdx.x & 1) == 0 ? 0x5410 : 0x7632);
          //   L%4=0: A0p = [05, 04, 01, 00]   needs to go to: [2 2 0 0]
          //   L%4=1: A0p = [07, 06, 03, 02]                   [3 3 1 1]
          //   L%4=2: A0p = [0d, 0c, 09, 08]                   [2 2 0 0]
          //   L%4=3: A0p = [0f, 0e, 0b, 0a]                   [3 3 1 1]
          int A0q = __shfl_xor_sync(0xffffffffU, A0, 0x02);
          int A1q = __shfl_xor_sync(0xffffffffU, A1, 0x02);
          //   L%4=0: A0q = [0d, 0c, 09, 08]
          //   L%4=1: A0q = [0f, 0e, 0b, 0a]
          //   L%4=2: A0q = [05, 04, 01, 00]
          //   L%4=3: A0q = [07, 06, 03, 02]
          int A0n =
              __byte_perm(A0p, A0q, (threadIdx.x & 2) == 0 ? 0x5401 : 0x3276);
          int A1n =
              __byte_perm(A1p, A1q, (threadIdx.x & 2) == 0 ? 0x5401 : 0x3276);
          //   L%4=0: A0n = [09, 08, 01, 00]
          //   L%4=1: A0n = [0b, 0a, 03, 02]
          //   L%4=2: A0n = [0d, 0c, 05, 04]
          //   L%4=3: A0n = [0f, 0e, 07, 06]
          //   L%4=0: A1n = [19, 18, 11, 10]
          //   L%4=1: A1n = [1b, 1a, 13, 12]
          //   L%4=2: A1n = [1d, 1c, 15, 14]
          //   L%4=3: A1n = [1f, 1e, 17, 16]

          // Extract real and imaginary parts by interleaving the dishes
          // In this table, the MSB is on the left, LSB is on the right
          //   A0n = [d09re, d09im, d08re, d08im, d01re, d01im, d00re, d00im]
          //   A1n = [d19re, d19im, d18re, d18im, d11re, d11im, d10re, d10im]
          Are.x[i] =
              ((unsigned int)(A0n & 0xf0f0f0f0U) >> 4) | (A1n & 0xf0f0f0f0U);
          Aim.x[i] =
              (A0n & 0x0f0f0f0fU) | ((unsigned int)(A1n & 0x0f0f0f0fU) << 4);
          //   Are = [d19re, d09re, d18re, d08re, d11re, d01re, d10re, d00re]
          //   Aim = [d19im, d09im, d18im, d08im, d11im, d01im, d10im, d00im]
        }

#endif

        // E[time][dish]
        // wmma::A[m][k]   (must be column major)
        fragment<wmma::matrix_a, m, n, k, experimental::precision::s4,
                 row_major>
            E[4];
        // const ucomplex4 E = Earray[Elinear(t, f, d, p, 0) / 2];
        assert(Elinear(t + 1, f, d, 0, 0) - Elinear(t, f, d, 0, 0) ==
               nfrequencies * ndishes * npolarizations * ncomplex);
        load_matrix_sync(E[0], &Earray[Elinear(t, f, d + 0 * k / 4, 0, 0) / 2],
                         nfrequencies * ndishes * npolarizations * ncomplex);
        load_matrix_sync(E[1], &Earray[Elinear(t, f, d + 1 * k / 4, 0, 0) / 2],
                         nfrequencies * ndishes * npolarizations * ncomplex);
        load_matrix_sync(E[2], &Earray[Elinear(t, f, d + 2 * k / 4, 0, 0) / 2],
                         nfrequencies * ndishes * npolarizations * ncomplex);
        load_matrix_sync(E[3], &Earray[Elinear(t, f, d + 3 * k / 4, 0, 0) / 2],
                         nfrequencies * ndishes * npolarizations * ncomplex);

        fragment<wmma::matrix_a, m, n, k, experimental::precision::s4,
                 row_major>
            Ere[npolarizations], Eim[npolarizations];
        // This loop handles 8 elements at a time
        for (int i = 0; i < E[0].num_storage_elements; ++i) {
          // Remove bias
          int E0 = E[0].x[i] ^ 0x88888888U;
          int E1 = E[1].x[i] ^ 0x88888888U;
          int E2 = E[2].x[i] ^ 0x88888888U;
          int E3 = E[3].x[i] ^ 0x88888888U;

          // Extract polarizations
          // We want to end up with the same layout as for A0 and A1 above.
          // In this table, the MSB is on the left, LSB is on the right
          //   E0  = [d01+, d01-, d00+, d00-]
          //   E1  = [d09+, d09-, d08+, d08-]
          //   E2  = [d11+, d11-, d10+, d10-]
          //   E3  = [d19+, d19-, d18+, d18-]
          int Em0 = __byte_perm(E0, E1, 0x6420);
          int Ep0 = __byte_perm(E0, E1, 0x7531);
          int Em1 = __byte_perm(E2, E3, 0x6420);
          int Ep1 = __byte_perm(E2, E3, 0x7531);
          //   Em0 = [d09-, d08-, d08-, d00-]
          //   Ep0 = [d09+, d08+, d08+, d00+]
          //   Em1 = [d19-, d18-, d11-, d10-]
          //   Ep1 = [d19+, d18+, d11+, d10+]

          // Extract real and imaginary parts by interleaving the dishes.
          // This works as for Are, Aim above.
          //   Em0 = [d09-re, d09-im, d08-re, d08-im,
          //          d01-re, d01-im, d00-re, d00-im]
          //   Em1 = [d19-re, d19-im, d18-re, d18-im,
          //          d11-re, d11-im, d10-re, d10-im]
          //   Ep0 = ...
          //   Ep1 = ...
          Ere[0].x[i] =
              ((unsigned int)(Em0 & 0xf0f0f0f0U) >> 4) | (Em1 & 0xf0f0f0f0U);
          Eim[0].x[i] =
              (Em0 & 0x0f0f0f0fU) | ((unsigned int)(Em1 & 0x0f0f0f0fU) << 4);
          Ere[1].x[i] =
              ((unsigned int)(Ep0 & 0xf0f0f0f0U) >> 4) | (Ep1 & 0xf0f0f0f0U);
          Eim[1].x[i] =
              (Ep0 & 0x0f0f0f0fU) | ((unsigned int)(Ep1 & 0x0f0f0f0fU) << 4);
          //   Ere[0] = [d19-re, d18-re, d11-re, d10-re,
          //             d09-re, d08-re, d01-re, d00-re]
          //   Eim[0] = [d19-im, d18-im, d11-im, d10-im,
          //             d09-im, d08-im, d01-im, d00-im]
          //   Ere[1] = ...
          //   Eim[1] = ...
        }

        // Multiply
        for (size_t p = 0; p < npolarizations; ++p) {
          mma_sync(rawJre[p], Ere[p], Are, rawJre[p]);
          mma_sync(rawJreNeg[p], Eim[p], Aim, rawJreNeg[p]);
          mma_sync(rawJim[p], Eim[p], Are, rawJim[p]);
          mma_sync(rawJim[p], Ere[p], Aim, rawJim[p]);
        }

      } // for dish

#if 0

      __shared__ int rawJreArray[npolarizations][m][n],
          rawJreNegArray[npolarizations][m][n],
          rawJimArray[npolarizations][m][n];
      for (size_t p = 0; p < npolarizations; ++p) {
        store_matrix_sync(&rawJreArray[p][0][0], rawJre[p], n, mem_row_major);
        store_matrix_sync(&rawJreNegArray[p][0][0], rawJreNeg[p], n,
                          mem_row_major);
        store_matrix_sync(&rawJimArray[p][0][0], rawJim[p], n, mem_row_major);
      }

      for (int i = 0; i < 2; ++i) {
        const size_t b1 = 2 * threadIdx.x + i;
        const size_t t1 = threadIdx.y;
        const float G = Garray[Glinear(f, b + b1)];
        for (size_t p = 0; p < npolarizations; ++p) {
          int Jre =
              max(-7, min(7, int(lrint(G * float(rawJreArray[p][t1][b1] -
                                                 rawJreNegArray[p][t1][b1])))));
          int Jim =
              max(-7, min(7, int(lrint(G * float(rawJimArray[p][t1][b1])))));
          Jarray[Jlinear(b + b1, f, p, t + t1, 0) / 2] = ucomplex4(Jre, Jim);
        }
      }

#else

      static_assert(rawJre[0].num_storage_elements == J_num_storage_elements,
                    "");
      for (int i = 0; i < rawJre[0].num_storage_elements; ++i) {
        const size_t b1 = J_num_storage_elements * threadIdx.x + i;
        const size_t t1 = threadIdx.y;
        const float G = Garray[Glinear(f, b + b1)];
        for (size_t p = 0; p < npolarizations; ++p) {
          int Jre = max(-7, min(7, int(lrint(G * float(rawJre[p].x[i] -
                                                       rawJreNeg[p].x[i])))));
          int Jim = max(-7, min(7, int(lrint(G * float(rawJim[p].x[i])))));
          Jarray[Jlinear(b + b1, f, p, t + t1, 0) / 2] = ucomplex4(Jre, Jim);
        }
      }

#endif
    }
  }
#endif
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
  cout << "beamforming.cuda\n";

  vector<ucomplex4> Earray;
  vector<ucomplex4> Aarray;
  vector<float> Garray;
  vector<ucomplex4> Jarray;
  setup(Earray, Aarray, Garray, Jarray);

  cout << "Forming beams...\n";
  ucomplex4 *Earray2 = nullptr;
  cudaMalloc(&Earray2, Earray.size() * sizeof(ucomplex4));
  cudaMemcpy(Earray2, Earray.data(), Earray.size() * sizeof(ucomplex4),
             cudaMemcpyHostToDevice);
  ucomplex4 *Aarray2 = nullptr;
  cudaMalloc(&Aarray2, Aarray.size() * sizeof(ucomplex4));
  cudaMemcpy(Aarray2, Aarray.data(), Aarray.size() * sizeof(ucomplex4),
             cudaMemcpyHostToDevice);
  float *Garray2 = nullptr;
  cudaMalloc(&Garray2, Garray.size() * sizeof(float));
  cudaMemcpy(Garray2, Garray.data(), Garray.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  ucomplex4 *Jarray2 = nullptr;
  cudaMalloc(&Jarray2, Jarray.size() * sizeof(ucomplex4));

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const int m = 8;
  const int n = 8;
  const dim3 numBlocks(nfrequencies);
  const dim3 threadsPerBlock(m / 2, n, 1);
  form_beams<<<numBlocks, threadsPerBlock>>>(Jarray2, Earray2, Aarray2,
                                             Garray2);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Earray2);
  Earray2 = nullptr;
  cudaFree(Aarray2);
  Aarray2 = nullptr;
  cudaFree(Garray2);
  Garray2 = nullptr;
  cudaMemcpy(Jarray.data(), Jarray2, Jarray.size() * sizeof(ucomplex4),
             cudaMemcpyDeviceToHost);
  cudaFree(Jarray2);
  Jarray2 = nullptr;

  check(Jarray);

  cout << "Done.\n";
  return 0;
}
