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
// Accessors handling memory layout

// E[time][frequency][dish][polarization][complex]
// A[frequency][beam][polarization][complex][dish][polarization][complex]
// J[time][frequency][beam][polarization][complex]
// G[frequency][beam][polarization][complex]

constexpr size_t Esize1 = Esize;
constexpr device_host size_t Elinear1(size_t t, size_t f, size_t d, size_t p,
                                      size_t c) {
  return Elinear(t, f, d, p, c);
}

constexpr size_t Jsize1 =
    ntimes * nfrequencies * nbeams * npolarizations * ncomplex;
constexpr device_host size_t Jlinear1(size_t t, size_t f, size_t b, size_t p,
                                      size_t c) {
  assert(t < ntimes);
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(p < npolarizations);
  assert(c < ncomplex);
  const auto ind =
      c +
      ncomplex * (p + npolarizations * (b + nbeams * (f + nfrequencies * t)));
  assert(ind < Jsize1);
  return ind;
}

constexpr size_t Asize1 = nfrequencies * nbeams * npolarizations * ncomplex *
                          ndishes * npolarizations * ncomplex;
constexpr device_host size_t Alinear1(size_t f, size_t b, size_t p1, size_t c1,
                                      size_t d, size_t p2, size_t c2) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(p1 < npolarizations);
  assert(c1 < ncomplex);
  assert(d < ndishes);
  assert(p2 < npolarizations);
  assert(c2 < ncomplex);
  const auto ind =
      c2 +
      ncomplex *
          (p2 +
           npolarizations *
               (d + ndishes * (c1 + ncomplex * (p1 + npolarizations *
                                                         (b + nbeams * f)))));
  assert(ind < Asize1);
  return ind;
}

constexpr size_t Gsize1 = nfrequencies * nbeams * npolarizations * ncomplex;
constexpr device_host size_t Glinear1(size_t f, size_t b, size_t p, size_t c) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(p < npolarizations);
  assert(c < ncomplex);
  const auto ind = c + ncomplex * (p + npolarizations * (b + nbeams * f));
  assert(ind < Gsize1);
  return ind;
}

static_assert(Jsize1 <= UINT_MAX);
static_assert(Esize1 <= UINT_MAX);
static_assert(Asize1 <= UINT_MAX);
static_assert(Gsize1 <= UINT_MAX);

// Reshape arrays

vector<icomplex4> prepare_A(const vector<icomplex4> &Aarray) {
  // for (size_t f = 0; f < nfrequencies; ++f) {
  //   for (size_t b = 0; b < nbeams; ++b) {
  //     for (size_t d = 0; d < ndishes; ++d) {
  //       for (size_t c = 0; c < ncomplex; ++c) {
  //         cout << "A["
  //              // << ",f=" << f << ",b=" << b << ",d=" << d
  //              << ",c=" << c
  //              << "]=" << int(Aarray.at(Alinear(f, b, d, c) / 2)[c]) << "\n";
  //       }
  //     }
  //   }
  // }

  vector<icomplex4> Aarray1(Asize1 / 2);
  for (size_t f = 0; f < nfrequencies; ++f) {
    for (size_t b = 0; b < nbeams; ++b) {
      for (size_t p1 = 0; p1 < npolarizations; ++p1) {
        for (size_t c1 = 0; c1 < ncomplex; ++c1) {
          for (size_t d = 0; d < ndishes; ++d) {
            for (size_t p2 = 0; p2 < npolarizations; ++p2) {
              const icomplex4 Ac = Aarray[Alinear(f, b, d, 0) / 2];
              // [1]=real, [0]=imag
              const signed char A[2] = {Ac.imag(), Ac.real()};
              signed char A1[2];
              if (p1 == p2) {
                // Want:
                //   J.re = A.re * E.re - A.im * E.im
                //   J.im = A.im * E.re + A.re * E.im
                // Old layout:
                //   J[p][0] = A[0] * E[p][1] + A[1] * E[p][0]
                //   J[p][1] = A[1] * E[p][1] - A[0] * E[p][0]
                // New layout:
                //   J[p][0] = A1[p][q][0] * E[p][1] + A1[p][q][1] * E[p][0]
                //   J[p][1] = A1[p][q][1] * E[p][1] - A1[p][q][0] * E[p][0]
                // Coefficients:
                //   A1[p][q][0][0] = delta[p][q] A[1]
                //   A1[p][q][0][1] = delta[p][q] A[0]
                //   A1[p][q][1][0] = delta[p][q] (-A[0])
                //   A1[p][q][1][1] = delta[p][q] A[1]
                //
                // Setting A1[c1][c2]
                if (c1 == 1) { // real part
                  A1[1] = A[1];
                  A1[0] = -A[0];
                } else { // imaginary part
                  A1[1] = A[0];
                  A1[0] = A[1];
                }
              } else {
                A1[1] = 0;
                A1[0] = 0;
              }
              const icomplex4 A1c(A1[1], A1[0]);
              Aarray1[Alinear1(f, b, p1, c1, d, p2, 0) / 2] = A1c.debias();
            }
          }
        }
      }
    }
  }

  // for (size_t f = 0; f < nfrequencies; ++f) {
  //   for (size_t b = 0; b < nbeams; ++b) {
  //     for (size_t p1 = 0; p1 < npolarizations; ++p1) {
  //       for (size_t c1 = 0; c1 < ncomplex; ++c1) {
  //         for (size_t d = 0; d < ndishes; ++d) {
  //           for (size_t p2 = 0; p2 < npolarizations; ++p2) {
  //             for (size_t c2 = 0; c2 < ncomplex; ++c2) {
  //               cout << "A1["
  //                    // << ",f=" << f << ",b=" << b
  //                    << ",p1=" << p1 << ",c1="
  //                    << c1
  //                    // << ",d=" << d
  //                    << ",p2=" << p2 << ",c2=" << c2 << "]="
  //                    << int(Aarray1.at(Alinear1(f, b, p1, c1, d, p2, c2) /
  //                                      2)[c2])
  //                    << "\n";
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  return Aarray1;
}

vector<float> prepare_G(const vector<float> &Garray) {
  vector<float> Garray1(Gsize1);
  for (size_t f = 0; f < nfrequencies; ++f) {
    for (size_t b = 0; b < nbeams; ++b) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t c = 0; c < ncomplex; ++c) {
          Garray1[Glinear1(f, b, p, c)] = Garray[Glinear(f, b)];
        }
      }
    }
  }
  return Garray1;
}

vector<icomplex4> prepare_J(const vector<icomplex4> &Jarray) {
  vector<icomplex4> Jarray1(Jsize1);
  return Jarray1;
}

void restore_J(vector<icomplex4> &Jarray, const vector<icomplex4> &Jarray1) {
  // for (size_t t = 0; t < ntimes; ++t) {
  //   for (size_t f = 0; f < nfrequencies; ++f) {
  //     for (size_t b = 0; b < nbeams; ++b) {
  //       for (size_t p = 0; p < npolarizations; ++p) {
  //         for (size_t c = 0; c < ncomplex; ++c) {
  //           cout << "J1["
  //                << ",t=" << t << ",f=" << f << ",b=" << b << ",p=" << p
  //                << ",c=" << c
  //                << "]=" << int(Jarray1.at(Jlinear1(t, f, b, p, c) / 2)[c])
  //                << "\n";
  //         }
  //       }
  //     }
  //   }
  // }

  for (size_t b = 0; b < nbeams; ++b) {
    for (size_t f = 0; f < nfrequencies; ++f) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t t = 0; t < ntimes; ++t) {
          Jarray[Jlinear(b, f, p, t, 0) / 2] =
              Jarray1[Jlinear1(t, f, b, p, 0) / 2];
        }
      }
    }
  }
}

__global__ void form_beams(unsigned char *restrict const Jarray,
                           const unsigned char *restrict const Earray,
                           const unsigned char *restrict const Aarray,
                           const float *restrict Garray) {
  // This is the array layout. Having the polarization inside requires
  // making A four times as large, and doubles the number of floating
  // point operations. Avoiding this requires changing the memory
  // layout of E.
  //
  // We double A once more to implement complex multiplication. Since
  // A is rather small, and is read only once for each beam forming,
  // the latter seems fine.
  //
  // J = A * E
  // J.re = A.re * E.re - A.im * E.im
  // J.im = A.im * E.re + A.re * E.im

  // operation:    D = A * B + C
  // matrix sizes: C, D: m * n
  //               A:    m * k
  //               B:    k * n
  //               D[m,n] = C[m,n] + A[m,k] B[k,n]
  // consequences: m ranges over beams
  //               k ranges over dishes
  //               n ranges over times
  // Nothing ranges over frequencies since all of E, A, and J depend
  // on frequencies.

  // These sizes are dictated by CUDA
  constexpr int m = 8;  // beams
  constexpr int n = 8;  // times
  constexpr int k = 32; // dishes
  assert(nbeams * npolarizations * ncomplex % m == 0);
  assert(ndishes * npolarizations * ncomplex % k == 0);
  assert(ntimes % n == 0);

  assert(blockDim.x == n);
  assert(blockDim.y == m / 2);
  assert(blockDim.z == 1);

  assert(ndishes * npolarizations * ncomplex % 32 == 0); // for load_matrix_sync

  for (unsigned int f = 0; f < nfrequencies; ++f) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
      printf("f=%u\n", f);
    for (unsigned int t = 0; t < ntimes; t += n) {
      for (unsigned int b = 0; b < nbeams * npolarizations * ncomplex; b += m) {

        fragment<wmma::accumulator, m, n, k, int32_t> C;
        fill_fragment(C, 0);

        for (unsigned int d = 0; d < ndishes * npolarizations * ncomplex;
             d += k) {

          // A must be row major
          fragment<wmma::matrix_a, m, n, k, experimental::precision::s4,
                   row_major>
              A;
          const unsigned int Aindex =
              Alinear1(f, b / 4, b / 2 % 2, b % 2, d / 4, d / 2 % 2, d % 2);
          const unsigned int dlast = d + k - 1;
          const unsigned int Aindexlast = Alinear1(
              f, b / 4, b / 2 % 2, b % 2, dlast / 4, dlast / 2 % 2, dlast % 2);
          assert(Aindex < Asize1);
          assert(Aindexlast < Asize1);
          const unsigned char *const Aptr = &Aarray[Aindex / 2];
          // assert(intptr_t(Aptr) % (128 / 8) == 0);
          const unsigned int bnext = b + 1;
          const unsigned int Aindexnext = Alinear1(
              f, bnext / 4, bnext / 2 % 2, bnext % 2, d / 4, d / 2 % 2, d % 2);
          assert(Aindexnext - Aindex == ndishes * npolarizations * ncomplex);
          load_matrix_sync(A, Aptr, ndishes * npolarizations * ncomplex);

          // B must be column major
          fragment<wmma::matrix_b, m, n, k, experimental::precision::s4,
                   col_major>
              B;
          const unsigned int Eindex = Elinear1(t, f, d / 4, d / 2 % 2, d % 2);
          const unsigned char *const Eptr = &Earray[Eindex / 2];
          const unsigned int Eindexlast =
              Elinear1(t, f, dlast / 4, dlast / 2 % 2, dlast % 2);
          assert(Eindex < Esize1);
          assert(Eindexlast < Esize1);
          // assert(intptr_t(Eptr) % (128 / 8) == 0);
          const unsigned int tnext = t + 1;
          const unsigned int Eindexnext =
              Elinear1(tnext, f, d / 4, d / 2 % 2, d % 2);
          assert(Eindexnext - Eindex ==
                 ndishes * npolarizations * ncomplex * nfrequencies);
          load_matrix_sync(B, Eptr,
                           ndishes * npolarizations * ncomplex * nfrequencies);
          for (int i = 0; i < B.num_elements; ++i)
            B.x[i] -= 8;

          // Multiply
          mma_sync(C, A, B, C);
        }

        __shared__ int rawJarray[m][n];
        // assert(intptr_t(&rawJarray[0][0]) % (256 / 8) == 0);
        store_matrix_sync(&rawJarray[0][0], C, n, mem_row_major);

        // Apply gain
        // We use threads for times
        // TODO: Make use of all threads
        assert(blockDim.x == n);
        assert(2 * blockDim.y == m);
        const unsigned int t1 = threadIdx.x;
        const unsigned int b1 = 2 * threadIdx.y;

        int rawJ0 = rawJarray[b1 + 0][t1];
        int rawJ1 = rawJarray[b1 + 1][t1];
        // Remove offset from 4-bit complex representation in E
        // rawJ0 -= 8 * ndishes * npolarizations * ncomplex;
        // rawJ1 -= 8 * ndishes * npolarizations * ncomplex;
        const unsigned int Gindex =
            Glinear1(f, (b + b1) / 4, (b + b1) / 2 % 2, (b + b1) % 2);
        assert(Gindex + 1 < Gsize1);
        signed char Jint0 =
            min(7, max(-7, int(lrintf(rawJ0 * Garray[Gindex + 0]))));
        signed char Jint1 =
            min(7, max(-7, int(lrintf(rawJ1 * Garray[Gindex + 1]))));
        // Assemble 4-bit complex number
        unsigned char Juchar0 = Jint0 + 8;
        unsigned char Juchar1 = Jint1 + 8;
        unsigned char J = Juchar0 | (Juchar1 << 4);

        const unsigned int Jindex =
            Jlinear1(t + t1, f, (b + b1) / 4, (b + b1) / 2 % 2, (b + b1) % 2);
        assert(Jindex < Jsize1);
        unsigned char *const Jptr = &Jarray[Jindex / 2];
        *Jptr = J;
      }
    }
  }
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

  vector<icomplex4> Earray;
  vector<icomplex4> Aarray;
  vector<float> Garray;
  vector<icomplex4> Jarray;
  setup(Earray, Aarray, Garray, Jarray);

  // Modify layouts
  // We don't modify the layout of E
  const vector<icomplex4> Earray1 = Earray;
  vector<icomplex4> Aarray1 = prepare_A(Aarray);
  vector<float> Garray1 = prepare_G(Garray);
  vector<icomplex4> Jarray1 = prepare_J(Jarray);

  cout << "Forming beams...\n";
  unsigned char *Earray2 = nullptr;
  cudaMalloc(&Earray2, Earray1.size() * sizeof(unsigned char));
  cudaMemcpy(Earray2, Earray1.data(), Earray1.size() * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  unsigned char *Aarray2 = nullptr;
  cudaMalloc(&Aarray2, Aarray1.size() * sizeof(unsigned char));
  cudaMemcpy(Aarray2, Aarray1.data(), Aarray1.size() * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  float *Garray2 = nullptr;
  cudaMalloc(&Garray2, Garray1.size() * sizeof(float));
  cudaMemcpy(Garray2, Garray1.data(), Garray1.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  unsigned char *Jarray2 = nullptr;
  cudaMalloc(&Jarray2, Jarray1.size() * sizeof(unsigned char));

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const dim3 numBlocks(1);
  const dim3 threadsPerBlock(8, 4);
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
  cudaMemcpy(Jarray1.data(), Jarray2, Jarray1.size() * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
  cudaFree(Jarray2);
  Jarray2 = nullptr;

  // Undo layout modification
  restore_J(Jarray, Jarray1);

  check(Jarray);

  cout << "Done.\n";
  return 0;
}
