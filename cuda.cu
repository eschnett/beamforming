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

const unsigned int ncomplex = 2; // complex number components

// Accessors handling memory layout
constexpr size_t Eindex1(size_t t, size_t f, size_t d, size_t p) {
  return Eindex(t, f, d, p);
}
constexpr size_t Jindex1(size_t t, size_t f, size_t b, size_t p) {
  assert(t < ntimes);
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(p < npolarizations);
  return p + npolarizations * (b + nbeams * (f + nfrequencies * t));
}
constexpr size_t Aindex1(size_t f, size_t b, size_t p1, size_t c1, size_t d,
                         size_t p2) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(p1 < npolarizations);
  assert(c1 < ncomplex);
  assert(d < ndishes);
  assert(p2 < npolarizations);
  return p2 + npolarizations *
                  (d + ndishes * (c1 + ncomplex * (p1 + npolarizations *
                                                            (b + nbeams * f))));
}
constexpr size_t Gindex1(size_t f, size_t b, size_t p, size_t c) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(p < npolarizations);
  assert(c < ncomplex);
  return c + ncomplex * (p + npolarizations * (b + nbeams * f));
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

  // E[time][frequency][dish][polarization][complex]
  // A[frequency][beam][polarization][complex][dish][polarization][complex]
  // J[time][frequency][beam][polarization][complex]
  // G[frequency][beam][polarization][complex]
  const size_t Jsize =
      size_t(1) * ntimes * nfrequencies * nbeams * npolarizations * ncomplex;
  const size_t Esize =
      size_t(1) * ntimes * nfrequencies * ndishes * npolarizations * ncomplex;
  const size_t Asize = size_t(1) * nfrequencies * nbeams * npolarizations *
                       ncomplex * ndishes * npolarizations * ncomplex;
  const size_t Gsize =
      size_t(1) * nfrequencies * nbeams * npolarizations * ncomplex;
  assert(Jsize <= UINT_MAX);
  assert(Esize <= UINT_MAX);
  assert(Asize <= UINT_MAX);
  assert(Gsize <= UINT_MAX);

  // operation:    D = A * B + C
  // matrix sizes: C, D: m * n
  //               A:    m * k
  //               B:    k * n
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
              d + ndishes * npolarizations * ncomplex *
                      (b + nbeams * npolarizations * ncomplex * f);
          const unsigned int Aindexlast =
              d + k - 1 +
              ndishes * npolarizations * ncomplex *
                  (b + (m - 1) + nbeams * npolarizations * ncomplex * f);
          assert(Aindex < Asize);
          assert(Aindexlast < Asize);
          const unsigned char *const Aptr = &Aarray[Aindex / 2];
          // assert(intptr_t(Aptr) % (128 / 8) == 0);
          load_matrix_sync(A, Aptr, ndishes * npolarizations * ncomplex);

          // B must be column major
          fragment<wmma::matrix_b, m, n, k, experimental::precision::s4,
                   col_major>
              B;
          const unsigned int Eindex =
              d + ndishes * npolarizations * ncomplex * (f + nfrequencies * t);
          const unsigned int Eindexlast = d + k - 1 +
                                          ndishes * npolarizations * ncomplex *
                                              (f + nfrequencies * (t + n - 1));
          assert(Eindex < Esize);
          assert(Eindexlast < Esize);
          const unsigned char *const Eptr = &Earray[Eindex / 2];
          // assert(intptr_t(Eptr) % (128 / 8) == 0);
          load_matrix_sync(B, Eptr,
                           ndishes * npolarizations * ncomplex * nfrequencies);

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
        assert(blockDim.y == m / 2);
        const unsigned int t = threadIdx.x;
        const unsigned int b1 = 2 * threadIdx.y;

        int rawJ0 = rawJarray[b1 + 0][threadIdx.x];
        int rawJ1 = rawJarray[b1 + 1][threadIdx.x];
        // Remove offset from 4-bit complex representation in E
        rawJ0 -= 8 * ndishes * npolarizations * ncomplex;
        rawJ1 -= 8 * ndishes * npolarizations * ncomplex;
        const unsigned int Gindex =
            b + b1 + nbeams * npolarizations * ncomplex * f;
        assert(Gindex + 1 < Gsize);
        int Jint0 = min(7, max(-7, int(lrintf(rawJ0 * Garray[Gindex + 0]))));
        int Jint1 = min(7, max(-7, int(lrintf(rawJ1 * Garray[Gindex + 1]))));
        // Assemble 4-bit complex number
        unsigned char Juchar0 = Jint0 + 8;
        unsigned char Juchar1 = Jint1 + 8;
        unsigned char J = Juchar0 | (Juchar1 << 4);

        const unsigned int Jindex = b + b1 +
                                    nbeams * npolarizations * ncomplex *
                                        (f + nfrequencies * (t + threadIdx.x));
        assert(Jindex < Jsize);
        unsigned char *const Jptr = &Jarray[Jindex / 2];
        *Jptr = J;
      }
    }
  }
}

vector<icomplex4> prepare_A(const vector<icomplex4> &Aarray) {
  vector<icomplex4> Aarray1(nfrequencies * nbeams * npolarizations * ncomplex *
                            ndishes * npolarizations);
  for (size_t f = 0; f < nfrequencies; ++f) {
    for (size_t b = 0; b < nbeams; ++b) {
      for (size_t p1 = 0; p1 < npolarizations; ++p1) {
        for (size_t c1 = 0; c1 < ncomplex; ++c1) {
          for (size_t d = 0; d < ndishes; ++d) {
            for (size_t p2 = 0; p2 < npolarizations; ++p2) {
              const icomplex4 Ac = Aarray[Aindex(f, b, d)];
              const signed char A[2] = {Ac.real(), Ac.imag()};
              signed char A1[2];
              if (p1 == p2) {
                // J.re = A.re * E.re - A.im * E.im
                // J.im = A.im * E.re + A.re * E.im
                if (c1 == 0) {
                  A1[0] = A[0];
                  A1[1] = -A[1];
                } else {
                  A1[0] = A[1];
                  A1[1] = A[0];
                }
              } else {
                A1[0] = 0;
                A1[1] = 0;
              }
              const icomplex4 A1c(A1[0], A1[1]);
              Aarray1[Aindex1(f, b, p1, c1, d, p2)] = A1c;
            }
          }
        }
      }
    }
  }
  return Aarray1;
}

vector<float> prepare_G(const vector<float> &Garray) {
  vector<float> Garray1(nfrequencies * nbeams * npolarizations * ncomplex);
  for (size_t f = 0; f < nfrequencies; ++f) {
    for (size_t b = 0; b < nbeams; ++b) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t c = 0; c < ncomplex; ++c) {
          Garray1[Gindex1(f, b, p, c)] = Garray[Gindex(f, b)];
        }
      }
    }
  }
  return Garray1;
}

vector<icomplex4> prepare_J(const vector<icomplex4> &Jarray) {
  vector<icomplex4> Jarray1(ntimes * nfrequencies * nbeams * npolarizations);
  return Jarray1;
}

void restore_J(vector<icomplex4> &Jarray, const vector<icomplex4> &Jarray1) {
  for (size_t b = 0; b < nbeams; ++b) {
    for (size_t f = 0; f < nfrequencies; ++f) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t t = 0; t < ntimes; ++t) {
          Jarray[Jindex(b, f, p, t)] = Jarray1[Jindex1(t, f, b, p)];
        }
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
