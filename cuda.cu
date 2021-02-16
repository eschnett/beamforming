// -*-c++-*-
// Beamforming with CUDA

#include "adler32.h"

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

const unsigned int ntimes = 32768;    // per chunk
const unsigned int nfrequencies = 32; // per GPU
const unsigned int ndishes = 512;
const unsigned int npolarizations = 2;
const unsigned int nbeams = 128;
const unsigned int ncomplex = 2; // complex number components

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
  // G[frequency][beam]

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

  assert(ndishes * npolarizations * ncomplex % 32 == 0); // for load_matrix_sync

  for (unsigned int f = 0; f < nfrequencies; ++f) {
    for (unsigned int t = 0; t < ntimes; t += n) {
      for (unsigned int b = 0; b < nbeams * npolarizations * ncomplex; b += m) {

        fragment<wmma::accumulator, m, n, k, int32_t> C;
        fill_fragment(C, 0);

        for (unsigned int d = 0; d < ndishes * npolarizations * ncomplex;
             d += k) {
          // must be row major
          fragment<wmma::matrix_a, m, n, k, experimental::precision::s4,
                   row_major>
              A;
          load_matrix_sync(
              A,
              &reinterpret_cast<const unsigned char *>(
                  Aarray)[(d +
                           ndishes * npolarizations * ncomplex *
                               (b + nbeams * npolarizations * ncomplex * f)) /
                          2],
              ndishes * npolarizations * ncomplex);
          // must be column major
          fragment<wmma::matrix_b, m, n, k, experimental::precision::s4,
                   col_major>
              B;
          load_matrix_sync(
              B,
              &reinterpret_cast<const unsigned char *>(
                  Earray)[(d + ndishes * npolarizations * ncomplex *
                                   (f + nfrequencies * t)) /
                          2],
              ndishes * npolarizations * ncomplex * nfrequencies);
          mma_sync(C, A, B, C);
        }

        __shared__ int32_t rawJarray[m][n];
        store_matrix_sync(&rawJarray[0][0], C, n, mem_row_major);

        // Apply gain
        // We use threads for times
        for (unsigned b1 = 0; b1 < m; b1 += 2) {
          int32_t rawJ0 = rawJarray[b1 + 0][threadIdx.x];
          int32_t rawJ1 = rawJarray[b1 + 1][threadIdx.x];
          int32_t Jint0 = min(
              7, max(-7, int(lrintf(rawJ0 * Garray[b + b1 + 0 + nbeams * f]))));
          int32_t Jint1 = min(
              7, max(-7, int(lrintf(rawJ1 * Garray[b + b1 + 1 + nbeams * f]))));
          // Assemble 4-bit complex number
          unsigned char Juchar0 = Jint0 + 8;
          unsigned char Juchar1 = Jint1 + 8;
          unsigned char J = Juchar0 | (Juchar1 << 4);
          reinterpret_cast<unsigned char *>(
              Jarray)[(b + b1 +
                       nbeams * npolarizations * ncomplex *
                           (f + nfrequencies * (t + threadIdx.x))) /
                      2] = J;
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
  cout << "Setting up input data...\n";
  vector<unsigned char> Earray(ntimes * nfrequencies * ndishes *
                               npolarizations);
  vector<unsigned char> Aarray(nfrequencies * nbeams * ndishes);
  vector<float> Garray(nfrequencies * nbeams);
  for (size_t n = 0; n < Earray.size(); ++n)
    Earray[n] = ((n % 15) << 4) | ((n + 1) % 15);
  for (size_t n = 0; n < Aarray.size(); ++n)
    Aarray[n] = ((n % 15) << 4) | ((n + 1) % 15);
  for (size_t n = 0; n < Garray.size(); ++n)
    Garray[n] = (n / ndishes) * (15 + n % 15) / 30;
  vector<unsigned char> Jarray(nbeams * nfrequencies * npolarizations * ntimes);
  cout << "Forming beams...\n";
  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  const dim3 numBlocks(1);
  const dim3 threadsPerBlock(32);
  form_beams<<<numBlocks, threadsPerBlock>>>(Jarray.data(), Earray.data(),
                                             Aarray.data(), Garray.data());
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  cout << "Calculating checksum...\n";
  uint32_t checksum =
      adler32(reinterpret_cast<unsigned char *>(Jarray.data()), Jarray.size());
  cout << "Checksum: 0x" << hex << setfill('0') << setw(8) << checksum << "\n";
  assert(checksum == 0x59b6a388);
  cout << "Done.\n";
  return 0;
}
