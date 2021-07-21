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

////////////////////////////////////////////////////////////////////////////////

constexpr size_t nreals_int = 8; // number of real numbers per 32-bit int

constexpr size_t nlanes = 32; // warp size
constexpr size_t nwarps = 32;

// Memory layout of intput/output arrays:

// E[time][frequency][dish][polarization][complex]
// A[frequency][beam][dish][complex]
// J[beam][frequency][polarization][time][complex]
// G[frequency][beam][complex]

// Matrix layout for tensor cores:

// These sizes are dictated by CUDA
constexpr int m = 8;  // beams
constexpr int n = 8;  // times
constexpr int k = 32; // dishes
static_assert(nbeams % m == 0);
static_assert(ndishes % k == 0);
static_assert(ntimes % n == 0);

constexpr size_t nbeams_per_matrix = m;
constexpr size_t ntimes_per_matrix = n;
constexpr size_t ndishes_per_matrix = k;

// Warp layout of A:

constexpr size_t nwarps_for_dishes = 8; // Number of warps over which dishes are distributed
constexpr size_t nwarps_for_beams = 4;  // Number of warps over which beams are distributed
static_assert(nwarps_for_dishes * nwarps_for_beams == nwarps);

static_assert(ndishes % nwarps_for_dishes == 0);
static_assert(nbeams % nwarps_for_beams == 0);

static_assert(ndishes % (ndishes_per_matrix * nwarps_for_dishes) == 0);
static_assert(nbeams % (nbeams_per_matrix * nwarps_for_beams) == 0);

constexpr size_t ndish_matrices_per_thread = ndishes / (ndishes_per_matrix * nwarps_for_dishes);
constexpr size_t nbeam_matrices_per_thread = nbeams / (nbeams_per_matrix * nwarps_for_beams);
// For ndishes = 512, ndish_matrices_per_thread = 2
// For nbeams = 96, nbeam_matrices_per_thread = 3

// Shuffling E:

constexpr size_t ntimes_per_iteration_E = 16;
static_assert(ntimes % ntimes_per_iteration_E == 0);
static_assert(ntimes_per_iteration_E % ntimes_per_matrix == 0);

constexpr size_t padding_E = 4; // padding to avoid bank conflicts

// Calculate Ju:

constexpr size_t ntimes_per_iteration_Ju = 4;

// Storing J:
constexpr size_t ntimes_per_iteration_J = 64;
constexpr size_t padding_J = 4; // padding to avoid bank conflicts

////////////////////////////////////////////////////////////////////////////////

__device__ int32_t clamp(int32_t i, int32_t i0, int32_t i1) { return max(i0, min(i1, i)); }

__device__ int32_t extract_real(const int32_t x0, const int32_t x1) {
  return ((uint32_t)(x0 & 0xf0f0f0f0U) >> 4) | (x1 & 0xf0f0f0f0U);
}

__device__ int32_t extract_imag(const int32_t x0, const int32_t x1) {
  return (x0 & 0x0f0f0f0fU) | ((uint32_t)(x1 & 0x0f0f0f0fU) << 4);
}

////////////////////////////////////////////////////////////////////////////////

__device__ void load_A(fragment<wmma::matrix_a, m, n, k, experimental::precision::s4, row_major> (
                               &restrict Aregister)[ncomplex][nbeam_matrices_per_thread][ndish_matrices_per_thread],
                       const ucomplex4 *restrict const Aarray, const size_t f) {
  assert(blockDim.x == nlanes); // one warp
  assert(blockDim.y == nwarps);
  assert(blockDim.z == 1); // just checking
  const size_t dish_warpidx = blockDim.y % nwarps_for_dishes;
  const size_t beam_warpidx = blockDim.y / nwarps_for_dishes;

  // These must be true if basic arithmetic holds
  assert(nwarps_for_dishes * ndish_matrices_per_thread * ndishes_per_matrix == ndishes);
  assert(nwarps_for_beams * nbeam_matrices_per_thread * nbeames_per_matrix == nbeames);
  const size_t dish0 = dish_warpidx * ndish_matrices_per_thread * ndishes_per_matrix;
  const size_t beam0 = beam_warpidx * nbeam_matrices_per_thread * nbeams_per_matrix;

  for (size_t bm = 0; bm < nbeam_matrices_per_thread; ++bm) {
    const size_t b = beam0 + bm * nbeams_per_matrix;
    for (size_t dm = 0; dm < ndish_matrices_per_thread; ++dm) {
      const size_t d = dish0 + dm * ndishes_per_matrix;

      // Note: This is the wrong ordering for A; need to shuffle entries the
      // same way as for E

      fragment<wmma::matrix_a, m, n, k, experimental::precision::s4, row_major> A0[ncomplex];
      for (size_t c = 0; c < ncomplex; ++c) {
        load_matrix_sync(A0[c], &Aarray[Alinear(f, b, d + c * k / 2, 0) / 2], ndishes * ncomplex);
      }

      static_assert(Aregister[0][bm][dm].num_storage_elements == 1);
      for (int i = 0; i < Aregister[0][bm][dm].num_storage_elements; ++i) {
        // Extract complex components and remove bias
        Aregister[0][bm][dm].x[i] = extract_real(A0[0].x[i], A0[1].x[i]) ^ 0x88888888U;
        Aregister[1][bm][dm].x[i] = extract_imag(A0[0].x[i], A0[1].x[i]) ^ 0x88888888U;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

__device__ void
shuffle_E(int32_t (&restrict Eshared)[ncomplex][ntimes_per_iteration_E][npolarizations][ndishes / nreals_int + padding_E],
          const ucomplex4 *restrict const Earray, const size_t f, const size_t time0) {
  assert(time0 % ntimes_per_iteration_E == 0);

  constexpr size_t nwarps_for_dishes_E = 2;
  constexpr size_t nwarps_for_times_E = 16;
  static_assert(nwarps_for_dishes_E * nwarps_for_times_E == nwarps);

  constexpr size_t ndishes_per_warp_E = ndishes / nwarps_for_dishes_E;              // 256
  constexpr size_t ntimes_per_warp_E = ntimes_per_iteration_E / nwarps_for_times_E; // 1
  static_assert(ntimes_per_warp_E == 1);

  const size_t dish_warpidx_E = threadIdx.y % nwarps_for_dishes_E; // [0..1]
  const size_t time_warpidx_E = threadIdx.y / nwarps_for_dishes_E; // [0..15]

  const size_t dish_laneidx_E = threadIdx.x; // [0..31]

  const size_t t = time0 + time_warpidx_E;
  const size_t dish0 = dish_warpidx_E * ndishes_per_warp_E + dish_laneidx_E * nreals_int;

  // Load E-field from global memory
  // Note: These are not yet split into polarizations complex components
  int32_t E0[npolarizations][ncomplex];
  for (size_t p = 0; p < npolarizations; ++p) {
    for (size_t c = 0; c < ncomplex; ++c) {
      const size_t d = dish0 + (ncomplex * p + c) * nlanes * nreals_int;
      E0[p][c] = *(const int32_t *)&Earray[Elinear(t, f, d, 0, 0) / 2];
    }
  }

  // First we split out the complex components and remove the bias
  int32_t E1[npolarizations][ncomplex];
  for (size_t p = 0; p < npolarizations; ++p) {
    E1[p][0] = extract_real(E0[p][0], E0[p][1]) ^ 0x88888888U;
    E1[p][1] = extract_imag(E0[p][0], E0[p][1]) ^ 0x88888888U;
  }

  // Next we separate the polarizations
  int32_t E2[npolarizations][ncomplex];
  for (size_t c = 0; c < ncomplex; ++c) {
    E2[0][c] = __byte_perm(E1[0][c], E1[1][c], 0x6420);
    E2[1][c] = __byte_perm(E1[0][c], E1[1][c], 0x7531);
  }

  // Store into shared memory
  for (size_t c = 0; c < ncomplex; ++c) {
    for (size_t p = 0; p < npolarizations; ++p) {
      Eshared[c][time_warpidx_E][p][nlanes * dish_warpidx_E + dish_laneidx_E] = E2[p][c];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

__device__ void
compute_Ju(int8_t (&restrict Jushared)[nwarps_for_dishes][nbeams][ntimes_per_iteration_Ju][npolarizations][ncomplex],
           const fragment<wmma::matrix_a, m, n, k, experimental::precision::s4, row_major> (
                   &restrict Aregister)[ncomplex][nbeam_matrices_per_thread][ndish_matrices_per_thread],
           const int32_t (&restrict Eshared)[ncomplex][ntimes_per_iteration_E][npolarizations][ndishes / nreals_int + padding_E],
           const float *restrict const Garray, const size_t f, const size_t time0) {
  const size_t dish_warpidx = threadIdx.y % nwarps_for_dishes;
  const size_t beam_warpidx = threadIdx.y / nwarps_for_dishes;
  assert(beam_warpidx < nwarps_for_beams);

  // These depend on the wmma C matrix layout
  const size_t beam_threadidx = threadIdx.x % (nlanes / ntimes_per_matrix);
  const size_t time_threadidx = threadIdx.x / (nlanes / ntimes_per_matrix);
  assert(time_threadidx < ntimes_per_iteration_Ju);

  const size_t dish0 = dish_warpidx * ndishes_per_matrix;
  const size_t beam0 = beam_warpidx * nbeams_per_matrix;

  const size_t t = time0 + time_threadidx;
  const size_t t1 = t % ntimes_per_iteration_E;
  assert(t1 % ntimes_per_iteration_Ju == 0);

  // Load E-field from shared memory
  // wmma::B[k][n]   (must be row major)
  fragment<wmma::matrix_b, m, n, k, experimental::precision::s4, col_major> E[ncomplex][ndish_matrices_per_thread];

  for (size_t c = 0; c < ncomplex; ++c) {
    for (int dm = 0; dm < ndish_matrices_per_thread; ++dm) {
      const size_t d = dish0 + nwarps_for_dishes * ndishes_per_matrix * dm;
      load_matrix_sync(E[c][dm], &Eshared[c][t1][0][d], (&Eshared[c][t1][1][d] - &Eshared[c][t1][0][d]) * nreals_int);
    }
  }

  for (size_t bm = 0; bm < nbeam_matrices_per_thread; ++bm) {
    fragment<wmma::accumulator, m, n, k, int32_t> JurePos, JureNeg, JuimPos;

    // Initialize Ju
    fill_fragment(JurePos, 0);
    fill_fragment(JureNeg, 0);
    fill_fragment(JuimPos, 0);

    // Multiply
    for (int dm = 0; dm < ndish_matrices_per_thread; ++dm) {
      mma_sync(JurePos, Aregister[0][bm][dm], E[0][dm], JurePos);
      mma_sync(JureNeg, Aregister[1][bm][dm], E[1][dm], JureNeg);
      mma_sync(JuimPos, Aregister[0][bm][dm], E[1][dm], JuimPos);
      mma_sync(JuimPos, Aregister[1][bm][dm], E[0][dm], JuimPos);
    }

    // Extract result from Ju matrix
    static_assert(JurePos.num_storage_elements == npolarizations);
    for (size_t i = 0; i < JurePos.num_storage_elements; ++i) {
      const size_t p = i;
      const size_t b = beam0 + nbeam_matrices_per_thread * bm + nbeam_matrices_per_thread * beam_threadidx;
      // Combine positive and negative J values, and reduce from 32 to 16 bits
      int32_t Ju[ncomplex];
      Ju[0] = JurePos.x[i] - JureNeg.x[i];
      Ju[1] = JuimPos.x[i];
      for (size_t c = 0; c < ncomplex; ++c) {
        const float G = Garray[Glinear(f, b)];
        const int8_t Ju8 = clamp(int32_t(lrintf(G * float(Ju[c]))), -127, 127);
        // TODO: Combine writes
        Jushared[dish_warpidx][bm][t1][p][c] = Ju8;
      }
    }
  } // bm
}

__device__ void
reduce_to_J(ucomplex4 (&restrict Jshared)[nbeams][ntimes_per_iteration_J + padding_J][npolarizations],
            const int8_t (&restrict Jushared)[nwarps_for_dishes][nbeams][ntimes_per_iteration_Ju][npolarizations][ncomplex],
            const size_t time0) {
  constexpr size_t beams_per_warp_Ju = 8;
  constexpr size_t times_per_warp_Ju = 4;
  static_assert(beams_per_warp_Ju * times_per_warp_Ju == nlanes);

  assert(nbeams % beams_per_warp_Ju == 0);
  const size_t beam_warpidx = threadIdx.y / beams_per_warp_Ju;
  const size_t beam0 = beam_warpidx * beams_per_warp_Ju;
  // We don't need all warps
  if (beam0 < nbeams) {
    const size_t beam_threadidx = threadIdx.x % beams_per_warp_Ju;
    const size_t time_threadidx = threadIdx.x / beams_per_warp_Ju;
    assert(time_threadidx < times_per_warp_Ju);
    const size_t b = beam0 + beam_threadidx;
    assert(b < nbeams);
    const size_t t = time_threadidx;
    assert(t < ntimes_per_iteration_Ju);

    // TODO: Vectorize this
    int8_t J[npolarizations][ncomplex];
    for (size_t p = 0; p < npolarizations; ++p) {
      for (size_t c = 0; c < ncomplex; ++c) {
        J[p][c] = 0;
      }
    }
    for (size_t d = 0; d < nwarps_for_dishes; ++d) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t c = 0; c < ncomplex; ++c) {
          J[p][c] += Jushared[d][b][t][p][c];
        }
      }
    }
    for (size_t p = 0; p < npolarizations; ++p) {
      // Convert to 4 bits and add bias
      Jshared[b][time0 % ntimes_per_iteration_J + t][p] = ucomplex4(J[p][0], J[p][1]);
    }
  }
}

__device__ void transpose_J(const ucomplex4 (&restrict Jshared)[nbeams][ntimes_per_iteration_J + padding_J][npolarizations],
                            const size_t time0) {}

////////////////////////////////////////////////////////////////////////////////

__global__ void form_beams(ucomplex4 *restrict const Jarray, const ucomplex4 *restrict const Earray,
                           const ucomplex4 *restrict const Aarray, const float *restrict const Garray) {

  // Each frequency is transformed independently. We use one thread
  // block per frequency.

  const size_t f = blockIdx.x;

  // Load A into registers
  fragment<wmma::matrix_a, m, n, k, experimental::precision::s4, row_major> Aregister[ncomplex][nbeam_matrices_per_thread]
                                                                                     [ndish_matrices_per_thread];
  load_A(Aregister, Aarray, f);

  // Loop over times

  for (size_t time0 = 0; time0 < ntimes; time0 += ntimes_per_iteration_Ju) {

    // Shuffle E-array
    __shared__ int32_t Eshared[ncomplex][ntimes_per_iteration_E][npolarizations][ndishes / nreals_int + padding_E];
    if (time0 % ntimes_per_iteration_E == 0) {
      shuffle_E(Eshared, Earray, f, time0);
      __syncthreads();
    }

    // Calculate Ju
    __shared__ int8_t Jushared[nwarps_for_dishes][nbeams][ntimes_per_iteration_Ju][npolarizations][ncomplex];
    assert(time0 % ntimes_per_iteration_Ju == 0);
    compute_Ju(Jushared, Aregister, Eshared, Garray, f, time0);
    __syncthreads();

    // Reduce to J
    __shared__ ucomplex4 Jshared[nbeams][ntimes_per_iteration_J + padding_J][npolarizations];
    assert(time0 % ntimes_per_iteration_Ju == 0);
    reduce_to_J(Jshared, Jushared, time0);
    __syncthreads();

    if ((time0 + ntimes_per_iteration_Ju) % ntimes_per_iteration_J == 0)
      transpose_J(Jshared, time0);

  } // for time0
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
