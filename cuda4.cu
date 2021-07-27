// -*-c++-*-
// Beamforming with CUDA

#include "arraysizes.hxx"
#include "icomplex4.hxx"

#include <mma.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

using namespace std;

using namespace nvcuda;
using namespace nvcuda::wmma;

#undef DEBUG_A_ARRAY_READ
#undef DEBUG_E_ARRAY_READ
#undef DEBUG_J_ARRAY_WRITE

#undef DEBUG_E_SHARED_WRITE
#undef DEBUG_E_SHARED_READ
#undef DEBUG_JU_SHARED_WRITE
#undef DEBUG_JU_SHARED_READ
#undef DEBUG_J_SHARED_WRITE
#undef DEBUG_J_SHARED_READ

////////////////////////////////////////////////////////////////////////////////

// J[b,f,t,p] = G[f,b] sum[d] A[f,b,d] E[t,f,d,p]

////////////////////////////////////////////////////////////////////////////////

// Index spaces

// Idea:
//
// Create "index spaces" that define the set of indices over which a kernel is applied.
//
// Index spaces cannot be split, but they can be combined.
//
// They can be combined to index arrays (and to define arrays).
//
// They can be combined into threads, warps, blocks, loop iterations, or be "explicit" (as in complex number components).
//
// There are checks that the mappings (to arrays and threads etc.) are complete, and don't overlap.
//
// Bank conflicts for shared and global memory should fall out naturally.

// Let's define the index spaces we are using:

// Note: We ignore frequencies here since we can map them to blocks.

// CUDA:
//
// 32 threads per warp
// 32 warps
//
// matrix m = 8    (beams)
// matrix n = 8    (times)
// matrix k = 32   (dishes)
//
// A[m][k]
// B[k][n]
// C[m][n]

// Input and output arrays:
//
// ntimes = 32768      (per chunk)
// nfrequencies = 32   (per GPU)
// ndishes = 512
// npolarizations = 2
// nbeams = 96
// ncomplex = 2        (complex number components)

// load_A:
//
// 3 iterations for beams
// 2 iterations for dishes
// 4 warps for beams
// 8 warps for dishes
// 32 threads for matrices
// 32 k matrix elements for dishes
// 8 m matrix elements for beams
// 2 explicit for complex numbers
//
// beam = (beam iteration) * (beam warp) * (beam m matrix element)
// dish = (dish warp) * (dish iteration) * (dish k matrix element)
// complex = (explicit)
//
// input: A[frequency][beam][dish][complex]
// output: A_register[complex][beam / A_register_beam_divisor][dish / A_register_dish_divisor % A_register_dish_modulo]
//         [beam % A_register_beam_matrix_modulo][dish % A_register_dish_matrix_modulo]
// A_register_beam_divisor = (# beam warp) * (# beam m matrix element)
// A_register_dish_divisor = (# dish k matrix element)
// A_register_dish_modulo = (# dish iterations)
// A_register_beam_matrix_modulo = (# beam m matrix element)
// A_register_dish_matrix_modulo = (# dish k matrix element)
//
// Replace dish by dish' above

// shuffle_E:
//
// 512 iterations (outer) for times
// 4 iterations (inner) for times
// 16 warps for times
// 2 warps for dishes
// 32 threads for dishes
// 4 explicit (outer) for dishes
// 2 explicit (inner) for dishes
// 2 explicit for polarization
// 2 explicit for complex numbers
//
// time = (time iteration outer) * (time iteration inner) * (time warp)
// dish = (dish warp) * (dish explicit outer) * (dish thread) * (dish explicit inner)
// polarization = (explicit)
// complex = (explicit)
// dish' = (dish warp) * (dish thread) * (dish explicit inner) * (dish explicit outer)
//
// input: E[time][frequency][dish][polarization][complex]
// output: E_shared[complex][time % E_shared_time_modulo][polarization][dish' + padding]
// E_shared_time_modulo = (# time warp)
//
// E layout:
//   dish warp
//     dish explicit outer
//        dish thread
//              dish explicit inner
//               polarization explicit
//                complex explicit
//   d dd ddddd dpc
//   8 76 54321 000
//
// Extract complex components:
//   cd d ddddd dpd
//   08 7 54321 006
//
// Extract polarization:
//   pcd ddddd ddd
//   008 54321 076

// compute_Ju:
//
// 512 iterations (outer) for times
// 4 iterations (inner) for times
// 4 iterations (inner2) for times
// 3 iterations for beams
// 2 iterations for dishes'
// 4 warps for beams
// 8 warps for dishes'
// 8 m matrix elements for beams
// 4 n matrix elements for time
// 2 n matrix elements for polarization
// 32 k matrix elements for dishes
// 2 explicit for complex numbers
//
// time = (time iteration outer) * (time iteration inner) * (time iteration inner2) * (time n matrix element)
// beam = (beam iteration) * (beam warp) * (beam m matrix element)
// dish' = (dish' warp) * (dish' iteration) * (dish' k matrix element)
// polarization = (polarization n matrix element)
// complex = explicit
//
// input: A_register[complex][beam][dish]   [beam][dish]
// input: E_shared[complex][time][polarization][dish' + padding][complex]
// output: Ju_shared[dish' / Ju_shared_dish'_divisor][beam][time % Ju_shared_time_modulo]   [polarization][complex]
// u_shared_dish'_divisor = # dish' iterations) * (# dish' k matrex elements)
// Ju_shared_time_modulo = (# time iteration inner2)

// reduce_to_J:
//
// 512 iterations (outer) for times
// 4 iterations (inner) for times
// 4 iterations (inner2) for times
// 12 warps for beams (other warps unused)
// 8 threads for beams
// 4 threads for times
// 8 iterations over dishes' (reduction)
// 2 explicit for polarization
// 2 explicit for complex numbers
//
// beam = (beam warp) * (beam thread)
// time = (time iteration (outer)) * (time iteration (inner)) * (time iteration (inner2)) * (time thread)
//
// input: Ju_shared[dish' / Ju_shared_dish'_divisor][beam][time % Ju_shared_time_modulo][polarization][complex]
// output: J_shared[beam][time % J_shared_time_modulo + padding][polarization][complex]
// J_shared_time_modulo = (# iterations (inner) for times) * (# iterations (inner2) for times) * (# threads for times)

// transpose_J:
//
// 512 iterations (outer) for times
// 3 iterations for beams
// 32 warps for beams
// 16 threads for time
// 2 threads for polarization
// 2 explicit for complex
//
// time = (time iteration (outer)) * (time thread)
// beam = (beam iteration) * (beam warp)
//
// input: J_shared[beam][time % J_shared_time_modulo + padding][polarization][complex]
// output: J[beam][frequency][time][polarization][complex]

////////////////////////////////////////////////////////////////////////////////

// Helper functions

__device__ int32_t clamp(int32_t i, int32_t i0, int32_t i1) { return max(i0, min(i1, i)); }

__device__ int32_t extract_real(const int32_t x0, const int32_t x1) {
  return (uint32_t(x0 & 0xf0f0f0f0U) >> 4) | (x1 & 0xf0f0f0f0U);
}

__device__ int32_t extract_imag(const int32_t x0, const int32_t x1) {
  return (x0 & 0x0f0f0f0fU) | (uint32_t(x1 & 0x0f0f0f0fU) << 4);
}

////////////////////////////////////////////////////////////////////////////////

// CUDA related constants

constexpr size_t num_reals_int = 8; // number of real numbers (complex number components) per 32-bit int

constexpr size_t num_threads = 32; // 32 threads per warp
constexpr size_t num_warps = 32;   // 32 warps

constexpr size_t num_m_elements = 8;  // (mostly) beams
constexpr size_t num_n_elements = 8;  // (mostly) times
constexpr size_t num_k_elements = 32; // (mostly) dishes

// Import algorithm settings

constexpr size_t num_times = ntimes;
constexpr size_t num_frequencies = nfrequencies;
constexpr size_t num_dishes = ndishes;
constexpr size_t num_polarizations = npolarizations;
constexpr size_t num_beams = nbeams;
constexpr size_t num_complex = ncomplex;

constexpr size_t num_dishes_prime = num_dishes;

////////////////////////////////////////////////////////////////////////////////

#ifdef DEBUG_A_ARRAY_READ
__device__ int A_mask[num_frequencies * num_beams * num_dishes];
#endif
#ifdef DEBUG_E_ARRAY_READ
__device__ int E_mask[num_times * num_frequencies * num_dishes * num_polarizations];
#endif
#ifdef DEBUG_J_ARRAY_WRITE
__device__ int J_mask[num_beams * num_frequencies * num_times * num_polarizations];
#endif

////////////////////////////////////////////////////////////////////////////////

namespace load_A {

constexpr size_t num_beam_iterations = 3;
constexpr size_t num_dish_prime_iterations = 2;
constexpr size_t num_beam_warps = 4;
constexpr size_t num_dish_prime_warps = 8;
constexpr size_t num_dish_prime_k_elements = 32;
constexpr size_t num_beam_m_elements = 8;
constexpr size_t num_complex_explicit = 2;

static_assert(num_beam_warps * num_dish_prime_warps == num_warps);
static_assert(num_beam_warps * num_beam_iterations * num_beam_m_elements == num_beams);
static_assert(num_dish_prime_iterations * num_dish_prime_warps * num_dish_prime_k_elements == num_dishes);

constexpr size_t A_register_beam_divisor = num_beam_warps * num_beam_m_elements;
constexpr size_t A_register_dish_prime_divisor = num_dish_prime_k_elements;
constexpr size_t A_register_dish_prime_modulo = num_dish_prime_iterations;
constexpr size_t A_register_beam_matrix_modulo = num_beam_m_elements;
constexpr size_t A_register_dish_prime_matrix_modulo = num_dish_prime_k_elements;

using A_register_t = fragment<wmma::matrix_a, num_m_elements, num_n_elements, num_k_elements, experimental::precision::s4,
                              row_major>[num_complex][num_beams / A_register_beam_divisor][A_register_dish_prime_modulo];

__device__ void load_A(A_register_t &restrict A_register, const ucomplex4 *restrict const A_array, const size_t frequency) {
  const size_t beam_warp = threadIdx.y / num_dish_prime_warps;
  const size_t dish_prime_warp = threadIdx.y % num_dish_prime_warps;
  for (size_t beam_iteration = 0; beam_iteration < num_beam_iterations; ++beam_iteration) {
    for (size_t dish_prime_iteration = 0; dish_prime_iteration < num_dish_prime_iterations; ++dish_prime_iteration) {
      const size_t beam = (beam_iteration * num_beam_warps + beam_warp) * num_beam_m_elements;
      const size_t dish_prime = (dish_prime_warp * num_dish_prime_iterations + dish_prime_iteration) * num_dish_prime_k_elements;
      assert(beam < num_beams);
      assert(dish_prime < num_dishes);

      // wmma::A[m][k]
      fragment<wmma::matrix_a, num_m_elements, num_n_elements, num_k_elements, experimental::precision::s4, row_major>
          A0[num_complex];
      for (size_t c = 0; c < num_complex; ++c) {

#ifdef DEBUG_A_ARRAY_READ
        if (threadIdx.x == 0) {
          const size_t A_offset = &A_array[A2linear(frequency, beam, c, dish_prime) / 2] - A_array;
          const size_t A_stride = A2linear(0, 1, 0, 0) / 2;
          for (size_t m = 0; m < 8; ++m) {
            for (size_t k = 0; k < 32 / 2; ++k) {
              const int oldval =
                  atomicMax(&A_mask[A_offset + m * A_stride + k], (blockIdx.x * 32 + threadIdx.y) * 32 + threadIdx.x);
              assert(oldval == -1);
            }
          }
        }
        __syncthreads();
#endif

        // TOOD: Use __ldcs
        // Load 2 consecutive sets of elements of A
        load_matrix_sync(A0[c], &A_array[A2linear(frequency, beam, c, dish_prime) / 2], A2linear(0, 1, 0, 0));
      }

      assert(beam / A_register_beam_divisor == beam_iteration);
      assert(dish_prime / A_register_dish_prime_divisor % A_register_dish_prime_modulo == dish_prime_iteration);
      static_assert(A_register[0][beam_iteration][dish_prime_iteration].num_storage_elements == 1);
      for (int i = 0; i < A_register[0][beam_iteration][dish_prime_iteration].num_storage_elements; ++i) {
        // Extract complex components and remove bias
        for (size_t c = 0; c < num_complex; ++c) {
          A_register[c][beam_iteration][dish_prime_iteration].x[i] = A0[c].x[i] ^ 0x88888888U;

          // #warning "TODO"
          // {
          //   for (int w = 0; w < 32; ++w) {
          //     for (int t = 0; t < 32; ++t) {
          //       __syncthreads();
          //       if (threadIdx.y == w && threadIdx.x == t) {
          //         if (A_register[c][beam_iteration][dish_prime_iteration].x[i] != 0) {
          //           printf("w=%02d t=%02d c=%01d b=%02d d'=%03d A=0x%08x\n", int(w), int(t), int(c), int(beam), int(dish_prime),
          //                  unsigned(A_register[c][beam_iteration][dish_prime_iteration].x[i]));
          //         }
          //       }
          //     }
          //   }
          // }
          //
        }
      }
    }
  }
}

} // namespace load_A

using load_A::num_beam_m_elements;
using load_A::num_dish_prime_k_elements;

using load_A::A_register_beam_divisor;
using load_A::A_register_beam_matrix_modulo;
using load_A::A_register_dish_prime_divisor;
using load_A::A_register_dish_prime_matrix_modulo;
using load_A::A_register_dish_prime_modulo;

using load_A::A_register_t;

////////////////////////////////////////////////////////////////////////////////

namespace shuffle_E {

constexpr size_t num_time_iterations_outer = num_times / 64;
constexpr size_t num_time_iterations_inner = 4;
constexpr size_t num_time_warps = 16;
constexpr size_t num_dish_warps = 2;
constexpr size_t num_dish_threads = 32;
constexpr size_t num_dish_explicit_outer = 4;
constexpr size_t num_dish_explicit_inner = 2;
constexpr size_t num_polarization_explicit = 2;
constexpr size_t num_complex_explicit = 2;

static_assert(num_time_warps * num_dish_warps == num_warps);
static_assert(num_dish_threads == num_threads);
static_assert(num_time_iterations_outer * num_time_iterations_inner * num_time_warps == num_times);
static_assert(num_dish_warps * num_dish_threads * num_dish_explicit_outer * num_dish_explicit_inner == num_dishes);

constexpr size_t E_shared_time_modulo = num_time_warps;
constexpr size_t E_shared_dish_prime_divisor = num_reals_int;
constexpr size_t E_shared_padding = 4;

using E_shared_t = uint32_t[num_complex][E_shared_time_modulo][num_polarizations]
                           [num_dishes_prime / E_shared_dish_prime_divisor + E_shared_padding];

#ifdef DEBUG_E_SHARED_WRITE
__device__ int E_shared_write_mask[num_complex][E_shared_time_modulo][num_polarizations]
                                  [num_dishes_prime / E_shared_dish_prime_divisor + E_shared_padding];
#endif
#ifdef DEBUG_E_SHARED_READ
__device__ int E_shared_read_mask[num_complex][E_shared_time_modulo][num_polarizations]
                                 [num_dishes_prime / E_shared_dish_prime_divisor + E_shared_padding];
#endif

__device__ void shuffle_E(E_shared_t &restrict E_shared, const ucomplex4 *restrict const E_array, const size_t frequency,
                          const size_t time_iteration_outer, const size_t time_iteration_inner) {
  const size_t time_warp = threadIdx.y / num_dish_warps;
  const size_t dish_warp = threadIdx.y % num_dish_warps;
  const size_t dish_thread = threadIdx.x % num_dish_threads;
  const size_t time = (time_iteration_outer * num_time_iterations_inner + time_iteration_inner) * num_time_warps + time_warp;
  const size_t dish0 = (dish_warp * num_dish_explicit_outer * num_dish_threads + dish_thread) * num_dish_explicit_inner;
  const size_t dish0_prime = (dish_warp * num_dish_threads + dish_thread) * num_dish_explicit_inner * num_dish_explicit_outer;
  assert(time < num_times);
  assert(dish0 < num_dishes);
  assert(dish0_prime < num_dishes);

  // Load E-field from global memory
  // Note: These are not yet split into polarizations complex components. `p` and `c` are the "outer" explicit indices.
  uint32_t E0[num_polarizations][num_complex];
  for (size_t p = 0; p < num_polarizations; ++p) {
    for (size_t c = 0; c < num_complex; ++c) {
      static_assert(num_dish_explicit_outer == num_polarizations * num_complex);
      static_assert(num_dish_explicit_outer * num_dish_explicit_inner == num_reals_int);
      const size_t dish = dish0 + (p * num_complex + c) * num_dish_threads * num_dish_explicit_inner;
      assert(dish < num_dishes);
      assert(uintptr_t(&E_array[Elinear(time, frequency, dish, 0, 0) / 2]) % sizeof(uint32_t) == 0);

#ifdef DEBUG_E_ARRAY_READ
      for (size_t i = 0; i < 4; ++i) {
        const int oldval =
            atomicMax(&E_mask[Elinear(time, frequency, dish, 0, 0) / 2 + i], (blockIdx.x * 32 + threadIdx.y) * 32 + threadIdx.x);
        assert(oldval == -1);
      }
#endif

      // #warning "TODO"
      //       {
      //         if (threadIdx.y == 0) {
      //           for (int t = 0; t < 32; ++t) {
      //             __syncthreads();
      //             if (threadIdx.x == t) {
      //               printf("t=%d p=%d c=%d d=%d idx=%d\n", int(t), int(p), int(c), int(dish), int(Elinear(time, frequency, dish,
      //               0, 0)));
      //             }
      //           }
      //         }
      //       }

      // TOOD: Use __ldcs
      E0[p][c] = *(const uint32_t *)&E_array[Elinear(time, frequency, dish, 0, 0) / 2];

      // #warning "TODO"
      // {
      //   for (int w = 0; w < 32; ++w) {
      //     for (int t = 0; t < 32; ++t) {
      //       __syncthreads();
      //       if (threadIdx.y == w && threadIdx.x == t) {
      //         if ((E0[p][c] ^ 0x88888888) != 0) {
      //           printf("w=%02d t=%02d p=%01d c=%01d d=%03d E0=0x%08x\n", int(w), int(t), int(p), int(c), int(dish),
      //                  unsigned(E0[p][c] ^ 0x88888888));
      //         }
      //       }
      //     }
      //   }
      // }
      //
    }
  }

  // Layout: p=0..1 (1 bit), p=0..1 (1 bit), e=0..7 ("element" inside each int, 3 bits), t=0..31 (thread)
  // Notation: p0 := bit 0 of p; e2 := bit 2 of e, etc.
  //
  // E0[p][c][e] = E[d=p0c0t4t3t2t1t0e2, p=e1, c=e0]
  // E1[p][c][e] = E[d=p0e0t4t3t2t1t0e2, p=e1, c=c0]
  // E2[p][c][e] = E[d=e1e0t4t3t2t1t0e2, p=p0, c=c0]

  // Dish layout over ints:
  // E0[0][0] = [1 1 1 1 0 0 0 0]
  // E0[0][1] = [65 65 65 65 64 64 64 64]
  // E1[0][0] = [65 1 65 1 64 0 64 0]
  // E1[1][0] = [193 129 193 129 192 128 192 128]
  // E2[0][0] = [193 129 65 1 192 128 64 0]

  // First we split out the complex components and remove the bias
  uint32_t E1[num_polarizations][num_complex];
  for (size_t p = 0; p < num_polarizations; ++p) {
    E1[p][0] = extract_real(E0[p][0], E0[p][1]) ^ 0x88888888U;
    E1[p][1] = extract_imag(E0[p][0], E0[p][1]) ^ 0x88888888U;
  }

  // Next we separate the polarizations
  uint32_t E2[num_polarizations][num_complex];
  for (size_t c = 0; c < num_complex; ++c) {
    E2[0][c] = __byte_perm(E1[0][c], E1[1][c], 0x6240);
    E2[1][c] = __byte_perm(E1[0][c], E1[1][c], 0x7351);
  }

  // Store into shared memory
  for (size_t c = 0; c < num_complex; ++c) {
    for (size_t p = 0; p < num_polarizations; ++p) {
      const size_t dish_prime = dish0_prime;
      assert(dish_prime < num_dishes_prime);

      // #warning "TODO"
      // {
      //   for (int w = 0; w < 32; ++w) {
      //     for (int t = 0; t < 32; ++t) {
      //       __syncthreads();
      //       if (threadIdx.y == w && threadIdx.x == t) {
      //         if (E2[p][c] != 0) {
      //           printf("w=%02d t=%02d p=%01d c=%01d d'=%03d E2=0x%08x\n", int(w), int(t), int(p), int(c), int(dish_prime),
      //                  unsigned(E2[p][c]));
      //         }
      //       }
      //     }
      //   }
      // }

#ifdef DEBUG_E_SHARED_WRITE
      const int oldval =
          atomicMax(&E_shared_write_mask[c][time % E_shared_time_modulo][p][dish_prime / E_shared_dish_prime_divisor],
                    (blockIdx.x * 32 + threadIdx.y) * 32 + threadIdx.x);
      assert(oldval == -1);
#endif

      E_shared[c][time % E_shared_time_modulo][p][dish_prime / E_shared_dish_prime_divisor] = E2[p][c];
    }
  }
}

} // namespace shuffle_E

using shuffle_E::num_time_iterations_inner;
using shuffle_E::num_time_iterations_outer;

using shuffle_E::E_shared_dish_prime_divisor;
using shuffle_E::E_shared_time_modulo;

using shuffle_E::E_shared_t;

////////////////////////////////////////////////////////////////////////////////

namespace compute_Ju {

constexpr size_t num_time_iterations_inner2 = 4;
constexpr size_t num_beam_iterations = 3;
constexpr size_t num_dish_prime_iterations = 2;
constexpr size_t num_beam_warps = 4;
constexpr size_t num_dish_prime_warps = 8;
constexpr size_t num_beam_m_elements = 8;
constexpr size_t num_time_n_elements = 4;
constexpr size_t num_polarization_n_elements = 2;
constexpr size_t num_dish_prime_k_elements = 32;
constexpr size_t num_complex_explicit = 2;

static_assert(num_beam_warps * num_dish_prime_warps == num_warps);
static_assert(num_time_iterations_outer * num_time_iterations_inner * num_time_iterations_inner2 * num_time_n_elements ==
              num_times);
static_assert(num_beam_iterations * num_beam_warps * num_beam_m_elements == num_beams);
static_assert(num_dish_prime_iterations * num_dish_prime_warps * num_dish_prime_k_elements == num_dishes);

constexpr size_t Ju_shared_dish_prime_divisor = num_dish_prime_iterations * num_dish_prime_k_elements;
constexpr size_t Ju_shared_time_modulo = num_time_iterations_inner2;
using Ju_shared_t =
    uint32_t[num_dishes_prime / Ju_shared_dish_prime_divisor][num_beams][Ju_shared_time_modulo]; // [polarization][complex]

#ifdef DEBUG_JU_SHARED_WRITE
__device__ int Ju_shared_write_mask[num_dishes_prime / Ju_shared_dish_prime_divisor][num_beams]
                                   [Ju_shared_time_modulo]; // [polarization][complex]
#endif
#ifdef DEBUG_JU_SHARED_READ
__device__ int Ju_shared_read_mask[num_dishes_prime / Ju_shared_dish_prime_divisor][num_beams]
                                  [Ju_shared_time_modulo]; // [polarization][complex]
#endif

__device__ void compute_Ju(Ju_shared_t &restrict Ju_shared, const A_register_t &restrict A_register,
                           const E_shared_t &restrict E_shared, const float *restrict const G_array, const size_t frequency,
                           const size_t time_iteration_outer, const size_t time_iteration_inner,
                           const size_t time_iteration_inner2) {
  const size_t beam_warp = threadIdx.y / num_dish_prime_warps;
  const size_t dish_prime_warp = threadIdx.y % num_dish_prime_warps;

  const size_t time0 = ((time_iteration_outer * num_time_iterations_inner + time_iteration_inner) * num_time_iterations_inner2 +
                        time_iteration_inner2) *
                       num_time_n_elements;
  const size_t dish_prime0 = dish_prime_warp * num_dish_prime_iterations * num_dish_prime_k_elements;

  // Load E-field from shared memory
  // wmma::B[k][n]
  fragment<wmma::matrix_b, num_m_elements, num_n_elements, num_k_elements, experimental::precision::s4, col_major>
      E[num_complex][num_dish_prime_iterations];

  for (size_t c = 0; c < num_complex; ++c) {
    for (size_t dish_prime_iteration = 0; dish_prime_iteration < num_dish_prime_iterations; ++dish_prime_iteration) {
      const size_t dish_prime1 = (dish_prime_warp * num_dish_prime_iterations + dish_prime_iteration) * num_dish_prime_k_elements;

#ifdef DEBUG_E_SHARED_READ
      if (threadIdx.x == 0) {
        const size_t E_stride = (&E_shared[0][0][1][0] - &E_shared[0][0][0][0]) * num_reals_int / 8;
        for (size_t n = 0; n < 8; ++n) {
          for (size_t k = 0; k < 32 / 8; ++k) {
            atomicAdd(
                &shuffle_E::E_shared_read_mask[c][time0 % E_shared_time_modulo][0][dish_prime1 / E_shared_dish_prime_divisor] +
                    n * E_stride + k,
                num_beam_iterations * num_beam_m_elements);
          }
        }
      }
#endif

      load_matrix_sync(E[c][dish_prime_iteration],
                       &E_shared[c][time0 % E_shared_time_modulo][0][dish_prime1 / E_shared_dish_prime_divisor],
                       (&E_shared[0][0][1][0] - &E_shared[0][0][0][0]) * num_reals_int);

      // #warning "TODO"
      // {
      //   const size_t beam_iteration = 0;
      //   const size_t beam0 = (beam_iteration * num_beam_warps + beam_warp) * num_beam_m_elements;
      //   for (int w = 0; w < 32; ++w) {
      //     for (int t = 0; t < 32; ++t) {
      //       __syncthreads();
      //       if (threadIdx.y == w && threadIdx.x == t) {
      //         for (int i = 0; i < E[c][dish_prime_iteration].num_storage_elements; ++i) {
      //           if (E[c][dish_prime_iteration].x[i] != 0) {
      //             printf("load w=%02d t=%02d time=%04d c=%01d b=%02d d'=%03d E=0x%08x\n", int(w), int(t), int(time0), int(c),
      //                    int(beam0), int(dish_prime1), unsigned(E[c][dish_prime_iteration].x[i]));
      //           }
      //         }
      //       }
      //     }
      //   }
      // }
      //
    }
  }

  for (size_t beam_iteration = 0; beam_iteration < num_beam_iterations; ++beam_iteration) {
    const size_t beam0 = (beam_iteration * num_beam_warps + beam_warp) * num_beam_m_elements;

    fragment<wmma::accumulator, num_m_elements, num_n_elements, num_k_elements, int32_t> JurePos, JureNeg, JuimPos;

    // These depend on the undocumented fragment layout
    static_assert(JurePos.num_storage_elements == num_polarizations);
    const size_t element0 = threadIdx.x * JurePos.num_storage_elements;
    const size_t beam = beam0 + element0 / num_polarizations / num_time_n_elements % num_beam_m_elements;
    const size_t time = time0 + element0 / num_polarizations % num_time_n_elements;

    // Initialize Ju
    fill_fragment(JurePos, 0);
    fill_fragment(JureNeg, 0);
    fill_fragment(JuimPos, 0);

    // Multiply
    for (size_t dish_prime_iteration = 0; dish_prime_iteration < num_dish_prime_iterations; ++dish_prime_iteration) {
      const size_t dish_prime1 = (dish_prime_warp * num_dish_prime_iterations + dish_prime_iteration) * num_dish_prime_k_elements;

      // #warning "TODO"
      // {
      //   for (int w = 0; w < 32; ++w) {
      //     for (int t = 0; t < 32; ++t) {
      //       __syncthreads();
      //       if (threadIdx.y == w && threadIdx.x == t) {
      //         for (int c = 0; c < 2; ++c) {
      //           for (int i = 0; i < A_register[c][beam_iteration][dish_prime_iteration].num_storage_elements; ++i) {
      //             if (A_register[c][beam_iteration][dish_prime_iteration].x[i] != 0) {
      //               printf("mma w=%02d t=%02d time=%04d c=%01d b=%02d d'=%03d A=0x%08x\n", int(w), int(t), int(time), int(c),
      //                      int(beam), int(dish_prime1), unsigned(A_register[c][beam_iteration][dish_prime_iteration].x[i]));
      //             }
      //           }
      //         }
      //       }
      //     }
      //   }
      // }

      // #warning "TODO"
      // {
      //   for (int w = 0; w < 32; ++w) {
      //     for (int t = 0; t < 32; ++t) {
      //       __syncthreads();
      //       if (threadIdx.y == w && threadIdx.x == t) {
      //         for (int c = 0; c < 2; ++c) {
      //           for (int i = 0; i < E[c][dish_prime_iteration].num_storage_elements; ++i) {
      //             if (E[c][dish_prime_iteration].x[i] != 0) {
      //               printf("mma w=%02d t=%02d time=%04d c=%01d b=%02d d'=%03d E=0x%08x\n", int(w), int(t), int(time), int(c),
      //                      int(beam), int(dish_prime1), unsigned(E[c][dish_prime_iteration].x[i]));
      //             }
      //           }
      //         }
      //       }
      //     }
      //   }
      // }

      static_assert(num_beam_iterations == num_beams / A_register_beam_divisor);
      static_assert(num_dish_prime_iterations == A_register_dish_prime_modulo);
      mma_sync(JurePos, A_register[0][beam_iteration][dish_prime_iteration], E[0][dish_prime_iteration], JurePos);
      mma_sync(JureNeg, A_register[1][beam_iteration][dish_prime_iteration], E[1][dish_prime_iteration], JureNeg);
      mma_sync(JuimPos, A_register[0][beam_iteration][dish_prime_iteration], E[1][dish_prime_iteration], JuimPos);
      mma_sync(JuimPos, A_register[1][beam_iteration][dish_prime_iteration], E[0][dish_prime_iteration], JuimPos);
    }

    assert(uintptr_t(&G_array[Glinear(frequency, beam)]) % sizeof(float) == 0);
    const float G = G_array[Glinear(frequency, beam)];

    // Extract result from Ju matrix
    int8_t Ju8[num_polarizations][num_complex];
    static_assert(JurePos.num_storage_elements == num_polarizations);
    for (size_t i = 0; i < JurePos.num_storage_elements; ++i) {
      const size_t p = i;
      // Combine positive and negative J values, and reduce from 32 to 8 bits
      int32_t Ju[num_complex];
      Ju[0] = JurePos.x[i] - JureNeg.x[i];
      Ju[1] = JuimPos.x[i];
      for (size_t c = 0; c < num_complex; ++c) {
#warning "TODO: add overflow checks"
#warning "TODO: sum/round is in different order"
        Ju8[p][c] = clamp(int32_t(lrintf(G * float(Ju[c]))), -127, 127);
      }
    }
    // CUDA is little endian
    const uint32_t Ju8all = uint32_t(uint8_t(Ju8[0][0])) | (uint32_t(uint8_t(Ju8[0][1])) << 8) |
                            (uint32_t(uint8_t(Ju8[1][0])) << 16) | (uint32_t(uint8_t(Ju8[1][1])) << 24);

    assert(dish_prime0 / Ju_shared_dish_prime_divisor == dish_prime_warp);

#ifdef DEBUG_JU_SHARED_WRITE
    const int oldval = atomicMax(&Ju_shared_write_mask[dish_prime_warp][beam][time % Ju_shared_time_modulo],
                                 (blockIdx.x * 32 + threadIdx.y) * 32 + threadIdx.x);
    assert(oldval == -1);
#endif

    Ju_shared[dish_prime_warp][beam][time % Ju_shared_time_modulo] = Ju8all;
  }
}
} // namespace compute_Ju

using compute_Ju::num_time_iterations_inner2;

using compute_Ju::Ju_shared_dish_prime_divisor;
using compute_Ju::Ju_shared_time_modulo;

using compute_Ju::Ju_shared_t;

////////////////////////////////////////////////////////////////////////////////

namespace reduce_to_J {

constexpr size_t num_beam_warps = 12; // other warps are unused
constexpr size_t num_beam_threads = 8;
constexpr size_t num_time_threads = 4;
constexpr size_t num_dish_prime_iterations = 8; // for reduction
constexpr size_t num_polarizations_explicit = 2;
constexpr size_t num_complex_explicit = 2;

static_assert(num_beam_warps <= num_warps);
static_assert(num_beam_threads * num_time_threads == num_threads);
static_assert(num_time_iterations_outer * num_time_iterations_inner * num_time_iterations_inner2 * num_time_threads == num_times);
static_assert(num_beam_warps * num_beam_threads == num_beams);
static_assert(num_dish_prime_iterations == num_dishes_prime / Ju_shared_dish_prime_divisor);

constexpr size_t J_shared_time_modulo = num_time_iterations_inner * num_time_iterations_inner2 * num_time_threads;
constexpr size_t J_shared_padding = 4;

using J_shared_t = uint16_t[num_beams][J_shared_time_modulo + J_shared_padding]; // [polarization][complex]

#ifdef DEBUG_J_SHARED_WRITE
__device__ int J_shared_write_mask[num_beams][J_shared_time_modulo + J_shared_padding]; // [polarization][complex]
#endif
#ifdef DEBUG_J_SHARED_READ
__device__ int J_shared_read_mask[num_beams][J_shared_time_modulo + J_shared_padding]; // [polarization][complex]
#endif

__device__ void reduce_to_J(J_shared_t &restrict J_shared, const Ju_shared_t &restrict Ju_shared, const size_t time_iteration_outer,
                            const size_t time_iteration_inner, const size_t time_iteration_inner2) {
  const size_t beam_warp = threadIdx.y;
  if (beam_warp < num_beam_warps) {
    // Other warps are unused
    const size_t beam_thread = threadIdx.x / num_time_threads;
    const size_t time_thread = threadIdx.x % num_time_threads;
    const size_t beam = beam_warp * num_beam_threads + beam_thread;
    const size_t time = ((time_iteration_outer * num_time_iterations_inner + time_iteration_inner) * num_time_iterations_inner2 +
                         time_iteration_inner2) *
                            num_time_threads +
                        time_thread;

    // TODO: Vectorize this
    int8_t J[num_polarizations][num_complex];
    for (size_t p = 0; p < num_polarizations; ++p) {
      for (size_t c = 0; c < num_complex; ++c) {
        J[p][c] = 0;
      }
    }
    for (size_t dish_prime_iteration = 0; dish_prime_iteration < num_dish_prime_iterations; ++dish_prime_iteration) {
      assert(dish_prime_iteration < num_dishes_prime / Ju_shared_dish_prime_divisor);

#ifdef DEBUG_JU_SHARED_READ
      const int oldval = atomicMax(&compute_Ju::Ju_shared_read_mask[dish_prime_iteration][beam][time % Ju_shared_time_modulo],
                                   (blockIdx.x * 32 + threadIdx.y) * 32 + threadIdx.x);
      assert(oldval == -1);
#endif

      const uint32_t Ju = Ju_shared[dish_prime_iteration][beam][time % Ju_shared_time_modulo];
      int8_t Ju8[num_polarizations][num_complex];
      Ju8[0][0] = int8_t(Ju & 0xffU);
      Ju8[0][1] = int8_t((Ju >> 8) & 0xffU);
      Ju8[1][0] = int8_t((Ju >> 16) & 0xffU);
      Ju8[1][1] = int8_t((Ju >> 24) & 0xffU);
      for (size_t p = 0; p < num_polarizations; ++p) {
        for (size_t c = 0; c < num_complex; ++c) {
          J[p][c] += Ju8[p][c];
        }
      }
    }
    // Convert to 4 bits and add bias
    uint8_t J4[2];
    for (size_t p = 0; p < num_polarizations; ++p) {
      J4[p] = ((uint32_t(clamp(J[p][0], -7, 7)) & 0x0f) << 4) | (uint32_t(clamp(J[p][1], -7, 7)) & 0x0f);
    }
    // Combine polarizations and add bias
    const uint16_t J4all = (uint32_t(J4[0]) | (uint32_t(J4[1]) << 8)) ^ 0x8888U;

#ifdef DEBUG_J_SHARED_WRITE
    const int oldval =
        atomicMax(&J_shared_write_mask[beam][time % J_shared_time_modulo], (blockIdx.x * 32 + threadIdx.y) * 32 + threadIdx.x);
    assert(oldval == -1);
#endif

    J_shared[beam][time % J_shared_time_modulo] = J4all;
  }
}
} // namespace reduce_to_J

using reduce_to_J::J_shared_time_modulo;

using reduce_to_J::J_shared_t;

////////////////////////////////////////////////////////////////////////////////

namespace transpose_J {

constexpr size_t num_beam_iterations = 3;
constexpr size_t num_beam_warps = 32;
constexpr size_t num_time_threads = 32;
constexpr size_t num_time_explicit = 2;
constexpr size_t num_polarizations_explicit = 2;
constexpr size_t num_complex_explicit = 2;

static_assert(num_beam_warps == num_warps);
static_assert(num_time_threads == num_threads);
static_assert(num_time_explicit * num_polarizations_explicit * num_complex_explicit == num_reals_int);
static_assert(num_time_iterations_outer * num_time_threads * num_time_explicit == num_times);
static_assert(num_beam_iterations * num_beam_warps == num_beams);

__device__ void transpose_J(ucomplex4 *restrict const J_array, const J_shared_t &restrict J_shared, const size_t frequency,
                            const size_t time_iteration_outer) {
  const size_t beam_warp = threadIdx.y;
  const size_t time_thread = threadIdx.x;
  const size_t time0 = (time_iteration_outer * num_time_threads + time_thread) * num_time_explicit;
  for (size_t beam_iteration = 0; beam_iteration < num_beam_iterations; ++beam_iteration) {
    const size_t beam = beam_iteration * num_beam_warps + beam_warp;
    // Load from shared memory

#ifdef DEBUG_J_SHARED_READ
    for (int i = 0; i < 2; ++i) {
      const int oldval = atomicMax(&reduce_to_J::J_shared_read_mask[beam][time0 % J_shared_time_modulo] + i,
                                   (blockIdx.x * 32 + threadIdx.y) * 32 + threadIdx.x);
      assert(oldval == -1);
    }
#endif

    const uint32_t Jall = *(const uint32_t *)&J_shared[beam][time0 % J_shared_time_modulo];

#ifdef DEBUG_J_ARRAY_WRITE
    for (size_t i = 0; i < 4; ++i) {
      const int oldval =
          atomicMax(&J_mask[J2linear(beam, frequency, time0, 0, 0) / 2 + i], (blockIdx.x * 32 + threadIdx.y) * 32 + threadIdx.x);
      assert(oldval == -1);
    }
#endif

    // Write to global memory
    // TOOD: Use __stcs
    *(uint32_t *)&J_array[J2linear(beam, frequency, time0, 0, 0) / 2] = Jall;
  }
}
} // namespace transpose_J

////////////////////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(num_threads *num_warps, 1)
    form_beams(ucomplex4 *restrict const J_array, const ucomplex4 *restrict const E_array, const ucomplex4 *restrict const A_array,
               const float *restrict const G_array) {

  // Each frequency is transformed independently. We use one thread block per frequency.

  const size_t frequency = blockIdx.x;

#ifdef DEBUG_A_ARRAY_READ
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (size_t i = 0; i < num_frequencies * num_beams * num_dishes; ++i) {
      A_mask[i] = -1;
    }
  }
  __syncthreads();
#endif
#ifdef DEBUG_E_ARRAY_READ
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (size_t i = 0; i < num_times * num_frequencies * num_dishes * num_polarizations; ++i) {
      E_mask[i] = -1;
    }
  }
  __syncthreads();
#endif
#ifdef DEBUG_J_ARRAY_WRITE
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (size_t i = 0; i < num_beams * num_frequencies * num_times * num_polarizations; ++i) {
      J_mask[i] = -1;
    }
  }
  __syncthreads();
#endif

  // Load A into registers
  load_A::A_register_t A_register;
  load_A::load_A(A_register, A_array, frequency);

  for (size_t time_iteration_outer = 0; time_iteration_outer < num_time_iterations_outer; ++time_iteration_outer) {
    __shared__ E_shared_t E_shared;
    __shared__ Ju_shared_t Ju_shared;
    __shared__ J_shared_t J_shared;

#ifdef DEBUG_J_SHARED_WRITE
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      for (size_t b = 0; b < num_beams; ++b) {
        for (size_t t = 0; t < J_shared_time_modulo; ++t) {
          reduce_to_J::J_shared_write_mask[b][t] = -1;
        }
      }
    }
    __syncthreads();
#endif

    for (size_t time_iteration_inner = 0; time_iteration_inner < num_time_iterations_inner; ++time_iteration_inner) {

#ifdef DEBUG_E_SHARED_WRITE
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (size_t c = 0; c < num_complex; ++c) {
          for (size_t t = 0; t < E_shared_time_modulo; ++t) {
            for (size_t p = 0; p < num_polarizations; ++p) {
              for (size_t d = 0; d < num_dishes_prime / E_shared_dish_prime_divisor; ++d) {
                shuffle_E::E_shared_write_mask[c][t][p][d] = -1;
              }
            }
          }
        }
      }
      __syncthreads();
#endif

      shuffle_E::shuffle_E(E_shared, E_array, frequency, time_iteration_outer, time_iteration_inner);
      __syncthreads();

#ifdef DEBUG_E_SHARED_WRITE
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (size_t c = 0; c < num_complex; ++c) {
          for (size_t t = 0; t < E_shared_time_modulo; ++t) {
            for (size_t p = 0; p < num_polarizations; ++p) {
              for (size_t d = 0; d < num_dishes_prime / E_shared_dish_prime_divisor; ++d) {
                assert(shuffle_E::E_shared_write_mask[c][t][p][d] >= 0);
              }
            }
          }
        }
      }
#endif

#ifdef DEBUG_E_SHARED_READ
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (size_t c = 0; c < num_complex; ++c) {
          for (size_t t = 0; t < E_shared_time_modulo; ++t) {
            for (size_t p = 0; p < num_polarizations; ++p) {
              for (size_t d = 0; d < num_dishes_prime / E_shared_dish_prime_divisor; ++d) {
                shuffle_E::E_shared_read_mask[c][t][p][d] = 0;
              }
            }
          }
        }
      }
      __syncthreads();
#endif

      for (size_t time_iteration_inner2 = 0; time_iteration_inner2 < num_time_iterations_inner2; ++time_iteration_inner2) {

#ifdef DEBUG_JU_SHARED_WRITE
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          for (size_t d = 0; d < num_dishes_prime / Ju_shared_dish_prime_divisor; ++d) {
            for (size_t b = 0; b < num_beams; ++b) {
              for (size_t t = 0; t < Ju_shared_time_modulo; ++t) {
                compute_Ju::Ju_shared_write_mask[d][b][t] = -1;
              }
            }
          }
        }
        __syncthreads();
#endif

        compute_Ju::compute_Ju(Ju_shared, A_register, E_shared, G_array, frequency, time_iteration_outer, time_iteration_inner,
                               time_iteration_inner2);
        __syncthreads();

#ifdef DEBUG_JU_SHARED_WRITE
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          for (size_t d = 0; d < num_dishes_prime / Ju_shared_dish_prime_divisor; ++d) {
            for (size_t b = 0; b < num_beams; ++b) {
              for (size_t t = 0; t < Ju_shared_time_modulo; ++t) {
                assert(compute_Ju::Ju_shared_write_mask[d][b][t] >= 0);
              }
            }
          }
        }
#endif

#ifdef DEBUG_JU_SHARED_READ
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          for (size_t d = 0; d < num_dishes_prime / Ju_shared_dish_prime_divisor; ++d) {
            for (size_t b = 0; b < num_beams; ++b) {
              for (size_t t = 0; t < Ju_shared_time_modulo; ++t) {
                compute_Ju::Ju_shared_read_mask[d][b][t] = -1;
              }
            }
          }
        }
        __syncthreads();
#endif

        reduce_to_J::reduce_to_J(J_shared, Ju_shared, time_iteration_outer, time_iteration_inner, time_iteration_inner2);

#ifdef DEBUG_JU_SHARED_READ
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          for (size_t d = 0; d < num_dishes_prime / Ju_shared_dish_prime_divisor; ++d) {
            for (size_t b = 0; b < num_beams; ++b) {
              for (size_t t = 0; t < Ju_shared_time_modulo; ++t) {
                assert(compute_Ju::Ju_shared_read_mask[d][b][t] >= 0);
              }
            }
          }
        }
#endif

      } // for time_iteration_inner2

#ifdef DEBUG_E_SHARED_READ
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        // E_shared needs to be read for every beam
        for (size_t c = 0; c < num_complex; ++c) {
          for (size_t t = 0; t < E_shared_time_modulo; ++t) {
            for (size_t p = 0; p < num_polarizations; ++p) {
              for (size_t d = 0; d < num_dishes_prime / E_shared_dish_prime_divisor; ++d) {
                assert(shuffle_E::E_shared_read_mask[c][t][p][d] == num_beams);
              }
            }
          }
        }
      }
#endif

    } // for time_iteration_inner

#ifdef DEBUG_J_SHARED_WRITE
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      for (size_t b = 0; b < num_beams; ++b) {
        for (size_t t = 0; t < J_shared_time_modulo; ++t) {
          assert(reduce_to_J::J_shared_write_mask[b][t] >= 0);
        }
      }
    }
#endif

#ifdef DEBUG_J_SHARED_READ
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      for (size_t b = 0; b < num_beams; ++b) {
        for (size_t t = 0; t < J_shared_time_modulo; ++t) {
          reduce_to_J::J_shared_read_mask[b][t] = -1;
        }
      }
    }
    __syncthreads();
#endif

    __syncthreads();
    transpose_J::transpose_J(J_array, J_shared, frequency, time_iteration_outer);

#ifdef DEBUG_J_SHARED_READ
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      for (size_t b = 0; b < num_beams; ++b) {
        for (size_t t = 0; t < J_shared_time_modulo; ++t) {
          assert(reduce_to_J::J_shared_read_mask[b][t] >= 0);
        }
      }
    }
#endif

  } // for time_iteration_outer

#ifdef DEBUG_A_ARRAY_READ
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (size_t i = 0; i < num_frequencies * num_beams * num_dishes; ++i) {
      assert(A_mask[i] >= 0);
    }
  }
#endif
#ifdef DEBUG_E_ARRAY_READ
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (size_t i = 0; i < num_times * num_frequencies * num_dishes * num_polarizations; ++i) {
      assert(E_mask[i] >= 0);
    }
  }
#endif
#ifdef DEBUG_J_ARRAY_WRITE
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (size_t i = 0; i < num_beams * num_frequencies * num_times * num_polarizations; ++i) {
      assert(J_mask[i] >= 0);
    }
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////

#define CHECK_RESULT(err) check_result(__FILE__, __LINE__, err)
void check_result(const char *file, int line, cudaError_t err) {
  if (err != cudaSuccess) {
    cerr << file << ":" << line << ": CUDA error " << err << ": " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err)
         << "\n";
    exit(1);
  }
}

////////////////////////////////////////////////////////////////////////////////

void setup_one(vector<ucomplex4> &Earray, vector<ucomplex4> &Aarray, vector<float> &Garray, vector<ucomplex4> &J2array,
               vector<ucomplex4> &J2array_expected) {
  // Test: Choose one beam, frequency, time, polarization, and dish. Set everything else to zero.

  // mt19937 engine(42);
  mt19937 engine(time(0));

  const size_t beam = uniform_int_distribution<size_t>(0, num_beams - 1)(engine);
  const size_t frequency = uniform_int_distribution<size_t>(0, num_frequencies - 1)(engine);
  const size_t time = uniform_int_distribution<size_t>(0, num_times - 1)(engine);
  const size_t polarization = uniform_int_distribution<size_t>(0, num_polarizations - 1)(engine);
  const size_t dish = uniform_int_distribution<size_t>(0, num_dishes - 1)(engine);
  // const size_t beam = 0;
  // const size_t frequency = 0;
  // const size_t time = 0;
  // const size_t polarization = 0;
  // const size_t dish = 0;

  const auto has_overflow = [](auto G, auto A, auto E) {
    return abs(real(A * E)) > 7 || abs(imag(A * E)) > 7 || abs(real(G * A * E)) > 7 || abs(imag(G * A * E)) > 7;
  };

  complex<int> Aval, Eval;
  int Gval;
  do {
    Aval = complex<int>(uniform_int_distribution<int>(-2, 2)(engine), uniform_int_distribution<int>(-2, 2)(engine));
    Eval = complex<int>(uniform_int_distribution<int>(-2, 2)(engine), uniform_int_distribution<int>(-2, 2)(engine));
    Gval = uniform_int_distribution<int>(-2, 2)(engine);
    // Aval = complex<int>(1, 0);
    // Eval = complex<int>(1, 0);
    // Gval = 1;
  } while (has_overflow(Gval, Aval, Eval));
  const complex<int> Jval = Gval * Aval * Eval;

  const ucomplex4 Aval4(Aval);
  const ucomplex4 Eval4(Eval);
  const float Gvalf(Gval);
  const ucomplex4 Jval4(Jval);

  cout << "beam=" << beam << " frequency=" << frequency << " time=" << time << " polarization=" << polarization << " dish=" << dish
       << "\n";
  cout << "Aval=" << Aval << " Eval=" << Eval << " Gval=" << Gval << " Jval=" << Jval << "\n";

  Earray.resize(Esize / 2);
  Aarray.resize(Asize / 2);
  Garray.resize(Gsize);
  J2array.resize(Jsize / 2);
  J2array_expected.resize(Jsize / 2);

  for (size_t t = 0; t < num_times; ++t) {
    for (size_t f = 0; f < num_frequencies; ++f) {
      for (size_t d = 0; d < num_dishes; ++d) {
        for (size_t p = 0; p < num_polarizations; ++p) {
          Earray.at(Elinear(t, f, d, p, 0) / 2) =
              t == time && f == frequency && d == dish && p == polarization ? Eval4 : ucomplex4(0, 0);
          // Earray.at(Elinear(t, f, d, p, 0) / 2) = ucomplex4(1,0);
        }
      }
    }
  }

  for (size_t f = 0; f < num_frequencies; ++f) {
    for (size_t b = 0; b < num_beams; ++b) {
      for (size_t d = 0; d < num_dishes; ++d) {
        Aarray.at(Alinear(f, b, d, 0) / 2) = f == frequency && b == beam && d == dish ? Aval4 : ucomplex4(0, 0);
        // Aarray.at(Alinear(f, b, d, 0) / 2) = ucomplex4(1, 0);
      }
    }
  }

  for (size_t f = 0; f < num_frequencies; ++f) {
    for (size_t b = 0; b < num_beams; ++b) {
      Garray.at(Glinear(f, b)) = f == frequency && b == beam ? Gvalf : 0;
      // Garray.at(Glinear(f, b)) = 1;
    }
  }

  for (size_t b = 0; b < num_beams; ++b) {
    for (size_t f = 0; f < num_frequencies; ++f) {
      for (size_t t = 0; t < num_times; ++t) {
        for (size_t p = 0; p < num_polarizations; ++p) {
          J2array_expected.at(J2linear(b, f, t, p, 0) / 2) =
              b == beam && f == frequency && t == time && p == polarization ? Jval4 : ucomplex4(0, 0);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  cout << "beamforming.cuda4\n";

  // vector<ucomplex4> Earray;
  // vector<ucomplex4> Aarray;
  // vector<float> Garray;
  // vector<ucomplex4> J2array;
  // setup(Earray, Aarray, Garray, J2array);

  vector<ucomplex4> Earray;
  vector<ucomplex4> Aarray;
  vector<float> Garray;
  vector<ucomplex4> J2array;
  vector<ucomplex4> J2array_expected;
  setup_one(Earray, Aarray, Garray, J2array, J2array_expected);

  // Change index order
  vector<ucomplex4> A2array(Aarray.size());
  {
    // for (size_t f = 0; f < num_frequencies; ++f) {
    //   for (size_t b = 0; b < num_beams; ++b) {
    //     for (size_t d = 0; d < num_dishes; d += 2) {
    //       for (size_t c = 0; c < num_complex / 2; ++c) {
    //         const uint8_t val = *(const uint8_t *)&Aarray.at(Alinear(f, b, d, c) / 2) ^ 0x88U;
    //         if (val != 0) {
    //           printf("f=%01d b=%02d d=%02d c=%01d A=0x%02x\n", int(f), int(b), int(d), int(c), unsigned(val));
    //         }
    //       }
    //     }
    //   }
    // }

    // bool Amask[num_frequencies][num_beams][num_dishes][num_complex];
    // for (size_t f = 0; f < num_frequencies; ++f) {
    //   for (size_t b = 0; b < num_beams; ++b) {
    //     for (size_t d = 0; d < num_dishes; ++d) {
    //       for (size_t c = 0; c < num_complex; ++c) {
    //         Amask[f][b][d][c] = false;
    //       }
    //     }
    //   }
    // }
    // bool A2mask[num_frequencies][num_beams][num_complex][num_dishes];
    // for (size_t f = 0; f < num_frequencies; ++f) {
    //   for (size_t b = 0; b < num_beams; ++b) {
    //     for (size_t c = 0; c < num_complex; ++c) {
    //       for (size_t d = 0; d < num_dishes; ++d) {
    //         A2mask[f][b][c][d] = false;
    //       }
    //     }
    //   }
    // }
    for (size_t f = 0; f < num_frequencies; ++f) {
      for (size_t b = 0; b < num_beams; ++b) {
        for (size_t c = 0; c < num_complex; ++c) {
          for (size_t d = 0; d < num_dishes; ++d) {
            const size_t d8 = (d >> 8) & 0b1;
            const size_t d67 = (d >> 6) & 0b11;
            const size_t d012345 = (d >> 0) & 0b111111;
            assert(((d8 << 8) | (d67 << 6) | (d012345 << 0)) == d);
            const size_t d_prime = (d8 << 8) | (d012345 << 2) | (d67 << 0);

            // assert(!Amask[f][b][d][c]);
            // Amask[f][b][d][c] = true;
            // assert(!A2mask[f][b][c][d_prime]);
            // A2mask[f][b][c][d_prime] = true;

            const ucomplex4 &Aelt = Aarray.at(Alinear(f, b, d, c) / 2);
            ucomplex4 &A2elt = A2array.at(A2linear(f, b, c, d_prime) / 2);
            const signed char arrval = c == 0 ? Aelt.real() : Aelt.imag();
            if (d_prime % 2 == 0) {
              A2elt = ucomplex4(A2elt.real(), arrval);
            } else {
              A2elt = ucomplex4(arrval, A2elt.imag());
            }
          }
        }
      }
    }
    // for (size_t f = 0; f < num_frequencies; ++f) {
    //   for (size_t b = 0; b < num_beams; ++b) {
    //     for (size_t d = 0; d < num_dishes; ++d) {
    //       for (size_t c = 0; c < num_complex; ++c) {
    //         assert(Amask[f][b][d][c]);
    //       }
    //     }
    //   }
    // }
    // for (size_t f = 0; f < num_frequencies; ++f) {
    //   for (size_t b = 0; b < num_beams; ++b) {
    //     for (size_t c = 0; c < num_complex; ++c) {
    //       for (size_t d = 0; d < num_dishes; ++d) {
    //         assert(A2mask[f][b][c][d]);
    //       }
    //     }
    //   }
    // }

    // for (size_t f = 0; f < num_frequencies; ++f) {
    //   for (size_t b = 0; b < num_beams; ++b) {
    //     for (size_t c = 0; c < num_complex; ++c) {
    //       for (size_t d = 0; d < num_dishes; d += 2) {
    //         const uint8_t val = *(const uint8_t *)&A2array.at(A2linear(f, b, c, d) / 2) ^ 0x88U;
    //         if (val != 0) {
    //           printf("f=%01d b=%02d c=%01d d'=%02d A2=0x%02x\n", int(f), int(b), int(c), int(d), unsigned(val));
    //         }
    //       }
    //     }
    //   }
    // }
  }

  constexpr size_t num_iters = 0; // benchmark iterations
  // constexpr size_t num_iters = 100; // benchmark iterations

  cout << "Forming beams...\n";
  ucomplex4 *Eptr = nullptr;
  cudaMalloc(&Eptr, Earray.size() * sizeof(ucomplex4));
  cudaMemcpy(Eptr, Earray.data(), Earray.size() * sizeof(ucomplex4), cudaMemcpyHostToDevice);
  ucomplex4 *A2ptr = nullptr;
  cudaMalloc(&A2ptr, A2array.size() * sizeof(ucomplex4));
  cudaMemcpy(A2ptr, A2array.data(), A2array.size() * sizeof(ucomplex4), cudaMemcpyHostToDevice);
  float *Gptr = nullptr;
  cudaMalloc(&Gptr, Garray.size() * sizeof(float));
  cudaMemcpy(Gptr, Garray.data(), Garray.size() * sizeof(float), cudaMemcpyHostToDevice);
  ucomplex4 *J2ptr = nullptr;
  cudaMalloc(&J2ptr, J2array.size() * sizeof(ucomplex4));
  vector<ucomplex4 *> J2ptrs(num_iters, nullptr);
  for (size_t iter = 0; iter < num_iters; ++iter) {
    cudaMalloc(&J2ptrs.at(iter), J2array.size() * sizeof(ucomplex4));
  }

  cudaError_t err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);

  const dim3 numBlocks(num_frequencies);
  const dim3 threadsPerBlock(num_threads, num_warps);
  form_beams<<<numBlocks, threadsPerBlock>>>(J2ptr, Eptr, A2ptr, Gptr);
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);

  cout << "Launching " << num_iters << " kernels...\n";
  const auto t0 = gettime();

  for (size_t iter = 0; iter < num_iters; ++iter) {
    form_beams<<<numBlocks, threadsPerBlock>>>(J2ptrs.at(iter), Eptr, A2ptr, Gptr);
  }
  err = cudaGetLastError();
  CHECK_RESULT(err);
  err = cudaDeviceSynchronize();
  CHECK_RESULT(err);

  const auto t1 = gettime();
  cout << "Elapsed time: " << ((t1 - t0) / num_iters) << " seconds per iteration (with " << num_iters << " iterations)\n";

  err = cudaGetLastError();
  CHECK_RESULT(err);

  cudaFree(Eptr);
  Eptr = nullptr;
  cudaFree(A2ptr);
  A2ptr = nullptr;
  cudaFree(Gptr);
  Gptr = nullptr;
  cudaMemcpy(J2array.data(), J2ptr, J2array.size() * sizeof(ucomplex4), cudaMemcpyDeviceToHost);
  cudaFree(J2ptr);
  J2ptr = nullptr;
  for (size_t iter = 0; iter < num_iters; ++iter) {
    cudaFree(J2ptrs.at(iter));
    J2ptrs.at(iter) = nullptr;
  }

  // // Change index order
  // vector<ucomplex4> Jarray(J2array.size());
  // for (size_t b = 0; b < nbeams; ++b) {
  //   for (size_t f = 0; f < nfrequencies; ++f) {
  //     for (size_t p = 0; p < npolarizations; ++p) {
  //       for (size_t t = 0; t < ntimes; ++t) {
  //         Jarray.at(Jlinear(b, f, p, t, 0) / 2) = J2array.at(J2linear(b, f, t, p, 0) / 2);
  //       }
  //     }
  //   }
  // }
  //
  // check(Jarray);

  bool allcorrect = true;
  for (size_t b = 0; b < num_beams; ++b) {
    for (size_t f = 0; f < num_frequencies; ++f) {
      for (size_t t = 0; t < num_times; ++t) {
        for (size_t p = 0; p < num_polarizations; ++p) {
          if (!(J2array.at(J2linear(b, f, t, p, 0) / 2) == J2array_expected.at(J2linear(b, f, t, p, 0) / 2))) {
            allcorrect = false;
            cout << "J2[b=" << b << ",f=" << f << ",t=" << t << ",p=" << p
                 << "]=" << complex<int>(J2array.at(J2linear(b, f, t, p, 0) / 2))
                 << "; J2_expected=" << complex<int>(J2array_expected.at(J2linear(b, f, t, p, 0) / 2)) << "\n";
          }
          // assert(J2array.at(J2linear(b, f, t, p, 0) / 2) == J2array_expected.at(J2linear(b, f, t, p, 0) / 2));
        }
      }
    }
  }
  if (!allcorrect) {
    cout << "ERROR FOUND\n";
    exit(1);
  }

  cout << "Done.\n";
  return 0;
}

// TODO:
// run on A40
// run benchmarks on full GPU (daisy-chain, multiple kernels, need different ouputs)
// test correctness; quantizing is different from standard implementation
// test fragment layout on A40
// KS has benchmark numbers
