// -*-c++-*-
// Beamforming with CUDA

#include "arraysizes.hxx"
#include "icomplex4.hxx"

#include <mma.h>

#include <iostream>

using namespace std;

using namespace nvcuda;
using namespace nvcuda::wmma;

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
// dish = (dish iteration) * (dish warp) * (dish k matrix element)
// complex = (explicit)
//
// input: A[frequency][beam][dish][complex]
// output: A_register[complex][beam / A_register_beam_divisor][dish / A_register_dish_divisor]
//         [beam % A_register_beam_matrix_modulo][dish % A_register_dish_matrix_modulo]
// A_register_beam_divisor = (# beam warp) * (# beam m matrix element)
// A_register_dish_divisor = (# dish warp) * (# dish k matrix element)
// A_register_beam_matrix_modulo = (# beam m matrix element)
// A_register_dish_matrix_modulo = (# dish k matrix element)
//
// TODO: Use dish' instead of dish

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
// dish' = (dish' warp) * (dish' iteration) * (dish' matrix element)
// polarization = (polarization n matrix element)
// complex = explicit
//
// input: A_register[complex][beam][dish]   [beam][dish]
// input: E_shared[complex][time][polarization][dish' + padding][complex]
// output: Ju_shared[dish' / Ju_shared_dish'_divisor][beam][time / Ju_shared_time_divisor %  Ju_shared_time_modulo]
//                  [polarization][complex]
// u_shared_dish'_divisor = (# dish' warp)
// Ju_shared_time_divisor = (# time n matrix elements)
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
// output: J[beam][frequency][polarization][time][complex]

////////////////////////////////////////////////////////////////////////////////

// Helper functions

__device__ int32_t clamp(int32_t i, int32_t i0, int32_t i1) { return max(i0, min(i1, i)); }

__device__ int32_t extract_real(const int32_t x0, const int32_t x1) {
  return ((uint32_t)(x0 & 0xf0f0f0f0U) >> 4) | (x1 & 0xf0f0f0f0U);
}

__device__ int32_t extract_imag(const int32_t x0, const int32_t x1) {
  return (x0 & 0x0f0f0f0fU) | ((uint32_t)(x1 & 0x0f0f0f0fU) << 4);
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

namespace load_A {

constexpr size_t num_beam_iterations = 3;
constexpr size_t num_dish_iterations = 2;
constexpr size_t num_beam_warps = 4;
constexpr size_t num_dish_warps = 8;
constexpr size_t num_dish_k_elements = 32;
constexpr size_t num_beam_m_elements = 8;
constexpr size_t num_complex_explicit = 2;

static_assert(num_beam_warps * num_dish_warps == num_warps);
static_assert(num_beam_iterations * num_beam_warps * num_beam_m_elements == num_beams);
static_assert(num_dish_iterations * num_dish_warps * num_dish_k_elements == num_dishes);

constexpr size_t A_register_beam_divisor = num_beam_warps * num_beam_m_elements;
constexpr size_t A_register_dish_divisor = num_dish_warps * num_dish_k_elements;
constexpr size_t A_register_beam_matrix_modulo = num_beam_m_elements;
constexpr size_t A_register_dish_matrix_modulo = num_dish_k_elements;

using A_register_t = fragment<wmma::matrix_a, num_m_elements, num_n_elements, num_k_elements, experimental::precision::s4,
                              row_major>[ncomplex][nbeams / A_register_beam_divisor][ndishes / A_register_dish_divisor];

__device__ void load_A(A_register_t &restrict A_register, const ucomplex4 *restrict const A_array, const size_t frequency) {
  const size_t beam_warp = threadIdx.y / num_dish_warps;
  const size_t dish_warp = threadIdx.y % num_dish_warps;
  for (size_t beam_iteration = 0; beam_iteration < num_beam_iterations; ++beam_iteration) {
    for (size_t dish_iteration = 0; dish_iteration < num_dish_iterations; ++dish_iteration) {
      const size_t beam = (beam_iteration * num_beam_warps + beam_warp) * num_beam_m_elements;
      const size_t dish = (dish_iteration * num_dish_warps + dish_warp) * num_dish_k_elements;
      assert(beam < nbeams);
      assert(dish < ndishes);

      // Note: This is the wrong ordering for A; need to shuffle dish
      // indices the same way as for E

      fragment<wmma::matrix_a, num_m_elements, num_n_elements, num_k_elements, experimental::precision::s4, row_major> A0[ncomplex];
      for (size_t c = 0; c < ncomplex; ++c) {
        // TOOD: Use __ldcs
        // Load 2 consecutive sets of elements of A
        load_matrix_sync(A0[c], &A_array[Alinear(frequency, beam, dish + c * num_dish_k_elements / 2, 0) / 2],
                         Alinear(0, 0, 1, 0) / 2);
      }

      assert(beam / A_register_beam_divisor == beam_iteration);
      assert(dish / A_register_dish_divisor == dish_iteration);
      static_assert(A_register[0][beam_iteration][dish_iteration].num_storage_elements == 1);
      for (int i = 0; i < A_register[0][beam_iteration][dish_iteration].num_storage_elements; ++i) {
        // Extract complex components and remove bias
        A_register[0][beam_iteration][dish_iteration].x[i] = extract_real(A0[0].x[i], A0[1].x[i]) ^ 0x88888888U;
        A_register[1][beam_iteration][dish_iteration].x[i] = extract_imag(A0[0].x[i], A0[1].x[i]) ^ 0x88888888U;
      }
    }
  }
}

} // namespace load_A

using load_A::num_beam_m_elements;
using load_A::num_dish_k_elements;

using load_A::A_register_beam_divisor;
using load_A::A_register_beam_matrix_modulo;
using load_A::A_register_dish_divisor;
using load_A::A_register_dish_matrix_modulo;

using load_A::A_register_t;

////////////////////////////////////////////////////////////////////////////////

namespace shuffle_E {

constexpr size_t num_time_iterations_outer = 512;
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
      // TOOD: Use __ldcs
      // TOOD: Use __stcs for J array
      E0[p][c] = *(const uint32_t *)&E_array[Elinear(time, frequency, dish, 0, 0) / 2];
    }
  }

  // First we split out the complex components and remove the bias
  uint32_t E1[num_polarizations][num_complex];
  for (size_t p = 0; p < num_polarizations; ++p) {
    E1[p][0] = extract_real(E0[p][0], E0[p][1]) ^ 0x88888888U;
    E1[p][1] = extract_imag(E0[p][0], E0[p][1]) ^ 0x88888888U;
  }

  // Next we separate the polarizations
  uint32_t E2[num_polarizations][num_complex];
  for (size_t c = 0; c < num_complex; ++c) {
    E2[0][c] = __byte_perm(E1[0][c], E1[1][c], 0x6420);
    E2[1][c] = __byte_perm(E1[0][c], E1[1][c], 0x7531);
  }

  // Store into shared memory
  for (size_t c = 0; c < num_complex; ++c) {
    for (size_t p = 0; p < num_polarizations; ++p) {
      const size_t dish_prime = dish0_prime;
      assert(dish_prime < num_dishes_prime);
      E_shared[c][time % shuffle_E::E_shared_time_modulo][p][dish_prime / shuffle_E::E_shared_dish_prime_divisor] = E2[p][c];
    }
  }
}

} // namespace shuffle_E

using shuffle_E::num_time_iterations_inner;
using shuffle_E::num_time_iterations_outer;

using shuffle_E::E_shared_dish_prime_divisor;
using shuffle_E::E_shared_padding;
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
constexpr size_t Ju_shared_time_divisor = num_time_n_elements;
constexpr size_t Ju_shared_time_modulo = num_time_iterations_inner2;
using Ju_shared_t =
    uint32_t[num_dishes_prime / Ju_shared_dish_prime_divisor][num_beams][Ju_shared_time_modulo]; // [polarization][complex]

__device__ void compute_Ju(Ju_shared_t &restrict Ju_shared, const A_register_t &restrict A_register,
                           const E_shared_t &restrict E_shared, const float *restrict const G_array, const size_t frequency,
                           const size_t time_iteration_outer, const size_t time_iteration_inner,
                           const size_t time_iteration_inner2) {
  const size_t beam_warp = threadIdx.y / num_dish_prime_warps;
  const size_t dish_prime_warp = threadIdx.y % num_dish_prime_warps;

  // Load E-field from shared memory
  // wmma::B[k][n]   (must be row major)
  fragment<wmma::matrix_b, num_m_elements, num_n_elements, num_k_elements, experimental::precision::s4, col_major>
      E[num_complex][num_dish_prime_iterations];

  for (size_t c = 0; c < num_complex; ++c) {
    for (size_t dish_prime_iteration = 0; dish_prime_iteration < num_dish_prime_iterations; ++dish_prime_iteration) {
      const size_t time0 = ((time_iteration_outer * num_time_iterations_inner + time_iteration_inner) * num_time_iterations_inner2 +
                            time_iteration_inner2) *
                           num_time_n_elements;
      const size_t dish_prime0 = (dish_prime_warp * num_dish_prime_iterations + dish_prime_iteration) * num_dish_prime_k_elements;
      load_matrix_sync(
          E[c][dish_prime_iteration],
          &E_shared[c][time0 / Ju_shared_time_divisor % Ju_shared_time_modulo][0][dish_prime0 / E_shared_dish_prime_divisor],
          (&E_shared[0][0][1][0] - &E_shared[0][0][0][0]) * num_reals_int);
    }
  }

  for (size_t beam_iteration = 0; beam_iteration < num_beam_iterations; ++beam_iteration) {
    const size_t time0 = ((time_iteration_outer * num_time_iterations_inner + time_iteration_inner) * num_time_iterations_inner2 +
                          time_iteration_inner2) *
                         num_time_n_elements;
    const size_t beam0 = (beam_iteration * num_beam_warps + beam_warp) * num_beam_m_elements;
    const size_t dish_prime0 = dish_prime_warp * num_dish_prime_iterations * num_dish_k_elements;

    fragment<wmma::accumulator, num_m_elements, num_n_elements, num_k_elements, int32_t> JurePos, JureNeg, JuimPos;

    // Initialize Ju
    fill_fragment(JurePos, 0);
    fill_fragment(JureNeg, 0);
    fill_fragment(JuimPos, 0);

    // Multiply
    for (size_t dish_prime_iteration = 0; dish_prime_iteration < num_dish_prime_iterations; ++dish_prime_iteration) {
      mma_sync(JurePos, A_register[0][beam_iteration][dish_prime_iteration], E[0][dish_prime_iteration], JurePos);
      mma_sync(JureNeg, A_register[1][beam_iteration][dish_prime_iteration], E[1][dish_prime_iteration], JureNeg);
      mma_sync(JuimPos, A_register[0][beam_iteration][dish_prime_iteration], E[1][dish_prime_iteration], JuimPos);
      mma_sync(JuimPos, A_register[1][beam_iteration][dish_prime_iteration], E[0][dish_prime_iteration], JuimPos);
    }

    // Extract result from Ju matrix
    int8_t Ju8[num_polarizations][num_complex];
    static_assert(JurePos.num_storage_elements == npolarizations);
    for (size_t i = 0; i < JurePos.num_storage_elements; ++i) {
      const size_t element = threadIdx.x * JurePos.num_storage_elements + i;
      // const size_t time = time0 + element / num_m_elements;
      const size_t beam = beam0 + element % num_m_elements / num_polarizations;
      const size_t p = element % num_m_elements % num_polarizations;
      // Combine positive and negative J values, and reduce from 32 to 16 bits
      int32_t Ju[num_complex];
      Ju[0] = JurePos.x[i] - JureNeg.x[i];
      Ju[1] = JuimPos.x[i];
      for (size_t c = 0; c < num_complex; ++c) {
        assert(uintptr_t(&G_array[Glinear(frequency, beam)]) % sizeof(float) == 0);
        const float G = G_array[Glinear(frequency, beam)];
        Ju8[p][c] = clamp(int32_t(lrintf(G * float(Ju[c]))), -127, 127);
      }
    }
    // CUDA is little endian
    // TODO: Use make_char4
    // TODO: Use char4/uchar4 instead of uint32_t?
    const uint32_t Ju8all =
        ((uint32_t)Ju8[0][0]) | ((uint32_t)Ju8[0][1] << 8) | ((uint32_t)Ju8[1][0] << 16) | ((uint32_t)Ju8[1][1] << 24);

    const size_t element0 = threadIdx.x * JurePos.num_storage_elements;
    const size_t beam = beam0 + element0 % num_m_elements / num_polarizations;
    assert(dish_prime0 / Ju_shared_dish_prime_divisor == dish_prime_warp);
    Ju_shared[dish_prime_warp][beam][time0 / Ju_shared_time_divisor % Ju_shared_time_modulo] = Ju8all;
  }
}
} // namespace compute_Ju

using compute_Ju::num_time_iterations_inner2;

using compute_Ju::Ju_shared_dish_prime_divisor;
using compute_Ju::Ju_shared_time_divisor;
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
      for (size_t p = 0; p < num_polarizations; ++p) {
        for (size_t c = 0; c < num_complex; ++c) {
          uint32_t Ju = Ju_shared[dish_prime_iteration][beam][time / Ju_shared_time_divisor % Ju_shared_time_modulo];
          J[p][c] += int8_t((Ju >> (8 * (p * num_complex + c))) & 0xffU);
        }
      }
    }
    // Convert to 4 bits and add bias
    uint8_t J4[2];
    for (size_t p = 0; p < num_polarizations; ++p) {
      J4[p] = (uint32_t(clamp(J[p][0], -7, 7)) << 4) | uint32_t(clamp(J[p][1], -7, 7));
    }
    // Combine polarizations and add bias
    // TODO: Use make_uchar2?
    const uint16_t J4all = (uint32_t(J4[0]) | (uint32_t(J4[1]) << 8)) ^ 0x8888U;
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
constexpr size_t num_time_threads = 16;
constexpr size_t num_polarization_threads = 2;
constexpr size_t num_time_explicit = 4;
constexpr size_t num_complex_explicit = 2;

static_assert(num_beam_warps == num_warps);
static_assert(num_time_threads * num_polarization_threads == num_threads);
static_assert(num_time_explicit * num_complex_explicit == num_reals_int);
static_assert(num_time_iterations_outer * num_time_threads * num_time_explicit == num_times);
static_assert(num_beam_iterations * num_beam_warps == num_beams);

__device__ void transpose_J(ucomplex4 *restrict const J_array, const J_shared_t &restrict J_shared, const size_t frequency,
                            const size_t time_iteration_outer) {
  const size_t beam_warp = threadIdx.y;
  const size_t time_thread = threadIdx.x / num_polarization_threads;
  const size_t polarization_thread = threadIdx.x % num_polarization_threads;
  const size_t time0 = (time_iteration_outer * num_time_threads + time_thread) * num_time_explicit;
  const size_t polarization = polarization_thread;
  for (size_t beam_iteration = 0; beam_iteration < num_beam_iterations; ++beam_iteration) {
    const size_t beam = beam_iteration * num_beam_warps + beam_warp;
    // Load data
    // (We load twice as much as we need from shared memory)
    // (We could avoid bank conflicts here by interchanging shared memory reads on every other thread)
    uint32_t Jall0[2];
    Jall0[0] = *(const uint32_t *)&J_shared[beam][(time0 + 0 * num_time_explicit / 2) % J_shared_time_modulo];
    Jall0[1] = *(const uint32_t *)&J_shared[beam][(time0 + 1 * num_time_explicit / 2) % J_shared_time_modulo];
    // Extract polarization
    const uint32_t Jall1 = __byte_perm(Jall0[0], Jall0[1], polarization == 0 ? 0x6420 : 0x7531);
    // Write to global memory
    *(uint32_t *)&J_array[Jlinear(beam, frequency, polarization, time0, 0) / 2] = Jall1;
  }
}
} // namespace transpose_J

////////////////////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(num_threads *num_warps, 1)
    form_beams(ucomplex4 *restrict const J_array, const ucomplex4 *restrict const E_array, const ucomplex4 *restrict const A_array,
               const float *restrict const G_array) {

  // Each frequency is transformed independently. We use one thread block per frequency.

  const size_t frequency = blockIdx.x;

  // Load A into registers
  load_A::A_register_t A_register;
  load_A::load_A(A_register, A_array, frequency);

  for (size_t time_iteration_outer = 0; time_iteration_outer < num_time_iterations_outer; ++time_iteration_outer) {
    __shared__ E_shared_t E_shared;
    __shared__ Ju_shared_t Ju_shared;
    __shared__ J_shared_t J_shared;

    for (size_t time_iteration_inner = 0; time_iteration_inner < num_time_iterations_inner; ++time_iteration_inner) {
      shuffle_E::shuffle_E(E_shared, E_array, frequency, time_iteration_outer, time_iteration_inner);
      __syncthreads();

      for (size_t time_iteration_inner2 = 0; time_iteration_inner2 < num_time_iterations_inner2; ++time_iteration_inner2) {
        compute_Ju::compute_Ju(Ju_shared, A_register, E_shared, G_array, frequency, time_iteration_outer, time_iteration_inner,
                               time_iteration_inner2);
        __syncthreads();
        reduce_to_J::reduce_to_J(J_shared, Ju_shared, time_iteration_outer, time_iteration_inner, time_iteration_inner2);
      }
    }
    __syncthreads();
    transpose_J::transpose_J(J_array, J_shared, frequency, time_iteration_outer);
  }
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

int main(int argc, char **argv) {
  cout << "beamforming.cuda4\n";

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

  const dim3 numBlocks(num_frequencies);
  const dim3 threadsPerBlock(num_threads, num_warps);
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
