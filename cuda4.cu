// -*-c++-*-
// Beamforming with CUDA

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
// output: A_register[complex][beam / A_register_beam_divisor][dish / A_register_dish_divisor]   [beam % A_register_beam_matrix_modulo][dish % A_register_dish_matrix_modulo]
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
// dish' = (dish warp) * (dish thread) * (dish explicit) * (dish explicit inner) * (dish explicit outer)
//
// input: E[time][frequency][dish][polarization][complex]
// output: E_shared[complex][time % E_shared_time_modulo][polarization][dish' + padding][complex]
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
// output: Ju_shared[dish' / Ju_shared_dish'_divisor][beam][time % Ju_shared_time_modulo][polarization][complex]
// Ju_shared_dish'_divisor = (# dish' warp)
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
