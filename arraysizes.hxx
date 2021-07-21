#ifndef ARRAYSIZES
#define ARRAYSIZES

#include "icomplex4.hxx"

#include <cassert>
#include <cstdint>
#include <vector>

using namespace std;

constexpr size_t ntimes = 32768;    // per chunk
constexpr size_t nfrequencies = 32; // per GPU
constexpr size_t ndishes = 512;
constexpr size_t npolarizations = 2;
constexpr size_t nbeams = 96;
constexpr size_t ncomplex = 2; // complex number components

// constexpr size_t ntimes = 32768;    // per chunk
// constexpr size_t nfrequencies = 32; // per GPU
// constexpr size_t ndishes = 512;
// constexpr size_t npolarizations = 2;
// constexpr size_t nbeams = 128;
// constexpr size_t ncomplex = 2; // complex number components

// constexpr size_t ntimes = 32;       // per chunk
// constexpr size_t nfrequencies = 32; // per GPU
// constexpr size_t ndishes = 512;
// constexpr size_t npolarizations = 2;
// constexpr size_t nbeams = 128;
// constexpr size_t ncomplex = 2; // complex number components

// constexpr size_t ntimes = 8;       // per chunk
// constexpr size_t nfrequencies = 1; // per GPU
// constexpr size_t ndishes = 32;
// constexpr size_t npolarizations = 2;
// constexpr size_t nbeams = 8;
// constexpr size_t ncomplex = 2; // complex number components

// constexpr size_t ntimes = 1;       // per chunk
// constexpr size_t nfrequencies = 1; // per GPU
// constexpr size_t ndishes = 1;
// constexpr size_t npolarizations = 2;
// constexpr size_t nbeams = 1;
// constexpr size_t ncomplex = 2; // complex number components

// Accessors handling memory layout

// E[time][frequency][dish][polarization][complex]
// A[frequency][beam][dish][complex]
// J[beam][frequency][polarization][time][complex]
// G[frequency][beam][complex]

constexpr size_t Esize = ntimes * nfrequencies * ndishes * npolarizations * ncomplex;
constexpr device_host size_t Elinear(size_t t, size_t f, size_t d, size_t p, size_t c) {
  assert(t < ntimes);
  assert(f < nfrequencies);
  assert(d < ndishes);
  assert(p < npolarizations);
  assert(c < ncomplex);
  const auto ind = c + ncomplex * (p + npolarizations * (d + ndishes * (f + nfrequencies * t)));
  assert(ind < Esize);
  return ind;
}

constexpr size_t Jsize = nbeams * nfrequencies * npolarizations * ntimes * ncomplex;
constexpr device_host size_t Jlinear(size_t b, size_t f, size_t p, size_t t, size_t c) {
  assert(b < nbeams);
  assert(f < nfrequencies);
  assert(p < npolarizations);
  assert(t < ntimes);
  assert(c < ncomplex);
  const auto ind = c + ncomplex * (t + ntimes * (p + npolarizations * (f + nfrequencies * b)));
  assert(ind < Jsize);
  return ind;
}

constexpr size_t Asize = nfrequencies * nbeams * ndishes * ncomplex;
constexpr device_host size_t Alinear(size_t f, size_t b, size_t d, size_t c) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(d < ndishes);
  assert(c < ncomplex);
  const auto ind = c + ncomplex * (d + ndishes * (b + nbeams * f));
  assert(ind < Asize);
  return ind;
}

constexpr size_t Gsize = nfrequencies * nbeams;
constexpr device_host size_t Glinear(size_t f, size_t b) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  const auto ind = b + nbeams * f;
  assert(ind < Gsize);
  return ind;
}

void setup(vector<ucomplex4> &Earray, vector<ucomplex4> &Aarray, vector<float> &Garray, vector<ucomplex4> &Jarray);
void check(const vector<ucomplex4> &Jarray);

#endif // #ifndef ARRAYSIZES
