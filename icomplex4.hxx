#ifndef ICOMPLEX4
#define ICOMPLEX4

#include <cassert>
#include <cstdint>
#include <vector>

using namespace std;

// a 4-bit complex number
struct icomplex4 {
  // CHIME uses the following conventions:
  // - real part is stored in upper 4 bits, imaginary part in lower 4 bits
  // - each value is in the range -7 ... 7 (the value cannot be -8)
  // - each value x is stored as unsigned number as x + 8
  unsigned char data;
  constexpr icomplex4() : data(0) {}
  constexpr icomplex4(signed char real, signed char imag)
      : data(((real + 8) << 4) | (imag + 8)) {}
  constexpr signed char real() const { return (data >> 4) - 8; }
  constexpr signed char imag() const { return (data & 0x0f) - 8; }
  constexpr signed char operator[](int c) const {
    return c == 0 ? imag() : real();
  }
};

static_assert(icomplex4(1, 2).data == 0x9a);
static_assert(icomplex4(-1, 2).data == 0x7a);
static_assert(icomplex4(1, -2).data == 0x96);
static_assert(icomplex4(-1, -2).data == 0x76);
static_assert(icomplex4(1, 2).real() == 1);
static_assert(icomplex4(1, 2).imag() == 2);
static_assert(icomplex4(-1, 2).real() == -1);
static_assert(icomplex4(-1, 2).imag() == 2);
static_assert(icomplex4(1, -2).real() == 1);
static_assert(icomplex4(1, -2).imag() == -2);
static_assert(icomplex4(-1, -2).real() == -1);
static_assert(icomplex4(-1, -2).imag() == -2);

////////////////////////////////////////////////////////////////////////////////

constexpr size_t ntimes = 1;       // 32768;    // per chunk
constexpr size_t nfrequencies = 1; // 32; // per GPU
constexpr size_t ndishes = 1;      // 512;
constexpr size_t npolarizations = 2;
constexpr size_t nbeams = 1;   // 128;
constexpr size_t ncomplex = 2; // complex number components

// Accessors handling memory layout

constexpr size_t Esize =
    ntimes * nfrequencies * ndishes * npolarizations * ncomplex;
constexpr size_t Elinear(size_t t, size_t f, size_t d, size_t p, size_t c) {
  assert(t < ntimes);
  assert(f < nfrequencies);
  assert(d < ndishes);
  assert(p < npolarizations);
  assert(c < ncomplex);
  const auto ind =
      c +
      ncomplex * (p + npolarizations * (d + ndishes * (f + nfrequencies * t)));
  assert(ind < Esize);
  return ind;
}

constexpr size_t Jsize =
    nbeams * nfrequencies * npolarizations * ntimes * ncomplex;
constexpr size_t Jlinear(size_t b, size_t f, size_t p, size_t t, size_t c) {
  assert(b < nbeams);
  assert(f < nfrequencies);
  assert(p < npolarizations);
  assert(t < ntimes);
  assert(c < ncomplex);
  const auto ind =
      c +
      ncomplex * (t + ntimes * (p + npolarizations * (f + nfrequencies * b)));
  assert(ind < Jsize);
  return ind;
}

constexpr size_t Asize = nfrequencies * nbeams * ndishes * ncomplex;
constexpr size_t Alinear(size_t f, size_t b, size_t d, size_t c) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(d < ndishes);
  assert(c < ncomplex);
  const auto ind = c + ncomplex * (d + ndishes * (b + nbeams * f));
  assert(ind < Asize);
  return ind;
}

constexpr size_t Gsize = nfrequencies * nbeams;
constexpr size_t Glinear(size_t f, size_t b) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  const auto ind = b + nbeams * f;
  assert(ind < Gsize);
  return ind;
}

void setup(vector<icomplex4> &Earray, vector<icomplex4> &Aarray,
           vector<float> &Garray, vector<icomplex4> &Jarray);
void check(const vector<icomplex4> &Jarray);

#endif // #ifndef ICOMPLEX4
