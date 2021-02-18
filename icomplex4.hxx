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
  // CUDA might use the opposite indexing order
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

const size_t ntimes = 32768;    // per chunk
const size_t nfrequencies = 32; // per GPU
const size_t ndishes = 512;
const size_t npolarizations = 2;
const size_t nbeams = 128;

// Accessors handling memory layout
constexpr size_t Eindex(size_t t, size_t f, size_t d, size_t p) {
  assert(t < ntimes);
  assert(f < nfrequencies);
  assert(d < ndishes);
  assert(p < npolarizations);
  return p + npolarizations * (d + ndishes * (f + nfrequencies * t));
}
constexpr size_t Jindex(size_t b, size_t f, size_t p, size_t t) {
  assert(b < nbeams);
  assert(f < nfrequencies);
  assert(p < npolarizations);
  assert(t < ntimes);
  return t + ntimes * (p + npolarizations * (f + nfrequencies * b));
}
constexpr size_t Aindex(size_t f, size_t b, size_t d) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(d < ndishes);
  return d + ndishes * (b + nbeams * f);
}
constexpr size_t Gindex(size_t f, size_t b) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  return b + nbeams * f;
}

void setup(vector<icomplex4> &Earray, vector<icomplex4> &Aarray,
           vector<float> &Garray, vector<icomplex4> &Jarray);
void check(const vector<icomplex4> &Jarray);

#endif // #ifndef ICOMPLEX4
