#ifndef ICOMPLEX4
#define ICOMPLEX4

#ifdef __CUDACC__
#define device_host __device__ __host__
#else
#define device_host
#endif

#include <cassert>
#include <cstdint>
#include <vector>

using namespace std;

struct ucomplex4;
struct icomplex4;

// a 4-bit complex number with an offset encoding
struct ucomplex4 {
  // CHIME uses the following conventions:
  // - real part is stored in upper 4 bits, imaginary part in lower 4 bits
  // - each value is in the range -7 ... 7 (the value cannot be -8)
  // - each value x is stored as unsigned number as x + 8
  unsigned char data;
  constexpr device_host ucomplex4() : data(0) {}
  constexpr device_host ucomplex4(signed char real, signed char imag)
      : data((((unsigned char)real << 4) | (imag & 0x0f)) ^ 0x88) {}
  constexpr device_host signed char real() const {
    return (signed char)(data ^ 0x88) >> 4;
  }
  constexpr device_host signed char imag() const {
    return (signed char)((unsigned char)(data ^ 0x88) << 4) >> 4;
  }
  constexpr device_host ucomplex4 conj() const {
    return ucomplex4(real(), -imag());
  }
  constexpr device_host ucomplex4 swap() const {
    return ucomplex4(imag(), real());
  }
  constexpr device_host signed char operator[](int c) const {
    return c == 0 ? imag() : real();
  }
  constexpr device_host icomplex4 debias() const;
};

static_assert(ucomplex4(1, 2).data == 0x9a);
static_assert(ucomplex4(-1, 2).data == 0x7a);
static_assert(ucomplex4(1, -2).data == 0x96);
static_assert(ucomplex4(-1, -2).data == 0x76);
static_assert(ucomplex4(1, 2).real() == 1);
static_assert(ucomplex4(1, 2).imag() == 2);
static_assert(ucomplex4(-1, 2).real() == -1);
static_assert(ucomplex4(-1, 2).imag() == 2);
static_assert(ucomplex4(1, -2).real() == 1);
static_assert(ucomplex4(1, -2).imag() == -2);
static_assert(ucomplex4(-1, -2).real() == -1);
static_assert(ucomplex4(-1, -2).imag() == -2);

// a 4-bit complex number
struct icomplex4 {
  unsigned char data;
  constexpr device_host icomplex4() : data(0) {}
  constexpr device_host icomplex4(signed char real, signed char imag)
      : data((real << 4) | (imag & 0x0f)) {}
  constexpr device_host signed char real() const {
    return (signed char)data >> 4;
  }
  constexpr device_host signed char imag() const {
    return (signed char)(data << 4) >> 4;
  }
  constexpr device_host icomplex4 conj() const {
    return icomplex4(real(), -imag());
  }
  constexpr device_host icomplex4 swap() const {
    return icomplex4(imag(), real());
  }
  constexpr device_host signed char operator[](int c) const {
    return c == 0 ? imag() : real();
  }
  constexpr device_host ucomplex4 bias() const;
};

constexpr device_host icomplex4 ucomplex4::debias() const {
  icomplex4 r;
  r.data = data ^ 0x88;
  return r;
}

constexpr device_host ucomplex4 icomplex4::bias() const {
  ucomplex4 r;
  r.data = data ^ 0x88;
  return r;
}

////////////////////////////////////////////////////////////////////////////////

constexpr size_t ntimes = 32768;    // per chunk
constexpr size_t nfrequencies = 32; // per GPU
constexpr size_t ndishes = 512;
constexpr size_t npolarizations = 2;
constexpr size_t nbeams = 128;
constexpr size_t ncomplex = 2; // complex number components

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

constexpr size_t Esize =
    ntimes * nfrequencies * ndishes * npolarizations * ncomplex;
constexpr device_host size_t Elinear(size_t t, size_t f, size_t d, size_t p,
                                     size_t c) {
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
constexpr device_host size_t Jlinear(size_t b, size_t f, size_t p, size_t t,
                                     size_t c) {
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

double gettime();

void setup(vector<ucomplex4> &Earray, vector<ucomplex4> &Aarray,
           vector<float> &Garray, vector<ucomplex4> &Jarray);
void check(const vector<ucomplex4> &Jarray);

#endif // #ifndef ICOMPLEX4
