#ifndef ICOMPLEX4
#define ICOMPLEX4

#ifdef __CUDACC__
#define device_host __device__ __host__
#else
#define device_host
#endif

#include <sys/time.h>

#include <cassert>
#include <cstdint>

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
  ucomplex4() = default;
  constexpr device_host bool operator==(const ucomplex4 &other) const { return data == other.data; }
  constexpr device_host bool operator!=(const ucomplex4 &other) const { return !(*this == other); }
  // constexpr device_host ucomplex4() : data(0) {}
  constexpr device_host ucomplex4(signed char real, signed char imag) : data((((unsigned char)real << 4) | (imag & 0x0f)) ^ 0x88) {}
  constexpr device_host signed char real() const { return (signed char)(data ^ 0x88) >> 4; }
  constexpr device_host signed char imag() const { return (signed char)((unsigned char)(data ^ 0x88) << 4) >> 4; }
  constexpr device_host ucomplex4 conj() const { return ucomplex4(real(), -imag()); }
  constexpr device_host ucomplex4 swap() const { return ucomplex4(imag(), real()); }
  constexpr device_host signed char operator[](int c) const { return c == 0 ? imag() : real(); }
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
  icomplex4() = default;
  // constexpr device_host icomplex4() : data(0) {}
  constexpr device_host icomplex4(signed char real, signed char imag) : data((real << 4) | (imag & 0x0f)) {}
  constexpr device_host signed char real() const { return (signed char)data >> 4; }
  constexpr device_host signed char imag() const { return (signed char)(data << 4) >> 4; }
  constexpr device_host icomplex4 conj() const { return icomplex4(real(), -imag()); }
  constexpr device_host icomplex4 swap() const { return icomplex4(imag(), real()); }
  constexpr device_host signed char operator[](int c) const { return c == 0 ? imag() : real(); }
  constexpr device_host ucomplex4 bias() const;
};

constexpr device_host icomplex4 ucomplex4::debias() const {
  icomplex4 r(0, 0);
  r.data = data ^ 0x88;
  return r;
}

constexpr device_host ucomplex4 icomplex4::bias() const {
  ucomplex4 r(0, 0);
  r.data = data ^ 0x88;
  return r;
}

inline double gettime() {
  timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec / 1.0e+6;
}

#endif // #ifndef ICOMPLEX4
