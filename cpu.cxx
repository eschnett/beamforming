// Beamforming on the CPU

// Build with
// g++ -std=c++17 -Ofast -march=native cpu.cxx -o cpu

#define restrict __restrict__

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

using namespace std;

const size_t ntimes = 32768;    // per chunk
const size_t nfrequencies = 32; // per GPU
const size_t ndishes = 512;
const size_t npolarizations = 2;
const size_t nbeams = 128;

// a 4-bit complex number
struct icomplex4 {
  unsigned char data;
  constexpr icomplex4() : data(0) {}
  constexpr icomplex4(signed char real, signed char imag)
      : data((real + 8) | ((imag + 8) << 4)) {}
  constexpr signed char real() const { return (data & 0x0f) - 8; }
  constexpr signed char imag() const { return (data >> 4) - 8; }
};

static_assert(icomplex4(1, 2).data == 0xa9);
static_assert(icomplex4(-1, 2).data == 0xa7);
static_assert(icomplex4(1, -2).data == 0x69);
static_assert(icomplex4(-1, -2).data == 0x67);
static_assert(icomplex4(1, 2).real() == 1);
static_assert(icomplex4(1, 2).imag() == 2);
static_assert(icomplex4(-1, 2).real() == -1);
static_assert(icomplex4(-1, 2).imag() == 2);
static_assert(icomplex4(1, -2).real() == 1);
static_assert(icomplex4(1, -2).imag() == -2);
static_assert(icomplex4(-1, -2).real() == -1);
static_assert(icomplex4(-1, -2).imag() == -2);

void form_beams(icomplex4 *restrict const Jarray,
                const icomplex4 *restrict const Earray,
                const icomplex4 *restrict const Aarray,
                const float *restrict Garray) {
  // Accessors handling memory layout
  const auto Eelem = [&](size_t t, size_t f, size_t d, size_t p) {
    assert(t >= 0 && t < ntimes);
    assert(f >= 0 && f < nfrequencies);
    assert(d >= 0 && d < ndishes);
    assert(p >= 0 && p < npolarizations);
    return Earray[p + npolarizations * (d + ndishes * (f + nfrequencies * t))];
  };
  const auto Jelem = [&](size_t b, size_t f, size_t p,
                         size_t t) -> icomplex4 & {
    assert(b >= 0 && b < nbeams);
    assert(f >= 0 && f < nfrequencies);
    assert(p >= 0 && p < npolarizations);
    assert(t >= 0 && t < ntimes);
    return Jarray[t + ntimes * (p + npolarizations * (f + nfrequencies * b))];
  };
  const auto Aelem = [&](size_t f, size_t b, size_t d) {
    assert(f >= 0 && f < nfrequencies);
    assert(b >= 0 && b < nbeams);
    assert(d >= 0 && d < ndishes);
    return Aarray[d + ndishes * (b + nbeams * f)];
  };
  const auto Gelem = [&](size_t f, size_t b) {
    assert(f >= 0 && f < nfrequencies);
    assert(b >= 0 && b < nbeams);
    return Garray[b + nbeams * f];
  };

  for (size_t b = 0; b < nbeams; ++b) {
    for (size_t f = 0; f < nfrequencies; ++f) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t t = 0; t < ntimes; ++t) {

          int rawJre = 0;
          int rawJim = 0;
          for (int d = 0; d < ndishes; ++d) {
            const icomplex4 A = Aelem(f, b, d);
            signed char Are = A.real();
            signed char Aim = A.imag();
            const icomplex4 E = Eelem(t, f, d, p);
            signed char Ere = E.real();
            signed char Eim = E.imag();
            rawJre += Are * Ere - Aim * Eim;
            rawJim += Are * Eim + Aim * Ere;
          }

          const float G = Gelem(f, b);
          int Jre = max(-7, min(7, int(lrint(G * float(rawJre)))));
          int Jim = max(-7, min(7, int(lrint(G * float(rawJim)))));
          icomplex4 &Jitem = Jelem(b, f, p, t);
          Jitem = icomplex4(Jre, Jim);
        }
      }
    }
  }
}

int main(int argc, char **argv) { return 0; }
