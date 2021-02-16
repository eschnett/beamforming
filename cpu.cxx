// Beamforming on the CPU

#include "adler32.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

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
      : data((real + 8) << 4 | ((imag + 8))) {}
  constexpr signed char real() const { return (data >> 4) - 8; }
  constexpr signed char imag() const { return (data & 0x0f) - 8; }
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

#pragma omp parallel for
  for (size_t b = 0; b < nbeams; ++b) {
    for (size_t f = 0; f < nfrequencies; ++f) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t t = 0; t < ntimes; ++t) {

          int rawJre = 0;
          int rawJim = 0;
          for (size_t d = 0; d < ndishes; ++d) {
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

int main(int argc, char **argv) {
  cout << "beamforming.cpu\n";
  cout << "Setting up input data...\n";
  vector<icomplex4> Earray(ntimes * nfrequencies * ndishes * npolarizations);
  vector<icomplex4> Aarray(nfrequencies * nbeams * ndishes);
  vector<float> Garray(nfrequencies * nbeams);
  for (size_t n = 0; n < Earray.size(); ++n)
    Earray[n] = icomplex4(n % 15 - 7, (n + 1) % 15 - 7);
  for (size_t n = 0; n < Aarray.size(); ++n)
    Aarray[n] = icomplex4(n % 15 - 7, (n + 1) % 15 - 7);
  for (size_t n = 0; n < Garray.size(); ++n)
    Garray[n] = (n / ndishes) * (15 + n % 15) / 30;
  vector<icomplex4> Jarray(nbeams * nfrequencies * npolarizations * ntimes);
  cout << "Forming beams...\n";
  form_beams(Jarray.data(), Earray.data(), Aarray.data(), Garray.data());
  cout << "Calculating checksum...\n";
  uint32_t checksum =
      adler32(reinterpret_cast<unsigned char *>(Jarray.data()), Jarray.size());
  cout << "Checksum: 0x" << hex << setfill('0') << setw(8) << checksum << "\n";
  assert(checksum == 0x59b6a388);
  cout << "Done.\n";
  return 0;
}
