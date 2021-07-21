#include "arraysizes.hxx"
#include "adler32.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>

void setup(vector<ucomplex4> &Earray, vector<ucomplex4> &Aarray, vector<float> &Garray, vector<ucomplex4> &Jarray) {
  cout << "Setting up input data...\n";

  Earray.resize(Esize / 2);
  Aarray.resize(Asize / 2);
  Garray.resize(Gsize);
  Jarray.resize(Jsize / 2);

  for (size_t n = 0; n < Earray.size(); ++n) {
    Earray[n] = ucomplex4(n % 15 - 7, (n + 1) % 15 - 7);
    // Earray[n] = ucomplex4(1, 0);
    // Earray[n] = ucomplex4(n % 2, 0);
    // Earray[n] = ucomplex4(0, n % 2);
    // Earray[n] = ucomplex4(n % 15 - 7, 0);
  }
  for (size_t n = 0; n < Aarray.size(); ++n) {
    Aarray[n] = ucomplex4(n % 15 - 7, (n + 1) % 15 - 7);
    /// Aarray[n] = ucomplex4(1, 1);
    // Aarray[n] = ucomplex4(1, 0);
    // Aarray[n] = ucomplex4(0, 1);
    // Aarray[n] = ucomplex4(n == 0, n == 1);
    // Aarray[n] = ucomplex4(n % 15 - 7, 0);
  }
  for (size_t n = 0; n < Garray.size(); ++n) {
    Garray[n] = float(n) / Garray.size() / ndishes * (15 + n % 15) / 30;
    // Garray[n] = float(1) / ndishes;
    // Garray[n] = float(1);
    // Garray[n] = n == 0;
  }

  // for (size_t t = 0; t < ntimes; ++t)
  //   for (size_t f = 0; f < nfrequencies; ++f)
  //     for (size_t d = 0; d < ndishes; ++d)
  //       for (size_t p = 0; p < npolarizations; ++p)
  //         Earray.at(Elinear(t, f, d, p, 0) / 2) = ucomplex4(0, 0);

  // for (size_t f = 0; f < nfrequencies; ++f)
  //   for (size_t b = 0; b < nbeams; ++b)
  //     for (size_t d = 0; d < ndishes; ++d)
  //       Aarray.at(Alinear(f, b, d, 0) / 2) = ucomplex4(0, 0);

  // for (size_t f = 0; f < nfrequencies; ++f)
  //   for (size_t b = 0; b < nbeams; ++b)
  //     Garray.at(Glinear(f, b)) = 1;
}

constexpr uint32_t correct_checksum = ntimes == 32768 && nbeams == 128  ? 0x4de6498f
                                      : ntimes == 32768 && nbeams == 96 ? 0xc4fb6a4c
                                      : ntimes == 32                    ? 0xab69dfee
                                      : ntimes == 8                     ? 0x3a3344bc
                                      : ntimes == 1                     ? 0x019a0111
                                                                        : 0;

void check(const vector<ucomplex4> &Jarray) {
  // for (size_t b = 0; b < nbeams; ++b) {
  //   for (size_t f = 0; f < nfrequencies; ++f) {
  //     for (size_t p = 0; p < npolarizations; ++p) {
  //       for (size_t t = 0; t < ntimes; ++t) {
  //         const auto J = Jarray.at(Jlinear(b, f, p, t, 0) / 2);
  //         cout << "J["
  //              << "b=" << b << ",f=" << f << ",p=" << p << ",t=" << t << "] =
  //              ("
  //              << int(J.real()) << "," << int(J.imag()) << ")\n";
  //       }
  //     }
  //   }
  // }

  cout << "Calculating checksum...\n";

  uint32_t checksum = adler32(reinterpret_cast<const unsigned char *>(Jarray.data()), Jarray.size());
  cout << "Checksum: 0x" << hex << setfill('0') << setw(8) << checksum << "\n";

  if (checksum != correct_checksum) {
    cout << "Expected: 0x" << hex << setfill('0') << setw(8) << correct_checksum << "\n";
    cout << "ERROR -- CHECKSUM MISMATCH\n";
    exit(1);
  }
  cout << "Checksum matches.\n";
}
