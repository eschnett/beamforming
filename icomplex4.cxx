#include "icomplex4.hxx"
#include "adler32.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>

void setup(vector<icomplex4> &Earray, vector<icomplex4> &Aarray,
           vector<float> &Garray, vector<icomplex4> &Jarray) {
  cout << "Setting up input data...\n";

  Earray.resize(Esize / 2);
  Aarray.resize(Asize / 2);
  Garray.resize(Gsize);
  Jarray.resize(Jsize / 2);

  for (size_t n = 0; n < Earray.size(); ++n)
    Earray[n] = icomplex4(n % 15 - 7, (n + 1) % 15 - 7);
  for (size_t n = 0; n < Aarray.size(); ++n)
    Aarray[n] = icomplex4(n % 15 - 7, (n + 1) % 15 - 7);
  for (size_t n = 0; n < Garray.size(); ++n)
    Garray[n] = (float(n) / ndishes) * (15 + n % 15) / 30;
}

constexpr uint32_t correct_checksum = ntimes == 32768 ? 0xdeba4178
                                      : ntimes == 32  ? 0x4acd2311
                                                      : 0;

void check(const vector<icomplex4> &Jarray) {
  cout << "Calculating checksum...\n";

  uint32_t checksum = adler32(
      reinterpret_cast<const unsigned char *>(Jarray.data()), Jarray.size());
  cout << "Checksum: 0x" << hex << setfill('0') << setw(8) << checksum << "\n";

  if (checksum != correct_checksum) {
    cout << "Expected: 0x" << hex << setfill('0') << setw(8) << correct_checksum
         << "\n";
    cout << "ERROR -- CHECKSUM MISMATCH\n";
    exit(1);
  }
  cout << "Checksum matches.\n";
}
