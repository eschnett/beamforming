// Beamforming on the CPU

#include "arraysizes.hxx"
#include "icomplex4.hxx"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace std;

void form_beams(ucomplex4 *restrict const Jarray, const ucomplex4 *restrict const Earray, const ucomplex4 *restrict const Aarray,
                const float *restrict Garray) {

#pragma omp parallel for
  for (size_t b = 0; b < nbeams; ++b) {
    for (size_t f = 0; f < nfrequencies; ++f) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t t = 0; t < ntimes; ++t) {

          int rawJre = 0;
          int rawJim = 0;
          for (size_t d = 0; d < ndishes; ++d) {
            const ucomplex4 A = Aarray[Alinear(f, b, d, 0) / 2];
            signed char Are = A.real();
            signed char Aim = A.imag();
            const ucomplex4 E = Earray[Elinear(t, f, d, p, 0) / 2];
            signed char Ere = E.real();
            signed char Eim = E.imag();
            rawJre += Are * Ere - Aim * Eim;
            rawJim += Are * Eim + Aim * Ere;
          }

          const float G = Garray[Glinear(f, b)];
          int Jre = max(-7, min(7, int(lrint(G * float(rawJre)))));
          int Jim = max(-7, min(7, int(lrint(G * float(rawJim)))));
          ucomplex4 &Jitem = Jarray[Jlinear(b, f, p, t, 0) / 2];
          Jitem = ucomplex4(Jre, Jim);
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  cout << "beamforming.cpu\n";

  vector<ucomplex4> Earray;
  vector<ucomplex4> Aarray;
  vector<float> Garray;
  vector<ucomplex4> Jarray;
  setup(Earray, Aarray, Garray, Jarray);

  cout << "Forming beams...\n";
  const auto t0 = gettime();
  form_beams(Jarray.data(), Earray.data(), Aarray.data(), Garray.data());
  const auto t1 = gettime();
  cout << "Elapsed time: " << (t1 - t0) << " seconds\n";

  check(Jarray);

  cout << "Done.\n";
  return 0;
}
