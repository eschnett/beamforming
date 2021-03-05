// -*-c++-*-
// Beamforming with CUDA's memory layout

#include "adler32.h"
#include "icomplex4.hxx"

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

// Accessors handling memory layout

// E[time][frequency][dish][polarization][complex]
// A[frequency][beam][polarization][complex][dish][polarization][complex]
// J[time][frequency][beam][polarization][complex]
// G[frequency][beam][polarization][complex]

constexpr size_t Esize1 = Esize;
constexpr size_t Elinear1(size_t t, size_t f, size_t d, size_t p, size_t c) {
  return Elinear(t, f, d, p, c);
}

constexpr size_t Jsize1 =
    ntimes * nfrequencies * nbeams * npolarizations * ncomplex;
constexpr size_t Jlinear1(size_t t, size_t f, size_t b, size_t p, size_t c) {
  assert(t < ntimes);
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(p < npolarizations);
  assert(c < ncomplex);
  const auto ind =
      c +
      ncomplex * (p + npolarizations * (b + nbeams * (f + nfrequencies * t)));
  assert(ind < Jsize1);
  return ind;
}

constexpr size_t Asize1 = nfrequencies * nbeams * npolarizations * ncomplex *
                          ndishes * npolarizations * ncomplex;
constexpr size_t Alinear1(size_t f, size_t b, size_t p1, size_t c1, size_t d,
                          size_t p2, size_t c2) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(p1 < npolarizations);
  assert(c1 < ncomplex);
  assert(d < ndishes);
  assert(p2 < npolarizations);
  assert(c2 < ncomplex);
  const auto ind =
      c2 +
      ncomplex *
          (p2 +
           npolarizations *
               (d + ndishes * (c1 + ncomplex * (p1 + npolarizations *
                                                         (b + nbeams * f)))));
  assert(ind < Asize1);
  return ind;
}

constexpr size_t Gsize1 = nfrequencies * nbeams * npolarizations * ncomplex;
constexpr size_t Glinear1(size_t f, size_t b, size_t p, size_t c) {
  assert(f < nfrequencies);
  assert(b < nbeams);
  assert(p < npolarizations);
  assert(c < ncomplex);
  const auto ind = c + ncomplex * (p + npolarizations * (b + nbeams * f));
  assert(ind < Gsize1);
  return ind;
}

static_assert(Jsize <= UINT_MAX);
static_assert(Esize <= UINT_MAX);
static_assert(Asize <= UINT_MAX);
static_assert(Gsize <= UINT_MAX);

// Reshape arrays

vector<icomplex4> prepare_A(const vector<icomplex4> &Aarray) {
  for (size_t f = 0; f < nfrequencies; ++f) {
    for (size_t b = 0; b < nbeams; ++b) {
      for (size_t d = 0; d < ndishes; ++d) {
        for (size_t c = 0; c < ncomplex; ++c) {
          cout << "A["
               // << ",f=" << f << ",b=" << b << ",d=" << d
               << ",c=" << c
               << "]=" << int(Aarray.at(Alinear(f, b, d, c) / 2)[c]) << "\n";
        }
      }
    }
  }

  vector<icomplex4> Aarray1(Asize1 / 2);
  for (size_t f = 0; f < nfrequencies; ++f) {
    for (size_t b = 0; b < nbeams; ++b) {
      for (size_t p1 = 0; p1 < npolarizations; ++p1) {
        for (size_t c1 = 0; c1 < ncomplex; ++c1) {
          for (size_t d = 0; d < ndishes; ++d) {
            for (size_t p2 = 0; p2 < npolarizations; ++p2) {
              const icomplex4 Ac = Aarray[Alinear(f, b, d, 0) / 2];
              // [1]=real, [0]=imag
              const signed char A[2] = {Ac.imag(), Ac.real()};
              signed char A1[2];
              if (p1 == p2) {
                // Want:
                //   J.re = A.re * E.re - A.im * E.im
                //   J.im = A.im * E.re + A.re * E.im
                // Old layout:
                //   J[p][0] = A[0] * E[p][1] + A[1] * E[p][0]
                //   J[p][1] = A[1] * E[p][1] - A[0] * E[p][0]
                // New layout:
                //   J[p][0] = A1[p][q][0] * E[p][1] + A1[p][q][1] * E[p][0]
                //   J[p][1] = A1[p][q][1] * E[p][1] - A1[p][q][0] * E[p][0]
                // Coefficients:
                //   A1[p][q][0][0] = delta[p][q] A[1]
                //   A1[p][q][0][1] = delta[p][q] A[0]
                //   A1[p][q][1][0] = delta[p][q] (-A[0])
                //   A1[p][q][1][1] = delta[p][q] A[1]
                //
                // Setting A1[c1][c2]
                if (c1 == 1) { // real part
                  A1[1] = A[1];
                  A1[0] = -A[0];
                } else { // imaginary part
                  A1[1] = A[0];
                  A1[0] = A[1];
                }
              } else {
                A1[1] = 0;
                A1[0] = 0;
              }
              const icomplex4 A1c(A1[1], A1[0]);
              Aarray1[Alinear1(f, b, p1, c1, d, p2, 0) / 2] = A1c;
            }
          }
        }
      }
    }
  }

  for (size_t f = 0; f < nfrequencies; ++f) {
    for (size_t b = 0; b < nbeams; ++b) {
      for (size_t p1 = 0; p1 < npolarizations; ++p1) {
        for (size_t c1 = 0; c1 < ncomplex; ++c1) {
          for (size_t d = 0; d < ndishes; ++d) {
            for (size_t p2 = 0; p2 < npolarizations; ++p2) {
              for (size_t c2 = 0; c2 < ncomplex; ++c2) {
                cout << "A1["
                     // << ",f=" << f << ",b=" << b
                     << ",p1=" << p1 << ",c1="
                     << c1
                     // << ",d=" << d
                     << ",p2=" << p2 << ",c2=" << c2 << "]="
                     << int(Aarray1.at(Alinear1(f, b, p1, c1, d, p2, c2) /
                                       2)[c2])
                     << "\n";
              }
            }
          }
        }
      }
    }
  }

  return Aarray1;
}

vector<float> prepare_G(const vector<float> &Garray) {
  vector<float> Garray1(Gsize1);
  for (size_t f = 0; f < nfrequencies; ++f) {
    for (size_t b = 0; b < nbeams; ++b) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t c = 0; c < ncomplex; ++c) {
          Garray1[Glinear1(f, b, p, c)] = Garray[Glinear(f, b)];
        }
      }
    }
  }
  return Garray1;
}

vector<icomplex4> prepare_J(const vector<icomplex4> &Jarray) {
  vector<icomplex4> Jarray1(Jsize1);
  return Jarray1;
}

void restore_J(vector<icomplex4> &Jarray, const vector<icomplex4> &Jarray1) {
  for (size_t t = 0; t < ntimes; ++t) {
    for (size_t f = 0; f < nfrequencies; ++f) {
      for (size_t b = 0; b < nbeams; ++b) {
        for (size_t p = 0; p < npolarizations; ++p) {
          for (size_t c = 0; c < ncomplex; ++c) {
            cout << "J1["
                 << ",t=" << t << ",f=" << f << ",b=" << b << ",p=" << p
                 << ",c=" << c
                 << "]=" << int(Jarray1.at(Jlinear1(t, f, b, p, c) / 2)[c])
                 << "\n";
          }
        }
      }
    }
  }

  for (size_t b = 0; b < nbeams; ++b) {
    for (size_t f = 0; f < nfrequencies; ++f) {
      for (size_t p = 0; p < npolarizations; ++p) {
        for (size_t t = 0; t < ntimes; ++t) {
          Jarray[Jlinear(b, f, p, t, 0) / 2] =
              Jarray1[Jlinear1(t, f, b, p, 0) / 2];
        }
      }
    }
  }
}

void form_beams(unsigned char *restrict const Jarray,
                const unsigned char *restrict const Earray,
                const unsigned char *restrict const Aarray,
                const float *restrict Garray) {
  // This is the array layout. Having the polarization inside requires
  // making A four times as large, and doubles the number of floating
  // point operations. Avoiding this requires changing the memory
  // layout of E.
  //
  // We double A once more to implement complex multiplication. Since
  // A is rather small, and is read only once for each beam forming,
  // the latter seems fine.
  //
  // J = A * E
  // J.re = A.re * E.re - A.im * E.im
  // J.im = A.im * E.re + A.re * E.im

  for (unsigned int f = 0; f < nfrequencies; ++f) {
    printf("f=%u\n", f);
    for (unsigned int t = 0; t < ntimes; ++t) {
      for (unsigned int b = 0; b < nbeams * npolarizations * ncomplex; ++b) {

        int C = 0;

        for (unsigned int d = 0; d < ndishes * npolarizations * ncomplex; ++d) {

          // const unsigned int Aindex =
          //     d + ndishes * npolarizations * ncomplex *
          //             (b + nbeams * npolarizations * ncomplex * f);
          // assert(Aindex < Asize1);
          const unsigned int Aindex =
              Alinear1(f, b / 4, (b / 2) % 2, b % 2, d / 4, (d / 2) % 2, d % 2);
          const icomplex4 *const Aptr = (const icomplex4 *)&Aarray[Aindex / 2];
          const signed char A = (*Aptr)[Aindex % 2];

          // const unsigned int Eindex =
          //     d + ndishes * npolarizations * ncomplex * (f +
          //     nfrequencies * t);
          // assert(Eindex < Esize1);
          const unsigned int Eindex = Elinear1(t, f, d / 4, (d / 2) % 2, d % 2);
          const icomplex4 *const Eptr = (const icomplex4 *)&Earray[Eindex / 2];
          const signed char B = (*Eptr)[Eindex % 2];

          // Multiply
          C += A * B;
        }

        const int rawJ = C;
        cout << "rawJ["
             << ",f=" << f << ",t=" << t / 4 << ",b=" << b / 4
             << ",p=" << (b / 2) % 2 << ",c=" << b % 2 << "]=" << rawJ << "\n";

        // Remove offset from 4-bit complex representation in E
        // rawJ -= 8 * ndishes * npolarizations * ncomplex;

        // Apply gain
        // const unsigned int Glinear = b + nbeams * npolarizations *
        // ncomplex * f; assert(Gindex < Gsize);
        const unsigned int Gindex = Glinear1(f, b / 4, (b / 2) % 2, b % 2);
        const int Jint =
            min(7, max(-7, int(lrintf(Garray[Gindex] * float(rawJ)))));

        // Assemble 4-bit complex number
        // const unsigned char Juchar = Jint + 8;
        const unsigned char J = Jint;

        const unsigned int Jlinear =
            b + nbeams * npolarizations * ncomplex * (f + nfrequencies * t);
        assert(Jlinear < Jsize);
        unsigned char *const Jptr = &Jarray[Jlinear / 2];
        *(icomplex4 *)Jptr =
            icomplex4(Jlinear % 2 == 1 ? J : ((const icomplex4 *)Jptr)->real(),
                      Jlinear % 2 == 0 ? J : ((const icomplex4 *)Jptr)->imag());
      }
    }
  }
}

int main(int argc, char **argv) {
  cout << "beamforming.cpu2\n";

  vector<icomplex4> Earray;
  vector<icomplex4> Aarray;
  vector<float> Garray;
  vector<icomplex4> Jarray;
  setup(Earray, Aarray, Garray, Jarray);

  // Modify layouts
  // We don't modify the layout of E
  const vector<icomplex4> Earray1 = Earray;
  vector<icomplex4> Aarray1 = prepare_A(Aarray);
  vector<float> Garray1 = prepare_G(Garray);
  vector<icomplex4> Jarray1 = prepare_J(Jarray);

  cout << "Forming beams...\n";
  form_beams((unsigned char *)Jarray1.data(),
             (const unsigned char *)Earray1.data(),
             (const unsigned char *)Aarray1.data(), Garray1.data());

  // Undo layout modification
  restore_J(Jarray, Jarray1);

  check(Jarray);

  cout << "Done.\n";
  return 0;
}
