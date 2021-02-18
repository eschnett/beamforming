// Taken from <https://en.wikipedia.org/wiki/Adler-32>

#include "adler32.h"

const uint32_t MOD_ADLER = 65521;

uint32_t adler32(const unsigned char *data, size_t len)
// Where data is the location of the data in physical memory and len
// is the length of the data in bytes
{
  uint32_t a = 1, b = 0;

  // Process each byte of the data in order
  for (size_t index = 0; index < len; ++index) {
    a = (a + data[index]) % MOD_ADLER;
    b = (b + a) % MOD_ADLER;
  }

  return (b << 16) | a;
}
