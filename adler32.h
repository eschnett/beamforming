#ifndef ADLER32_H
#define ADLER32_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t adler32(unsigned char *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif // #ifndef ADLER32_H
