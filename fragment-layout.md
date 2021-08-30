# CUDA Matrix Fragment Layout

## Layout for 4-bit matrices

(As discussed.)

## Layout for 8-bit matrices

8-bit matrices can be loaded in row-major or column-major mode. It
seems that CUDA has a preferred register layout, and uses multiple
memory loads and then permute instructions to shuffle from the given
memory to the preferred register layout if memory and preferred
register layout don't match. If they layouts do match, then loading a
matrix translates to a simple and efficient load instructions that
load 32-bit registers from memory, one full cache line at a time.
Given this, it would seem beneficial to load matrices from the
preferred memory layout.

Preferred (efficient) memory layouts:

| m  | n  | k  | M | major | efficient? |
|----|----|----|---|-------|------------|
|  8 | 32 | 16 | A | col   | no         |
|  8 | 32 | 16 | A | row   | yes        |
|  8 | 32 | 16 | B | col   | yes        |
|  8 | 32 | 16 | B | row   | no         |
| 16 | 16 | 16 | A | col   | no         |
| 16 | 16 | 16 | A | row   | yes        |
| 16 | 16 | 16 | B | col   | yes        |
| 16 | 16 | 16 | B | row   | no         |
| 32 |  8 | 16 | A | col   | no         |
| 32 |  8 | 16 | A | row   | yes        |
| 32 |  8 | 16 | B | col   | yes        |
| 32 |  8 | 16 | B | row   | no         |

The preferred layout is always row-major for A and column-major for B.

For example:
- Store bytes with values 0x00...0xff consecutively in memory
- Load these values a row-major matrix A, with m=16,n=16,k=16
- A is then stored in 2 registers
- The first register of thread 0 holds the values 00 01 02 03, the
  second register holds 80 81 82 83 (all values in hex)
- For m=32,n=8,k=16, the matrix A is stored in 4 registers. On thread
  0, these hold the values at offsets 000 001 002 003, 080 081 082
  083, 100 101 102 103, and 180 181 182 183.

## Layout for 32-bit accumulators

The register layout of the accumulator depends only on m, n, and k. It
is independent of any memory layout, or of the element type of the A
and B matrices (as per the standard). It is also independent of the
element type of the accumulator (as per the Nvidia source code).

For 8-bit integers, the allowed matrix sizes imply that the matrix C
has 256 elements, and thus each thread holds 8 values (in 8
registers). The tables below shows which of these 256 values are held
in the registers of thread 0. The first two digits encode the row, the
last two digits encode the column number.

### NVIDIA GeForce RTX 2080 Ti (on Symmetry):

| m  | n  | k  | #R | elements on thread 0                    |
|----|----|----|----|-----------------------------------------|
|  8 |  8 | 32 |  2 | 0000 0001                               |
|  8 | 32 | 16 |  8 | 0000 0001 0008 0009 0010 0011 0018 0019 |
| 16 | 16 | 16 |  8 | 0000 0001 0800 0801 0008 0009 0808 0809 |
| 32 |  8 | 16 |  8 | 0000 0001 0800 0801 1000 1001 1800 1801 |

For 8-bit operations, the threads are grouped into 8 groups of 4
threads each. The registers are grouped into 4 pairs of 2 registers
each. These 4x2-blocks hold 8 consecutive columns of the same row of
C, always in the same pattern:

```
    0000 0001
    0002 0003
    0004 0005
    0006 0007
```
(Here the rows correspond to threads, the columns correspond to registers.)

The mapping to these thread/register blocks is different for each C
matrix size, and there is no obvious pattern.

### A40 (on Sky):

Here we finally find a difference between compute capabilities 8.0 and
7.5! The case m=8, n=32, k=16 (and only this case) differs from the
table above. The pattern has the 4x2-blocks of the same shape as
above, but the layout within these blocks is transposed: The
4x2-blocks hold 8 consecutive rows of the same column of C.

```
    0000 0100 0008 0108 0010 0110 0018 0118
    0200 0300 0208 0308 0210 0310 0218 0318
    0400 0500 0408 0508 0410 0510 0418 0518
    0600 0700 0608 0708 0610 0710 0618 0718
    0001 0101 0009 0109 0011 0111 0019 0119
    0201 0301 0209 0309 0211 0311 0219 0319
    0401 0501 0409 0509 0411 0511 0419 0519
    0601 0701 0609 0709 0611 0711 0619 0719
    0002 0102 000a 010a 0012 0112 001a 011a
    0202 0302 020a 030a 0212 0312 021a 031a
    0402 0502 040a 050a 0412 0512 041a 051a
    0602 0702 060a 070a 0612 0712 061a 071a
    0003 0103 000b 010b 0013 0113 001b 011b
    0203 0303 020b 030b 0213 0313 021b 031b
    0403 0503 040b 050b 0413 0513 041b 051b
    0603 0703 060b 070b 0613 0713 061b 071b
    0004 0104 000c 010c 0014 0114 001c 011c
    0204 0304 020c 030c 0214 0314 021c 031c
    0404 0504 040c 050c 0414 0514 041c 051c
    0604 0704 060c 070c 0614 0714 061c 071c
    0005 0105 000d 010d 0015 0115 001d 011d
    0205 0305 020d 030d 0215 0315 021d 031d
    0405 0505 040d 050d 0415 0515 041d 051d
    0605 0705 060d 070d 0615 0715 061d 071d
    0006 0106 000e 010e 0016 0116 001e 011e
    0206 0306 020e 030e 0216 0316 021e 031e
    0406 0506 040e 050e 0416 0516 041e 051e
    0606 0706 060e 070e 0616 0716 061e 071e
    0007 0107 000f 010f 0017 0117 001f 011f
    0207 0307 020f 030f 0217 0317 021f 031f
    0407 0507 040f 050f 0417 0517 041f 051f
    0607 0707 060f 070f 0617 0717 061f 071f
```

## Gory Details

No insights below, just the raw data without explanations.

### NVIDIA GeForce RTX 2080 Ti (on Symmetry):

```
fragment2.cuda

matrix_a [bits=4,m=8,n=8,k=32]:
  Amat:
    00 08 10 18
    20 28 30 38
    40 48 50 58
    60 68 70 78
    80 88 90 98
    a0 a8 b0 b8
    c0 c8 d0 d8
    e0 e8 f0 f8
  fragmentA:
    00
    08
    10
    18
    20
    28
    30
    38
    40
    48
    50
    58
    60
    68
    70
    78
    80
    88
    90
    98
    a0
    a8
    b0
    b8
    c0
    c8
    d0
    d8
    e0
    e8
    f0
    f8

matrix_b [bits=4,m=8,n=8,k=32]:
  Bmat:
    00 20 40 60 80 a0 c0 e0
    01 21 41 61 81 a1 c1 e1
    02 22 42 62 82 a2 c2 e2
    03 23 43 63 83 a3 c3 e3
  fragmentB:
    00
    01
    02
    03
    20
    21
    22
    23
    40
    41
    42
    43
    60
    61
    62
    63
    80
    81
    82
    83
    a0
    a1
    a2
    a3
    c0
    c1
    c2
    c3
    e0
    e1
    e2
    e3

accumulator [m=8,n=8,k=32]:
  Cmat:
    0000 0001 0002 0003 0004 0005 0006 0007
    0100 0101 0102 0103 0104 0105 0106 0107
    0200 0201 0202 0203 0204 0205 0206 0207
    0300 0301 0302 0303 0304 0305 0306 0307
    0400 0401 0402 0403 0404 0405 0406 0407
    0500 0501 0502 0503 0504 0505 0506 0507
    0600 0601 0602 0603 0604 0605 0606 0607
    0700 0701 0702 0703 0704 0705 0706 0707
  fragmentC:
    0000 0001
    0002 0003
    0004 0005
    0006 0007
    0100 0101
    0102 0103
    0104 0105
    0106 0107
    0200 0201
    0202 0203
    0204 0205
    0206 0207
    0300 0301
    0302 0303
    0304 0305
    0306 0307
    0400 0401
    0402 0403
    0404 0405
    0406 0407
    0500 0501
    0502 0503
    0504 0505
    0506 0507
    0600 0601
    0602 0603
    0604 0605
    0606 0607
    0700 0701
    0702 0703
    0704 0705
    0706 0707

matrix_a [bits=8,m=16,n=16,k=16,layout=row_major]:
  Amat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f
    90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf
    d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef
    f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
  fragmentA:
    00 01 02 03 80 81 82 83
    04 05 06 07 84 85 86 87
    08 09 0a 0b 88 89 8a 8b
    0c 0d 0e 0f 8c 8d 8e 8f
    10 11 12 13 90 91 92 93
    14 15 16 17 94 95 96 97
    18 19 1a 1b 98 99 9a 9b
    1c 1d 1e 1f 9c 9d 9e 9f
    20 21 22 23 a0 a1 a2 a3
    24 25 26 27 a4 a5 a6 a7
    28 29 2a 2b a8 a9 aa ab
    2c 2d 2e 2f ac ad ae af
    30 31 32 33 b0 b1 b2 b3
    34 35 36 37 b4 b5 b6 b7
    38 39 3a 3b b8 b9 ba bb
    3c 3d 3e 3f bc bd be bf
    40 41 42 43 c0 c1 c2 c3
    44 45 46 47 c4 c5 c6 c7
    48 49 4a 4b c8 c9 ca cb
    4c 4d 4e 4f cc cd ce cf
    50 51 52 53 d0 d1 d2 d3
    54 55 56 57 d4 d5 d6 d7
    58 59 5a 5b d8 d9 da db
    5c 5d 5e 5f dc dd de df
    60 61 62 63 e0 e1 e2 e3
    64 65 66 67 e4 e5 e6 e7
    68 69 6a 6b e8 e9 ea eb
    6c 6d 6e 6f ec ed ee ef
    70 71 72 73 f0 f1 f2 f3
    74 75 76 77 f4 f5 f6 f7
    78 79 7a 7b f8 f9 fa fb
    7c 7d 7e 7f fc fd fe ff

matrix_a [bits=8,m=16,n=16,k=16,layout=col_major]:
  Amat:
    00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
    01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
    02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
    03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
    04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
    05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
    06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
    07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7
    08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8
    09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9
    0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa
    0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb
    0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc
    0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd
    0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe
    0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff
  fragmentA:
    00 10 20 30 08 18 28 38
    40 50 60 70 48 58 68 78
    80 90 a0 b0 88 98 a8 b8
    c0 d0 e0 f0 c8 d8 e8 f8
    01 11 21 31 09 19 29 39
    41 51 61 71 49 59 69 79
    81 91 a1 b1 89 99 a9 b9
    c1 d1 e1 f1 c9 d9 e9 f9
    02 12 22 32 0a 1a 2a 3a
    42 52 62 72 4a 5a 6a 7a
    82 92 a2 b2 8a 9a aa ba
    c2 d2 e2 f2 ca da ea fa
    03 13 23 33 0b 1b 2b 3b
    43 53 63 73 4b 5b 6b 7b
    83 93 a3 b3 8b 9b ab bb
    c3 d3 e3 f3 cb db eb fb
    04 14 24 34 0c 1c 2c 3c
    44 54 64 74 4c 5c 6c 7c
    84 94 a4 b4 8c 9c ac bc
    c4 d4 e4 f4 cc dc ec fc
    05 15 25 35 0d 1d 2d 3d
    45 55 65 75 4d 5d 6d 7d
    85 95 a5 b5 8d 9d ad bd
    c5 d5 e5 f5 cd dd ed fd
    06 16 26 36 0e 1e 2e 3e
    46 56 66 76 4e 5e 6e 7e
    86 96 a6 b6 8e 9e ae be
    c6 d6 e6 f6 ce de ee fe
    07 17 27 37 0f 1f 2f 3f
    47 57 67 77 4f 5f 6f 7f
    87 97 a7 b7 8f 9f af bf
    c7 d7 e7 f7 cf df ef ff

matrix_b [bits=8,m=16,n=16,k=16,layout=row_major]:
  Bmat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f
    90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf
    d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef
    f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
  fragmentB:
    00 10 20 30 08 18 28 38
    40 50 60 70 48 58 68 78
    80 90 a0 b0 88 98 a8 b8
    c0 d0 e0 f0 c8 d8 e8 f8
    01 11 21 31 09 19 29 39
    41 51 61 71 49 59 69 79
    81 91 a1 b1 89 99 a9 b9
    c1 d1 e1 f1 c9 d9 e9 f9
    02 12 22 32 0a 1a 2a 3a
    42 52 62 72 4a 5a 6a 7a
    82 92 a2 b2 8a 9a aa ba
    c2 d2 e2 f2 ca da ea fa
    03 13 23 33 0b 1b 2b 3b
    43 53 63 73 4b 5b 6b 7b
    83 93 a3 b3 8b 9b ab bb
    c3 d3 e3 f3 cb db eb fb
    04 14 24 34 0c 1c 2c 3c
    44 54 64 74 4c 5c 6c 7c
    84 94 a4 b4 8c 9c ac bc
    c4 d4 e4 f4 cc dc ec fc
    05 15 25 35 0d 1d 2d 3d
    45 55 65 75 4d 5d 6d 7d
    85 95 a5 b5 8d 9d ad bd
    c5 d5 e5 f5 cd dd ed fd
    06 16 26 36 0e 1e 2e 3e
    46 56 66 76 4e 5e 6e 7e
    86 96 a6 b6 8e 9e ae be
    c6 d6 e6 f6 ce de ee fe
    07 17 27 37 0f 1f 2f 3f
    47 57 67 77 4f 5f 6f 7f
    87 97 a7 b7 8f 9f af bf
    c7 d7 e7 f7 cf df ef ff

matrix_b [bits=8,m=16,n=16,k=16,layout=col_major]:
  Bmat:
    00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
    01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
    02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
    03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
    04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
    05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
    06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
    07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7
    08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8
    09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9
    0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa
    0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb
    0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc
    0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd
    0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe
    0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff
  fragmentB:
    00 01 02 03 80 81 82 83
    04 05 06 07 84 85 86 87
    08 09 0a 0b 88 89 8a 8b
    0c 0d 0e 0f 8c 8d 8e 8f
    10 11 12 13 90 91 92 93
    14 15 16 17 94 95 96 97
    18 19 1a 1b 98 99 9a 9b
    1c 1d 1e 1f 9c 9d 9e 9f
    20 21 22 23 a0 a1 a2 a3
    24 25 26 27 a4 a5 a6 a7
    28 29 2a 2b a8 a9 aa ab
    2c 2d 2e 2f ac ad ae af
    30 31 32 33 b0 b1 b2 b3
    34 35 36 37 b4 b5 b6 b7
    38 39 3a 3b b8 b9 ba bb
    3c 3d 3e 3f bc bd be bf
    40 41 42 43 c0 c1 c2 c3
    44 45 46 47 c4 c5 c6 c7
    48 49 4a 4b c8 c9 ca cb
    4c 4d 4e 4f cc cd ce cf
    50 51 52 53 d0 d1 d2 d3
    54 55 56 57 d4 d5 d6 d7
    58 59 5a 5b d8 d9 da db
    5c 5d 5e 5f dc dd de df
    60 61 62 63 e0 e1 e2 e3
    64 65 66 67 e4 e5 e6 e7
    68 69 6a 6b e8 e9 ea eb
    6c 6d 6e 6f ec ed ee ef
    70 71 72 73 f0 f1 f2 f3
    74 75 76 77 f4 f5 f6 f7
    78 79 7a 7b f8 f9 fa fb
    7c 7d 7e 7f fc fd fe ff

accumulator [m=16,n=16,k=16]:
  Cmat:
    0000 0001 0002 0003 0004 0005 0006 0007 0008 0009 000a 000b 000c 000d 000e 000f
    0100 0101 0102 0103 0104 0105 0106 0107 0108 0109 010a 010b 010c 010d 010e 010f
    0200 0201 0202 0203 0204 0205 0206 0207 0208 0209 020a 020b 020c 020d 020e 020f
    0300 0301 0302 0303 0304 0305 0306 0307 0308 0309 030a 030b 030c 030d 030e 030f
    0400 0401 0402 0403 0404 0405 0406 0407 0408 0409 040a 040b 040c 040d 040e 040f
    0500 0501 0502 0503 0504 0505 0506 0507 0508 0509 050a 050b 050c 050d 050e 050f
    0600 0601 0602 0603 0604 0605 0606 0607 0608 0609 060a 060b 060c 060d 060e 060f
    0700 0701 0702 0703 0704 0705 0706 0707 0708 0709 070a 070b 070c 070d 070e 070f
    0800 0801 0802 0803 0804 0805 0806 0807 0808 0809 080a 080b 080c 080d 080e 080f
    0900 0901 0902 0903 0904 0905 0906 0907 0908 0909 090a 090b 090c 090d 090e 090f
    0a00 0a01 0a02 0a03 0a04 0a05 0a06 0a07 0a08 0a09 0a0a 0a0b 0a0c 0a0d 0a0e 0a0f
    0b00 0b01 0b02 0b03 0b04 0b05 0b06 0b07 0b08 0b09 0b0a 0b0b 0b0c 0b0d 0b0e 0b0f
    0c00 0c01 0c02 0c03 0c04 0c05 0c06 0c07 0c08 0c09 0c0a 0c0b 0c0c 0c0d 0c0e 0c0f
    0d00 0d01 0d02 0d03 0d04 0d05 0d06 0d07 0d08 0d09 0d0a 0d0b 0d0c 0d0d 0d0e 0d0f
    0e00 0e01 0e02 0e03 0e04 0e05 0e06 0e07 0e08 0e09 0e0a 0e0b 0e0c 0e0d 0e0e 0e0f
    0f00 0f01 0f02 0f03 0f04 0f05 0f06 0f07 0f08 0f09 0f0a 0f0b 0f0c 0f0d 0f0e 0f0f
  fragmentC:
    0000 0001 0800 0801 0008 0009 0808 0809
    0002 0003 0802 0803 000a 000b 080a 080b
    0004 0005 0804 0805 000c 000d 080c 080d
    0006 0007 0806 0807 000e 000f 080e 080f
    0100 0101 0900 0901 0108 0109 0908 0909
    0102 0103 0902 0903 010a 010b 090a 090b
    0104 0105 0904 0905 010c 010d 090c 090d
    0106 0107 0906 0907 010e 010f 090e 090f
    0200 0201 0a00 0a01 0208 0209 0a08 0a09
    0202 0203 0a02 0a03 020a 020b 0a0a 0a0b
    0204 0205 0a04 0a05 020c 020d 0a0c 0a0d
    0206 0207 0a06 0a07 020e 020f 0a0e 0a0f
    0300 0301 0b00 0b01 0308 0309 0b08 0b09
    0302 0303 0b02 0b03 030a 030b 0b0a 0b0b
    0304 0305 0b04 0b05 030c 030d 0b0c 0b0d
    0306 0307 0b06 0b07 030e 030f 0b0e 0b0f
    0400 0401 0c00 0c01 0408 0409 0c08 0c09
    0402 0403 0c02 0c03 040a 040b 0c0a 0c0b
    0404 0405 0c04 0c05 040c 040d 0c0c 0c0d
    0406 0407 0c06 0c07 040e 040f 0c0e 0c0f
    0500 0501 0d00 0d01 0508 0509 0d08 0d09
    0502 0503 0d02 0d03 050a 050b 0d0a 0d0b
    0504 0505 0d04 0d05 050c 050d 0d0c 0d0d
    0506 0507 0d06 0d07 050e 050f 0d0e 0d0f
    0600 0601 0e00 0e01 0608 0609 0e08 0e09
    0602 0603 0e02 0e03 060a 060b 0e0a 0e0b
    0604 0605 0e04 0e05 060c 060d 0e0c 0e0d
    0606 0607 0e06 0e07 060e 060f 0e0e 0e0f
    0700 0701 0f00 0f01 0708 0709 0f08 0f09
    0702 0703 0f02 0f03 070a 070b 0f0a 0f0b
    0704 0705 0f04 0f05 070c 070d 0f0c 0f0d
    0706 0707 0f06 0f07 070e 070f 0f0e 0f0f

matrix_a [bits=8,m=32,n=8,k=16,layout=row_major]:
  Amat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f
    90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf
    d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef
    f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f
    90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf
    d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef
    f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
  fragmentA:
    00 01 02 03 80 81 82 83 00 01 02 03 80 81 82 83
    04 05 06 07 84 85 86 87 04 05 06 07 84 85 86 87
    08 09 0a 0b 88 89 8a 8b 08 09 0a 0b 88 89 8a 8b
    0c 0d 0e 0f 8c 8d 8e 8f 0c 0d 0e 0f 8c 8d 8e 8f
    10 11 12 13 90 91 92 93 10 11 12 13 90 91 92 93
    14 15 16 17 94 95 96 97 14 15 16 17 94 95 96 97
    18 19 1a 1b 98 99 9a 9b 18 19 1a 1b 98 99 9a 9b
    1c 1d 1e 1f 9c 9d 9e 9f 1c 1d 1e 1f 9c 9d 9e 9f
    20 21 22 23 a0 a1 a2 a3 20 21 22 23 a0 a1 a2 a3
    24 25 26 27 a4 a5 a6 a7 24 25 26 27 a4 a5 a6 a7
    28 29 2a 2b a8 a9 aa ab 28 29 2a 2b a8 a9 aa ab
    2c 2d 2e 2f ac ad ae af 2c 2d 2e 2f ac ad ae af
    30 31 32 33 b0 b1 b2 b3 30 31 32 33 b0 b1 b2 b3
    34 35 36 37 b4 b5 b6 b7 34 35 36 37 b4 b5 b6 b7
    38 39 3a 3b b8 b9 ba bb 38 39 3a 3b b8 b9 ba bb
    3c 3d 3e 3f bc bd be bf 3c 3d 3e 3f bc bd be bf
    40 41 42 43 c0 c1 c2 c3 40 41 42 43 c0 c1 c2 c3
    44 45 46 47 c4 c5 c6 c7 44 45 46 47 c4 c5 c6 c7
    48 49 4a 4b c8 c9 ca cb 48 49 4a 4b c8 c9 ca cb
    4c 4d 4e 4f cc cd ce cf 4c 4d 4e 4f cc cd ce cf
    50 51 52 53 d0 d1 d2 d3 50 51 52 53 d0 d1 d2 d3
    54 55 56 57 d4 d5 d6 d7 54 55 56 57 d4 d5 d6 d7
    58 59 5a 5b d8 d9 da db 58 59 5a 5b d8 d9 da db
    5c 5d 5e 5f dc dd de df 5c 5d 5e 5f dc dd de df
    60 61 62 63 e0 e1 e2 e3 60 61 62 63 e0 e1 e2 e3
    64 65 66 67 e4 e5 e6 e7 64 65 66 67 e4 e5 e6 e7
    68 69 6a 6b e8 e9 ea eb 68 69 6a 6b e8 e9 ea eb
    6c 6d 6e 6f ec ed ee ef 6c 6d 6e 6f ec ed ee ef
    70 71 72 73 f0 f1 f2 f3 70 71 72 73 f0 f1 f2 f3
    74 75 76 77 f4 f5 f6 f7 74 75 76 77 f4 f5 f6 f7
    78 79 7a 7b f8 f9 fa fb 78 79 7a 7b f8 f9 fa fb
    7c 7d 7e 7f fc fd fe ff 7c 7d 7e 7f fc fd fe ff

matrix_a [bits=8,m=32,n=8,k=16,layout=col_major]:
  Amat:
    00 20 40 60 80 a0 c0 e0 00 20 40 60 80 a0 c0 e0
    01 21 41 61 81 a1 c1 e1 01 21 41 61 81 a1 c1 e1
    02 22 42 62 82 a2 c2 e2 02 22 42 62 82 a2 c2 e2
    03 23 43 63 83 a3 c3 e3 03 23 43 63 83 a3 c3 e3
    04 24 44 64 84 a4 c4 e4 04 24 44 64 84 a4 c4 e4
    05 25 45 65 85 a5 c5 e5 05 25 45 65 85 a5 c5 e5
    06 26 46 66 86 a6 c6 e6 06 26 46 66 86 a6 c6 e6
    07 27 47 67 87 a7 c7 e7 07 27 47 67 87 a7 c7 e7
    08 28 48 68 88 a8 c8 e8 08 28 48 68 88 a8 c8 e8
    09 29 49 69 89 a9 c9 e9 09 29 49 69 89 a9 c9 e9
    0a 2a 4a 6a 8a aa ca ea 0a 2a 4a 6a 8a aa ca ea
    0b 2b 4b 6b 8b ab cb eb 0b 2b 4b 6b 8b ab cb eb
    0c 2c 4c 6c 8c ac cc ec 0c 2c 4c 6c 8c ac cc ec
    0d 2d 4d 6d 8d ad cd ed 0d 2d 4d 6d 8d ad cd ed
    0e 2e 4e 6e 8e ae ce ee 0e 2e 4e 6e 8e ae ce ee
    0f 2f 4f 6f 8f af cf ef 0f 2f 4f 6f 8f af cf ef
    10 30 50 70 90 b0 d0 f0 10 30 50 70 90 b0 d0 f0
    11 31 51 71 91 b1 d1 f1 11 31 51 71 91 b1 d1 f1
    12 32 52 72 92 b2 d2 f2 12 32 52 72 92 b2 d2 f2
    13 33 53 73 93 b3 d3 f3 13 33 53 73 93 b3 d3 f3
    14 34 54 74 94 b4 d4 f4 14 34 54 74 94 b4 d4 f4
    15 35 55 75 95 b5 d5 f5 15 35 55 75 95 b5 d5 f5
    16 36 56 76 96 b6 d6 f6 16 36 56 76 96 b6 d6 f6
    17 37 57 77 97 b7 d7 f7 17 37 57 77 97 b7 d7 f7
    18 38 58 78 98 b8 d8 f8 18 38 58 78 98 b8 d8 f8
    19 39 59 79 99 b9 d9 f9 19 39 59 79 99 b9 d9 f9
    1a 3a 5a 7a 9a ba da fa 1a 3a 5a 7a 9a ba da fa
    1b 3b 5b 7b 9b bb db fb 1b 3b 5b 7b 9b bb db fb
    1c 3c 5c 7c 9c bc dc fc 1c 3c 5c 7c 9c bc dc fc
    1d 3d 5d 7d 9d bd dd fd 1d 3d 5d 7d 9d bd dd fd
    1e 3e 5e 7e 9e be de fe 1e 3e 5e 7e 9e be de fe
    1f 3f 5f 7f 9f bf df ff 1f 3f 5f 7f 9f bf df ff
  fragmentA:
    00 20 40 60 08 28 48 68 10 30 50 70 18 38 58 78
    80 a0 c0 e0 88 a8 c8 e8 90 b0 d0 f0 98 b8 d8 f8
    00 20 40 60 08 28 48 68 10 30 50 70 18 38 58 78
    80 a0 c0 e0 88 a8 c8 e8 90 b0 d0 f0 98 b8 d8 f8
    01 21 41 61 09 29 49 69 11 31 51 71 19 39 59 79
    81 a1 c1 e1 89 a9 c9 e9 91 b1 d1 f1 99 b9 d9 f9
    01 21 41 61 09 29 49 69 11 31 51 71 19 39 59 79
    81 a1 c1 e1 89 a9 c9 e9 91 b1 d1 f1 99 b9 d9 f9
    02 22 42 62 0a 2a 4a 6a 12 32 52 72 1a 3a 5a 7a
    82 a2 c2 e2 8a aa ca ea 92 b2 d2 f2 9a ba da fa
    02 22 42 62 0a 2a 4a 6a 12 32 52 72 1a 3a 5a 7a
    82 a2 c2 e2 8a aa ca ea 92 b2 d2 f2 9a ba da fa
    03 23 43 63 0b 2b 4b 6b 13 33 53 73 1b 3b 5b 7b
    83 a3 c3 e3 8b ab cb eb 93 b3 d3 f3 9b bb db fb
    03 23 43 63 0b 2b 4b 6b 13 33 53 73 1b 3b 5b 7b
    83 a3 c3 e3 8b ab cb eb 93 b3 d3 f3 9b bb db fb
    04 24 44 64 0c 2c 4c 6c 14 34 54 74 1c 3c 5c 7c
    84 a4 c4 e4 8c ac cc ec 94 b4 d4 f4 9c bc dc fc
    04 24 44 64 0c 2c 4c 6c 14 34 54 74 1c 3c 5c 7c
    84 a4 c4 e4 8c ac cc ec 94 b4 d4 f4 9c bc dc fc
    05 25 45 65 0d 2d 4d 6d 15 35 55 75 1d 3d 5d 7d
    85 a5 c5 e5 8d ad cd ed 95 b5 d5 f5 9d bd dd fd
    05 25 45 65 0d 2d 4d 6d 15 35 55 75 1d 3d 5d 7d
    85 a5 c5 e5 8d ad cd ed 95 b5 d5 f5 9d bd dd fd
    06 26 46 66 0e 2e 4e 6e 16 36 56 76 1e 3e 5e 7e
    86 a6 c6 e6 8e ae ce ee 96 b6 d6 f6 9e be de fe
    06 26 46 66 0e 2e 4e 6e 16 36 56 76 1e 3e 5e 7e
    86 a6 c6 e6 8e ae ce ee 96 b6 d6 f6 9e be de fe
    07 27 47 67 0f 2f 4f 6f 17 37 57 77 1f 3f 5f 7f
    87 a7 c7 e7 8f af cf ef 97 b7 d7 f7 9f bf df ff
    07 27 47 67 0f 2f 4f 6f 17 37 57 77 1f 3f 5f 7f
    87 a7 c7 e7 8f af cf ef 97 b7 d7 f7 9f bf df ff

matrix_b [bits=8,m=32,n=8,k=16,layout=row_major]:
  Bmat:
    00 01 02 03 04 05 06 07
    08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17
    18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27
    28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37
    38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47
    48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57
    58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67
    68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77
    78 79 7a 7b 7c 7d 7e 7f
  fragmentB:
    00 08 10 18
    20 28 30 38
    40 48 50 58
    60 68 70 78
    01 09 11 19
    21 29 31 39
    41 49 51 59
    61 69 71 79
    02 0a 12 1a
    22 2a 32 3a
    42 4a 52 5a
    62 6a 72 7a
    03 0b 13 1b
    23 2b 33 3b
    43 4b 53 5b
    63 6b 73 7b
    04 0c 14 1c
    24 2c 34 3c
    44 4c 54 5c
    64 6c 74 7c
    05 0d 15 1d
    25 2d 35 3d
    45 4d 55 5d
    65 6d 75 7d
    06 0e 16 1e
    26 2e 36 3e
    46 4e 56 5e
    66 6e 76 7e
    07 0f 17 1f
    27 2f 37 3f
    47 4f 57 5f
    67 6f 77 7f

matrix_b [bits=8,m=32,n=8,k=16,layout=col_major]:
  Bmat:
    00 10 20 30 40 50 60 70
    01 11 21 31 41 51 61 71
    02 12 22 32 42 52 62 72
    03 13 23 33 43 53 63 73
    04 14 24 34 44 54 64 74
    05 15 25 35 45 55 65 75
    06 16 26 36 46 56 66 76
    07 17 27 37 47 57 67 77
    08 18 28 38 48 58 68 78
    09 19 29 39 49 59 69 79
    0a 1a 2a 3a 4a 5a 6a 7a
    0b 1b 2b 3b 4b 5b 6b 7b
    0c 1c 2c 3c 4c 5c 6c 7c
    0d 1d 2d 3d 4d 5d 6d 7d
    0e 1e 2e 3e 4e 5e 6e 7e
    0f 1f 2f 3f 4f 5f 6f 7f
  fragmentB:
    00 01 02 03
    04 05 06 07
    08 09 0a 0b
    0c 0d 0e 0f
    10 11 12 13
    14 15 16 17
    18 19 1a 1b
    1c 1d 1e 1f
    20 21 22 23
    24 25 26 27
    28 29 2a 2b
    2c 2d 2e 2f
    30 31 32 33
    34 35 36 37
    38 39 3a 3b
    3c 3d 3e 3f
    40 41 42 43
    44 45 46 47
    48 49 4a 4b
    4c 4d 4e 4f
    50 51 52 53
    54 55 56 57
    58 59 5a 5b
    5c 5d 5e 5f
    60 61 62 63
    64 65 66 67
    68 69 6a 6b
    6c 6d 6e 6f
    70 71 72 73
    74 75 76 77
    78 79 7a 7b
    7c 7d 7e 7f

accumulator [m=32,n=8,k=16]:
  Cmat:
    0000 0001 0002 0003 0004 0005 0006 0007
    0100 0101 0102 0103 0104 0105 0106 0107
    0200 0201 0202 0203 0204 0205 0206 0207
    0300 0301 0302 0303 0304 0305 0306 0307
    0400 0401 0402 0403 0404 0405 0406 0407
    0500 0501 0502 0503 0504 0505 0506 0507
    0600 0601 0602 0603 0604 0605 0606 0607
    0700 0701 0702 0703 0704 0705 0706 0707
    0800 0801 0802 0803 0804 0805 0806 0807
    0900 0901 0902 0903 0904 0905 0906 0907
    0a00 0a01 0a02 0a03 0a04 0a05 0a06 0a07
    0b00 0b01 0b02 0b03 0b04 0b05 0b06 0b07
    0c00 0c01 0c02 0c03 0c04 0c05 0c06 0c07
    0d00 0d01 0d02 0d03 0d04 0d05 0d06 0d07
    0e00 0e01 0e02 0e03 0e04 0e05 0e06 0e07
    0f00 0f01 0f02 0f03 0f04 0f05 0f06 0f07
    1000 1001 1002 1003 1004 1005 1006 1007
    1100 1101 1102 1103 1104 1105 1106 1107
    1200 1201 1202 1203 1204 1205 1206 1207
    1300 1301 1302 1303 1304 1305 1306 1307
    1400 1401 1402 1403 1404 1405 1406 1407
    1500 1501 1502 1503 1504 1505 1506 1507
    1600 1601 1602 1603 1604 1605 1606 1607
    1700 1701 1702 1703 1704 1705 1706 1707
    1800 1801 1802 1803 1804 1805 1806 1807
    1900 1901 1902 1903 1904 1905 1906 1907
    1a00 1a01 1a02 1a03 1a04 1a05 1a06 1a07
    1b00 1b01 1b02 1b03 1b04 1b05 1b06 1b07
    1c00 1c01 1c02 1c03 1c04 1c05 1c06 1c07
    1d00 1d01 1d02 1d03 1d04 1d05 1d06 1d07
    1e00 1e01 1e02 1e03 1e04 1e05 1e06 1e07
    1f00 1f01 1f02 1f03 1f04 1f05 1f06 1f07
  fragmentC:
    0000 0001 0800 0801 1000 1001 1800 1801
    0002 0003 0802 0803 1002 1003 1802 1803
    0004 0005 0804 0805 1004 1005 1804 1805
    0006 0007 0806 0807 1006 1007 1806 1807
    0100 0101 0900 0901 1100 1101 1900 1901
    0102 0103 0902 0903 1102 1103 1902 1903
    0104 0105 0904 0905 1104 1105 1904 1905
    0106 0107 0906 0907 1106 1107 1906 1907
    0200 0201 0a00 0a01 1200 1201 1a00 1a01
    0202 0203 0a02 0a03 1202 1203 1a02 1a03
    0204 0205 0a04 0a05 1204 1205 1a04 1a05
    0206 0207 0a06 0a07 1206 1207 1a06 1a07
    0300 0301 0b00 0b01 1300 1301 1b00 1b01
    0302 0303 0b02 0b03 1302 1303 1b02 1b03
    0304 0305 0b04 0b05 1304 1305 1b04 1b05
    0306 0307 0b06 0b07 1306 1307 1b06 1b07
    0400 0401 0c00 0c01 1400 1401 1c00 1c01
    0402 0403 0c02 0c03 1402 1403 1c02 1c03
    0404 0405 0c04 0c05 1404 1405 1c04 1c05
    0406 0407 0c06 0c07 1406 1407 1c06 1c07
    0500 0501 0d00 0d01 1500 1501 1d00 1d01
    0502 0503 0d02 0d03 1502 1503 1d02 1d03
    0504 0505 0d04 0d05 1504 1505 1d04 1d05
    0506 0507 0d06 0d07 1506 1507 1d06 1d07
    0600 0601 0e00 0e01 1600 1601 1e00 1e01
    0602 0603 0e02 0e03 1602 1603 1e02 1e03
    0604 0605 0e04 0e05 1604 1605 1e04 1e05
    0606 0607 0e06 0e07 1606 1607 1e06 1e07
    0700 0701 0f00 0f01 1700 1701 1f00 1f01
    0702 0703 0f02 0f03 1702 1703 1f02 1f03
    0704 0705 0f04 0f05 1704 1705 1f04 1f05
    0706 0707 0f06 0f07 1706 1707 1f06 1f07

matrix_a [bits=8,m=8,n=32,k=16,layout=row_major]:
  Amat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
  fragmentA:
    00 01 02 03
    04 05 06 07
    08 09 0a 0b
    0c 0d 0e 0f
    10 11 12 13
    14 15 16 17
    18 19 1a 1b
    1c 1d 1e 1f
    20 21 22 23
    24 25 26 27
    28 29 2a 2b
    2c 2d 2e 2f
    30 31 32 33
    34 35 36 37
    38 39 3a 3b
    3c 3d 3e 3f
    40 41 42 43
    44 45 46 47
    48 49 4a 4b
    4c 4d 4e 4f
    50 51 52 53
    54 55 56 57
    58 59 5a 5b
    5c 5d 5e 5f
    60 61 62 63
    64 65 66 67
    68 69 6a 6b
    6c 6d 6e 6f
    70 71 72 73
    74 75 76 77
    78 79 7a 7b
    7c 7d 7e 7f

matrix_a [bits=8,m=8,n=32,k=16,layout=col_major]:
  Amat:
    00 08 10 18 20 28 30 38 40 48 50 58 60 68 70 78
    01 09 11 19 21 29 31 39 41 49 51 59 61 69 71 79
    02 0a 12 1a 22 2a 32 3a 42 4a 52 5a 62 6a 72 7a
    03 0b 13 1b 23 2b 33 3b 43 4b 53 5b 63 6b 73 7b
    04 0c 14 1c 24 2c 34 3c 44 4c 54 5c 64 6c 74 7c
    05 0d 15 1d 25 2d 35 3d 45 4d 55 5d 65 6d 75 7d
    06 0e 16 1e 26 2e 36 3e 46 4e 56 5e 66 6e 76 7e
    07 0f 17 1f 27 2f 37 3f 47 4f 57 5f 67 6f 77 7f
  fragmentA:
    00 08 10 18
    20 28 30 38
    40 48 50 58
    60 68 70 78
    01 09 11 19
    21 29 31 39
    41 49 51 59
    61 69 71 79
    02 0a 12 1a
    22 2a 32 3a
    42 4a 52 5a
    62 6a 72 7a
    03 0b 13 1b
    23 2b 33 3b
    43 4b 53 5b
    63 6b 73 7b
    04 0c 14 1c
    24 2c 34 3c
    44 4c 54 5c
    64 6c 74 7c
    05 0d 15 1d
    25 2d 35 3d
    45 4d 55 5d
    65 6d 75 7d
    06 0e 16 1e
    26 2e 36 3e
    46 4e 56 5e
    66 6e 76 7e
    07 0f 17 1f
    27 2f 37 3f
    47 4f 57 5f
    67 6f 77 7f

matrix_b [bits=8,m=8,n=32,k=16,layout=row_major]:
  Bmat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f 70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f 90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f 70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f 90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
  fragmentB:
    00 20 40 60 08 28 48 68 10 30 50 70 18 38 58 78
    80 a0 c0 e0 88 a8 c8 e8 90 b0 d0 f0 98 b8 d8 f8
    00 20 40 60 08 28 48 68 10 30 50 70 18 38 58 78
    80 a0 c0 e0 88 a8 c8 e8 90 b0 d0 f0 98 b8 d8 f8
    01 21 41 61 09 29 49 69 11 31 51 71 19 39 59 79
    81 a1 c1 e1 89 a9 c9 e9 91 b1 d1 f1 99 b9 d9 f9
    01 21 41 61 09 29 49 69 11 31 51 71 19 39 59 79
    81 a1 c1 e1 89 a9 c9 e9 91 b1 d1 f1 99 b9 d9 f9
    02 22 42 62 0a 2a 4a 6a 12 32 52 72 1a 3a 5a 7a
    82 a2 c2 e2 8a aa ca ea 92 b2 d2 f2 9a ba da fa
    02 22 42 62 0a 2a 4a 6a 12 32 52 72 1a 3a 5a 7a
    82 a2 c2 e2 8a aa ca ea 92 b2 d2 f2 9a ba da fa
    03 23 43 63 0b 2b 4b 6b 13 33 53 73 1b 3b 5b 7b
    83 a3 c3 e3 8b ab cb eb 93 b3 d3 f3 9b bb db fb
    03 23 43 63 0b 2b 4b 6b 13 33 53 73 1b 3b 5b 7b
    83 a3 c3 e3 8b ab cb eb 93 b3 d3 f3 9b bb db fb
    04 24 44 64 0c 2c 4c 6c 14 34 54 74 1c 3c 5c 7c
    84 a4 c4 e4 8c ac cc ec 94 b4 d4 f4 9c bc dc fc
    04 24 44 64 0c 2c 4c 6c 14 34 54 74 1c 3c 5c 7c
    84 a4 c4 e4 8c ac cc ec 94 b4 d4 f4 9c bc dc fc
    05 25 45 65 0d 2d 4d 6d 15 35 55 75 1d 3d 5d 7d
    85 a5 c5 e5 8d ad cd ed 95 b5 d5 f5 9d bd dd fd
    05 25 45 65 0d 2d 4d 6d 15 35 55 75 1d 3d 5d 7d
    85 a5 c5 e5 8d ad cd ed 95 b5 d5 f5 9d bd dd fd
    06 26 46 66 0e 2e 4e 6e 16 36 56 76 1e 3e 5e 7e
    86 a6 c6 e6 8e ae ce ee 96 b6 d6 f6 9e be de fe
    06 26 46 66 0e 2e 4e 6e 16 36 56 76 1e 3e 5e 7e
    86 a6 c6 e6 8e ae ce ee 96 b6 d6 f6 9e be de fe
    07 27 47 67 0f 2f 4f 6f 17 37 57 77 1f 3f 5f 7f
    87 a7 c7 e7 8f af cf ef 97 b7 d7 f7 9f bf df ff
    07 27 47 67 0f 2f 4f 6f 17 37 57 77 1f 3f 5f 7f
    87 a7 c7 e7 8f af cf ef 97 b7 d7 f7 9f bf df ff

matrix_b [bits=8,m=8,n=32,k=16,layout=col_major]:
  Bmat:
    00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0 00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
    01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1 01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
    02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2 02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
    03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3 03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
    04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4 04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
    05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5 05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
    06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6 06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
    07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7 07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7
    08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8 08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8
    09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9 09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9
    0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa 0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa
    0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb 0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb
    0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc 0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc
    0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd 0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd
    0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe 0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe
    0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff 0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff
  fragmentB:
    00 01 02 03 80 81 82 83 00 01 02 03 80 81 82 83
    04 05 06 07 84 85 86 87 04 05 06 07 84 85 86 87
    08 09 0a 0b 88 89 8a 8b 08 09 0a 0b 88 89 8a 8b
    0c 0d 0e 0f 8c 8d 8e 8f 0c 0d 0e 0f 8c 8d 8e 8f
    10 11 12 13 90 91 92 93 10 11 12 13 90 91 92 93
    14 15 16 17 94 95 96 97 14 15 16 17 94 95 96 97
    18 19 1a 1b 98 99 9a 9b 18 19 1a 1b 98 99 9a 9b
    1c 1d 1e 1f 9c 9d 9e 9f 1c 1d 1e 1f 9c 9d 9e 9f
    20 21 22 23 a0 a1 a2 a3 20 21 22 23 a0 a1 a2 a3
    24 25 26 27 a4 a5 a6 a7 24 25 26 27 a4 a5 a6 a7
    28 29 2a 2b a8 a9 aa ab 28 29 2a 2b a8 a9 aa ab
    2c 2d 2e 2f ac ad ae af 2c 2d 2e 2f ac ad ae af
    30 31 32 33 b0 b1 b2 b3 30 31 32 33 b0 b1 b2 b3
    34 35 36 37 b4 b5 b6 b7 34 35 36 37 b4 b5 b6 b7
    38 39 3a 3b b8 b9 ba bb 38 39 3a 3b b8 b9 ba bb
    3c 3d 3e 3f bc bd be bf 3c 3d 3e 3f bc bd be bf
    40 41 42 43 c0 c1 c2 c3 40 41 42 43 c0 c1 c2 c3
    44 45 46 47 c4 c5 c6 c7 44 45 46 47 c4 c5 c6 c7
    48 49 4a 4b c8 c9 ca cb 48 49 4a 4b c8 c9 ca cb
    4c 4d 4e 4f cc cd ce cf 4c 4d 4e 4f cc cd ce cf
    50 51 52 53 d0 d1 d2 d3 50 51 52 53 d0 d1 d2 d3
    54 55 56 57 d4 d5 d6 d7 54 55 56 57 d4 d5 d6 d7
    58 59 5a 5b d8 d9 da db 58 59 5a 5b d8 d9 da db
    5c 5d 5e 5f dc dd de df 5c 5d 5e 5f dc dd de df
    60 61 62 63 e0 e1 e2 e3 60 61 62 63 e0 e1 e2 e3
    64 65 66 67 e4 e5 e6 e7 64 65 66 67 e4 e5 e6 e7
    68 69 6a 6b e8 e9 ea eb 68 69 6a 6b e8 e9 ea eb
    6c 6d 6e 6f ec ed ee ef 6c 6d 6e 6f ec ed ee ef
    70 71 72 73 f0 f1 f2 f3 70 71 72 73 f0 f1 f2 f3
    74 75 76 77 f4 f5 f6 f7 74 75 76 77 f4 f5 f6 f7
    78 79 7a 7b f8 f9 fa fb 78 79 7a 7b f8 f9 fa fb
    7c 7d 7e 7f fc fd fe ff 7c 7d 7e 7f fc fd fe ff

accumulator [m=8,n=32,k=16]:
  Cmat:
    0000 0001 0002 0003 0004 0005 0006 0007 0008 0009 000a 000b 000c 000d 000e 000f 0010 0011 0012 0013 0014 0015 0016 0017 0018 0019 001a 001b 001c 001d 001e 001f
    0100 0101 0102 0103 0104 0105 0106 0107 0108 0109 010a 010b 010c 010d 010e 010f 0110 0111 0112 0113 0114 0115 0116 0117 0118 0119 011a 011b 011c 011d 011e 011f
    0200 0201 0202 0203 0204 0205 0206 0207 0208 0209 020a 020b 020c 020d 020e 020f 0210 0211 0212 0213 0214 0215 0216 0217 0218 0219 021a 021b 021c 021d 021e 021f
    0300 0301 0302 0303 0304 0305 0306 0307 0308 0309 030a 030b 030c 030d 030e 030f 0310 0311 0312 0313 0314 0315 0316 0317 0318 0319 031a 031b 031c 031d 031e 031f
    0400 0401 0402 0403 0404 0405 0406 0407 0408 0409 040a 040b 040c 040d 040e 040f 0410 0411 0412 0413 0414 0415 0416 0417 0418 0419 041a 041b 041c 041d 041e 041f
    0500 0501 0502 0503 0504 0505 0506 0507 0508 0509 050a 050b 050c 050d 050e 050f 0510 0511 0512 0513 0514 0515 0516 0517 0518 0519 051a 051b 051c 051d 051e 051f
    0600 0601 0602 0603 0604 0605 0606 0607 0608 0609 060a 060b 060c 060d 060e 060f 0610 0611 0612 0613 0614 0615 0616 0617 0618 0619 061a 061b 061c 061d 061e 061f
    0700 0701 0702 0703 0704 0705 0706 0707 0708 0709 070a 070b 070c 070d 070e 070f 0710 0711 0712 0713 0714 0715 0716 0717 0718 0719 071a 071b 071c 071d 071e 071f
  fragmentC:
    0000 0001 0008 0009 0010 0011 0018 0019
    0002 0003 000a 000b 0012 0013 001a 001b
    0004 0005 000c 000d 0014 0015 001c 001d
    0006 0007 000e 000f 0016 0017 001e 001f
    0100 0101 0108 0109 0110 0111 0118 0119
    0102 0103 010a 010b 0112 0113 011a 011b
    0104 0105 010c 010d 0114 0115 011c 011d
    0106 0107 010e 010f 0116 0117 011e 011f
    0200 0201 0208 0209 0210 0211 0218 0219
    0202 0203 020a 020b 0212 0213 021a 021b
    0204 0205 020c 020d 0214 0215 021c 021d
    0206 0207 020e 020f 0216 0217 021e 021f
    0300 0301 0308 0309 0310 0311 0318 0319
    0302 0303 030a 030b 0312 0313 031a 031b
    0304 0305 030c 030d 0314 0315 031c 031d
    0306 0307 030e 030f 0316 0317 031e 031f
    0400 0401 0408 0409 0410 0411 0418 0419
    0402 0403 040a 040b 0412 0413 041a 041b
    0404 0405 040c 040d 0414 0415 041c 041d
    0406 0407 040e 040f 0416 0417 041e 041f
    0500 0501 0508 0509 0510 0511 0518 0519
    0502 0503 050a 050b 0512 0513 051a 051b
    0504 0505 050c 050d 0514 0515 051c 051d
    0506 0507 050e 050f 0516 0517 051e 051f
    0600 0601 0608 0609 0610 0611 0618 0619
    0602 0603 060a 060b 0612 0613 061a 061b
    0604 0605 060c 060d 0614 0615 061c 061d
    0606 0607 060e 060f 0616 0617 061e 061f
    0700 0701 0708 0709 0710 0711 0718 0719
    0702 0703 070a 070b 0712 0713 071a 071b
    0704 0705 070c 070d 0714 0715 071c 071d
    0706 0707 070e 070f 0716 0717 071e 071f
Done.
```

### A40 (on Sky):

```
fragment2.cuda

matrix_a [bits=4,m=8,n=8,k=32]:
  Amat:
    00 08 10 18
    20 28 30 38
    40 48 50 58
    60 68 70 78
    80 88 90 98
    a0 a8 b0 b8
    c0 c8 d0 d8
    e0 e8 f0 f8
  fragmentA:
    00
    08
    10
    18
    20
    28
    30
    38
    40
    48
    50
    58
    60
    68
    70
    78
    80
    88
    90
    98
    a0
    a8
    b0
    b8
    c0
    c8
    d0
    d8
    e0
    e8
    f0
    f8

matrix_b [bits=4,m=8,n=8,k=32]:
  Bmat:
    00 20 40 60 80 a0 c0 e0
    01 21 41 61 81 a1 c1 e1
    02 22 42 62 82 a2 c2 e2
    03 23 43 63 83 a3 c3 e3
  fragmentB:
    00
    01
    02
    03
    20
    21
    22
    23
    40
    41
    42
    43
    60
    61
    62
    63
    80
    81
    82
    83
    a0
    a1
    a2
    a3
    c0
    c1
    c2
    c3
    e0
    e1
    e2
    e3

accumulator [m=8,n=8,k=32]:
  Cmat:
    0000 0001 0002 0003 0004 0005 0006 0007
    0100 0101 0102 0103 0104 0105 0106 0107
    0200 0201 0202 0203 0204 0205 0206 0207
    0300 0301 0302 0303 0304 0305 0306 0307
    0400 0401 0402 0403 0404 0405 0406 0407
    0500 0501 0502 0503 0504 0505 0506 0507
    0600 0601 0602 0603 0604 0605 0606 0607
    0700 0701 0702 0703 0704 0705 0706 0707
  fragmentC:
    0000 0001
    0002 0003
    0004 0005
    0006 0007
    0100 0101
    0102 0103
    0104 0105
    0106 0107
    0200 0201
    0202 0203
    0204 0205
    0206 0207
    0300 0301
    0302 0303
    0304 0305
    0306 0307
    0400 0401
    0402 0403
    0404 0405
    0406 0407
    0500 0501
    0502 0503
    0504 0505
    0506 0507
    0600 0601
    0602 0603
    0604 0605
    0606 0607
    0700 0701
    0702 0703
    0704 0705
    0706 0707

matrix_a [bits=8,m=16,n=16,k=16,layout=row_major]:
  Amat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f
    90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf
    d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef
    f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
  fragmentA:
    00 01 02 03 80 81 82 83
    04 05 06 07 84 85 86 87
    08 09 0a 0b 88 89 8a 8b
    0c 0d 0e 0f 8c 8d 8e 8f
    10 11 12 13 90 91 92 93
    14 15 16 17 94 95 96 97
    18 19 1a 1b 98 99 9a 9b
    1c 1d 1e 1f 9c 9d 9e 9f
    20 21 22 23 a0 a1 a2 a3
    24 25 26 27 a4 a5 a6 a7
    28 29 2a 2b a8 a9 aa ab
    2c 2d 2e 2f ac ad ae af
    30 31 32 33 b0 b1 b2 b3
    34 35 36 37 b4 b5 b6 b7
    38 39 3a 3b b8 b9 ba bb
    3c 3d 3e 3f bc bd be bf
    40 41 42 43 c0 c1 c2 c3
    44 45 46 47 c4 c5 c6 c7
    48 49 4a 4b c8 c9 ca cb
    4c 4d 4e 4f cc cd ce cf
    50 51 52 53 d0 d1 d2 d3
    54 55 56 57 d4 d5 d6 d7
    58 59 5a 5b d8 d9 da db
    5c 5d 5e 5f dc dd de df
    60 61 62 63 e0 e1 e2 e3
    64 65 66 67 e4 e5 e6 e7
    68 69 6a 6b e8 e9 ea eb
    6c 6d 6e 6f ec ed ee ef
    70 71 72 73 f0 f1 f2 f3
    74 75 76 77 f4 f5 f6 f7
    78 79 7a 7b f8 f9 fa fb
    7c 7d 7e 7f fc fd fe ff

matrix_a [bits=8,m=16,n=16,k=16,layout=col_major]:
  Amat:
    00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
    01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
    02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
    03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
    04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
    05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
    06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
    07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7
    08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8
    09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9
    0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa
    0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb
    0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc
    0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd
    0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe
    0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff
  fragmentA:
    00 10 20 30 08 18 28 38
    40 50 60 70 48 58 68 78
    80 90 a0 b0 88 98 a8 b8
    c0 d0 e0 f0 c8 d8 e8 f8
    01 11 21 31 09 19 29 39
    41 51 61 71 49 59 69 79
    81 91 a1 b1 89 99 a9 b9
    c1 d1 e1 f1 c9 d9 e9 f9
    02 12 22 32 0a 1a 2a 3a
    42 52 62 72 4a 5a 6a 7a
    82 92 a2 b2 8a 9a aa ba
    c2 d2 e2 f2 ca da ea fa
    03 13 23 33 0b 1b 2b 3b
    43 53 63 73 4b 5b 6b 7b
    83 93 a3 b3 8b 9b ab bb
    c3 d3 e3 f3 cb db eb fb
    04 14 24 34 0c 1c 2c 3c
    44 54 64 74 4c 5c 6c 7c
    84 94 a4 b4 8c 9c ac bc
    c4 d4 e4 f4 cc dc ec fc
    05 15 25 35 0d 1d 2d 3d
    45 55 65 75 4d 5d 6d 7d
    85 95 a5 b5 8d 9d ad bd
    c5 d5 e5 f5 cd dd ed fd
    06 16 26 36 0e 1e 2e 3e
    46 56 66 76 4e 5e 6e 7e
    86 96 a6 b6 8e 9e ae be
    c6 d6 e6 f6 ce de ee fe
    07 17 27 37 0f 1f 2f 3f
    47 57 67 77 4f 5f 6f 7f
    87 97 a7 b7 8f 9f af bf
    c7 d7 e7 f7 cf df ef ff

matrix_b [bits=8,m=16,n=16,k=16,layout=row_major]:
  Bmat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f
    90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf
    d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef
    f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
  fragmentB:
    00 10 20 30 08 18 28 38
    40 50 60 70 48 58 68 78
    80 90 a0 b0 88 98 a8 b8
    c0 d0 e0 f0 c8 d8 e8 f8
    01 11 21 31 09 19 29 39
    41 51 61 71 49 59 69 79
    81 91 a1 b1 89 99 a9 b9
    c1 d1 e1 f1 c9 d9 e9 f9
    02 12 22 32 0a 1a 2a 3a
    42 52 62 72 4a 5a 6a 7a
    82 92 a2 b2 8a 9a aa ba
    c2 d2 e2 f2 ca da ea fa
    03 13 23 33 0b 1b 2b 3b
    43 53 63 73 4b 5b 6b 7b
    83 93 a3 b3 8b 9b ab bb
    c3 d3 e3 f3 cb db eb fb
    04 14 24 34 0c 1c 2c 3c
    44 54 64 74 4c 5c 6c 7c
    84 94 a4 b4 8c 9c ac bc
    c4 d4 e4 f4 cc dc ec fc
    05 15 25 35 0d 1d 2d 3d
    45 55 65 75 4d 5d 6d 7d
    85 95 a5 b5 8d 9d ad bd
    c5 d5 e5 f5 cd dd ed fd
    06 16 26 36 0e 1e 2e 3e
    46 56 66 76 4e 5e 6e 7e
    86 96 a6 b6 8e 9e ae be
    c6 d6 e6 f6 ce de ee fe
    07 17 27 37 0f 1f 2f 3f
    47 57 67 77 4f 5f 6f 7f
    87 97 a7 b7 8f 9f af bf
    c7 d7 e7 f7 cf df ef ff

matrix_b [bits=8,m=16,n=16,k=16,layout=col_major]:
  Bmat:
    00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
    01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
    02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
    03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
    04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
    05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
    06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
    07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7
    08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8
    09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9
    0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa
    0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb
    0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc
    0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd
    0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe
    0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff
  fragmentB:
    00 01 02 03 80 81 82 83
    04 05 06 07 84 85 86 87
    08 09 0a 0b 88 89 8a 8b
    0c 0d 0e 0f 8c 8d 8e 8f
    10 11 12 13 90 91 92 93
    14 15 16 17 94 95 96 97
    18 19 1a 1b 98 99 9a 9b
    1c 1d 1e 1f 9c 9d 9e 9f
    20 21 22 23 a0 a1 a2 a3
    24 25 26 27 a4 a5 a6 a7
    28 29 2a 2b a8 a9 aa ab
    2c 2d 2e 2f ac ad ae af
    30 31 32 33 b0 b1 b2 b3
    34 35 36 37 b4 b5 b6 b7
    38 39 3a 3b b8 b9 ba bb
    3c 3d 3e 3f bc bd be bf
    40 41 42 43 c0 c1 c2 c3
    44 45 46 47 c4 c5 c6 c7
    48 49 4a 4b c8 c9 ca cb
    4c 4d 4e 4f cc cd ce cf
    50 51 52 53 d0 d1 d2 d3
    54 55 56 57 d4 d5 d6 d7
    58 59 5a 5b d8 d9 da db
    5c 5d 5e 5f dc dd de df
    60 61 62 63 e0 e1 e2 e3
    64 65 66 67 e4 e5 e6 e7
    68 69 6a 6b e8 e9 ea eb
    6c 6d 6e 6f ec ed ee ef
    70 71 72 73 f0 f1 f2 f3
    74 75 76 77 f4 f5 f6 f7
    78 79 7a 7b f8 f9 fa fb
    7c 7d 7e 7f fc fd fe ff

accumulator [m=16,n=16,k=16]:
  Cmat:
    0000 0001 0002 0003 0004 0005 0006 0007 0008 0009 000a 000b 000c 000d 000e 000f
    0100 0101 0102 0103 0104 0105 0106 0107 0108 0109 010a 010b 010c 010d 010e 010f
    0200 0201 0202 0203 0204 0205 0206 0207 0208 0209 020a 020b 020c 020d 020e 020f
    0300 0301 0302 0303 0304 0305 0306 0307 0308 0309 030a 030b 030c 030d 030e 030f
    0400 0401 0402 0403 0404 0405 0406 0407 0408 0409 040a 040b 040c 040d 040e 040f
    0500 0501 0502 0503 0504 0505 0506 0507 0508 0509 050a 050b 050c 050d 050e 050f
    0600 0601 0602 0603 0604 0605 0606 0607 0608 0609 060a 060b 060c 060d 060e 060f
    0700 0701 0702 0703 0704 0705 0706 0707 0708 0709 070a 070b 070c 070d 070e 070f
    0800 0801 0802 0803 0804 0805 0806 0807 0808 0809 080a 080b 080c 080d 080e 080f
    0900 0901 0902 0903 0904 0905 0906 0907 0908 0909 090a 090b 090c 090d 090e 090f
    0a00 0a01 0a02 0a03 0a04 0a05 0a06 0a07 0a08 0a09 0a0a 0a0b 0a0c 0a0d 0a0e 0a0f
    0b00 0b01 0b02 0b03 0b04 0b05 0b06 0b07 0b08 0b09 0b0a 0b0b 0b0c 0b0d 0b0e 0b0f
    0c00 0c01 0c02 0c03 0c04 0c05 0c06 0c07 0c08 0c09 0c0a 0c0b 0c0c 0c0d 0c0e 0c0f
    0d00 0d01 0d02 0d03 0d04 0d05 0d06 0d07 0d08 0d09 0d0a 0d0b 0d0c 0d0d 0d0e 0d0f
    0e00 0e01 0e02 0e03 0e04 0e05 0e06 0e07 0e08 0e09 0e0a 0e0b 0e0c 0e0d 0e0e 0e0f
    0f00 0f01 0f02 0f03 0f04 0f05 0f06 0f07 0f08 0f09 0f0a 0f0b 0f0c 0f0d 0f0e 0f0f
  fragmentC:
    0000 0001 0800 0801 0008 0009 0808 0809
    0002 0003 0802 0803 000a 000b 080a 080b
    0004 0005 0804 0805 000c 000d 080c 080d
    0006 0007 0806 0807 000e 000f 080e 080f
    0100 0101 0900 0901 0108 0109 0908 0909
    0102 0103 0902 0903 010a 010b 090a 090b
    0104 0105 0904 0905 010c 010d 090c 090d
    0106 0107 0906 0907 010e 010f 090e 090f
    0200 0201 0a00 0a01 0208 0209 0a08 0a09
    0202 0203 0a02 0a03 020a 020b 0a0a 0a0b
    0204 0205 0a04 0a05 020c 020d 0a0c 0a0d
    0206 0207 0a06 0a07 020e 020f 0a0e 0a0f
    0300 0301 0b00 0b01 0308 0309 0b08 0b09
    0302 0303 0b02 0b03 030a 030b 0b0a 0b0b
    0304 0305 0b04 0b05 030c 030d 0b0c 0b0d
    0306 0307 0b06 0b07 030e 030f 0b0e 0b0f
    0400 0401 0c00 0c01 0408 0409 0c08 0c09
    0402 0403 0c02 0c03 040a 040b 0c0a 0c0b
    0404 0405 0c04 0c05 040c 040d 0c0c 0c0d
    0406 0407 0c06 0c07 040e 040f 0c0e 0c0f
    0500 0501 0d00 0d01 0508 0509 0d08 0d09
    0502 0503 0d02 0d03 050a 050b 0d0a 0d0b
    0504 0505 0d04 0d05 050c 050d 0d0c 0d0d
    0506 0507 0d06 0d07 050e 050f 0d0e 0d0f
    0600 0601 0e00 0e01 0608 0609 0e08 0e09
    0602 0603 0e02 0e03 060a 060b 0e0a 0e0b
    0604 0605 0e04 0e05 060c 060d 0e0c 0e0d
    0606 0607 0e06 0e07 060e 060f 0e0e 0e0f
    0700 0701 0f00 0f01 0708 0709 0f08 0f09
    0702 0703 0f02 0f03 070a 070b 0f0a 0f0b
    0704 0705 0f04 0f05 070c 070d 0f0c 0f0d
    0706 0707 0f06 0f07 070e 070f 0f0e 0f0f

matrix_a [bits=8,m=32,n=8,k=16,layout=row_major]:
  Amat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f
    90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf
    d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef
    f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f
    90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf
    d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef
    f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
  fragmentA:
    00 01 02 03 80 81 82 83 00 01 02 03 80 81 82 83
    04 05 06 07 84 85 86 87 04 05 06 07 84 85 86 87
    08 09 0a 0b 88 89 8a 8b 08 09 0a 0b 88 89 8a 8b
    0c 0d 0e 0f 8c 8d 8e 8f 0c 0d 0e 0f 8c 8d 8e 8f
    10 11 12 13 90 91 92 93 10 11 12 13 90 91 92 93
    14 15 16 17 94 95 96 97 14 15 16 17 94 95 96 97
    18 19 1a 1b 98 99 9a 9b 18 19 1a 1b 98 99 9a 9b
    1c 1d 1e 1f 9c 9d 9e 9f 1c 1d 1e 1f 9c 9d 9e 9f
    20 21 22 23 a0 a1 a2 a3 20 21 22 23 a0 a1 a2 a3
    24 25 26 27 a4 a5 a6 a7 24 25 26 27 a4 a5 a6 a7
    28 29 2a 2b a8 a9 aa ab 28 29 2a 2b a8 a9 aa ab
    2c 2d 2e 2f ac ad ae af 2c 2d 2e 2f ac ad ae af
    30 31 32 33 b0 b1 b2 b3 30 31 32 33 b0 b1 b2 b3
    34 35 36 37 b4 b5 b6 b7 34 35 36 37 b4 b5 b6 b7
    38 39 3a 3b b8 b9 ba bb 38 39 3a 3b b8 b9 ba bb
    3c 3d 3e 3f bc bd be bf 3c 3d 3e 3f bc bd be bf
    40 41 42 43 c0 c1 c2 c3 40 41 42 43 c0 c1 c2 c3
    44 45 46 47 c4 c5 c6 c7 44 45 46 47 c4 c5 c6 c7
    48 49 4a 4b c8 c9 ca cb 48 49 4a 4b c8 c9 ca cb
    4c 4d 4e 4f cc cd ce cf 4c 4d 4e 4f cc cd ce cf
    50 51 52 53 d0 d1 d2 d3 50 51 52 53 d0 d1 d2 d3
    54 55 56 57 d4 d5 d6 d7 54 55 56 57 d4 d5 d6 d7
    58 59 5a 5b d8 d9 da db 58 59 5a 5b d8 d9 da db
    5c 5d 5e 5f dc dd de df 5c 5d 5e 5f dc dd de df
    60 61 62 63 e0 e1 e2 e3 60 61 62 63 e0 e1 e2 e3
    64 65 66 67 e4 e5 e6 e7 64 65 66 67 e4 e5 e6 e7
    68 69 6a 6b e8 e9 ea eb 68 69 6a 6b e8 e9 ea eb
    6c 6d 6e 6f ec ed ee ef 6c 6d 6e 6f ec ed ee ef
    70 71 72 73 f0 f1 f2 f3 70 71 72 73 f0 f1 f2 f3
    74 75 76 77 f4 f5 f6 f7 74 75 76 77 f4 f5 f6 f7
    78 79 7a 7b f8 f9 fa fb 78 79 7a 7b f8 f9 fa fb
    7c 7d 7e 7f fc fd fe ff 7c 7d 7e 7f fc fd fe ff

matrix_a [bits=8,m=32,n=8,k=16,layout=col_major]:
  Amat:
    00 20 40 60 80 a0 c0 e0 00 20 40 60 80 a0 c0 e0
    01 21 41 61 81 a1 c1 e1 01 21 41 61 81 a1 c1 e1
    02 22 42 62 82 a2 c2 e2 02 22 42 62 82 a2 c2 e2
    03 23 43 63 83 a3 c3 e3 03 23 43 63 83 a3 c3 e3
    04 24 44 64 84 a4 c4 e4 04 24 44 64 84 a4 c4 e4
    05 25 45 65 85 a5 c5 e5 05 25 45 65 85 a5 c5 e5
    06 26 46 66 86 a6 c6 e6 06 26 46 66 86 a6 c6 e6
    07 27 47 67 87 a7 c7 e7 07 27 47 67 87 a7 c7 e7
    08 28 48 68 88 a8 c8 e8 08 28 48 68 88 a8 c8 e8
    09 29 49 69 89 a9 c9 e9 09 29 49 69 89 a9 c9 e9
    0a 2a 4a 6a 8a aa ca ea 0a 2a 4a 6a 8a aa ca ea
    0b 2b 4b 6b 8b ab cb eb 0b 2b 4b 6b 8b ab cb eb
    0c 2c 4c 6c 8c ac cc ec 0c 2c 4c 6c 8c ac cc ec
    0d 2d 4d 6d 8d ad cd ed 0d 2d 4d 6d 8d ad cd ed
    0e 2e 4e 6e 8e ae ce ee 0e 2e 4e 6e 8e ae ce ee
    0f 2f 4f 6f 8f af cf ef 0f 2f 4f 6f 8f af cf ef
    10 30 50 70 90 b0 d0 f0 10 30 50 70 90 b0 d0 f0
    11 31 51 71 91 b1 d1 f1 11 31 51 71 91 b1 d1 f1
    12 32 52 72 92 b2 d2 f2 12 32 52 72 92 b2 d2 f2
    13 33 53 73 93 b3 d3 f3 13 33 53 73 93 b3 d3 f3
    14 34 54 74 94 b4 d4 f4 14 34 54 74 94 b4 d4 f4
    15 35 55 75 95 b5 d5 f5 15 35 55 75 95 b5 d5 f5
    16 36 56 76 96 b6 d6 f6 16 36 56 76 96 b6 d6 f6
    17 37 57 77 97 b7 d7 f7 17 37 57 77 97 b7 d7 f7
    18 38 58 78 98 b8 d8 f8 18 38 58 78 98 b8 d8 f8
    19 39 59 79 99 b9 d9 f9 19 39 59 79 99 b9 d9 f9
    1a 3a 5a 7a 9a ba da fa 1a 3a 5a 7a 9a ba da fa
    1b 3b 5b 7b 9b bb db fb 1b 3b 5b 7b 9b bb db fb
    1c 3c 5c 7c 9c bc dc fc 1c 3c 5c 7c 9c bc dc fc
    1d 3d 5d 7d 9d bd dd fd 1d 3d 5d 7d 9d bd dd fd
    1e 3e 5e 7e 9e be de fe 1e 3e 5e 7e 9e be de fe
    1f 3f 5f 7f 9f bf df ff 1f 3f 5f 7f 9f bf df ff
  fragmentA:
    00 20 40 60 08 28 48 68 10 30 50 70 18 38 58 78
    80 a0 c0 e0 88 a8 c8 e8 90 b0 d0 f0 98 b8 d8 f8
    00 20 40 60 08 28 48 68 10 30 50 70 18 38 58 78
    80 a0 c0 e0 88 a8 c8 e8 90 b0 d0 f0 98 b8 d8 f8
    01 21 41 61 09 29 49 69 11 31 51 71 19 39 59 79
    81 a1 c1 e1 89 a9 c9 e9 91 b1 d1 f1 99 b9 d9 f9
    01 21 41 61 09 29 49 69 11 31 51 71 19 39 59 79
    81 a1 c1 e1 89 a9 c9 e9 91 b1 d1 f1 99 b9 d9 f9
    02 22 42 62 0a 2a 4a 6a 12 32 52 72 1a 3a 5a 7a
    82 a2 c2 e2 8a aa ca ea 92 b2 d2 f2 9a ba da fa
    02 22 42 62 0a 2a 4a 6a 12 32 52 72 1a 3a 5a 7a
    82 a2 c2 e2 8a aa ca ea 92 b2 d2 f2 9a ba da fa
    03 23 43 63 0b 2b 4b 6b 13 33 53 73 1b 3b 5b 7b
    83 a3 c3 e3 8b ab cb eb 93 b3 d3 f3 9b bb db fb
    03 23 43 63 0b 2b 4b 6b 13 33 53 73 1b 3b 5b 7b
    83 a3 c3 e3 8b ab cb eb 93 b3 d3 f3 9b bb db fb
    04 24 44 64 0c 2c 4c 6c 14 34 54 74 1c 3c 5c 7c
    84 a4 c4 e4 8c ac cc ec 94 b4 d4 f4 9c bc dc fc
    04 24 44 64 0c 2c 4c 6c 14 34 54 74 1c 3c 5c 7c
    84 a4 c4 e4 8c ac cc ec 94 b4 d4 f4 9c bc dc fc
    05 25 45 65 0d 2d 4d 6d 15 35 55 75 1d 3d 5d 7d
    85 a5 c5 e5 8d ad cd ed 95 b5 d5 f5 9d bd dd fd
    05 25 45 65 0d 2d 4d 6d 15 35 55 75 1d 3d 5d 7d
    85 a5 c5 e5 8d ad cd ed 95 b5 d5 f5 9d bd dd fd
    06 26 46 66 0e 2e 4e 6e 16 36 56 76 1e 3e 5e 7e
    86 a6 c6 e6 8e ae ce ee 96 b6 d6 f6 9e be de fe
    06 26 46 66 0e 2e 4e 6e 16 36 56 76 1e 3e 5e 7e
    86 a6 c6 e6 8e ae ce ee 96 b6 d6 f6 9e be de fe
    07 27 47 67 0f 2f 4f 6f 17 37 57 77 1f 3f 5f 7f
    87 a7 c7 e7 8f af cf ef 97 b7 d7 f7 9f bf df ff
    07 27 47 67 0f 2f 4f 6f 17 37 57 77 1f 3f 5f 7f
    87 a7 c7 e7 8f af cf ef 97 b7 d7 f7 9f bf df ff

matrix_b [bits=8,m=32,n=8,k=16,layout=row_major]:
  Bmat:
    00 01 02 03 04 05 06 07
    08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17
    18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27
    28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37
    38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47
    48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57
    58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67
    68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77
    78 79 7a 7b 7c 7d 7e 7f
  fragmentB:
    00 08 10 18
    20 28 30 38
    40 48 50 58
    60 68 70 78
    01 09 11 19
    21 29 31 39
    41 49 51 59
    61 69 71 79
    02 0a 12 1a
    22 2a 32 3a
    42 4a 52 5a
    62 6a 72 7a
    03 0b 13 1b
    23 2b 33 3b
    43 4b 53 5b
    63 6b 73 7b
    04 0c 14 1c
    24 2c 34 3c
    44 4c 54 5c
    64 6c 74 7c
    05 0d 15 1d
    25 2d 35 3d
    45 4d 55 5d
    65 6d 75 7d
    06 0e 16 1e
    26 2e 36 3e
    46 4e 56 5e
    66 6e 76 7e
    07 0f 17 1f
    27 2f 37 3f
    47 4f 57 5f
    67 6f 77 7f

matrix_b [bits=8,m=32,n=8,k=16,layout=col_major]:
  Bmat:
    00 10 20 30 40 50 60 70
    01 11 21 31 41 51 61 71
    02 12 22 32 42 52 62 72
    03 13 23 33 43 53 63 73
    04 14 24 34 44 54 64 74
    05 15 25 35 45 55 65 75
    06 16 26 36 46 56 66 76
    07 17 27 37 47 57 67 77
    08 18 28 38 48 58 68 78
    09 19 29 39 49 59 69 79
    0a 1a 2a 3a 4a 5a 6a 7a
    0b 1b 2b 3b 4b 5b 6b 7b
    0c 1c 2c 3c 4c 5c 6c 7c
    0d 1d 2d 3d 4d 5d 6d 7d
    0e 1e 2e 3e 4e 5e 6e 7e
    0f 1f 2f 3f 4f 5f 6f 7f
  fragmentB:
    00 01 02 03
    04 05 06 07
    08 09 0a 0b
    0c 0d 0e 0f
    10 11 12 13
    14 15 16 17
    18 19 1a 1b
    1c 1d 1e 1f
    20 21 22 23
    24 25 26 27
    28 29 2a 2b
    2c 2d 2e 2f
    30 31 32 33
    34 35 36 37
    38 39 3a 3b
    3c 3d 3e 3f
    40 41 42 43
    44 45 46 47
    48 49 4a 4b
    4c 4d 4e 4f
    50 51 52 53
    54 55 56 57
    58 59 5a 5b
    5c 5d 5e 5f
    60 61 62 63
    64 65 66 67
    68 69 6a 6b
    6c 6d 6e 6f
    70 71 72 73
    74 75 76 77
    78 79 7a 7b
    7c 7d 7e 7f

accumulator [m=32,n=8,k=16]:
  Cmat:
    0000 0001 0002 0003 0004 0005 0006 0007
    0100 0101 0102 0103 0104 0105 0106 0107
    0200 0201 0202 0203 0204 0205 0206 0207
    0300 0301 0302 0303 0304 0305 0306 0307
    0400 0401 0402 0403 0404 0405 0406 0407
    0500 0501 0502 0503 0504 0505 0506 0507
    0600 0601 0602 0603 0604 0605 0606 0607
    0700 0701 0702 0703 0704 0705 0706 0707
    0800 0801 0802 0803 0804 0805 0806 0807
    0900 0901 0902 0903 0904 0905 0906 0907
    0a00 0a01 0a02 0a03 0a04 0a05 0a06 0a07
    0b00 0b01 0b02 0b03 0b04 0b05 0b06 0b07
    0c00 0c01 0c02 0c03 0c04 0c05 0c06 0c07
    0d00 0d01 0d02 0d03 0d04 0d05 0d06 0d07
    0e00 0e01 0e02 0e03 0e04 0e05 0e06 0e07
    0f00 0f01 0f02 0f03 0f04 0f05 0f06 0f07
    1000 1001 1002 1003 1004 1005 1006 1007
    1100 1101 1102 1103 1104 1105 1106 1107
    1200 1201 1202 1203 1204 1205 1206 1207
    1300 1301 1302 1303 1304 1305 1306 1307
    1400 1401 1402 1403 1404 1405 1406 1407
    1500 1501 1502 1503 1504 1505 1506 1507
    1600 1601 1602 1603 1604 1605 1606 1607
    1700 1701 1702 1703 1704 1705 1706 1707
    1800 1801 1802 1803 1804 1805 1806 1807
    1900 1901 1902 1903 1904 1905 1906 1907
    1a00 1a01 1a02 1a03 1a04 1a05 1a06 1a07
    1b00 1b01 1b02 1b03 1b04 1b05 1b06 1b07
    1c00 1c01 1c02 1c03 1c04 1c05 1c06 1c07
    1d00 1d01 1d02 1d03 1d04 1d05 1d06 1d07
    1e00 1e01 1e02 1e03 1e04 1e05 1e06 1e07
    1f00 1f01 1f02 1f03 1f04 1f05 1f06 1f07
  fragmentC:
    0000 0001 0800 0801 1000 1001 1800 1801
    0002 0003 0802 0803 1002 1003 1802 1803
    0004 0005 0804 0805 1004 1005 1804 1805
    0006 0007 0806 0807 1006 1007 1806 1807
    0100 0101 0900 0901 1100 1101 1900 1901
    0102 0103 0902 0903 1102 1103 1902 1903
    0104 0105 0904 0905 1104 1105 1904 1905
    0106 0107 0906 0907 1106 1107 1906 1907
    0200 0201 0a00 0a01 1200 1201 1a00 1a01
    0202 0203 0a02 0a03 1202 1203 1a02 1a03
    0204 0205 0a04 0a05 1204 1205 1a04 1a05
    0206 0207 0a06 0a07 1206 1207 1a06 1a07
    0300 0301 0b00 0b01 1300 1301 1b00 1b01
    0302 0303 0b02 0b03 1302 1303 1b02 1b03
    0304 0305 0b04 0b05 1304 1305 1b04 1b05
    0306 0307 0b06 0b07 1306 1307 1b06 1b07
    0400 0401 0c00 0c01 1400 1401 1c00 1c01
    0402 0403 0c02 0c03 1402 1403 1c02 1c03
    0404 0405 0c04 0c05 1404 1405 1c04 1c05
    0406 0407 0c06 0c07 1406 1407 1c06 1c07
    0500 0501 0d00 0d01 1500 1501 1d00 1d01
    0502 0503 0d02 0d03 1502 1503 1d02 1d03
    0504 0505 0d04 0d05 1504 1505 1d04 1d05
    0506 0507 0d06 0d07 1506 1507 1d06 1d07
    0600 0601 0e00 0e01 1600 1601 1e00 1e01
    0602 0603 0e02 0e03 1602 1603 1e02 1e03
    0604 0605 0e04 0e05 1604 1605 1e04 1e05
    0606 0607 0e06 0e07 1606 1607 1e06 1e07
    0700 0701 0f00 0f01 1700 1701 1f00 1f01
    0702 0703 0f02 0f03 1702 1703 1f02 1f03
    0704 0705 0f04 0f05 1704 1705 1f04 1f05
    0706 0707 0f06 0f07 1706 1707 1f06 1f07

matrix_a [bits=8,m=8,n=32,k=16,layout=row_major]:
  Amat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
    50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
    70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
  fragmentA:
    00 01 02 03
    04 05 06 07
    08 09 0a 0b
    0c 0d 0e 0f
    10 11 12 13
    14 15 16 17
    18 19 1a 1b
    1c 1d 1e 1f
    20 21 22 23
    24 25 26 27
    28 29 2a 2b
    2c 2d 2e 2f
    30 31 32 33
    34 35 36 37
    38 39 3a 3b
    3c 3d 3e 3f
    40 41 42 43
    44 45 46 47
    48 49 4a 4b
    4c 4d 4e 4f
    50 51 52 53
    54 55 56 57
    58 59 5a 5b
    5c 5d 5e 5f
    60 61 62 63
    64 65 66 67
    68 69 6a 6b
    6c 6d 6e 6f
    70 71 72 73
    74 75 76 77
    78 79 7a 7b
    7c 7d 7e 7f

matrix_a [bits=8,m=8,n=32,k=16,layout=col_major]:
  Amat:
    00 08 10 18 20 28 30 38 40 48 50 58 60 68 70 78
    01 09 11 19 21 29 31 39 41 49 51 59 61 69 71 79
    02 0a 12 1a 22 2a 32 3a 42 4a 52 5a 62 6a 72 7a
    03 0b 13 1b 23 2b 33 3b 43 4b 53 5b 63 6b 73 7b
    04 0c 14 1c 24 2c 34 3c 44 4c 54 5c 64 6c 74 7c
    05 0d 15 1d 25 2d 35 3d 45 4d 55 5d 65 6d 75 7d
    06 0e 16 1e 26 2e 36 3e 46 4e 56 5e 66 6e 76 7e
    07 0f 17 1f 27 2f 37 3f 47 4f 57 5f 67 6f 77 7f
  fragmentA:
    00 08 10 18
    20 28 30 38
    40 48 50 58
    60 68 70 78
    01 09 11 19
    21 29 31 39
    41 49 51 59
    61 69 71 79
    02 0a 12 1a
    22 2a 32 3a
    42 4a 52 5a
    62 6a 72 7a
    03 0b 13 1b
    23 2b 33 3b
    43 4b 53 5b
    63 6b 73 7b
    04 0c 14 1c
    24 2c 34 3c
    44 4c 54 5c
    64 6c 74 7c
    05 0d 15 1d
    25 2d 35 3d
    45 4d 55 5d
    65 6d 75 7d
    06 0e 16 1e
    26 2e 36 3e
    46 4e 56 5e
    66 6e 76 7e
    07 0f 17 1f
    27 2f 37 3f
    47 4f 57 5f
    67 6f 77 7f

matrix_b [bits=8,m=8,n=32,k=16,layout=row_major]:
  Bmat:
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f 70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f 90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f 70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f 90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
  fragmentB:
    00 20 40 60 08 28 48 68 10 30 50 70 18 38 58 78
    80 a0 c0 e0 88 a8 c8 e8 90 b0 d0 f0 98 b8 d8 f8
    00 20 40 60 08 28 48 68 10 30 50 70 18 38 58 78
    80 a0 c0 e0 88 a8 c8 e8 90 b0 d0 f0 98 b8 d8 f8
    01 21 41 61 09 29 49 69 11 31 51 71 19 39 59 79
    81 a1 c1 e1 89 a9 c9 e9 91 b1 d1 f1 99 b9 d9 f9
    01 21 41 61 09 29 49 69 11 31 51 71 19 39 59 79
    81 a1 c1 e1 89 a9 c9 e9 91 b1 d1 f1 99 b9 d9 f9
    02 22 42 62 0a 2a 4a 6a 12 32 52 72 1a 3a 5a 7a
    82 a2 c2 e2 8a aa ca ea 92 b2 d2 f2 9a ba da fa
    02 22 42 62 0a 2a 4a 6a 12 32 52 72 1a 3a 5a 7a
    82 a2 c2 e2 8a aa ca ea 92 b2 d2 f2 9a ba da fa
    03 23 43 63 0b 2b 4b 6b 13 33 53 73 1b 3b 5b 7b
    83 a3 c3 e3 8b ab cb eb 93 b3 d3 f3 9b bb db fb
    03 23 43 63 0b 2b 4b 6b 13 33 53 73 1b 3b 5b 7b
    83 a3 c3 e3 8b ab cb eb 93 b3 d3 f3 9b bb db fb
    04 24 44 64 0c 2c 4c 6c 14 34 54 74 1c 3c 5c 7c
    84 a4 c4 e4 8c ac cc ec 94 b4 d4 f4 9c bc dc fc
    04 24 44 64 0c 2c 4c 6c 14 34 54 74 1c 3c 5c 7c
    84 a4 c4 e4 8c ac cc ec 94 b4 d4 f4 9c bc dc fc
    05 25 45 65 0d 2d 4d 6d 15 35 55 75 1d 3d 5d 7d
    85 a5 c5 e5 8d ad cd ed 95 b5 d5 f5 9d bd dd fd
    05 25 45 65 0d 2d 4d 6d 15 35 55 75 1d 3d 5d 7d
    85 a5 c5 e5 8d ad cd ed 95 b5 d5 f5 9d bd dd fd
    06 26 46 66 0e 2e 4e 6e 16 36 56 76 1e 3e 5e 7e
    86 a6 c6 e6 8e ae ce ee 96 b6 d6 f6 9e be de fe
    06 26 46 66 0e 2e 4e 6e 16 36 56 76 1e 3e 5e 7e
    86 a6 c6 e6 8e ae ce ee 96 b6 d6 f6 9e be de fe
    07 27 47 67 0f 2f 4f 6f 17 37 57 77 1f 3f 5f 7f
    87 a7 c7 e7 8f af cf ef 97 b7 d7 f7 9f bf df ff
    07 27 47 67 0f 2f 4f 6f 17 37 57 77 1f 3f 5f 7f
    87 a7 c7 e7 8f af cf ef 97 b7 d7 f7 9f bf df ff

matrix_b [bits=8,m=8,n=32,k=16,layout=col_major]:
  Bmat:
    00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0 00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
    01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1 01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
    02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2 02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
    03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3 03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
    04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4 04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
    05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5 05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
    06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6 06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
    07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7 07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7
    08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8 08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8
    09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9 09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9
    0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa 0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa
    0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb 0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb
    0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc 0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc
    0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd 0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd
    0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe 0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe
    0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff 0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff
  fragmentB:
    00 01 02 03 80 81 82 83 00 01 02 03 80 81 82 83
    04 05 06 07 84 85 86 87 04 05 06 07 84 85 86 87
    08 09 0a 0b 88 89 8a 8b 08 09 0a 0b 88 89 8a 8b
    0c 0d 0e 0f 8c 8d 8e 8f 0c 0d 0e 0f 8c 8d 8e 8f
    10 11 12 13 90 91 92 93 10 11 12 13 90 91 92 93
    14 15 16 17 94 95 96 97 14 15 16 17 94 95 96 97
    18 19 1a 1b 98 99 9a 9b 18 19 1a 1b 98 99 9a 9b
    1c 1d 1e 1f 9c 9d 9e 9f 1c 1d 1e 1f 9c 9d 9e 9f
    20 21 22 23 a0 a1 a2 a3 20 21 22 23 a0 a1 a2 a3
    24 25 26 27 a4 a5 a6 a7 24 25 26 27 a4 a5 a6 a7
    28 29 2a 2b a8 a9 aa ab 28 29 2a 2b a8 a9 aa ab
    2c 2d 2e 2f ac ad ae af 2c 2d 2e 2f ac ad ae af
    30 31 32 33 b0 b1 b2 b3 30 31 32 33 b0 b1 b2 b3
    34 35 36 37 b4 b5 b6 b7 34 35 36 37 b4 b5 b6 b7
    38 39 3a 3b b8 b9 ba bb 38 39 3a 3b b8 b9 ba bb
    3c 3d 3e 3f bc bd be bf 3c 3d 3e 3f bc bd be bf
    40 41 42 43 c0 c1 c2 c3 40 41 42 43 c0 c1 c2 c3
    44 45 46 47 c4 c5 c6 c7 44 45 46 47 c4 c5 c6 c7
    48 49 4a 4b c8 c9 ca cb 48 49 4a 4b c8 c9 ca cb
    4c 4d 4e 4f cc cd ce cf 4c 4d 4e 4f cc cd ce cf
    50 51 52 53 d0 d1 d2 d3 50 51 52 53 d0 d1 d2 d3
    54 55 56 57 d4 d5 d6 d7 54 55 56 57 d4 d5 d6 d7
    58 59 5a 5b d8 d9 da db 58 59 5a 5b d8 d9 da db
    5c 5d 5e 5f dc dd de df 5c 5d 5e 5f dc dd de df
    60 61 62 63 e0 e1 e2 e3 60 61 62 63 e0 e1 e2 e3
    64 65 66 67 e4 e5 e6 e7 64 65 66 67 e4 e5 e6 e7
    68 69 6a 6b e8 e9 ea eb 68 69 6a 6b e8 e9 ea eb
    6c 6d 6e 6f ec ed ee ef 6c 6d 6e 6f ec ed ee ef
    70 71 72 73 f0 f1 f2 f3 70 71 72 73 f0 f1 f2 f3
    74 75 76 77 f4 f5 f6 f7 74 75 76 77 f4 f5 f6 f7
    78 79 7a 7b f8 f9 fa fb 78 79 7a 7b f8 f9 fa fb
    7c 7d 7e 7f fc fd fe ff 7c 7d 7e 7f fc fd fe ff

accumulator [m=8,n=32,k=16]:
  Cmat:
    0000 0001 0002 0003 0004 0005 0006 0007 0008 0009 000a 000b 000c 000d 000e 000f 0010 0011 0012 0013 0014 0015 0016 0017 0018 0019 001a 001b 001c 001d 001e 001f
    0100 0101 0102 0103 0104 0105 0106 0107 0108 0109 010a 010b 010c 010d 010e 010f 0110 0111 0112 0113 0114 0115 0116 0117 0118 0119 011a 011b 011c 011d 011e 011f
    0200 0201 0202 0203 0204 0205 0206 0207 0208 0209 020a 020b 020c 020d 020e 020f 0210 0211 0212 0213 0214 0215 0216 0217 0218 0219 021a 021b 021c 021d 021e 021f
    0300 0301 0302 0303 0304 0305 0306 0307 0308 0309 030a 030b 030c 030d 030e 030f 0310 0311 0312 0313 0314 0315 0316 0317 0318 0319 031a 031b 031c 031d 031e 031f
    0400 0401 0402 0403 0404 0405 0406 0407 0408 0409 040a 040b 040c 040d 040e 040f 0410 0411 0412 0413 0414 0415 0416 0417 0418 0419 041a 041b 041c 041d 041e 041f
    0500 0501 0502 0503 0504 0505 0506 0507 0508 0509 050a 050b 050c 050d 050e 050f 0510 0511 0512 0513 0514 0515 0516 0517 0518 0519 051a 051b 051c 051d 051e 051f
    0600 0601 0602 0603 0604 0605 0606 0607 0608 0609 060a 060b 060c 060d 060e 060f 0610 0611 0612 0613 0614 0615 0616 0617 0618 0619 061a 061b 061c 061d 061e 061f
    0700 0701 0702 0703 0704 0705 0706 0707 0708 0709 070a 070b 070c 070d 070e 070f 0710 0711 0712 0713 0714 0715 0716 0717 0718 0719 071a 071b 071c 071d 071e 071f
  fragmentC:
    0000 0100 0008 0108 0010 0110 0018 0118
    0200 0300 0208 0308 0210 0310 0218 0318
    0400 0500 0408 0508 0410 0510 0418 0518
    0600 0700 0608 0708 0610 0710 0618 0718
    0001 0101 0009 0109 0011 0111 0019 0119
    0201 0301 0209 0309 0211 0311 0219 0319
    0401 0501 0409 0509 0411 0511 0419 0519
    0601 0701 0609 0709 0611 0711 0619 0719
    0002 0102 000a 010a 0012 0112 001a 011a
    0202 0302 020a 030a 0212 0312 021a 031a
    0402 0502 040a 050a 0412 0512 041a 051a
    0602 0702 060a 070a 0612 0712 061a 071a
    0003 0103 000b 010b 0013 0113 001b 011b
    0203 0303 020b 030b 0213 0313 021b 031b
    0403 0503 040b 050b 0413 0513 041b 051b
    0603 0703 060b 070b 0613 0713 061b 071b
    0004 0104 000c 010c 0014 0114 001c 011c
    0204 0304 020c 030c 0214 0314 021c 031c
    0404 0504 040c 050c 0414 0514 041c 051c
    0604 0704 060c 070c 0614 0714 061c 071c
    0005 0105 000d 010d 0015 0115 001d 011d
    0205 0305 020d 030d 0215 0315 021d 031d
    0405 0505 040d 050d 0415 0515 041d 051d
    0605 0705 060d 070d 0615 0715 061d 071d
    0006 0106 000e 010e 0016 0116 001e 011e
    0206 0306 020e 030e 0216 0316 021e 031e
    0406 0506 040e 050e 0416 0516 041e 051e
    0606 0706 060e 070e 0616 0716 061e 071e
    0007 0107 000f 010f 0017 0117 001f 011f
    0207 0307 020f 030f 0217 0317 021f 031f
    0407 0507 040f 050f 0417 0517 041f 051f
    0607 0707 060f 070f 0617 0717 061f 071f
Done.
```
