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
registers). The table below shows which of these 256 values are held
in the registers of thread 0.

| m  | n  | k  | #R | elements on thread 0    | offset |
|----|----|----|----|-------------------------|--------|
|  8 |  8 | 32 |  2 | 00 01                   | 02     |
| 16 | 16 | 16 |  8 | 00 01 80 81 08 09 88 89 | 02     |
| 32 |  8 | 16 |  8 | 00 01 40 41 80 81 c0 c1 | 02     |
|  8 | 32 | 16 |  8 | 00 01 08 09 10 11 18 19 | 02     |

The values held by the other threads follow naturally. That is, thread
1 holds the next values that aren't already held by thread 0. To give
an equation: If thread 0 holds value i in register r, then thread t
holds value i + 2*t in its register r.

## Gory Details

No insights below, just the raw data without explanations.

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
    00 01 02 03 04 05 06 07
    08 09 0a 0b 0c 0d 0e 0f
    10 11 12 13 14 15 16 17
    18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27
    28 29 2a 2b 2c 2d 2e 2f
    30 31 32 33 34 35 36 37
    38 39 3a 3b 3c 3d 3e 3f
  fragmentC:
    00 01
    02 03
    04 05
    06 07
    08 09
    0a 0b
    0c 0d
    0e 0f
    10 11
    12 13
    14 15
    16 17
    18 19
    1a 1b
    1c 1d
    1e 1f
    20 21
    22 23
    24 25
    26 27
    28 29
    2a 2b
    2c 2d
    2e 2f
    30 31
    32 33
    34 35
    36 37
    38 39
    3a 3b
    3c 3d
    3e 3f

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
  fragmentC:
    00 01 80 81 08 09 88 89
    02 03 82 83 0a 0b 8a 8b
    04 05 84 85 0c 0d 8c 8d
    06 07 86 87 0e 0f 8e 8f
    10 11 90 91 18 19 98 99
    12 13 92 93 1a 1b 9a 9b
    14 15 94 95 1c 1d 9c 9d
    16 17 96 97 1e 1f 9e 9f
    20 21 a0 a1 28 29 a8 a9
    22 23 a2 a3 2a 2b aa ab
    24 25 a4 a5 2c 2d ac ad
    26 27 a6 a7 2e 2f ae af
    30 31 b0 b1 38 39 b8 b9
    32 33 b2 b3 3a 3b ba bb
    34 35 b4 b5 3c 3d bc bd
    36 37 b6 b7 3e 3f be bf
    40 41 c0 c1 48 49 c8 c9
    42 43 c2 c3 4a 4b ca cb
    44 45 c4 c5 4c 4d cc cd
    46 47 c6 c7 4e 4f ce cf
    50 51 d0 d1 58 59 d8 d9
    52 53 d2 d3 5a 5b da db
    54 55 d4 d5 5c 5d dc dd
    56 57 d6 d7 5e 5f de df
    60 61 e0 e1 68 69 e8 e9
    62 63 e2 e3 6a 6b ea eb
    64 65 e4 e5 6c 6d ec ed
    66 67 e6 e7 6e 6f ee ef
    70 71 f0 f1 78 79 f8 f9
    72 73 f2 f3 7a 7b fa fb
    74 75 f4 f5 7c 7d fc fd
    76 77 f6 f7 7e 7f fe ff

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
    80 81 82 83 84 85 86 87
    88 89 8a 8b 8c 8d 8e 8f
    90 91 92 93 94 95 96 97
    98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7
    a8 a9 aa ab ac ad ae af
    b0 b1 b2 b3 b4 b5 b6 b7
    b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7
    c8 c9 ca cb cc cd ce cf
    d0 d1 d2 d3 d4 d5 d6 d7
    d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7
    e8 e9 ea eb ec ed ee ef
    f0 f1 f2 f3 f4 f5 f6 f7
    f8 f9 fa fb fc fd fe ff
  fragmentC:
    00 01 40 41 80 81 c0 c1
    02 03 42 43 82 83 c2 c3
    04 05 44 45 84 85 c4 c5
    06 07 46 47 86 87 c6 c7
    08 09 48 49 88 89 c8 c9
    0a 0b 4a 4b 8a 8b ca cb
    0c 0d 4c 4d 8c 8d cc cd
    0e 0f 4e 4f 8e 8f ce cf
    10 11 50 51 90 91 d0 d1
    12 13 52 53 92 93 d2 d3
    14 15 54 55 94 95 d4 d5
    16 17 56 57 96 97 d6 d7
    18 19 58 59 98 99 d8 d9
    1a 1b 5a 5b 9a 9b da db
    1c 1d 5c 5d 9c 9d dc dd
    1e 1f 5e 5f 9e 9f de df
    20 21 60 61 a0 a1 e0 e1
    22 23 62 63 a2 a3 e2 e3
    24 25 64 65 a4 a5 e4 e5
    26 27 66 67 a6 a7 e6 e7
    28 29 68 69 a8 a9 e8 e9
    2a 2b 6a 6b aa ab ea eb
    2c 2d 6c 6d ac ad ec ed
    2e 2f 6e 6f ae af ee ef
    30 31 70 71 b0 b1 f0 f1
    32 33 72 73 b2 b3 f2 f3
    34 35 74 75 b4 b5 f4 f5
    36 37 76 77 b6 b7 f6 f7
    38 39 78 79 b8 b9 f8 f9
    3a 3b 7a 7b ba bb fa fb
    3c 3d 7c 7d bc bd fc fd
    3e 3f 7e 7f be bf fe ff

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
    00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
    40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
    60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f 70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
    80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f 90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
  fragmentC:
    00 01 08 09 10 11 18 19
    02 03 0a 0b 12 13 1a 1b
    04 05 0c 0d 14 15 1c 1d
    06 07 0e 0f 16 17 1e 1f
    20 21 28 29 30 31 38 39
    22 23 2a 2b 32 33 3a 3b
    24 25 2c 2d 34 35 3c 3d
    26 27 2e 2f 36 37 3e 3f
    40 41 48 49 50 51 58 59
    42 43 4a 4b 52 53 5a 5b
    44 45 4c 4d 54 55 5c 5d
    46 47 4e 4f 56 57 5e 5f
    60 61 68 69 70 71 78 79
    62 63 6a 6b 72 73 7a 7b
    64 65 6c 6d 74 75 7c 7d
    66 67 6e 6f 76 77 7e 7f
    80 81 88 89 90 91 98 99
    82 83 8a 8b 92 93 9a 9b
    84 85 8c 8d 94 95 9c 9d
    86 87 8e 8f 96 97 9e 9f
    a0 a1 a8 a9 b0 b1 b8 b9
    a2 a3 aa ab b2 b3 ba bb
    a4 a5 ac ad b4 b5 bc bd
    a6 a7 ae af b6 b7 be bf
    c0 c1 c8 c9 d0 d1 d8 d9
    c2 c3 ca cb d2 d3 da db
    c4 c5 cc cd d4 d5 dc dd
    c6 c7 ce cf d6 d7 de df
    e0 e1 e8 e9 f0 f1 f8 f9
    e2 e3 ea eb f2 f3 fa fb
    e4 e5 ec ed f4 f5 fc fd
    e6 e7 ee ef f6 f7 fe ff
Done.
```
