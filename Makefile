CC = gcc
CXX = g++
CU = nvcc
CFLAGS = -std=c11 -fopenmp -march=native -O3
CXXFLAGS = -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -O3
CUFLAGS = -std=c++17 --compiler-options -std=c++17 -Drestrict=__restrict__ --gpu-architecture sm_75 --compiler-options -march=native -O3 --compiler-options -O3

all: cpu

cpu: adler32.o cpu.o icomplex4.o
	$(CXX) $(CXXFLAGS) -o $@ $^
cuda: adler32.o cuda.o icomplex4.o
	$(CU) $(CUFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $*.c
%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $*.cxx
%.o: %.cu
	$(CU) $(CUFLAGS) -c $*.cu

adler32.o: adler32.h
cpu.o: icomplex4.hxx
cuda.o: adler32.h
icomplex4.o: adler32.h icomplex4.hxx

format:
	clang-format -i adler32.h adler32.c cpu.cxx cuda.cu icomplex4.cxx icomplex4.hxx
clean:
	rm -f cpu cuda adler32.o cpu.o cuda.o icomplex4.o

.PHONY: all format clean
