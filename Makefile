CC = gcc
CXX = g++
CU = nvcc
CFLAGS = -std=c11 -fopenmp -march=native -Ofast
CXXFLAGS = -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -Ofast
CUFLAGS = -std=c++17 --compiler-options -std=c++17 -Drestrict=__restrict__ --gpu-architecture sm_75 --compiler-options -march=native -O3 --compiler-options -Ofast

all: cpu

cpu: adler32.o cpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^
cuda: adler32.o cuda.o
	$(CU) $(CUFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $*.c
%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $*.cxx
%.o: %.cu
	$(CU) $(CUFLAGS) -c $*.cu

adler32.o: adler32.h
cpu.o: adler32.h
cuda.o: adler32.h

format:
	clang-format -i adler32.h adler32.c cpu.cxx cuda.cu
clean:
	rm -f cpu cuda adler32.o cpu.o cuda.o

.PHONY: all format clean
