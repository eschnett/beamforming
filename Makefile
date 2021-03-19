CC = gcc
CXX = g++
CU = nvcc
# Debug
# CFLAGS = -std=c11 -fopenmp -march=native -g
# CXXFLAGS = -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -g
# CUFLAGS = -std=c++17 --compiler-options -std=c++17 -Drestrict=__restrict__ --gpu-architecture sm_75 --compiler-options -march=native -g --compiler-options -g
# Optimize
CFLAGS = -std=c11 -fopenmp -march=native -O3
CXXFLAGS = -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -O3
CUFLAGS = -std=c++17 --compiler-options -std=c++17 -Drestrict=__restrict__ --gpu-architecture sm_75 --compiler-options -march=native -O3 --compiler-options -O3

all: cpu cpu2

cpu: adler32.o cpu.o icomplex4.o
	$(CXX) $(CXXFLAGS) -o $@ $^
cpu2: adler32.o cpu2.o icomplex4.o
	$(CXX) $(CXXFLAGS) -o $@ $^
cuda: adler32.o cuda.o icomplex4.o
	$(CU) $(CUFLAGS) -o $@ $^
fragment: fragment.o
	$(CU) $(CUFLAGS) -o $@ $^
matmul: matmul.o
	$(CU) $(CUFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $*.c
%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $*.cxx
%.o: %.cu
	$(CU) $(CUFLAGS) -c $*.cu

adler32.o: adler32.h
cpu.o: icomplex4.hxx
cpu2.o: icomplex4.hxx
cuda.o: adler32.h
fragment.o:
matmul.o:
icomplex4.o: adler32.h icomplex4.hxx

format:
	clang-format -i adler32.h adler32.c cpu.cxx cpu2.cxx cuda.cu icomplex4.cxx icomplex4.hxx fragment.cu matmul.cu
clean:
	rm -f cpu cpu2 cuda fragment matmul
	rm -f adler32.o cpu.o cpu2.o cuda.o icomplex4.o fragment.o matmul.o

.PHONY: all format clean
