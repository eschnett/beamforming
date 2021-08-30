CC = gcc
CXX = g++
CU = nvcc

# Symmetry
# # Debug
# CFLAGS = -std=c11 -fopenmp -march=native -g
# CXXFLAGS = -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -g
# CUFLAGS = -std=c++17 --compiler-options -std=c++17 -Drestrict=__restrict__ --expt-relaxed-constexpr --gpu-architecture sm_75 --compiler-options -march=native -g --compiler-options -g
# Optimize
CFLAGS = -std=c11 -fopenmp -march=native -O3 -DNDEBUG
CXXFLAGS = -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -O3 -DNDEBUG
CUFLAGS = -std=c++17 --compiler-options -std=c++17 -Drestrict=__restrict__ --expt-relaxed-constexpr --gpu-architecture sm_75 --compiler-options -march=native -O3 --compiler-options -O3 -DNDEBUG

# Sky
# # Debug
# CFLAGS = -std=c11 -fopenmp -march=native -g
# CXXFLAGS = -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -g
# CUFLAGS = -std=c++17 --compiler-options -std=c++17 -Drestrict=__restrict__ --expt-relaxed-constexpr --gpu-architecture sm_80 --compiler-options -march=native -g --compiler-options -g
# Optimize
# CFLAGS = -std=c11 -fopenmp -march=native -O3 -DNDEBUG
# CXXFLAGS = -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -O3 -DNDEBUG
# CUFLAGS = -std=c++17 --compiler-options -std=c++17 -Drestrict=__restrict__ --expt-relaxed-constexpr --ftz=true --extra-device-vectorization --gpu-architecture sm_80 --compiler-options -march=native -O3 --compiler-options -O3 -DNDEBUG



all: cpu cpu2

cpu: adler32.o cpu.o arraysizes.o
	$(CXX) $(CXXFLAGS) -o $@ $^
cpu2: adler32.o cpu2.o arraysizes.o
	$(CXX) $(CXXFLAGS) -o $@ $^
cuda: adler32.o cuda.o arraysizes.o
	$(CU) $(CUFLAGS) -o $@ $^
cuda2: adler32.o cuda2.o arraysizes.o
	$(CU) $(CUFLAGS) -o $@ $^
cuda3: adler32.o cuda3.o arraysizes.o
	$(CU) $(CUFLAGS) -o $@ $^
cuda4: adler32.o cuda4.o arraysizes.o
	$(CU) $(CUFLAGS) -o $@ $^
fragment: fragment.o
	$(CU) $(CUFLAGS) -o $@ $^
fragment2: fragment2.o
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
cpu.o: arraysizes.hxx icomplex4.hxx
cpu2.o: arraysizes.hxx icomplex4.hxx
cuda.o: adler32.h arraysizes.hxx icomplex4.hxx
cuda2.o: adler32.h arraysizes.hxx icomplex4.hxx
cuda3.o: adler32.h arraysizes.hxx icomplex4.hxx
cuda4.o: adler32.h arraysizes.hxx icomplex4.hxx
fragment.o:
fragment2.o:
matmul.o:
arraysizes.o: adler32.h arraysizes.hxx icomplex4.hxx

format:
	clang-format -i				\
		adler32.c			\
		adler32.h			\
		arraysizes.cxx			\
		arraysizes.hxx			\
		cpu.cxx				\
		cpu2.cxx			\
		cuda.cu				\
		cuda2.cu			\
		cuda3.cu			\
		cuda4.cu			\
		fragment.cu			\
		fragment2.cu			\
		icomplex4.hxx			\
		matmul.cu
clean:
	rm -f cpu cpu2 cuda cuda2 cuda3 cuda4 fragment fragment2 matmul
	rm -f adler32.o arraysizes.o cpu.o cpu2.o cuda.o cuda2.o cuda3.o cuda4.o fragment.o fragment2.o matmul.o

.PHONY: all format clean
