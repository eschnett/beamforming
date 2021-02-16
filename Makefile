all: cpu
cpu: adler32.o cpu.o
	g++ -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -Ofast -o $@ $^
%.o: %.c
	gcc -std=c11 -fopenmp -march=native -Ofast -c $*.c
%.o: %.cxx
	g++ -std=c++17 -fopenmp -Drestrict=__restrict__ -march=native -Ofast -c $*.cxx
adler32.o: adler32.h
cpu.o: adler32.h
format:
	clang-format -i adler32.h adler32.c cpu.cxx
clean:
	rm -f cpu adler32.o cpu.o
.PHONY: all format clean
