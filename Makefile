CC=mpic++
FLAGS = -std=c++0x

all:
	$(CC) $(FLAGS) -o PPrombergMPI.out rombergMPI.cpp
		
clean:
	rm -f PPrombergMPI*
	
testMPI: all
	qsub script_rombergMPI
	qstat

testHybrid:
	nvcc -Xcompiler -fopenmp romberg_hybrid.cu
	./a.out

test:
	nvcc -o v1.o romberg_v1.cu
	nvcc -o v2.o romberg_v2.cu
	nvcc -o ref.o ref_simpson.cu
	nvcc -Xcompiler -fopenmp -o hybrid.o romberg_hybrid.cu
	g++ -o seq.o Sequential.cpp 
	./v1.o
	./v2.o
	./ref.o
	./hybrid.o
	./seq.o
