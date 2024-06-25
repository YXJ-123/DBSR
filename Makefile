CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp
# AVX512
# CXXFLAGS = -std=c++11 -O3 -mavx512f -march=native -fopenmp
# NEON
# CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

INCS = -I../include
LIBS = 

file = Laplacian27pt.cpp Smoothing.cpp OptData.cpp OptimizedData.cpp SparseMatrix.cpp

all: DBSR

DBSR: Laplacian27pt.cpp
	$(CXX) $(CXXFLAGS) -o ILU0-DBSR $(file) $(INCS) $(LIBS)
	
clean:
	rm -f ILU0-DBSR
