
#include <stdlib.h>
#include <string.h>
#include "Laplacian27pt.hpp"


int main(int argc, char* argv[])
{
	
	int nx = 128;
	int ny = 128;
	int nz = 128;

	int arg_index = 1;

	while (arg_index < argc)
	{
		if (strcmp(argv[arg_index], "-n") == 0)
		{
			nx = atoi(argv[++arg_index]);
			ny = atoi(argv[++arg_index]);
			nz = atoi(argv[++arg_index]);
		}
		++arg_index;
	}

	Laplacian27pt LMG(nx, ny, nz);
	
	LMG.MaxIters = 500;
	LMG.Tolerance = 1.0e-08;
	LMG.PrintStats = 1;
	
	LMG.Setup();
	
	Vector b, x;
	b.Resize(nx * ny * nz);
	x.Resize(nx * ny * nz);
	b.Fill(1.0);
	x.Fill(0.0);

	int iter;
	double relres;

	x.Fill(0.0);
	OptData data;
	LMG.OptimizeProblem(data, b, x);
	LMG.Smoothing_DBSR(data, b, x, iter, relres);


	return 0;
}