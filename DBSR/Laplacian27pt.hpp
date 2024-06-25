#include <sys/time.h>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <omp.h>
#include "OptData.hpp"
#include "config.h"



static inline int sub2ind(int nx, int ny, int nz, int ix, int iy, int iz)
{
	return iz * nx * ny + iy * nx + ix;
}

static void CSRGenerator27pt(int nx, int ny, int nz, CSRMatrix& A)
{
	double c = 1.0;
	double b = -1.0;
	if (nx > 1) b *= 3;
	if (ny > 1) b *= 3;
	if (nz > 1) b *= 3;
	b += 1.0;

	int size = nx * ny * nz;

	int* local_rowptr = new int[size + 1];

	local_rowptr[0] = 0;
	for (int ix = 0; ix < nx; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < nz; ++iz)
			{
				int cnt = 1;
				if (ix > 0)
				{
					++cnt;
					if (iy > 0)
					{
						++cnt;
						if (iz > 0) ++cnt;
						if (iz < nz - 1) ++cnt;
					}
					if (iy < ny - 1)
					{
						++cnt;
						if (iz > 0) ++cnt;
						if (iz < nz - 1) ++cnt;
					}
					if (iz > 0) ++cnt;
					if (iz < nz - 1) ++cnt;
				}

				if (ix < nx - 1)
				{
					++cnt;
					if (iy > 0)
					{
						++cnt;
						if (iz > 0) ++cnt;
						if (iz < nz - 1) ++cnt;
					}
					if (iy < ny - 1)
					{
						++cnt;
						if (iz > 0) ++cnt;
						if (iz < nz - 1) ++cnt;
					}
					if (iz > 0) ++cnt;
					if (iz < nz - 1) ++cnt;
				}

				if (iy > 0)
				{
					++cnt;
					if (iz > 0) ++cnt;
					if (iz < nz - 1) ++cnt;
				}

				if (iy < ny - 1)
				{
					++cnt;
					if (iz > 0) ++cnt;
					if (iz < nz - 1) ++cnt;
				}

				if (iz > 0) ++cnt;
				if (iz < nz - 1) ++cnt;

				local_rowptr[sub2ind(nx, ny, nz, ix, iy, iz) + 1] = cnt;
			}
		}
	}

	for (int i = 0; i < size; ++i)
		local_rowptr[i + 1] += local_rowptr[i];

	int* local_colind = new int[local_rowptr[size]];
	double* local_values = new double[local_rowptr[size]];

	for (int ix = 0; ix < nx; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < nz; ++iz)
			{
				int r = sub2ind(nx, ny, nz, ix, iy, iz);
				int j = local_rowptr[r];

				if (ix > 0)
				{
					if (iy > 0)
					{
						if (iz > 0)
						{
							local_colind[j] = sub2ind(nx, ny, nz, ix - 1, iy - 1, iz - 1);
							local_values[j++] = c;
						}

						local_colind[j] = sub2ind(nx, ny, nz, ix - 1, iy - 1, iz);
						local_values[j++] = c;

						if (iz < nz - 1)
						{
							local_colind[j] = sub2ind(nx, ny, nz, ix - 1, iy - 1, iz + 1);
							local_values[j++] = c;
						}
					}

					if (iz > 0)
					{
						local_colind[j] = sub2ind(nx, ny, nz, ix - 1, iy, iz - 1);
						local_values[j++] = c;
					}

					local_colind[j] = sub2ind(nx, ny, nz, ix - 1, iy, iz);
					local_values[j++] = c;

					if (iz < nz - 1)
					{
						local_colind[j] = sub2ind(nx, ny, nz, ix - 1, iy, iz + 1);
						local_values[j++] = c;
					}

					if (iy < ny - 1)
					{
						if (iz > 0)
						{
							local_colind[j] = sub2ind(nx, ny, nz, ix - 1, iy + 1, iz - 1);
							local_values[j++] = c;
						}

						local_colind[j] = sub2ind(nx, ny, nz, ix - 1, iy + 1, iz);
						local_values[j++] = c;

						if (iz < nz - 1)
						{
							local_colind[j] = sub2ind(nx, ny, nz, ix - 1, iy + 1, iz + 1);
							local_values[j++] = c;
						}
					}
				}

				if (iy > 0)
				{
					if (iz > 0)
					{
						local_colind[j] = sub2ind(nx, ny, nz, ix, iy - 1, iz - 1);
						local_values[j++] = c;
					}

					local_colind[j] = sub2ind(nx, ny, nz, ix, iy - 1, iz);
					local_values[j++] = c;

					if (iz < nz - 1)
					{
						local_colind[j] = sub2ind(nx, ny, nz, ix, iy - 1, iz + 1);
						local_values[j++] = c;
					}
				}

				if (iz > 0)
				{
					local_colind[j] = sub2ind(nx, ny, nz, ix, iy, iz - 1);
					local_values[j++] = c;
				}

				local_colind[j] = sub2ind(nx, ny, nz, ix, iy, iz);
				local_values[j++] = b;

				if (iz < nz - 1)
				{
					local_colind[j] = sub2ind(nx, ny, nz, ix, iy, iz + 1);
					local_values[j++] = c;
				}

				if (iy < ny - 1)
				{
					if (iz > 0)
					{
						local_colind[j] = sub2ind(nx, ny, nz, ix, iy + 1, iz - 1);
						local_values[j++] = c;
					}

					local_colind[j] = sub2ind(nx, ny, nz, ix, iy + 1, iz);
					local_values[j++] = c;

					if (iz < nz - 1)
					{
						local_colind[j] = sub2ind(nx, ny, nz, ix, iy + 1, iz + 1);
						local_values[j++] = c;
					}
				}

				if (ix < nx - 1)
				{
					if (iy > 0)
					{
						if (iz > 0)
						{
							local_colind[j] = sub2ind(nx, ny, nz, ix + 1, iy - 1, iz - 1);
							local_values[j++] = c;
						}

						local_colind[j] = sub2ind(nx, ny, nz, ix + 1, iy - 1, iz);
						local_values[j++] = c;

						if (iz < nz - 1)
						{
							local_colind[j] = sub2ind(nx, ny, nz, ix + 1, iy - 1, iz + 1);
							local_values[j++] = c;
						}
					}

					if (iz > 0)
					{
						local_colind[j] = sub2ind(nx, ny, nz, ix + 1, iy, iz - 1);
						local_values[j++] = c;
					}

					local_colind[j] = sub2ind(nx, ny, nz, ix + 1, iy, iz);
					local_values[j++] = c;

					if (iz < nz - 1)
					{
						local_colind[j] = sub2ind(nx, ny, nz, ix + 1, iy, iz + 1);
						local_values[j++] = c;
					}

					if (iy < ny - 1)
					{
						if (iz > 0)
						{
							local_colind[j] = sub2ind(nx, ny, nz, ix + 1, iy + 1, iz - 1);
							local_values[j++] = c;
						}

						local_colind[j] = sub2ind(nx, ny, nz, ix + 1, iy + 1, iz);
						local_values[j++] = c;

						if (iz < nz - 1)
						{
							local_colind[j] = sub2ind(nx, ny, nz, ix + 1, iy + 1, iz + 1);
							local_values[j++] = c;
						}
					}
				}
			}
		}
	}

	A.Free();
	A.size[0] = size;
	A.size[1] = size;
	A.rowptr = local_rowptr;
	A.colind = local_colind;
	A.values = local_values;
}







	class Laplacian27pt
	{
	private:

		CSRMatrix A;

	public:
		int    nx;
		int    ny;
		int    nz;

		int    MaxIters;
		double Tolerance;
		int    PrintStats;

		Laplacian27pt(
		int    _nx = 64,
		int    _ny = 64,
		int    _nz = 64)
		: 
		nx(_nx),
		ny(_ny),
		nz(_nz),
		MaxIters(500),
		Tolerance(1.0e-08),
		PrintStats(1)
		{
		}
		
		~Laplacian27pt()
		{
		}

		void Setup()
		{
			if (PrintStats)
			{
				std::cout << "\n3D Laplacian Problem with 27-point Stencil\n";
				std::cout << "Local Domain Size: (" << nx << ", " << ny << ", " << nz << ")\n";
				std::cout << "--------------------------------------------------" << std::endl;
			}

  			struct timeval t1,t2;
			gettimeofday(&t1, NULL);

			int _nx = nx;
			int _ny = ny;
			int _nz = nz;
			
			CSRGenerator27pt(_nx, _ny, _nz, A);

			gettimeofday(&t2, NULL);
			double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
			
			if (PrintStats)
			{
				std::cout << "--------------------------------------------------\n";
				std::cout << "Setup Time: " << timeuse << std::endl;
			}
		}
		
		void OptimizeProblem(OptData & data, Vector & b, Vector & x);

		void ILU0_DBSR(OptData & data, Vector & b, Vector & x, bool xis);

		void Smoothing_DBSR(OptData & data, Vector& b, Vector& x, int& iter, double& relres)
		{
			int n = A.OutSize();

			double normb, res, alpha, beta, rho, rho1;
			Vector r, z, p, v;

			r.Resize(n);
			z.Resize(n);
			p.Resize(n);
			v.Resize(n);

			if (PrintStats)
			{
			#ifdef _AVX512	
				std::cout << "\nSmoothing_AVX Iterations:\n";
			#elif defined _NEON	
				std::cout << "\nSmoothing_NEON Iterations:\n";
			#else
				std::cout << "\nSmoothing_DBSR Iterations:\n";
			#endif
			}

  			struct timeval t1,t2;
			gettimeofday(&t1, NULL);

			iter = 0;

			A.Apply(x, r);
			VecAXPBY(1.0, b, -1.0, r);

			res = sqrt(VecDot(r, r));
			normb = sqrt(VecDot(b, b));

			while (1)
			{
				if (res / normb <= Tolerance || iter == MaxIters) break;

				z.Fill(0.0);
				ILU0_DBSR(data, r, z, 0);

				rho = VecDot(r, z);

				if (iter == 0)
					p.Copy(z);
				else
				{
					beta = rho / rho1;
					VecAXPBY(1.0, z, beta, p);
				}

				A.Apply(p, v);

				alpha = rho / VecDot(v, p);
				rho1 = rho;

				x.AddScaled(alpha, p);
				r.AddScaled(-alpha, v);

				res = sqrt(VecDot(r, r));

				++iter;
			}

			relres = res / normb;

			gettimeofday(&t2, NULL);
			double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;

			if (PrintStats)
			{
				std::cout << "Iterations: " << iter << "\n";
				std::cout << "Final Relative Residual: " << relres << "\n";
				std::cout << "Solve Time: " << timeuse << std::endl;
			}
		}

	};
