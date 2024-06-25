#include "Laplacian27pt.hpp"
// #include "OptData.hpp"


static inline void swap_int(int& a, int& b)
{
	int temp = a;
	a = b;
	b = temp;
}

static inline void swap(double* v, int a, int b, int bsize)
{
	double temp[bsize];
	for(int i = 0; i < bsize; i++)	temp[i] = v[a + i];
	for(int i = 0; i < bsize; i++)	v[a + i] = v[b + i];
	for(int i = 0; i < bsize; i++)	v[b + i] = temp[i];
}

static void SortIndAsc(int* a, int* offset, double* value, int left, int right, int bsize)
{
	if (left >= right) return;

	swap_int(a[left], a[(left + right) / 2]);
	swap_int(offset[left], offset[(left + right) / 2]);
	swap(value, left * bsize, (left + right) / 2 * bsize, bsize);

	int last = left;
	for (int i = left + 1; i <= right; ++i)
	{
		if (a[i] < a[left])
		{
			swap_int(a[++last], a[i]);
			swap_int(offset[last], offset[i]);
			swap(value, last * bsize, i * bsize, bsize);
		}
	}

	swap_int(a[left], a[last]);
	swap_int(offset[left], offset[last]);
	swap(value, left * bsize, last * bsize, bsize);

	SortIndAsc(a, offset, value, left, last - 1, bsize);
	SortIndAsc(a, offset, value, last + 1, right, bsize);
}



static void Factorization(CSRMatrix & A, OptData & data)
{
	int n = A.size[0];
	int* Ap = A.rowptr;
	int* Ai = A.colind;
	double* Av = A.values;
	

	int * Rmap = data.ReorderMap;

	int * invRmap = (int*)malloc(n * sizeof(int));
	for(int i = 0; i < n; i++)
		invRmap[Rmap[i]] = i;

//************* CSR -> BMC -> DBSR Begin ****************

	int bsize = data.Lower.bsize;
	int nb = (n - 1) / bsize + 1;
	int mb = (n - 1) / bsize + 1;

	int* Bp = new int[nb + 1];
	Bp[0] = 0;

#pragma omp parallel
	{
		int* w = new int[mb];
		for (int i = 0; i < mb; ++i)
			w[i] = -1;
#pragma omp for schedule(guided)
		for (int i = 0; i < nb; ++i)
		{
			int cnt = 0;
			for (int ii = 0; ii < bsize; ++ii)
			{
				int ia = i * bsize + ii;
				int inv_ia = invRmap[ia];
				for (int j = Ap[inv_ia]; j < Ap[inv_ia + 1]; ++j)
				{
					int jcolb = Rmap[Ai[j]] / bsize;
					if (w[jcolb] != i)
					{
						w[jcolb] = i;
						++cnt;
					}
				}
			}
			Bp[i + 1] = cnt;
		}
		delete[] w;
	}

	for (int i = 0; i < nb; ++i)
		Bp[i + 1] += Bp[i];

	int _nnz = Bp[nb] * bsize;
	int * Bd = new int[nb];
	int * Bi = new int[Bp[nb]];
	int * Bo = new int[Bp[nb]];
	double * Bv = new double[_nnz];
	double * Bv2 = new double[_nnz];

#pragma omp parallel for
	for(int i = 0; i < _nnz; i++)
		Bv[i] = 0;

#pragma omp parallel
	{
		int* w = new int[mb];
		for (int i = 0; i < mb; ++i)
			w[i] = -1;
#pragma omp for schedule(guided)
		for (int i = 0; i < nb; ++i)
		{
			for (int ii = 0, r = Bp[i], r0 = Bp[i]; ii < bsize; ++ii)
			{
				int ia = i * bsize + ii;
				int inv_ia = invRmap[ia];
				for (int j = Ap[inv_ia]; j < Ap[inv_ia + 1]; ++j)
				{
					int jcol = Rmap[Ai[j]];
					int bjcol = jcol / bsize;
					int bjres = jcol - bjcol * bsize;
					if (w[bjcol] < r0)
					{
						w[bjcol] = r;
						Bi[r] = bjcol;
						Bo[r] = bjres - ii;
						++r;
					}
					Bv[w[bjcol] * bsize + ii] = Av[j];
				}
			}

			SortIndAsc(Bi, Bo, Bv, Bp[i], Bp[i+1] - 1, bsize);

			for(int k = Bp[i]; k < Bp[i + 1]; k++)
				if(Bi[k] == i) {
					Bd[i] = k;
					break;
				}
		}
		delete[] w;
	}

	delete invRmap;

	for(int i = 0; i < _nnz; i++)
		Bv2[i] = Bv[i];

//********* CSR -> BMC -> DBSR End  *****************



//************* ILU(0) Begin *********************

  	struct timeval t1,t2;
	gettimeofday(&t1, NULL);

	int* Lp = new int[nb + 1];
	int* Up = new int[nb + 1];
	Lp[0] = 0;
	Up[0] = 0;

#pragma omp parallel for
	for(int i = 0; i < nb; i++){
		Lp[i + 1] = Bd[i] - Bp[i];
		Up[i + 1] = Bp[i + 1] - 1 - Bd[i];
	}

	for(int i = 1; i < nb; i++){
		Lp[i + 1] += Lp[i];
		Up[i + 1] += Up[i];
	}
	assert(Lp[nb] == Up[nb]);
	int* Li = new int[Lp[nb]];
	int* Lo = new int[Lp[nb]];
	double* Lv = (double*)aligned_alloc(64, Lp[nb] * bsize * sizeof(double));
	int* Ui = new int[Up[nb]];
	int* Uo = new int[Up[nb]];
	double* Uv = (double*)aligned_alloc(64, Up[nb] * bsize * sizeof(double));
	double* D = (double*)aligned_alloc(64, nb * bsize * sizeof(double));
	double * Drcp = (double*)aligned_alloc(64, nb * bsize * sizeof(double));

#pragma omp parallel for schedule(guided)
	for(int i = 0; i < nb; i++){
		int r = Lp[i];
		int s = Up[i];
		for(int j = Bp[i]; j < Bp[i + 1]; j++){
			if(j < Bd[i]){
				Li[r] = Bi[j];
				Lo[r] = Bo[j];
				r++;
			}
			else if(j > Bd[i]){
				Ui[s] = Bi[j];
				Uo[s] = Bo[j];
				s++;
			}
		}
	}

	int NumColors = data.NumColors;
	int blkNumPerColor = nb / NumColors;
	int blksPerCore = data.BlockSize;

	for (int color = 0; color < NumColors; ++color)	
#pragma omp parallel for schedule(guided)
		for(int ib = blkNumPerColor * color; ib < blkNumPerColor * (color + 1); ib += blksPerCore)
			for(int i = ib; i < ib + blksPerCore; i++)
			{
				// do B[i, i] block ILU Factorized

				for(int k = Lp[i]; k < Lp[i + 1]; k++){
					int kcol = Li[k];
					int koffset = k * bsize;
					int kB = Bp[i] + k - Lp[i];
					int voffset = kB * bsize;
					int height = kcol * bsize;
					for(int ii = 0; ii < bsize; ii++) // SIMD
						Lv[koffset + ii] = Bv[voffset + ii] * Drcp[height + Lo[k] + ii];

					kB++;
					int j = Up[kcol];
					while(kB < Bp[i + 1] && j < Up[kcol + 1]){
						int kBcol = Bi[kB];
						int jcol = Ui[j];
						if(kBcol > jcol) j++;
						else if(kBcol < jcol) kB++;
						else{
							if(Lo[k] + Uo[j] == Bo[kB]){
								int kBoffset = kB * bsize;
								int joffset = j * bsize + Lo[k];
								for(int ii = 0; ii < bsize; ii++) // Can be changed to SIMD instruction
									Bv[kBoffset + ii] -= Lv[koffset + ii] * Uv[joffset + ii];
							}
							kB++; j++;
						}
					}
				}
				for(int ii = 0, height = i * bsize, ioffset = Bd[i] * bsize; ii < bsize; ii++){ // Can be changed to SIMD instruction
					D[height + ii] = Bv[ioffset + ii];
					Drcp[height + ii] = 1.0 / D[height + ii];
				}
				for (int j = Up[i]; j < Up[i + 1]; ++j)
					for(int ii = 0, joffset = j * bsize, voffset = (j - Up[i] + Bd[i] + 1) * bsize; ii < bsize; ii++) // Can be changed to SIMD instruction
						Uv[joffset + ii] = Bv[voffset + ii];
			}
					
	gettimeofday(&t2, NULL);
	double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
	std::cout << "\n--------------------------------------------------\n";
	std::cout << "LU(0) factorization Time: " << timeuse << std::endl;



//************* A = L + U  End ***********************

	delete[] Bd;
	// delete[] Bp;
	// delete[] Bi;
	// delete[] Bo;
	delete[] Bv;

	data.Ori.bsize = bsize;
	data.Ori.brow = nb;
	data.Ori.blk_ptr = Bp;
	data.Ori.col_ind = Bi;
	data.Ori.dia_offset = Bo;
	data.Ori.val = Bv2;

	data.Lower.brow = nb;
	// data.Lower.bsize = bsize;
	data.Lower.blk_ptr = Lp;
	data.Lower.col_ind = Li;
	data.Lower.dia_offset = Lo;
	data.Lower.val = Lv;

	data.Upper.brow = nb;
	// data.Upper.bsize = bsize;
	data.Upper.blk_ptr = Up;
	data.Upper.col_ind = Ui;
	data.Upper.dia_offset = Uo;
	data.Upper.val = Uv;

	data.Diagonal = D;
	data.DiagonalRecip = Drcp;
	
}



static void BuildOptData_BMC(CSRMatrix & A, OptData & data, int nx, int ny, int nz, int bx, int by, int bz, int cx, int cy, int cz)
{

	data.nx = nx;
	data.ny = ny;
	data.nz = nz;

	assert(!(nx % (cx * bx)));
	assert(!(ny % (cy * by)));
	assert(!(nz % (cz * bz)));

	int nxny = nx * ny;
	int LocalSize = nxny * nz;

	int bxby = bx * by;
	int cxcy = cx * cy;
	int NumColors = cz * cxcy;
	int BlockSize = bz * bxby;

	int veclen = LocalSize / NumColors / BlockSize;
	int len = VEC_LEN;
	while(veclen % len != 0)
		len /= 2;
	if( veclen >= len)
		veclen = len;
	data.Lower.bsize = veclen;
	data.Upper.bsize = veclen;

	data.LocalSize = LocalSize;
	data.NumColors = NumColors;
	data.BlockSize = BlockSize;
    data.ReorderMap = new int[LocalSize];

#pragma omp parallel for schedule(guided)
	for (int color = 0; color < NumColors; ++color)
		for(int iz = (color / cxcy) * bz; iz < nz; iz += cz * bz)
			for(int iy = ((color % cxcy) / cx) * by; iy < ny; iy += cy * by)
				for(int ix = (color % cx) * bx; ix < nx; ix += cx * bx)
					for(int ibz = 0; ibz < bz; ibz++)					
						for(int iby = 0; iby < by; iby++)
							for(int ibx = 0; ibx < bx; ibx++){
								int CurBlockNum = (iz/bz/cz) * (nx/bx/cx) * (ny/by/cy) + (iy/by/cy) * (nx/bx/cx) + (ix/bx/cx);
								int CurBlockToCoreNum = CurBlockNum / veclen;
								data.ReorderMap[(iz + ibz)*nxny + (iy+iby)*nx + ix+ibx] = color * (LocalSize / NumColors)
								+ CurBlockToCoreNum * BlockSize * veclen + veclen*(ibz*bxby+iby*bx+ibx) + (CurBlockNum - CurBlockToCoreNum*veclen);
							}
	
	Factorization(A, data);
}

static void ReorderVector(const OptData & data, const Vector & xorg, Vector & xrdr)
{
	int* ReorderMap = data.ReorderMap;
	double* xov = xorg.values;
	double* xrv = xrdr.values;
	int length = xorg.size;
	for (int i = 0; i < length; ++i)
		xrv[ReorderMap[i]] = xov[i];
}

void sizePrintf(CSRMatrix & A, OptData & data){
	int n = A.size[0];
	int* Ap = A.rowptr;
	int* Ai = A.colind;
	double* Av = A.values;
	printf("CSR:\n");
	printf("\tn = %d\n", n);
	printf("\trowptr = %d(int)\n", n+1);
	printf("\tcolind = %d(int)\n", Ap[n]);
	printf("\tvalues = %d(double)\n", Ap[n]);

	int nb = data.Ori.brow;
    int bsize = data.Ori.bsize;
	int * Bp = data.Ori.blk_ptr;
    int * Bi = data.Ori.col_ind;
    int * Bo = data.Ori.dia_offset;
    double * Bv = data.Ori.val;
	printf("DBSR:\n");
	printf("\tbsize = %d\n", bsize);
	printf("\tnb = %d\n", nb);
	printf("\tblkptr = %d(int)\n", nb+1);
	printf("\tcolind = %d(int)\n", Bp[nb]);
	printf("\toffset = %d(int)\n", Bp[nb]);
	printf("\tvalues = %d(double)\n", Bp[nb]*bsize);
}



void Laplacian27pt::OptimizeProblem(OptData & data, Vector & b, Vector & x){

#ifdef BMC_FIX
	BuildOptData_BMC(A, data, nx, ny, nz, 4, 4, 4, 2, 2, 2);

#elif defined(BMC_AUTO)
	int num_threads = VEC_LEN * omp_get_max_threads();
	int cx, cy, cz, mx, my, mz;
	cx = 2, cy = 2, cz = 2;
	mx = 1, my = 1, mz = 1;
	while (mx * my * mz < num_threads)
		(mx <= mz ? (my <= mx ? my : mx) : mz) *= 2;

	if(nx/(cx*mx)>1 && ny/(cy*my)>1 && nz/(cz*mz)>1)
		BuildOptData_BMC(A, data, nx, ny, nz, nx/(cx*mx), ny/(cy*my), nz/(cz*mz), cx, cy, cz);	
	else
		BuildOptData_BMC(A, data, nx, ny, nz, 4, 4, 4, cx, cy, cz);

#else
	BuildOptData_BMC(A, data, nx, ny, nz, nx/(2*BMC_MX), ny/(2*BMC_MY), nz/(2*BMC_MZ), 2, 2, 2);
#endif

	data.Reorder_CSR(A);

	Vector b1, x1;
	b1.Resize(b.size);
	x1.Resize(x.size);
	b1.Copy(b);
	x1.Copy(x);
	ReorderVector(data, b1, b);
	ReorderVector(data, x1, x);
	b1.Free();
	x1.Free();

	// sizePrintf(A, data);

}
