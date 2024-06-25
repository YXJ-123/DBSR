#include "OptData.hpp"


DataStruct::DataStruct(/* args */)
{
}

DataStruct::~DataStruct()
{
}

MatrixStruct::MatrixStruct(/* args */)
{
}

MatrixStruct::~MatrixStruct()
{
}


DBSR::DBSR()
	: brow(0),
	bsize(0),
	blk_ptr(0),
	col_ind(0),
	dia_offset(0),
	val(0)
{
}

DBSR::DBSR(int _brow, int _bsize, int* _blk_ptr, int* _col_ind, int* _dia_offset, double* _val)
	:brow(_brow),
	bsize(_bsize),
	blk_ptr(_blk_ptr),
	col_ind(_col_ind),
	dia_offset(_dia_offset),
	val(_val)
{
}

DBSR::~DBSR()
{
	if(blk_ptr) delete[] blk_ptr;
	if(col_ind) delete[] col_ind;
	if(dia_offset) delete[] dia_offset;
	if(val) delete[] val;
}


OptData::OptData()
	:nx(0),
	ny(0),
	nz(0),
	LocalSize(0),	
	NumColors(0),
	BlockSize(0),
	ReorderMap(0),
	Diagonal(0),
	DiagonalRecip(0)
{
}

OptData::~OptData()
{
	if(ReorderMap) delete[] ReorderMap;
	if(Diagonal) delete[] Diagonal;
	if(DiagonalRecip) delete[] DiagonalRecip;
}


static inline void swap(int& a, int& b)
{
	int temp = a;
	a = b;
	b = temp;
}

static inline void swap(double& a, double& b)
{
	double temp = a;
	a = b;
	b = temp;
}

static void SortIndAsc(int* a, int left, int right)
{
	if (left >= right) return;

	swap(a[left], a[(left + right) / 2]);

	int last = left;
	for (int i = left + 1; i <= right; ++i)
	{
		if (a[i] < a[left])
		{
			swap(a[++last], a[i]);
		}
	}

	swap(a[left], a[last]);

	SortIndAsc(a, left, last - 1);
	SortIndAsc(a, last + 1, right);
}

static void SortIndAsc(int* a, double* x, int left, int right)
{
	if (left >= right) return;

	swap(a[left], a[(left + right) / 2]);
	swap(x[left], x[(left + right) / 2]);

	int last = left;
	for (int i = left + 1; i <= right; ++i)
	{
		if (a[i] < a[left])
		{
			swap(a[++last], a[i]);
			swap(x[last], x[i]);
		}
	}

	swap(a[left], a[last]);
	swap(x[left], x[last]);

	SortIndAsc(a, x, left, last - 1);
	SortIndAsc(a, x, last + 1, right);
}

void OptData::Reorder_CSR(CSRMatrix & A){
	int n = A.size[0];
	int* Ap = A.rowptr;
	int* Ai = A.colind;
	double* Av = A.values;

	int* Rp = new int[n + 1];
	int* Ri = new int[Ap[n]];
	double* Rv = new double[Ap[n]];
	Rp[0] = 0;

#pragma omp parallel for
	for(int i = 0; i < n; i++)
		Rp[ReorderMap[i] + 1] = Ap[i + 1] - Ap[i];
	
	for(int i = 0; i < n; i++)
		Rp[i + 1] += Rp[i];

	#pragma omp parallel for schedule(guided)
		for(int i = 0; i < n; i++){
			for(int j = Ap[i], t = Rp[ReorderMap[i]]; j < Ap[i + 1]; j++, t++){
				Ri[t] = ReorderMap[Ai[j]];
				Rv[t] = Av[j];
			}
			SortIndAsc(Ri, Rv, Rp[ReorderMap[i]], Rp[ReorderMap[i] + 1] - 1);
		}

	A.rowptr = Rp;
	A.colind = Ri;
	A.values = Rv;
	delete[] Ap;
	delete[] Ai;
	delete[] Av;

}
