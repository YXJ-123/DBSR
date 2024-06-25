#include "SparseMatrix.hpp"



void CSRMatVec(const CSRMatrix& A, const Vector& x, const Vector& y)
{
	int n = A.size[0];
	int* Ap = A.rowptr;
	int* Ai = A.colind;
	double* Av = A.values;
	double* xv = x.values;
	double* yv = y.values;

#pragma omp parallel for schedule(guided)
	for (int i = 0; i < n; ++i)
	{
		double temp = 0.0;
		for(int j = Ap[i]; j < Ap[i+1]; j++)
			temp += Av[j] * xv[Ai[j]];
		yv[i] = temp;
	}
}




Vector::Vector()
	: ref(0),
	size(0),
	values(0)
{
}

Vector::Vector(int _n)
	: ref(0),
	size(_n),
	values(new double[_n])
{
}

Vector::Vector(int _n, double* _values, int _ref)
	: ref(_ref),
	size(_n),
	values(_values)
{
}

Vector::Vector(const Vector& x)
	: ref(0),
	size(x.size),
	values(new double[x.size])
{
	double* xv = x.values;
#pragma omp parallel for
	for(int i = 0; i < size; i++)
		values[i] = xv[i];
}

Vector::~Vector()
{
	if (!ref && values)
		delete[] values;
}

void Vector::Resize(int n)
{
	if (!ref && values)
		delete[] values;
	size = n;
	// values = new double[n];
	values = (double *)aligned_alloc(64, n * sizeof(double));
	ref = 0;
}

void Vector::Fill(double a) const
{
#pragma omp parallel for
	for(int i = 0; i < size; i++)
		values[i] = a;
}

void Vector::Copy(const Vector& x) const
{
	int size = x.size;
	double* xv = x.values;
#pragma omp parallel for
	for( int i = 0; i < size; i++)
		values[i] = xv[i];
}

void Vector::AddScaled(double a, const Vector& x) const
{
	double* xv = x.values;
#pragma omp parallel for
	for(int i = 0; i < size; i++)
		values[i] += a * xv[i];

}

void VecAXPBY(double alpha, const Vector& x, double beta, const Vector& y)
{
	int n = y.size;
	double* xv = x.values;
	double* yv = y.values;
#pragma omp parallel for
	for(int i = 0; i < n; i++)
		yv[i] = alpha * xv[i] + beta * yv[i];
}

double VecDot(const Vector& x, const Vector& y)
{
	double result = 0.0;
	int n = x.size;
	double* xv = x.values;
	double* yv = y.values;

#pragma omp parallel for reduction(+:result)
	for (int i = 0; i < n; ++i)
		result += xv[i] * yv[i];

	return result;
}

void Vector::Free()
{
	if (!ref && values)
		delete[] values;
	size = 0;
	values = 0;
	ref = 0;
}





CSRMatrix::CSRMatrix()
	: ref(0),
	size{0, 0},
	rowptr(0),
	colind(0),
	values(0)
{
}

CSRMatrix::CSRMatrix(int _n, int _m, int* _rowptr, int* _colind, double* _values, int _ref)
	: ref(_ref),
	size{_n, _m},
	rowptr(_rowptr),
	colind(_colind),
	values(_values)
{
}

CSRMatrix::~CSRMatrix()
{
	if (!ref)
	{
		if (rowptr) delete[] rowptr;
		if (colind) delete[] colind;
		if (values) delete[] values;
	}
}

void CSRMatrix::Free()
{
	if (!ref)
	{
		if (rowptr) delete[] rowptr;
		if (colind) delete[] colind;
		if (values) delete[] values;
	}
	size[0] = 0;
	size[1] = 0;
	rowptr = 0;
	colind = 0;
	values = 0;
	ref = 0;
}

void CSRMatrix::Apply(const Vector& x, const Vector& y) const
{
	CSRMatVec(*this, x, y);
}

int CSRMatrix::InSize() const
{
	return size[1];
}

int CSRMatrix::OutSize() const
{
	return size[0];
}
