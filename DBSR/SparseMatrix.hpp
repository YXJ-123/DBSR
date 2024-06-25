
#include <assert.h>
#include <iostream>

struct Vector
{
	int ref;
	int size;
	double* values;

	Vector();
	Vector(int n);
	Vector(int n, double* values, int ref);
	Vector(const Vector& x);
	~Vector();

	void Free();
	void Resize(int n);
	void Fill(double a) const;
	void Copy(const Vector& x) const;
	void AddScaled(double a, const Vector& x) const;
};

void VecAXPBY(double alpha, const Vector& x, double beta, const Vector& y);
double VecDot(const Vector& x, const Vector& y);


struct CSRMatrix
{
	int ref;
	int size[2];

	int* rowptr;
	int* colind;
	double* values;

	CSRMatrix();
	CSRMatrix(int n, int m, int* rowptr, int* colind, double* values, int ref);
	// CSRMatrix(const COOMatrix& A);
	~CSRMatrix();

	void Free();
	int InSize() const;
	int OutSize() const;
	void Apply(const Vector& x, const Vector& y) const;
};