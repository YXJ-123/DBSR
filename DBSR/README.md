### DBSR

The Sparse Triangular Solver (SPTRSV) plays a critical role in solving the sparse linear system $Ax=b$ on structured grids.
Yet, the commonly used sparse matrix storage formats for structured grid methods do not efficiently support SPTRSV in utilizing the instruction parallelism offered by modern multi-core CPUs.  
This work introduce DBSR, a new sparse storage format to enable SPTRSV to take advantage of the SIMD instructions. 
The DBSR format provides high parallelism and reliable convergence for iterative-based sparse linear solver.
It reduces the matrix storage overhead and supports continuous memory access and vectorized computation.

This work integrate DBSR into the zero-fill ILU (ILU(0)) preconditioner to effectively support the implementation of a vectorized ILU(0) algorithm. 
ILU(0) is one of the most widely used preconditioners. The DBSR strategy is applicable in both the LU factorization phase and the smoothing phase.

### Default

DBSR is based on a block multi-color parallelism strategy to vectorize and reorder parallelizable blocks. 
In the current code, the default block multicolor strategy is adaptively divided based on the number of threads, and the reordering vector length (i.e., the size of the data block) is 8. 
To modify them, follow the instructions in *"config.h"*.

### Compile and install

You can use the command `` make`` to build the binary file ``ILU0-DBSR``.

If you want to use the SIMD instruction, please change the options in the Makefile.

### Getting Started

This work starts by creating a 3D 27-point stencil structured matrix. This matrix is then used to solve a system of sparse linear equations using the ILU(0) preconditioner in DBSR format. 
Smoothing stops until the residual is small enough. Finally, the times for the factorization and smoothing phases will be printed separately.

The command 

```
export OMP_NUM_THREADS=16
./ILU0-DBSR -n 128 128 128
```

will execute the program, where `-n` is used to determine the sacle of the 3D grid in direction x, y, z.

### Note

We will keep these files and datas maintained. Please contact me if there are any new findings in the testing.

