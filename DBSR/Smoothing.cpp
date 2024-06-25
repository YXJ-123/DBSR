#include "Laplacian27pt.hpp"

#ifdef _AVX512
  #include <immintrin.h>
#endif

#ifdef _NEON
#include <arm_neon.h>
#define step_size 2
#endif

#ifdef _AVX512
void DBSRMatVec_AVX(double alpha, const OptData & data, const Vector& x, double beta, const Vector& y)
{
    double * xv = x.values;
    double * yv = y.values;

    int * Ap = data.Ori.blk_ptr;
    int * Ai = data.Ori.col_ind;
    int * Ao = data.Ori.dia_offset;
    double * Av = data.Ori.val;

    int nb = data.Ori.brow;
    int bsize = data.Ori.bsize;

if(alpha == -1 && beta == 1){

    #pragma omp parallel for schedule(guided)
	for(int i = 0; i < nb; i++){
		int height = bsize * i;
		__m512d temp = _mm512_load_pd(yv + height);

		for(int j = Ap[i]; j < Ap[i + 1]; j++){
			int joffset = Ai[j] * bsize + Ao[j];
            __m512d A_mul_x;
            if(Ao[j] != 0){
					__m512i ind_x = _mm512_set_epi64(joffset + 7, joffset + 6, joffset + 5, joffset + 4,
												 joffset + 3, joffset + 2, joffset + 1, joffset);
					A_mul_x = _mm512_mul_pd(_mm512_load_pd(Av + j * bsize), _mm512_i64gather_pd(ind_x, xv, 8));
				}
            else{
        		A_mul_x = _mm512_mul_pd(_mm512_load_pd(Av + j * bsize), _mm512_load_pd(xv + joffset));
            }
            temp = _mm512_sub_pd(temp, A_mul_x);
		}

		_mm512_store_pd(yv + height, temp);
    }
}
}
#endif

#ifdef _NEON
void DBSRMatVec_NEON(double alpha, const OptData & data, const Vector& x, double beta, const Vector& y)
{
    double * xv = x.values;
    double * yv = y.values;

    int * Ap = data.Ori.blk_ptr;
    int * Ai = data.Ori.col_ind;
    int * Ao = data.Ori.dia_offset;
    double * Av = data.Ori.val;

    int nb = data.Ori.brow;
    int bsize = data.Ori.bsize;
    int neon_steps = bsize / step_size;

if(alpha == -1 && beta == 1){

    #pragma omp parallel for schedule(guided)
	for(int i = 0; i < nb; i++){
		int height = bsize * i;

        float64x2_t temp[neon_steps];
        for(int c = 0; c < neon_steps; c++)
			temp[c] = vld1q_f64(yv + height + c * step_size);

		for(int j = Ap[i]; j < Ap[i + 1]; j++){
			int joffset = Ai[j] * bsize + Ao[j];
            for(int c = 0; c < neon_steps; c++){
				float64x2_t A_mul_x = vmulq_f64( vld1q_f64(Av + j * bsize + c * step_size), vld1q_f64(xv + joffset + c * step_size));
                temp[c] = vsubq_f64(temp[c], A_mul_x);
            }
		}
        for(int c = 0; c < neon_steps; c++)
			vst1q_f64(yv + height + c * step_size, temp[c]);
    }
}
}
#endif

void DBSRMatVec(double alpha, const OptData & data, const Vector& x, double beta, const Vector& y)
{
    double * xv = x.values;
    double * yv = y.values;

    int * Ap = data.Ori.blk_ptr;
    int * Ai = data.Ori.col_ind;
    int * Ao = data.Ori.dia_offset;
    double * Av = data.Ori.val;

    int nb = data.Ori.brow;
    int bsize = data.Ori.bsize;

if(alpha == -1 && beta == 1){

    #pragma omp parallel for schedule(guided)
	for(int i = 0; i < nb; i++){
		int height = bsize * i;
		double temp[bsize];
		for(int ii = 0; ii < bsize; ii++)		// Can be changed to SIMD instruction
			temp[ii] = yv[height + ii];

		for(int j = Ap[i]; j < Ap[i + 1]; j++){
            int joffset = Ai[j] * bsize + Ao[j];
            int len = j * bsize;
			for(int k = 0, joffset = Ai[j] * bsize + Ao[j], len = j * bsize; k < bsize; k++)		// Can be changed to SIMD instruction
				temp[k] -= Av[len + k] * xv[joffset + k];
        }

		for(int ii = 0; ii < bsize; ii++)		// Can be changed to SIMD instruction
			yv[height + ii] = temp[ii];
    }
}

}

void OptMatVec(double alpha, const CSRMatrix& A, const OptData & data, const Vector& x, double beta, const Vector& y)
{
    #ifdef _AVX512
        DBSRMatVec_AVX(alpha, data, x, beta, y);
    #elif defined _NEON
        DBSRMatVec_NEON(alpha, data, x, beta, y);
    #else
        DBSRMatVec(alpha, data, x, beta, y);
    #endif
}

void DBSR_ILU0_Solve(CSRMatrix & A, OptData & data, const Vector & b, Vector & x, bool xis = 0 )
{
    double * Drcp = data.DiagonalRecip;
    double * D = data.Diagonal;
    double * xv = x.values;

    int * Lp = data.Lower.blk_ptr;
    int * Li = data.Lower.col_ind;
    int * Lo = data.Lower.dia_offset;
    double * Lv = data.Lower.val;
    int * Up = data.Upper.blk_ptr;
    int * Ui = data.Upper.col_ind;
    int * Uo = data.Upper.dia_offset;
    double * Uv = data.Upper.val;

	int NumColors = data.NumColors;
    int nb = data.Lower.brow;
	int blkNumPerColor = nb / NumColors;
	int blksPerCore = data.BlockSize;
    int bsize = data.Lower.bsize;

    Vector r;
    r.Resize(b.size);
    r.Copy(b);
    double * rv = r.values;
	
    if(xis)
        OptMatVec(-1.0, A, data, x, 1.0, r);

    // L_Solve
	for (int color = 0; color < NumColors; ++color)
	#pragma omp parallel for schedule(guided)
		for(int ib = blkNumPerColor * color; ib < blkNumPerColor * (color + 1); ib += blksPerCore)
            for(int i = ib; i < ib + blksPerCore; i++){
                int height = bsize * i;
                
                double temp[bsize];
                for(int ii = 0; ii < bsize; ii++)
                    temp[ii] = rv[height + ii];

                for(int j = Lp[i]; j < Lp[i + 1]; j++)
                    for(int k = 0, joffset = Li[j] * bsize + Lo[j], len = j * bsize; k < bsize; k++)		// Can be changed to SIMD instruction
                        temp[k] -= Lv[len + k] * rv[joffset + k];

                for(int ii = 0; ii < bsize; ii++)
                    rv[height + ii] = temp[ii];
            }


    // U_Solve
    for (int color = NumColors - 1; color >= 0; --color)
    #pragma omp parallel for schedule(guided)
        for(int ib = blkNumPerColor * (color + 1) - 1; ib >= blkNumPerColor * color; ib -= blksPerCore)
            for(int i = ib; i > ib - blksPerCore; i--){
                int height = bsize * i;

                double temp[bsize];
                for(int ii = 0; ii < bsize; ii++)
                    temp[ii] = rv[height + ii];

                for(int j =  Up[i + 1] - 1; j >= Up[i]; j--)
                    for(int k = bsize - 1, joffset = Ui[j] * bsize + Uo[j], len = j * bsize; k >= 0; k--)		// Can be changed to SIMD instruction
                        temp[k] -= Uv[len + k] * rv[joffset + k];

                for(int ii = 0; ii < bsize; ii++){
                    temp[ii] *= Drcp[height + ii];
                    rv[height + ii] = temp[ii];
                    xv[height + ii] += temp[ii];
                }
            }
}

#ifdef _AVX512
void DBSR_ILU0_Solve_AVX(CSRMatrix & A, OptData & data, const Vector & b, Vector & x, bool xis = 0 )
{
    double * Drcp = data.DiagonalRecip;
    double * D = data.Diagonal;
    double * xv = x.values;

    int * Lp = data.Lower.blk_ptr;
    int * Li = data.Lower.col_ind;
    int * Lo = data.Lower.dia_offset;
    double * Lv = data.Lower.val;
    int * Up = data.Upper.blk_ptr;
    int * Ui = data.Upper.col_ind;
    int * Uo = data.Upper.dia_offset;
    double * Uv = data.Upper.val;

	int NumColors = data.NumColors;
    int nb = data.Lower.brow;
	int blkNumPerColor = nb / NumColors;
	int blksPerCore = data.BlockSize;
    int bsize = data.Lower.bsize;

    Vector r;
    r.Resize(b.size);
    r.Copy(b);
    double * rv = r.values;
	
    if(xis)
        OptMatVec(-1.0, A, data, x, 1.0, r);

    // L_Solve
	for (int color = 0; color < NumColors; ++color)
	#pragma omp parallel for schedule(guided)
		for(int ib = blkNumPerColor * color; ib < blkNumPerColor * (color + 1); ib += blksPerCore)
            for(int i = ib; i < ib + blksPerCore; i++){
                int height = bsize * i;
                __m512d temp = _mm512_load_pd(rv + height);

                for(int j = Lp[i]; j < Lp[i + 1]; j++){
                    int joffset = Li[j] * bsize + Lo[j];
                    __m512d L_mul_x = _mm512_mul_pd(_mm512_load_pd(Lv + j * bsize), _mm512_load_pd(rv + joffset));
                    temp = _mm512_sub_pd(temp, L_mul_x);
                }
                _mm512_store_pd(rv + height, temp);
            }


    // U_Solve
    for (int color = NumColors - 1; color >= 0; --color)
    #pragma omp parallel for schedule(guided)
        for(int ib = blkNumPerColor * (color + 1) - 1; ib >= blkNumPerColor * color; ib -= blksPerCore)
            for(int i = ib; i > ib - blksPerCore; i--){
                int height = bsize * i;

                __m512d temp = _mm512_load_pd(rv + height);
                __m512d temp2 = _mm512_load_pd(rv + height + 8);

                for(int j =  Up[i + 1] - 1; j >= Up[i]; j--){
                    int joffset = Ui[j] * bsize + Uo[j];
                    __m512d U_mul_x = _mm512_mul_pd(_mm512_load_pd(Uv + j * bsize), _mm512_load_pd(rv + joffset));
                    temp = _mm512_sub_pd(temp, U_mul_x);
                }

                temp = _mm512_mul_pd(temp, _mm512_load_pd(Drcp + height));
                _mm512_store_pd(rv + height, temp);
                temp = _mm512_add_pd(temp, _mm512_load_pd(xv + height));
                _mm512_store_pd(xv + height, temp);
            }
}
#endif

#ifdef _NEON
void DBSR_ILU0_Solve_NEON(CSRMatrix & A, OptData & data, const Vector & b, Vector & x, bool xis = 0 )
{
    double * Drcp = data.DiagonalRecip;
    double * D = data.Diagonal;
    double * xv = x.values;

    int * Lp = data.Lower.blk_ptr;
    int * Li = data.Lower.col_ind;
    int * Lo = data.Lower.dia_offset;
    double * Lv = data.Lower.val;
    int * Up = data.Upper.blk_ptr;
    int * Ui = data.Upper.col_ind;
    int * Uo = data.Upper.dia_offset;
    double * Uv = data.Upper.val;

	int NumColors = data.NumColors;
    int nb = data.Lower.brow;
	int blkNumPerColor = nb / NumColors;
	int blksPerCore = data.BlockSize;
    int bsize = data.Lower.bsize;
    int neon_steps = bsize / step_size;

    Vector r;
    r.Resize(b.size);
    r.Copy(b);
    double * rv = r.values;
	
    if(xis)
        // ParCSRMatVec(-1.0, A, x, 1.0, r);
        OptMatVec(-1.0, A, data, x, 1.0, r);

    // L_Solve
	for (int color = 0; color < NumColors; ++color)
	#pragma omp parallel for schedule(guided)
		for(int ib = blkNumPerColor * color; ib < blkNumPerColor * (color + 1); ib += blksPerCore)
            for(int i = ib; i < ib + blksPerCore; i++){
                int height = bsize * i;

                float64x2_t temp[neon_steps];
                for(int c = 0; c < neon_steps; c++)
			        temp[c] = vld1q_f64(rv + height + c * step_size);

                for(int j = Lp[i]; j < Lp[i + 1]; j++){
                    int joffset = Li[j] * bsize + Lo[j];
                    for(int c = 0; c < neon_steps; c++){
                        float64x2_t L_mul_x = vmulq_f64( vld1q_f64(Lv + j * bsize + c * step_size), vld1q_f64(rv + joffset + c * step_size));
                        temp[c] = vsubq_f64(temp[c], L_mul_x);
                    }
                }
                for(int c = 0; c < neon_steps; c++)
			        vst1q_f64(rv + height + c * step_size, temp[c]);
            }


    // U_Solve
    for (int color = NumColors - 1; color >= 0; --color)
    #pragma omp parallel for schedule(guided)
        for(int ib = blkNumPerColor * (color + 1) - 1; ib >= blkNumPerColor * color; ib -= blksPerCore)
            for(int i = ib; i > ib - blksPerCore; i--){
                int height = bsize * i;

                float64x2_t temp[neon_steps];
                for(int c = 0; c < neon_steps; c++)
                    temp[c] = vld1q_f64(rv + height + c * step_size);

                for(int j = Up[i]; j < Up[i + 1]; j++){
                    int joffset = Ui[j] * bsize + Uo[j];
                    for(int c = 0; c < neon_steps; c++){
                        float64x2_t U_mul_x = vmulq_f64( vld1q_f64(Uv + j * bsize + c * step_size), vld1q_f64(rv + joffset + c * step_size));
                        temp[c] = vsubq_f64(temp[c], U_mul_x);
                    }
                }
                for(int c = 0; c < neon_steps; c++){
                    temp[c] = vmulq_f64(temp[c], vld1q_f64(Drcp + height + c * step_size));
                    vst1q_f64(rv + height + c * step_size, temp[c]);
                    temp[c] = vaddq_f64(temp[c], vld1q_f64(xv + height + c * step_size));
                    vst1q_f64(xv + height + c * step_size, temp[c]);
                }
            }
}
#endif

void ILU0_Solve(CSRMatrix & A, OptData & data, const Vector & b, Vector & x, bool xis = 0 ){
    #ifdef _AVX512
        DBSR_ILU0_Solve_AVX(A, data, b, x, xis);
    #elif defined _NEON
        DBSR_ILU0_Solve_NEON(A, data, b, x, xis);
    #else
        DBSR_ILU0_Solve(A, data, b, x, xis);
    #endif
}

void Laplacian27pt::ILU0_DBSR(OptData & data, Vector & b, Vector & x, bool xis) {

    ILU0_Solve(A, data, b, x, xis);

}
