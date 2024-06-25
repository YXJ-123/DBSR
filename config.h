

// Please select BMC's division scheme and keep only one.
#define BMC_AUTO        // Adaptive division
// #define BMC_FIX      // Fixed block size 4x4x4
// #define BMC_DYNAMIC  // Custom Dynamic Division

// If BMC_DYNAMIC is selected, redefine the number of blocks, otherwise ignore them!
 #define BMC_MX 4
 #define BMC_MY 4
 #define BMC_MZ 4

// Please select the vector length
#define VEC_LEN 8


// If VEC_LEN = 8, select the appropriate SIMD, otherwise turn them all off!
// #define _AVX512
// #define _NEON
