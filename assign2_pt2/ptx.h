#include <stdint.h>
__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
}

__forceinline__ 
__device__ void load_matrix_x4(unsigned *destReg, uint4 *srcAddr) {
  unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
  asm volatile(
    // This function loads four 8x8 matrices which forms a 16x16 matrix for A
    // Hint: Look at the function definition to figure out what ldmatrix ptx instruction
    // Hint: We are loading matrices of F16 from shared so .shared, and memory access width is 16-bit or .b16
      " TODO " // TODO Fill in the " " with the correct ldmatrix instruction
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(destReg[0]), "=r"(destReg[1]), "=r"(destReg[2]), "=r"(destReg[3])
      :  "r"(ptxSrcAddr)
      );
}

__forceinline__ 
__device__ void load_matrix_x2(unsigned *destReg, uint4 *srcAddr) {
  unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
  asm volatile(
    // This function loads 2 8x8 matrices which forms a 16x8 matrix for B
    // Hint: This time we are loading 2 matrix fragments
      " TODO " // TODO Fill in the " " with the correct ldmatrix instruction
      "{%0, %1}, [%2];\n"
      : "=r"(destReg[0]), "=r"(destReg[1])
      :  "r"(ptxSrcAddr)
      );
}

__forceinline__ 
__device__ void mma_m16n8k16(const unsigned *A, const unsigned *B, float *C, float *D) {
  asm(
    // Formula: D = A*B + C 
    // A tile has shape 16x16, B tile has shape 16x8, so D will have final shape 16x8 
    // Hint: Look at the function definition to figure out what mma ptx instruction and shape we should use.
    // Hint: C, D are the same matrix
    // Hint: A,B is F16 and C, D is F32
      " TODO " // TODO Fill in the " " with the correct mma instruction
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
      "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
     );
}

__forceinline__
__device__ void cp_async(uint4 *dstAddr, const uint4 *srcAddr) {
  unsigned ptxDstAddr = __cvta_generic_to_shared(dstAddr);
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
      :: "r"(ptxDstAddr),
      "l"(srcAddr),
      "n"(16));
}

