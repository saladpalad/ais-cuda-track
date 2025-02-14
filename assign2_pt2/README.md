# Assignment 2 Part 2: Implementing HGEMM Kernels
Building upon the first part of the assignment, we now turn our attention to matrix multiplication in half-precision. Instead of the input matrices being FP32 (`A*B`), we now perform computation on FP16 matrices (`A_half*B_half`). To take advantage of the smaller size of FP16 values, we utilize tensor cores in the SM, which are designed to take advantage of the faster FP16 computation. There are 2 CUDA APIs to program the tensor cores in a RTX 3090, `wmma` and `mma`. In this assignment, we will be utilizing the `mma` PTX instructions. Finish the TODO PTX instructions in `ptx.h` and the TODO lines in `hgemm.cu`. Refer to [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#instructions) to do the assignment. Once done, run `python3 test_hgemm.py` to make sure each kernel passes the sanity checks. Then, profile them with `ncu --set full --export reports/hgemm_all python3 test_hgemm.py` and answer the following questions.

## Nsight Compute Profiling
### Kernel 1: Naive MMA
Given that we use matrices of size `M=N=K=4096` and each warp computes the mma instruction of size `m16n8k16`. How many mma instructions do we need to execute in total to compute the matrices of the given shape? Given that an RTX 3090 has 328 tensor cores where each one executes a single mma instruction at a time, how many instructions will each tensor core need to compute to execute all mma instructions? 

Look up the metric `sm__cycles_elapsed.avg.per_second` in NCU, using this number calculate the number of clock cycles it took the SM to compute a single MMA instruction (`sm__cycles_elapsed.avg` / number of mma instructions executed by a tensor core). For a 3090 the ideal number of clock cycles per MMA instruction is 32, was the number of clock cycles it took the SM to compute a single MMA instruction close at all? If not, why?

### Kernel 2: Permutated Shared Memory
What bitwise operator is used to implement the swizzling pattern in shared memory? Why do we use this bitwise operator? Compare the shared memory bank conflicts between Kernel 1 and 2, which has more? Why? Has our performance improved after accessing the elements in shared memory in this swizzling pattern?

### Kernel 3: CP Async
Look at the Memory Workload Analysis section and Memory Chart. \
Compare the `Global Memory Instructions`, and `Memory Throughput` between Kernel 2 and Kernel 3. What differences do you observe when using cp_async? Explain how the asynchronous memory operations contribute to these changes.
