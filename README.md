# Week 1: Intro to GPUs and writing your first kernel!
![gpu-devotes-more-transistors-to-data-processing](https://github.com/user-attachments/assets/2aca8245-ad88-4613-8b73-f94ad395edf4)

#### Can you guess which architecture more closely resembles a CPU? What about a GPU?
### Recommended Readings:
[Motivation for GPUs in Deep Learning](https://horace.io/brrr_intro.html)\
[A gentle introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
### Further resources/references to use:
[PMPP Book Access](https://dokumen.pub/qdownload/programming-massively-parallel-processors-a-hands-on-approach-4nbsped-9780323912310.html)\
[NVIDIA GPU Glossary](https://modal.com/gpu-glossary/device-hardware)

# Week 2 and 3: Learning to optimize your kernels! 
![matmul](https://github.com/user-attachments/assets/494a758f-cc52-4dc3-8454-63181d3786c8)

#### From the image, how many FLOPS (floating point operations) are in matrix multiplication?

### Recommended Readings: 
[Aalto University's Course on GPU Programming](https://ppc.cs.aalto.fi/ch4/)\
[Simon's Blog on SGEMM (Kernels 1-5 are the most relevant for the assignment)](https://siboehm.com/articles/22/CUDA-MMM)\
[How to use NCU profiler](https://www.youtube.com/watch?v=04dJ-aePYpE)

### Further references to use:
[NCU Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)

# Week 4: Learning to optimize your kernels with Tensor Cores!
![Tensor-Core-Matrix](https://github.com/user-attachments/assets/d6209037-dd9b-4285-b71e-d3df5184ea2a)
#### How much faster are Tensor Core operations compared to F32 CUDA Cores?


### Recommended Readings:
[A sequel to Simon's Blog in HGEMM](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html)\
[Bruce's Blog on HGEMM](https://bruce-lee-ly.medium.com/nvidia-tensor-core-cuda-hgemm-advanced-optimization-5a17eb77dd85)\
[NVIDIA's Presentation on A100 Tensor Cores](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf)

### Further references to use:
[Primer on Inline PTX Assembly](https://docs.nvidia.com/cuda/pdf/Inline_PTX_Assembly.pdf)\
[CUTLASS GEMM Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md#shared-memory-layouts)
[NVIDIA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma#)
