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
![gemm1](https://github.com/user-attachments/assets/d0349f57-d436-459e-920f-5b445a3771fa)

#### From the image, how many FLOPS (floating point operations) are in matrix multiplication?

### Recommended Readings: 
[Aalto University's Course on GPU Programming](https://ppc.cs.aalto.fi/ch4/)\
[Simon's Blog on SGEMM (Kernels 1-5 are the most relevant for the assignment)](https://siboehm.com/articles/22/CUDA-MMM)\
[How to use NCU profiler](https://www.youtube.com/watch?v=04dJ-aePYpE)\
[Roofline Models](https://www.telesens.co/2018/07/26/understanding-roofline-charts/)

### Further references to use:
[NCU Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)

# Week 4 and 5: Learning to optimize with Tensor Cores!
![Tensor-Core-Matrix](https://github.com/user-attachments/assets/d6209037-dd9b-4285-b71e-d3df5184ea2a)
#### How much faster are Tensor Core operations compared to F32 CUDA Cores?


### Recommended Readings:
[A sequel to Simon's Blog in HGEMM](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html)\
[Bruce's Blog on HGEMM](https://bruce-lee-ly.medium.com/nvidia-tensor-core-cuda-hgemm-advanced-optimization-5a17eb77dd85)\
[Spatter's Blog on HGEMM](https://www.spatters.ca/mma-matmul)\
[NVIDIA's Presentation on A100 Tensor Cores](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf)

### Further references to use:
[Primer on Inline PTX Assembly](https://docs.nvidia.com/cuda/pdf/Inline_PTX_Assembly.pdf)\
[CUTLASS GEMM Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md#shared-memory-layouts)\
[NVIDIA PTX ISA Documentation (Chapter 9.7 is most relevant)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma#)

# Week 6: Exploring other optimization parallel techniques!
![1_l1uoTZpQUW8YaSjFpcMNlw](https://github.com/user-attachments/assets/3d2997f7-d149-4668-a48c-39b3fc516f1a)

#### How could we compute the sum of all the elements in a 1-million sized vector?

### Recommended Readings:
[Primer on Parallel Reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)\
[Warp level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)\
[Vectorization](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)\
[Efficient Softmax Kernel](https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031)\
[Online Softmax Paper](https://arxiv.org/pdf/1805.02867)

# Week 7 & 8: Putting it all together in Flash Attention!
![0_maKQLOzxf4mK3B4O](https://github.com/user-attachments/assets/89814742-9d3c-47b2-b2f2-ee9304a71dce)

#### Is the self-attention layer in LLMs compute-bound or memory-bound?

### Recommended Readings:
[Flash Attention V1 Paper](https://arxiv.org/pdf/2205.14135)\
[Aleksa Gordic's Flash Attention Blog](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
