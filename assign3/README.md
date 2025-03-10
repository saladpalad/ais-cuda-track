# Assignment 3: CUDA Softmax: The Need for Speed
You're in a lab working on AGI but the softmax layer is a bottleneck to your lab's frontier LLM models. After profiling the training pipeline, the lab has identified PyTorch's native softmax function as the primary performance limitation. 
The previous CUDA implementation developed by our team failed to deliver meaningful speedups, so you've been assigned to create a more optimized version. 
The goal is to develop a CUDA-based softmax algorithm that can run on your lab's cluster of NVIDIA GPUs, while also significantly increasing GFLOP performance. Use techniques such as parallel reduction, warp-level primitives and vectorization to improve performance to make it go fast!

## To get started:
Optimize the naive kernel in `softmax.cu`,  \
Use profiling tools like NSight Compute to identify any bottlenecks \
Run `python3 bench.py` to benchmark the softmax kernel 

Good luck!
