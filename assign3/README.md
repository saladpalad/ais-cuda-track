# Assignment 3: CUDA Softmax: You're too slow!
<img src="https://github.com/user-attachments/assets/44c161d3-9b11-4ea8-9c65-ccc1202d165c" width="250" alt="artworks-000296633391-rpazic">

<img src="https://github.com/user-attachments/assets/67bd7a9a-d9ee-4447-a17e-1ac8aa59e1ad" width="300px" alt="System diagram">

You're in a lab working towards AGI but unfortunately a competing open-source lab has been getting all the praise for its frontier LLMs. After profiling the training pipeline, the lab has identified PyTorch's native softmax function as the primary performance limitation. The previous CUDA implementation developed by your team failed to deliver meaningful speedups, so you've been assigned to create a more optimized version. The goal is to develop a CUDA-based softmax algorithm that can beat out the competitor's GFLOP performance and put your lab back at the forefront of AGI research. Use techniques such as **parallel reduction**, **warp-level primitives** and **vectorization** to improve throughput.

## To get started:
- Install the necessary packages in your environment
- Optimize the naive kernel in `softmax.cu`
- Use profiling tools like `NSight Compute` to identify any bottlenecks
- Run `python3 bench.py` to benchmark the softmax kernel 

Good luck!
