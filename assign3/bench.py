import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import time
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Ampere

print("\nCompiling softmax module...\n\n")

softmax_module = load(
    name="softmax_module",
    sources=['softmax.cu'],
    extra_cflags=["-O3"],
    verbose=True
)

naive_correctness_message = 'YOUR SOFTMAX PRODUCED INCORRECT RESULTS'

def time_softmax(func, *args, num_runs=1):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_runs):
        result = func(*args)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / num_runs
    return result, elapsed_time

def naive_softmax(x, y, bz, c):
    softmax_module.naive_softmax_forward(x, y, bz, c)
    return y

if __name__ == "__main__":
    batch_size, num_classes = 10000, 10000

    x = torch.randn(batch_size, num_classes, device='cuda', dtype=torch.float32)    
    print(f"\nTesting batch_size={batch_size}, num_classes={num_classes}")
    
    y_naive = torch.empty_like(x)
    
    result_student, time_student = time_softmax(
        naive_softmax, x, y_naive, batch_size, num_classes 
    )
    result_pytorch = F.softmax(x, dim=1)

    assert torch.allclose(result_pytorch, result_student, rtol=1e-5, atol=1e-8), naive_correctness_message

    print("Sanity check naive: ", torch.allclose(result_pytorch, result_student, rtol=1e-5, atol=1e-8))

    print("\nExecution time:")
    print(f"Your Softmax time:   {time_student:.4f} ms")

    NUM_FLOPS = (5*num_classes)*batch_size
    your_throughput = NUM_FLOPS / (time_student * 1e-3) / 1e9   

    print(f"\nThroughput:")
    print(f"Your Softmax:   {your_throughput:.3f} GFLOPS/s")
