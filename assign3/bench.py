import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import softmax_ref as softmax_ref
import time
import os
import random

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Ampere
print("\nCompiling softmax module...\n\n")
softmax_module = load(
    name="softmax_module",
    sources=['softmax.cu'],
    extra_cflags=["-O3"],
    verbose=True
)

naive_correctness_message = 'YOUR SOFTMAX PRODUCED INCORRECT RESULTS'
optimized_correctness_message = 'REFERENCE SOFTMAX PRODUCED INCORRECT RESULTS'

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

def pytorch_softmax(x, y):
    return F.softmax(x, dim=1)

def naive_softmax(x, y, bz, c):
    softmax_module.naive_softmax_forward(x, y, bz, c)
    return y

def optimized_softmax(x, y, bz, c):
    softmax_ref.optimized_softmax_forward(x, y, bz, c)
    return y

if __name__ == "__main__":
    batch_size, num_classes = 10000, 10000
    x = torch.randn(batch_size, num_classes, device='cuda', dtype=torch.float32)    
    print(f"\nTesting batch_size={batch_size}, num_classes={num_classes}")
    
    y_torch = torch.empty_like(x)
    y_naive = torch.empty_like(x)
    y_optimized = torch.empty_like(x)
    
    result_pytorch = F.softmax(x, dim=1)
    result_naive, time_naive = time_softmax(
        naive_softmax, x, y_naive, batch_size, num_classes 
    )
    result_optimized, time_optimized = time_softmax(
        optimized_softmax, x, y_optimized, batch_size, num_classes 
    )
   
    assert torch.allclose(result_pytorch, result_naive, rtol=1e-5, atol=1e-8), naive_correctness_message
    assert torch.allclose(result_pytorch, result_optimized, rtol=1e-5, atol=1e-8), optimized_correctness_message
    
    print("\nExecution time:")
    print(f"YOUR Softmax time:   {time_naive:.4f} ms")
    print(f"REFERENCE Softmax time:   {time_optimized:.4f} ms")   
    
    NUM_FLOPS = (5*num_classes)*batch_size
    naive_throughput = NUM_FLOPS / (time_naive * 1e-3) / 1e9   
    optimized_throughput = NUM_FLOPS / (time_optimized * 1e-3) / 1e9   
    
    print(f"\nThroughput:")
    print(f"YOUR Softmax:   {naive_throughput:.3f} GFLOPS/s")
    print(f"REFERENCE Softmax:   {optimized_throughput:.3f} GFLOPS/s")
    
    if naive_throughput < optimized_throughput:
        temescal_inner_thoughts = [
            "You're too slow!",
            "Try harder next time buddy!",
            "Waiting for nvidia-smi on ynez, is somehow faster than you."
        ]
        print(random.choice(temescal_inner_thoughts))
    else:
        print("I apologize, I was not familiar with your GFLOPs...")
