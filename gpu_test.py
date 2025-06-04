#!/usr/bin/env python
import torch
import time

print("=== GPU Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test GPU memory allocation
    print("\n=== Memory Test ===")
    print("Creating large tensor on GPU...")
    
    # Create a large tensor that should show up in nvidia-smi
    x = torch.randn(5000, 5000, device='cuda')
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    print("Tensor created. Check nvidia-smi now!")
    print("Sleeping for 10 seconds...")
    time.sleep(10)
    
    # Test computation
    print("\n=== Computation Test ===")
    print("Running matrix multiplication on GPU...")
    start_time = time.time()
    y = torch.mm(x, x)
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start_time
    print(f"GPU computation time: {gpu_time:.3f} seconds")
    
    # Compare with CPU
    print("Running same computation on CPU...")
    x_cpu = x.cpu()
    start_time = time.time()
    y_cpu = torch.mm(x_cpu, x_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU computation time: {cpu_time:.3f} seconds")
    print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")
    
    print(f"\nFinal GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
else:
    print("CUDA not available!") 