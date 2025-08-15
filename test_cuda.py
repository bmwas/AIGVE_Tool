#!/usr/bin/env python
"""Simple CUDA diagnostic script"""
import sys
import os

print("=== CUDA Diagnostic ===")
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

# Environment variables
print("\n--- Environment ---")
for var in ['CUDA_HOME', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH']:
    print(f"{var}: {os.environ.get(var, 'NOT SET')}")

# Try importing torch
try:
    import torch
    print(f"\n--- PyTorch Info ---")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Try creating a tensor on GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"Successfully created tensor on GPU: {x.device}")
        except Exception as e:
            print(f"Failed to create tensor on GPU: {e}")
    else:
        print("\nCUDA not available - checking why...")
        print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
        print(f"torch.backends.cuda.is_built(): {torch.backends.cuda.is_built()}")
        
        # Check if this is a CPU-only build
        if not torch.backends.cuda.is_built():
            print("ERROR: This is a CPU-only PyTorch build!")
        
except ImportError as e:
    print(f"Failed to import torch: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
