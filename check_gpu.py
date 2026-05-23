import torch

print(f"PyTorch version   : {torch.__version__}")
print(f"CUDA available    : {torch.cuda.is_available()}")
print(f"CUDA version      : {torch.version.cuda}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"\nGPU {i}: {p.name}")
        print(f"  VRAM total : {total / 1024**3:.1f} GB")
        print(f"  VRAM free  : {free / 1024**3:.1f} GB")
        print(f"  Compute    : {p.major}.{p.minor}")
else:
    print("\n[!] No CUDA GPU detected by PyTorch.")
    print("    Most likely cause: PyTorch was installed WITHOUT CUDA support.")
    print("    Fix: reinstall PyTorch with CUDA (see below)")
    print()
    print("    For CUDA 12.1 (most common, RTX 30xx/40xx):")
    print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("    For CUDA 11.8:")
    print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("    Check your CUDA version with: nvidia-smi")
