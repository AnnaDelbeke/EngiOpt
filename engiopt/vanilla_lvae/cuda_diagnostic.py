"""Standalone CUDA diagnostic for debugging SIGFPE on cluster.

Run with: python -u engiopt/vanilla_lvae/cuda_diagnostic.py
"""

import subprocess
import sys

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def p(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"ERROR: {e}"


p("=" * 60)
p("CUDA DIAGNOSTIC")
p("=" * 60)

# 1. Environment info
p(f"\nPython: {sys.version}")
p(f"PyTorch: {torch.__version__}")
p(f"PyTorch CUDA compiled version: {torch.version.cuda}")
p(f"cuDNN version: {torch.backends.cudnn.version()}")
p(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    p("FATAL: No CUDA device available")
    sys.exit(1)

p(f"GPU: {torch.cuda.get_device_name(0)}")
p(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
p(f"CUDA runtime version: {torch.version.cuda}")

# nvidia-smi driver version
p(f"\nnvidia-smi output:")
p(run_cmd("nvidia-smi --query-gpu=driver_version,cuda_version,name --format=csv,noheader"))

device = torch.device("cuda")

# 2. Basic CUDA operations
p("\n--- Basic CUDA ops ---")

p("  matmul (100x100)...", )
a = torch.randn(100, 100, device=device)
b = torch.randn(100, 100, device=device)
c = a @ b
torch.cuda.synchronize()
p(f"  OK, result norm={c.norm().item():.4f}")

p("  matmul (100x25088)...")
a = torch.randn(2, 100, device=device)
b = torch.randn(100, 25088, device=device)
c = a @ b
torch.cuda.synchronize()
p(f"  OK, result shape={c.shape}")

# 3. Linear layer (no spectral norm)
p("\n--- Linear layer (no spectral norm) ---")
p("  Linear(100, 25088)...")
layer = nn.Linear(100, 25088).to(device)
x = torch.randn(2, 100, device=device)
y = layer(x)
torch.cuda.synchronize()
p(f"  OK, output shape={y.shape}")

# 4. Spectral norm linear - the suspected culprit
p("\n--- Spectral norm linear ---")
p("  spectral_norm(Linear(100, 25088))...")
sn_layer = spectral_norm(nn.Linear(100, 25088)).to(device)
x = torch.randn(2, 100, device=device)
y = sn_layer(x)
torch.cuda.synchronize()
p(f"  OK, output shape={y.shape}")

# 5. Spectral norm + ReLU (matches decoder.proj)
p("\n--- Spectral norm + ReLU (decoder.proj pattern) ---")
proj = nn.Sequential(
    spectral_norm(nn.Linear(100, 512 * 7 * 7)),
    nn.ReLU(inplace=True),
).to(device)
x = torch.randn(2, 100, device=device)
y = proj(x)
torch.cuda.synchronize()
p(f"  OK, output shape={y.shape}")

# 6. Spectral norm ConvTranspose2d
p("\n--- Spectral norm ConvTranspose2d ---")
deconv = spectral_norm(
    nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, bias=False)
).to(device)
x = torch.randn(2, 512, 7, 7, device=device)
y = deconv(x)
torch.cuda.synchronize()
p(f"  OK, output shape={y.shape}")

# 7. BatchNorm2d
p("\n--- BatchNorm2d ---")
bn = nn.BatchNorm2d(256).to(device)
x = torch.randn(2, 256, 13, 13, device=device)
y = bn(x)
torch.cuda.synchronize()
p(f"  OK, output shape={y.shape}")

# 8. Full encoder-decoder pattern
p("\n--- Full encoder/decoder round-trip ---")
from engiopt.vanilla_lvae.components import Encoder2D, TrueSNDecoder2D

enc = Encoder2D(latent_dim=100, design_shape=(101, 101)).to(device)
dec = TrueSNDecoder2D(latent_dim=100, design_shape=(101, 101)).to(device)

x = torch.randn(2, 1, 101, 101, device=device)
p("  Encoder forward...")
z = enc(x)
torch.cuda.synchronize()
p(f"  Encoder OK, z shape={z.shape}")

p("  Decoder forward...")
x_hat = dec(z)
torch.cuda.synchronize()
p(f"  Decoder OK, x_hat shape={x_hat.shape}")

# 9. Backward pass
p("\n--- Backward pass ---")
z2 = enc(x)
x_hat2 = dec(z2)
loss = torch.nn.functional.mse_loss(x, x_hat2)
p(f"  loss={loss.item():.6f}")
p("  backward...")
loss.backward()
torch.cuda.synchronize()
p("  backward OK")

p("\n" + "=" * 60)
p("ALL TESTS PASSED")
p("=" * 60)
