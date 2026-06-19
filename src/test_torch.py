import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("Testing torch import with CUDA disabled...")
import torch
print(f"Torch version: {torch.__version__}")
print("Torch import successful!")
