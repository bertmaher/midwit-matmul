import torch

# Set tensor print options so we can see the 64x256 result in all its glory
torch.set_printoptions(sci_mode=False, precision=3, profile="full", linewidth=12*256)

# Create a synthetic tensor with 128 threads each holding 128 values
r = torch.arange(128)[:, None] + torch.arange(128) * 0.001
print(r)

# Permute the tensor so that it arranges the results properly in memory
r = r.view(4, 8, 4, 32, 2, 2).permute(0, 4, 1, 3, 2, 5).reshape(64, 256)
print(r)
