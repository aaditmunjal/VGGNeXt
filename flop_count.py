import torch
from vggnext import VGGNeXt # Example model

from thop import profile  # Import the profile function from THOP

# Load model
model = VGGNeXt(num_classes=200)

# Dummy input tensor matching the model's expected input shape
dummy_input = torch.randn(1, 3, 64, 64)

# Profile the model
macs, params = profile(model, inputs=(dummy_input,))

print(f"Model: vgg16_bn")
print(f"FLOPs: {macs / 1e9:.2f} GFLOPs")
print(f"Params: {params / 1e6:.2f} M Params")