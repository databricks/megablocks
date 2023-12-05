# write transformer engine example 
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)
a = torch.randn(out_features, hidden_size, device="cuda").cuda(torch.cuda.current_device()).to(torch.bfloat16)
b = a.detach().clone()
model.weight_tensor = a
inp = torch.randn(hidden_size, in_features, device="cuda")

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
with torch.autocast(device_type='cuda', dtype=torch.float16):
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model(inp)
loss = out.sum()
loss.backward()
print(loss)