import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048
num_experts = 4
torch.manual_seed(0)
# Initialize model and inputs.
model = [te.Linear(in_features, out_features, bias=True) for _ in range(num_experts)]
inp = torch.randn(hidden_size, in_features, device="cuda").cuda().to(torch.bfloat16)
inp.requires_grad_(True)
# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
res = []
# with torch.autocast(device_type='cuda', dtype=torch.float16):
#     with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
#         with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
for ne in range(num_experts):
    out = model[ne](inp)
    res.append(out)

total = torch.cat(res)
loss = total.sum()
print(loss)
loss.backward()
print(inp.grad.sum())