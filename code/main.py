#Internal
from time import time

#External
import torch
import torch.nn as nn

EMBED_DIM = 32

batch_size = 1
num_modules = 2
height = 4
width = 4

print(f'Processing {2} input modules at dimensions {height}x{width} ')

patch_size = 2 # Rectangular, number of pixels


num_patches_per_image = int((height / patch_size) * (width / patch_size))
print(f'Patching images into {num_patches_per_image} patches with patch size: {patch_size}')

input_data = torch.randn(batch_size, num_modules, height, width)

target = torch.ones(batch_size, 1, height, width)

#print(input_data)
#print(input_data.shape)


# Patch transform
patches = input_data.unfold(dimension = 2, size = patch_size, step = patch_size) \
                    .unfold(dimension = 3, size = patch_size, step = patch_size)

patches = patches.contiguous().view(
    batch_size,
    num_modules,
    num_patches_per_image,
    patch_size * patch_size
)

#print(patches)
#print(patches.shape)

# Patch embedding (Pixels --> Tokens)

patch_embed = nn.Linear(patch_size*patch_size, EMBED_DIM)
tokens = patch_embed(patches)

#print(tokens)
#print(tokens.shape)

# Positional Encoding
pos_embed = nn.Parameter(
    torch.randn(batch_size, 1, num_patches_per_image, EMBED_DIM)
)

tokens = tokens + pos_embed # Add embedding, addition is automatically broadcasted correctly

#print(tokens)
#print(tokens.shape)

## Define Self-Attention block
# Singler transformer layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model = EMBED_DIM, # Embedding dimension of input
    nhead = 4,           # Number of attention heads
    dim_feedforward = 128,
    dropout = 0.01,
    batch_first = True,
)

# Blocks of transformer layers
encoder = nn.TransformerEncoder(
    encoder_layer = encoder_layer,
    num_layers = 4
)

# Perform per-module, per-batch self-attention
encoded_modules = []
for module in range(num_modules):
    mod_tokens = tokens[:, module, :, :] # (Batch, Patch, Embedding) - Slice along module
    encoded_module = encoder(mod_tokens)
    encoded_modules.append(encoded_module)

encoded_tokens = torch.stack(encoded_modules, dim=1)



## Define Cross-Attention block
# Single Cross-Attention layer
cross_attention = nn.MultiheadAttention(embed_dim = EMBED_DIM,
                                        num_heads = 4,
                                        batch_first = True)

#Block of Cross-Attention layers
num_cross_layers = 3
cross_encoder = nn.ModuleList([cross_attention for _ in range(num_cross_layers)])

mod0_tokens = encoded_tokens[:, 0, :, :]
mod1_tokens = encoded_tokens[:, 1, :, :]

mod0_tokens, _ = cross_attention(query = mod0_tokens, # 0 queries 1
                                 key = mod1_tokens,
                                 value = mod1_tokens)

tokens = torch.stack([mod0_tokens, mod1_tokens], dim = 1)

fused = tokens[:, 0, :, :] # Isolated for prediction

feat = fused.view(batch_size, height // patch_size, width // patch_size, EMBED_DIM).permute(0, 3, 1, 2)

decoder = nn.Sequential(
    nn.Conv2d(EMBED_DIM, EMBED_DIM//2, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False),
    nn.Conv2d(EMBED_DIM//2, 1, kernel_size=1)
)

head = decoder(feat).squeeze(1)
print(head)