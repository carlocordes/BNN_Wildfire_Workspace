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

## Define Attention blocks

# Singler transformer layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model = EMBED_DIM, # Embedding dimension of input
    nhead = 2,           # Number of attention heads
    dim_feedforward = 128,
    dropout = 0.01,
    batch_first = True,
)

# Blocks of transformer layers
encoder = nn.TransformerEncoder(
    encoder_layer = encoder_layer,
    num_layers = 2
)

tokens = tokens[0] # Temporary, including first of batch only

encoded_tokens = encoder(tokens)
print(encoded_tokens.shape)