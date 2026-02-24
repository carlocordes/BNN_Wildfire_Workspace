import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class STViT(nn.Module):
    """
    Spatio-temporal Vision Transformer
        Args:
        batch_size: Number of like-sized samples to be passed through at one forward feed
        num_modules: 
        patch_size:
        embedding_dim : 
    """
    def __init__(self,
                 num_modules : int,
                 patch_size : int,
                 embedding_dim: int,
        ):
        
        super().__init__()
        
        #Parameters
        self.num_modules = num_modules
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
        # Network Classes
        self.patch_embedding = nn.Conv2d(1, embedding_dim, kernel_size=patch_size, stride=patch_size)
        
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model = self.embedding_dim, # Embedding dimension of input
                    nhead = 4,                    # Number of attention heads
                    dim_feedforward = 128,
                    dropout = 0.01,
                    batch_first = True,
                ),
                num_layers = 4
            )
            for _ in range(num_modules)
        ])

        self.num_cross_layers = 4
        self.cross_encoder = nn.ModuleList([
            nn.MultiheadAttention(embed_dim = self.embedding_dim,
                                  num_heads = 4,
                                  batch_first = True
            )
            for _ in range(self.num_cross_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Conv2d(self.embedding_dim, self.embedding_dim // 2,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=self.patch_size,
                        mode="bilinear",
                        align_corners=False),
            nn.Conv2d(self.embedding_dim // 2, 1, kernel_size=1)
        )

    def get_2d_pos_embed(self, grid_h, grid_w, device):
        grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
        grid_y, grid_x = grid_y.to(device).float(), grid_x.to(device).float()
        dims = self.embedding_dim // 4
        omega = torch.arange(dims).to(device).float() / dims
        omega = 1. / (10000**omega)
        out_y = torch.einsum('hw,d->hwd', grid_y, omega)
        out_x = torch.einsum('hw,d->hwd', grid_x, omega)
        pos_embed = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
        return pos_embed.flatten(0, 1).unsqueeze(0)

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        pad_h = (self.patch_size - orig_h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - orig_w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        batch_size, _, h_padded, w_padded = x.shape
        h_grid, w_grid = h_padded // self.patch_size, w_padded // self.patch_size

        x = x.view(batch_size * self.num_modules, 1, h_padded, w_padded)
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        x = x + self.get_2d_pos_embed(h_grid, w_grid, x.device)

        # Self-Attention
        x = x.view(batch_size, self.num_modules, -1, self.embedding_dim)
        encoded_x = []
        for i, encoder in enumerate(self.encoders):
            encoded_x.append(encoder(x[:, i]))
        x = torch.stack(encoded_x, dim = 1)

        # Cross-Attention
        mod0_tokens = x[:, 0, :, :]
        mod1_tokens = x[:, 1, :, :]

        for i, crossattention in enumerate(self.cross_encoder):
            attn_out, _ = crossattention(query = mod0_tokens, # 0 queries 1
                                            key = mod1_tokens,
                                            value = mod1_tokens)
            
            mod0_tokens = mod0_tokens + attn_out

        x = mod0_tokens # Isolating module 0 for prediction head

        x = x.transpose(1, 2).view(batch_size, self.embedding_dim, h_grid, w_grid)

        # Decode
        x = self.decoder(x)
        
        return x[:, 0, :orig_h, :orig_w]