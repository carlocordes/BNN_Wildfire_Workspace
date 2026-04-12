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
                 num_static_channels : int,
                 num_dynamic_channels : int,
                 num_timestamps_per_sample : int,
                 patch_size : int,
                 embedding_dim: int,
        ):
        
        super().__init__()
        
        #Parameters
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.num_static_channels= num_static_channels
        self.num_dynamic_channels = num_dynamic_channels

        self.num_timestamps_per_sample = num_timestamps_per_sample
        
        # Static embeds
        self.static_embeds = nn.ModuleList([
            nn.Conv2d(in_channels = 1,
                      out_channels = embedding_dim,
                      kernel_size = patch_size,
                      stride = patch_size)
            for _ in range(self.num_static_channels)
        ])    

        # Static modality tags
        self.static_tags = nn.Parameter(torch.randn(num_static_channels, 1, 1, embedding_dim))

        # Dynamic embeds
        self.dynamic_embeds = nn.ModuleList([
            nn.Conv3d(in_channels = 1,
                      out_channels = self.embedding_dim,
                      # Kernel definition
                      kernel_size = (self.num_timestamps_per_sample, self.patch_size, self.patch_size),
                      stride = (self.num_timestamps_per_sample, self.patch_size, self.patch_size))
            for _ in range(self.num_dynamic_channels)
        ])

        # Dynamic modality tags
        self.dynamic_tags = nn.Parameter(torch.randn(self.num_dynamic_channels, 1, 1, self.embedding_dim))

        # Self-Attention blocks
        self.static_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model = self.embedding_dim, nhead = 4, batch_first = True)
            for _ in range(self.num_static_channels)
        ])

        self.dynamic_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model = self.embedding_dim, nhead = 4, batch_first = True)
            for _ in range(self.num_dynamic_channels)
        ])

        # Mixers for concat operations
        self.static_mixer = nn.Linear(self.num_static_channels * self.embedding_dim, self.embedding_dim)
        self.dynamic_mixer = nn.Linear(self.num_dynamic_channels * self.embedding_dim, self.embedding_dim)

        # Cross-Attention block
        self.module_fusion = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads = 8,
            batch_first = True
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Step 1: Feature compression and initial refinement
            nn.Conv2d(self.embedding_dim, self.embedding_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.embedding_dim // 2),
            nn.ReLU(inplace=True),
            
            # Step 2: First Upsample (4x)
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(self.embedding_dim // 2, self.embedding_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.embedding_dim // 4),
            nn.ReLU(inplace=True),
            
            # Step 3: Second Upsample (4x) to reach original resolution
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            
            # Step 4: Final prediction head (Collapse to 1 channel for binary Fire/No-Fire)
            nn.Conv2d(self.embedding_dim // 4, 1, kernel_size=1)
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

    def forward(self, x_static, x_dynamic):

        ## Pad input to be divisible by patch size via reflect method
        # Pad static (4D): (B, C, H, W) -> (B, C, H+pad_h, W+pad_w)
        orig_h, orig_w = x_static.shape[-2:]
        pad_h = (self.patch_size - orig_h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - orig_w % self.patch_size) % self.patch_size

        pad_4d = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)

        x_static = F.pad(x_static, pad_4d, mode='reflect')

        # Pad dynamic (5D): (B, C, T, H, W) -> (B, C, T, H+pad_h, W+pad_w)
        pad_5d = (0, pad_w, 0, pad_h, 0, 0)
        x_dynamic = F.pad(x_dynamic, pad_5d, mode='reflect')



        ## Pre-produce 2d embeddings
        padded_h, padded_w = x_static.shape[-2:]
        grid_h = padded_h // self.patch_size
        grid_w = padded_w // self.patch_size

        spatial_pos_embed = self.get_2d_pos_embed(
            grid_h = grid_h,
            grid_w = grid_w,
            device = x_static.device
        )



        ## Embedding
        # Patch embedding for static
        embedded_static_tokens = []
        for i, embed_layer in enumerate(self.static_embeds):
            single_channel = x_static[:, i:i+1, :, :] # Slice

            tokens = embed_layer(single_channel) # Embed
            tokens = tokens.flatten(2).transpose(1,2) # Rearrange

            tokens = tokens + spatial_pos_embed # Add 2D spatial embedding
            tokens = tokens + self.static_tags[i] # Add static modularity token

            embedded_static_tokens.append(tokens)
            
        # Tublet embedding for dynamic
        embedded_dynamic_tokens = []
        for i, embed_layer in enumerate(self.dynamic_embeds):
            single_channel = x_dynamic[:, i:i+1, :, :, :] # Slice

            tokens = embed_layer(single_channel)
            tokens = tokens.squeeze(2) # Rearrange
            tokens = tokens.flatten(2).transpose(1,2)

            tokens = tokens + spatial_pos_embed # Add 2D spatial embedding
            
            tokens = tokens + self.dynamic_tags[i] # Add dynamic modularity token

            embedded_dynamic_tokens.append(tokens)



        ## Encoding
        encoded_static = []
        for i, tokens in enumerate(embedded_static_tokens):
            encoded_static.append(self.static_encoders[i](tokens))

        encoded_dynamic = []
        for i, tokens in enumerate(embedded_dynamic_tokens):
            encoded_dynamic.append(self.dynamic_encoders[i](tokens))



        ## Modality mixing (stacking modalities)
        # Stack
        static_concat = torch.cat(encoded_static, dim = -1)
        dynamic_concat = torch.cat(encoded_dynamic, dim = -1)
        
        # Retain embedding dimension
        static_mixed = self.static_mixer(static_concat)
        dynamic_mixed = self.dynamic_mixer(dynamic_concat)



        ## Final fusion
        fused_output, _ = self.module_fusion(
            query = dynamic_mixed,
            key = static_mixed,
            value = static_mixed
        )

        fused_tokens = dynamic_mixed + fused_output # Add residual

        

        ## Decoder
        batch_size, _, __ = fused_tokens.shape
        x_2d = fused_tokens.transpose(1,2).contiguous().view(
                batch_size, self.embedding_dim, grid_h, grid_w
        )
        pred_map = self.decoder(x_2d)

        pred_map = pred_map[:, 0:1, :orig_h, :orig_w] # "Un-pad" to original dimensions
        return(pred_map)