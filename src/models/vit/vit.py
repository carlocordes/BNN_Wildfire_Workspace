import torch
import torch.nn as nn
import torch.nn.functional as F

class STViT(nn.Module):
    """
    Spatio-temporal Channel-Fused Vision Transformer (ST-FlexViT)
    
    Fixes implemented:
      1. Early cross-channel fusion (Unified Conv layers instead of individual loops).
      2. Temporal tracking preservation (Conv3D kernel size temporal extent = 1).
      3. Wider, shallower transformer backbone (replaces 64 narrow layers with 8 robust layers).
      4. Progressive, symmetrical decoder.
    """
    def __init__(self,
                 num_static_channels : int,
                 num_dynamic_channels : int,
                 num_timestamps_per_sample : int,
                 patch_size : int,
                 embedding_dim: int,
                 encoder_depth: int = 8,  # CHANGED: Controlled shallow depth
                 num_heads: int = 8       # ORIGINAL: Matches your attention setups
        ):
        super().__init__()
        
        # [ORIGINAL] Structural tracking parameters
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_static_channels = num_static_channels
        self.num_dynamic_channels = num_dynamic_channels
        self.num_timestamps_per_sample = num_timestamps_per_sample
        
        # [ORIGINAL CONTENT / CHANGED EXECUTION] 
        # You combined static and single_dynamic channels together in forward(). 
        # We calculate the total combined input channels here.
        self.total_static_inputs = num_static_channels
        
        # ==========================================
        # PHASE 1 & 2: CHANGED - UNIFIED TOKENIZATION LAYER
        # ==========================================
        
        # CHANGED: Instead of a ModuleList of separate Conv2Ds for each channel,
        # we pass all combined static channels into ONE Conv2D. This forces 
        # cross-channel features to mix immediately at the patch level.
        self.static_embed = nn.Conv2d(
            in_channels=self.total_static_inputs,
            out_channels=self.embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )    

        # CHANGED: Modified Conv3D kernel to (1, patch, patch) instead of (10, patch, patch).
        # This treats each day in your 10-day timeline as its own distinct sequence step, 
        # preventing immediate temporal collapse.
        self.dynamic_embed = nn.Conv3d(
            in_channels=num_dynamic_channels,
            out_channels=self.embedding_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        # CHANGED: Learnable Temporal Embedding vector to track chronological order
        self.temporal_embed = nn.Parameter(
            torch.randn(1, num_timestamps_per_sample, 1, self.embedding_dim)
        )

        # CHANGED: Learnable modality identity tokens to separate static vs dynamic inputs
        self.static_type_tag = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.dynamic_type_tag = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        # ==========================================
        # PHASE 3: CHANGED - JOINT SPATIOTEMPORAL BACKBONE
        # ==========================================
        
        # CHANGED: Replaced your 12 isolated transformers and 16 fusion blocks 
        # (64 layers total) with a single, highly expressive joint backbone.
        # This allows cross-variable, spatial, and temporal attention to occur simultaneously.
        self.joint_backbone = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,   # Increase this via config (e.g., 256 or 512)
                nhead=num_heads,
                dim_feedforward=self.embedding_dim * 4,
                dropout=0.45,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ),
            num_layers=encoder_depth
        )
        
        # CHANGED: Sequence aggregation layer to project joint tokens back to a single 2D grid spatial layout
        self.sequence_projector = nn.Linear(self.embedding_dim, self.embedding_dim)

        # ==========================================
        # PHASE 4: CHANGED - PROGRESSIVE SYMMETRICAL DECODER
        # ==========================================
        
        # CHANGED: Replaced the brute-force abrupt bilinear scaling blocks.
        # This progressively restores resolution step-by-step using convolutions 
        # to ensure local pixel representations remain cohesive.
        self.decoder = nn.Sequential(
            # Grid resolution up to Intermediate resolution (e.g., 16x16 patches upscaled by 4x -> 4x4 resolution maps)
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(self.embedding_dim, self.embedding_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.embedding_dim // 2),
            nn.ReLU(inplace=True),
            
            # Intermediate upsample (4x) to reach original target resolution
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(self.embedding_dim // 2, self.embedding_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.embedding_dim // 4),
            nn.ReLU(inplace=True),
            
            # Final output mapping projection (1 channel output for binary logit cross-entropy prediction)
            nn.Conv2d(self.embedding_dim // 4, 1, kernel_size=1)
        )

    # [ORIGINAL] Kept your exact 2D Sin-Cos Meshgrid calculation mechanism intact
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

    def forward(self, x_static, x_dynamic, x_single_dynamic, ablate_static_idxs = None, ablate_dynamic_idxs = None,
                return_tokens = False):
        B = x_static.shape[0]

        # [ORIGINAL] Concat single-layer dynamic into the static block sequence
        x_static = torch.cat([x_static, x_single_dynamic], dim=1)

        if ablate_static_idxs is not None:
            print(f'[ABLATION]: Muting static channels', *ablate_static_idxs)

            x_static = x_static.clone()
            for idx in ablate_static_idxs:
                x_static[:, idx, :, :] = 0.0

        if ablate_dynamic_idxs is not None:
            print(f'[ABLATION]: Muting dynamic channels', *ablate_dynamic_idxs)

            x_dynamic = x_dynamic.clone()
            for idx in ablate_dynamic_idxs:
                x_dynamic[:, idx, :, :] = 0.0

        # [ORIGINAL] Reflection padding to ensure patch size divisibility
        orig_h, orig_w = x_static.shape[-2:]
        pad_h = (self.patch_size - orig_h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - orig_w % self.patch_size) % self.patch_size
        
        x_static = F.pad(x_static, (0, pad_w, 0, pad_h), mode='reflect')
        x_dynamic = F.pad(x_dynamic, (0, pad_w, 0, pad_h, 0, 0), mode='reflect')

        # [ORIGINAL] Grid dimension calculations
        padded_h, padded_w = x_static.shape[-2:]
        grid_h = padded_h // self.patch_size
        grid_w = padded_w // self.patch_size
        num_patches_per_frame = grid_h * grid_w

        # [ORIGINAL] Positional tracking map initialization
        spatial_pos_embed = self.get_2d_pos_embed(grid_h, grid_w, x_static.device)

        # ==========================================
        # CHANGED: SIMPLIFIED & FUSED FORWARD PROJECTIONS
        # ==========================================
        
        # 1. Process Unified Static Tokens
        # Shape: (B, Static_Ch, H, W) -> Conv2d -> (B, Embed_Dim, Grid_H, Grid_W)
        static_tokens = self.static_embed(x_static)
        static_tokens = static_tokens.flatten(2).transpose(1, 2) # (B, Num_Patches, Embed_Dim)
        
        # Apply structural identifiers
        static_tokens = static_tokens + spatial_pos_embed
        static_tokens = static_tokens + self.static_type_tag

        # 2. Process Unified Dynamic Tokens (Preserving Timeline Axis)
        # Shape: (B, Dynamic_Ch, T, H, W) -> Conv3D -> (B, Embed_Dim, T, Grid_H, Grid_W)
        dynamic_tokens = self.dynamic_embed(x_dynamic)
        
        # Reshape to easily inject distinct temporal spatial identifiers
        # Shape: (B, Embed_Dim, T, Grid_H * Grid_W) -> Permute -> (B, T, Grid_H * Grid_W, Embed_Dim)
        dynamic_tokens = dynamic_tokens.flatten(3).permute(0, 2, 3, 1)
        
        # Inject matching 2D coordinate embeddings across all individual days
        dynamic_tokens = dynamic_tokens + spatial_pos_embed.unsqueeze(1)
        # Inject the learnable chronological 1D temporal embedding
        dynamic_tokens = dynamic_tokens + self.temporal_embed
        # Add dynamic feature source identifier tag
        dynamic_tokens = dynamic_tokens + self.dynamic_type_tag
        
        # Flatten time and space tracking dimensions into a single continuous token array sequence
        # Shape: (B, T * Num_Patches, Embed_Dim)
        dynamic_tokens = dynamic_tokens.flatten(1, 2)

        # 3. Join Token Chains Sequentially 
        # Sequence Length = (Num_Patches) + (10 * Num_Patches)
        joint_token_sequence = torch.cat([static_tokens, dynamic_tokens], dim=1)

        # 4. Joint Attention Sequence Backbone Execution
        enrolled_tokens = self.joint_backbone(joint_token_sequence)

        # ==========================================
        # CHANGED: AGGREGATE SEQUENCES BACK TO 2D RECONSTRUCTION GRID
        # ==========================================
        
        # Separate the computed unified stream tokens back into their original spatial blocks
        processed_static = enrolled_tokens[:, :num_patches_per_frame, :]
        processed_dynamic = enrolled_tokens[:, num_patches_per_frame:, :]
        
        # Reshape dynamic tokens back to structure: (B, T, Num_Patches, Embed_Dim)
        processed_dynamic = processed_dynamic.view(B, self.num_timestamps_per_sample, num_patches_per_frame, self.embedding_dim)
        # Aggregate temporal vectors across the 10-day timeline via a mean reduction step
        temporal_aggregated_dynamic = processed_dynamic.mean(dim=1) 
        
        # Fuse the spatial representations of static features and temporal histories together
        fused_spatial_tokens = processed_static + temporal_aggregated_dynamic
        fused_spatial_tokens = self.sequence_projector(fused_spatial_tokens)

        # Reshape the token sequence chain back into a 2D Feature Map Grid format for standard CNN convolution reading
        x_2d = fused_spatial_tokens.transpose(1, 2).contiguous().view(
            B, self.embedding_dim, grid_h, grid_w
        )

        # 5. Execute Progressive Resolution Upsampling Decoder Path
        pred_map = self.decoder(x_2d)

        # [ORIGINAL] Slice off padding adjustments back to user raw target specifications
        pred_map = pred_map[:, 0:1, :orig_h, :orig_w]
        
        if return_tokens == True:
            return pred_map, x_2d
        else:
            return pred_map