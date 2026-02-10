import torch
import torch.nn as nn

class STViT(nn.Module):
    """
    Spatio-temporal Vision Transformer
        Args:
        batch_size:
        num_modules:
        patch_size:
        embedding_dim : 
    """
    def __init__(self,
                 batch_size : int,
                 num_modules : int,
                 patch_size : int,
                 img_height : int,
                 img_width : int,
                 embedding_dim: int,
        ):
        
        super().__init__()
        
        #Parameters
        self.batch_size = batch_size
        self.num_modules = num_modules
        self.patch_size = patch_size
        self.img_height = img_height
        self.img_width = img_width
        self.embedding_dim = embedding_dim

        self.num_patches_per_image = int((img_height / patch_size) * (img_width / patch_size))
        
        # Network Classes
        self.patch_embedding = nn.Linear(patch_size*patch_size, embedding_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(self.batch_size, 1, self.num_patches_per_image, self.embedding_dim)
        )

        
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


    def forward(self, x):
        x = x.unfold(dimension = 2, size = self.patch_size, step = self.patch_size) \
             .unfold(dimension = 3, size = self.patch_size, step = self.patch_size)

        x = x.contiguous().view(
            self.batch_size,
            self.num_modules,
            self.num_patches_per_image,
            self.patch_size * self.patch_size
        )

        x = self.patch_embedding(x)

        x = x + self.pos_encoding

        # Self-Attention
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


        x = torch.stack([mod0_tokens, mod1_tokens], dim = 1)

        x = x[:, 0, :, :] # Isolating module 0 for prediction head

        x = x.view(self.batch_size,
                   self.img_height // self.patch_size, self.img_width // self.patch_size,
                   self.embedding_dim).permute(0, 3, 1, 2)

        # Decode
        x = self.decoder(x).squeeze(1)

        return x