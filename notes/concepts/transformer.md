# Transformer
- Also: Vision Transformer (ViT)
- A [[neural-network]] architecture using the concept of [[self-attention]]
- Originally developed for translation, a concept where both input and output are sequences), also hint to [[self-attention]] (words have different meanings in different contexts)
Involves multiple steps:

Transformer layers:
1. [[patching]]: dividing the image into smaller, non-overlapping patches
2. [[embedding]]: [[flattening]] and projecting  patch into a token (or 1D vector)
3. [[self-attention]]: processing of tokens from previous step

Transformers capture a global receptive field, producing an output that captures strong and weak relationships between patches of an image that are far apart, regardless of their distance.