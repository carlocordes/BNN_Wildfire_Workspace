# Transformer
- Also: Vision Transformer (ViT)
- A [[deep-learning]] architecture
- Solves previous bottleneck problem by [[LSTM]] and [[RNN]] that data needed to be processed sequentially

- Translating transformers from processing text to images, the concept of a token is translated from a word to a patch of pixels (3D) called tubelets (e.g. 16x16x3)
- Divides 3D space into smaller cube chunks

Transformer layers:
1. [[patching]]: dividing the image into smaller, non-overlapping patches
2. [[embedding]]: [[flattening]] and projecting  patch into a token (or 1D vector)
3. [[attention]]: processing of tokens from previous step, updating 

Transformers capture a global receptive field, producing an output that captures strong and weak relationships between patches of an image that are far apart, regardless of their distance.