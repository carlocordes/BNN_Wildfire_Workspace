#import "../template.typ": *

= Methodology <chap:methodology>

== Workflow & System Overview
#figure(
```yaml
data:
  spatial_extent:
    longmin: -9.4900
    longmax: -6.0234
    latmin: 36.812
    latmax: 42.2724
  crs: "EPSG:3763"
  scale: 1000 # meters
  temporal_extent:
    start_date: "2020-01-01"
    end_date: "2024-12-31"
    day_interval: 1
    sequence_period : 4 # Time between respective sequences
    sample_extent : 10 # Total time of sample
    sample_period : 1 # Time between images in sample
    target_extent : 4
    lead_time : 0 # Time between last sample and target
```
)

== Data 


== Data Acquisition & Processing
=== Earth Engine & Harmonization
=== Feature Engineering
=== Ground Truth

== Dataset Construction
=== Timeframe Configuration
=== Torch Datasets & Tensors


== Model Architecture
=== Encoding
The initial objective of the transfomrer model revolves around turning input data into tokens via patch embedding. The patch size, a model hyperparameter, defines the pixel extent of square patches every input image is divided into. As we handle only single-band data we treat the image as a grid of height and width: $h times w$. 

// Conv2d for patch embedding
This is realized via a 2D convolution with stride equal to the desired patch size. This ensures zero overlap between neighboring patches. Essentially the patch is first flattened from a matrix to a vector:
$ P in RR ^ (16 times 16) -> P_(f l a t t e n e d) in RR ^256 $
To produce the embedding, the convolution operation then maps every flattened patch into the embedding dimension:

$ e = w P_(f l a t t e n e d) +  b $
where
$ w  = mat(
          w_(1, 1), w_(1, 2), ..., w(d_(p a t c h), 1);
          w_(2, 1), w_(2, 1), ..., w(d_(p a t c h), 2);
          dots.v, dots.v, dots.down, dots.v;
          w_(d_(e m b e d), 1), w_(d_(e m b e d), 2), ..., w_(d_(p a t c h), d_(e m b e d))
  )  in RR ^(128 x 256) $

// Reflective padding (h_pad x w_pad)


// Tensor
As the model is fed multiple modules $M$ for consecutive time steps $t$ that are processed in batches $B$ of variable sizes the input tensor has dimension:


//Dynamic to feature amount -> config defines no. of static/dynamic channels
=== Temporal Embedding
=== Spatial Embedding
=== Attention Layers
=== Fusion Layers
=== Decoder Layer

== Training Routine
=== Model Parameters
=== Loss Function
=== Back Propagation & Optimizer