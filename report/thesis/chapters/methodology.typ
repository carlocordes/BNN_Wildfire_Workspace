#import "../template.typ": *
#set heading(numbering: "1.")

= Methodology <chap:methodology>

== Workflow & System Overview <sec:method>
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

== Data <sec:data>


== Data Acquisition & Processing
=== Earth Engine & Harmonization
=== Feature Engineering
=== Ground Truth

== Dataset Construction
=== Timeframe Configuration
=== Torch Datasets & Tensors


== Model Architecture <sec:model>
=== Encoding <subsec:encoding>
The initial objective of the transfomrer model revolves around turning input data into tokens via patch embedding. The patch size, a model hyperparameter, defines the pixel extent of square patches every input image is divided into. As we handle only single-band data we treat the image as a grid of height and width: $h times w$. 

// Reflective padding (h_pad x w_pad)
This is realized via a 2D convolution with stride equal to the desired patch size. This ensures zero overlap between neighboring patches. An important element to consider is that both height and width of images need not be divisible without rest by the patch size. To counter this we use reflective padding, a process which mirrors input images around their edges to fill in pixel data until the next complete patch. Extending height and width to to $h_(p a d)$ & $w_(p a d)$ allows for perfect image division into patches. As the model input consists of multiple modules $M$ for consecutive time steps $t$ that are processed in batches $B$ of variable sizes the input tensor has dimensionality:
$ macron(T) in  RR ^(B times M times t times h_(p a d) times w_(p a d)) $ 

#todo(stroke: orange)[Figure of reflective patch padding]

// Conv2d for patch embedding
Every image of dimension $h_(p a d) times w_(p a d)$ is then subject to the flattening operation, here illustrated with a patch size of 16 pixels. 

$ P in RR ^ (16 times 16) -> P_(f l a t) in RR ^256 $

Up to this point, pixel values have not changed, simply the indexing is altered. To produce the embedding, the convolution operation then maps every flattened patch into the embedding dimension. Here we showcase the operation for a patch size of 16 pixels, mapping to the embedding of dimension 128:

#todo(stroke: orange)[Figure of conv  with stride equal to patch]


$ e = w P_(f l a t) +  b $ <eq:2dkernel>

where

$ w  = mat(
          w_(1, 1), w_(1, 2), ..., w_(d_(p a t c h, 1));
          w_(2, 1), w_(2, 1), ..., w_(d_(p a t c h, 2));
          dots.v, dots.v, dots.down, dots.v;
          w_(d_(e m b e d, 1)), w_(d_(e m b e d, 2)), ..., w_(d_(p a t c h), d_(e m b e d))
  )  in RR ^ (128 times 256) $

//Dynamic to feature amount -> config defines no. of static/dynamic channels
The convolution weights are learnable and are only shared between samples of the same module. Every module is assigned an independent convolution operation and individual weights.

=== Temporal embedding
The model architecture demands a separate routing of static and dynamic data channels. All dynamic modules with a more than one time steps receive an implicit temporal encoding that is an extension to the one described in @subsec:encoding. The objective of this is to give patches that describe the same physical area in different time steps a closer connection to one another. This is done by treating the different time steps for one module together. The single-image 2D convolution is replaced by a 3D convolution that treats the series of images of timesteps $t$ as a 3-dimensional set of values and projects it into the embedding dimension $d_(e m b e d)$:

$ RR ^ (t times d_(p a t c h) times d_(p a t c h)) ->  RR ^ (d_(e m b e d)) $ <eq:3d_conv>

The convolutional operation is analogous to @eq:2dkernel for a static embedding with a kernal of dimension:

$ w in RR ^ (d_(e m b e d) times t times P ^ 2) $

In essence, the image series is partitioned into tubelets instead of patches. These tubelets include the information of one patch for all time steps together in one. This information is then fused into one embedding through the convolution operation. As this is carried out with independent convolutionary weights for all modules we call this step the implicit temporal embedding. Tokens describing temporally dynamic patches are projected into the same embedding dimension as static ones.

#todo(stroke : orange)[Figure of tubelet embedding]

=== Spatial Embedding
Having obtained a set of embeddings for every image in the model input, we now induce spatial relationships between patches. The spatial embeddings is applied to both temporally static and dynamic embeddings. The idea is to assert a learnable idea of proximity from one patch to another. As wildfire is a localized phenomena the spatial embedding the objective here is to give neighboring patches a strong correlation. Distant patches, while sharing lower correlation, can still influence one another, with limited effect. The important difference to a convolution here is that we do not simply rule out distant relationships, but keep them contained. This step sets up the relationships of patch embeddings as it will later be used in the attention processes, in which patches with more similar spatial embeddings will more likely attend to one another. 

In order to create such a relationship we come up with an individual positional encoding for every patch that is added onto each embedding:

$ e_"token" ' = e_"token" + e_"pos"  in RR ^ (d_"embed") $

To create a positional embedding that retains similarity between proximal patches, we use sinusoidal positional embedding. By imposing sine and cosine waves of varying frequencies over the patch grid, we are able to establish a unique encoding for every patch that has distinct relationships to other patches, depending on their euclidean distance. This is done by first defining a set of frequencies with  $d = d_"embed" / 4$:

$ omega_i = 1 / 10000 ^ (i / d) $ 

The nature of the integer $4$ is that in order to create the most unique positional embeddings, we split the added partial embeddings into both height and width and also sine and cosine categories. We scale the frequencies with 10000 in order to fit multiple periods into the image. Interpreting height and width as coordinates we can thus describe the positional encoding of a patch as the vertical concatentation of these frequencies:

$ e_"pos" = vec(sin(h omega), cos(h omega), sin(w omega), cos(w omega)) in RR ^ (d_"embed") $

@fig:embedding_dims shows samples of the modes of the positional embeddings for every patch in the input image, in this case $80 times 40$. These include waves that propagate in direction of height and width. For every patch we add the embedding values of all dimensions. 

#figure(
  image("../figs/encoding_dims_visual.png", width:120%),
  placement: auto, 
  caption: [Additive encoded values per encoding dimension],
) <fig:embedding_dims>

Given the multitude of modes we can form very individualised encodings for all patches. @fig:embedding_simil shows the cosine similarity of patches to one another for different embedding dimensions. Here the central patch serves as an anchor-patch, with the values of all patches describing the its similarity to it. This shows that sinusoidal positional embedding is able to establish similarities and differences between patches, depending on their respective proximity. Furthermore, it is important to mention that the utility of this process heavily depends on the embedding dimension chosen. A higher embedding dimension allows to pass more sinusoidal modes and hence to create a more distinct characterization of each patch. We here show the cosine similarity for three embedding dimensions, which are more descriptive for larger values.  

#figure(
  image("../figs/encoding_simil_visual.png", width:120%),
  placement: auto, 
  caption: [Cosine similarity of patches relative to central anchor-patch at varying embedding dimension],
) <fig:embedding_simil>

#todo(stroke: red)[Static tags]

Having embedded all tokens of different modalities into embedding space, we additionally give each token a modality tag. A tag is unique per module and is a learnable parameter with the objective to give the attention heads a clearer idea of where tokens came from. Throughout the learning process the transformer will be able to differentiate between different modules as the tag injects a numeric meaning to all tokens. The complete spatial embedding process, with the additional static tag $arrow(s)_m$ for each module $m$ thus takes form:
$ arrow(z)^(m)_n = arrow(e)^("  m")_"token, n" + arrow(e)_"pos" + arrow(s)_m $ 

=== Attention Layers



=== Fusion Layers
=== Decoder Layer

== Training Routine
=== Model Parameters
=== Loss Function
=== Back Propagation & Optimizer