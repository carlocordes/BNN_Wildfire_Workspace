#import "../template.typ": *
#set heading(numbering: "1.")

= Methodology <chap:methodology>
#todo(stroke : green)[Define key terms (maybe somewhere else)]

#todo(stroke : green)[Concept: apply idea of embeddings to "climate vector space"]

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
- *Module: *


== Data Acquisition & Processing


=== Earth Engine & Harmonization


=== Feature Engineering


=== Ground Truth




== Dataset Construction
=== Timeframe Configuration <subsec:time_config>
As part of the research objectives are to investigate the temporal scopes of both data and predictions, it is important to be clear about terminology. The datasets are created according to configurations. These can be adjusted depending on the desired test case. As we handle temporally dynamic datasets we need to define how timesteps of samples are represented in the dataset. To ensure consistency in naming these, we here declare a series of term definitions relevant for this step:


- *Sample:* Input data the model sees to form predictions; describes the set of timestamps for one forward pass
- *Target:* The collection of images, that is variable length, that is used for testing the model predictions
- *Sequence:* A pair of a sample and target
- *Dataset:* A collection of sequences
- *Sequence Period:* Separation between individual unique sequences
- *Extent:* Used both sample and target, defines their respective variable length
- *Lead Time:* The time between the last seen timestamp in the sample and first one in the target. In other words: the amount of time the model predicts into the future

#figure(
  image("../figs/temporal_configs.png", width:100%),
  placement: auto, 
  caption : [Definions of temporal scope in project datasets]
) <fig:temporal_scope>
#todo(stroke : orange)[Update figure of timestamp configurations]

@fig:temporal_scope illustrates the terms defined above. It shows two consecutive sequences as well as their their respective samples and targets. The configuration file governs the total time of interest. The dataset builder then generates a set of timestamps as governed by the rest of the temporal configurations and iterates through the respective file for every module. Sequences with missing images are skipped entirely to ensure consistency. 


=== Torch Datasets & Tensors
Moving data from images into the pytorch framework is done using its fundamental data structure, the *Tensor*. Like in the traditional mathematical sense this is an array of arbitrary dimension containing elements of a single data type, but has a few computation-driven features. Most importantly, it is optimized to be handled by GPU. Tensors are directly assosciated with the model instance (elaborated in @subsec:backprop) and have an attribute that defines whether they are adjustable model parameters. This concept is specifically designed for those tensors containing model weights. PyTorch can track the operations imposed on a tensor, which makes later backpropagation to these easier #citep(<pytorch_tensor>). The sequences described in @subsec:time_config are appended into such a tensor structures and then saved as portable datasets to be made accessible on larger compute resources.



== Model Architecture <sec:model>
In order to motivate the structure of the here applied transformer, this section walks through the sequential steps incoming data takes through the model. This represents one forward propagation. The later showcased training regiment makes this forward pass once for every batch in the training data, thus cycling through it an abundance of times.

#todo(stroke : orange)[Complete model architecture diagram and reference]



=== Encoding <subsec:encoding>
The initial objective of the transfomer model revolves around turning input data into tokens via patch embedding. The patch size, a model hyperparameter, defines the pixel extent of square patches every input image is divided into. As we handle only single-band data we treat the image as a grid of height and width: $h times w$. 

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

The convolutional operation is analogous to @eq:2dkernel for a static embedding with a kernel of dimension:

$ w in RR ^ (d_(e m b e d) times t times P ^ 2) $

In essence, the image series is partitioned into tubelets instead of patches. These tubelets include the information of one patch for all time steps together in one. This information is then fused into one embedding through the convolution operation. As this is carried out with independent convolutionary weights for all modules we call this step the implicit temporal embedding. Tokens describing temporally dynamic patches are projected into the same embedding dimension as static ones.

#todo(stroke : orange)[Figure of tubelet embedding]

=== Spatial Embedding
Having obtained a set of embeddings for every image in the model input, we now induce spatial relationships between patches. The spatial embeddings is applied to both temporally static and dynamic embeddings. The idea is to assert a learnable idea of proximity from one patch to another. As wildfire is a localized phenomena the spatial embedding the objective here is to give neighboring patches a strong correlation. Distant patches, while sharing lower correlation, can still influence one another, with limited effect. The important difference to a convolution here is that we do not simply rule out distant relationships, but keep them contained. This step sets up the relationships of patch embeddings as it will later be used in the attention processes, in which patches with more similar spatial embeddings will more likely attend to one another. 

In order to create such a relationship we come up with an individual positional encoding for every patch that is added onto each embedding:

$ arrow(e)_"token" ' = arrow(e)_"token" + arrow(e)_"pos"  in RR ^ (d_"embed") $

To create a positional embedding that retains similarity between proximal patches, we use sinusoidal positional embedding. By imposing sine and cosine waves of varying frequencies over the patch grid, we are able to establish a unique encoding for every patch that has distinct relationships to other patches, depending on their euclidean distance. This is done by first defining a set of frequencies with  $d = d_"embed" / 4$:

$ omega_i = 1 / 10000 ^ (i / d) $ 

The nature of the integer $4$ is that in order to create the most unique positional embeddings, we split the added partial embeddings into both height and width and also sine and cosine categories. We scale the frequencies with 10000 in order to fit multiple periods into the image. Interpreting height and width as coordinates we can thus describe the positional encoding of a patch as the vertical concatentation of these frequencies:

$ arrow(e)_"pos" = vec(sin(h omega), cos(h omega), sin(w omega), cos(w omega)) in RR ^ (d_"embed") $

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

Having embedded all tokens of different modalities into embedding space, we additionally give each token a modality tag. A tag is unique per module and is a learnable parameter with the objective to give the attention heads a clearer idea of where tokens came from. Throughout the learning process the transformer will be able to differentiate between different modules as the tag injects a numeric meaning to all tokens. The complete spatial embedding process, with the additional static tag $arrow(s)_m$ for each module $m$ thus takes form:
$ arrow(z)^(m)_n = arrow(e)^("  m")_"token, n" + arrow(e)_"pos" + arrow(s)_m $ 


=== Attention Layers
Having obtained a set of tokens from the batch of the dataset, the next step dictates how these tokens communicate information between each other and consequently update their embeddings. The core priniples demanded from this step are:

- Module isolation
- Shallow self-attention

The idea of keeping the modules in isolation at first is to give later cross-attention process a richer input. By allowing e.g. a patch embedding stemming from the DTM to understand not only itself but its surrounding patches, we save computational cost for later steps as this module delivers an already pre-trained version of every patch. Model weights in this step are only shared between and trained by embeddings of the same module. The idea of keeping this initial self-attention shallow is motivated by the goal of the model. The objective is not for every set of module tokens to best understand its surroundings, but much rather get an initial idea of it, to carry it further into the next model step.

#todo(stroke: green)[Specify number of heads, number of blocks]
#todo(stroke: orange)[Diagram of attention heads]

=== Fusion Layers
With a set of pre-attended tokens we now move onto the fusion step, which represents the first time tokens from different modules are exposed to each other. The traditional implementation with this goal in mind is to use cross-attention heads in which the query vector belongs to the first, and both key and value vector belonging to the other. This step then fuses patch information accross modules. 

The methodology chosen here for cross-attending such data is implicit. Rather than taking every token on its own and letting it communicate to all other tokens of all datasets we simplify the process, as the computational complecity of such a module would scale exponentially with the amount of patches and the amount of datasets as input. The objective here is to rule out potentially unnecessary attention computations e.g. two patches describing vegetation stress and wind that lie far apart in the area of interest. To motivate the following simplification, a short revisit to the core idea of the original objective of cross-attention for translation learning tasks. A sentence, in which every word forms a token, will likely be rearranged in the translation to another language (see @fig:translation). In the english language we expect the embedding of #emph[red] to have a large influence that of #emph[fire] and likewise in its french counter part. When cross-attending these tokens of the two datasets, we expect no consistency in the arrangements in the positions of directly translatable words. Therefore, a cross-attention mechanism considering all tokens is mandatory.

#figure(
  rect[#strong[EN:] The red fire spreads accross the land \
      #strong[FR:] Le feu rouge se propage à travers le pays],
  caption: "Example of token order of two cross-attending datasets"
) <fig:translation>

This concept is however not consistent in the vision transformer. When dealing with datasets that embody the same area of interest and in the same order, we can confidently expect token 1 to describe the same entity in both images. This very effect allows for an elegant simplification of cross-attention with which we can greatly constrain cross-attending tokens and computational cost. 

The fusion layer concatenates tokens for every patch and stacks their values into one new embedding vector. The model thus holds all information later used for prediction in only one set of patch tokens. With this set it then performs self-attention or since multiple modalities are used a quasi-cross-attention block. Since every token in one module cross-attends to every other token in another module very similarly, this step simply carries out this operation together as we expect these token pairs to weight on each other quite similarly. This part of the encoder carries the weight of deep self-attenion with n blocks of x attention heads each. The product of these attended tokens then represent the final embeddings of each patch in the vector space that characterizes the prediction of ground truth. Tokens with similar physical properties are represented by similar locations in this embedding space.
#todo(stroke : red)[Fill head & block values]
#todo(stroke: red)[Ammend fusion part, explain convolution]

=== Decoder Layer
The final task of the vision transformer is to take the embedded vector representations in the embedding space and project them back into a form which represents our ground truth: a 2D image. To clarify, the tensor at this point carries an embedding for each patch, which needs to be projected back from embedding space into 2D:

$ T_"encoder" in RR ^(B times n_"token" times d_"embed") $

It is of course possible to make predictions for every patch. This is however not just an oversimplification but does provide the opportunity to predict for as many samples as possible. @fig:decoder shows the complete structure of the decoder process that projects back to image space:

#figure(
  table(
    columns: 3,
    align: horizon,
    inset: 10pt,
    stroke: (bottom: 1pt + black),
    [*Step*], [*Tensor Size $macron(T)$*], [*Description*],
    [ Input ], [ $B times n_"token" times d_"embed"$ ], [ Output of attention blocks ],
    [ Reshape 1], [ $B times d_"embed" times h_"pad" times w_"pad"$ ], [ Reshape back to 4D feature map ],
    [ Convolution 1 + BatchNorm / ReLU], [ $B times d_"embed" / 2 times h_"pad" times w_"pad"$ ], [ $3 times 3$ kernel with padding, reduce channel depth],
    [ Upsample 1 ], [ $B times d_"embed" / 2 times 4 h_"pad" times 4 w_"pad"$ ], [ $4 times$ scale, bilinear interpolation ],
    [ Convolution 2 + BatchNorm / RELU], [ $B times d_"embed" / 4 times 4 h_"pad" times 4 w_"pad"$ ], [ $3 times 3$ kernel with padding, reduce channel depth ],
    [ Upsample 2 ], [ $B times d_"embed" / 4 times 16 h_"pad" times 16 w_"pad"$ ], [ $4 times$ scale, bilinear interpolation ],
    [ Convolution 3], [ $B times 1 times 16 h_"pad" times 16 w_"pad"$ ], [ $1 times 1$ kernel, pixel-wise classifier],
    [ Slice], [ $B times 1 times h times w$ ], [ Un-pad operation to original input size ],
  ),
  caption : "Step-wise decoder mechanism & prediction head"
) <fig:decoder>

=== Model Parameters

#figure(
  table(
    columns : 2,

    stroke: (x, y) => if y == 0 {
    (bottom: 0.7pt + black)
    },

    align: (x, y) => (
      if x > 0 { center }
      else { left }
    ),

    table.header(
      [ *Module Name* ], [ *Parameter Count* ]
    ),
    [static_embeds], [98,688],
    [dynamic_embeds], [1,311,232],
    [static_encoders], [1,779,072] ,
    [dynamic_encoders], [2,372,096],
    [static_mixer], [49,280],
    [dynamic_mixer], [65,664],
    [module_fusion], [66,048],
    [decoder], [92,481],
    [Standalone Parameters], [896],
    table.hline(),
    [TOTAL], [5,835,457],
  ),
  caption : "Model Parameters per class
   "
)
#todo(stroke : green)[Update with final params]

== Training Routine
#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Training routine],
  pseudocode-list[
    + *Input:* Load dataset, initialize dataloader, move to GPU
    + Initialize loss function, optimizer, model from config
    + $"epoch_loss" = 0$
    + *for* eopoch in num_epochs:
      + *for* batch in dataloader:
        + *Predict* for batches given static & dynamic inputs
        + Compute *losses* based on WBCE, predictions & *ground truth*
        + Set *optimizer* gradients to zero
        + Backward pass
        + Update *weights*
        + $"epoch_loss" += "loss"$
      + if epoch_loss <= best_loss:
        + Save *model*
  ]
)
#todo(stroke : green)[Update with validation workflow]



=== Loss Function

Succeding a succesfull forward pass, we evaluate the accuracy of the model by comparing its prediction to the ground truth as defined by a loss function. The main specification the loss function has to adhere to in this problem set-up is the type of target domain in its binary form (fire / no fire). The other is the lopsidedness of our target categories. Due to the very rare nature of fire events, most of the target data describes no-fire, or to the model: zeros. We therefore choose a categorical loss function that in addition to being sensitive to class imbalances is scaleable by a custom weight, the Binary Cross-Entropy (WBCE) loss function. It depends on the model prediction $p_i$, the ground truth $y_i$ and positive weight $w$. The loss function sums $N$ comparisons in total, one for every pixel in the ground truth.

$ L =  -1 / N sum_(i = 1)^N w y_i log(p_i) + (1-y_i)log(1-p_i) $ <eq:wbce>

This setup allows for customizable weighing of output cases. As a concrete example, @eq:wbce allows us to treat categorical errors made by the model differently. The penalty given by the loss function should be much higher for an incorrect prediction of high-likelihood of fire than a missed prediction of a fire as the no-fire category dominates in overall frequency. This also prevents the model from simply predicting zero to all images, in which case it would still be correct in the majority of cases, however wildly missing the point of its training exercise.

@fig:loss_3d shows the loss plane for all possible prediction/ground-truth combinations and a class imbalance $w=5$. This illustratess a zero loss for all correct predictions along with the two extreme parts of the domains in which the model has made an error. By predicting less than the actual ground truth the model is conistently penalized harsher than vice-versa.

Parameter choice of the positive weight $w$ is determined by the entirety of the training dataset before training and is then passed as an input parameter. In essence, all ground truth images are summed to yield a balance of categories.

#figure(
  image("../figs/wbce_loss.svg", width:120%),
  placement: auto, 
  caption: [Loss surface of Weighted Binary Cross Entropy (WBCE) loss function],
) <fig:loss_3d>



=== Back Propagation & Optimizer <subsec:backprop>
Having obtained a loss for a pair of data and target, the penultimate step requires computing the required weight updates due to the loss. PyTorch has an efficient implementation of this via its *optimizer* modules, chosen here is the Adam optimizer, a stochastic algorithm for first-order gradient-based optimization 
#citep(<optimizer_adam>)#citep(<pytorch_optim>). The optimizer has access to the model parameters as well as computed. For one set of parameters $theta_t$ the optimizer computes:


$ theta_(t+1) = theta_t + nabla_(theta)L $

A baseline learning rate is defined via the project configuration and passed to the optimizer class. It uses this as a baseline value and adjusts it adaptively in order to stabilise and accelerate parameter convergence.

#todo(stroke : green)[Reference pytorch docs]

=== Test Cases