#import "../template.typ": *
#set heading(numbering: "1.")

= Methodology <chap:methodology>
This section provides an overview of the methodological choices made in the investigation. First, a short account of the system overview is given, before an account for the construction of data into datasets is made. Further, a complete walk-through of the model architecure is given. Finally, the integration of the model into a training workflow is explained.


== Workflow & System Overview <sec:method>
To ensure consistency throughout the training, here a short explanation of the operating modules as referenced in @fig:system_arch. All information regarding harmonization and location of input data, temporal extents and specification on training parameters are stored in a configuration file. Upon retrieving data via download, an instance of the #emph[GoldenGrid] class is made, carrying all spatial and temporal parameters of the configuration to make sure data from different sources is projected alike. 

#figure(
  image("../figs/system-overview.svg"),
  caption : [Overview of System architecture]
) <fig:system_arch>

Upon training, the dataset builder is given all information regarding the configurations of data scopes as well as the splits of data into training, validation and test sets, the nature of which will be elaborated on in this section. The dataset builder is directly integrated into the training routine as the data is read directly from disk. The final products of the training routine are a model file which inherits the specifications of the model (.pt extension) as well as its trained weights along with a log file, containing time and performance records.g

== Dataset Construction
=== Timeframe Configuration <subsec:time_config>
A section of the research objective is to assess the influences of the temporal scopes of both data and predictions. Both can be variably adjusted depending on the desired test case. As we handle temporally dynamic datasets we need to define how timesteps of samples are represented in the dataset. To ensure consistency in naming these, we here declare a series of term definitions relevant for this step:


- *Sample:* Input data the model sees to form predictions; describes the set of timestamps for one forward pass
- *Target:* The collection of images, that is variable length, that is used for testing the model predictions
- *Sequence:* A pair of a sample and target
- *Dataset:* A collection of sequences
- *Sequence Period:* Separation between individual unique sequences
- *Extent:* Used both sample and target, defines their respective variable length
- *Lead Time:* The time between the last seen timestamp in the sample and first one in the target. In other words: the amount of time the model predicts into the future

#figure(
  image("../figs/timeframes.svg", width:100%),
  placement: auto, 
  caption : [Definitions of temporal scope in project datasets]
) <fig:temporal_scope>

Every sample is constructed by 3 different kinds of data, that are read, packaged together and channeled into the model:
- Static: equivalent in every sample, describes non-variable data (e.g. terrain)
- Dynamic: time-series data consisting of multiple images (e.g. wind)
- Single-dynamic: dynamic data that is less temporally dependent and assumed static over the extent of the sample (e.g. 30-day precipitation sum)

@fig:temporal_scope illustrates the terms defined above. It shows two consecutive sequences as well as their their respective samples and targets. It shows the temporal division of the sample into subsequent timesteps. The distance between sample and target corresponds to the above definition of #emph[lead-time].

The configuration file that belongs to a model training governs the total time of interest and all parameters defining the temporal relationship of input data. The dataset builder then generates a set of timestamps as governed by the rest of the temporal configurations and iterates through the respective file for every module. Sequences with missing images are skipped entirely to ensure consistency. These dynamic modules are accompanied by the static data, which is the same in every sample. The result is a set of sample/target combinations spanning the entire domain of temporal interest.


=== Torch Datasets & Tensors
Moving data from images into the Pytorch framework is done using its fundamental data structure: the *Tensor*. Like in the traditional mathematical sense this is an array of arbitrary dimension containing elements of a single data type but has a few computation-driven features. Most importantly, it is optimized to be handled by GPU. Tensors are directly associated with the model instance (elaborated in @subsec:backprop) and have an attribute that defines whether they are adjustable model parameters. This concept is specifically designed for those tensors containing model weights. PyTorch can track the operations imposed on a tensor, which makes later backpropagation to these easier #citep(<pytorch_tensor>). The sequences described in @subsec:time_config are appended into such a tensor structures and then saved as portable datasets to be made accessible on larger compute resources.



== Model Architecture <sec:model>
In order to motivate the structure of the here applied transformer, this section walks through the sequential steps incoming data takes through the model. This represents one forward propagation. The later showcased training regiment makes this forward pass once for every batch in the training data, thus cycling through it an abundance of times. The complete model architecture of the here proposed spatio-temporal vision transformer is depicted in @fig:architecture.

#figure(
  image("../figs/architecture.svg", width : 100%),
  caption : [Architecture diagram of STViT (Spatio-Temporal Vision Transformer)]
) <fig:architecture>



=== Spatial Encoding <subsec:encoding>
The initial objective of the transformer model revolves around turning input data into tokens by segmenting it into equal patchs (e.g. 16x16 pixels). The patch size, a model hyperparameter, defines the pixel extent of square patches every input image is divided into. As we handle only single-band data we treat the image as a grid of height and width: $h times w$. 

// Reflective padding (h_pad x w_pad)
This is realized via a 2D convolution with stride equal to the desired patch size. This ensures zero overlap between neighbouring patches. An important element to consider is the fact that both height and width of images need not be divisible without rest by the patch size. To counter this we use reflective padding, a process which mirrors input images around their edges to fill in pixel data until the next complete patch. Extending height and width to $h_(p a d)$ & $w_(p a d)$ allows for perfect image division into patches. As the model input consists of multiple modules $M$ for consecutive time steps $t$ that are processed in batches $B$ of variable sizes the input tensor has dimensionality:
$ macron(T) in  RR ^(B times M times t times h_(p a d) times w_(p a d)) $ 

#todo(stroke: orange)[Figure of reflective patch padding]

// Conv2d for patch embedding
Every image of dimension $h_(p a d) times w_(p a d)$ is then subject to the flattening operation, here illustrated with a patch size of 16 pixels. 

$ P in RR ^ (16 times 16) -> P_(f l a t) in RR ^256 $

Up to this point, pixel values have not changed, simply the indexing is altered. To produce the embedding, the convolution operation then maps every flattened patch into the embedding dimension. Here we showcase the operation for a patch size of 16 pixels, mapping to the embedding of dimension 128:


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

=== Temporal Encoding
The model architecture demands a separate routing of static and dynamic data channels. All dynamic modules with multiple time steps receive a temporal encoding that is an extension to the one described in @subsec:encoding. The objective of this is to give patches that describe the same physical area in different time steps a closer connection to one another. This is done by treating the different time steps for one module together. The single-image 2D convolution is replaced by a 3D convolution that treats the series of images of timesteps $t$ as a 3-dimensional set of values and projects it into the embedding dimension $d_(e m b e d)$:

$ RR ^ (t times d_(p a t c h) times d_(p a t c h)) ->  RR ^ (d_(e m b e d)) $ <eq:3d_conv>

The convolutional operation is analogous to @eq:2dkernel for a static embedding with a kernel of dimension:

$ w in RR ^ (d_(e m b e d) times 1 times P ^ 2) $

In essence, this step results in a token for every patch in every timestep of every image. This effectively patches every timestep in every module independently. The objective here is to prevent the immediate blending of time steps, as this is the objective of the transformer layers at a later stage.



=== Spatial Embedding
Having obtained a set of embeddings for all images of varying modalities in the model input, this part of the transformer architecture intends to introduce spatial and temporal relationships between the patches called #emph[embeddings]. These greatly influence the magnitude with which patches learn from each other.

The spatial embedding is applied to every patch regardless of being temporally dependent or not. The idea is to assert a learnable idea of proximity from one patch to another. As wildfires are localized phenomena, the the objective of the spatial embedding here is to give neighbouring patches a strong correlation. Distant patches, while sharing lower correlation, can still influence one another, with limited effect. The important difference to a convolution here is that we do not simply rule out distant relationships but keep them contained. Rather than imposing a relationship on the transformer, we give it rough guidelines and freedom to give high weight to the relationship even between distant patches @rotarypositionembedding. Having created a notion of spatial similarity is crucial for the later attention process in which more similar patches will conduct higher weighted attention with one another.

In order to create such a relationship we come up with an individual positional encoding for every patch that is added onto each embedding:

$ arrow(e)_"token" ' = arrow(e)_"token" + arrow(e)_"pos"  in RR ^ (d_"embed") $

To create a positional embedding that retains similarity between proximal patches, we use sinusoidal positional embedding. By imposing sine and cosine waves of varying frequencies over the patch grid, we are able to establish a unique encoding for every patch that has distinct relationships to other patches, depending on their Euclidean distance. This is done by first defining a set of frequencies with  $d = d_"embed" / 4$:

$ omega_i = 1 / 10000 ^ (i / d) $ 

The nature of the integer $4$ is that in order to create the most unique positional embeddings, we split the added partial embeddings into both height and width and also sine and cosine categories. We scale the frequencies with 10000 in order to fit multiple periods into the image. Interpreting height and width as coordinates we can thus describe the positional encoding of a patch as the vertical concatenation of these frequencies:

$ arrow(e)_"pos" = vec(sin(h omega), cos(h omega), sin(w omega), cos(w omega)) in RR ^ (d_"embed") $

@fig:embedding_dims shows samples of the modes of the positional embeddings for every patch in the input image, in this case $80 times 40$. These include waves that propagate in direction of height and width. For every patch we add the embedding values of all dimensions. 

#figure(
  image("../figs/encoding_dims_visual.png", width:100%),
  placement: auto, 
  caption: [Additive encoded values per encoding dimension],
) <fig:embedding_dims>

Given the multitude of modes, we can form very individualised encodings for all patches. @fig:embedding_simil shows the cosine similarity of patches to one another for different embedding dimensions as demanded from the spatial embedding. Every pixel in the images represents a single patch. Here, the central patch serves as an anchor-patch, with the values of all patches describing its similarity to it. This shows that sinusoidal positional embedding are able to establish similarities and differences between patches, depending on their respective proximity. Furthermore, it is important to mention that the utility of this process heavily depends on the embedding dimension chosen. A higher embedding dimension allows to pass more sinusoidal modes and hence to create a more distinct characterization of each patch. We here show the cosine similarity for three embedding dimensions. We detect a clear trend that when treating choosing higher embedding dimensions will lead to sharper definitions of similarity and differences of patches. 

#figure(
  image("../figs/encoding_simil_visual.png", width:110%),
  placement: auto, 
  caption: [Cosine similarity of patches relative to central anchor-patch at varying embedding dimension],
) <fig:embedding_simil>


=== Temporal Embedding
Weighing patches of the same modality against each other should be entirely up to the transformer model. Afterall, the idea is to find nuanced temporal relationships both within a modularity and between pairs of them. To enforce distinct temporal relationships between dynmic tokens as well as a global perception of the model for seasonality we include an additional temporal encoding for all dynamic modules.Used here is an encoding of the Julian day, which is simply an integer number of the day of the year (0-365). To make use of this in the best way, we encode the Julian day so that it captures the true nature of the climatological conditions: its cyclical nature. The embedding of a patch from a day in December should be close to one in the beginning of January, despite their Julian day being far apart.

Making such an embedding is realized through a two-step process, first mapping the Julian day $J_t$ into a clock-like cyclical two-dimensional space and then translating it into the embedding dimension. The translation into a cyclical representation $v_t$ is done using a sinusoidal projection via:

$ v_t = [sin((2 pi J_t) / (365.25)), cos((2 pi J_t) / (365.25))] $

#figure(
  image("../figs/Julian_day.png"),
  caption : "Julian Day encoding into 2D vector space"
) <fig:Julian_day>

@fig:Julian_day shows the representation of each Julian day in a two-dimensional vector space, effectively capturing cyclical relationships of months even with distant Julian days. The next step entails moving this to to the same embedding dimension which the tokens occupy. Similarly to convolution processes shown earlier, we apply a learnable weight matrix to this vector representation projecting these values into $RR^(d_"embed")$:

$ arrow(e)_"t" = W_(2 times d_"embed") dot arrow(v)_t + arrow(b) $

Having embedded all tokens of different modalities into embedding space, we additionally give each token a modality tag $arrow(s)_m$. A tag is unique per module and is a learnable parameter with the objective to give the attention heads a clearer idea of where tokens came from. Throughout the learning process the transformer will be able to differentiate between different modules as the tag injects a numeric meaning to all tokens.

The final transformer-ready embeddings for static and dynamic tokens are additions of their individual spatial, temporal and static embeddings like so:

- Static Embedding: $arrow(z)^(m)_n = arrow(e)^("  m")_"token, n" + arrow(e)_"pos" + arrow(s)_m$

- Temporal Embedding: $arrow(z)^(m)_(n,t) = arrow(e)^("  m")_"token, n" + arrow(e)_"pos" + arrow(e)_"t" + arrow(s)_m$

Simply adding large vectors to our representations of patches might seem counterintuitive at first, however it is crucial to remember that the representations of data modularities are learnable. The longer the transformer architecture trains, them more it knows to contextualize a patch of precipitation data differently, depending on the time of year. The most elegant analogy to this process is music. If the raw encoding of a patch is analogous to a melody, the positional and temporal encoding represent the harmony accompanying it. Music is propagated as the sums of interfering frequencies, however they are discretely distinguishable to the human ear. The same is true for the transformer, it learns to contextualize the raw encoding (melody) given an extra set of information, the embeddings (harmony).

=== Token Unification
The heart of the concept in this model relies on early fusion. Rather than separating tokens per module, this approach allows for patches accross time and modularity to freely communicate with all others. Despite having the drawback of computational load here, this is a crucial design choice with the purpose of not overconstraining the system. With respect to wildfire modelling, this is a great asset. Data modularities will have varying levels of temporal depndencies. As an example, while the last timestep of wind speed might be sufficient to cast a valuable prediction, the land surface temperature might need to be tracked over the course of a few days to cause wildfire favoring conditions. The setup of keeping all tokens of all modules separate rather than concatenating them early allows for exactly these dependencies to be found in attention.

 The token unification step therefore concatenates embeddings into a joint sequence, as a preparation for the next step. Crucially, the order of tokens is of no importance here as the attention steps that follow are invariant to permutation of tokens. Furthermore, due to the spatial and temporal embeddings and embedding tags, tokens are uniquely identifiable. 
The tensor at this stage holds embeddings accross all timesteps, patches and modules in one place:
$ macron(T) in  RR ^ (B times [h dot w(n_"dyn" dot t plus n_"static")] times d_"embed") = RR ^(B times n_"token" times d_"embed") $


=== Joint Attention
The tokens in the concatinated tensor at this stage query each others information for the first time. This step can be considered self-attention by mathematical standards, as all embeddings reside in the same tensor, however this really allows embeddings to query one another accross modules. This joint encoder is composed of $8$ identical blocks, consisting of $8$ attention heads each, equating to a total of 64 attention heads. Each of these have the ability to learn individual feature relationships.

To motivate the design choice of the join attention block here, a short revisit to the core idea of the original objective of cross-attention for translation learning tasks. A sentence, in which every word forms a token, will likely be rearranged in the translation to another language (see @fig:translation). In the English language we expect the embedding of #emph[red] to have a large influence that of #emph[fire] and likewise in its French counterpart. When cross-attending these tokens of the two datasets, we expect no consistency in the arrangements in the positions of directly translatable words. Therefore, a cross-attention mechanism considering all tokens accross space and time is mandatory.

#figure(
  rect[#strong[EN:] The red fire spreads across the land \
      #strong[FR:] Le feu rouge se propage à travers le pays],
  caption: "Example of token order of two cross-attending datasets"
) <fig:translation>

The same concept is true for a spatio-temporal transformer. Rather than separating modularities and conduction isolated self-attention, we consider all possibilities. An example here could be the effect of precipitation on future burn probabilities. When predicting for wildfires, the further the latest rain surge was recorded, the higher the probability for an event will be. This underlines that allowing tokens to attend accross time and modules is crucial at this point.

An indispensible mechansim that is implemented at this position during training is dropout. When learning, models tend to follow their most probable direction of learning, meaning once it locked onto characterizing a specific learnt feature, this is all it will be able to understand. Dropout, systematically works against this, as the goal is not to understand just single relationships between data, but as many as possible. Therefore, at random intervals and attention heads, dropout will temporarily zero-out all attention weights. This signifies to the encoder that for this specific iteration of epoch it will not have access to relating pieces of data in the exact way it is used to. Consequently, it will have to rely on making new connections. 


=== Disaggregation
Having attended the modular tokens, the goal is to reshape the obtained information into a prediction map. Important to consider however is that despite having obtained a set of learned tokens, they still stem from different modalities. In order to make use of dynamic and static data in the best way, the tensor contained all attended global embeddings, is disaggregated back into dynamic and static tensors. The patches from dynamic modules are first averaged accross the temporal dimension:

$ RR  ^ (B times M times t times d_"embed") -> RR ^ (B times M times d_"embed") $

It might seem that information is lost via this operation. However, when considering the previous attention block, which has the purpose to weight patches accross timesteps by their importance, this stop only aggregates this information into one time step. During training, patches from the more relevant timesteps will receive higher weights, thus the averaged representation of the time series will represent this. 

Tokens of static and dynamic layers are consequently concatinated into tensor T_decoder, setting up for the decoder:

$ T in RR ^(B times n_"token" times d_"embed") $

=== Decoder Layer
The final task of the vision transformer is to take the embedded vector representations in the embedding space and project them back into a form which represents our ground truth: a 2D image. 



It is of course possible to make predictions for every patch. This is however not just an oversimplification but also does not provide the opportunity to predict for as many samples as possible. @fig:decoder shows the complete structure of the decoder process that projects back to image space:

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
  caption : "Stepwise decoder mechanism & prediction head"
) <fig:decoder>


In this stage we use a progressive resolution decoding. After first reshaping the tensor into four dimensions, equal to the dimension of the output we require, a series of convolution and upsample steps are used. Each convolution halves the dimension of the embedding, after which the upsample redistributes this via bilinear interpolation to widen the sizes of the patches back into a full-size image. Finally, the decoder features the #emph[un-pad] operation which discards the padding created intially to prevent data disturbance. 

=== Model Parameters
By design, the total amount of parameters in the model is dependent on the temporal extent of the sample, as every patch and every timestep receives an individual encoding. The model uses 90% of its parameters on encoding while the remainder is equally distributed on embedding and decoding. @fig:param_count shows the total parameters for a baseline model with a sequence extent of 5 days.

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
    [static_embeds], [49,280],
    [dynamic_embeds], [57,472],
    [join_backbone], [1,586,176],
    [sequence_projector], [16,512],
    [decoder], [92,481],
    [Standalone Parameters], [1,536],
    table.hline(),
    [TOTAL], [1,803,457],
  ),
  caption : "Model Parameters per class
   "
) <fig:param_count>

== Training Routine
The setup of the training regimen of models is crucial as without certain guidelines, the model could make faulty predictions. The most important vocabulary in this regard is #emph[overfitting] and #emph[generalization]. Generalization describes the ability of the model to transfer its learned patterns from training data to previously unseen data. This showcases the ability to concretely learn the physical underlying the data, rather than simply memorizing it. A model is said to be overfitting, if the prediction results rely on the training data including all its inaccuracies and noise rather than actual representation of data. The training routine used herein caters to both of these issues. As a guide, @algo:training shows the pseudo-code implementation of the training regiment. 

To promote generalization of a trained model, we use a train-validation-test split of 70-15-15 respectively. These are the proportions the dataset is split into at the beginning of training. Every dataset has a distinct purpose. The training data is seen by the model on every iteration, losses are computed accordingly. The model achieves a more refined understanding of this training data in every epoch, however there is a minute distinction to be made here. We must differentiate those model improvements in the loss to those which describe a model gaining an understanding of physical phenomena and those in which it is taking a short-cut and is learning the actual data training data. The safeguard for this is the validation data. This data has no effect on the model. The losses of the model are computed against it after every epoch. We therefore only consider a model to be improving after a new lowest training loss, if the validation loss has also improved. A model cannot learn the validation data, hence if its validation loss is decreasing, it must be learning the science and not the data. The testing data is only used once at the end of the training run. Its entire purpose is to ensure the global generalization of the model. If it is vastly greater than the best validation score, the model is over-fitting to its input data.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Training routine],
  pseudocode-list[
    + *Input:* Load dataset, initialize dataloader, move to GPU
    + Calculate class imbalance positive weight $w$
    + Initialize loss function, optimizer, model from config
    + Split data (70/15/15) into training- validation- and testing sets
    + $"best_loss" = infinity$
    + *for* eopoch in num_epochs:
      + $"train_loss" = 0$
      + *for* batch in train_data:
        + *Predict* for batches given static & dynamic inputs
        + Compute *losses* based on WBCE, predictions & *ground truth*
        + Set *optimizer* gradients to zero
        + Backward pass
        + Update *weights*
        + $"train_loss" += "batch_loss"$
      + Compute: *validation loss* on validation dataset
      + *if* validation_loss < best loss:
        + best_loss = validation_loss
        + *Save model*
      + *if* no improvement for 7 epochs:
        + *break*
      + Compute: *test loss* on test dataset
  ]
) <algo:training>


=== Loss Function

Following a successful forward pass, we evaluate the accuracy of the model by comparing its prediction to the ground truth as defined by a loss function. The main specification the loss function has to adhere to in this problem set-up is the type of target domain in its binary form (fire / no fire). The other is the lopsidedness of our target categories. Due to the very rare nature of fire events, most of the target data describes no-fire, or to the model: zeros. We therefore choose a categorical loss function that in addition to being sensitive to class imbalances is scaleable by a custom weight, the weighted Binary Cross-Entropy loss function (WBCE). It depends on the model prediction $p_i$, the ground truth $y_i$ and positive weight $w$. The loss function sums $N$ comparisons in total, one for every pixel in the ground truth.

$ L =  -1 / N sum_(i = 1)^N w y_i log(p_i) + (1-y_i)log(1-p_i) $ <eq:wbce>

This setup allows for customizable weighing of output cases. As a concrete example, @eq:wbce allows us to treat categorical errors made by the model differently. The penalty given by the loss function should be much higher for an incorrect prediction of high-likelihood of fire than a missed prediction of a fire as the no-fire category dominates in overall frequency. This also prevents the model from simply predicting zero to all images, in which case it would still be correct in most cases, however wildly missing the point of its training exercise.

@fig:loss_3d shows the loss plane for all possible prediction/ground-truth combinations and a class imbalance $w=5$. This illustrates a zero loss for all correct predictions along with the two extreme parts of the domains in which the model has made an error. By predicting less than the actual ground truth the model is consistently penalized harsher than vice-versa.

Parameter choice of the positive weight $w$ is determined by the entirety of the training dataset before training and is then passed as an input parameter. In essence, all ground truth images are summed to yield a balance of categories.

#figure(
  image("../figs/wbce_loss.svg", width:120%),
  placement: auto, 
  caption: [Loss surface of Weighted Binary Cross Entropy (WBCE) loss function],
) <fig:loss_3d>



=== Back Propagation & Gradient Descent <subsec:backprop>
The goal of the training regimen is to produce the combination of model parameters that results in the least amount of loss, more specifically a minimal validation loss. To achieve this, the loss is evaluated against all model parameters after every batch. This optimization problem can be thought of a local or global minimum in the #emph[loss-landscape] of $L(w_1, w_2 ...w_i)$ to a model parameter combination that produces a lower loss, the following steps work to descend along the steepest gradient of this landscape. To do this, we calculate the gradient of the loss due to each model parameter: $ g_t = (partial L) / (partial w_i) $

This tries to approximate the extent to which the loss, which describes the discrepancy between prediction and target, is due to this specific parameter. Parameters with a high gradient shall be updated more aggressively than those being responsible for little loss. The method for updating these model parameters is accompanied by a further parameter, the learning rate $eta$. This dimensionless value scales the overall aggressiveness of model updates. It's value heavily influences the stability of the training process. If chosen inadequately, the model may never find an equilibrium state. The updated weight parameters $theta$ are sums of their original values and the loss gradient, scaled by the learning rate:
$ theta_(t+1) = theta_t + eta g_t $ <eq:weight_update>

=== Optimizer & Learning Rate Scheduler
Pytorch uses to further implementation to aid this model weight update process. First is the learning rate scheduler, in this case we use the OneCycle learning rate scheduler. This system is designed to dynamically change the learning rate to a dynamic one that changes with every epoch ($eta_t$). The job of this specific scheduler is to start with a lower learning rate and grant the system so-called warm-up epochs. This caters to one specific problem that is common in machine learning model training and specifically one of this kind which deals with high class imbalance. The model weights, as initialized before the first training loop are random and represent no coherence with the data or reality. Applying a full-scale learning rate to these can lead to erratic loss jumps in subsequent epochs. Using the earlier landscape analogy this corresponds to overshooting a downhill section in a landscape. Once the training process has found its momentum, the scheduler will gradually increase the learning rate to a higher value. This is the stage in which the gradient descent process has gained its momentum, and we require computed loss gradients to have maximum effects on the model parameters. It is this exact stage in which the model learns most about its tokens. As the loss function approaches its local minimum the scheduler ideally decreases the learn rate again, as not to overshoot the minimum. The effect of this is a user setting and is proportional to the total amount of epochs the system is in training for.

Another crucial implementation aiding training performance is the optimizer. Here chosen was the #emph[AdamW] optimizer. This creates a valuable synergy with the learning rate scheduler to produce and enhanced image of how weights should be updated, following their loss computation. Along with the gradient a weight produces for the loss, the optimizer tracks momentum vectors $accent(m, hat)_i$ for each weight over the course of a few epochs. The logic insinuates that a parameter improving in the same trajectory for subsequent epochs, will likely continue in the same direction. It can therefore confidently receive a larger update as compared to a parameter that has recently stagnated or fluctuated. Together with the scheduled learning rate the weight updates postulated in @eq:weight_update becomes:

$ theta_(i+1) = theta_i - eta_t dot (accent(m, hat) / (sqrt(accent(v, hat)_t) + epsilon) +  lambda theta_i) $

Another parameter $accent(v, hat)_i$ tracks the erraticism a parameter shows during epochs. Those values behaving unpredictably will receive suppressed updates while those showing consistency will be rewarded greater updates #citep(<optimizer_adam>)#citep(<pytorch_optim>).


== Experiments
To provide valuable discussion of the research questions defined in @fig:research_questions, the experiments carried out for this work each cater to one of them. The main study involves creating a baseline version of the spatio-temporal transformer as well as the general study of how the labeled binary ground truth data yields a predicted uncertainty map. By studying an entire year worth of predictions, this experiment is able to establish initial spatio-temporal pattern the transformer learns.

The next section focuses on the main objective of this investigation, which is to investigate how varying time configurations affect the model. More closely, we investigate how the amount of temporal context, or as here named sample extent, is useful or ideal for spatio-temporal transformers to cast the most precise predictions. The analysis will include a performance comparison of training parameters as well as quantitative discussion of spatial risk distribution is given.

The third experiment focuses on the capacity of the model to make accurate predictions when elongating lead-time and viewing further into the feature. By modifying the set-up of system configurations, we test five different versions and compare elongated lead-time model accuracy.

Experiment four focuses on evaluating the validity of the methodology in terms of its input data. An ablation study into feature impact is carried out, via predicting a set of targets with a constrained model. Essentially, removing information and tracking the added error in the outcome allows for a systematic ranking of feature impact.

In short, the experiments presented in @chap:results each address a research question of @fig:research_questions:

- *RQ 1* $arrow$ Experiment 1: Baseline Model
- *RQ 2* $arrow$ Experiment 2: Varying sample/target length
- *RQ 3* $arrow$ Experiment 3: Lead time horizon stress test
- *RQ 4* $arrow$ Experiment 4: Ablation studies