#import "../template.typ": *

= Related Work <chap:related_work>


== Drivers & Metrics of Wildfire Risk Modeling

// Motivation of why risk is being tracked
Wildfire events are highly complex phenomena. First and foremost, due to its highly violent nature, fires in the wild as well as urban areas pose as one of the most destructive of natural hazards. Furthermore, the stochastic yet extremely rare nature of such events pose as a challenge in quantifying, reasoning and mitigating such events. On top of being difficult to predict, such fire events are highly locally concentrated. In 25 years of recorded data in Spain, a region of 6% of the total landmass is responsible for the majority of outbreaks @galicia-fires. Having a close pulse on such events is crucial as forrest systems serve a crucial importance to establishing a healthy calbon balance and are a fundamental component of mitigation global carbon combustion @illarionova-carbon-forest.

In recent decades, many areas world-wid have been experiencing an increase of such events. While a great deal of the scientific discussion has been centered around correlating land use and risk mitigation to the occurence of these fires, a large factor can be attributed to a novel set of climate conditions. In the United States, more particularly in the western state of California, a large spike in suh events from the 1980s and onwards can be found to correlate significantly with the conditions a warming climate poses @westerling-cal-fire. With elongated dry summers, drought and stagnating vegetation health, especially regions of mid-elevation are found to be directly be affected by the characteristics of a warming climate.


// Splitting into root causes of ignition vs. spread & climate drivers
Assessing and describing the effects of wildfires can be essentially split into two disciplines. Conceptually, a fire disaster describes the propagation of a fire event through combustible material, given the initial start of a fire. The terminology to focus on here are the probability of ignition of a fire and the thereby causal spread of it. Describing these two disciplines separately as well as quantifying their interdependencies lies at the heart of quantifying the risk a fire poses @rego2010towards. In turn, this allows for authorities to make concentrated choices regarding mitigtion measures, in which both spark and ignition have to be addressed. In essence, the discipline can be contextualized as the conditional probability of a spread of a fire given an ignition.

Studies find global wildfire ignition with 90% likelihood to be caused by human activity @human-ignition. A further investigation of fire ignitions in California, US used 10 years worth of ignition data to show the local dispersion of events caused by humans versus other effects. It showed strong spatial between human-induced events and urban areas as well as remote nature and lightning strikes @chen-fire-california. It could be shown that this effect is also highly locally optimized and needs to be assessed regionally, once again underlining the importance of good interpretation of local climate health conditions.

Taking ignition probability out of the picture, there are a few crucial factors influencing the extent, intensity and probability of spread. Most importantly, average daily temperatures, correlating with summer season make the biggest effects. Along with a deprecated annual rainfall rate and sparsity thereof, the general distribution of anthropogenic activity is a notable factor. Furthermore, studies have linked gross domestic product (GDP) density to the extent of fire events @aldersley2011. Urban areas therefore not only contain more sealed surfaces, posing as a natural fire barrier but with rising GDP per area, extended mitigation measures could be the underlying factor of less burnt area occuring in such high value producing areas.

Other factors influencing burn extent include natural difference vegetation indices (NDVI). This monitors vegetation health through infrared radiation from satellite stations. Vapor pressure, humidity are contributing factors. A factor that can lead to containment or further acceleration of such events is topography, as terrain is more susceptible to propagating fire in high-slope conditions. Similarly, wind has the analogous ability to shift flame angle to set neighboring combustible material afire. Lastly, determing biomass density and fuel moisture are avid determinants of risk @rafaqat. Wind is an especially dangerous factor as at high velocities it can also lead to transport of burning materials, sparking new fires elsewhere, a factor that was especially harmful in the January 2025 fires in Los Angeles @fire-speed.


== The Transformer <sec:transformers>
=== Background & Motivation 
Processing data in the machine learning domain, whether text or image, was historically dominated by Recurrent Neural Networks (RNN) or Long-Short-Term Memory (LSTM). These types of learning networks rely on a chain-like structure, processing data in series to make a prediction. Speaking specifically for processing sequences of satellite imagery, a bottleneck arises when scaling to large datasets. As the proccessing of data sample $n_t$ is dependent on $n_(t-1)$, such a process is inherently inefficient due to its incompatibility to be parallelized. Furthermore, such models are subject to the #emph("Vanishing Gradient") effect. This describes the loss of information a sequential model encounters as the size of the input sequence increases. Every element is dependent on the one before it, so as a signal passes through larger amounts of computations, information will get lost #citep(<rnn-vanishing>).

A fundamental breakthrough was made in modern technology with the transformer architecture, the goal of which was to eliminate the sequential computation bottleneck. In direct consequence, this would tackle the the issue modeling accuracy being dependent on the distance of elements of the encoded input sequence. The transformer, as originally conceived for nerual translation tasks, was the pioneering idea in this regard. The proposed architecture relies not on processing data sequentially, but treating the input data entirely at once. Inputs no longer rely on their relative position in the sequence, but are much rather position invariant concepts. Every part has the direct ability to interact with all others without relying on those in between #citep(<attention>). The regressional analysis was consequently replaced by the attention mechanism, that allows sequence segments to self establish their relevance and corellation to each other.




=== Architecture

// Tokens, Embeddings, Idea of vector spaces
The core idea behind the transformer is to form predictions given an input, depending on the application. As mentioned above, the inital design involved translating text between languages not just word for word, but with strong relational semantic meaning. This same concept was later adapted for a variety of different tasks, perhaps most prominently text generation  tasks. Modern chatbots use this very mechanism to from educated prediction of outputs given a user input sequence.

Generative Transformers that produce either images of text from input sequences are pre-trained, meaning their weights are set. Under the hood are various matrix multiplication processes that turn inputs into outputs. Crucial for this to work however is the processing of input sequences into a stucture the transformer architecture can understand. To do this, we consider partitions of the input as so-called #emph[tokens] @embeddings as shown in @fig:tokenization.

#figure(
  rect[
    #strong[Raw: ] The capital of Italy is \
    #strong[Tokens:] [The] [capital] [of] [Italy] [is]
  ],
  caption: "Tokenization of an input"
) <fig:tokenization>

Each of the input tokens is then processed via a process called embedding. This simply assigns every token to a high dimensional vector:
$ "Italy" -> arrow(e) =  mat(e_1; e_2; e_3; dots.v;) in RR ^(d_"embed") $

To an untrained model this might appear like a random projection of a word into a vector, however this process is learnable, meaning that the way a token might be transformed into an embedding is optimized as a model trains. As a result, trained models gather their entire understanding of reality by assigning many tokens each a distinct position in a very high dimensional embedding vector space. A trained model not only has a position for every term in its vocabulary, but the relative position of these can directly embody similarities of terms in plain language. For example, consider a pre-trained transformer that holds embeddings for the terms #emph[brother], #emph[sister], #emph[aunt] and #emph[uncle] in @fig:embedding_relative.

#figure(
  image("../figs/2d_vectors_and_relative.png"),
  caption : "Oragnization of tokens into vector space and relative positional meanings"
) <fig:embedding_relative>


The imporant notion to realize is that a trained transformer is able to relate certain terms to one another. Subtracting the vector of #emph[aunt] and #emph[uncle] will yield a similar vector to the one when applying this logic to #emph[brother] and #emph[sister], a vector that to the transformer encodes a logic of relative gender. Furthermore, taking the dot product of two vectors will provide information about similarity of these two vectors. Analogously to vector geometry, the dot product of two unit vectors pointing in different directions will yield zero: 
$ accent(e, hat)_i dot accent(e, hat)_j = delta_(i j) = {1 "if" i = j, "else" 0 $
Suppose such vectors were describing directions in a coordinate system. A zero dot-product dictates that these two vectors are very dissimilar, as they encode different directions. The same applies to the embedding dimension. A large dot product of two embeddings signifies similarity between these terms @3b1b-transformers.



=== Attention
// Key, Query, Value
Embedding the tokens of an input sequence like in @fig:tokenization, for now only gives a vectoral meaning to each individual one, rather than as a complete input sequence. However, as the core concept here is to make predictions from an input sequence as a whole, a mechanism relating the meaning of each token into a conglomerate understanding is neccessary. This exact problem is solved by the concept of attention, a powerful matrix encoder that not only serves as the basis of this project, but is responsible for the immense growth in the artificial intelligence sector @3b1b-attention. 

Despite being powerful, this concept is very simple. It determines the relative importance of all embeddings to all other embeddings in the input and subsequently updates each embedding, depending on this relative importance. This step can be conceptualized as tokens absorbing information from others. For example, the embeddings of #emph[green apple] will hold much richer meaning after an attention block as the token #emph[apple] will have *attended* to #emph[green]. Thereafter, the embedding of apple is updated. It now sits in a slightly shifted position in embedding space, carrying this richer, updated information. 

Mathematically speaking, the attention block is a scaled matrix multiplication. To apply attention to the input sequence, first let the concatination operation of all tokens of a sequence into a structure that will be called the Value matix $V$:

$
  "[The] [capital] [of] [Italy] [is]" -> 
  mat(
    dots.v, dots.v, dots.v, dots.v, dots.v;
    arrow(E_1), arrow(E_2), arrow(E_3), arrow(E_4), arrow(E_5);
    dots.v, dots.v, dots.v, dots.v, dots.v;

  ) = V 
 $


For every embedding $E_i$ a query vector ($Q_i$) and a key vector ($K_i$) is created via convolution of weighted key and query matrices that are convolved with the embedings of each token:
$ Q_i = W_Q dot arrow(E)_i $
$ K_i = W_K dot arrow(E)_i $

Attention is then formulated as scaled the matrix multiplication of these two matrices, using softmax to scale the outputs to the domain $[0, 1]$ and scaled by the value matrix $arrow(V)$ @attention:

$ "Attention"(K, Q, V) = "softmax"((Q dot K^T)/(sqrt(d_"embed"))) dot V $ <eq:attention>

This formula acts as a bridge allowing each embedding to inform itself about the contents and relevance of all other embeddings. Every embedding posesses a query $Q$ to which the to other tokens respond with a key value $K$. The dot product between these two in @eq:attention thus defines the relevance of tokens to one another. The greater the output of this dot product, the higher the attention value will be for a token pair. The result of the scaled dot product is multiplied by the value matrix. In simple terms it can be read: one token uses its query vector to get information about another with its key vector. If that information is relevant to it and the dot product is high, a weighted part of the queried token is added onto the embedding of the querying token. 

This process is called-self attention, as the input sequence is only compared to itself. The analogous process of comparing two sequence, for example in translation tasks. The magnitude to which embeddings attend to one another is entirely governed by the weights in $W_k$ and $W_Q$, which are initialized at random and learned during model training. 

The resulting attention values are then used to update the embeddings to hold richer meaning:

$ arrow(E)'_i = arrow(E)_i + A_i $

//multi-head attention
Attention rarely is caried out in such single operations but in many blocks. The raw embeddings are fed to each of these blocks that might contain many of these operations, individually called heads. Each block and each head is thus able to learn slightly different relational attributes between the embeddings. This is called multi-headed attention #citep(<attention-positions>).

Crucially, the computations of each head and block are independent of each other, making this process highly parallelizable and computationally convenient to compute with large Graphics Processing Units (GPU).


=== Johnson-Lindenstrauß Lemma & Vector representations
#citep(<johnson-lindenstrauss>)


=== Vision Transformers
The concept of using embedded sequences as predictors quickly caught on for vision tasks. The bottleneck resided in establishing an analogy to what a token should represent in images. Using the naive approach in extracting a token for each pixel value in every image quickly scales above computable limits as using $N$ tokens for an image of height $h$ and width $w$ scales quadratically with image resolution ($N = h times w$). Furthermore, attention processes also scale as $O(n_"tokens"^2)$ as per @eq:attention, this would quickly lead to unfeasible computations @vision-trans-stats. The solution was found in extracting not raw pixel values, but patches of them. By flattening each patch into a vector and projecting this into an embedding dimension an analogous structure to embedding was found @transformer-image. It is hereby important that the embedded values are not the exact pixel values but are the product of a convolution operation. This mechanism outperformed convolutional approaches on predicting labeled such as ImageNet, CIFAR-100 or BTAB. 

// CNN do not capture distant relationships
The transformer notably solved one of the main converns of convolutional neural network approaches, which involves using a kernel as a sliding window to extract features. This process however, while being an asset at extacting local features, failed to correlate features that might have a large extent or distant relationships in an image @cnn-transformer-comparison. Because transformers have the ability to self-weigh relationships of features, this issue is almost entirely resolved. An investigation on the the large dataset Imagenet-21k could show that vision transformer models consistently outperform convolution approaches, amongst others due to its enhanced feature recognition @imagenet21k.

A new challenge with the transformer architecture is that in general, compared to its predecessors, such a model requires an extensive set of pre-labeled data in order not just to learn dependencies of its input, but to generalize well to unseen data @cnn-transformer-comparison2. 


Transformers performed well with simple classification tasks, but could initially not perform well in denser tasks, image segmentation being a prime example. This is in part due to the excess of attention calculation a model runs through for one prediction. An adaptation to this was found in the sliding-window (SWIN) which particularly for large-scale images bridged the gap of computational time from text tasks to the image interpretation @swintransformer. By only computing self-attention between non-overlapping local windows rather than the entire set, the approach brough greater efficiency.

An invaluable advancement to vision transformer came with frequent applications to remote sensing data. As satellite data is grid-like it is an ideal application. This coincided with a further enhancement to the pipeline: positional encodings. Here, the set of embedded image patches are pre-weighed, proportionally to their distance to other patches. This serves as a kickstarter to the later cross-patch attention process in which close patches will conduct attention more closely due to their initial similarity induced by the positional encoding @transformer-remote-sensing. Early appraoches adapted rotary positional encoding from natural language processing into vision approaches @rotarypositionembedding. Further progress was made with the proposal of using overlayed sinusoidal encodings, that do not just pass absolute encoding but relative ones. 

This concept could  be further extended the temporal dimension. Input images with multiple steps accross time could not only share weights given their relative position, but also would explicitly encode time via the same logic. Images of neighboring timesteps could receive more similar encodings while distant once could not conduct attention as closely @time-encoding @video-vision-transformer     .            

== Multimodal Learning Tasks in Remote Sensing
Spatio-temporal attention mechanism find a wide array of applications in the remote sensing sector as with sufficiently large datasets, they tend to outperform other architectures @rs-transformers. Specifically for climate phenomena prediction, learning architectures performed well, even before the initiation of vision transformers. Notably, a bayesian neural network architecture was able to produce numerical predictions of el Niño, taking a multi-modal, multi-year dataset to produce valuable prdiction results. The method elegantly predicts sea-surface temperatures under varying lead-time and at different time scales @el-nino. A 



Cuboid attention for space-time transformers used for forecasting time series of earth systems @earthformer

=== Wildfire Applications

@jain-ml-overview

@andrianarivony

@durlevic

// Fire spread stuff here
@spread-forecasting



== Memory Acceleration

== Research Gap
// Motivate with fig of difference vectors: create climate health vector space