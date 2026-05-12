#import "../template.typ": *

= Related Work <chap:related_work>

//== Wildfire Risk Assessment <sec:wildfire_risk>



== Transformers <sec:transformers>
=== Background & Motivation 
Processing data in the machine learning domain, whether text or image, was historically dominated by Recurrent Neural Networks (RNN) or Long-Short-Term Memory (LSTM). These types of learning networks rely on a chain-like structure, processing data in series to make a prediction. Speaking specifically for processing sequences of satellite imagery, a bottleneck arises when scaling to large datasets. As the proccessing of data sample $n_t$ is dependent on $n_(t-1)$, such a process is inherently inefficient due to its incompatibility to be parallelized. Furthermore, such models are subject to the #emph("Vanishing Gradient") effect. This describes the loss of information a sequential model encounters as the size of the input sequence increases. Every element is dependent on the one before it, so as a signal passes through larger amounts of computations, information will get lost #citep(<rnn-vanishing>).

A fundamental breakthrough was made in modern technology with the transformer architecture, the goal of which was to eliminate the sequential computation bottleneck. In direct consequence, this would tackle the the issue modeling accuracy being dependent on the distance of elements of the encoded input sequence. The transformer, as originally conceived for nerual translation tasks, was the pioneering idea in this regard. The proposed architecture relies not on processing data sequentially, but treating the input data entirely at once. Inputs no longer rely on their relative position in the sequence, but are much rather position invariant concepts. Every part has the direct ability to interact with all others without relying on those in between #citep(<attention>). The regressional analysis was consequently replaced by the attention mechanism, that allows sequence segments to self establish their relevance and corellation to each other.




=== Architecture
// Tokens, Embeddings, Idea of vector spaces

=== Attention mechanism
// Key, Query, Value

#citep(<attention-positions>)

=== Johnson-Lindenstrauß Lemma & Vector representations
#citep(<johnson-lindenstrauss>)

=== Positional Encodings
#citep(<wang2021on>)


=== Vision Transformers



//patch embedding, for image classifiction
#citep(<transformer-image>)

#citep(<video-vision-transformer>)

/*
== Deep Learning in Remote Sensing
=== CNN based approaches
=== Spatio-temporal learning
=== Multi-modal Earth Observation Learning


== Memory Acceleration
*/
