#import "../template.typ": *

= Related Work <chap:related_work>


== Drivers & Metrics of Wildfire Risk Modeling

Many areas accross the globe have experienced increase in wildfire activity in recent decades. While much scientific discussion has been made about correlating land use to fire occurrence, a study in the western United States finds a large spike in activity from the 1980s onwards, representing a significant correlation with elongated dry summers and drought especially in the mid-elevation regions, proposing a direct correlation to a changing climate. @westerling-cal-fire

Most important drivers of risk are not localized phenomena but apply on the regional scale. High temperature, deprecated annual rainfall and prolonged dry periods account for the biggest culprits along with anthropogenic activity. Study finds burned area is always greater in regions of minimal GDP density @aldersley2011.

Among climate factors like NDVI, vapor pressure, humidity and human factors, topography could be shown to have large effects not on wildfire ignition probability but with respect to wildfire propagation. Furthermore using vegetation indices is cruical to determining biomass density and fuel moisture in risk-prone areas @rafaqat

Study finds California fire ignition between 2000 and 2010 to be almost entirely distributable between human activity and lightning strikes, by assessing remote and population-proximal events #citep(<chen-fire-california>)

Forrest ecosystems serve crucial importance to a healthy carbon balance and are a fundamental component of mitigating carbon combustion.  @illarionova-carbon-forest

Wildfire events are localized phenomena. In 25 years of recorded data in Spain, a region of 6% of the total landmass is responsible for the majority of outbreaks. @galicia-fires

The investigation into wildfire mitigation measures can essentially be divided into two disciplines: ignition and propagation. The ability to make concentrated choices on mitigation measures starts with prevention. Therefore ignition is primary factor. Wildfire occurence can be thought of as the conditional probability of spread given an ignition. @rego2010towards



== Multimodal Learning Tasks in Remote Sensing
BNN for el nino climate prediction @el-nino
/*
== Deep Learning in Remote Sensing
=== CNN based approaches
=== Spatio-temporal learning
=== Multi-modal Earth Observation Learning
*/

== The Transformer <sec:transformers>
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



== Machine Learning in Wildfire Modeling

@jain-ml-overview

@durlevic


== Memory Acceleration

== Research Gap
