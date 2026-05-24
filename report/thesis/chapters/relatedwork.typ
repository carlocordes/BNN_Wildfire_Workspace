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

== Multimodal Learning Tasks in Remote Sensing

// Modeling of risk via climate markers
wildfire stats modeling @wildfire-stats-modeling



BNN for el nino climate prediction @el-nino

@durlevic
/*
== Deep Learning in Remote Sensing
=== CNN based approaches
=== Spatio-temporal learning
=== Multi-modal Earth Observation Learning
*/

Cuboid attention for space-time transformers used for forecasting time series of earth systems @earthformer

=== Wildfire Applications

@jain-ml-overview

@andrianarivony

// Fire spread stuff here
@spread-forecasting


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




== Memory Acceleration

== Research Gap
