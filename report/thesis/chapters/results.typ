#import "../template.typ"

= Results
To start guiding through the results of this investigation, the research questions formulated in @fig:research_questions will continuously be referenced in the following section to ensure concrete discussion of the defined goals. 

== The Baseline Model
// Basic introduction to results

As a proof of concept and a discussion of *RQ 1* in @fig:research_questions, here follows a description of the choices made to yield a working model that can confidently predict wildfire occurence probability.

// Difficulty with large penalization of loss function, lead to training instability.
All model training was conducted according to the specifications in @algo:training. The objective of it is to obtain a set of model parameters, that produce a highly skilled, but generalized prediction mechanism. With the subdivision of data into training, validation and testing data a few important guidelines were implemented here to ensure valid training performance of the models. First, the training loss should be a smoothly converging curve, asymptotically approaching a minimum. The same goes for the validation performance for each epoch. Once validation loss no longer improves but starts to fluctuate or increase, the model has shifted from learning underlying process to learning the training data. This can be seen in most training performance curves and is the condition under which training is stopped. These will later be presented in detail for each sub-experiment for this investigation.

The most troublesome phase of training in general was the initiation phase of the model. In the first iteration of the model, all weights are initialized at random. Combining this with an aggresive loss function like WBCE can cause quite a bit of instability. For every false positive prediction, the model weights produce high loss and receive large corrections. This does not improve the model, but pushes its weights into a new, random and unimproved direction. Phenomena like these can be tracked via the training loss curve, as it will result in sudden erratic spikes, rather than a smooth convergence. To counter this effect, a lower learning rate than originally desired was chosen. The learning rate scales the magnitude of corrections made to model weights and by lowering it, stability is enforced in the beginning of training. A drawback of this is training speed. As the learning rate is lower, more epochs will have to be run to achieve the same loss improvement. Additionally adjusting the learning rate scheduler to spend $30 percent$ of the epochs in a warm-up period with further reduced learning rate was a great asset. 


// General results, showing seasonal risk
To establish the feasibility of our model, @fig:seasonal_prediction, shows a few sample predictions of Portugal, with varying timeframes. Here, we predicted over the course of an entire calendar year (2025) and show excerpts for each season that can be qualitatively discussed.


#figure(
  image("../figs/season_pred.png"),
  caption : [Seasonal wildfire probability predictions over Portugal]
) <fig:seasonal_prediction>


First, it is highly notable that by exposing the transformer to binary classes (fire/no-fire) and infering a burn probability from them is possible. By allowing the WBCE loss function to put great emphasis on not missing fires, the model is able to infer similar risk for regions of similar conditions even if no fire was spotted there. This proves that the model operates as a bridge using binary classes to infer probabilities. This allows for outputs to hold much richer information about high risk regions which use infered information from other pixels.

Next, it is without surprise that the model has successfully learned the difference between land and water, as clearly visible by the values drawing a clear coastline in @fig:seasonal_prediction. By making concentrated choices with no-data values and masking operations, the attention heads in the backbone seem to know not to infer information into watermass, essentially treating it as a natural barrier to wildfire.

Furthermore, a strong seasonal trend can be detected. Apart from a few anomalies, the overall risk is near zero for early months as Portugal is in winter season. Peak fire periods are June to September as shown, after which rain season in October usually takes wildfire risk down to zero again. By encoding of the Julian day, the general disparity between winter and summer climate or a combination of both, the model has learned the ability to distinguish seasonality.

// Sampling logits back to a probability via class imbalance (log(600))

== Experiment 1: Sample Extent
The first sub-investigation as defined by *RQ 2* in @fig:research_questions. Conceptually, this investigation attempts to discover the amount of context necessary for a valuable prediction to be made. The setup of the spatio-transformer allows for dynamic scaling of input features. The temporal scope of dynamic features, or simply the amount of images ingested per data module is variable. However, the early fusion concept chosen here demands every input token to be treated equally. This has the effect that the number of images in the input sample linearly scales the amount of tokens, which in turn scale the computational time quadratically. The maximum sample length feasible on hardware available is seven days, above which could not be investigated. In this experiment a lead-time of zero days was chosen. Four otherwise equally defined models were trained with respective sample extents of 1, 3, 5 and 7 days.

#figure(
  image("../figs/model_seq_compare.png", width : 100%),
  caption : [Training performance of models with varying sample length]
) <fig:samp_train_results>

@fig:samp_train_results shows the 



#figure(
  image("../figs/lift_samp.png"),
  caption : [Lift curves and Gini-coefficients of comparable models]
) <fig:fsamp_lift>

== Experiment 2: Lead-Time

#figure(
  image("../figs/model_lead_compare.png", width : 100%),
  caption : [Training performance of models with varying lead time]
) <fig:lead_train_results>

#figure(
  image("../figs/lift_lead.png"),
  caption : [Lift curves and Gini-coefficients of comparable models]
) <fig:lead_lift>


== Experiment 3: Ablation Studies

#figure(
  image("../figs/seasonal_implicit_trend.png"),
  caption : [Seasonal Ablation Profile: Component Reliance Across Time]
)


#figure(
  image("../figs/ablation_mse_mae.png"),
  caption : [Model Sensitivity Analysis by Data Module]
)

// Experiment 4: Vector space representation in 2D