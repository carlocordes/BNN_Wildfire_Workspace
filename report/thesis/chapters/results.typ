#import "../template.typ"

= Results


== Model Training
// Basic introduction to results
// Difficulty with large penalization of loss function, lead to training instability.

All model training was conducted according to the specifications in @algo:training. The objective of it is to obtain a set of model parameters, that produce a highly skilled, but generalized prediction mechanism. With the subdivision of data into training, validation and testing data a few important guidelines were implemented here to ensure valid training performance of the models. First, the training loss should be a smoothly converging curve, asymptotically approaching a minimum. The same goes for the validation performance for each epoch. Once validation loss no longer improves but starts to fluctuate or increase, the model has shifted from learning underlying process to learning the training data. This can be seen in most training performance curves and is the condition under which training is stopped. These will be presented for each experiment for this investigation.

The most troublesome phase of training in this use case was the initiation phase of the model. In the first iteration of the model, all weights are initialized at random. Combining this with an aggresive loss function like WBCE in this case, can cause quite a bit of instability. For every false positive prediction, the model weights produce high loss and receiv large corrections. This does not improve the model, but pushes its weights in a new, random and unimproved direction. Phenomena like these can be tracked via the training loss curve, as it will result in sudden erratic spikes, rather than a smooth convergence. To counter this effect, a lower learning rate than originally desired was chosen. The learning rate scales the magnitude of corrections made to model weights and by lowering it, stability is enforced in the beginning of training. A drawback of this is training speed. As the learning rate is lower, more epochs will have to be run to achieve the same loss improvement. 

// General results, showing seasonal risk


// Abilities of 





== Experiment 1: Sample Extent

// Explain general setup of training curve

#figure(
  image("../figs/model_seq_compare.png", width : 100%),
  caption : [Training performance of models with varying sample length]
) <fig:samp_train_results>

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