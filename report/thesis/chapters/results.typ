#import "../template.typ"

= Results

// General results, showing seasonal risk




// Experiment 1: Sample Length

#figure(
  image("../figs/model_seq_compare.png", width : 100%),
  caption : [Training performance of models with varying sample length]
) <fig:samp_train_results>

#figure(
  image("../figs/lift_samp.png"),
  caption : [Lift curves and Gini-coefficients of comparable models]
)


// Experiment 2: Lead-time

#figure(
  image("../figs/model_lead_compare.png", width : 100%),
  caption : [Training performance of models with varying lead time]
) <fig:lead_train_results>

#figure(
  image("../figs/lift_lead.png"),
  caption : [Lift curves and Gini-coefficients of comparable models]
)


// Experiment 3: Ablation studies
== Ablation studies

#figure(
  image("../figs/seasonal_implicit_trend.png"),
  caption : [Seasonal Ablation Profile: Component Reliance Across Time]
)


#figure(
  image("../figs/ablation_mse_mae.png"),
  caption : [Model Sensitivity Analysis by Data Module]
)

// Experiment 4: Vector space representation in 2D