#import "../template.typ"

= Results & Discussions <chap:results>
To guide through the results of this investigation, the research questions formulated in @fig:research_questions will continuously be referenced in the following section to ensure concrete discussion of the defined goals. 

== Experiment 1: The Baseline Model
// Basic introduction to results

As a proof of concept and a discussion of *RQ 1* in @fig:research_questions, here follows a description of the choices made to yield a working model that can confidently predict wildfire occurence probability.

//TODO: Turning logits to probabilities


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


#figure(
  image("../figs/unified_comparison.png"),
  caption : [Complete and zoomed view of Sentinel-2 orthophoto, predictions and ground truth ]
)

// Talk about: it works, it predicts something right, it learned coastal contours

// Sampling logits back to a probability via class imbalance (log(600))

== Experiment 2: Sample Extent
The first sub-investigation as defined by *RQ 2* in @fig:research_questions. Conceptually, this investigation attempts to discover the amount of context necessary for a valuable prediction to be made, by varying the extent of the input sample. The setup of the spatio-transformer allows for dynamic scaling of input features. The temporal scope of dynamic features, or simply the amount of images ingested per data module is variable. However, the early fusion concept chosen here demands every input token to be treated equally. This has the effect that the number of images in the input sample linearly scales the amount of tokens, which in turn scale the computational time quadratically. The maximum sample length feasible on hardware available is seven days, above which could not be investigated. In this experiment a lead-time of zero days was chosen. Four otherwise equally defined models were trained with varying sample extents, the experiment descriptions of which are shown in @tab:samp_configurations.



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
      [ *Experiment Name* ], [ *Sample Extent* ],
    ),
    [SampleExt1], [1],
    [SampleExt3], [3],
    [SampleExt5], [5],
    [SampleExt7], [7],
  ),
  caption : [Sample extent experiment configurations]
) <tab:samp_configurations>


#figure(
  image("../figs/model_seq_compare.png", width : 100%),
  caption : [Training performance of models with varying sample length]
) <fig:samp_train_results>

@fig:samp_train_results shows the training performance of the four respective cases to be compared. Training loss histories show consistent convergence for all cases. Crucial for all of these is the region in the training loss after which the steepest slope is overcome. At this stage, models have learnt some basic feature relationships and now concretize them. It is at this stage in which the dropout provides the most value. By sporadically zeroing out weights between attending features, the model is forced to rely on different feature relationships to reduce its losses. 

All examples experience volatility in validation loss starting at epochs 20-30, while the training loss decreases steadily. This is a clear giveaway for overfitting, where the models are learning to memorize data rather than wildfire relationships.

Perhaps most critically, all runs show gaps between validation and testing scores. This can point to the general disparity of validation and testing sets, however by comparing these differences amongst the models brings some insight into their capabilities. #emph[SampleExt1] shows the biggest gap. More specifically, while obtaining the lowest best validation, it has the worst testing score. Being able to obtain non-complex seems to be feasible in the short-term but fails when data is completely unknown. As wildfires are highly dynamic and instationary phenomena, this shows that such a short sample time frame is insufficient.

On the other end of the spectrum, #emph[SampleExt7] uses the most contextual information to make predictions. While comparably having a higher average validation loss, it has the smallest disparity to its training loss, making it the best generalizer among the four. Richer, more contextualized information seems to have positive impact on both the prediction accuracy as well as the deployability of the model to previously unseen data. #emph[SampleExt3] shows quite similar conditions.

#emph[SampleExt5] acts as middle ground to the aforementioned models. Both sit in the middle for validation scores but fail to capture the enhanced generalization of the seven days of context.


#figure(
  image("../figs/lift_samp.png"),
  caption : [Lift curves and Gini-coefficients of comparable models]
) <fig:samp_lift>

To further demonstrate the models abilities and illustrate their differences, @fig:samp_lift illustrates cumulative gains of lift-charts of the models. Essentially, the methdology here argues that for such a rare event like wildfires, most of the risk should be contained in a fraction of the total area. The set of predictions, presented on the x-axis, has here been ordered in descending order of predicted risk. A model that would simply guess randomly would show a uniform distribution of risk over all pixels, as denoted by the random spatial baseline. A well-trained model on the other hand should be able to contain large amounts of the actual fires in very few image pixels.

All four models show this behaviour, proving they operate without having to guess randomly. The clear categorical winner from this perspective is again #emph[SampleExt3], with the highest overall Gini-coefficient of $0.620$. It is able to contain roughly $70%$ of actual fires within $20%$ of the total area. 

As previously #emph[SampleExt5] underperforms compared to all other models, being able to centralize significantly less burn pixels its riskiest $20%$ of its area, aligning with the previous loss analysis. 



// Summarize which model is the best SampleExt3
// Diminishing returns with overcontextualization
The results of this experiment shows, that context has immense influence on predictive validity. However, simply by adding more context and increasing sample length, accuracy will not always improve. Extremely short one-day samples obtain highest accuracy in finding patterns in known data, however struggle to generalize to unseenj samples, as shown by the testing losses. It is with less input features, that a transformer is able to specifically seek out inherent data noise and learn from it to overfit known training samples. On the other end of the spectrum lie the overcontextualized examples. As shown by the training performance and lift-cuve of the longest sample length, this maximum context example no longer has the learn trends as well as some the shorter examples. By introducing too much context, the model struggles to relate the amount of information together in a sensible way. This motivates for the existance of a point of diminishing returns. Limiting sample length to three days, provides only the most useful context and while still retaining some inaccuracies as shown by the lift-curve in @fig:samp_lift produces the best overall performance.


== Experiment 3: Lead-Time
The second experiment relates to *RQ 3* of @fig:research_questions and investigates the extent to which varying the lead-time influences prediction abilities of the transformer. To reiterate, the lead-time describes the amount of time passed between the last seen training sample and the beginning of the target, the time frame of prediction as per @fig:temporal_scope. In this instance we test for five different lead times, as shown in @tab:lead_configurations. The correct interpretation of #emph[Lead-5] is that the sample and target extents overlap by five days. It therefore does not function as a predictor, but more as a wildfire tracker and has here been included to show the influence of a temporal overlap or a negative lead time.

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
      [ *Experiment Name* ], [ *Lead-Time* ]
    ),
    [Lead-5], [-5], 
    [Lead0], [0],
    [Lead5], [5], 
    [Lead10], [10],
    [Lead20], [20], 
  ),
  caption : [Lead-time experiment configurations]
) <tab:lead_configurations>

#figure(
  image("../figs/model_lead_compare.png", width : 100%),
  caption : [Training performance of models with varying lead time]
) <fig:lead_train_results>

The training performance of all models is useful once again to determine important trends and comparisons betweeen them. Notably and without surprise, the training losses of #emph[Lead-5] are lower and steeper as compared to its counterparts. Nonetheless, despite increasing the lead-times to values as large as 20, both training and validation scores seem to be converge, showing learned trends.

Regardless, of the lead-time, all validation losses ecome unstable as late as epoch 30, showing first signs of overfitting. All models show some ability to learn fundamental relationships in the features. Even after the point of the best validation loss is reached even the longest lead time model #emph[Lead-20] shows the ability to overfit and predict targets that take place 20 days into the future. 

Most interesting in this case is the testing loss chart, which shows a staircase behaviour, when elongating the lead-time. This shows a very clear patter: the longer the lead-time, the more predictive accuracy degrades for unseen data. Despite its validation loss being exceptional ($0.3193$), #emph[Lead-20] far overfits the data as can be seen bby its test loss value ($0.8852$). 

#figure(
  image("../figs/lift_lead.png"),
  caption : [Lift curves and Gini-coefficients of comparable models]
) <fig:lead_lift>

The lift curve shows similar trends. With a Gini-coefficient of $0.635$ sets the baseline for all future predicting models. Unsurprisingly both #emph[Lead20] (Gini: $0.452$) and #emph[Lead10] (Gini: $0.440$) perform the worst. Surprisingly however, #emph[Lead5] (Gini : $0.512$) shows little to none prediction degradation compared to #emph[Lead0] (Gini : $0.508$), the baseline model, despite its extended target frame of reference. Predictions show to retain their predictive validity for up to a five-day period, however not for far greater lead-times.

Combining insights from both analyses shows, that a five-day lead-time is highly performant under predictive circumstances. Both a stable Gini-coefficient of the lift curve and relatively good validation to testing loss ratio can be seen for this model making it a viable option for fire management purposes. Furthermore, #emph[Lead-5] underlines the validity of the architecutre, by showing that it can spatially segment features. Long lead-times here stress test the network showing that inherent noise and missing data are issues that are especially extended into the future prediction domain. Nonetheless, the ability of the transformer to retain predictive performance without losing copious amounts of generalizability demonstrates that an applicable predictive lead-time is possible in this transformer architecture. 

== Experiment 4: Ablation Studies
Frequently, due to the complex nature of transformer architectures, not enough reasoning about the inner workings and performance is done (@fig:research_questions, *RQ4*). Hence, we here devote a section to ablation studies, more specifically to test the performance of the model under constrained circumstances. 

To offer a structural window into the behaviour of the model we here take a trained model and predict one year of data with one or multiple input features removed. For each, we track both the mean absolute error (MAE) and the mean squared error (MSE), the ladder penalizing especially large errors. All comparisons are made to the pre-trained baseline model #emph[SampleExt3] with a lead-time of zero , sequence extent of three and a target prediction of 14 days. Computationally, this is achieved by intervening in the forward pass of the prediction. By setting input data of single or combinations of models to zero, all attention blocks which rely on this information will yield zero as well. In essence, this constraint allows the model to behave as if certain information (e.g. terrain) is not available.

Other ablation approaches include retraining the model entirely under these constrained circumstances, however it is highly probable that the models will find new shortcuts to circumvent their limitations. This method would furthermore focus on investigating the datas redundancy rather than evaluating the focus of the architecture, the ladder being the goal of this investigation. By removing single data modules we investigate the models dependencies and structural vulnerabilities.

In total we here test 12 different scenarios, which include single or combinations of input data modules:


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
      [ *Ablation Configurations* ], [ *Number of Modules* ]
    ),
    [Aspect], [2],
    [Slope], [1],
    [Aspect + Slope], [3],
    [Roads Proximity], [1],
    [Burn History], [1],
    [Precipitation], [1],
    [NDVI], [1],
    [NDWI], [1],
    [Wind Direction], [2],
    [Wind Speed], [1],
    [Wind Direction + Speed], [3],
    [LST], [1],
  ),
  caption : "Ablation configurations investigated for single and combinations of data modules"
) <fig:ablation_configurations>

The global sensitivity analysis by data showing the ablation results for all 12 configurations is shown in @fig:ablation_seasonal. The most important confirmation here is one that as was expected from discussions in @chap:related_work, terrain is a main driver of predictional robustness. Removing the combined aspect and slope modules (MAE: $0.07080$, MSE: $0.01336$) removes all systematic overview of the model, making it the single most important metric of the model. Another notable result is that it by removing just one of the two terrain types (aspect or slope), the error is somewhat contained. Both modularities appear to be appear an error safeguard for the respective other, as described by their comparably small MSE (Aspect: $0.00102$, Slope: $0.00029$). The dominating feature under the dynamic features is LST (MAE: $0.05223$, MSE: $0.00945$). It acts as the anchor of thermal context, naturally scaling wildfire probabilities.

Roads (MSE: $0.03627$), burn history (MSE: $0.03258$) and wind components (MSE: $0.02941$) represent the second tier of feature imortance. They don't carry copious amounts weight, but in combination provide the model with critical boundary logic. While both burn history and road proximity are not dynamic features, but aid the preciction crucial secondary information regarding proximity to civilization.

NDVI (MSE: $0.00973$) is only about half as impactful as NDWI (MSE: $0.01987$). Given their similar derivations from satellite imagery, the two correlate quite closely, however plant moisture stress (NDWI) appears far more meaningful than proxies for photosynthetic capacity as NDVI.

#figure(
  image("../figs/ablation_mse_mae.png"),
  caption : [Model Sensitivity Analysis by Data Module]
) <fig:ablation_absolute>


To show not just the overall importance of features, this investigation additionally tracked feature importance over the period of an entire year, for equal periods of 14 days. In order to highlight seasonal impact of modules, here illustrated is a temporal sensitivity analysis (@fig:ablation_seasonal).

#figure(
  image("../figs/seasonal_implicit_trend.png"),
  caption : [Seasonal Ablation Profile: Component Reliance Across Time]
) <fig:ablation_seasonal>

The key notion shown in @fig:ablation_seasonal is that the induced error from removing modalities is variable by season. For most ablations, the model prediction error remains stable for the first nine indices, or about four months. This is the probable time for wildfire occurrence, hence none of the conditions in any module are fulfilled to cause significant risk. Quite clearly, since the model is predicting imbalanced binary classes, it requires less information to predict the more likely class as more information exists for it. For all ablations considered, MAE shifts rapidly as soon as wildfires are in season, at around indices 9-10. It is at this point, that the structural reasoning changes from relying on the broadness of all modules in winter, to predicting higher probabilities from a few extreme values in the summer months. Perhaps most notably, removing terrain features and LST causes significant errors of upto $16 percent$ in peak season. Furthermore, LST and terrain features show correlating error behaviour in their peaks from indices 14-16, stating that during this peak season, the features interact with one another.

Features contributng significantly less error especially in the seasonal perspective are NDVI and wind speed despite both of these being large datasets producing significant amounts of input tokens. Both stay almost perfectly level, causing little inaccuracies to the model when omitted. Rather than being heavily reliant on them, the model treats these features as a baseline offset and does not adjust them by season.

An important take-away from this ablation study, especially when evaluating RQ4, is that evaluating feature importance is entirely possible. By showing a distributed feature reliance, it is clear that the model does not simply use shortcut learning, depending on a single variable to produce as little less as possible, but uses the entire arsenal of its data types. @fig:ablation_absolute showed that features do have relative levels of impact on the prediction outcome, while none exceed an error of $7 %$. This shows that the cross-attention backbone of the architecture is succesfully using information from all modalities. Another important argument to be made, is that despite dealing with immense cloud-cover in the winter months, all dynamic wheather-dependent data types suppress this noise, producing no larger error than the static features. The model is however to some extent sensitive to imbalance of classes, which can be detected from rising error for all module ablations in fire-prone months. 
