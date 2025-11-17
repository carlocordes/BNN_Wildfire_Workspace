
Full Title: **[A Bayesian Deep Learning Approach to Near_term Climate Prediction](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022MS003058)**

First Author: Xihaier Luo

---
### Short Summary
An application of a network interfered with [[bayesian-statistics]] is made to predict the variations in sea surface temperatures. 800 years of partially simulated climate data is used to predict outputs at varying [[lead-time]]. 

A comparison is made between traditional [[deep-learning]](BDL), [[convLSTM]] and the bayesian variant. Results show the BDL architecture producing more accurate crisp results. 

Finally, an evaluation is made to what extent the [[uncertainty]] of the output is valuable to describing the [[prediction-error]] of predictions in the future.

--- 
### Notes

**Problem Setup** <br>
Predictions are made about the natural valiability in sea surface temperature in the North Atlantic sea both spatially and temporally. A catalogue of 800 years of data has been simulated to aid training and testing data. The data is preprocessed to reduce the mean (as variation is to be predicted) and the seasonal variation. 

**Background of BDL / BNN Setup** <br>
The bayesian approach is a probabalistic one, as it assumes all weights as random variables, which have a [[probability-density-function]] with $\bar x = 0$. Given a training dataset $D$, a [[posterior-distribution]] of the network parameters $\vec w $ can be formulated [[bayes-rule]]:
$$P(\vec w \mid D) = \dfrac{P(D \mid \vec w)\,P(\vec w)}{P(D)}$$

The formulation of a probabalistic neural network is thus formulated by further considering a term of random noise $ \vec n $ which describes the irreducible [[aleatoric-uncertainty]] in the data:
$$ \vec y = f(\vec x, \vec w) + \vec n $$

**Architectural design** <br>
The network as opposed to dimensionality reduction techniques aims to use the entire dimensionality of the data. Next [[convolution]] layers are the centerpiece of the architecture. To permit the learning of multiscale interactions, a [[bottleneck]] is employed. 

Down- and up-sampling processes are used to reduce the number of network parameters, which accelerates the training process. 

**Results** <br>
Bayesian architectures successfully capture the key features of the expected target. Comparing to [[convLSTM]], predictions are more correllated and sharper for a random sample. 

This result is confirmed, when averaging the test data over 128 test samples. Error increases linearly with [[lead-time]]. The error for these samples is usually lower for BDL as compared to [[convLSTM]].

Both DL and BDL models are better scalable and require less training time than convLSTM.

Paper makes a argues the extent to which [[prediction error]] can be assessed for future predictions by the [[uncertainty]].

---

### Take-aways
- Problem setup requires lots of labeled data to be produced, a risk index needs to be derived and calculated.
- Network architecture is very specific to the problem
- Bayesian Deep learning succesfully deployed for spatio-temporal prediction problems
- Lead time is a potential sub-research question
    - Not to be confused with prediction time, e.g. monthly, seasonal, decadal
- Pre-processing data is key as learning setup might catch on to more obvious trends
- More research into bayesian statistics needed
    - Specifically prior/ posterior distributions for BNN applications