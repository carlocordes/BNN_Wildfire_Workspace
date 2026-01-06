## Main Overview of Thesis Topic

### Guiding Research Questions
**Main RQ:** 
To what extent can (Bayesian) spatio-temporal transformers be used to predict sequence-to-sequence problems such as wildfire risk maps?

**Hypothesis:**

**Sub RQ:**
1. Can DL architectures be used to reliably quantify uncertainty in data and model?
2. How does the temporal scope of training data affect the temporal scope of trends learnt?
3. How does prediction uncertainty vary under extensive lead-time?
4. *(If Bayesian)* How can transformers be combined with uncertainty quantification such as bayesian statistics?

### Data

| Name | Description |Temporal Res. | Temporal Extent |Spatial Res. | d-type | Delivery Method |
| --- | --- | --- | --- | --- | --- | --- |
| MODIS | | | | | |
| FIRMS | | | | | |
| NVDI | | | | | |
| DTM | | | | | |

**Pipeline**: <br>

```mermaid
---
title: Network Architecture
---
flowchart LR
    subgraph Data
        Input[("Input Tensor
                (B x T x h x w)")]
        Static[("Temporal Data
                    (B x T x h x w)")]
        Dynamic[("Static Data
                    (h x w)")]
        Input --> Static
        Input --> Dynamic
    end

    subgraph Embedding
        Patch("$$E_{patch} = Flatten(Patch)  W + b$$")
                Dynamic & Static --> Patch

        subgraph Spatiotemporal Embedding
            STE("$$E_{patch} + E_{spatial} + E_{temporal}$$")
        end

        subgraph Spatial Embedding
            SE("$$E_{patch} + E_{spatial}$$")
        end

        Patch --> STE
        Patch --> SE
    end 

    subgraph Encoding
        subgraph Shallow Individual Spatial Self-Attention
            MHE1(Multi-Head Self-Attention)
            MLP1(Multi-Layer-Perceptron)
            ADDNORM1@{ shape: dbl-circ, label: "Add &
                                         Norm" }
            SE -- N x B x --> MHE1 --> ADDNORM1 --> MLP1 --> MHE1
        end

        subgraph Shallow Individual Spatiotemporal Self-Attention
            MHE2(Multi-Head Self-Attention)
            MLP2(Multi-Layer-Perceptron)
            ADDNORM2@{ shape: dbl-circ, label: "Add &
                                         Norm" }
            STE -- N x B x --> MHE2 --> ADDNORM2 --> MLP2 --> MHE2
        end
    end
```