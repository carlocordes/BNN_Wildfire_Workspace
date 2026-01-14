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

| Name | Description |Temporal Res. | Temporal Extent |Spatial Res. | Spatial Extent | d-type | Delivery Method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [NASA FIRMS MODIS](https://firms.modaps.eosdis.nasa.gov/country/)| Active fire / hotspot information available as Points(lon, lat, t), brightness, confidence  | continuous | 2001 - 2024| 1000m | worldwide | csv | Download |
| [Land Cover](https://www.sciencebase.gov/catalog/item/6345b637d34e342aee0863aa) | Type of Land Cover, Tree Canopy Cover| - | - | 30m | CONUS | GeoTIFF | Download |
| [DEM, ASPECT](https://apps.nationalmap.gov/downloader/) | Digital Elevation Model of US| - | - | 1m | CONUS | GeoTIFF, shp, GDB, GPKG |Download|
| ERA5 Re-analysis| Meteorological Products: Temperature, wind speed, wind direction, humidity, precipitation| daily | 1959 - 2025 | 31km | worldwide | NetCDF | Data Store / API |
| GRIDMET | Surface meteorology, precipitation, humidity | daily | 2001 - 2025| 4 km | CONUS | NetCDF | HTTPS / API |

### Network Architecture

```mermaid 
flowchart TD
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

    subgraph Iterative Fusion
        subgraph Cross-Attention
        MHE3(Cross-Attention)
        MLP3(Multi-Layer-Perceptron)
        ADDNORM3@{ shape: dbl-circ, label: "Add &
                                        Norm" }
        
        end

        Static1(Static Embedding)
        MLP2 --> MHE3
        MHE3--> ADDNORM3 --> Static1 --> MLP3 --> MHE3
        MLP1 --> Static1
    end

    subgraph Joint-Deep-Self-Attention
        MHE4(Multi-Head Self-Attention)
        MLP4(Multi-Layer-Perceptron)
        ADDNORM4@{ shape: dbl-circ, label: "Add &
                                        Norm" }
        MLP3 -- N x B x--> MHE4
        MHE4 --> ADDNORM4 --> MLP4 --> MHE4
    end

    subgraph Convolution / Upsampling
        ST1(Stage 1
            D'' x n'' x n'')
        ST2(Stage 2
            D' x n' x n')
        ST3(Stage 3
            1 x n x n)
        MLP4 --> ST1
        ST1 --> ST2 --> ST3
    end

    subgraph Prediction Head
        Output(Probabalistic Output Layer
                0-1)

        ST3 --> Output
    end

```