#import "../template.typ": *

= Data <chap:data>



== Data Acquisition
As shown in the introductory sections, transformer models thrive on abundance of data. In order to construct a meaningful model, a large harmonized dataset is necessary. The collection of datapoints used to gather data along with information on temporal and spatial extent and resolution is shown in @fig:ee_endpoints:

//Table of datatype, source, resolution, coverage,
#figure(
  table(
    columns: (auto, 4cm, 2cm, 2.4cm, 2cm, auto),
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black)
    },
    align: (x, y) => (
      if x > 0 { left }
      else { left }
    ),
    table.header(
      [ *Endpoint Name* ], [ *Description* ], [ *Temporal Res.* ], [ *Temporal Extent* ], [ *Spatial Res.* ], [ *Spatial Extent* ]
    ),
    [MODIS/061/MCD64A1], [Global monthly gridded burned area and quality], [Monthly], [2000-01-01], [500 m], [Global ($90 degree upright(N)$ to $90 degree upright(S)$)],

    [CGIAR/SRTM90_V4], [DTM  data processed by CGIAR-CSI to fill voids.], [Static], [2000-01-01], [90 m], [Global land masses ($60 degree upright(N)$ to $56 degree upright(S)$)],

    [ERA5_LAND/HOURLY], [High-resolution hourly atmospheric and land-surface reanalysis dataset.], [Hourly], [1950-01-01], [\~9 km], [Global ($90 degree upright(N)$ to $90 degree upright(S)$)],

    [MODIS/061/MOD11A2], [Terra MODIS 8-day composite Land Surface Temperature and Emissivity (LST)], [8 Days], [2000-01-01], [1,000 m], [Global ($90 degree upright(N)$ to $90 degree upright(S)$)],

    [MODIS/061/MOD09GQ], [Terra MODIS Daily Level-2G Surface Reflectance  (NDVI)], [Daily], [2000-01-01], [250 m], [Global ($90 degree upright(N)$ to $90 degree upright(S)$)],

    [MODIS/061/MOD09GA], [Terra MODIS Daily Level-2G Surface Reflectance (NDWI)], [Daily], [2000-01-01], [500 m], [Global ($90 degree upright(N)$ to $90 degree upright(S)$)],

    [UCSB-CHG/CHIRPS/DAILY], [CHIRPS daily rainfall estimates.], [Daily], [1981-01-01], [\~5,500 m], [Quasi-global ($50 degree upright(N)$ to $50 degree upright(S)$)],
  ),
  caption: "Overview of data sources"
) <fig:ee_endpoints>

The data used herein stems from a variety of different sensors and data products using varying calibrations and accuracy. Also, since all datasets describe different physical phenomena both the temporal and spatial resolutions must be considered when harmonizing this dataset. The data gathered in this investigation stems entirely from Google Earth-Engine (EE) and was retrieved using API requests.


=== System configuration
// Configure area of interest, crs, projection, grid -> ensures uniformity
To govern data harmonization, the core defining parameters of the area of interested are globally defined in the system configurations and is passed to each of the downloading and processing steps throughout the process. The interpretation of the area of interest, time, projection, scale and coordinate reference system (CRS) is handled internally by the #emph[GoldenGrid] class. It inherits start- and end-date from the system configurations and contains a variable list of timestamps for which to retrieve data. Furthermore, the coordinate settings in the inputs are interpreted here as a bounding box structure which earth engine can understand, while also passing along the system CRS and scale. 

// UML of golden grid
#todo(stroke : orange)[Add UML figure of GG]


=== Earth Engine
// Endpoints, retrieval

The instance of golden-grid is handed to each of the classes responsible for retrieving data from EE. Specifications defined in the golden-grid are directly handed over as query attributes in the request. All cropping operations and projections are done server-side. Resulting rasters are stored for each day in a tagged image file format (TIFF). A script iterates over all endpoints, the product of which is a daily for every module and every day in the time of interest. 


== Feature Engineering
In the following steps we take a few important steps from raw data, towards a dataset that a transformer can understand and reliably extract information to learn from. There are a few significant pitfalls when preparing data for a model relying on simple correlation, which will be addressed here.

A choice made to promote easier organisation and fast access of data is to store it in the commonly used Zarr-format. This is a data type can store 3D raster data, making it an ideal fit for time dependent phenomena. Furthermore, it is possible to consolidate data of different modalities into this structure as shown in @fig:zarr. Zarr data furthermore has the ability to use a method called chunking, which allows for random access and slicing of all three dimensions, here two spatial and one temporal dimension and accross multiple datasets. More specifically, this allows for rapid data selection and loading from disk as opposed to loading georeferenced raster files. 

#figure(
  image("../figs/zarr.svg", width : 70%),
  caption: [Organization of multi-modular temporal rasters into zarr format]
) <fig:zarr>


=== Ground Truth
// Raw yearly day-of-year dataset to binary burn zarr
Most important for a valid prediction mechanism is a valuable ground truth dataset. Chosen for this work is the monthly gridded burned area from MODIS. Despite being delivered on a monthly basis, the dataset contains yearly images which specify the Julian day of a burn event per pixel. The data is processed from a yearly perspective into daily rasters. Essentially, this process creates binary burn maps for every calendar day in the input specifications.

The target extent, a system paramter passed by the config file, defines the temporal extent for which a prediction should be made (e.g. 14 days). We use the previously created zarr containing daily burn records as the basis for this feature. In essence, we use a continuous sliding window that computes the union of 14 consecutive days of burn records and stores it as a new module in the zarr container. These images will later represent the feature which the model predictions are compared to and updated accordingly. As some pixels may never have burnt, we must opt for a maximum value, hence introducing a small bias. Here it was set at 7300 days or roughly 20 years.

=== Burn History
// Retrieval from binary burn zarr
Another feature which is derived from historical burn records is the burn history map. The idea here is to encode some information into the system, which describes the frequency and temporal behaviour of fire events for every location. As for all other features, we produce a burn history map for every day. The methodology here once again involves reading the binary burn history zarr file and extracting all burn records previous to the desired date. As for one pixel, multiple fires might have occured, we compute the amount of days since the last seen event for each pixel. This historical map is computed for all days availlable in the catalogue and is stored back to a new zarr modality.

By using this kind of information the model has the ability to recognize patterns areas which are highly susceptible to fires. Additionally, we here encode an important idea about burned area, which is that once it has been exposed to a fire, it is highly unlikely to burn again in near time, or might even pose as a natural fire barrier. 

#todo(stroke : orange)[Figure of zarr slicing and union]

=== Road Proximity
As discussed in @chap:related_work, the main driver of wildfire ignition is human activity. This information is implicitly encoded via proximity to roads. The basis of this serves a vector dataset that encodeds roads for the entirety of Europe, which was synthesized as part of a study to describe patterns in road infrastructure @roads_dataset. 

The process here includes selecting roads in the area of interest and creating a small buffer around them, essentially converting from #emph[LineString] to #emph[Polygon]. These are then converted in a binary road map raster, which has exact dimensions as all other inputs, as dictated by the golden-grid. Ones indicate roads, zeros indicate any other land use. On the basis of this we conduct a grid search for every pixel in the image. Here we compute the physical proximity from the center of each pixel to the nearest road pixel. 

#todo(stroke : orange)[Figure of 3 step process here?]

=== Ratio-Scale data types
// Split Interval scales into ratio scales (which have absolute zero)
Processing the digital terrain model (DTM) as a raw feature is possible, and is valid, however we enrich enrich terrain information passed to the model by additionally including derivatives of it. After all it is less the total altitude a model is interested in when predicting fire, but more the contextual appearance. Therefore, the terrain is processed into slope and aspect maps. 

Slope is simple to compute as for every pixel we simply conduct a grid search of the immediate pixel neighbors and compute the joint maximum slope for each position. This feature is crucial to fire modeling, as fire propagates well in highly slanted terrain. Adding a representation of aspect, which is valuable to the model, is more complicated. Aspect is denoted as an angle, which is an interval scale as per @eq:angle:

$ phi in [0, 360) $ <eq:angle>

A transformer whose mechanisms rely on correlation and scaled dot-products is inherently flawed in determining correlations of such interval scales. As an example an aspect of $355 degree$ is quite similar to $0 degree$, however given on a ratio scale, these are far apart, these would yield a low correlation. To circumvent this effect, the aspect dataset is split up into two components, one describing the east-west component and north-south for the other. Both run on a ratio scale, together describing the angle of aspect (@fig:decomposing_aspect). The two aspect and slope datasets complete the suite of terrain information. It is up to the transformer to find spatial correlation between these in order to gain an understanding. 

#figure(
  image("../figs/aspect_decomposition.png", width : 80%),
  caption: "Decomposition of Ratio scale datasets into components"
) <fig:decomposing_aspect>

The same process is carried out for the wind datasets. Daily data is delivered via a cyclical wind-direction, and wind speed map, which both together denote the wind vector. As for the terrain we dissect wind direction into east-west and north-south component and deliver them to the transformer as separate datasets and allow it to cross-correlate the information back together.


== Processing
// Normalization
With all information processed into value scales a transformer can interpret, we must additionally normalize datasets. The choice here was made to normalize all values to the $[0, 1]$ domain. This was done by taking the entirety of 10 years of data available for each data module and recording the global maximum and minimum values for each. Next, every image in the respective module was normalized by this metric. Ruthlessly merging metric scales into a normalized domain might seem counter-intuitive at first, but is justifiable and follows very concrete reasoning. Primarily, the SI-derived units in which the datasets are measured in are mostly arbitrary. The transformer model cannot distinguish the extent to which precipitation, measured in mm, is relevant to land surface temperature ($degree C$). Furthermore, combining datasets which include values in different orders of magnitudes will lead to processing instability in the transformer. When cross-attending elevation, which in this caries varies between [0, 2500], and land surface temperature, the weights that correlate these values to one another will have to scale up and down drastically to achieve stable cross attention. This also shows, that were values not to be normalized, the transformer de facto will do it nevertheless.

// Masking and nodata value
We additionally must treat the datasets missing data points. Many of the datasets are influenced by cloudcover. Those relying on near-infrared data like NDVI, which does not penetrate cloud cover long streches of missing data can be observed especially in the winter months. Regardless of its relevance to a wildfire prediction, we additionally mask out waterbodies on all images. Not all modularities produce values over oceans and rivers, making it a safer option to remove them entirely. A joint no-data value both for missing data and water-cover of $-1$ has been chosen as a unified value. Through its learning process, the transformer will learn to give especially those pixels over water less weight, effectively removing them from the data scope.

@fig:all_input_data shows the final resulting input dataset for one day as it would appear to the transformer.


#figure(
  image("../figs/all_samples.png"),
  caption: "Snapshot of input data types"
) <fig:all_input_data>
