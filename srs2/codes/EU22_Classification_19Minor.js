var geo1 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/EU28_Divided_1_6"),
    geo3 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/EU28_Divided_3_6"),
    geo5 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/EU28_Divided_5_6"),
    geo6 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/EU28_Divided_6_6"),
    geo4 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/EU28_Divided_4_1_6"),
    geo7 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/EU28_Divided_4_2_6"),
    geo2 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/EU27_Divided_2_6"),
    train_p1 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_1"),
    train_p2 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_2"),
    train_p3 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_3"),
    train_p4 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_4"),
    f_name1 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_important_features2");

//******************************************************************************************************/
//*****************     Authors: Babak Ghassemi, Emma Izquierdo-Verdiguier, Francesco Vuolo ************/
//*****************     Contact: babak.ghassemi@boku.ac.at                            ******************/
//******************************************************************************************************/
//******************************************************************************************************/
//*****************                       Institute of Geomatics                       *****************/
//*****************        University of Natural Resources and Life Sciences (BOKU)    *****************/
//*****************             Peter Jordan Strasse 82, 1190 Vienna, Austria          *****************/
//******************************************************************************************************/
//******************************************************************************************************/
// If you use this code, please cite this paper: B. Ghassemi et al., 
// "EU Crop Map 2022: Earth Observation's 10-Meter Dive into Europe's Crop Tapestry", in preparation.

// Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) 2023, 
// Babak Ghassemi, Emma Izquierdo-Verdiguier, Francesco Vuolo.
//******************************************************************************************************/
//******************************************************************************************************/
//******************************************************************************************************/
// Script to classify EU in 19 minor arable land classes. This script classifies the study area into 19 minor 
// arable land classes using a predefined training data and important features by 
// employing earth observation data and the Random Forest algorithm.
// Inputs:
//         - Study area: geometry of the study area. In this case, it was divided in seven different  
//                       geometries (geo1, geo2, ... , geo7) in order to predict the full study area.
//         - Training samples: the balanced training samples are uploaded in 4 separate FC (train_p1, 
//                             ... ,train_p4) to training multiple times.
//         - feat_sel: FC which contain the name of the selected features set as a property in the 
//                     features.
//Outputs: 
//         - EU27 classification map of 19 minor classes. It is exported to user assets folder.
//******************************************************************************************************/
//******************************************************************************************************/

// Load functions require:
var functions = require('users/BabakGhassemi9/Eucropmap22:functions');
var clip_lts = functions.clipic;
var feat_s1 = functions.s1_features;
var preproces_S2 = functions.preproS2;
var preproces_S1 = functions.preproS1;
var newNames_func = functions.newNames;
var S2month = functions.generateS2month;
var S1month = functions.generateS1month;
var vegetation_index = functions.vegetation_indices;

// Input constant variables:
var year = '2022';
var date_start = ee.Date(year + '-01-01');
var date_end= ee.Date(year + '-12-31');

var cloud_cover = 50;
var cloud_prob = 75;
var bands=['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

var geos = [geo1, geo2, geo3, geo4, geo5, geo6, geo7];

//////////////////////////////////////////////////////////////////////////////////////////////
//// Provide the selected features:
var feat_select=f_name1.aggregate_array('0').distinct();
//////////////////////////////////////////////////////////////////////////////////////////////////////

//********************************* TRAINING A RF ********************************//
// Train multiple times a Random Forest using provided training samples and  
// important features:
var model = ee.Classifier.smileRandomForest({
  numberOfTrees: 150,        
  variablesPerSplit: null, 
  minLeafPopulation: 2,    
  bagFraction: 0.5,        
  maxNodes: null,          
  seed: 0                  
  });

model = model.train(train_p1.filter(ee.Filter.eq('Lbl_cls_major', 1)),'Label_clas',feat_select);
model = model.train(train_p2.filter(ee.Filter.eq('Lbl_cls_major', 1)),'Label_clas',feat_select);
model = model.train(train_p3.filter(ee.Filter.eq('Lbl_cls_major', 1)),'Label_clas',feat_select);
model = model.train(train_p4.filter(ee.Filter.eq('Lbl_cls_major', 1)),'Label_clas',feat_select);

//****************************************************************************//

for(var g=0;g<7;g++){
  // Region to be classified [to cover the EU27 region, change one by one]:
  var geometry= geos[g].geometry();
  
  //********************************* LOAD DATA ********************************//
  
  //*************************** Load Sentinel-2 (S2) ***************************//
  var cloudprob= ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
  .filterDate(date_start, date_end)
  .filterBounds(geometry);
  
  var dataSE=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(date_start, date_end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',cloud_cover))
  .filterBounds(geometry);
  
  // Join Sentinel 2 and cloud probablity IC:
  var idJoin = ee.Filter.equals({
    leftField: 'system:index', 
    rightField: 'system:index'
  });
  var innerJoin = ee.Join.saveFirst('Cloud', 'system:time_start', true, null);
  var dataSE = ee.ImageCollection(innerJoin.apply(dataSE, cloudprob, idJoin));
  
  //*************************** Load Sentinel-1 (S2) ***************************//
  var s1=ee.ImageCollection("COPERNICUS/S1_GRD");
  
  //****************************** Load LTS (MODIS) ****************************//
  
  var lst = ee.ImageCollection('MODIS/061/MOD21C3')
  .filter(ee.Filter.date('2022-01-01', '2022-12-31')).select('LST_Day');
  lst = clip_lts(lst, geometry).toBands();
  
  //********************************** Load DEM ********************************//
  var dem=ee.Image("projects/sat-io/open-datasets/ASTER/GDEM")
  .clip(geometry.bounds());
  //****************************************************************************//
  
  //********************** Sentinel-2 (S2) pre-processing **********************//
  // Images with a cloud fraction cover lower than 50% were selected and masked 
  // considering the cloud probability (higher 75%)and Scene Classification (SCL) 
  // to avoid defective, all kinds of clouds and snow/ice.
  dataSE = preproces_S2(dataSE,cloud_prob);
  dataSE = dataSE.select(bands);
  
  // Calculate Spectral Indices of S2 data:
  var Collection_S2_filtered = vegetation_index(dataSE);
  
  //Compute monthly features from S2:
  // Median for the first three month of the year, after that the median is calculated per month 
  var months = ee.List.sequence(3,10);
  var S2_Monthly = S2month(months, Collection_S2_filtered, year);
  
  //Compute yearly features from S2:
  var S2_Yearly = ee.Image(Collection_S2_filtered.reduce(ee.Reducer.percentile([5,50,98])));
  // Rename the bands:
  var NewBandNames = newNames_func(S2_Yearly, year);
  S2_Yearly = S2_Yearly.rename(NewBandNames);
  //****************************************************************************//
  
  //********************** Sentinel-1 (S1) pre-processing **********************//
  // S1 image selections were made in IW mode and with vertical transmit/vertical 
  // receive (VV) polarization. Then, VV and vertical transmit/horizontal receive 
  // (VH) bands, and using the focal median method, a speckle filter with a 
  // circular kernel with 30 m of radius was applied to reduce noise in the images.
  var radar = s1
  .filterBounds(geometry)
  .filterDate('2022-01-01','2023-01-01')
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .select('VV','VH');
  radar =preproces_S1(radar, geometry);
  
  // Extract features of S1 data:
  radar = feat_s1(radar);
  
  //Compute monthly features from S1:
  var S1_Monthly = S1month(radar,'2022-01-01', 12, 1, 'month');
  
  //Compute yearly features from S1:
  var S1_Yearly = radar.reduce(ee.Reducer.percentile([5, 50, 98]));
  
  //// Stack all the input data in one image:
  var data= S2_Monthly.toBands()
  .addBands(S2_Yearly)
  .addBands(S1_Monthly.toBands())
  .addBands(S1_Yearly)
  .addBands(lst)
  .addBands(dem);
  
  //// Take the selected bands and clip the image:
  var final_data = data.select(feat_select);
  var final_data = final_data.clip(geometry);
  
  // Predict the classification map for a specific region:
  var class_map = final_data.classify(model);
  //print(class_map);
  
  //// Export the classification map to assets:
  Export.image.toAsset({
    image: class_map,
    description: 'classificaon_map_19classes_geo'+(g+1).toString(),
    assetId: 'EU_19_minor_geo'+(g+1).toString(),
    scale:10,
    region: geometry,
    maxPixels: 10e12
  });
}