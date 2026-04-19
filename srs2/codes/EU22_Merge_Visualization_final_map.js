var image1 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_7Maj_Part1_6"),
    image2 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_7Maj_Part2_6"),
    image3 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_7Maj_Part3_6"),
    image4 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_7Maj_Part4_1_6"),
    image5 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_7Maj_Part4_2_6"),
    image6 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_7Maj_Part5_6"),
    image7 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_7Maj_Part6_6"),
    image8 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_19Cropland_Part1_6"),
    image9 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_19Cropland_Part2_6"),
    image10 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_19Cropland_Part3_6"),
    image11 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_19Cropland_Part4_1_6"),
    image12 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_19Cropland_Part4_2_6"),
    image13 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_19Cropland_Part5_6"),
    image14 = ee.Image("users/BabakGhassemi9/LU22_ClassifiedToAssetEU27_S1S2MYA_training75_CentroPoly_19Cropland_Part6_6"),
    EU27 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/EU27_EPSG3035");

//******************************************************************************************************/
//*****************     Authors: Babak Ghassemi, Emma Izquierdo-Verdiguier, Francesco Vuolo ************/
//*****************     Contact: babak.Ghassemi@boku.ac.at                            ******************/
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
// Script to merge major and minor classification of EU. This selects the arable land in the major map 
// to select the crop class from the minor map. A post-processing step is applied to reduce the noise 
// of the final map.
// Inputs:
//         - major classification: load the seven regions in the assets obtained by 
//                                 "EU22_Classification_7Major" script.
//         - minor classification: load the seven regions in the assets obtained by 
//                                 "EU22_Classification_19Minor" script.
//         - EU27: FC cointain the region of the 27 countries classified.
//Outputs: 
//         - final_map: post.
//******************************************************************************************************/
//******************************************************************************************************/

// Functions require:
var functions = require('users/BabakGhassemi9/Eucropmap22:functions');
var Add_legend = functions.legend;
var postprocessing = functions.postMap;
var Visualization = functions.Visualization;
var remap_classes = functions.rename_classes;

// Load the major (7 classes) and minor (19 classes) maps:
var major = ee.ImageCollection([image1,image2, image3, image4, image5, image6, image7]).mosaic();
var minor = ee.ImageCollection([image8,image9, image10, image11, image12, image13, image14]).mosaic();

// Change the classification values of major image:
var major = major.remap(ee.List([0, 1, 2, 4, 5, 6, 7]), ee.List([0, 200, 20, 21, 22, 23, 24]));

// Merge both classifiaction maps:
var merge_map = major.where(major.eq(200), minor);


// Remap the final map classes to be consistented to EUCROPMAP 2018:
var merge_map = remap_classes(merge_map);

// Postprocessing step of the final map:
var final_map =  postprocessing(merge_map);

// Display the merge map
Map.addLayer(Visualization(merge_map), {}, 'Classified_S2_final') ;

// Display the final map
Map.addLayer(Visualization(final_map), {}, 'Classified_S2_final_filtered');


//// Add legend:
Add_legend();

/// Export the final map to Assets
Export.image.toAsset({
  image: final_map,
  description: 'EU27_Final_Map_EPSG3035',
  assetId: 'EU27_Final_Map_EPSG3035',
  scale:10,
  crs:'EPSG:3035',
  region: EU27,
  maxPixels: 10e12
});
