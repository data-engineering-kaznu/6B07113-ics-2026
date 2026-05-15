var train_p1 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_1"),
    train_p2 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_2"),
    train_p3 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_3"),
    train_p4 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_4"),
    f_name1  = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_important_features2");

var aoi = ee.FeatureCollection('projects/srs1-488806/assets/test1_srs2');

var functions        = require('users/BabakGhassemi9/Eucropmap22:functions');
var feat_s1          = functions.s1_features;
var preproces_S2     = functions.preproS2;
var preproces_S1     = functions.preproS1;
var newNames_func    = functions.newNames;
var S2month          = functions.generateS2month;
var S1month          = functions.generateS1month;
var vegetation_index = functions.vegetation_indices;

var year     = '2025';
var bands    = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
var geometry = aoi.geometry();

Map.centerObject(aoi, 13);
Map.addLayer(aoi, {color: 'red'}, 'AOI');

// ─── Sentinel-2 ───
var cloudprob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
  .filterDate(year+'-01-01', year+'-12-31').filterBounds(geometry);

var dataSE = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(year+'-01-01', year+'-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
  .filterBounds(geometry);

var idJoin    = ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'});
var innerJoin = ee.Join.saveFirst('Cloud', 'system:time_start', true, null);
dataSE = ee.ImageCollection(innerJoin.apply(dataSE, cloudprob, idJoin));

dataSE = preproces_S2(dataSE, 75).select(bands);
var S2_col = vegetation_index(dataSE);

var months     = ee.List.sequence(3, 10);
var S2_Monthly = S2month(months, S2_col, year);
var S2_Yearly  = ee.Image(S2_col.reduce(ee.Reducer.percentile([5, 50, 98])));
S2_Yearly = S2_Yearly.rename(newNames_func(S2_Yearly, year));

// ─── Sentinel-1 ───
var radar = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(geometry)
  .filterDate(year+'-01-01', year+'-12-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select('VV', 'VH');
radar = preproces_S1(radar, geometry);
radar = feat_s1(radar);
var S1_Monthly = S1month(radar, year+'-01-01', 12, 1, 'month');
var S1_Yearly  = radar.reduce(ee.Reducer.percentile([5, 50, 98]));

// ─── DEM ───
var dem = ee.Image("projects/sat-io/open-datasets/ASTER/GDEM").clip(geometry.bounds());

// ─── Стек данных ───
var data = S2_Monthly.toBands()
  .addBands(S2_Yearly)
  .addBands(S1_Monthly.toBands())
  .addBands(S1_Yearly)
  .addBands(dem);

// ─── Фильтр признаков ───
var feat_select = f_name1.aggregate_array('0').distinct();
var available_bands = data.clip(geometry).bandNames();
feat_select = feat_select.filter(ee.Filter.inList('item', available_bands));

var final_data = data.select(feat_select).unmask(0).clip(geometry);

// ════════════════════════════════════════════
// 7 major классов
// ════════════════════════════════════════════
var model_major = ee.Classifier.smileRandomForest({
  numberOfTrees: 130, minLeafPopulation: 2, bagFraction: 0.5, seed: 0
});
model_major = model_major.train(train_p1, 'Lbl_cls_major', feat_select);
model_major = model_major.train(train_p2, 'Lbl_cls_major', feat_select);
model_major = model_major.train(train_p3, 'Lbl_cls_major', feat_select);
model_major = model_major.train(train_p4, 'Lbl_cls_major', feat_select);

var class_map_major = final_data.classify(model_major);

// Визуализация
var palette_major = ['e60000','ffffa8','006400','98ff00','d3d3d3','0064c8','6096ff'];
Map.addLayer(class_map_major, {min:0, max:6, palette: palette_major}, 'Землепользование 7 классов 2025');

// Легенда в консоли
print('Классы: 0=Искусственные, 1=Пашня, 2=Леса/кустарники, 3=Луга, 4=Голые земли, 5=Вода, 6=Болота');

// Экспорт CSV
Export.table.toDrive({
  collection:  class_map_major.sample({region: geometry, scale: 10, geometries: true}),
  description: 'crop_7major_2025',
  fileFormat:  'CSV'
});
