var functions = require('users/BabakGhassemi9/Eucropmap22:functions');
var preproces_S2     = functions.preproS2;
var preproces_S1     = functions.preproS1;
var newNames_func    = functions.newNames;
var S2month          = functions.generateS2month;
var S1month          = functions.generateS1month;
var vegetation_index = functions.vegetation_indices;
var feat_s1          = functions.s1_features;

var train_p1 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_1");
var train_p2 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_2");
var train_p3 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_3");
var train_p4 = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_CentroPolyPointsFull_training75_part_4");
var f_name1  = ee.FeatureCollection("projects/ee-babakghassemi9/assets/LU22_S1S2MYA_important_features2");

var aoi      = ee.FeatureCollection('projects/srs1-488806/assets/test1_srs2');
var geometry = aoi.geometry();

var year  = '2025';
var bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

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
var Collection_S2_filtered = vegetation_index(dataSE);
var months     = ee.List.sequence(3, 10);
var S2_Monthly = S2month(months, Collection_S2_filtered, year);
var S2_Yearly  = ee.Image(Collection_S2_filtered.reduce(ee.Reducer.percentile([5, 50, 98])));
S2_Yearly      = S2_Yearly.rename(newNames_func(S2_Yearly, year));

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
var data = S2_Monthly.toBands().addBands(S2_Yearly)
  .addBands(S1_Monthly.toBands()).addBands(S1_Yearly).addBands(dem);

// Переименовываем признаки модели с '2022' на '2025',
// чтобы они совпадали с нашими band names
var feat_select_all = f_name1.aggregate_array('0').distinct();
var feat_select_2025 = feat_select_all.map(function(name) {
  return ee.String(name).replace('2022', '2025', 'g');
});
var available_bands = data.clip(geometry).bandNames();
var feat_select     = feat_select_2025.filter(ee.Filter.inList('item', available_bands));

print('Количество признаков:', feat_select.size());

var final_data = data.select(feat_select).unmask(0).clip(geometry);

// ─── Модель: 19 культур (только пашня) ───
var model = ee.Classifier.smileRandomForest({
  numberOfTrees: 150, minLeafPopulation: 2, bagFraction: 0.5, seed: 0
});
model = model.train(train_p1.filter(ee.Filter.eq('Lbl_cls_major', 1)), 'Label_clas', feat_select);
model = model.train(train_p2.filter(ee.Filter.eq('Lbl_cls_major', 1)), 'Label_clas', feat_select);
model = model.train(train_p3.filter(ee.Filter.eq('Lbl_cls_major', 1)), 'Label_clas', feat_select);
model = model.train(train_p4.filter(ee.Filter.eq('Lbl_cls_major', 1)), 'Label_clas', feat_select);

var class_map = final_data.classify(model);

Map.centerObject(geometry, 14);

// ─── Экспорт ───
Export.table.toDrive({
  collection:  class_map.sample({region: geometry, scale: 10, geometries: true}),
  description: 'crop_19classes_2025',
  fileFormat:  'CSV'
});

print('Готово! Запусти задачу во вкладке Tasks.');
