// ===============================
// CROP MAP SRS - FINAL SCRIPT
// ===============================

// 1. Загружаем обучающие точки
var rawPoints = ee.FeatureCollection('projects/crop-map-srs/assets/group3');

// 2. Добавляем числовой класс
var trainingPoints = rawPoints.map(function(f) {
  var crop = ee.String(f.get('class'));

  var classNum = ee.Number(
    ee.Algorithms.If(crop.compareTo('лен').eq(0), 0, 1)
  );

  return f.set('class_num', classNum);
});

print('Training points', trainingPoints.limit(5));
Map.addLayer(trainingPoints, {color: 'red'}, 'Training points');

// 3. Область интереса вокруг точек
var roi = trainingPoints.geometry().bounds().buffer(2000);
Map.centerObject(trainingPoints, 11);
Map.addLayer(roi, {color: 'blue'}, 'ROI');

// 4. Sentinel-2
var dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2023-05-01', '2023-08-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

// 5. Медианное изображение
var image = dataset.median().clip(roi);

// 6. RGB
Map.addLayer(
  image,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
  'Sentinel-2 RGB'
);

// 7. NDVI
var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
Map.addLayer(
  ndvi,
  {min: 0, max: 1, palette: ['white', 'yellow', 'green']},
  'NDVI'
);

// 8. EVI
var evi = image.expression(
  '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
  {
    'NIR': image.select('B8'),
    'RED': image.select('B4'),
    'BLUE': image.select('B2')
  }
).rename('EVI');

Map.addLayer(
  evi,
  {min: 0, max: 1, palette: ['white', 'lightgreen', 'darkgreen']},
  'EVI'
);

// 9. NDWI
var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
Map.addLayer(
  ndwi,
  {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']},
  'NDWI'
);

// 10. Feature stack
var features = image
  .select(['B2', 'B3', 'B4', 'B8'])
  .addBands(ndvi)
  .addBands(evi)
  .addBands(ndwi);

print('Feature stack', features);

// 11. Получаем обучающую выборку
var training = features.sampleRegions({
  collection: trainingPoints,
  properties: ['class_num', 'class'],
  scale: 10
});

print('Training sample', training.limit(10));
print('Training sample size', training.size());

// 12. Обучаем Random Forest
var classifier = ee.Classifier.smileRandomForest(30).train({
  features: training,
  classProperty: 'class_num',
  inputProperties: features.bandNames()
});

// 13. Классифицируем изображение
var classified = features.classify(classifier);

// 14. Показываем результат
// yellow = лен
// green  = пшеница
Map.addLayer(
  classified,
  {min: 0, max: 1, palette: ['yellow', 'green']},
  'Crop classification'
);

// =======================
// Оценка точности модели
// =======================

// Делим данные на train / test
var withRandom = training.randomColumn('random');

var trainSet = withRandom.filter(ee.Filter.lt('random', 0.7));
var testSet = withRandom.filter(ee.Filter.gte('random', 0.7));

// Обучаем модель
var classifier = ee.Classifier.smileRandomForest(30).train({
  features: trainSet,
  classProperty: 'class_num',
  inputProperties: features.bandNames()
});

// Проверяем на тестовых данных
var validated = testSet.classify(classifier);

// Матрица ошибок
var confusionMatrix = validated.errorMatrix('class_num', 'classification');

print('Confusion Matrix', confusionMatrix);
print('Overall Accuracy', confusionMatrix.accuracy());

// =======================
// EXPORT TO GEO TIFF
// =======================
Export.image.toDrive({
  image: classified,
  description: 'crop_classification_group3',
  folder: 'GEE_exports',
  fileNamePrefix: 'crop_classification_group3',
  region: roi,
  scale: 10,
  maxPixels: 1e13
});

// =======================
// VISUAL IMAGE FOR PNG-LIKE EXPORT
// =======================
var classifiedVis = classified.visualize({
  min: 0,
  max: 1,
  palette: ['yellow', 'green']
});

Export.image.toDrive({
  image: classifiedVis,
  description: 'crop_classification_png_style',
  folder: 'GEE_exports',
  fileNamePrefix: 'crop_classification_png_style',
  region: roi,
  scale: 10,
  maxPixels: 1e13
});