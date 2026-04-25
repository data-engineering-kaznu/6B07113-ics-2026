// Pixel-based land cover workflow for a user AOI uploaded from test1.zip.
// Output 1: colored raster for Google Earth Pro.
// Output 2: CSV with one row per 10 m pixel inside the AOI.
// Output 3: CSV summary by administrative region based on those pixels.

// ---------------------------
// User inputs
// ---------------------------

// Upload test1.zip to Earth Engine Assets and replace this path.
var userAoi = ee.FeatureCollection('users/your_username/test1');

// Google Drive folder for export tasks.
var driveFolder = 'GoogleEarthEngine';

// Administrative regions dataset.
var regions = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level1');

// Land cover raster. Resolution is 10 m.
var landcover = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .select('Map');

// ---------------------------
// Class dictionaries
// ---------------------------

var classNames = ee.Dictionary({
  '10': 'Tree cover',
  '20': 'Shrubland',
  '30': 'Grassland',
  '40': 'Cropland',
  '50': 'Built-up',
  '60': 'Bare or sparse vegetation',
  '70': 'Snow and ice',
  '80': 'Permanent water bodies',
  '90': 'Herbaceous wetland',
  '95': 'Mangroves',
  '100': 'Moss and lichen'
});

var classColors = ee.Dictionary({
  '10': '006400',
  '20': 'ffbb22',
  '30': 'ffff4c',
  '40': 'f096ff',
  '50': 'fa0000',
  '60': 'b4b4b4',
  '70': 'f0f0f0',
  '80': '0064c8',
  '90': '0096a0',
  '95': '00cf75',
  '100': 'fae6a0'
});

var palette = [
  '006400', 'ffbb22', 'ffff4c', 'f096ff', 'fa0000',
  'b4b4b4', 'f0f0f0', '0064c8', '0096a0', '00cf75', 'fae6a0'
];

var legendItems = [
  {code: 10, name: 'Tree cover', color: '006400'},
  {code: 20, name: 'Shrubland', color: 'ffbb22'},
  {code: 30, name: 'Grassland', color: 'ffff4c'},
  {code: 40, name: 'Cropland', color: 'f096ff'},
  {code: 50, name: 'Built-up', color: 'fa0000'},
  {code: 60, name: 'Bare or sparse vegetation', color: 'b4b4b4'},
  {code: 70, name: 'Snow and ice', color: 'f0f0f0'},
  {code: 80, name: 'Permanent water bodies', color: '0064c8'},
  {code: 90, name: 'Herbaceous wetland', color: '0096a0'},
  {code: 95, name: 'Mangroves', color: '00cf75'},
  {code: 100, name: 'Moss and lichen', color: 'fae6a0'}
];

// ---------------------------
// AOI and raster preparation
// ---------------------------

var aoi = userAoi.geometry();
var aoiBounds = aoi.bounds(1);
var clippedLandcover = landcover.clip(aoi);

// Add region id/name as raster bands so every sampled pixel can carry region info.
var regionIdImage = regions.reduceToImage({
  properties: ['ADM1_CODE'],
  reducer: ee.Reducer.first()
}).rename('region_code').clip(aoi);

var sampleImage = clippedLandcover.rename('landcover_class')
  .addBands(regionIdImage);

var landcoverVisual = clippedLandcover.visualize({
  min: 10,
  max: 100,
  palette: palette,
  forceRgbOutput: true
});

// ---------------------------
// Pixel table: one row per 10 m pixel
// ---------------------------

var pixelSamples = sampleImage.sample({
  region: aoi,
  scale: 10,
  geometries: true,
  dropNulls: true,
  tileScale: 4
}).map(function(feature) {
  var coords = ee.List(feature.geometry().coordinates());
  var classValue = ee.Number(feature.get('landcover_class'));
  var regionCode = ee.Number(feature.get('region_code'));
  var regionFeature = ee.Feature(
    regions.filter(ee.Filter.eq('ADM1_CODE', regionCode)).first()
  );
  return feature.set({
    longitude: coords.get(0),
    latitude: coords.get(1),
    class_name: classNames.get(classValue.format()),
    class_color: classColors.get(classValue.format()),
    region_name: regionFeature.get('ADM1_NAME'),
    country_name: regionFeature.get('ADM0_NAME')
  });
});

// ---------------------------
// Region summary based on sampled pixels
// ---------------------------

var validPixels = pixelSamples.filter(ee.Filter.notNull(['region_code']));
var regionCodes = ee.List(validPixels.aggregate_array('region_code')).distinct().sort();

var summarizeRegion = function(regionCode) {
  regionCode = ee.Number(regionCode);
  var pixels = validPixels.filter(ee.Filter.eq('region_code', regionCode));
  var firstPixel = ee.Feature(pixels.first());

  var histogram = ee.Dictionary(pixels.aggregate_histogram('landcover_class'));
  var classKeys = histogram.keys().map(ee.Number.parse);
  var classStats = ee.FeatureCollection(classKeys.map(function(key) {
    key = ee.Number(key);
    return ee.Feature(null, {
      class_value: key,
      pixel_count: ee.Number(histogram.get(key.format()))
    });
  }));

  var dominant = ee.Feature(classStats.sort('pixel_count', false).first());
  var dominantValue = ee.Number(dominant.get('class_value'));
  var dominantCount = ee.Number(dominant.get('pixel_count'));
  var totalCount = ee.Number(pixels.size());

  return ee.Feature(null, {
    region_code: regionCode,
    region_name: firstPixel.get('region_name'),
    country_name: firstPixel.get('country_name'),
    dominant_class: dominantValue,
    dominant_name: classNames.get(dominantValue.format()),
    dominant_color: classColors.get(dominantValue.format()),
    dominant_pixels: dominantCount,
    total_pixels: totalCount,
    dominant_share_pct: dominantCount.divide(totalCount).multiply(100)
  });
};

var regionSummary = ee.FeatureCollection(regionCodes.map(summarizeRegion));

var styledRegions = regions
  .filter(ee.Filter.inList('ADM1_CODE', regionCodes))
  .map(function(feature) {
    var match = ee.Feature(
      regionSummary.filter(ee.Filter.eq('region_code', feature.get('ADM1_CODE'))).first()
    );
    return feature.set('style', {
      color: '202020',
      width: 1,
      fillColor: ee.String(match.get('dominant_color')).cat('AA')
    });
  });

// ---------------------------
// Map display
// ---------------------------

Map.centerObject(userAoi, 11);
Map.addLayer(landcoverVisual, {}, 'Pixel land cover');
Map.addLayer(styledRegions.style({styleProperty: 'style'}), {}, 'Dominant class by region');
Map.addLayer(userAoi.style({color: '000000', fillColor: '00000000', width: 2}), {}, 'AOI');

print('AOI feature count', userAoi.size());
print('Pixel sample count', pixelSamples.size());
print('Pixel samples', pixelSamples.limit(10));
print('Region summary', regionSummary);

// ---------------------------
// Legend
// ---------------------------

var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 12px'
  }
});

legend.add(ui.Label({
  value: 'Pixel land cover',
  style: {fontWeight: 'bold', fontSize: '14px'}
}));

legendItems.forEach(function(item) {
  var row = ui.Panel({
    widgets: [
      ui.Label('', {
        backgroundColor: '#' + item.color,
        padding: '8px',
        margin: '0 6px 4px 0'
      }),
      ui.Label(item.code + ' - ' + item.name, {margin: '0 0 4px 0'})
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
  legend.add(row);
});

Map.add(legend);

// ---------------------------
// Exports
// ---------------------------

Export.image.toDrive({
  image: landcoverVisual,
  description: 'test1_pixels_landcover_rgb_geotiff',
  folder: driveFolder,
  fileNamePrefix: 'test1_pixels_landcover_rgb',
  region: aoiBounds,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.table.toDrive({
  collection: pixelSamples.select([
    'longitude',
    'latitude',
    'landcover_class',
    'class_name',
    'class_color',
    'region_code',
    'region_name',
    'country_name'
  ]),
  description: 'test1_pixels_landcover_csv',
  folder: driveFolder,
  fileNamePrefix: 'test1_pixels_landcover',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: regionSummary.select([
    'country_name',
    'region_code',
    'region_name',
    'dominant_class',
    'dominant_name',
    'dominant_color',
    'dominant_pixels',
    'total_pixels',
    'dominant_share_pct'
  ]),
  description: 'test1_regions_dominant_landcover_csv',
  folder: driveFolder,
  fileNamePrefix: 'test1_regions_dominant_landcover',
  fileFormat: 'CSV'
});
