# Crop Map SRS

This project performs crop classification using Sentinel-2 satellite imagery and Random Forest in Google Earth Engine.

## Data
Training dataset contains crop samples:
- flax (лен)
- wheat (пшеница)

## Features used
- Sentinel-2 spectral bands (B2, B3, B4, B8)
- NDVI
- EVI
- NDWI

## Method
Random Forest classifier.

## Results
Classification map of crop types in the study area.

Overall accuracy ≈ 0.56

Google Earth Engine script:
https://code.earthengine.google.com/450517689abd2bd8f948a98b89a8be16