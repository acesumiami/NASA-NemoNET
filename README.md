# NASA-NemoNET
This repository contains a deep learning solution for semantic segmentation of coral reef satellite imagery. It implements a U-Net architecture with attention mechanisms for satellite image segmentation, supporting both Sentinel-2 and WorldView-2 satellite data.

## Installation
Clone this repository and install the required dependencies:
```
git clone https://github.com/acesumiami/NASA-NemoNET.git
cd NASA-NemoNET
pip install -r requirements.txt
```

## Configuration
Edit the 'config.yml' file to set up the project:

```
satellite_type: 'sn2'
# Paths configuration
paths:
  output_dir: '/path/to/output'
# Image directories
image_set_loader:
  train:
    image_dir: '/path/to/TrainImages/'
    label_dir: '/path/to/LabelImages/'
    .
    .
    .
```

## Directory Structure
For optimal performance, organize the data like this:
```
data/
├── TrainImages/
│   ├── Class1/
│   │   ├── image1.tif
│   │   ├── image2.tif
│   │   └── …
│   ├── Class2/
│   └── …
├── LabelImages/
│   ├── Class1/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── …
│   ├── Class2/
│   └── …
├── ValidationImages/
│   └── …
└── ValidationLabels/
    └── …
```

Labels should be RGB images where each class is represented by a specific color:

| Class | Name | Hex Color |
|-------|------|-----------|
| 0 | Coral | #ff6347 |
| 1 | Coral Fore Reef | #ffa500 |
| 2 | Reef Crest - Coralline Algal Ridge | #2ca02c |
| 3 | Algae | #17becf |
| 4 | Seagrassae | #32cd32 |
| 5 | Sediment & Rubble | #f5deb3 |
| 6 | Terrestrial Vegetated | #8b4513 |
| 7 | Mangroves | #e377c2 |
| 8 | Deep Water | #00008b |
| 9 | Terrestrial Other & No Data | #808080 |
| 10 | Clouds | #ffffff |

## Training
To train a new model:

```
python train.py --config_path config.yml
```

## Prediction
To run predictions on large satellite images:

```
python predict.py --config_path config.yml --image_path /path/to/image.tif
```
## Satellite Bands
The model supports two satellite types:
### Sentinel-2 (sn2)
Uses bands [1, 2, 3, 4] (Blue, Green, Red, NIR)
### WorldView-2 (wv2)
Uses bands [2, 3, 5, 7] (Blue, Green, Red, NIR1)

You can select the satellite type in the config file or use the '--satellite' argument.

## Output Files
The prediction script generates:
- Visual comparison of original image, ground truth, and prediction
- GeoTIFF segmentation map with proper coloring and georeferencing
- Confusion matrix and metrics plots (if ground truth available)
- CSV file with detailed evaluation metrics
