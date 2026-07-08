# [dataset.py](../30_experiments/dataset.py)

Defines the OilSlickDataset class for loading and preprocessing Sentinel-1 SAR images with their binary labels.

## Overview

A PyTorch Dataset class that implements proper loading of Sentinel-1 Synthetic Aperture Radar (SAR) imagery from GeoTIFF format and their corresponding binary labels. Handles sample filtering, image loading, tensor conversion, and optional transformations for training and evaluation pipelines.

## Core Components

### OilSlickDataset Class

A custom PyTorch `Dataset` subclass for the oil slick detection task. Manages:
- Loading filtered metadata and binary labels
- Spatial data organization by split type (random/geographic)
- GeoTIFF image reading and tensor conversion
- Optional transform application (normalization, augmentation)

## Initialization Parameters

* **`data_root`** Root directory containing the OilSlick dataset. Expected structure: `{data_root}/images_s1/` for images and `{data_root}/splits/{split_type}/` for split definitions.

* **`split_type`** Data split strategy. Options: `["random", "geographic"]`. Determines which train/val/test split to load.

* **`split`** Which data subset to load. Options: `["train", "val", "test"]`. Loads corresponding split file.

* **`transform`** Optional PyTorch transform function to apply to images. Applied after tensor conversion. Examples: normalization, augmentation, resizing.

## Core Methods

* **`__init__(data_root, split_type, split, transform)`** Initializes dataset by:
  1. Loading filtered metadata from `../20_data_analysis/filtered_metadata.csv`
  2. Filtering to valid samples only (`valid_sample == True`)
  3. Reading split file from `splits/{split_type}/{split}.txt`
  4. Filtering to samples with both metadata and image files
  5. Storing labels dictionary and image IDs list

* **`__len__()`** Returns the total number of valid samples in the dataset.

* **`__getitem__(idx)`** Retrieves a single sample:
  1. Gets image ID from index
  2. Loads 2-band GeoTIFF file using `tifffile.imread()`
  3. Converts to float32 PyTorch tensor
  4. Handles dimension conversion (ensures shape: [2, H, W] for VV and VH bands)
  5. Retrieves binary label (0 = no slick, 1 = oil slick)
  6. Applies optional transform
  7. **Returns:** tuple of (image_tensor, label_tensor)

## Data Structure

### Input GeoTIFF Files

- **Location**: `{data_root}/images_s1/{sample_id}_s1.tif`
- **Format**: GeoTIFF with 2 bands (VV and VH Sentinel-1 polarizations)
- **Dimensions**: 2 channels × H × W (H and W typically 224×224 after transform)
- **Data Type**: Float32

### Metadata

- **Source**: `../20_data_analysis/filtered_metadata.csv`
- **Columns**: `sample_id`, `label`, `valid_sample`
- **Filtering**: Only samples with `valid_sample == True` are included

### Split Files

- **Location**: `{data_root}/splits/{split_type}/{split}.txt`
- **Format**: Text file with one sample ID per line
- **Types**: 
  - `random/{train,val,test}.txt` - Random spatial splits
  - `geographic/{train,val,test}.txt` - Geographic (out-of-distribution) splits

## Split Statistics

### Random Split (In-distribution)

| Subset | Samples |
|--------|---------|
| Train | 793 |
| Validation | 127 |
| Test | 266 |
| **Total** | **1,186** |

### Geographic Split (Out-of-distribution)

| Subset | Samples |
|--------|---------|
| Train | 749 |
| Validation | 189 |
| Test | 141 |
| **Total** | **1,079** |

## Data Loading Pipeline

| Step | Description | Output |
|------|-------------|--------|
| 1. Metadata Load | Read `filtered_metadata.csv` | Dict mapping sample_id → label |
| 2. Filtering | Keep only `valid_sample == True` | Reduced metadata dict |
| 3. Split Load | Read `{split_type}/{split}.txt` | List of sample IDs for subset |
| 4. File Validation | Check image existence on disk | Final valid sample IDs |
| 5. Image Load | Read GeoTIFF via `tifffile.imread()` | Numpy array [2, H, W] |
| 6. Tensor Conversion | Convert to float32 torch tensor | Torch tensor [2, H, W] |
| 7. Transform Application | Apply normalization/augmentation | Transformed tensor [2, 224, 224] |
| 8. Return | Package image and label | (image_tensor, label_tensor) |

## Dependencies

* **Internal:** Metadata from `../20_data_analysis/filtered_metadata.csv`
* **External:** `torch`, `torch.utils.data`, `tifffile`, `pandas`, `numpy`, `os`

## Notes

- Images with missing metadata or file paths are automatically excluded
- All statistics (for z-score normalization) are derived from **training set only** to prevent data leakage
- Transform is applied after tensor creation for efficient processing
