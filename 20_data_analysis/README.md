# [**`20_data_analysis`**](../20_data_analysis/)

This directory contains the Exploratory Data Analysis (EDA) for the OilSlick Sentinel-1 dataset. The analysis computes statistics, visualizes distributions, filters invalid samples, and obtains split-specific normalization parameters (mean, std) used in the training pipeline in [`../30_experiments/`](../30_experiments/).

## Contents

| File | Description |
|---|---|
| [`oilslick_eda.ipynb`](oilslick_eda.ipynb) | Jupyter Notebook for EDA workflow |
| [`filtered_metadata.csv`](filtered_metadata.csv) | Per-sample metadata with validity flags after filtering |
| [`split_stats.json`](split_stats.json) | Per-split means, standard deviations, and sample counts for z-score normalization |

## EDA Workflow

The notebook [`oilslick_eda.ipynb`](oilslick_eda.ipynb) is structured as follows:

### 1. Data Loading & Overview

- Loads raw [`../10_waterbench_data/data/OilSlick/metadata.csv`](../10_waterbench_data/data/OilSlick/metadata.csv)
- Maps labels (`0` = negative, `1` = positive) and subcategories (Ships, Platforms, Natural seeps)

### 2. Spatial & Label Distribution

- Bar charts of binary label distribution and subcategory counts
- Interactive `folium` world map colored by positive subcategory vs. negative samples

### 3. Value Range Analysis

- Pixel-level statistics (mean, std, median, percentiles) for VV and VH channels across all valid pixels
- Log-scale histograms and boxplots showing the dB value distributions

### 4. Nodata Calculation & Filtering

Since the original `nodata_fraction` column describes Sentinel-2 data, it is not applicable to Sentinel-1 SAR. A custom validity check is applied per GeoTIFF:

- **Nodata threshold**: Pixels with values <= −50 dB or `NaN` are considered nodata; chips with **>= 40 % nodata** are removed
- **Corruption check**: Chips with zero standard deviation in any band are discarded

| Metric | Count |
|---|---|
| Total samples | 1,363 |
| Valid after filtering | 1,196 (87.7 %) |
| Removed | 167 (12.3 %) |

### 5. Split Statistics (Filtering Comparison)

Compares train / val / test sizes for both split types before and after filtering:

| Split | Phase | Before | After |
|---|---|---|---|
| **Random** | Train | 900 | 793 |
| | Val | 150 | 127 |
| | Test | 300 | 266 |
| **Geographic** | Train | 864 | 749 |
| | Val | 217 | 189 |
| | Test | 150 | 141 |

### 6. Train-Set Normalization Parameters

For each split type, mean and standard deviation are computed exclusively from the training set (to prevent data leakage) and stored in [`split_stats.json`](split_stats.json) in a manner like this:

```json
{
  "random": {
    "means": {"0": -17.87, "1": -25.61},
    "stds":  {"0": 26.34,  "1": 19.91},
    "train": {"n_samples": 793},
    "val":   {"n_samples": 127},
    "test":  {"n_samples": 266}
  },
  "geographic": {
    "means": {"0": -19.42, "1": -26.84},
    "stds":  {"0": 22.68,  "1": 17.84},
    "train": {"n_samples": 749},
    "val":   {"n_samples": 189},
    "test":  {"n_samples": 141}
  }
}
```

These values are used by the `SARzScore` transform in [`../30_experiments/dataloader.py`](../30_experiments/dataloader.py) to normalize VV (channel `0`) and VH (channel `1`) before model training.

### 7. Example Visualizations

Plots are presented in the notebook to illustrate example TIFF chips, including both valid and invalid samples (for each VV and VH channel).

## Final Structure

After execution, the final structure of the [`20_data_analysis/`](./) directory is as follows:

```bash
20_data_analysis/
├── oilslick_eda.ipynb
├── filtered_metadata.csv
└── split_stats.json
```
