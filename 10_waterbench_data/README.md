# [**`10_waterbench_data`**](10_waterbench_data/)

This directory contains the OilSlick subset of the [WaterBench](https://huggingface.co/datasets/ayushprd/WaterBench) dataset - a collection of Sentinel-1 SAR image chips (with VV and VH channels) for binary oil slick detection. The dataset also includes Sentinel-2 optical imagery, but this is not used in the experiments of this project.

## Contents

| Path | Description |
|---|---|
| `data/OilSlick/metadata.json` | Dataset schema (columns, task, spatial OOD info) |
| `data/OilSlick/metadata.csv` | Raw metadata for all 1,363 samples (Sentinel-2 based) |
| `data/OilSlick/images_s1/` | 2-band GeoTIFFs (VV + VH, Sentinel-1 SAR) |
| `data/OilSlick/splits/random/` | In-distribution random split (train / val / test) |
| `data/OilSlick/splits/geographic/` | Geographic split (train / val / test) with Mediterranean as OOD test set |

## Setup

After running the [`../setup_data.sh`](../setup_data.sh) script, the desired dataset is located in the [`10_waterbench_data/`](./) directory.

> **Note**: The full OilSlick SAR imagery may not be fully available on Hugging Face. Please get the whole imagery set after running the script from the [Google Drive mirror](https://drive.google.com/file/d/1yv23B9PtfBD10j2VcYs5JCcdHOOXcI4S/view?usp=sharing) into `data/OilSlick/images_s1/`.

## Metadata Disclaimer

The main purpose of WaterBench is focusing on the Sentinel-2 optical imagery, but in this project, we only use the Sentinel-1 SAR imagery. By this, the [`metadata.csv`](data/OilSlick/metadata.csv) and [`metadata.json`](data/OilSlick/metadata.json) are mainly useful for the Sentinel-2 imagery (e.g. cloud cover, reflectance), and not applicable to the Sentinel-1 SAR chips throughout this project.

Based on this, by filtering the dataset according to corrupted imagery or chips with nodata >= 40%, the [`../20_data_analysis/`](../20_data_analysis/) directory provides a filtered dataset CSV ([`filtered_metadata.csv`](../20_data_analysis/filtered_metadata.csv)) and JSON split statistics for normalization ([`split_stats.json`](../20_data_analysis/split_stats.json)). These are going to be created after running the [`oilslick_eda.ipynb`](../20_data_analysis/oilslick_eda.ipynb) notebook.

## Final Structure

After execution, the final structure of the [`10_waterbench_data/`](./) directory is as follows:

```bash
data/OilSlick/
├── metadata.json
├── metadata.csv
├── images_s1/
└── splits/
    ├── random/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    └── geographic/
        ├── train.txt
        ├── val.txt
        └── test.txt
```
