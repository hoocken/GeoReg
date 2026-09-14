# GeoReg: Registration of CTA and Fluoroscopy Scans

This is a fork of the repository from the MIDL 2026 submission:

**"Direct biplanar DSA-to-CTA registration with geodesic consistency for acute ischemic stroke"**
*Rudolf L. M. van Herten, Robert Graf, Felix Bitzer, Jan S. Kirschke, Johannes C. Paetzold*

## Overview

This repository extends the original GeoReg to work with 2D fluoroscopy scans for the torso area, but without the biplanar registration.

## Installation
This repository uses [uv](https://docs.astral.sh/uv/) as a package manager. To install the packages, run
```sh
uv sync
```

## Usage
To use this repository, the inputs must be put in the folder `data`. In the end, you should have the following directory structure:
```
data
├── CTA_maskTr          # CTA Segmentation
│   └── sub-*.nii.gz
├── CTATr               # CTA Volume
│   └── sub-*_0000.nii.gz
├── F_maskTr            # Fluoroscopy Segmentation
│   └── sub-*.npy  
├── F_metadataTr        # Fluoroscopy Metadata
│   └── sub-*.json
└── FTr                 # Fluoroscopy Scan
    └── sub-*.nii.gz
```
where the input in CTATr must be of the form `sub-*_0000.nii.gz`, with files in other folders following `sub-*.nii.gz`.

To run the program, simply set the index in `configs/register_local.yaml` to select which data to fit and run
```sh
uv run main.py
```

### CTA Segmentation
For CTA Segmentation, we use [VIBESegmentator](https://github.com/robert-graf/VIBESegmentator). For tutorial on how to use them, refer to the repository website.

### Fluoroscopy Segmentation
For fluoroscopy segmentation, we use this implementation of [UNet](https://github.com/hoocken/UNet). Please refer to the repository for usage instructions.

### Config
You can modify how the optimization is run from the configs in `configs`. Primarily, `register_local.yaml` is used, but you can change the path in `main.py`.

The `use_truncated` field determines if we use CT segmentations that are truncated at the inferior side. You may need to tweak it to either `false` or `true` to get a good result.

## License

© 2025 CC-BY 4.0
