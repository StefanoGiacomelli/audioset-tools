# AudioSet-EV - Preparation Guide

This file provides instructions on downloading, extracting, and formatting AudioSet-EV dataset to be ready for use. Follow the steps below to set up the dataset properly.

Commands pipeline is tested on Unix-based systems.

## 1. Download

Move inside the appropriate folder and download the dataset (from Zenodo), using the following commands:
```bash
cd ./Ev-benchmark/AudioSet-EV/
wget -O AudioSet-EV.zip [DATASET_DOWNLOAD_URL]
```
or directly download it from [Zenodo](URL) an move ```.zip``` file inside this folder.

## 2. UnZip contents

Unzip contents in the CWD with:
```bash
unzip AudioSet-EV.zip -d ./
rm -rf AudioSet-EV.zip          # Optional (remove .zip archive)
```
Resulting size of AudioSet-EV must be around 8.3GiB (without ```.zip``` file).

## 3. Check Directory contents

Check CWD contents 
```bash
ls                              # Output: Negatives_files, Positives_files, EV_Negatives.csv, EV_Positives.csv
```

# License

License Here ToDo

# Citation

If you use this dataset, please cite:
```
bibtex citation ToDo
```