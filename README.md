# Medical-Image-Classification
 
 
## Overview

This repository provides a naive baseline and submission demo for the  [MedFM Challenge 2023](https://medfm2023.grand-challenge.org/medfm2023/).

## Installation

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
pip install mmcls==0.25.0 openmim scipy scikit-learn ftfy regex tqdm
mim install mmcv-full==1.6.0

```

## Results

The results of ChestDR, ColonPath and Endo in MedFMC dataset and their corresponding configs on each task are shown as below.

### Few-shot Learning Results

We utilize [Visual Prompt Tuning](https://github.com/KMnP/vpt) method as the few-shot learning baseline, whose backbone is Swin Transformer.
The results are shown as below:

#### ChestDR

| N Shot | Crop Size | Epoch |  mAP  |  AUC  |                                      Config                                      |
| :----: | :-------: | :---: | :---: | :---: | :------------------------------------------------------------------------------: |
|   1    |  384x384  |  20   | 13.14 | 56.49 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest_adamw.py)  |
|   5    |  384x384  |  20   | 17.05 | 64.86 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_5-shot_chest_adamw.py)  |
|   10   |  384x384  |  20   | 19.01 | 66.68 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_chest_adamw.py) |

#### ColonPath

| N Shot | Crop Size | Epoch |  Acc  |  AUC  |                                      Config                                      |
| :----: | :-------: | :---: | :---: | :---: | :------------------------------------------------------------------------------: |
|   1    |  384x384  |  20   | 77.60 | 84.69 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon_adamw.py)  |
|   5    |  384x384  |  20   | 89.29 | 96.07 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_5-shot_colon_adamw.py)  |
|   10   |  384x384  |  20   | 91.21 | 97.14 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_colon_adamw.py) |

#### Endo

| N Shot | Crop Size | Epoch |  mAP  |  AUC  |                                     Config                                      |
| :----: | :-------: | :---: | :---: | :---: | :-----------------------------------------------------------------------------: |
|   1    |  384x384  |  20   | 19.70 | 62.18 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo_adamw.py)  |
|   5    |  384x384  |  20   | 23.88 | 67.48 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_5-shot_endo_adamw.py)  |
|   10   |  384x384  |  20   | 25.62 | 71.41 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_endo_adamw.py) |

### Transfer Learning on 20% (Fully Supervised Task)

Noted that MedFMC mainly focuses on few-shot learning i.e., transfer learning task.
Thus, fully supervised learning tasks below only use 20% training data to make corresponding comparisons.

#### ChestDR

|    Backbone     | Crop Size | Epoch |  mAP  |  AUC  |                        Config                         |
| :-------------: | :-------: | :---: | :---: | :---: | :---------------------------------------------------: |
|   DenseNet121   |  384x384  |  20   | 24.48 | 75.25 |     [config](configs/densenet/dense121_chest.py)      |
| EfficientNet-B5 |  384x384  |  20   | 29.08 | 77.21 |    [config](configs/efficientnet/eff-b5_chest.py)     |
|     Swin-B      |  384x384  |  20   | 31.07 | 78.56 | [config](configs/swin_transformer/swin-base_chest.py) |

#### ColonPath

|    Backbone     | Crop Size | Epoch |  Acc  |  AUC  |                        Config                         |
| :-------------: | :-------: | :---: | :---: | :---: | :---------------------------------------------------: |
|   DenseNet121   |  384x384  |  20   | 92.73 | 98.27 |     [config](configs/densenet/dense121_colon.py)      |
| EfficientNet-B5 |  384x384  |  20   | 94.04 | 98.58 |    [config](configs/efficientnet/eff-b5_colon.py)     |
|     Swin-B      |  384x384  |  20   | 94.68 | 98.35 | [config](configs/swin_transformer/swin-base_colon.py) |

#### Endo

|    Backbone     | Crop Size | Epoch |  mAP  |  AUC  |                        Config                        |
| :-------------: | :-------: | :---: | :---: | :---: | :--------------------------------------------------: |
|   DenseNet121   |  384x384  |  20   | 41.13 | 80.19 |     [config](configs/densenet/dense121_endo.py)      |
| EfficientNet-B5 |  384x384  |  20   | 36.95 | 78.23 |    [config](configs/efficientnet/eff-b5_endo.py)     |
|     Swin-B      |  384x384  |  20   | 41.38 | 79.42 | [config](configs/swin_transformer/swin-base_endo.py) |

## Usage

### Data Preparation

Prepare data following  [MMClassification](https://github.com/open-mmlab/mmclassification).

### Training and Evaluation

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/train.py $CONFIG
python tools/test.py $CONFIG $CHECKPOINT --metrics mAP

```

### Generating Submission Results

```bash
python tools/test_prediction.py $DATASETPATH/test_WithoutLabel.txt $DATASETPATH/images/ $CONFIG $CHECKPOINT --output-prediction $DATASET_N-shot_submission.csv

```

Zip the generated files into  `result.zip`  and upload to Grand Challenge website.

```sql
result/
├── endo_1-shot_submission.csv
├── endo_5-shot_submission.csv
├── endo_10-shot_submission.csv
├── colon_1-shot_submission.csv
├── colon_5-shot_submission.csv
├── colon_10-shot_submission.csv
├── chest_1-shot_submission.csv
├── chest_5-shot_submission.csv
├── chest_10-shot_submission.csv
```
