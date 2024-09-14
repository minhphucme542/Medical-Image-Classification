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

Dataset

Backbone

mAP

AUC

ChestDR

Swin-B

31.07

78.56

ColonPath

Swin-B

94.68

98.35

Endo

Swin-B

41.38

79.42

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
