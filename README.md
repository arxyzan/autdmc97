# autdmc97
Amir Kabir University Of Technology Data Mining Competition

## Overview
Alibaba dataset contains about 2 million samples with 8 features. the first task is to predict sales of airplane tickets from a specific city to a specific destination represented by columns FROM and TO. the sales column is produced in data_preparation.py file.
We used Lightgbm to build a tree-based model to solve this problem.

## Running Code
To produce the dataset ready for the training run the command below which runs the file data_preparation.py.

```
python data_preparation.py
```
### Training
lgbm parameters are configured in lgbm.py, the model can be trained on the dataset by running:

```
python train_model.py
```
The eval metrics are MAPE and RMSE and will be verbosed while training.

