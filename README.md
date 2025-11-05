# DataScienceToolbox-Project1

## Project Group

Kyra Le Quelenec, Neva Fradd, Harry Clarke, Nadia Kusneraitis.

Equal equity split between group.

## Reading order

All report content is in the directory Report/

The reading order of the content is:
* 01-Data.ipynb
* 02-LogisticRegression-Model.Rmd
* 03-RandomForest-model.ipynb
* 04-XGboost-model.ipynb
* 05-SVM-model.ipynb
* 06-Wrapup.md

## Requirements

Requirements for the Rmd files are given within each script.

Requirements for each python file is given in individual requirements.txt files:
* `XGBoost-requirments.txt`
* `RandomForest-requirements.txt`
* `SVM-requirements.txt`

To install these, in a virtual environment run:
```{sh}
pip3 install -r requirements.txt
```
replacing the general requirments.txt with the specific file for that dataset.

## Description

In this project we have each built a classification model for the stroke dataset. Our goal is to find the most powerful model in predicting cases of stroke. We have chosen to use the AUC-PR performance metric due to the imbalance in our data. We will each build our model independently then compare the output performance metrics on the test dataset. In our wrapup we will consider these results in the real-world context of the dataset.
