# DataScienceToolbox-Project1

## Project Group

Kyra Le Quelenec, Neva Fradd, Harry Clarke, Nadia Kusneraitis.

Equal equity split between group.

## Reading order

All report content is in the directory Report/

The reading order of the content is:
* 01- (data stuff)
* 02- (regression model)
* 03- (SVM model)
* 04- (XGBoost model)
* 05- (Random forest model)
* 06-Wrapup.md

## Requirements

Requirements for the Rmd files are given within each script.

Requirements for each python file is given in individual requirements.txt files:
* `XGBoost-requirments.txt`
* 
* 

To install these, in a virtual environment run:
```{sh}
pip3 install -r requirements.txt
```
replacing the general requirments.txt with the specific file for that dataset.

## Description

In this project we have each built a classification model for the stroke dataset. Our goal is to find the most powerful model in predicting cases of stroke. We have chosen to use the AUC-PR performance metric due to the imbalance in our data. We will each build our model independently then compare the output performance metrics on the test dataset. In our wrapup we will consider these results in the context of the dataset.
