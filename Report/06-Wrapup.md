## Comparison of results
All values stated are given to 4 s.f.


### XGBoost model

* Final test AUC-PR score: 0.1572
* Type I error rate: 0.1914
* Type II error rate: 0.3800

Model strengths: low type I error rate suggests the model makes relatively few false positives, which is particuarly good given the imbalance of the data, with few positive cases to train on.
Model weakness: significant overfitting to the training data - my best model output an AUC_PR score of 0.2575 during training which is significantly higher than the score on the test data indicating overfitting.

### SVM model

* Final test AUC-PR score: 0.06291
* Type I error rate: 0.0
* Type II error rate: 0.04728

Model strengths: Applicable and computationally inexpensive for size of dataset and number of features.
Model weakness: Sensitive to noisy / irrelevant feature variables in the dataset. Ended up guessing 'no stroke' for all samples in testing set.

### Random Forest model

* Final test AUC-PR score: 0.2386
* Type I error rate: 0.1002
* Type II error rate: 0.5263

Model strengths: Highest AUC-PR score thanks to hyperparameter tuning and minimal overfitting since robust to noise.
Model weakness: High type II error indicating that the model is too conservative when predicting positive cases. Also, computationally expensive with some parameter selection taking almost 2 minutes to run despite the dataset being quite small, which suggests it’s not scalable.


## Final ranking of the models

1. Random Forest
2.
3.
4. SVM

## Final ranking of the models


## Further comparsion of the models and application to real world
If this model were to be used as a screening method in the real world then the false negative rate would be more important than the false positive rate. This is because a high false negative rate could directly cause loss of life, if fatal cases of stroke are missed. Whereas a high false negative rate might cause some unnecessary further examination of patients but this is a much safer option than missing true cases of stroke. Hence if our models were to be used in a real life scenario, it might not be that the winning model is actually most appropriate.

Random Forest achieved the highest score for our performance metric - AUC-PR, which focuses on correctly identifying positive cases. This means the Random Forest model was the best at predicting true stroke cases while minimising false alarms. If the model was used as a key diagnosis tool, this attribute would be essential to avoid misdiagnosis. 

If, however, the model was used as an initial screening tool, we would also want prioritise minimising false negatives (type II error). The SVM model has a low type II error rate but has a very low AUC-PR score and wouldn't be useful. On the other hand, the XGBoost model has a relatively low type II error rate while maintaining the second highest AUC-PR score. For this purpose, the XGBoost model would likely be the most useful.
