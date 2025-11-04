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


## Final ranking of the models

1.
2.
3.
4. SVM

## Final ranking of the models


## Further comparsion of the models and application to real world
If this model were to be used as a screening method in the real world then the false negative rate would be more important than the false positive rate. This is because a high false negative rate could directly cause loss of life, if fatal cases of stroke are missed. Whereas a high false negative rate might cause some unnecessary further examination of patients but this is a much safer option than missing true cases of stroke. Hence if our models were to be used in a real life scenario, it might not be that the winning model is actually most appropriate.
