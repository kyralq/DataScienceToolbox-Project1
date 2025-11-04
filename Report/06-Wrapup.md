## Comparison of results
All values stated are given to 4 s.f.


### XGBoost model

* Final test AUC-PR score: 0.1572
* Type I error rate: 0.1914
* Type II error rate: 0.3800

Model strengths: low type I error rate suggests the model makes relatively few false negatives which was the main focus I had when building the model.
Model weakness: significant overfitting to the training data - my best model output an AUC_PR score of 0.2575 which is significantly higher than the score on the test data indicating overfitting. 

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

## Further comparsion of the models and application to real world

