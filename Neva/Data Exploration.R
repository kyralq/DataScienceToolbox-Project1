
install.packages("tidymodels")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("rpivotTable")

library("tidyverse")
library("tidymodels")
library("rlang")
library("corrplot")
library("GGally")
library("patchwork")
library("recipes")
library("glmnet")

stroke_train <- read.csv("Data/stroke_training_dataset.csv")

head(stroke_train)
summary(stroke_train)
str(stroke_train)

## get rid of identity column

stroke_train <- stroke_train[ , 2:12]
head(stroke_train)
##overall class imbalance, <5% of the data set are positive for stroke 

table(stroke_train$stroke)
prop.table(table(stroke_train$stroke))

## look at overall distributions of numerical distributions

numeric_vars <- c("age" , "bmi" , "avg_glucose_level")
summary(stroke_train[ , numeric_vars])

## several of our categorical variables are being treated like they're continuous

stroke_train <- stroke_train %>%
  mutate(
    gender = factor(gender),
    hypertension = factor(hypertension) , 
    ever_married = factor(ever_married),
    work_type = factor(work_type),
    heart_disease = factor(heart_disease) , 
    Residence_type = factor(Residence_type),
    smoking_status = factor(smoking_status),
    stroke = factor(stroke)
  )


summary(stroke_train)

## look for missing values in all columns
colSums(is.na(stroke_train))

## there are 173 missing values in BMI, we'll revisit this

##look at age distribution

ggplot(stroke_train , aes(age)) +
  geom_histogram(bins = 30)

## look at glucose distribution 

ggplot(stroke_train, aes(avg_glucose_level)) +
  geom_histogram(bins = 30)

## look at BMI distribution

ggplot(stroke_train, aes(bmi)) +
  geom_histogram(bins = 30)
## hasn't included the 173 missing values


## below plots all 3 together
plots <- lapply(numeric_vars, function(var){
  ggplot(stroke_train, aes_string(x = var)) +
    geom_histogram(bins = 30, fill = "blue", color = "black") +
    theme_minimal() +
    ggtitle(paste(var, "distribution"))
})

print(wrap_plots(plots, ncol = 3))


##for categorical data, we can look at categorical frequency

stroke_train %>% 
  select(gender, work_type, Residence_type, smoking_status, ever_married, heart_disease, hypertension) %>%
  map(~table(.)) 
## this gives us counts

ggplot(stroke_train, aes(gender)) + geom_bar()
ggplot(stroke_train, aes(work_type)) + geom_bar()
ggplot(stroke_train, aes(smoking_status)) + geom_bar()
ggplot(stroke_train , aes(Residence_type)) + geom_bar()
ggplot(stroke_train , aes(ever_married)) +geom_bar()
ggplot(stroke_train , aes(heart_disease)) +geom_bar()
ggplot(stroke_train , aes(hypertension)) +geom_bar()

## consider rates of hypertension/heart disease in population vs sample
## is there  a higher percentage of hypertension/ heart distease in sample than in real life
## this gives visualisations


## continuing with visualisation, I want to plot stroke against single factors
## I'm doing this to understand ....
ggplot(stroke_train, aes(age, fill = factor(stroke))) +
  geom_histogram(position = "identity", alpha = 0.5)

ggplot(stroke_train, aes(age, stroke)) +
  geom_jitter(alpha = 0.2) + 
  ggtitle("Scatter Stroke vs Age")

## from the scatter plot we can see the two outliers - patients significantly younger than 40 who have had strokes
## I want to look at proportional stroke rates by age

ggplot(stroke_train, aes(x = age, fill = stroke)) +
  geom_histogram(position = "fill", binwidth = 5) +
  ylab("Proportion") +
  ggtitle("Proportion of Stroke by Age")

## Age has a significant effect on stroke rate - can see this from proportion histogram

## I want to look at proportional stroke rates by variable for each covariate

##Numeric Covariates:

ggplot(stroke_train, aes(bmi, stroke)) +
  geom_jitter(alpha = 0.2) + 
  ggtitle("Scatter Stroke vs BMI")

ggplot(stroke_train, aes(x = bmi, fill = stroke)) +
  geom_histogram(position = "fill", binwidth = 5) +
  ylab("Proportion") +
  ggtitle("Proportion of Stroke by BMI")

##potential outliers>75 may affect model

ggplot(stroke_train, aes(x = avg_glucose_level, fill = stroke)) +
  geom_histogram(position = "fill", binwidth = 5) +
  ylab("Proportion") +
  ggtitle("Proportion of Stroke by Average Glucose Level")

##steady increase

##look further at density plots of stroke/no stroke against glucose as it should be a high predictor
ggplot(stroke_train, aes(avg_glucose_level, fill = factor(stroke))) +
  geom_density(alpha = 0.5)

ggplot(stroke_train, aes(gender, fill = factor(stroke))) +
  geom_bar(position = "fill") +
  ylab("Proportion with stroke")

## potentially remove 'other' for simpler analysis - same proportion accross gender

ggplot(stroke_train, aes(work_type, fill = factor(stroke))) +
  geom_bar(position = "fill") +
  coord_flip()

## can see a difference between job (self employed, private, govt) verses no job (never worked, children) - could i combine these covariates?

ggplot(stroke_train, aes(smoking_status, fill = factor(stroke))) +
  geom_bar(position = "fill")
## differences between proportions

ggplot(stroke_train, aes(factor(hypertension), fill = factor(stroke))) +
  geom_bar(position = "fill")

ggplot(stroke_train, aes(factor(heart_disease), fill = factor(stroke))) +
  geom_bar(position = "fill")

## I have done all this EDA to look at how individual covariates affect stroke rate, and to determine any outliers.
## Logistic regression is particularly sensitive to extreme values so I'm going to remove them

####handling the missing BMI - we replace N/A with the median of the training data, and add a labelling column 
bmi_plot_1 <- ggplot(stroke_train, aes(bmi)) +
  geom_histogram(bins = 30)


bmi_median <- median(stroke_train$bmi , na.rm = TRUE)
stroke_train <- stroke_train %>% 
  mutate(
    bmi_imputed = ifelse(is.na(bmi), bmi_median , bmi) , 
    bmi_imputed_flag = ifelse(is.na(bmi), 1, 0)
  )

bmi_plot_2 <- ggplot(stroke_train , aes(bmi_imputed)) +
  geom_histogram(bins = 30)

bmi_plot_1 + bmi_plot_2

## look at outliers

ggplot(stroke_train, aes(x = "", y = bmi)) + geom_boxplot()
ggplot(stroke_train, aes(x = "", y = avg_glucose_level)) + geom_boxplot()
## get rid of BMI > 75?
## IQR is mainly overweight


## now I want to look more deeply at interactions. 
## pairwise scatter plot for numeric variables

numeric_df <- stroke_train %>% 
  select(age, bmi_imputed, avg_glucose_level)

corrplot(cor(numeric_df), method = "color", addCoef.col = "black")
## for a more detailed plot, look at ggpairs
GGally::ggpairs(
  stroke_train[, numeric_vars],
  lower = list(continuous = wrap("smooth", color = "magenta", alpha = 0.1)),
  diag = list(continuous = wrap("densityDiag")),
  upper = list(continuous = wrap("cor"))
)

## looking at individual categorical variables to determine if they're relevant 

categorical_vars <- c("gender", "ever_married", "work_type", 
                      "Residence_type", "smoking_status", "heart_disease" , "hypertension")

for (v in categorical_vars) {
  print(
    stroke_train %>%
      group_by(.data[[v]]) %>%
      summarise(stroke_rate = mean(stroke == 1)) %>%
      arrange(desc(stroke_rate))
    )
}

## recall stroke rate = 0.4868

##Here I'm looking at individual covariates to see if stroke rates change accross their levels
## covariates showing minimal differences in stroke rates are unlikely to contribute a strong main 
## effect in the model but may still be involved in interaction terms.


## one hot encoding

numeric_vars <- c("age", "bmi", "avg_glucose_level")
categorical_vars <- c("gender", "ever_married", "work_type",
                      "Residence_type", "smoking_status",
                      "hypertension", "heart_disease")

stroke_recipe <- recipe(stroke ~ ., data = stroke_train) %>%
  
  # 1. Ensure categorical predictors are factors
  step_mutate(across(all_of(categorical_vars), as.factor)) %>%
  
  # 2. Add imputation flag BEFORE imputing BMI (flag stays 0/1)
  step_mutate(bmi_imputed_flag = ifelse(is.na(bmi), 1, 0)) %>%
  
  # 3. Impute BMI using *training-set* median
  step_impute_median(bmi) %>%
  
  # 4. One-hot encode ALL categorical predictors (0/1)
  step_dummy(all_nominal_predictors()) %>%
  
  # 5. Scale ONLY numeric columns (age, bmi, glucose) AFTER dummy encoding
  step_normalize(all_of(numeric_vars)) %>%
  
  # 6. Remove zero-variance predictors (safe clean-up)
  step_zv(all_predictors())

prep_recipe <- prep(stroke_recipe , training = stroke_train)
train_processed <- bake(prep_recipe , new_data = stroke_train)

stroke_test <- read.csv("Data/stroke_testing_dataset.csv")

stroke_test <- stroke_test[ , 2:12]
stroke_test <- stroke_test %>%
  mutate(
    gender = factor(gender),
    hypertension = factor(hypertension) , 
    ever_married = factor(ever_married),
    work_type = factor(work_type),
    heart_disease = factor(heart_disease) , 
    Residence_type = factor(Residence_type),
    smoking_status = factor(smoking_status),
    stroke = factor(stroke)
  )
stroke_test <- stroke_test %>% 
  mutate(
    bmi_imputed = ifelse(is.na(bmi), bmi_median , bmi) , 
    bmi_imputed_flag = ifelse(is.na(bmi), 1, 0)
  )
test_processed <- bake(prep_recipe , new_data = stroke_test)

head(train_processed)
head(test_processed)

setdiff(names(train_processed), names(test_processed))
setdiff(names(test_processed), names(train_processed))
glimpse(train_processed)

colSums(is.na(train_processed))
colSums(is.na(test_processed))

summary(train_processed)
summary(test_processed)

## now i have one hot encoded data, i'm going to build my model, use k-fold cv, evalute models and pick the best one based on AUC

logit_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

tidy(logit_model)

stroke_workflow <- workflow() %>%
  add_recipe(stroke_recipe) %>%
  add_model(logit_model)
## connect model to recipe - same preprocessing used every time, no leakage, k fold cv resuses pre processing the same way
## 10-fold CV
set.seed(123)  # outputs are reproducible
folds <- vfold_cv(stroke_train, v = 10, strata = stroke)

cv_results <- fit_resamples(
  stroke_workflow,
  resamples = folds,
  metrics = metric_set(roc_auc, accuracy, sensitivity, specificity),
  control = control_resamples(save_pred = TRUE)
)

stroke_train %>%
  recipe(stroke ~ ., .) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep() %>%
  bake(new_data = NULL) %>%
  select(where(is.numeric)) %>%
  colnames()

collect_metrics(cv_results)

final_model <- stroke_workflow %>%
  fit(data = stroke_train)

test_predictions <- predict(final_model , new_data = stroke_test , type = "prob") %>%
  bind_cols(stroke_test %>% select(stroke))

# ROC AUC
roc_auc(test_predictions, truth = stroke, .pred_1)

# Confusion matrix
test_predictions %>%
  mutate(
    pred_class = factor(
      ifelse(.pred_1 >= 0.5, 1, 0) , 
      levels = levels(stroke))) %>%
  conf_mat(truth = stroke, estimate = pred_class)

levels(stroke_train$stroke)
levels(stroke_test$stroke)
head(test_predictions)
stroke_train$stroke <- factor(stroke_train$stroke , levels = c(0 ,1))
stroke_test$stroke <- factor(stroke_test$stroke , levels = c(0 ,1))

## logistic regression collapses on inbalanced data sets - hence AUC value of 0.178
## going to try upsampling
library(themis)
stroke_recipe <- recipe(stroke ~ ., data = stroke_train) %>%
  
  # 1. Ensure categorical predictors are factors
  step_mutate(across(all_of(categorical_vars), as.factor)) %>%
  
  # 2. Add imputation flag BEFORE imputing BMI (flag stays 0/1)
  step_mutate(bmi_imputed_flag = ifelse(is.na(bmi), 1, 0)) %>%
  
  # 3. Impute BMI using *training-set* median
  step_impute_median(bmi) %>%
  
  # 4. One-hot encode ALL categorical predictors (0/1)
  step_dummy(all_nominal_predictors()) %>%
  
  # 5. Scale ONLY numeric columns (age, bmi, glucose) AFTER dummy encoding
  step_normalize(all_of(numeric_vars)) %>%
  
  step_upsample(stroke , over_ratio = 1)
  
  # 6. Remove zero-variance predictors (safe clean-up)
  step_zv(all_predictors())

logit_model <- logistic_reg() %>%
    set_engine("glm") %>%
    set_mode("classification")
  
stroke_workflow <- workflow() %>%
    add_recipe(stroke_recipe) %>%
    add_model(logit_model)


set.seed(123)  # outputs are reproducible
folds <- vfold_cv(stroke_train, v = 10, strata = stroke)

cv_results <- fit_resamples(
  stroke_workflow,
  resamples = folds,
  metrics = metric_set(roc_auc, accuracy, sensitivity, specificity),
  control = control_resamples(save_pred = TRUE)
)

collect_metrics(cv_results)
final_model <- stroke_workflow %>%
  fit(stroke_train)

test_predictions <- predict(final_model , new_data = stroke_test , type = "prob") %>%
  bind_cols(stroke_test %>% select(stroke))

colnames(test_predictions)

test_predictions <- test_predictions %>%
  mutate(
    pred_class = factor(
      ifelse(.pred_1 >= 0.5, "1", "0"),
      levels = c("0", "1")
    )
  )

roc_auc(test_predictions, truth = stroke, .pred_1)
conf_mat(test_predictions, truth = stroke, estimate = pred_class)
accuracy(test_predictions, truth = stroke, estimate = pred_class)
sensitivity(test_predictions, truth = stroke, estimate = pred_class)
specificity(test_predictions, truth = stroke, estimate = pred_class)

