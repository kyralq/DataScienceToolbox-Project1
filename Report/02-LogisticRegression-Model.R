library("tidyverse")
library("tidymodels")
library("rlang")
library("corrplot")
library("GGally")
library("patchwork")
library("recipes")
library(themis)
library(PRROC)

stroke_train <- read.csv("Data/stroke_training_dataset.csv")
stroke_test <-  read.csv("Data/stroke_testing_dataset.csv")

head(stroke_train)
summary(stroke_train)
str(stroke_train)

head(stroke_test)
summary(stroke_test)
str(stroke_test)

#Remove Identity Column
stroke_train <- stroke_train[ , 2:12]
stroke_test <- stroke_test[ , 2:12]

table(stroke_train$stroke)
prop.table(table(stroke_train$stroke))

table(stroke_test$stroke)
prop.table(table(stroke_test$stroke))

####################
#Exploratory Data Analysis
####################

numeric_vars <- c("age" , "bmi" , "avg_glucose_level")
summary(stroke_train[ , numeric_vars])
# Plot histograms of numerical variables
plots <- lapply(numeric_vars, function(var){
  ggplot(stroke_train, aes_string(x = var)) +
    geom_histogram(bins = 30, fill = "blue", color = "black") +
    theme_minimal() +
    ggtitle(paste(var, "distribution"))
})

print(wrap_plots(plots, ncol = 3))

categorical_vars <- c("gender", "ever_married", "work_type", 
                      "Residence_type", "smoking_status", "heart_disease" , "hypertension")
#Convert categorical data to numerical 
stroke_train <- stroke_train %>%
  mutate(
    gender = factor(gender),
    hypertension = factor(hypertension) , 
    ever_married = factor(ever_married),
    work_type = factor(work_type),
    heart_disease = factor(heart_disease) , 
    Residence_type = factor(Residence_type),
    smoking_status = factor(smoking_status)
  )
#Look at counts of categorical data
stroke_train %>% 
  select(categorical_vars) %>%
  map(~table(.)) 

#Generate bar charts
for (var in categorical_vars) {
  p <- ggplot(stroke_train , aes_string(x = var)) +
    geom_bar() +
    ggtitle(paste("Bar plot of" , var))
  print(p)
}

#I want to look at the proportional stroke rate by each covariate

#By Age:
age_proportion <- ggplot(stroke_train, aes(age, fill = factor(stroke))) +
      geom_histogram(position = "identity", binwidth = 3 ,  alpha = 0.5) + 
      ylab("Proportion") +
      ggtitle("Proportion of Strokes by Age")

age_scatter <- ggplot(stroke_train, aes(age, stroke)) +
  geom_jitter(alpha = 0.2) + 
  ggtitle("Scatter Stroke vs Age")

age_density <- ggplot(stroke_train, aes(age, fill = factor(stroke))) +
  geom_density(alpha = 0.5)

#By BMI

bmi_proportion <- ggplot(stroke_train, aes(bmi, fill = factor(stroke))) +
      geom_histogram(position = "identity", binwidth = 3 ,  alpha = 0.5) + 
      ylab("Proportion") +
      ggtitle("Proportion of Strokes by BMI")

bmi_scatter <- ggplot(stroke_train, aes(bmi, stroke)) +
  geom_jitter(alpha = 0.2) + 
  ggtitle("Scatter Stroke vs BMI")

bmi_density <- ggplot(stroke_train, aes(bmi, fill = factor(stroke))) +
  geom_density(alpha = 0.5)

#By Average Glucose Level

glucose_proportion <- ggplot(stroke_train, aes(avg_glucose_level, fill = factor(stroke))) +
  geom_histogram(position = "identity", binwidth = 3 ,  alpha = 0.5) + 
  xlab("Average Glucose Level") +
  ylab("Proportion") +
  ggtitle("Proportion of Strokes by Glucose")

glucose_scatter <- ggplot(stroke_train, aes(avg_glucose_level, stroke)) +
  geom_jitter(alpha = 0.2) + 
  ggtitle("Scatter Stroke vs Glucose")

glucose_density <- ggplot(stroke_train, aes(avg_glucose_level, fill = factor(stroke))) +
  geom_density(alpha = 0.5)

#Compare numeric covariates

print(wrap_plots(age_proportion , bmi_proportion , glucose_proportion , ncol = 3))
print(wrap_plots(age_scatter , bmi_scatter , glucose_scatter , ncol = 3))
print(wrap_plots(age_density , bmi_density , glucose_density , ncol = 3))

#Look at proportional stroke rates within categorical covariates. This prints all bar graphs of
#categorical covariates divided by proportion of strokes

stroke_train$stroke <- factor(stroke_train$stroke)

for (var in categorical_vars) {
  prop_chart <- ggplot(stroke_train, aes_string(x = var , fill = "stroke")) +
    geom_bar(position = "fill") +
    ggtitle(paste("Proportion of strokes by" , var))
  print(prop_chart)
}

##########
#DATA CLEANING
##########
#Clean before looking at correlations
#Create a recipe to apply the same cleaning to both training and testing data

#Missing values:
colSums(is.na(stroke_train))

stroke_recipe <- recipe(stroke ~ ., data = stroke_train) %>%
  step_mutate(across(all_of(categorical_vars), as.factor)) %>%
  step_mutate(bmi_imputed_flag = ifelse(is.na(bmi), 1, 0)) %>%
  step_impute_median(bmi) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_of(numeric_vars)) 

prep_recipe <- prep(stroke_recipe , training = stroke_train)

train_processed <- bake(prep_recipe , new_data = stroke_train)

test_processed <- bake(prep_recipe , new_data = stroke_test)

#Make sure recipe has worked
colSums(is.na(train_processed))
colSums(is.na(test_processed))

summary(train_processed)
summary(test_processed)

train_processed$stroke <- factor(train_processed$stroke, levels = c(0,1), labels = c("No", "Yes"))
test_processed$stroke <- factor(test_processed$stroke, levels = c(0,1), labels = c("No", "Yes"))
#look at outliers

stroke_train <- stroke_train %>%
  mutate(bmi = ifelse(bmi > 70, 70, bmi))


######
#Fit initial model
######

full_model <- glm(stroke ~ ., family = binomial , data = train_processed)
summary(full_model)

car::vif(full_model)

#use tidymodels pipeline to make it easier
library(caret)

set.seed(42)
cv_control <- trainControl(method = "cv" , number = 5 , classProbs = TRUE , summaryFunction = twoClassSummary)
full_cv <- train( stroke ~., data = train_processed , method = "glm" , family = binomial , metric = "ROC" , trControl = cv_control)
full_cv$results

library(PRROC)

pred_probs <- predict(full_cv, newdata = train_processed, type = "prob")$Yes

# PR AUC
pr_obj <- pr.curve(
  scores.class0 = pred_probs[train_processed$stroke == "Yes"],  # positives
  scores.class1 = pred_probs[train_processed$stroke == "No"],   # negatives
  curve = TRUE
)
pr_obj$auc.integral
plot(pr_obj)

# PR AUC value of 0.2344 - recall issues with rare encoded factors

# remove factors that are rare/ not statistically significant

train_reduced <- train_processed %>% 
  select(-gender_Other,-gender_Male , , -Residence_type_Urban , -ever_married_Yes , -work_type_Never_worked)

reduced_model <- glm(stroke ~ ., family = binomial, data = train_reduced)
summary(reduced_model)

interaction_model <- glm(
  stroke ~ . + age:heart_disease_X1,
  family = binomial,
  data = train_reduced
)
summary(interaction_model)

set.seed(42)
cv_control <- trainControl(method = "cv" , number = 5 , classProbs = TRUE , summaryFunction = twoClassSummary)
full_cv <- train( stroke ~., data = train_reduced , method = "glm" , family = binomial , metric = "ROC" , trControl = cv_control)
full_cv$results

pred_probs <- predict(full_cv, newdata = train_reduced, type = "prob")$Yes

# PR AUC

pr_obj <- pr.curve(
  scores.class0 = pred_probs[train_reduced$stroke == "Yes"],  # positives
  scores.class1 = pred_probs[train_reduced$stroke == "No"],   # negatives
  curve = TRUE
)
pr_obj$auc.integral
plot(pr_obj)

# what happens if we remove BMI 
train_reduced <-  train_reduced %>% 
  select(-bmi , -bmi_imputed_flag)

reduced_model <- glm(stroke ~ ., family = binomial, data = train_reduced)
summary(reduced_model)

set.seed(42)
cv_control <- trainControl(method = "cv" , number = 5 , classProbs = TRUE , summaryFunction = twoClassSummary)
full_cv <- train( stroke ~., data = train_reduced , method = "glm" , family = binomial , metric = "ROC" , trControl = cv_control)
full_cv$results

pred_probs <- predict(full_cv, newdata = train_reduced, type = "prob")$Yes

pr_obj <- pr.curve(
  scores.class0 = pred_probs[train_reduced$stroke == "Yes"],  # positives
  scores.class1 = pred_probs[train_reduced$stroke == "No"],   # negatives
  curve = TRUE
)
pr_obj$auc.integral
plot(pr_obj)

# worse - keep bmi in

# Instead of manually seeing what works - create a function to compute PR AUC

compute_pr_auc <- function(model, data, outcome = "stroke") {
  
  pred_probs <- predict(model, newdata = data, type = "response")  # glm probabilities
  pr_obj <- pr.curve(
    scores.class0 = pred_probs[data[[outcome]] == "Yes"],  # positives
    scores.class1 = pred_probs[data[[outcome]] == "No"],   # negatives
    curve = FALSE
  )
  return(pr_obj$auc.integral)
}

predictors <- setdiff(names(train_processed), "stroke")
pr_auc_results <- data.frame(variable = character(), PR_AUC = numeric(), stringsAsFactors = FALSE)

full_model <- glm(stroke ~ ., family = binomial, data = train_processed)
baseline_pr <- compute_pr_auc(full_model, train_processed)
pr_auc_results <- rbind(pr_auc_results, data.frame(variable = "ALL", PR_AUC = baseline_pr))

for (var in predictors) {
  formula_str <- paste("stroke ~", paste(setdiff(predictors, var), collapse = " + "))
  model <- glm(as.formula(formula_str), family = binomial, data = train_processed)
  pr_val <- compute_pr_auc(model, train_processed)
  pr_auc_results <- rbind(pr_auc_results, data.frame(variable = var, PR_AUC = pr_val))
}

pr_auc_results <- pr_auc_results %>% arrange(desc(PR_AUC))
pr_auc_results

# Remove the variables that increase PR AUC when dropped (have a higher number next to them)
#opposite for interactions - only keep those with higher pr auc

interactions <- c("age:hypertension_X1", "age:heart_disease_X1", "age:avg_glucose_level")

for (inter in interactions) {
  formula_str <- paste("stroke ~ . +", inter)
  model <- glm(as.formula(formula_str), family = binomial, data = train_processed)
  pr_val <- compute_pr_auc(model, train_processed)
  cat(inter, ": PR AUC =", pr_val, "\n")
}

train_reduced <- train_processed %>%
  select(-work_type_Govt_job , work_type_Private , avg_glucose_level , work_type_Self.employed , hypertension_X1, -gender_Other, -work_type_Never_worked)

reduced_model <- glm(stroke ~ ., family = binomial, data = train_reduced)
summary(reduced_model)

set.seed(42)
cv_control <- trainControl(method = "cv" , number = 5 , classProbs = TRUE , summaryFunction = twoClassSummary)
full_cv <- train( stroke ~., data = train_reduced , method = "glm" , family = binomial , metric = "ROC" , trControl = cv_control)
full_cv$results

pred_probs <- predict(full_cv, newdata = train_reduced, type = "prob")$Yes

pr_obj <- pr.curve(
  scores.class0 = pred_probs[train_reduced$stroke == "Yes"],  # positives
  scores.class1 = pred_probs[train_reduced$stroke == "No"],   # negatives
  curve = TRUE
)
pr_obj$auc.integral
plot(pr_obj)
# better!
