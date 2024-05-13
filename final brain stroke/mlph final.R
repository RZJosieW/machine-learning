library(readr)
brain_stroke <- read_csv("Desktop/Machine Learning/brain_stroke.csv")
View(brain_stroke)

#recode
# check the missing value 
sum(is.na(brain_stroke))
#marry
table(brain_stroke$ever_married)
brain_stroke$ever_married <- ifelse(brain_stroke$ever_married == "No", 0, ifelse(brain_stroke$ever_married == "Yes", 1, NA))
table(brain_stroke$ever_married)

#Residence_type
table(brain_stroke$Residence_type)
brain_stroke$Residence_type <- ifelse(brain_stroke$Residence_type == "Urban", 0, ifelse(brain_stroke$Residence_type == "Rural", 1, NA))
table(brain_stroke$Residence_type)

#gender 
table(brain_stroke$gender)
brain_stroke$gender <- ifelse(brain_stroke$gender == "Male", 0, ifelse(brain_stroke$gender == "Female", 1, NA))
table(brain_stroke$gender)

# one_hot encoding for smoking and work type
table(brain_stroke$smoking_status)
table(brain_stroke$work_type )

one_hot_encoded <- model.matrix(~ smoking_status + work_type    - 1, data = brain_stroke)
head(one_hot_encoded)

df2 <- cbind(brain_stroke, one_hot_encoded)
df2_new <- subset(df2, select = -c(work_type, smoking_status))

colnames(df2_new)[colnames(df2_new) == 'smoking_statusformerly smoked'] <- 'smoking_statusformerly_smoked'
colnames(df2_new)[colnames(df2_new) == 'smoking_statusnever smoked'] <- 'smoking_statusnever_smoked'
colnames(df2_new)[colnames(df2_new) == 'work_typeSelf-employed'] <- 'work_typeSelf_employed'


class(brain_stroke$hypertension)



#feature selection - interpret PPT 
library(ggcorrplot)
cor_matrix2 <- data.frame(cor(df2_new))
cor_matrix2
ggcorrplot(cor_matrix2)



#split the data 
library(caret)
set.seed(123)
train_indices<- createDataPartition(df2_new$stroke, p = .7, list = FALSE, times = 1)
train_data <- df2_new[train_indices, ]
test_data <- df2_new[-train_indices, ]
sapply(test_data  , class)

## imblance 
# data imblance 
train_counts <- table(train_data$stroke)
train_counts
barplot(train_counts, 
        main = "Distribution of train_counts",
        xlab = "Classes",
        ylab = "Frequency",
        col = c("blue", "red"),  
        ylim = c(0, max(train_counts) + 2000)) 
midpoints <- barplot(train_counts, plot = FALSE)
text(midpoints, train_counts + 100, labels = train_counts, pos = 3)

test_counts <- table(test_data$stroke)
test_counts
barplot(test_counts, 
        main = "Distribution of test_counts",
        xlab = "Classes",
        ylab = "Frequency",
        col = c("blue", "red"),  
        ylim = c(0, max(test_counts) + 2000)) 
midpoints <- barplot(test_counts, plot = FALSE)
text(midpoints, test_counts + 100, labels = test_counts, pos = 3)

#plot 
hist(train_data$age, main = "Histogram of Age", xlab = "Age", col = "blue", border = "black")
hist(train_data$bmi, main = "Histogram of Age", xlab = "Age", col = "blue", border = "black")
hist(train_data$avg_glucose_level, main = "Histogram of Age", xlab = "Age", col = "blue", border = "black")




#_standardized- not for tree model!!
train_data_standardized <-train_data  
train_data_standardized [, 2] <- scale(train_data [, 2], center = TRUE, scale = TRUE)
train_data_standardized[, 7:8] <- scale(train_data [, 7:8], center = TRUE, scale = TRUE)
head(train_data_standardized )


test_data_standardized <-test_data 
test_data_standardized [, 2] <- scale(test_data [, 2], center = TRUE, scale = TRUE)
test_data_standardized[, 7:8] <- scale(test_data [, 7:8], center = TRUE, scale = TRUE)
head(test_data_standardized )
table(train_data_standardized$stroke)

sapply(test_data_standardized , class)



## how to fix imblance smote? weight on model?
## Smote -- for no standardizaton model 
str(train_data)
summary(train_data$stroke)
library(smotefamily)
smote_output <- SMOTE(train_data, train_data$stroke, K = 5, dup_size = 17.8)
balanced_data <- smote_output$data
balanced_data_new <- subset(balanced_data, select = -c(class))
table(balanced_data_new$stroke)


## standardizaton
smote_output_standardizaton <- SMOTE(train_data_standardized, train_data_standardized$stroke, K = 5, dup_size = 17.8)
balanced_data_standardizaton <- smote_output_standardizaton$data
balanced_data_std_new <- subset(balanced_data_standardizaton, select = -c(class))
table(balanced_data_std_new$stroke)



## XGboost

train_data_xgboost <- as.matrix(balanced_data_new[, -which(names(balanced_data_new) == "stroke")])
train_label <- balanced_data_new$stroke


test_data_xgboost <- as.matrix(test_data[, -which(names(test_data) == "stroke")])
test_label <- test_data$stroke





library(xgboost)

# Create DMatrix for training and testing
dtrain <- xgb.DMatrix(data = train_data_xgboost, label = train_label)
dtest <- xgb.DMatrix(data = test_data_xgboost, label = test_label)

params <- list(
  booster = "gbtree",
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.5,
  colsample_bytree = 0.7,
  objective = "binary:logistic",
  eval_metric = "auc"
)

num_rounds <- 100
watchlist <- list(train = dtrain, test = dtest)
model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = num_rounds,
  watchlist = watchlist,
  early_stopping_rounds = 10,
  verbose = 1  
)
pred_probs <- predict(model, dtest)
pred_class <- ifelse(pred_probs > 0.5, 1, 0)
library(pROC)
conf_matrix <- confusionMatrix(factor(pred_class, levels = c(0, 1)), 
                               factor(test_label, levels = c(0, 1)))

accuracy_xgboost <- conf_matrix$overall['Accuracy']
precision_xgboost <- conf_matrix$byClass['Precision']
recall_xgboost <- conf_matrix$byClass['Sensitivity']


f1_score_xgboost <- 2 * (precision_xgboost * recall_xgboost) / (precision_xgboost + recall_xgboost)

print(accuracy_xgboost)
print(precision_xgboost)
print(recall_xgboost)
print(f1_score_xgboost)

roc_obj_xgboost <- roc(response = factor(test_label, levels = c(0, 1)), predictor = pred_probs)
auc_value_xgboost<- auc(roc_obj_xgboost)
roc_obj_xgboost <- roc(response = factor(test_label, levels = c(0, 1)), predictor = pred_probs)
auc_value_xgboost <- auc(roc_obj_xgboost)
print(auc_value_xgboost)

#shap plot
library(shap)
shap_values <- predict(model, dtrain, predcontrib = TRUE)

library(shapviz)
shp <- shapviz(model, train_data_xgboost )
sv_importance(shp, kind = "beeswarm")
sv_importance(shp)
sv_dependence(shp, train_data_xgboost)
sv_force(shp, row_id = 1)



## Random forest
library(randomForest)
balanced_data_new$stroke <- as.factor(balanced_data_new$stroke )
test_data$stroke <- as.factor(test_data$stroke)

# RF-Train 
rf_train <- randomForest(stroke ~ ., data = balanced_data_new, ntree = 500, importance = TRUE)
rf_predictions <- predict(rf_train, newdata = test_data)
rf_predictions_auc <- predict(rf_train, newdata = test_data, type = "prob")[,2]
varImpPlot(rf_train, type = 2, main = "Variable Importance in Random Forest Model")
conf_matrix_rf<- confusionMatrix(rf_predictions, test_data$stroke)
print(conf_matrix_rf)
accuracy_rf <- conf_matrix_rf$overall['Accuracy']
precision_rf <- conf_matrix_rf$byClass['Precision']
recall_rf <- conf_matrix_rf$byClass['Sensitivity']
f1_score_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
print(accuracy_rf)
print(precision_rf)
print(recall_rf)
print(f1_score_rf)
roc_obj_rf <- roc(test_data$stroke, rf_predictions_auc)
auc_value_rf <- auc(roc_obj_rf)

#KNN
#knn
train_data <- balanced_data_std_new[, -which(names(balanced_data_std_new) == "stroke")]
train_labels <- factor(balanced_data_std_new$stroke)
test_data_without_stroke <- test_data_standardized[, -which(names(test_data_standardized) == "stroke")]
test_labels <- factor(test_data_standardized$stroke)

y_pre <- knn(train = train_data, 
             test = test_data_without_stroke, 
             cl = train_labels, 
             k = k)

y_pre <- factor(y_pre, levels = levels(test_labels))

c_m <- confusionMatrix(data = y_pre, reference = test_labels)
print(c_m)

accuracy_knn <- c_m$overall['Accuracy']
precision_knn <- c_m$byClass['Precision']
recall_knn <- c_m$byClass['Sensitivity']
f1_score_knn <- 2 * (precision_knn * recall_knn) / (precision_knn + recall_knn)
print(paste("Accuracy:", accuracy_knn))
print(paste("Precision:", precision_knn))
print(paste("Recall:", recall_knn))
print(paste("F1 Score:", f1_score_knn))

knn_prob <- as.numeric(y_pre == "1")  
roc_knn <- roc(response = as.numeric(test_labels), predictor = knn_prob)
auc_knn <- auc(roc_knn)

k_values <- seq(1,20, by = 5) 

train_errors <- numeric(length(k_values))
test_errors <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  knn_train_pred <- knn(train_data_knn, train_data_knn, train_label, k = k)
  knn_test_pred <- knn(train_data_knn, test_data_knn, train_label, k = k)
  
  train_conf_matrix <- table(Predicted = knn_train_pred, Actual = train_label)
  test_conf_matrix <- table(Predicted = knn_test_pred, Actual = test_label)
  
  train_accuracy <- sum(diag(train_conf_matrix)) / sum(train_conf_matrix)
  test_accuracy <- sum(diag(test_conf_matrix)) / sum(test_conf_matrix)
  
  train_errors[i] <- 1 - train_accuracy
  test_errors[i] <- 1 - test_accuracy
}
error_data <- data.frame(
  k = k_values,
  TrainError = train_errors,
  TestError = test_errors
)
error_plot <- ggplot(error_data, aes(x = k)) +
  geom_line(aes(y = TrainError, colour = "Train Error")) +
  geom_point(aes(y = TrainError, colour = "Train Error"), size = 2) +  
  geom_line(aes(y = TestError, colour = "Test Error")) +
  geom_point(aes(y = TestError, colour = "Test Error"), size = 2) +  
  labs(title = "KNN Train vs Test Error", x = "Number of Neighbors (k)", y = "Error Rate") +
  scale_colour_manual(values = c("Train Error" = "blue", "Test Error" = "red")) +
  theme_minimal()
error_plot


#SVM
library(e1071)
library(caret)
library(pROC)

classifier <- svm(formula = stroke ~ ., 
                  data = balanced_data_std_new,  
                  type = 'C-classification', 
                  kernel = 'linear')

test_data_without_stroke <- test_data_standardized[, -which(names(test_data_standardized) == "stroke")]
y_pre <- predict(classifier, newdata = test_data_without_stroke)
test_labels <- factor(test_data_standardized[, "stroke"])
y_pre <- factor(y_pre, levels = levels(test_labels))


c_m <- confusionMatrix(data = y_pre, reference = test_labels)
print(c_m)

accuracy_svm <- c_m$overall['Accuracy']
precision_svm <- c_m$byClass['Precision']
recall_svm <- c_m$byClass['Sensitivity']
f1_score_svm <- 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm)
print(paste("Accuracy:", accuracy_svm))
print(paste("Precision:", precision_svm))
print(paste("Recall:", recall_svm))
print(paste("F1 Score:", f1_score_svm))

svm_scores <- predict(classifier, newdata = test_data_without_stroke, decision.values = TRUE)
decision_values <- attributes(svm_scores)$decision.values
roc_svm <- roc(response = as.numeric(test_labels), predictor = decision_values)
auc_svm <- auc(roc_svm)
plot(roc_svm, main="ROC Curve for SVM")
print(paste("AUC for SVM:", auc_svm))






#logistic Regression
logistic_model <- glm(stroke ~ . - smoking_statusUnknown, family = binomial(), data = balanced_data_std_new)
summary(logistic_model)
logi_pred <- predict(logistic_model, type = "response")
roc_logistic <- roc(balanced_data_std_new$stroke,logi_pred )
auc_logi <- auc(roc_logistic)
auc_logi
pred_classes <- ifelse(logi_pred > 0.5, 1, 0)
accuracy <- sum(pred_classes == balanced_data_std_new$stroke) / nrow(balanced_data_std_new)
print(paste("Accuracy:", accuracy))

## lasso
library(glmnet)
library(pROC)  
x <- model.matrix(stroke ~ . - 1, data = balanced_data_std_new)  
summary(x)
y <- balanced_data_std_new$stroke
set.seed(123)  
cv_model <- cv.glmnet(x, y, alpha = 1, family = "binomial")  
cv_model
lasso_pred <- predict(cv_model, newx = x, type = "response", s = cv_model$lambda.min)
roc_lasso <- roc(response = y, predictor = as.vector(lasso_pred))  # Ensure predictors are in correct format
auc_lasso <- auc(roc_lasso)
print(auc_lasso)


## plot
rocobjs <- list(
  XGboost = roc_obj_xgboost,
  RandomForest = roc_obj_rf,
  SVM = roc_svm,
  Logistic_Regression = roc_logistic,
  Lasso_Regression = roc_lasso,
  Deep_Neural_Network = roc_curve_DNN
)

methods_auc <- paste(
  c("XGboost", "RandomForest", "KNN","SVM", "Logistic Regression", "Lasso Regression", "Deep Neural Network"),
  "AUC =",
  round(c(auc(roc_obj_xgboost), auc(roc_obj_rf), auc(roc_knn),auc(roc_svm), auc(roc_logistic), auc(roc_lasso), auc(roc_curve_DNN)), 3)
)

ggplot_obj <- ggroc(rocobjs, size = 2, alpha = 0.5) +
  scale_color_discrete(labels = methods_auc) +
  theme_minimal() +  
  theme(
    plot.title = element_text(size = 15),
    legend.title = element_blank(), 
    legend.text = element_text(size = 10)
  )

print(ggplot_obj)


## deep neutral network 
train_indices<- createDataPartition(df2_new$stroke, p = .7, list = FALSE, times = 1)
train_data <- df2_new[train_indices, ]
test_data <- df2_new[-train_indices, ]

library(caret)

scale01 <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
train_data_normalized <- train_data
train_data_normalized[, 2] <- scale01(train_data[, 2])
train_data_normalized[, 7:8] <- apply(train_data[, 7:8], 2, scale01)

head(train_data_normalized)

test_data_normalized <- test_data
test_data_normalized[, 2] <- scale01(test_data[, 2])
test_data_normalized[, 7:8] <- apply(test_data[, 7:8], 2, scale01)

head(test_data_normalized)

table(train_data_normalized$stroke)

#smote 
library(smotefamily)
smote_output_nor <- SMOTE(train_data_normalized, train_data_normalized$stroke, K = 5, dup_size = 17.8)
balanced_data_normalized <- smote_output_nor$data
balanced_data_nor_new <- subset(balanced_data_normalized, select = -c(class))
table(balanced_data_normalized$stroke)

library(keras)

X_train <- as.matrix(subset(balanced_data_nor_new, select = -stroke))
y_train <-balanced_data_nor_new$stroke
X_test <- as.matrix(subset(test_data_normalized, select = -stroke))
y_test <- test_data_normalized$stroke

ncol(X_train)
ncol(X_test)
input_shape <- ncol(X_train)

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = input_shape) %>%
  layer_dropout(rate = 0.2) %>%
 layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid')








model %>% compile(
  optimizer = optimizer_adam(),
  loss = 'binary_crossentropy',
  metrics = 'accuracy'
)


history <- model %>% fit(
  X_train, y_train,
  epochs = 50, batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)

predictions <- model %>% predict(X_test)

roc_curve_DNN <- roc(y_test, predictions)
auc_value_DNN <- auc(roc_curve_DNN)
print(paste("AUC:", auc_value_DNN))

plot(roc_curve_DNN, main = "ROC Curve for Deep Neural Network")

predictions_binary <- ifelse(predictions > 0.5, 1, 0)

predictions_binary <- factor(predictions_binary, levels = c(0, 1))
y_test <- factor(y_test, levels = c(0, 1))

c_m <- confusionMatrix(predictions_binary, y_test)
print(c_m)

accuracy_dnn <- c_m$overall['Accuracy']
precision_dnn <- c_m$byClass['Precision']
recall_dnn <- c_m$byClass['Sensitivity']
f1_score_dnn <- 2 * (precision_dnn * recall_dnn) / (precision_dnn + recall_dnn)

print(paste("Accuracy:", accuracy_dnn))
print(paste("Precision:", precision_dnn))
print(paste("Recall:", recall_dnn))
print(paste("F1 Score:", f1_score_dnn))


