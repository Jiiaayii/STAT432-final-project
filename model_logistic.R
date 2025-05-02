## Logistical regression 

# set seed
set.seed(7)

# create training and testing set
x_train <- model.matrix(Depression ~ ., data = train)[, -1]
y_train <- ifelse(train$Depression == "Yes", 1, 0)
x_test <- model.matrix(Depression ~ ., data = test)[, -1]
y_test <- test$Depression

# fit LASSO logistic model with CV=10, optimizing for AUC
lasso_logistic_cv <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", 
                               type.measure = "auc")
plot(lasso_logistic_cv, main = "", label = TRUE)
mtext("LASSO Logistic Regression (AUC vs. log(lambda))", side = 3, line = 2, 
      cex = 1.2)

# extract best lambda
lambda_lasso <- lasso_logistic_cv$lambda.min
cat("Best Lambda for LASSO Logistic:", lambda_lasso, "\n")

# train the final model
lasso_logistic_model <- glmnet(x_train, y_train, alpha = 1, lambda = lambda_lasso, 
                               family = "binomial")

# predict on test set
lasso_logistic_probs <- predict(lasso_logistic_model, newx = x_test, type = "response")
lasso_logistic_preds <- factor(ifelse(lasso_logistic_probs > 0.5, "Yes", "No"), 
                               levels = c("No", "Yes"))

# compute model evaluation metrics
lasso_logistic_conf <- confusionMatrix(lasso_logistic_preds, y_test)
lasso_logistic_acc <- lasso_logistic_conf$overall["Accuracy"]
lasso_logistic_prec <- lasso_logistic_conf$byClass["Pos Pred Value"]
lasso_logistic_rec <- lasso_logistic_conf$byClass["Sensitivity"]
lasso_logistic_f1 <- 2 * lasso_logistic_prec * lasso_logistic_rec / 
  (lasso_logistic_prec + lasso_logistic_rec)

# calculate AUC and ROC
lasso_logistic_roc <- roc.curve(
  scores.class0 = as.numeric(lasso_logistic_probs),
  weights.class0 = ifelse(y_test == "Yes", 1, 0),
  curve = TRUE)
lasso_logistic_auc <- lasso_logistic_roc$auc

# display the model performance table
lasso_logistic_table <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "AUC"),
  Value = c(lasso_logistic_acc,lasso_logistic_prec,lasso_logistic_rec,
            lasso_logistic_f1,lasso_logistic_auc))
kable(lasso_logistic_table, digits = 6)

# plot ROC curve
plot(lasso_logistic_roc, main = "ROC Curve for LASSO Logistic Model")

# set seed
set.seed(7)

# fit Ridge logistic model with CV=10, optimizing for AUC
ridge_logistic_cv <- cv.glmnet(x_train, y_train, alpha = 0, family = "binomial", 
                               type.measure = "auc")
plot(ridge_logistic_cv, main = "", cex.main = 1.2)  
mtext("Ridge Logistic Regression (AUC vs. log(lambda))", side = 3, line = 2, 
      cex = 1.2)


# extract best lambda
lambda_ridge <- ridge_logistic_cv$lambda.min
cat("Best Lambda for Ridge Logistic:", lambda_ridge, "\n")

# train the final model
ridge_logistic_model <- glmnet(x_train, y_train, alpha = 0, lambda = lambda_ridge, 
                               family = "binomial")

# predict on test set
ridge_logistic_probs <- predict(ridge_logistic_model, newx = x_test, type = "response")
ridge_logistic_preds <- factor(ifelse(ridge_logistic_probs > 0.5, "Yes", "No"), 
                               levels = c("No", "Yes"))

# compute model evaluation metrics
ridge_logistic_conf <- confusionMatrix(ridge_logistic_preds, y_test)
ridge_logistic_acc <- ridge_logistic_conf$overall["Accuracy"]
ridge_logistic_prec <- ridge_logistic_conf$byClass["Pos Pred Value"]
ridge_logistic_rec <- ridge_logistic_conf$byClass["Sensitivity"]
ridge_logistic_f1 <- 2 * ridge_logistic_prec * ridge_logistic_rec / 
  (ridge_logistic_prec + ridge_logistic_rec)

# calculate AUC and ROC
ridge_logistic_roc <- roc.curve(
  scores.class0 = as.numeric(ridge_logistic_probs),
  weights.class0 = ifelse(y_test == "Yes", 1, 0),
  curve = TRUE)
ridge_logistic_auc <- ridge_logistic_roc$auc

# display the model performance table
ridge_logistic_table <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "AUC"),
  Value = c(ridge_logistic_acc, ridge_logistic_prec, ridge_logistic_rec,
            ridge_logistic_f1, ridge_logistic_auc))
kable(ridge_logistic_table, digits = 6)

# plot ROC curve
plot(ridge_logistic_roc, main = "ROC Curve for Ridge Logistic Model")

# Create data frames with model labels
lasso_logistic_table <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "AUC"),
  Value = c(lasso_logistic_acc, lasso_logistic_prec, lasso_logistic_rec,
            lasso_logistic_f1, lasso_logistic_auc),
  Model = "LASSO"
)

ridge_logistic_table <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "AUC"),
  Value = c(ridge_logistic_acc, ridge_logistic_prec, ridge_logistic_rec,
            ridge_logistic_f1, ridge_logistic_auc),
  Model = "Ridge"
)

# Combine into one table
combined_logistic_table <- rbind(lasso_logistic_table, ridge_logistic_table)

# Reshape to wide format for side-by-side comparison
wide_table <- pivot_wider(combined_logistic_table, names_from = Model, values_from = Value)

# Display table
kable(wide_table, digits = 6)