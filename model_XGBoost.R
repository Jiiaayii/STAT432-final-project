## XGBoost
library(xgboost)
set.seed(7)

# Define training control with ROC as metric
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

# Train XGBoost with hyperparameter tuning
xgb_grid <- expand.grid(
  nrounds = c(100, 200, 300,400),
  max_depth = c(3, 6),
  eta = c(0.01, 0.05, 0.1),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

xgb_model <- train(
  Depression ~ .,
  data = train_tree,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  metric = "ROC"
)

# View best parameters
print(xgb_model$bestTune)

ggplot(xgb_model$results, aes(x = nrounds, y = ROC, color = as.factor(max_depth))) +
  geom_line() +
  facet_wrap(~ eta, labeller = label_both) +
  labs(
    title = "ROC vs nrounds (by eta and max_depth)",
    x = "Number of Boosting Rounds (nrounds)",
    y = "ROC AUC",
    color = "Max Depth"
  ) +
  theme_minimal()
```
#### Selection of Parameters

**nrounds**:Number of boosting rounds. Need to find a proper value to avoid overfitting while still giving the model room to learn. The best value is 300.

**max_depth**:The maximum depth of each tree. Controls how complex each individual tree can be. The best value is 3, which keeps trees shallow, reducing risk of overfitting, good for generalization.

**eta**:Learning rate.It controls how much the model adjusts with each boosting round. Smaller values mean slower, more cautious learning. The best value is 0.05, which is low enough to reduce overfitting, but not so low that training becomes very slow. 

**gamma**:Minimum loss reduction required to make a further partition on a leaf node. Acts like a regularization parameter for pruning. We’re not applying additional pruning pressure yet—this gives the trees more freedom.

**colsample_bytree**:Fraction of features used when growing each tree, which set to 0.8 to encourage diversity among trees and help prevent overfitting by not using all features in every tree.

**min_child_weight**: Minimum sum of instance weights (or number of samples) in a child node. Controls model complexity. It sets to 1: Allows the model to create nodes with fewer samples, which can capture rare patterns. Might be adjusted higher if overfitting is observed.

**subsample**: Fraction of training data used for growing each tree. It sets to 0.8: Like colsample_bytree, this helps prevent overfitting and improves generalization by training on slightly different subsets of the data.

```{r}
# Predict on test set
pred_class <- predict(xgb_model, test_tree)
pred_prob <- predict(xgb_model, test_tree, type = "prob")

# Confusion matrix
conf_matrix <- confusionMatrix(pred_class, test_tree$Depression)
print(conf_matrix)

# Extract metrics
accuracy  <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall    <- conf_matrix$byClass["Recall"]
f1_score  <- conf_matrix$byClass["F1"]

# Print all
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1 Score:", round(f1_score, 4), "\n")

# AUC
roc_obj <- roc(response = test_tree$Depression, predictor = pred_prob[,"Yes"])
auc_value <- auc(roc_obj)
cat("AUC on test set:", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, main = "ROC Curve - Test Set", col = "blue", print.auc = TRUE)

# Plot variable importance
varImpPlot <- varImp(xgb_model)
plot(varImpPlot, top = 15)
