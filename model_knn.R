## KNN

# Load necessary libraries
library(caret)
library(pROC)
library(precrec)

set.seed(7)

# 1. Define train control for 10-fold CV optimizing for AUC
train_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# 2. Define candidate k values for tuning
k_grid <- data.frame(k = seq(5, 80, by = 2))

# 3. Train the KNN model
knn_model <- train(
  Depression ~ .,
  data = train,
  method = "knn",
  trControl = train_control,
  tuneGrid = k_grid,
  metric = "ROC"
)

# 4. Print optimal k
cat("Optimal k:", knn_model$bestTune$k, "\n")

# 5. Predict on test set
knn_pred_class <- predict(knn_model, newdata = test)
knn_pred_prob <- predict(knn_model, newdata = test, type = "prob")[, "Yes"]

# 6. Confusion Matrix & Metrics
knn_conf_matrix <- confusionMatrix(knn_pred_class, test$Depression, positive = "Yes")
knn_accuracy <- knn_conf_matrix$overall["Accuracy"]
knn_precision <- knn_conf_matrix$byClass["Pos Pred Value"]
knn_recall <- knn_conf_matrix$byClass["Sensitivity"]
knn_f1 <- 2 * (knn_precision * knn_recall) / (knn_precision + knn_recall)


# 7. Compute AUC
knn_roc <- roc(test$Depression, knn_pred_prob)
knn_auc <- pROC::auc(knn_roc)


# 8. Create summary table
knn_metrics_table <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "AUC"),
  Value = c(
    round(knn_accuracy, 4),
    round(knn_precision, 4),
    round(knn_recall, 4),
    round(knn_f1, 4),
    round(knn_auc, 4)
  )
)
print(knn_metrics_table)

# 9. Optional: Plot ROC curve
plot(knn_roc, main = "ROC Curve for Tuned KNN Model", col = "blue", lwd = 2)
library(ggplot2)

# Extract tuning results
tuning_results <- knn_model$results

# Find best k and corresponding AUC
best_k <- knn_model$bestTune$k
best_auc <- max(tuning_results$ROC)

# Plot
ggplot(tuning_results, aes(x = k, y = ROC)) +
  geom_line(color = "steelblue", size = 1.2) +
  geom_point(color = "darkred", size = 2) +
  geom_point(aes(x = best_k, y = best_auc), color = "red", size = 3) +
  geom_text(aes(x = best_k, y = best_auc, 
                label = paste0("K = ", best_k, "\nAUC = ", round(best_auc, 3))),
            hjust = -0.1, vjust = -1, color = "red", size = 4) +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "gray40") +
  geom_hline(yintercept = best_auc, linetype = "dashed", color = "gray40") +
  labs(title = "AUC vs. Number of Neighbors (K)",
       x = "K (Number of Neighbors)",
       y = "AUC") +
  scale_x_continuous(breaks = tuning_results$k) +
  theme_minimal(base_size = 13) +
  theme(panel.grid.minor = element_blank())






