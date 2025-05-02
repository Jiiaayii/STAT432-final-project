#Random forest
library(randomForest)
library(caret)
library(pROC)
library(ggplot2)
library(dplyr) 

# Convert target variable to factor
df_clean$Depression <- factor(df_clean$Depression, levels = c(0, 1), labels = c("No", "Yes"))

# Split the dataset into training (70%) and testing (30%) sets
set.seed(432)
split_index <- createDataPartition(df_clean$Depression, p = 0.7, list = FALSE)
train_data <- df_clean[split_index, ]
test_data <- df_clean[-split_index, ]

train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

# Train a Random Forest model
set.seed(432)
rf_model <- randomForest(Depression ~ ., data = train_data, ntree = 500, importance = TRUE)

# Predict and evaluate
rf_pred <- predict(rf_model, newdata = test_data)
confusionMatrix(rf_pred, test_data$Depression, positive = "Yes")

# === Improved Variable Importance Plot (Top 5) ===
# Extract and process variable importance
importance_df <- as.data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)

# Top 5 by MeanDecreaseGini
top5 <- importance_df %>%
  arrange(desc(MeanDecreaseGini)) %>%
  head(5)

# Plot top 5
ggplot(top5, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 5 Important Features - Random Forest",
       x = "Feature",
       y = "Mean Decrease in Gini") +
  theme_minimal(base_size = 13)

# === Calculate and Plot ROC Curve ===
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")

# Remove NA if any
roc_df <- data.frame(
  truth = test_data$Depression,
  prob = rf_prob[, "Yes"]
)
roc_df <- na.omit(roc_df)

# Compute and plot ROC
roc_obj <- roc(roc_df$truth, roc_df$prob)
plot(roc_obj, col = "blue", main = "ROC Curve - Random Forest")
auc(roc_obj)

