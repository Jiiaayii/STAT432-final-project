
## EDA

# load libraries
library(tidyverse)    
library(data.table)   
library(knitr) 
library(dplyr)
library(stringr)
library(ggplot2)     
library(reshape2)      
library(RColorBrewer)  
library(gridExtra) 
library(corrplot)
library(caret)
library(MLmetrics)
library(pROC)
library(PRROC)
library(glmnet)
library(tidyr)

# 1 Statistical Summary for Numerical Variables
cat("Statistical Summary for Numerical Variables:\n")
print(summary(df_final_noid[, num_vars]))

# Melt numeric data for ggplot2
df_num_melt <- melt(df_final_noid[, num_vars], measure.vars = num_vars)

# Define color palette
set2_color <- brewer.pal(8, "Set2")
num_fill_colors <- rep(set2_color, length.out = length(num_vars))

# Create histogram plot using facets
ggplot(df_num_melt, aes(x = value, fill = variable)) +
  geom_histogram(bins = 30, color = "white", alpha = 0.9) +
  facet_wrap(~variable, scales = "free", ncol = 3) +
  scale_fill_manual(values = num_fill_colors) +
  labs(title = "Histograms of Numeric Variables", x = NULL, y = "Count") +
  theme_minimal(base_size = 13) +
  theme(panel.grid.major = element_line(color = "#E0E0E0"),
        panel.grid.minor = element_blank(),
        strip.background = element_rect(fill = "#F9F9F9", color = NA),
        strip.text = element_text(face = "bold", color = "#333333"),
        plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

# 2 Histogram plot for categorical variables
# Highlight and plot the outcome variable Depression (only once)
ggplot(df_final_noid, aes(x = Depression, fill = Depression)) +
  geom_bar(color = "black") +
  scale_fill_manual(values = c("No" = "#66c2a5", "Yes" = "#fc8d62")) +
  labs(title = "Distribution of Depression", x = "Depression", y = "Count") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")


# Remove Depression from the categorical variable list to avoid repetition
cat_vars_filtered <- setdiff(cat_vars_filtered, "Depression")

# Use unified Set2 color palette for visual consistency
main_palette <- brewer.pal(8, "Set2")
count_plots <- list()

for (i in seq_along(cat_vars_filtered)) {
  var <- cat_vars_filtered[i]
  n_levels <- length(unique(df_final_noid[[var]]))
  
  # Rotate starting color to reduce color duplication across plots
  offset <- (i - 1) %% length(main_palette) + 1
  cur_colors <- c(main_palette[offset:length(main_palette)], main_palette[1:(offset - 1)])
  cur_colors <- rep(cur_colors, length.out = n_levels)
  
  # Create individual count plot
  p <- ggplot(df_final_noid, aes(x = .data[[var]], fill = .data[[var]])) +
    geom_bar(color = "black") +
    labs(x = var, y = "Count") +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      axis.title = element_text(size = 9),
      plot.title = element_blank(),
      legend.position = "none"
    ) +
    scale_fill_manual(values = cur_colors)
  
  count_plots[[var]] <- p
}
# Display the remaining categorical variable plots in a 2x2 layout per page
count_pages <- marrangeGrob(grobs = count_plots, nrow = 2, ncol = 2, top = NULL)
invisible(print(count_pages))

## 3 create correlation heatmap
# extract numeric data
numeric_data <- df_final_noid[, num_vars]
# compute Pearson correlation matrix
cor_matrix <- cor(numeric_data, method = "pearson")

# plot heatmap using corrplot
corrplot(cor_matrix,method = "color", type = "upper",               
         col = colorRampPalette(c("blue", "white", "red"))(200),  
         addCoef.col = "black",tl.col = "black",tl.srt = 45,                  
         number.cex = 0.8, diag = TRUE, mar = c(0, 0, 1, 0)) 

## 4 boxplot for outliers
# Define numerical variables for the depression dataset
num_vars <- c("Age", "Academic.Pressure", "CGPA", "Study.Satisfaction",  
              "Work.Study.Hours")

# Melt the dataframe for ggplot
df_melted <- melt(df_final_noid[, num_vars])

# Define a Set2 color palette
set2_color <- brewer.pal(8, "Set2")

# Draw boxplot
# Boxplot for Age only
ggplot(df_melted[df_melted$variable == "Age", ], aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 1.5) +
  scale_fill_manual(values = c("Age" = "#66c2a5")) +
  labs(title = "Boxplot of Age", x = "Variable", y = "Value") +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 0),
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none"
  )

# Boxplot for the rest (excluding Age)
ggplot(df_melted[df_melted$variable != "Age", ], aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 1.5) +
  scale_fill_manual(values = rep(set2_color, length.out = length(num_vars) - 1)) +
  labs(title = "Boxplots of Other Numerical Variables", x = "Variable", y = "Value") +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none"
  )

## Relationship bewteen depression and other variables
depression_colors <- c("Yes" = "#90a4ae", "No" = "#fdd835")

# pic 1: Suicidal Thoughts vs Depression
p1 <- ggplot(df_final_tree, aes(x = Have.you.ever.had.suicidal.thoughts.., fill = Depression)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = depression_colors) +
  labs(title = "Suicidal Thoughts", x = "Response", y = "Proportion") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

# pic 2: Academic Pressure vs Depression
p2 <- ggplot(df_final_tree, aes(x = Depression, y = Academic.Pressure, fill = Depression)) +
  geom_boxplot() +
  scale_fill_manual(values = depression_colors) +
  labs(title = "Academic Pressure", x = "Depression", y = "Academic Pressure") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

# pic 3: Financial Stress vs Depression
p3 <- ggplot(df_final_tree, aes(x = Financial.Stress, fill = Depression)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = depression_colors) +
  labs(title = "Financial Stress", x = "Financial Stress Level", y = "Proportion") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

grid.arrange(p1, p2, p3, ncol = 3)

# extract numeric variable
num_vars <- c("Age", "CGPA", "Work.Study.Hours", "Academic.Pressure", "Study.Satisfaction")
df_melted <- melt(df_final[, num_vars])

# boxplot after normalization and standardizarion
ggplot(df_melted, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 16, outlier.size = 1.5) +
  labs(title = "Boxplots of Scaled Variables", x = "Variable", y = "Value (Standardized/Normalized)") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

