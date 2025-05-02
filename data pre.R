## Data Preprocessing

# load data
df <- read.csv("~/Desktop/STAT432/final project/student_depression_dataset.csv",
               stringsAsFactors = TRUE)
dim(df)
head(df)

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

# Data cleaning 

## Check for missing and duplicated values
cat("Number of missing values:", sum(is.na(df)), "\n")
cat("Number of duplicated rows:", sum(duplicated(df)), "\n")

## Convert outcome variable to factor
df$Depression <- as.factor(df$Depression)

## step 1 Handling ambiguous values
# clean 'City' column and create cleaned dataset
df_city_clean <- df %>%
  mutate(
    City = as.character(City),
    City = trimws(City),
    City = case_when(
      City == "'Less Delhi'" ~ "Delhi",                     # Single-quote variant
      City == "'Less than 5 Kalyan'" ~ "Kalyan",            # Single-quote variant
      City == "Khaziabad" ~ "Ghaziabad",                    # Misspelled city
      # Mark invalid or suspicious entries as 'Unknown'
      City %in% c("City", "3.0", "Gaurav", "Mihir", "Nandini", 
                  "Nalini", "Nalyan", "Bhavna", "Kibara", "Harsh", 
                  "Harsha", "M.Com", "M.Tech", "ME", "Mira", 
                  "Rashi", "Reyansh", "Saanvi", "Vaanya") ~ "Unknown",
      TRUE ~ City
    )
  ) %>%
  filter(City != "Unknown") %>%         # Remove all 'Unknown' rows
  mutate(City = as.factor(City))        # Convert back to factor

# Clean remaining invalid entries from other categorical variables
df_clean <- df_city_clean %>%
  filter(as.character(Financial.Stress) != "?",
         `Sleep.Duration` != "Others",
         `Dietary.Habits` != "Others",
         Degree != "Others") %>%
  mutate(
    Financial.Stress = droplevels(as.factor(Financial.Stress)),
    `Sleep.Duration` = droplevels(as.factor(`Sleep.Duration`)),
    `Dietary.Habits` = droplevels(as.factor(`Dietary.Habits`)),
    Degree = droplevels(as.factor(Degree)))

## step 2: Variable Filtering and Scope Restriction
### Records with non-zero Work.Pressure
nonzero_wp <- df_clean %>% 
  filter(Work.Pressure != 0)

###  Records with non-zero Job.Satisfaction
nonzero_js <- df_clean %>% 
  filter(Job.Satisfaction != 0)

###  Records with Profession not equal to 'Student'
non_student_prof <- df_clean %>% 
  filter(Profession != "Student")

# filter to keep only student records, and remove Work.Pressure, Job.Satisfaction, and Profession columns
df_final <- df_clean %>%
  filter(Profession == "Student") %>%   # Only select students
  select(-Work.Pressure, -Job.Satisfaction, -Profession)  
# remove the id column from data_final
df_final_noid <- df_final %>% select(-id)
df_final_noid$Depression <- as.character(df_final_noid$Depression)
df_final_noid$Depression[df_final_noid$Depression == "0"] <- "No"
df_final_noid$Depression[df_final_noid$Depression == "1"] <- "Yes"
df_final_noid$Depression <- factor(df_final_noid$Depression, levels = c("No", "Yes"))

# dataset for tree model
df_final_tree <- df_final_noid
# Identify numeric and categorical variables in the tree-friendly dataset
num_vars <- names(df_final_tree)[sapply(df_final_tree, is.numeric)]
cat_vars <- names(df_final_tree)[sapply(df_final_tree, is.factor)]

## Step 3 data transformation and encoding
# Z-score standardization for continuous variables: Age, CGPA, Work.Study.Hours
df_final_noid <- df_final_noid %>%
  mutate(
    Age = scale(Age),
    CGPA = scale(CGPA),
    Work.Study.Hours = scale(Work.Study.Hours) )

# Min-max normalization for ordinal rating variables: Academic.Pressure and Study.Satisfaction
df_final_noid <- df_final_noid %>%
  mutate(
    Academic.Pressure = (Academic.Pressure - min(Academic.Pressure)) / 
      (max(Academic.Pressure) - min(Academic.Pressure)),
    Study.Satisfaction = (Study.Satisfaction - min(Study.Satisfaction)) / 
      (max(Study.Satisfaction) - min(Study.Satisfaction)) )

# df_encoded <- df_final_noid
# 1. Label encode binary variables
binary_vars <- c("Gender", "Have.you.ever.had.suicidal.thoughts.", "Family.History.of.Mental.Illness")

df_encoded <- df_encoded %>%
  mutate(
    Gender = ifelse(Gender == "Male", 1, 0),
    Have.you.ever.had.suicidal.thoughts.. = ifelse(Have.you.ever.had.suicidal.thoughts.. == "Yes", 1, 0),
    Family.History.of.Mental.Illness = ifelse(Family.History.of.Mental.Illness == "Yes", 1, 0)
  )

# 2. Ordinal encode ordered variables
# Sleep.Duration: 'Less than 5 hours' < '5-6 hours' < '7-8 hours' < 'More than 8 hours'
df_encoded$Sleep.Duration <- factor(df_encoded$Sleep.Duration, 
                                    levels = c("'Less than 5 hours'", "'5-6 hours'", 
                                               "'7-8 hours'", "'More than 8 hours'"),
                                    ordered = TRUE)
df_encoded$Sleep.Duration <- as.numeric(df_encoded$Sleep.Duration)

# Dietary.Habits: Healthy < Moderate < Unhealthy
df_encoded$Dietary.Habits <- factor(df_encoded$Dietary.Habits, 
                                    levels = c("Healthy", "Moderate", "Unhealthy"),
                                    ordered = TRUE)
df_encoded$Dietary.Habits <- as.numeric(df_encoded$Dietary.Habits)

# 3. One-hot encode Degree and City
# Make sure these are factors
df_encoded$Degree <- as.factor(df_encoded$Degree)
df_encoded$City <- as.factor(df_encoded$City)

# Use dummyVars from caret
dummies <- dummyVars(" ~ Degree + City", data = df_encoded)
dummy_data <- predict(dummies, newdata = df_encoded)
dummy_df <- as.data.frame(dummy_data)

# 4. Combine all together (excluding original Degree and City)
df_final <- df_encoded %>%
  select(-Degree, -City) %>%
  bind_cols(dummy_df)

# 5. Check final structure
# Convert to character first
df_final$Depression <- as.character(df_final$Depression)

# Map "0" to "No", "1" to "Yes"
df_final$Depression[df_final$Depression == "0"] <- "No"
df_final$Depression[df_final$Depression == "1"] <- "Yes"

# Convert back to factor with correct level order
df_final$Depression <- factor(df_final$Depression, levels = c("No", "Yes"))
str(df_final)

## Split training and testing set
# set seed 
set.seed(7)

# split into training and testing sets (70% vs 30%) for logistic and knn
train_index <- createDataPartition(df_final$Depression, p = 0.7, list = FALSE)
train <- df_final[train_index, ]
test <- df_final[-train_index, ]

# split into training and testing sets (70% vs 30%) for tree model
train_tree <- df_final_tree[train_index, ]
test_tree <- df_final_tree[-train_index, ]

