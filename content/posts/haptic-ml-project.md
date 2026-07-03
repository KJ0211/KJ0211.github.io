---
title: "Classifying Human Interaction Patterns in Inertial Sensor Data: A Machine Learning Approach to Inform Future Haptic and Robotic Systems"
date: 2026-05-02
draft: false
tags: ["Machine Learning", "Numpy", "Pandas", "Scikit-Learn", "Matplotlib"]
summary: "This project used machine learning to classify human activity patterns from smartphone inertial sensor data. Using the UCI Human Activity Recognition dataset, I trained a Random Forest classifier to distinguish between six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying. The project explored how accelerometer and gyroscope features can support human activity recognition and inform future haptic or robotic systems."
---

## 1. Problem
Human movement patterns can be difficult to classify accurately from sensor data, especially when different postures produce similar signals. This project investigated whether high-dimensional inertial features could distinguish between static and dynamic human activities, and which sensor features were most useful for classification.

## 2. Dataset
The project used the UCI Human Activity Recognition dataset, collected from 30 participants aged 19 to 48. Participants wore a Samsung Galaxy S II smartphone on the waist while performing six activities. The dataset contains 3-axis accelerometer and gyroscope readings captured at 50 Hz, with 561 pre-extracted time and frequency-domain features. The data was split by participant, with 70% used for training and 30% used for testing on unseen individuals.
![Activity Distribution](/images/figure-1.jpg)

## 3. Methodology
- Prepared the high-dimensional sensor dataset for machine learning
- Encoded activity labels into numerical classes
- Checked class distribution and confirmed the dataset was balanced
- Analysed correlations between the 561 sensor features
- Trained a Random Forest classifier with 100 estimators
- Used feature-importance analysis to identify the most predictive sensor features
- Evaluated model performance using a confusion matrix and classification report
![Confusion matrix](/images/figure-4.png)
![Classification](/images/figure-5.jpg)


## 4. Evaluation
The Random Forest model achieved 93% accuracy on the unseen test set. The model performed strongly on dynamic movements such as walking and on distinct static postures such as laying. Evaluation using the confusion matrix showed that the main source of error was confusion between sitting and standing, because both activities produce similar waist-level gravity orientation signals.
![Top 20 features](/images/figure-2.jpg)

## 5. Conclusion
The project showed that Random Forest can classify human activity patterns effectively from high-dimensional inertial sensor data. Feature-importance analysis showed that gravity-based orientation features were among the strongest predictors. The results suggest that tilt-sensing and movement-intensity features are especially useful for designing responsive haptic and robotic systems.

## 6. Reflection
This project strengthened my understanding of supervised machine learning, sensor-data classification, feature importance, model evaluation, and the practical limitations of classification models. It also helped me understand how machine learning results can be interpreted and connected to human-centred technology design.

# Features Extraction Coding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. Data Preparation & Cleaning
# ==========================================
# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#  (Requirement: Data Science Process - Cleaning)
print("--- Missing Values Check ---")
print(f"Missing values in Train: {train_df.isnull().values.any()}")
print(f"Missing values in Test: {test_df.isnull().values.any()}")

# Separate features (X) and labels (y)
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Label Encoding
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# ==========================================
# 2. Exploratory Data Analysis
# ==========================================
print("\n--- Exploratory Data Analysis ---")

# A. Activity Distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=y_train)
plt.title("Distribution of Activities in Training Set")
plt.xticks(rotation=45)
plt.show()

# B. Correlation Heatmap
plt.figure(figsize=(12, 10))
top_20_corr = X_train.iloc[:, :20].corr()
sns.heatmap(top_20_corr, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap (First 20 Features)")
plt.show()


# ==========================================
# 3. Feature Selection
# ==========================================
# Remove constant features (Variance Threshold)
selector = VarianceThreshold(threshold=0)
X_train_reduced = selector.fit_transform(X_train)
X_test_reduced = selector.transform(X_test)
print(f"Features reduced from {X_train.shape[1]} to {X_train_reduced.shape[1]} by removing constants.")

# ==========================================
# 4. Model Training
# ==========================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_reduced, y_train_encoded)

# ==========================================
# 5. Evaluation & Knowledge Extraction
# ==========================================
y_pred = rf_model.predict(X_test_reduced)

# A. Accuracy & Report
print(f"\nOverall Accuracy: {accuracy_score(y_test_encoded, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

# B. Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix: Visualizing Sitting vs Standing Conflict")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# C. Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 10 Most Predictive Features:")
for i in range(10):
    print(f"{i+1}. {X_train.columns[indices[i]]} ({importances[indices[i]]:.4f})")

print(f"Sum of importances: {importances.sum():.2f}")
plt.figure(figsize=(10, 6))
plt.title("Top 20 Feature Importances")
plt.bar(range(20), importances[indices[:20]], color='skyblue', align="center")
plt.xticks(range(20), X_train.columns[indices[:20]], rotation=90)
plt.tight_layout()
plt.show()