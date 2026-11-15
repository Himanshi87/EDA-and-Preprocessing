# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:49:14 2026

@author: Himu
"""

# ==========================================
# BANK MARKETING DATASET - COMPLETE EDA
# ==========================================

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load Dataset

df = pd.read_csv("D:/data understanding/bank_data (1).csv")

# Display first 5 records
print("First 5 Rows:")
print(df.head())

# Display last 5 records
print("\nLast 5 Rows:")
print(df.tail())

# ==========================================
# STEP 3: BASIC DATASET INFORMATION
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nData Types:")
print(df.dtypes)

# ==========================================
# STEP 4: CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Missing value percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing Value Percentage:")
print(missing_percentage)

# Missing value visualization
plt.figure(figsize=(12, 5))
missing_percentage.plot(kind='bar')
plt.title("Missing Values Percentage - Bank Dataset")
plt.xlabel("Columns")
plt.ylabel("Missing Percentage")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 5: CHECK DUPLICATE VALUES
# ==========================================

print("\nTotal Duplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate records if present
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

# ==========================================
# STEP 6: STATISTICAL ANALYSIS
# ==========================================

print("\nStatistical Summary:")
print(df.describe())

print("\nStatistical Summary Including All Columns:")
print(df.describe(include='all'))

# ==========================================
# STEP 7: UNIQUE VALUES
# ==========================================

print("\nUnique Values in Each Column:")

for col in df.columns:
    print(col, ":", df[col].nunique())

# ==========================================
# STEP 8: TARGET VARIABLE ANALYSIS
# ==========================================

print("\nTarget Variable Distribution:")
print(df['y'].value_counts())

print("\nTarget Variable Percentage:")
print(df['y'].value_counts(normalize=True) * 100)

# Target count plot
plt.figure(figsize=(6, 4))
df['y'].value_counts().sort_index().plot(
    kind='bar', rot=0
)
plt.title("Bank Subscription Distribution")
plt.xlabel("Subscription (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()

# Target pie chart
plt.figure(figsize=(6, 6))
df['y'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=['No', 'Yes']
)
plt.title("Bank Subscription Percentage")
plt.ylabel("")
plt.show()

# ==========================================
# STEP 9: NUMERICAL FEATURE DISTRIBUTION
# ==========================================

numerical_cols = [
    'age', 'balance', 'duration',
    'campaign', 'pdays', 'previous'
]

# Histograms
df[numerical_cols].hist(
    figsize=(14, 10),
    bins=30,
    edgecolor='black'
)

plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 10: BOXPLOTS FOR OUTLIER DETECTION
# ==========================================

for col in numerical_cols:

    plt.figure(figsize=(8, 4))

    plt.boxplot(df[col].dropna(), vert=False)

    plt.title("Boxplot of " + col)
    plt.xlabel(col)

    plt.show()

# ==========================================
# STEP 11: OUTLIER DETECTION USING IQR
# ==========================================

print("\nOutlier Detection:")

for col in numerical_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    print("\nColumn:", col)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Number of Outliers:", len(outliers))

# ==========================================
# STEP 12: CORRELATION ANALYSIS
# ==========================================

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(18, 14))

plt.imshow(
    correlation,
    cmap='coolwarm',
    aspect='auto',
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap - Bank Dataset")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 13: FEATURE CORRELATION WITH TARGET
# ==========================================

target_corr = correlation['y'].sort_values(
    ascending=False
)

print("\nCorrelation with Target Variable:")
print(target_corr)

plt.figure(figsize=(12, 8))

target_corr.drop('y').plot(kind='barh')

plt.title("Feature Correlation with Bank Subscription")
plt.xlabel("Correlation")
plt.ylabel("Features")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 14: AGE ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(df['age'], bins=25, edgecolor='black')

plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# ==========================================
# STEP 15: BALANCE ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(df['balance'], bins=40, edgecolor='black')

plt.title("Customer Balance Distribution")
plt.xlabel("Balance")
plt.ylabel("Frequency")

plt.show()

# ==========================================
# STEP 16: AGE VS TARGET
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(column='age', by='y')

plt.title("Age vs Bank Subscription")
plt.suptitle("")
plt.xlabel("Subscription (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.show()

# ==========================================
# STEP 17: BALANCE VS TARGET
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(column='balance', by='y')

plt.title("Balance vs Bank Subscription")
plt.suptitle("")
plt.xlabel("Subscription (0 = No, 1 = Yes)")
plt.ylabel("Balance")

plt.show()

# ==========================================
# STEP 18: DURATION VS TARGET
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(column='duration', by='y')

plt.title("Call Duration vs Bank Subscription")
plt.suptitle("")
plt.xlabel("Subscription (0 = No, 1 = Yes)")
plt.ylabel("Duration in Seconds")

plt.show()

# ==========================================
# STEP 19: CAMPAIGN ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

df.groupby('y')['campaign'].mean().plot(kind='bar')

plt.title("Average Campaign Contacts by Subscription")
plt.xlabel("Subscription (0 = No, 1 = Yes)")
plt.ylabel("Average Campaign Contacts")

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 20: FEATURE-WISE TARGET ANALYSIS
# ==========================================

binary_cols = [
    'default', 'housing', 'loan',
    'poutfailure', 'poutother', 'poutsuccess',
    'poutunknown', 'con_cellular',
    'con_telephone', 'con_unknown',
    'divorced', 'married', 'single',
    'joadmin.', 'joblue.collar',
    'joentrepreneur', 'johousemaid',
    'jomanagement', 'joretired',
    'joself.employed', 'joservices',
    'jostudent', 'jotechnician',
    'jounemployed', 'jounknown'
]

for col in binary_cols:

    result = df.groupby(col)['y'].mean() * 100

    plt.figure(figsize=(6, 4))

    result.plot(kind='bar')

    plt.title("Subscription Rate by " + col)
    plt.xlabel(col)
    plt.ylabel("Subscription Rate (%)")

    plt.xticks(rotation=0)
    plt.show()

# ==========================================
# STEP 21: FINAL EDA SUMMARY
# ==========================================

print("\n========== BANK DATASET EDA SUMMARY ==========")

print("Total Records:", len(df))
print("Total Features:", df.shape[1] - 1)

print("Missing Values:", df.isnull().sum().sum())

print("Duplicate Records:", df.duplicated().sum())

print("\nTarget Distribution:")
print(df['y'].value_counts())

print("\nTarget Percentage:")
print(df['y'].value_counts(normalize=True) * 100)

print("\nEDA COMPLETED SUCCESSFULLY!")


# ==========================================
# BANK MARKETING - DATA PREPROCESSING
# ==========================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# STEP 2: LOAD DATASET

df = pd.read_csv("D:/data understanding/bank_data (1).csv")

print("Original Dataset:")
print(df.head())

print("\nOriginal Shape:", df.shape)

# STEP 3: REMOVE DUPLICATES

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

# STEP 4: HANDLE MISSING VALUES

print("\nMissing Values:")
print(df.isnull().sum())

# STEP 5: SEPARATE FEATURES AND TARGET

X = df.drop('y', axis=1)
y = df['y']

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# STEP 6: IDENTIFY NUMERICAL AND BINARY FEATURES

numerical_cols = [
    'age', 'balance', 'duration',
    'campaign', 'pdays', 'previous'
]

binary_cols = [
    col for col in X.columns
    if col not in numerical_cols
]

print("\nNumerical Columns:")
print(numerical_cols)

print("\nBinary Columns:")
print(binary_cols)

# STEP 7: TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# STEP 8: CREATE PREPROCESSING PIPELINES

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_cols),
    ('binary', binary_pipeline, binary_cols)
])

# STEP 9: FIT AND TRANSFORM TRAINING DATA

X_train_processed = preprocessor.fit_transform(X_train)

# STEP 10: TRANSFORM TESTING DATA

X_test_processed = preprocessor.transform(X_test)

# STEP 11: CONVERT INTO DATAFRAME

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

# STEP 12: DISPLAY PROCESSED DATA

print("\nProcessed Training Data:")
print(X_train_processed.head())

print("\nProcessed Testing Data:")
print(X_test_processed.head())

# STEP 13: CHECK MISSING VALUES

print("\nMissing Values After Processing:")
print(X_train_processed.isnull().sum().sum())

# STEP 14: CHECK SCALING

print("\nScaled Numerical Feature Means:")
print(X_train_processed[
    ['numerical__' + col for col in numerical_cols]
].mean())

# STEP 15: TARGET DISTRIBUTION

print("\nTraining Target Distribution:")
print(y_train.value_counts())

# STEP 16: SAVE PROCESSED DATA

train_data = X_train_processed.copy()
train_data['y'] = y_train

test_data = X_test_processed.copy()
test_data['y'] = y_test

train_data.to_csv(
    "bank_train_processed.csv",
    index=False
)

test_data.to_csv(
    "bank_test_processed.csv",
    index=False
)

print("\nBank preprocessing completed successfully!")

print("Training Data:", X_train_processed.shape)
print("Testing Data:", X_test_processed.shape)