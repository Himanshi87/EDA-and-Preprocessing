# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:50:03 2026

@author: Himu
"""

# ==========================================
# HEART DISEASE DATASET - COMPLETE EDA
# ==========================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 2: LOAD DATASET

df = pd.read_csv("D:/data understanding/heart disease (1).csv")

print("First 5 Rows:")
print(df.head())

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

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

print("\nMissing Percentage:")
print(missing_percentage)

# Missing value visualization

plt.figure(figsize=(10, 5))

missing_percentage.plot(kind='bar')

plt.title("Missing Values - Heart Disease Dataset")
plt.xlabel("Features")
plt.ylabel("Missing Percentage")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 5: CHECK DUPLICATES
# ==========================================

print("\nDuplicate Records:")
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

print("\nComplete Statistical Summary:")
print(df.describe(include='all'))

# ==========================================
# STEP 7: UNIQUE VALUES
# ==========================================

print("\nUnique Values:")

for col in df.columns:

    print(col, ":", df[col].nunique())

# ==========================================
# STEP 8: TARGET VARIABLE ANALYSIS
# ==========================================

print("\nHeart Disease Target Distribution:")
print(df['target'].value_counts())

print("\nTarget Percentage:")
print(df['target'].value_counts(normalize=True) * 100)

# Target count plot

plt.figure(figsize=(6, 4))

df['target'].value_counts().sort_index().plot(
    kind='bar',
    rot=0
)

plt.title("Heart Disease Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Patients")

plt.show()

# Target pie chart

plt.figure(figsize=(6, 6))

df['target'].value_counts().sort_index().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=['No Disease', 'Disease']
)

plt.title("Heart Disease Percentage")
plt.ylabel("")

plt.show()

# ==========================================
# STEP 9: NUMERICAL FEATURE DISTRIBUTION
# ==========================================

numerical_cols = [
    'age', 'trestbps', 'chol',
    'thalach', 'oldpeak'
]

df[numerical_cols].hist(
    figsize=(14, 10),
    bins=20,
    edgecolor='black'
)

plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 10: BOXPLOTS
# ==========================================

for col in numerical_cols:

    plt.figure(figsize=(8, 4))

    plt.boxplot(
        df[col].dropna(),
        vert=False
    )

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

    print("\nFeature:", col)

    print("Lower Limit:", lower_limit)

    print("Upper Limit:", upper_limit)

    print("Number of Outliers:", len(outliers))

# ==========================================
# STEP 12: CORRELATION MATRIX
# ==========================================

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

# ==========================================
# STEP 13: CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12, 9))

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
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap - Heart Disease")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 14: CORRELATION WITH TARGET
# ==========================================

target_corr = correlation['target'].sort_values(
    ascending=False
)

print("\nCorrelation with Heart Disease:")
print(target_corr)

plt.figure(figsize=(10, 7))

target_corr.drop('target').plot(
    kind='barh'
)

plt.title("Feature Correlation with Heart Disease")
plt.xlabel("Correlation")
plt.ylabel("Features")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 15: AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df['age'],
    bins=20,
    edgecolor='black'
)

plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.show()

# ==========================================
# STEP 16: GENDER DISTRIBUTION
# ==========================================

print("\nGender Distribution:")
print(df['sex'].value_counts())

plt.figure(figsize=(6, 5))

df['sex'].value_counts().sort_index().plot(
    kind='bar',
    rot=0
)

plt.title("Gender Distribution")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Number of Patients")

plt.show()

# ==========================================
# STEP 17: HEART DISEASE BY GENDER
# ==========================================

gender_target = pd.crosstab(
    df['sex'],
    df['target']
)

print("\nHeart Disease by Gender:")
print(gender_target)

gender_target.plot(
    kind='bar',
    figsize=(7, 5)
)

plt.title("Heart Disease Distribution by Gender")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Number of Patients")

plt.legend(
    ['No Disease', 'Disease']
)

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 18: AGE VS HEART DISEASE
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column='age',
    by='target'
)

plt.title("Age vs Heart Disease")
plt.suptitle("")

plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Age")

plt.show()

# ==========================================
# STEP 19: CHOLESTEROL ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df['chol'],
    bins=20,
    edgecolor='black'
)

plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol Level")
plt.ylabel("Frequency")

plt.show()

# Cholesterol vs target

plt.figure(figsize=(8, 5))

df.boxplot(
    column='chol',
    by='target'
)

plt.title("Cholesterol vs Heart Disease")
plt.suptitle("")

plt.xlabel("Heart Disease")
plt.ylabel("Cholesterol")

plt.show()

# ==========================================
# STEP 20: MAXIMUM HEART RATE ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column='thalach',
    by='target'
)

plt.title("Maximum Heart Rate vs Heart Disease")
plt.suptitle("")

plt.xlabel("Heart Disease")
plt.ylabel("Maximum Heart Rate")

plt.show()

# ==========================================
# STEP 21: CHEST PAIN ANALYSIS
# ==========================================

print("\nChest Pain Distribution:")
print(df['cp'].value_counts())

cp_target = pd.crosstab(
    df['cp'],
    df['target']
)

print("\nChest Pain vs Heart Disease:")
print(cp_target)

cp_target.plot(
    kind='bar',
    figsize=(8, 5)
)

plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")

plt.legend(
    ['No Disease', 'Disease']
)

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 22: FASTING BLOOD SUGAR ANALYSIS
# ==========================================

print("\nFasting Blood Sugar:")
print(df['fbs'].value_counts())

fbs_target = pd.crosstab(
    df['fbs'],
    df['target']
)

fbs_target.plot(
    kind='bar',
    figsize=(7, 5)
)

plt.title("Fasting Blood Sugar vs Heart Disease")
plt.xlabel("FBS (0 = Normal, 1 = High)")
plt.ylabel("Number of Patients")

plt.legend(
    ['No Disease', 'Disease']
)

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 23: EXERCISE INDUCED ANGINA
# ==========================================

print("\nExercise Induced Angina:")
print(df['exang'].value_counts())

exang_target = pd.crosstab(
    df['exang'],
    df['target']
)

exang_target.plot(
    kind='bar',
    figsize=(7, 5)
)

plt.title("Exercise Induced Angina vs Heart Disease")
plt.xlabel("Exang (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.legend(
    ['No Disease', 'Disease']
)

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 24: ST DEPRESSION ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column='oldpeak',
    by='target'
)

plt.title("ST Depression vs Heart Disease")
plt.suptitle("")

plt.xlabel("Heart Disease")
plt.ylabel("Oldpeak")

plt.show()

# ==========================================
# STEP 25: CATEGORICAL FEATURE ANALYSIS
# ==========================================

categorical_cols = [
    'sex', 'cp', 'fbs', 'restecg',
    'exang', 'slope', 'ca', 'thal'
]

for col in categorical_cols:

    plt.figure(figsize=(7, 4))

    df[col].value_counts().sort_index().plot(
        kind='bar'
    )

    plt.title("Distribution of " + col)
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.xticks(rotation=0)
    plt.show()

# ==========================================
# STEP 26: CATEGORICAL FEATURES VS TARGET
# ==========================================

for col in categorical_cols:

    cross_tab = pd.crosstab(
        df[col],
        df['target']
    )

    cross_tab.plot(
        kind='bar',
        figsize=(7, 5)
    )

    plt.title(col + " vs Heart Disease")
    plt.xlabel(col)
    plt.ylabel("Number of Patients")

    plt.legend(
        ['No Disease', 'Disease']
    )

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ==========================================
# STEP 27: FINAL EDA SUMMARY
# ==========================================

print("\n========== HEART DISEASE EDA SUMMARY ==========")

print("Total Records:", len(df))

print("Total Features:", df.shape[1] - 1)

print("Missing Values:", df.isnull().sum().sum())

print("Duplicate Records:", df.duplicated().sum())

print("\nTarget Distribution:")
print(df['target'].value_counts())

print("\nTarget Percentage:")
print(df['target'].value_counts(normalize=True) * 100)

print("\nEDA COMPLETED SUCCESSFULLY!")


# ==========================================
# HEART DISEASE - DATA PREPROCESSING
# ==========================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# STEP 2: LOAD DATASET

df = pd.read_csv("D:/data understanding/heart disease (1).csv")

print("Original Dataset:")
print(df.head())

print("\nOriginal Shape:", df.shape)

# STEP 3: REMOVE DUPLICATES

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

# STEP 4: CHECK MISSING VALUES

print("\nMissing Values:")
print(df.isnull().sum())

# STEP 5: SEPARATE FEATURES AND TARGET

X = df.drop('target', axis=1)
y = df['target']

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# STEP 6: IDENTIFY FEATURE TYPES

numerical_cols = [
    'age', 'trestbps', 'chol',
    'thalach', 'oldpeak'
]

categorical_cols = [
    'cp', 'restecg', 'slope', 'ca', 'thal'
]

binary_cols = [
    'sex', 'fbs', 'exang'
]

print("\nNumerical Features:")
print(numerical_cols)

print("\nCategorical Features:")
print(categorical_cols)

print("\nBinary Features:")
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

# STEP 8: NUMERICAL PIPELINE

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# STEP 9: CATEGORICAL PIPELINE

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    ))
])

# STEP 10: BINARY PIPELINE

binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# STEP 11: COLUMN TRANSFORMER

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_cols),
    ('categorical', categorical_pipeline, categorical_cols),
    ('binary', binary_pipeline, binary_cols)
])

# STEP 12: FIT AND TRANSFORM TRAINING DATA

X_train_processed = preprocessor.fit_transform(X_train)

# STEP 13: TRANSFORM TESTING DATA

X_test_processed = preprocessor.transform(X_test)

# STEP 14: GET FEATURE NAMES

feature_names = preprocessor.get_feature_names_out()

# STEP 15: CONVERT TO DATAFRAME

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

# STEP 16: DISPLAY PROCESSED DATA

print("\nProcessed Training Data:")
print(X_train_processed.head())

print("\nProcessed Testing Data:")
print(X_test_processed.head())

# STEP 17: CHECK MISSING VALUES

print("\nMissing Values After Processing:")
print(X_train_processed.isnull().sum().sum())

# STEP 18: CHECK SCALING

print("\nNumerical Feature Means:")

scaled_cols = [
    'numerical__' + col for col in numerical_cols
]

print(X_train_processed[scaled_cols].mean())

# STEP 19: TARGET DISTRIBUTION

print("\nTraining Target Distribution:")
print(y_train.value_counts())

# STEP 20: SAVE PROCESSED DATA

train_data = X_train_processed.copy()
train_data['target'] = y_train

test_data = X_test_processed.copy()
test_data['target'] = y_test

train_data.to_csv(
    "heart_train_processed.csv",
    index=False
)

test_data.to_csv(
    "heart_test_processed.csv",
    index=False
)

print("\nHeart Disease Preprocessing Completed!")

print("Training Data:", X_train_processed.shape)
print("Testing Data:", X_test_processed.shape)