# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:50:55 2026

@author: Himu
"""

# ==========================================
# MT CARS DATASET - COMPLETE EDA
# ==========================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 2: LOAD DATASET

df = pd.read_csv("D:/data understanding/mtcars.csv")

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

missing_percentage = df.isnull().mean() * 100

print("\nMissing Value Percentage:")
print(missing_percentage)

plt.figure(figsize=(10, 5))
missing_percentage.plot(kind='bar')

plt.title("Missing Values - MT Cars Dataset")
plt.xlabel("Columns")
plt.ylabel("Missing Percentage")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 5: CHECK DUPLICATES
# ==========================================

print("\nDuplicate Records:")
print(df.duplicated().sum())

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
# STEP 8: NUMERICAL FEATURE DISTRIBUTION
# ==========================================

numerical_cols = [
    'mpg', 'cyl', 'disp', 'hp', 'drat',
    'wt', 'qsec', 'vs', 'am', 'gear', 'carb'
]

df[numerical_cols].hist(
    figsize=(15, 12),
    bins=10,
    edgecolor='black'
)

plt.suptitle("Distribution of MT Cars Features")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 9: BOXPLOT FOR ALL FEATURES
# ==========================================

for col in numerical_cols:

    plt.figure(figsize=(8, 4))

    plt.boxplot(df[col].dropna(), vert=False)

    plt.title("Boxplot of " + col)
    plt.xlabel(col)

    plt.show()

# ==========================================
# STEP 10: OUTLIER DETECTION USING IQR
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
# STEP 11: CORRELATION MATRIX
# ==========================================

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

# ==========================================
# STEP 12: CORRELATION HEATMAP
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

plt.title("Correlation Heatmap - MT Cars")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 13: MPG DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df['mpg'],
    bins=10,
    edgecolor='black'
)

plt.title("Miles Per Gallon Distribution")
plt.xlabel("MPG")
plt.ylabel("Frequency")

plt.show()

# ==========================================
# STEP 14: HORSEPOWER DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df['hp'],
    bins=10,
    edgecolor='black'
)

plt.title("Horsepower Distribution")
plt.xlabel("Horsepower")
plt.ylabel("Frequency")

plt.show()

# ==========================================
# STEP 15: WEIGHT DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df['wt'],
    bins=10,
    edgecolor='black'
)

plt.title("Car Weight Distribution")
plt.xlabel("Weight (1000 lbs)")
plt.ylabel("Frequency")

plt.show()

# ==========================================
# STEP 16: MPG VS HORSEPOWER
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(df['hp'], df['mpg'])

plt.title("MPG vs Horsepower")
plt.xlabel("Horsepower")
plt.ylabel("Miles Per Gallon")

plt.show()

# ==========================================
# STEP 17: MPG VS WEIGHT
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(df['wt'], df['mpg'])

plt.title("MPG vs Car Weight")
plt.xlabel("Weight (1000 lbs)")
plt.ylabel("Miles Per Gallon")

plt.show()

# ==========================================
# STEP 18: MPG VS DISPLACEMENT
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(df['disp'], df['mpg'])

plt.title("MPG vs Displacement")
plt.xlabel("Displacement")
plt.ylabel("Miles Per Gallon")

plt.show()

# ==========================================
# STEP 19: CYLINDER DISTRIBUTION
# ==========================================

print("\nCylinder Distribution:")
print(df['cyl'].value_counts().sort_index())

plt.figure(figsize=(7, 5))

df['cyl'].value_counts().sort_index().plot(
    kind='bar'
)

plt.title("Distribution of Cylinders")
plt.xlabel("Number of Cylinders")
plt.ylabel("Number of Cars")

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 20: MPG VS CYLINDERS
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column='mpg',
    by='cyl'
)

plt.title("MPG vs Number of Cylinders")
plt.suptitle("")

plt.xlabel("Number of Cylinders")
plt.ylabel("MPG")

plt.show()

# ==========================================
# STEP 21: TRANSMISSION ANALYSIS
# ==========================================

print("\nTransmission Distribution:")
print(df['am'].value_counts())

# 0 = Automatic
# 1 = Manual

plt.figure(figsize=(7, 5))

df['am'].value_counts().sort_index().plot(
    kind='bar'
)

plt.title("Transmission Type Distribution")
plt.xlabel("Transmission (0 = Automatic, 1 = Manual)")
plt.ylabel("Number of Cars")

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 22: TRANSMISSION VS MPG
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column='mpg',
    by='am'
)

plt.title("Transmission vs MPG")
plt.suptitle("")

plt.xlabel("Transmission (0 = Automatic, 1 = Manual)")
plt.ylabel("MPG")

plt.show()

# ==========================================
# STEP 23: GEAR DISTRIBUTION
# ==========================================

print("\nGear Distribution:")
print(df['gear'].value_counts().sort_index())

plt.figure(figsize=(7, 5))

df['gear'].value_counts().sort_index().plot(
    kind='bar'
)

plt.title("Distribution of Gears")
plt.xlabel("Number of Gears")
plt.ylabel("Number of Cars")

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 24: AVERAGE MPG BY CYLINDERS
# ==========================================

avg_mpg = df.groupby('cyl')['mpg'].mean()

print("\nAverage MPG by Cylinders:")
print(avg_mpg)

plt.figure(figsize=(7, 5))

avg_mpg.plot(kind='bar')

plt.title("Average MPG by Number of Cylinders")
plt.xlabel("Cylinders")
plt.ylabel("Average MPG")

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 25: AVERAGE MPG BY TRANSMISSION
# ==========================================

avg_transmission = df.groupby('am')['mpg'].mean()

print("\nAverage MPG by Transmission:")
print(avg_transmission)

plt.figure(figsize=(7, 5))

avg_transmission.plot(kind='bar')

plt.title("Average MPG by Transmission Type")
plt.xlabel("Transmission (0 = Automatic, 1 = Manual)")
plt.ylabel("Average MPG")

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 26: PAIRWISE FEATURE ANALYSIS
# ==========================================

selected_cols = [
    'mpg', 'hp', 'wt', 'disp', 'qsec'
]

pd.plotting.scatter_matrix(
    df[selected_cols],
    figsize=(12, 10),
    diagonal='hist'
)

plt.suptitle("Pairwise Relationships - MT Cars")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 27: FINAL EDA SUMMARY
# ==========================================

print("\n========== MT CARS EDA SUMMARY ==========")

print("Total Records:", len(df))
print("Total Features:", df.shape[1])

print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Records:", df.duplicated().sum())

print("\nAverage MPG:", df['mpg'].mean())
print("Average Horsepower:", df['hp'].mean())
print("Average Weight:", df['wt'].mean())

print("\nAverage MPG by Cylinders:")
print(df.groupby('cyl')['mpg'].mean())

print("\nAverage MPG by Transmission:")
print(df.groupby('am')['mpg'].mean())

print("\nEDA COMPLETED SUCCESSFULLY!")


# ==========================================
# MT CARS - DATA PREPROCESSING
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

df = pd.read_csv("D:/data understanding/mtcars.csv")

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

X = df.drop('mpg', axis=1)
y = df['mpg']

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# STEP 6: IDENTIFY FEATURE TYPES

numerical_cols = [
    'disp', 'hp', 'drat', 'wt', 'qsec'
]

categorical_cols = [
    'cyl', 'vs', 'am', 'gear', 'carb'
]

print("\nNumerical Features:")
print(numerical_cols)

print("\nCategorical Features:")
print(categorical_cols)

# STEP 7: TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
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

# STEP 10: COLUMN TRANSFORMER

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_cols),
    ('categorical', categorical_pipeline, categorical_cols)
])

# STEP 11: FIT TRAINING DATA

X_train_processed = preprocessor.fit_transform(X_train)

# STEP 12: TRANSFORM TESTING DATA

X_test_processed = preprocessor.transform(X_test)

# STEP 13: GET FEATURE NAMES

feature_names = preprocessor.get_feature_names_out()

# STEP 14: CONVERT TO DATAFRAME

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

# STEP 15: DISPLAY PROCESSED DATA

print("\nProcessed Training Data:")
print(X_train_processed.head())

print("\nProcessed Testing Data:")
print(X_test_processed.head())

# STEP 16: CHECK MISSING VALUES

print("\nMissing Values After Processing:")
print(X_train_processed.isnull().sum().sum())

# STEP 17: CHECK SCALING

scaled_cols = [
    'numerical__' + col for col in numerical_cols
]

print("\nScaled Feature Means:")
print(X_train_processed[scaled_cols].mean())

# STEP 18: DISPLAY TARGET VALUES

print("\nTraining Target Values:")
print(y_train.head())

# STEP 19: SAVE PROCESSED DATA

train_data = X_train_processed.copy()
train_data['mpg'] = y_train

test_data = X_test_processed.copy()
test_data['mpg'] = y_test

train_data.to_csv(
    "mtcars_train_processed.csv",
    index=False
)

test_data.to_csv(
    "mtcars_test_processed.csv",
    index=False
)

print("\nMT Cars Preprocessing Completed!")

print("Training Data:", X_train_processed.shape)
print("Testing Data:", X_test_processed.shape)