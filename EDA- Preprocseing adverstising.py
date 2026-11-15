# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:46:13 2026

@author: Himu
"""

# ==========================================
# ADVERTISING DATASET - COMPLETE EDA
# ==========================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 2: LOAD DATASET

df = pd.read_csv("D:/data understanding/advertising.csv")

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

plt.title("Missing Values - Advertising Dataset")
plt.xlabel("Columns")
plt.ylabel("Missing Percentage")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 5: CHECK DUPLICATE RECORDS
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
# STEP 8: TARGET VARIABLE ANALYSIS
# ==========================================

print("\nAdvertisement Click Distribution:")

print(df['Clicked_on_Ad'].value_counts())

print("\nTarget Percentage:")

print(
    df['Clicked_on_Ad'].value_counts(
        normalize=True
    ) * 100
)

# Target count plot

plt.figure(figsize=(6, 4))

df['Clicked_on_Ad'].value_counts().sort_index().plot(
    kind='bar',
    rot=0
)

plt.title("Advertisement Click Distribution")
plt.xlabel("Clicked on Ad (0 = No, 1 = Yes)")
plt.ylabel("Number of Users")

plt.show()

# Target pie chart

plt.figure(figsize=(6, 6))

df['Clicked_on_Ad'].value_counts().sort_index().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=['Not Clicked', 'Clicked']
)

plt.title("Advertisement Click Percentage")
plt.ylabel("")

plt.show()

# ==========================================
# STEP 9: NUMERICAL FEATURE DISTRIBUTION
# ==========================================

numerical_cols = [
    'Daily_Time_ Spent _on_Site',
    'Age',
    'Area_Income',
    'Daily Internet Usage'
]

df[numerical_cols].hist(
    figsize=(13, 9),
    bins=20,
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

correlation = df[
    numerical_cols + ['Male', 'Clicked_on_Ad']
].corr()

print("\nCorrelation Matrix:")
print(correlation)

# ==========================================
# STEP 13: CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(10, 8))

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

plt.title("Correlation Heatmap - Advertising Dataset")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 14: CORRELATION WITH TARGET
# ==========================================

target_corr = correlation['Clicked_on_Ad'].sort_values(
    ascending=False
)

print("\nCorrelation with Advertisement Click:")
print(target_corr)

plt.figure(figsize=(9, 6))

target_corr.drop('Clicked_on_Ad').plot(
    kind='barh'
)

plt.title("Feature Correlation with Advertisement Click")
plt.xlabel("Correlation")
plt.ylabel("Features")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 15: AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df['Age'],
    bins=20,
    edgecolor='black'
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Users")

plt.show()

# ==========================================
# STEP 16: AGE VS ADVERTISEMENT CLICK
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column='Age',
    by='Clicked_on_Ad'
)

plt.title("Age vs Advertisement Click")
plt.suptitle("")

plt.xlabel("Clicked on Ad (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.show()

# ==========================================
# STEP 17: WEBSITE TIME ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df['Daily_Time_ Spent _on_Site'],
    bins=20,
    edgecolor='black'
)

plt.title("Daily Time Spent on Website")
plt.xlabel("Time Spent")
plt.ylabel("Frequency")

plt.show()

# Website time vs target

plt.figure(figsize=(8, 5))

df.boxplot(
    column='Daily_Time_ Spent _on_Site',
    by='Clicked_on_Ad'
)

plt.title("Website Time vs Advertisement Click")
plt.suptitle("")

plt.xlabel("Clicked on Ad")
plt.ylabel("Daily Time Spent")

plt.show()

# ==========================================
# STEP 18: AREA INCOME ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df['Area_Income'],
    bins=20,
    edgecolor='black'
)

plt.title("Area Income Distribution")
plt.xlabel("Area Income")
plt.ylabel("Frequency")

plt.show()

# Income vs target

plt.figure(figsize=(8, 5))

df.boxplot(
    column='Area_Income',
    by='Clicked_on_Ad'
)

plt.title("Area Income vs Advertisement Click")
plt.suptitle("")

plt.xlabel("Clicked on Ad")
plt.ylabel("Area Income")

plt.show()

# ==========================================
# STEP 19: INTERNET USAGE ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df['Daily Internet Usage'],
    bins=20,
    edgecolor='black'
)

plt.title("Daily Internet Usage Distribution")
plt.xlabel("Daily Internet Usage")
plt.ylabel("Frequency")

plt.show()

# Internet usage vs target

plt.figure(figsize=(8, 5))

df.boxplot(
    column='Daily Internet Usage',
    by='Clicked_on_Ad'
)

plt.title("Internet Usage vs Advertisement Click")
plt.suptitle("")

plt.xlabel("Clicked on Ad")
plt.ylabel("Daily Internet Usage")

plt.show()

# ==========================================
# STEP 20: GENDER DISTRIBUTION
# ==========================================

print("\nGender Distribution:")

print(df['Male'].value_counts())

plt.figure(figsize=(6, 5))

df['Male'].value_counts().sort_index().plot(
    kind='bar',
    rot=0
)

plt.title("Gender Distribution")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Number of Users")

plt.show()

# ==========================================
# STEP 21: GENDER VS ADVERTISEMENT CLICK
# ==========================================

gender_target = pd.crosstab(
    df['Male'],
    df['Clicked_on_Ad']
)

print("\nGender vs Advertisement Click:")
print(gender_target)

gender_target.plot(
    kind='bar',
    figsize=(7, 5)
)

plt.title("Gender vs Advertisement Click")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Number of Users")

plt.legend(
    ['Not Clicked', 'Clicked']
)

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 22: AVERAGE NUMERICAL FEATURES
# ==========================================

average_values = df.groupby(
    'Clicked_on_Ad'
)[numerical_cols].mean()

print("\nAverage Features by Advertisement Click:")
print(average_values)

average_values.T.plot(
    kind='bar',
    figsize=(11, 6)
)

plt.title("Average User Features by Advertisement Click")
plt.xlabel("Features")
plt.ylabel("Average Value")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 23: CITY ANALYSIS
# ==========================================

print("\nTop 10 Cities:")

print(df['City'].value_counts().head(10))

city_counts = df['City'].value_counts().head(10)

plt.figure(figsize=(10, 5))

city_counts.plot(kind='bar')

plt.title("Top 10 Cities by Number of Users")
plt.xlabel("City")
plt.ylabel("Number of Users")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 24: COUNTRY ANALYSIS
# ==========================================

print("\nTop 10 Countries:")

print(df['Country'].value_counts().head(10))

country_counts = df['Country'].value_counts().head(10)

plt.figure(figsize=(10, 5))

country_counts.plot(kind='bar')

plt.title("Top 10 Countries by Number of Users")
plt.xlabel("Country")
plt.ylabel("Number of Users")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 25: COUNTRY-WISE CLICK RATE
# ==========================================

country_click_rate = (
    df.groupby('Country')['Clicked_on_Ad']
    .agg(['count', 'mean'])
)

country_click_rate['mean'] *= 100

print("\nCountries with Highest Click Rate:")
print(
    country_click_rate.sort_values(
        by='mean',
        ascending=False
    ).head(10)
)

# ==========================================
# STEP 26: TIMESTAMP ANALYSIS
# ==========================================

df['Timestamp'] = pd.to_datetime(
    df['Timestamp'],
    dayfirst=True,
    errors='coerce'
)

print("\nTimestamp Information:")
print(df['Timestamp'].describe())

df['Hour'] = df['Timestamp'].dt.hour

df['Month'] = df['Timestamp'].dt.month

print("\nHourly Advertisement Click Rate:")

hourly_click = df.groupby('Hour')['Clicked_on_Ad'].mean() * 100

print(hourly_click)

plt.figure(figsize=(9, 5))

hourly_click.plot(kind='bar')

plt.title("Advertisement Click Rate by Hour")
plt.xlabel("Hour")
plt.ylabel("Click Rate (%)")

plt.xticks(rotation=0)
plt.show()

# Monthly click rate

monthly_click = df.groupby('Month')['Clicked_on_Ad'].mean() * 100

plt.figure(figsize=(9, 5))

monthly_click.plot(kind='bar')

plt.title("Advertisement Click Rate by Month")
plt.xlabel("Month")
plt.ylabel("Click Rate (%)")

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 27: PAIRWISE FEATURE ANALYSIS
# ==========================================

selected_cols = [
    'Age',
    'Area_Income',
    'Daily Internet Usage',
    'Daily_Time_ Spent _on_Site'
]

pd.plotting.scatter_matrix(
    df[selected_cols],
    figsize=(12, 10),
    diagonal='hist'
)

plt.suptitle("Pairwise Relationships - Advertising Dataset")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 28: FINAL EDA SUMMARY
# ==========================================

print("\n========== ADVERTISING DATASET EDA SUMMARY ==========")

print("Total Records:", len(df))

print("Total Columns:", df.shape[1])

print("Missing Values:", df.isnull().sum().sum())

print("Duplicate Records:", df.duplicated().sum())

print("\nTarget Distribution:")

print(df['Clicked_on_Ad'].value_counts())

print("\nTarget Percentage:")

print(
    df['Clicked_on_Ad'].value_counts(
        normalize=True
    ) * 100
)

print("\nAverage Age:", df['Age'].mean())

print(
    "Average Website Time:",
    df['Daily_Time_ Spent _on_Site'].mean()
)

print(
    "Average Internet Usage:",
    df['Daily Internet Usage'].mean()
)

print("\nEDA COMPLETED SUCCESSFULLY!")

# ==========================================
# ADVERTISING DATASET - DATA PREPROCESSING
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

df = pd.read_csv("D:/data understanding/advertising.csv")

print("Original Dataset:")
print(df.head())

print("\nOriginal Shape:")
print(df.shape)

# STEP 3: CLEAN COLUMN NAMES

df.columns = df.columns.str.strip()

print("\nUpdated Column Names:")
print(df.columns.tolist())

# STEP 4: REMOVE DUPLICATE RECORDS

print("\nDuplicate Records:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

# STEP 5: CHECK MISSING VALUES

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# STEP 6: CONVERT TIMESTAMP INTO DATETIME

df['Timestamp'] = pd.to_datetime(
    df['Timestamp'],
    dayfirst=True,
    errors='coerce'
)

# STEP 7: EXTRACT DATE AND TIME FEATURES

df['Hour'] = df['Timestamp'].dt.hour

df['Month'] = df['Timestamp'].dt.month

df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

print("\nDataset After Timestamp Conversion:")
print(df[['Timestamp', 'Hour', 'Month', 'DayOfWeek']].head())

# STEP 8: DROP UNNECESSARY COLUMNS

drop_cols = [
    'Ad_Topic_Line',
    'City',
    'Country',
    'Timestamp'
]

df = df.drop(columns=drop_cols)

print("\nDataset After Removing Unnecessary Columns:")
print(df.head())

print("\nUpdated Shape:")
print(df.shape)

# STEP 9: SEPARATE FEATURES AND TARGET

X = df.drop('Clicked_on_Ad', axis=1)

y = df['Clicked_on_Ad']

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# STEP 10: IDENTIFY NUMERICAL FEATURES

numerical_cols = [
    'Daily_Time_ Spent _on_Site',
    'Age',
    'Area_Income',
    'Daily Internet Usage',
    'Hour',
    'Month',
    'DayOfWeek'
]

# Binary feature

binary_cols = ['Male']

print("\nNumerical Features:")
print(numerical_cols)

print("\nBinary Features:")
print(binary_cols)

# STEP 11: TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# STEP 12: CREATE NUMERICAL PIPELINE

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# STEP 13: CREATE BINARY PIPELINE

binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# STEP 14: CREATE COLUMN TRANSFORMER

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_cols),
    ('binary', binary_pipeline, binary_cols)
])

# STEP 15: FIT AND TRANSFORM TRAINING DATA

X_train_processed = preprocessor.fit_transform(X_train)

# STEP 16: TRANSFORM TESTING DATA

X_test_processed = preprocessor.transform(X_test)

# STEP 17: GET FEATURE NAMES

feature_names = preprocessor.get_feature_names_out()

# STEP 18: CONVERT INTO DATAFRAME

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

# STEP 19: DISPLAY PROCESSED DATA

print("\nProcessed Training Data:")
print(X_train_processed.head())

print("\nProcessed Testing Data:")
print(X_test_processed.head())

# STEP 20: CHECK MISSING VALUES

print("\nMissing Values After Processing:")

print(X_train_processed.isnull().sum().sum())

print(X_test_processed.isnull().sum().sum())

# STEP 21: CHECK SCALING

scaled_cols = [
    'numerical__' + col
    for col in numerical_cols
]

print("\nMean After Scaling:")
print(X_train_processed[scaled_cols].mean())

print("\nStandard Deviation After Scaling:")
print(X_train_processed[scaled_cols].std())

# STEP 22: CHECK TARGET DISTRIBUTION

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

# STEP 23: SAVE PROCESSED DATA

train_data = X_train_processed.copy()

train_data['Clicked_on_Ad'] = y_train

test_data = X_test_processed.copy()

test_data['Clicked_on_Ad'] = y_test

train_data.to_csv(
    "advertising_train_processed.csv",
    index=False
)

test_data.to_csv(
    "advertising_test_processed.csv",
    index=False
)

# STEP 24: FINAL SUMMARY

print("\n========== ADVERTISING PREPROCESSING SUMMARY ==========")

print("Original Records:", len(df))

print("Training Records:", len(X_train_processed))

print("Testing Records:", len(X_test_processed))

print("Total Features:", X_train_processed.shape[1])

print(
    "Missing Values:",
    X_train_processed.isnull().sum().sum()
)

print("\nFinal Training Shape:", X_train_processed.shape)

print("Final Testing Shape:", X_test_processed.shape)

print("\nADVERTISING DATA PREPROCESSING COMPLETED!")