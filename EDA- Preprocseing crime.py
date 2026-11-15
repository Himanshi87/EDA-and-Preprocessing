# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:51:57 2026

@author: Himu
"""

# ==========================================
# US CRIME DATASET - COMPLETE EDA
# ==========================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 2: LOAD DATASET

df = pd.read_csv("D:/data understanding/crime_data (1).csv")

# Rename first column
df.rename(
    columns={'Unnamed: 0': 'State'},
    inplace=True
)

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

plt.figure(figsize=(8, 5))

missing_percentage.plot(kind='bar')

plt.title("Missing Values - Crime Dataset")
plt.xlabel("Columns")
plt.ylabel("Missing Percentage")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 5: CHECK DUPLICATE RECORDS
# ==========================================

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nDuplicate State Names:")
print(df['State'].duplicated().sum())

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
    'Murder', 'Assault', 'UrbanPop', 'Rape'
]

df[numerical_cols].hist(
    figsize=(12, 9),
    bins=10,
    edgecolor='black'
)

plt.suptitle("Distribution of Crime Features")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 9: BOXPLOT FOR ALL NUMERICAL FEATURES
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

correlation = df[numerical_cols].corr()

print("\nCorrelation Matrix:")
print(correlation)

# ==========================================
# STEP 12: CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(9, 7))

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

plt.title("Correlation Heatmap - US Crime Dataset")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 13: MURDER ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(
    df['Murder'],
    bins=10,
    edgecolor='black'
)

plt.title("Murder Rate Distribution")
plt.xlabel("Murder Rate")
plt.ylabel("Number of States")

plt.show()

# ==========================================
# STEP 14: ASSAULT ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(
    df['Assault'],
    bins=10,
    edgecolor='black'
)

plt.title("Assault Rate Distribution")
plt.xlabel("Assault Rate")
plt.ylabel("Number of States")

plt.show()

# ==========================================
# STEP 15: URBAN POPULATION ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(
    df['UrbanPop'],
    bins=10,
    edgecolor='black'
)

plt.title("Urban Population Distribution")
plt.xlabel("Urban Population Percentage")
plt.ylabel("Number of States")

plt.show()

# ==========================================
# STEP 16: RAPE ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(
    df['Rape'],
    bins=10,
    edgecolor='black'
)

plt.title("Rape Rate Distribution")
plt.xlabel("Rape Rate")
plt.ylabel("Number of States")

plt.show()

# ==========================================
# STEP 17: STATE-WISE MURDER ANALYSIS
# ==========================================

murder_sorted = df.sort_values(
    by='Murder',
    ascending=False
)

print("\nTop 10 States by Murder Rate:")
print(
    murder_sorted[
        ['State', 'Murder']
    ].head(10)
)

plt.figure(figsize=(14, 6))

plt.bar(
    murder_sorted['State'],
    murder_sorted['Murder']
)

plt.title("State-wise Murder Rate")
plt.xlabel("States")
plt.ylabel("Murder Rate")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 18: STATE-WISE ASSAULT ANALYSIS
# ==========================================

assault_sorted = df.sort_values(
    by='Assault',
    ascending=False
)

print("\nTop 10 States by Assault Rate:")
print(
    assault_sorted[
        ['State', 'Assault']
    ].head(10)
)

plt.figure(figsize=(14, 6))

plt.bar(
    assault_sorted['State'],
    assault_sorted['Assault']
)

plt.title("State-wise Assault Rate")
plt.xlabel("States")
plt.ylabel("Assault Rate")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 19: STATE-WISE RAPE ANALYSIS
# ==========================================

rape_sorted = df.sort_values(
    by='Rape',
    ascending=False
)

print("\nTop 10 States by Rape Rate:")
print(
    rape_sorted[
        ['State', 'Rape']
    ].head(10)
)

plt.figure(figsize=(14, 6))

plt.bar(
    rape_sorted['State'],
    rape_sorted['Rape']
)

plt.title("State-wise Rape Rate")
plt.xlabel("States")
plt.ylabel("Rape Rate")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 20: MURDER VS ASSAULT
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df['Murder'],
    df['Assault']
)

plt.title("Murder vs Assault")
plt.xlabel("Murder Rate")
plt.ylabel("Assault Rate")

plt.show()

# ==========================================
# STEP 21: URBANPOP VS MURDER
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df['UrbanPop'],
    df['Murder']
)

plt.title("Urban Population vs Murder")
plt.xlabel("Urban Population (%)")
plt.ylabel("Murder Rate")

plt.show()

# ==========================================
# STEP 22: URBANPOP VS ASSAULT
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df['UrbanPop'],
    df['Assault']
)

plt.title("Urban Population vs Assault")
plt.xlabel("Urban Population (%)")
plt.ylabel("Assault Rate")

plt.show()

# ==========================================
# STEP 23: URBANPOP VS RAPE
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df['UrbanPop'],
    df['Rape']
)

plt.title("Urban Population vs Rape")
plt.xlabel("Urban Population (%)")
plt.ylabel("Rape Rate")

plt.show()

# ==========================================
# STEP 24: AVERAGE CRIME RATE ANALYSIS
# ==========================================

crime_average = df[numerical_cols].mean()

print("\nAverage Crime Statistics:")
print(crime_average)

plt.figure(figsize=(9, 5))

crime_average.plot(kind='bar')

plt.title("Average Crime Statistics")
plt.xlabel("Crime Categories")
plt.ylabel("Average Value")

plt.xticks(rotation=0)
plt.show()

# ==========================================
# STEP 25: STATE WITH HIGHEST AND LOWEST
# ==========================================

print("\nHighest Crime Values:")

for col in numerical_cols:

    max_state = df.loc[df[col].idxmax(), 'State']
    max_value = df[col].max()

    print(
        col, ":", max_state,
        "(", max_value, ")"
    )

print("\nLowest Crime Values:")

for col in numerical_cols:

    min_state = df.loc[df[col].idxmin(), 'State']
    min_value = df[col].min()

    print(
        col, ":", min_state,
        "(", min_value, ")"
    )

# ==========================================
# STEP 26: PAIRWISE FEATURE ANALYSIS
# ==========================================

pd.plotting.scatter_matrix(
    df[numerical_cols],
    figsize=(12, 10),
    diagonal='hist'
)

plt.suptitle("Pairwise Relationships - Crime Dataset")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 27: TOP 10 STATES BY OVERALL CRIME
# ==========================================

# Standardize crime variables before combining
# because their numerical scales differ.

crime_zscore = (
    df[numerical_cols] - df[numerical_cols].mean()
) / df[numerical_cols].std()

df['Overall_Crime_Score'] = crime_zscore.mean(axis=1)

overall_sorted = df.sort_values(
    by='Overall_Crime_Score',
    ascending=False
)

print("\nTop 10 States by Standardized Crime Score:")
print(
    overall_sorted[
        ['State', 'Overall_Crime_Score']
    ].head(10)
)

plt.figure(figsize=(12, 6))

plt.bar(
    overall_sorted['State'].head(10),
    overall_sorted['Overall_Crime_Score'].head(10)
)

plt.title("Top 10 States by Standardized Crime Score")
plt.xlabel("States")
plt.ylabel("Average Standardized Crime Score")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# STEP 28: FINAL EDA SUMMARY
# ==========================================

print("\n========== CRIME DATASET EDA SUMMARY ==========")

print("Total Records:", len(df))

print("Total Features:", len(numerical_cols))

print("Missing Values:", df.isnull().sum().sum())

print("Duplicate Records:", df.duplicated().sum())

print("\nAverage Crime Statistics:")
print(df[numerical_cols].mean())

print("\nMaximum Crime Statistics:")
print(df[numerical_cols].max())

print("\nMinimum Crime Statistics:")
print(df[numerical_cols].min())

print("\nEDA COMPLETED SUCCESSFULLY!")


# ==========================================
# US CRIME - DATA PREPROCESSING
# ==========================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# STEP 2: LOAD DATASET

df = pd.read_csv("D:/data understanding/crime_data (1).csv")

print("Original Dataset:")
print(df.head())

print("\nOriginal Shape:", df.shape)

# STEP 3: RENAME STATE COLUMN

df.rename(
    columns={'Unnamed: 0': 'State'},
    inplace=True
)

print("\nUpdated Column Names:")
print(df.columns.tolist())

# STEP 4: REMOVE DUPLICATES

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

# STEP 5: CHECK MISSING VALUES

print("\nMissing Values:")
print(df.isnull().sum())

# STEP 6: SEPARATE STATE AND NUMERICAL FEATURES

state = df['State']

X = df.drop('State', axis=1)

print("\nNumerical Features:")
print(X.head())

print("\nFeature Shape:", X.shape)

# STEP 7: HANDLE MISSING VALUES

imputer = SimpleImputer(strategy='median')

X_imputed = imputer.fit_transform(X)

X_imputed = pd.DataFrame(
    X_imputed,
    columns=X.columns,
    index=X.index
)

print("\nMissing Values After Imputation:")
print(X_imputed.isnull().sum())

# STEP 8: CHECK OUTLIERS USING IQR

print("\nOutlier Detection:")

for col in X_imputed.columns:

    Q1 = X_imputed[col].quantile(0.25)

    Q3 = X_imputed[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    outliers = X_imputed[
        (X_imputed[col] < lower_limit) |
        (X_imputed[col] > upper_limit)
    ]

    print("\nFeature:", col)
    print("Number of Outliers:", len(outliers))

# STEP 9: FEATURE SCALING

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)

print("\nScaled Dataset:")
print(X_scaled.head())

# STEP 10: CHECK SCALING

print("\nMean After Scaling:")
print(X_scaled.mean())

print("\nStandard Deviation After Scaling:")
print(X_scaled.std())

# STEP 11: ADD STATE NAMES

processed_df = X_scaled.copy()

processed_df.insert(
    0,
    'State',
    state
)

print("\nFinal Processed Dataset:")
print(processed_df.head())

# STEP 12: SAVE PROCESSED DATA

processed_df.to_csv(
    "crime_data_processed.csv",
    index=False
)

# Also save imputed but unscaled data

unscaled_df = X_imputed.copy()

unscaled_df.insert(
    0,
    'State',
    state
)

unscaled_df.to_csv(
    "crime_data_imputed.csv",
    index=False
)

print("\nCrime Dataset Preprocessing Completed!")

print("Final Shape:", processed_df.shape)