## Exploratory Data Analysis (EDA) and Data Preprocessing on 5 Datasets

## 📌 Project Overview

This project focuses on performing Exploratory Data Analysis (EDA) and Data Preprocessing on five different datasets using Python. The main objective is to understand the structure of data, identify patterns, detect missing values and outliers, visualize relationships between variables, and prepare clean datasets for further machine learning applications.

The project covers classification, regression, and clustering-related datasets to understand different types of data analysis and preprocessing techniques.

## 🎯 Objectives

- Understand the structure and characteristics of different datasets.
- Identify and handle missing values and duplicate records.
- Perform statistical analysis and data visualization.
- Detect and analyze outliers using boxplots and the IQR method.
- Study correlations and relationships between variables.
- Apply suitable encoding and feature scaling techniques.
- Prepare datasets for machine learning applications.

## 📂 Datasets Used

This project includes the following five datasets:

| No. | Dataset | Records | Features | Target / Purpose |
|---|---|---:|---:|---|
| 1 | Bank Marketing | 45,211 | 32 | `y` |
| 2 | Heart Disease | 303 | 14 | `target` |
| 3 | MT Cars | 32 | 11 | `mpg` (regression) |
| 4 | US Crime | 50 | 5 | Clustering analysis |
| 5 | Advertising | 1,000 | 10 | `Clicked_on_Ad` |

### 1. Bank Marketing Dataset
- Contains customer information and banking campaign details.
- Used to analyze customer behavior and predict whether a customer subscribes to a term deposit.
- Target variable: `y`
- Problem Type: Classification.

### 2. Heart Disease Dataset
- Contains medical attributes related to heart health.
- Used to analyze health-related features and predict the presence of heart disease.
- Target variable: `target`
- Problem Type: Classification.

### 3. MT Cars Dataset
- Contains information about different car models and their specifications.
- Used to analyze relationships between vehicle features such as mileage, horsepower, weight, and fuel efficiency.
- Target variable (optional): `mpg`
- Problem Type: Regression.

### 4. US Crime Dataset
- Contains crime statistics for 50 US states.
- Includes Murder, Assault, Urban Population, and Rape-related data.
- Used to identify patterns and relationships in crime statistics.
- Problem Type: Clustering and Exploratory Analysis.

### 5. Advertising Dataset
- Contains information about internet users, demographics, and online advertising behavior.
- Used to analyze user behavior and predict whether a user clicks on an advertisement.
- Target variable: `Clicked_on_Ad`
- Problem Type: Classification.

## 🔍 Exploratory Data Analysis (EDA)

The following EDA techniques are applied to the datasets:

- **Data Understanding:** Checking dataset shape, columns, data types, and sample records.
- **Missing Value Analysis:** Identifying null and missing values.
- **Duplicate Analysis:** Detecting duplicate records.
- **Statistical Analysis:** Calculating mean, median, standard deviation, minimum, maximum, and quartiles.
- **Univariate Analysis:** Studying individual feature distributions using histograms and count plots.
- **Bivariate Analysis:** Understanding relationships between two variables using scatter plots and other visualizations.
- **Outlier Detection:** Identifying outliers using boxplots and the Interquartile Range (IQR) method.
- **Correlation Analysis:** Analyzing relationships between numerical features using correlation matrices and heatmaps.
- **Target Analysis:** Studying target variable distributions for classification and regression datasets.

## ⚙️ Data Preprocessing

The following preprocessing techniques are used:

- Handling missing values using suitable imputation methods.
- Removing duplicate records.
- Correcting column names and data types where required.
- Encoding categorical variables using suitable encoding techniques.
- Detecting and handling outliers where appropriate.
- Applying feature scaling using StandardScaler or MinMaxScaler.
- Separating independent and dependent variables for supervised learning.
- Splitting data into training and testing sets.
- Preparing numerical features for clustering analysis in the US Crime dataset.

## 🛠️ Technologies and Libraries Used

- **Python** – Programming language.
- **Pandas** – Data manipulation and analysis.
- **NumPy** – Numerical operations.
- **Matplotlib** – Data visualization.
- **Seaborn** – Statistical data visualization.
- **Scikit-learn** – Data preprocessing, feature scaling, and dataset splitting.
- **Jupyter Notebook / Google Colab / Spyder** – Development environments.

## 📁 Project Structure

```text
EDA-and-Data-Preprocessing/
│
├── datasets/
│   ├── bank_data.csv
│   ├── heart disease.csv
│   ├── mtcars.csv
│   ├── crime_data.csv
│   └── advertising.csv
│
├── notebooks/
│   ├── bank_marketing_eda.ipynb
│   ├── heart_disease_eda.ipynb
│   ├── mtcars_eda.ipynb
│   ├── crime_data_eda.ipynb
│   └── advertising_eda.ipynb
│
├── preprocessing/
│   └── data_preprocessing.ipynb
│
├── images/
│   └── visualizations/
│
├── requirements.txt
│
└── README.md
```

*Note: The folder structure is a suggested organization. Adjust filenames according to the files uploaded to the repository.*

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd EDA-and-Data-Preprocessing
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Run the Notebooks

```bash
Spyder
```

Open the required notebook and execute the cells sequentially.

## 📊 Expected Outcomes

- Better understanding of different datasets and their characteristics.
- Identification of missing values, duplicates, and outliers.
- Visualization of data distributions and relationships.
- Improved data quality through preprocessing.
- Preparation of datasets for classification, regression, and clustering applications.
- Development of practical knowledge of Python-based data analysis.

## 🔮 Future Scope

- Apply machine learning algorithms to the preprocessed datasets.
- Perform feature selection and feature engineering.
- Compare model performance using suitable evaluation metrics.
- Explore advanced visualization techniques.
- Build interactive dashboards for data analysis.

## 👩‍💻 Author

**Himanshi Motghare**

Artificial Intelligence and Data Science

## 📄 License

This project is intended for educational and learning purposes.
