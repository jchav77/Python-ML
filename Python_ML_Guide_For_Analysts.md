# Python & Machine Learning Guide for Analysts

> A practical guide for analysts transitioning from SAS, SQL, and Excel to Python and Machine Learning

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Python Fundamentals](#python-fundamentals)
4. [Data Manipulation with Pandas](#data-manipulation-with-pandas)
5. [Data Preprocessing](#data-preprocessing)
6. [Machine Learning Basics](#machine-learning-basics)
7. [Clustering Analysis](#clustering-analysis)
8. [Statistical Testing](#statistical-testing)
9. [Visualization](#visualization)
10. [Project Structure](#project-structure)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)
13. [Resources](#resources)

---

## Introduction

### Who This Guide Is For

This guide is designed for analysts who:
- Have strong analytical skills and statistical knowledge
- Are proficient in SAS, SQL, and/or Excel
- Are new to Python programming
- Want to learn machine learning for business analytics
- Prefer practical, hands-on learning

### What You'll Learn

- Python syntax and data structures
- Data manipulation and preprocessing
- Machine learning concepts (with focus on clustering)
- Statistical hypothesis testing in Python
- Creating production-ready analysis pipelines
- Best practices for reproducible research

### Prerequisites

- Basic statistics knowledge (means, standard deviations, hypothesis testing)
- Understanding of data analysis workflows
- Familiarity with at least one: SAS, SQL, or Excel

---

## Getting Started

### Environment Setup

#### 1. Install Python

Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version
```

#### 2. Install an IDE

**Recommended: Visual Studio Code**
- Download from [code.visualstudio.com](https://code.visualstudio.com/)
- Install Python extension
- Lightweight and powerful

**Alternative: PyCharm Community Edition**
- More features out-of-the-box
- Heavier resource usage

#### 3. Install Essential Packages

Create a `requirements.txt` file:
```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

Install all packages:
```bash
pip install -r requirements.txt
```

### Your First Python Script

Create `hello_analysis.py`:
```python
# Import libraries
import pandas as pd
import numpy as np

# Create sample data
data = {
    'customer_id': [1, 2, 3, 4, 5],
    'age': [25, 35, 45, 55, 65],
    'balance': [1000, 2000, 3000, 4000, 5000]
}

df = pd.DataFrame(data)

# Display first few rows
print(df.head())

# Calculate mean balance
mean_balance = df['balance'].mean()
print(f"Mean balance: ${mean_balance:,.2f}")
```

Run it:
```bash
python hello_analysis.py
```

---

## Python Fundamentals

### For SAS/SQL/Excel Users

Python concepts translated to familiar tools:

| Concept | SAS | SQL | Excel | Python |
|---------|-----|-----|-------|--------|
| Dataset/Table | DATA step | TABLE | Worksheet | DataFrame |
| Column | Variable | Column | Column | Series |
| Missing value | . (dot) | NULL | (blank) | NaN, None |
| Filter rows | WHERE | WHERE | Filter | df[condition] |
| Select columns | KEEP/DROP | SELECT | Select columns | df[['col1', 'col2']] |
| Create column | Variable = | AS | Formula | df['new'] = ... |
| Group by | CLASS | GROUP BY | Pivot Table | groupby() |
| Sort | PROC SORT | ORDER BY | Sort | sort_values() |
| Join | MERGE | JOIN | VLOOKUP | merge() |

### Basic Syntax

#### Variables and Types

```python
# Numbers
age = 30                    # Integer
balance = 1500.50          # Float
rate = 0.05                # Float

# Strings
name = "John Doe"          # String
status = 'active'          # Single or double quotes work

# Booleans
is_active = True           # Boolean (capital T/F)
has_balance = False

# None (like NULL)
middle_name = None         # Represents missing/null

# Type checking
print(type(age))           # <class 'int'>
```

#### Lists (Arrays)

```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
names = ['Alice', 'Bob', 'Charlie']
mixed = [1, 'hello', True, 3.14]

# Accessing elements (0-indexed!)
first = numbers[0]         # 1 (first element)
last = numbers[-1]         # 5 (last element)
subset = numbers[1:3]      # [2, 3] (elements at index 1 and 2)

# Common operations
numbers.append(6)          # Add to end
length = len(numbers)      # Get length
total = sum(numbers)       # Sum all elements
```

#### Dictionaries (Key-Value Pairs)

```python
# Creating dictionaries
customer = {
    'id': 12345,
    'name': 'John Doe',
    'age': 30,
    'balance': 1500.50
}

# Accessing values
name = customer['name']              # 'John Doe'
age = customer.get('age')            # 30
balance = customer.get('salary', 0)  # 0 (default if key missing)

# Adding/updating
customer['email'] = 'john@example.com'
customer['age'] = 31
```

#### Control Flow

**If statements:**
```python
if balance > 1000:
    tier = 'Premium'
elif balance > 500:
    tier = 'Standard'
else:
    tier = 'Basic'

# Inline if (ternary operator)
status = 'Active' if balance > 0 else 'Inactive'
```

**⚠️ Indentation matters in Python!** Code blocks are defined by indentation, not brackets.

**For loops:**
```python
# Loop through range
for i in range(5):              # 0, 1, 2, 3, 4
    print(i)

# Loop through list
for name in names:
    print(f"Hello, {name}")

# Loop with index
for i, name in enumerate(names):
    print(f"{i}: {name}")
```

#### Functions

```python
def calculate_discount(price, discount_rate):
    """
    Calculate discounted price.
    
    Parameters:
    -----------
    price : float
        Original price
    discount_rate : float
        Discount as decimal (0.1 = 10%)
    
    Returns:
    --------
    float
        Discounted price
    """
    discount = price * discount_rate
    final_price = price - discount
    return final_price

# Using the function
original = 100
discounted = calculate_discount(original, 0.15)
print(f"Original: ${original}, Discounted: ${discounted}")
```

### String Operations

```python
text = "Hello World"

# Common operations
text.upper()              # "HELLO WORLD"
text.lower()              # "hello world"
text.replace("World", "Python")  # "Hello Python"
text.split()              # ['Hello', 'World']
len(text)                 # 11

# String formatting (modern way)
name = "Alice"
age = 30
message = f"{name} is {age} years old"  # "Alice is 30 years old"
```

---

## Data Manipulation with Pandas

### Introduction to DataFrames

A DataFrame is like a table in SQL or a dataset in SAS.

```python
import pandas as pd

# Creating a DataFrame from dictionary
data = {
    'customer_id': [1, 2, 3, 4, 5],
    'age': [25, 35, 45, 55, 65],
    'balance': [1000, 2000, 3000, 4000, 5000],
    'status': ['active', 'active', 'inactive', 'active', 'inactive']
}

df = pd.DataFrame(data)
```

### Loading Data

```python
# From CSV
df = pd.read_csv('data.csv')

# From Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# From SQL database
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM customers", conn)

# From multiple CSVs
import glob
files = glob.glob('data/*.csv')
df = pd.concat([pd.read_csv(f) for f in files])
```

### Exploring Data

```python
# Quick overview
df.head()           # First 5 rows
df.tail()           # Last 5 rows
df.shape            # (rows, columns)
df.columns          # Column names
df.dtypes           # Data types
df.info()           # Summary of DataFrame

# Statistical summary
df.describe()       # Summary statistics for numeric columns
df['column'].describe()  # Statistics for specific column

# Unique values
df['status'].unique()        # Array of unique values
df['status'].nunique()       # Count of unique values
df['status'].value_counts()  # Frequency of each value
```

### Selecting Data

**Selecting columns:**
```python
# Single column (returns Series)
ages = df['age']

# Multiple columns (returns DataFrame)
subset = df[['customer_id', 'age', 'balance']]

# Using loc (label-based)
subset = df.loc[:, ['age', 'balance']]
```

**Selecting rows:**
```python
# By position (iloc - integer location)
first_row = df.iloc[0]          # First row
first_5 = df.iloc[0:5]          # First 5 rows
last_row = df.iloc[-1]          # Last row

# By condition (boolean indexing)
active = df[df['status'] == 'active']
high_balance = df[df['balance'] > 2000]

# Multiple conditions (use parentheses and & or |)
filtered = df[(df['balance'] > 2000) & (df['status'] == 'active')]
filtered = df[(df['age'] < 30) | (df['age'] > 60)]
```

### Creating and Modifying Columns

```python
# Create new column
df['balance_category'] = 'Low'  # Set all to 'Low'
df['balance_squared'] = df['balance'] ** 2
df['age_group'] = df['age'] // 10  # Integer division

# Conditional column creation
df['tier'] = df['balance'].apply(
    lambda x: 'High' if x > 3000 else ('Medium' if x > 1500 else 'Low')
)

# Using np.where (like SQL CASE WHEN)
import numpy as np
df['senior'] = np.where(df['age'] >= 55, 'Yes', 'No')

# Rename columns
df = df.rename(columns={'customer_id': 'id', 'balance': 'account_balance'})
```

### Filtering and Sorting

```python
# Filter rows
active_customers = df[df['status'] == 'active']
high_value = df[df['balance'] > df['balance'].mean()]

# Filter with isin
selected = df[df['status'].isin(['active', 'pending'])]

# Sort
df_sorted = df.sort_values('balance', ascending=False)
df_sorted = df.sort_values(['status', 'balance'], ascending=[True, False])
```

### Grouping and Aggregating

```python
# Group by single column
by_status = df.groupby('status')['balance'].mean()

# Group by multiple columns
by_status_age = df.groupby(['status', 'age_group'])['balance'].sum()

# Multiple aggregations
summary = df.groupby('status').agg({
    'balance': ['mean', 'sum', 'count'],
    'age': ['min', 'max', 'median']
})

# Custom aggregations
summary = df.groupby('status').agg(
    avg_balance=('balance', 'mean'),
    total_balance=('balance', 'sum'),
    customer_count=('customer_id', 'count')
)
```

### Joining/Merging DataFrames

```python
# Inner join (like SQL INNER JOIN)
merged = pd.merge(df1, df2, on='customer_id', how='inner')

# Left join (like SQL LEFT JOIN)
merged = pd.merge(df1, df2, on='customer_id', how='left')

# Join on different column names
merged = pd.merge(df1, df2, left_on='id', right_on='customer_id')

# Join on multiple columns
merged = pd.merge(df1, df2, on=['customer_id', 'date'])

# Concatenate (stack DataFrames)
combined = pd.concat([df1, df2], axis=0)  # Stack vertically
combined = pd.concat([df1, df2], axis=1)  # Stack horizontally
```

### Handling Missing Values

```python
# Check for missing values
df.isnull().sum()           # Count missing per column
df.isnull().any()           # Which columns have any missing
total_missing = df.isnull().sum().sum()

# Remove missing values
df_clean = df.dropna()                    # Remove rows with ANY missing
df_clean = df.dropna(subset=['age'])      # Remove only if 'age' is missing
df_clean = df.dropna(thresh=3)            # Keep rows with at least 3 non-null

# Fill missing values
df['age'] = df['age'].fillna(0)                    # Fill with 0
df['age'] = df['age'].fillna(df['age'].mean())     # Fill with mean
df['age'] = df['age'].fillna(method='ffill')       # Forward fill
df = df.fillna({'age': 0, 'balance': df['balance'].median()})  # Different per column
```

---

## Data Preprocessing

### Why Preprocessing Matters

Machine learning algorithms require:
- **Numeric data** (no text categories)
- **Same scale** (features should be comparable)
- **No missing values** (or properly handled)
- **Clean data** (no outliers or errors)

### Encoding Categorical Variables

#### One-Hot Encoding

**What it does:** Converts categories into binary (0/1) columns

**When to use:** For categorical variables with no natural order

```python
# Before
# status: ['active', 'inactive', 'pending']

# After one-hot encoding
# status_active: [1, 0, 0]
# status_inactive: [0, 1, 0]
# status_pending: [0, 0, 1]

# Using pandas
df_encoded = pd.get_dummies(df['status'], prefix='status')

# For multiple columns
df_encoded = pd.get_dummies(df, columns=['status', 'category'])

# Using scikit-learn (more control)
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['status']])
```

#### Label Encoding

**What it does:** Converts categories to integers (0, 1, 2, ...)

**When to use:** For ordinal categories (low, medium, high) or tree-based models

```python
from sklearn.preprocessing import LabelEncoder

# Encode a single column
le = LabelEncoder()
df['status_encoded'] = le.fit_transform(df['status'])

# Map manually (more control)
status_map = {'inactive': 0, 'active': 1, 'premium': 2}
df['status_encoded'] = df['status'].map(status_map)
```

### Feature Scaling

#### Why Scale?

Algorithms like K-means, SVM, and neural networks are sensitive to feature scales. If one feature ranges from 0-1 and another from 0-10000, the algorithm will treat the larger-scale feature as more important.

#### Standardization (Z-score Normalization)

**Formula:** (x - mean) / std  
**Result:** Mean = 0, Std = 1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['age', 'balance']])

# Save scaler for later use on new data
import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```

#### Min-Max Normalization

**Formula:** (x - min) / (max - min)  
**Result:** Range = [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['age', 'balance']])
```

### Handling Outliers

```python
# Identify outliers using IQR method
Q1 = df['balance'].quantile(0.25)
Q3 = df['balance'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outliers
df_no_outliers = df[(df['balance'] >= lower_bound) & (df['balance'] <= upper_bound)]

# Cap outliers instead of removing
df['balance_capped'] = df['balance'].clip(lower=lower_bound, upper=upper_bound)
```

### Train-Test Split

Always split data before training to evaluate model performance on unseen data.

```python
from sklearn.model_selection import train_test_split

# Split features (X) and target (y)
X = df[['age', 'balance', 'status_active']]
y = df['purchased']

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42  # For reproducibility
)
```

---

## Machine Learning Basics

### Supervised vs. Unsupervised Learning

| Type | Definition | Examples | Use Cases |
|------|------------|----------|-----------|
| **Supervised** | Learn from labeled data (input → output pairs) | Classification, Regression | Predict customer churn, forecast sales |
| **Unsupervised** | Find patterns in unlabeled data | Clustering, Dimensionality Reduction | Customer segmentation, anomaly detection |

### Scikit-learn Workflow

Standard pattern for all ML in scikit-learn:

```python
from sklearn.some_module import SomeModel

# 1. Initialize the model
model = SomeModel(parameter1=value1, parameter2=value2)

# 2. Train the model
model.fit(X_train, y_train)  # Supervised
model.fit(X_train)           # Unsupervised

# 3. Make predictions
predictions = model.predict(X_test)

# 4. Evaluate (for supervised learning)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predictions)
```

### Common ML Tasks

#### Classification

**Goal:** Predict a category (yes/no, high/medium/low)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Train model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

#### Regression

**Goal:** Predict a continuous number (price, amount, percentage)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict
y_pred = reg.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, R²: {r2:.2f}")
```

---

## Clustering Analysis

### What is Clustering?

**Definition:** Grouping similar observations together without predefined labels.

**Business use cases:**
- Customer segmentation
- Product categorization
- Anomaly detection
- Market basket analysis

### K-Means Clustering

**How it works:**
1. Choose K (number of clusters)
2. Randomly place K centroids
3. Assign each point to nearest centroid
4. Move centroids to center of their points
5. Repeat steps 3-4 until convergence

#### Basic Implementation

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
X = df[['age', 'balance', 'tenure']].values

# Standardize features (important for K-means!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-means
kmeans = KMeans(
    n_clusters=5,      # Number of clusters
    random_state=42,   # For reproducibility
    n_init=10          # Number of times to run with different centroids
)

cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to DataFrame
df['cluster'] = cluster_labels
```

#### Choosing Optimal K

**Method 1: Elbow Method**

Plot inertia (within-cluster sum of squares) for different K values:

```python
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

Look for the "elbow" - where adding more clusters doesn't significantly reduce inertia.

**Method 2: Silhouette Score**

Measures how well-separated clusters are:

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()
```

**Score interpretation:**
- 0.7 - 1.0: Strong cluster structure
- 0.5 - 0.7: Reasonable structure
- 0.25 - 0.5: Weak structure
- < 0.25: No substantial structure

#### Analyzing Clusters

**Profile each cluster:**

```python
# Summary statistics per cluster
cluster_summary = df.groupby('cluster').agg({
    'age': ['mean', 'std', 'min', 'max'],
    'balance': ['mean', 'std', 'min', 'max'],
    'tenure': ['mean', 'std'],
    'customer_id': 'count'  # Size of each cluster
})

print(cluster_summary)

# Most common features per cluster
for cluster_id in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {len(cluster_data)} ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"  Avg Age: {cluster_data['age'].mean():.1f}")
    print(f"  Avg Balance: ${cluster_data['balance'].mean():,.2f}")
```

**Visualize clusters (2D):**

```python
# For 2D visualization, use first 2 features
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df['age'], 
    df['balance'], 
    c=df['cluster'], 
    cmap='viridis', 
    alpha=0.6
)
plt.xlabel('Age')
plt.ylabel('Balance')
plt.title('Customer Clusters')
plt.colorbar(scatter, label='Cluster')
plt.show()
```

### Other Clustering Algorithms

#### Hierarchical Clustering

**When to use:** Want to see hierarchy of clusters, or K is unknown

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Fit model
hierarchical = AgglomerativeClustering(n_clusters=5)
clusters = hierarchical.fit_predict(X_scaled)

# Create dendrogram
linkage_matrix = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer Index')
plt.ylabel('Distance')
plt.show()
```

#### DBSCAN

**When to use:** Clusters have arbitrary shapes, or you want to detect outliers

```python
from sklearn.cluster import DBSCAN

# Fit model
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# -1 indicates outliers
n_outliers = (clusters == -1).sum()
print(f"Outliers detected: {n_outliers}")
```

---

## Statistical Testing

### Why Statistical Testing?

Answers the question: "Is this difference real or just random chance?"

**Common business questions:**
- Did the new feature increase conversion rates?
- Do different customer segments have different retention rates?
- Is the difference between test and control groups significant?

### Hypothesis Testing Framework

**1. State hypotheses:**
- Null hypothesis (H₀): No difference exists
- Alternative hypothesis (H₁): A difference exists

**2. Choose significance level (α):**
- α = 0.10 → 90% confidence
- α = 0.05 → 95% confidence (most common)
- α = 0.01 → 99% confidence

**3. Calculate test statistic and p-value**

**4. Make decision:**
- If p-value < α: Reject null (difference is significant)
- If p-value ≥ α: Fail to reject null (difference not significant)

### Comparing Two Groups

#### T-Test (Continuous Variables)

**Use when:** Comparing means of two groups

```python
from scipy.stats import ttest_ind

# Example: Compare balance between two groups
group1 = df[df['segment'] == 'A']['balance']
group2 = df[df['segment'] == 'B']['balance']

# Perform t-test
t_stat, p_value = ttest_ind(group1, group2)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Significant difference (reject null hypothesis)")
else:
    print("✗ No significant difference (fail to reject null)")

# Effect size (Cohen's d)
mean_diff = group1.mean() - group2.mean()
pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
cohens_d = mean_diff / pooled_std
print(f"Effect size (Cohen's d): {cohens_d:.3f}")
```

**Interpretation of Cohen's d:**
- |d| < 0.2: Small effect
- |d| 0.2-0.5: Medium effect
- |d| > 0.8: Large effect

#### Two-Proportion Z-Test (Binary Variables)

**Use when:** Comparing percentages/rates between two groups

```python
from scipy import stats
import numpy as np

def two_proportion_z_test(successes1, n1, successes2, n2, alpha=0.05):
    """
    Perform two-proportion z-test.
    
    Parameters:
    -----------
    successes1, successes2 : int
        Number of successes in each group
    n1, n2 : int
        Sample sizes
    alpha : float
        Significance level
    """
    # Calculate proportions
    p1 = successes1 / n1
    p2 = successes2 / n2
    
    # Pooled proportion
    p_pool = (successes1 + successes2) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    # Z-statistic
    z_stat = (p1 - p2) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Results
    print(f"Group 1: {p1:.1%} ({successes1}/{n1})")
    print(f"Group 2: {p2:.1%} ({successes2}/{n2})")
    print(f"Difference: {(p1-p2)*100:.2f} percentage points")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"✓ Significant at {alpha} level")
    else:
        print(f"✗ Not significant at {alpha} level")
    
    return {'p1': p1, 'p2': p2, 'z_stat': z_stat, 'p_value': p_value}

# Example usage
# Test group: 120 conversions out of 1000
# Control group: 95 conversions out of 1000
results = two_proportion_z_test(120, 1000, 95, 1000, alpha=0.05)
```

### Multiple Group Comparisons

#### ANOVA (Analysis of Variance)

**Use when:** Comparing means across 3+ groups

```python
from scipy.stats import f_oneway

# Compare balance across multiple segments
group_a = df[df['segment'] == 'A']['balance']
group_b = df[df['segment'] == 'B']['balance']
group_c = df[df['segment'] == 'C']['balance']

f_stat, p_value = f_oneway(group_a, group_b, group_c)

print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ At least one group differs significantly")
else:
    print("✗ No significant difference between groups")
```

#### Chi-Square Test (Categorical Variables)

**Use when:** Testing association between two categorical variables

```python
from scipy.stats import chi2_contingency

# Create contingency table
contingency_table = pd.crosstab(df['segment'], df['converted'])

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("✓ Significant association")
else:
    print("✗ No significant association")
```

### Sample Size Considerations

**Minimum sample sizes for reliable tests:**

| Test Type | Minimum per Group | Notes |
|-----------|-------------------|-------|
| T-test | 30 | Can go lower if data is normally distributed |
| Two-proportion | 10 successes AND 10 failures | In EACH group |
| Chi-square | 5 expected count | In each cell of contingency table |

```python
# Check sample size adequacy for proportions
def check_proportion_sample_size(successes, failures):
    """Check if sample size is adequate for proportion test."""
    if min(successes, failures) < 10:
        return False, f"Insufficient: need ≥10 successes and failures"
    return True, f"Adequate: {successes} successes, {failures} failures"

is_adequate, message = check_proportion_sample_size(120, 880)
print(message)
```

---

## Visualization

### Matplotlib Basics

```python
import matplotlib.pyplot as plt

# Simple line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.grid(True)
plt.show()

# Multiple lines
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='Group 1', marker='o')
plt.plot(x, y2, label='Group 2', marker='s')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.legend()
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['balance'], alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Balance')
plt.title('Age vs Balance')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(df['balance'], bins=30, edgecolor='black')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.title('Distribution of Balance')
plt.show()
```

### Seaborn for Statistical Plots

```python
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(df['balance'], kde=True, bins=30)
plt.title('Balance Distribution')
plt.show()

# Box plot (compare distributions)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='segment', y='balance')
plt.title('Balance by Segment')
plt.show()

# Violin plot (more detailed than box plot)
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='segment', y='balance')
plt.title('Balance Distribution by Segment')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df[['age', 'balance', 'tenure']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Pair plot (scatter matrix)
sns.pairplot(df[['age', 'balance', 'tenure', 'cluster']], 
             hue='cluster', 
             diag_kind='kde')
plt.show()
```

### Saving Figures

```python
# Save to file
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
plt.savefig('figure.pdf', bbox_inches='tight')

# High-resolution for publication
plt.savefig('figure.png', dpi=600, bbox_inches='tight', transparent=True)
```

---

## Project Structure

### Recommended Directory Structure

```
project_name/
├── data/
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned, transformed data
│   └── external/               # External data sources
├── notebooks/
│   ├── 01_exploration.ipynb    # Initial data exploration
│   ├── 02_preprocessing.ipynb  # Data cleaning
│   └── 03_modeling.ipynb       # Model development
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_processing.py
│   ├── features.py
│   ├── models.py
│   └── visualization.py
├── outputs/
│   ├── figures/                # Generated plots
│   ├── models/                 # Saved models
│   └── results/                # Analysis results (CSV, Excel)
├── tests/                      # Unit tests
│   └── test_data_processing.py
├── requirements.txt            # Package dependencies
├── README.md                   # Project documentation
└── main.py                     # Main execution script
```

### Modular Code Structure

**Example: `src/data_processing.py`**

```python
"""
Data processing utilities
"""
import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from CSV or Excel."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format")

def clean_data(df):
    """Remove duplicates and handle missing values."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna(subset=['customer_id'])  # Remove if ID missing
    df['age'] = df['age'].fillna(df['age'].median())
    
    return df

def encode_features(df, categorical_columns):
    """One-hot encode categorical variables."""
    return pd.get_dummies(df, columns=categorical_columns)
```

**Example: `main.py`**

```python
"""
Main analysis pipeline
"""
from src.data_processing import load_data, clean_data, encode_features
from src.models import perform_clustering
from src.visualization import plot_clusters

def main():
    # Load data
    print("Loading data...")
    df = load_data('data/raw/customer_data.csv')
    
    # Clean data
    print("Cleaning data...")
    df = clean_data(df)
    
    # Encode features
    print("Encoding features...")
    categorical_cols = ['status', 'segment']
    df = encode_features(df, categorical_cols)
    
    # Perform clustering
    print("Performing clustering...")
    df, model = perform_clustering(df, n_clusters=5)
    
    # Visualize
    print("Creating visualizations...")
    plot_clusters(df, 'outputs/figures/clusters.png')
    
    # Save results
    print("Saving results...")
    df.to_csv('outputs/results/clustered_customers.csv', index=False)
    
    print("✓ Analysis complete!")

if __name__ == "__main__":
    main()
```

---

## Best Practices

### Code Quality

#### 1. Use Meaningful Variable Names

```python
# Bad
df1 = pd.read_csv('data.csv')
x = df1[['c1', 'c2']]

# Good
customer_data = pd.read_csv('customer_data.csv')
features = customer_data[['age', 'balance']]
```

#### 2. Add Comments and Docstrings

```python
def calculate_metrics(predictions, actuals):
    """
    Calculate performance metrics for predictions.
    
    Parameters:
    -----------
    predictions : array-like
        Predicted values
    actuals : array-like
        Actual values
    
    Returns:
    --------
    dict
        Dictionary containing MSE, RMSE, and R²
    """
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (mse / np.var(actuals))
    
    return {'mse': mse, 'rmse': rmse, 'r2': r2}
```

#### 3. Handle Errors Gracefully

```python
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Error: data.csv not found!")
    print("Please check the file path.")
except Exception as e:
    print(f"Unexpected error: {e}")
```

#### 4. Avoid Magic Numbers

```python
# Bad
if balance > 5000:
    tier = 'premium'

# Good
PREMIUM_THRESHOLD = 5000
if balance > PREMIUM_THRESHOLD:
    tier = 'premium'
```

### Reproducibility

#### 1. Set Random Seeds

```python
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Use in sklearn functions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
```

#### 2. Version Control Your Dependencies

Create `requirements.txt`:
```bash
pip freeze > requirements.txt
```

Install from requirements:
```bash
pip install -r requirements.txt
```

#### 3. Document Your Analysis

Create a `README.md` with:
- Project description
- Data sources
- How to run the analysis
- Key findings
- Dependencies

### Performance

#### 1. Vectorize Operations (Avoid Loops)

```python
# Slow: Loop through rows
for i in range(len(df)):
    df.loc[i, 'total'] = df.loc[i, 'price'] * df.loc[i, 'quantity']

# Fast: Vectorized operation
df['total'] = df['price'] * df['quantity']
```

#### 2. Use Efficient Data Types

```python
# Before
df['customer_id'].dtype  # int64 (uses 8 bytes per value)

# After (if values fit in smaller range)
df['customer_id'] = df['customer_id'].astype('int32')  # Uses 4 bytes

# For categories with few unique values
df['status'] = df['status'].astype('category')
```

#### 3. Filter Early

```python
# Slow: Load all data then filter
df = pd.read_csv('large_file.csv')
df = df[df['year'] == 2024]

# Fast: Filter during load (if CSV is structured)
df = pd.read_csv('large_file.csv')
df = df[df['year'] == 2024]  # Still need to load first with CSV

# Better: Use databases for large data
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table WHERE year = 2024", conn)
```

---

## Troubleshooting

### Common Errors and Solutions

#### ImportError / ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
pip install pandas
# or
pip install -r requirements.txt
```

#### KeyError

**Error:**
```
KeyError: 'column_name'
```

**Cause:** Column doesn't exist in DataFrame

**Solution:**
```python
# Check available columns
print(df.columns.tolist())

# Use .get() method for safe access
value = df.get('column_name', 'default_value')
```

#### ValueError: could not convert string to float

**Cause:** Trying to do math on text data

**Solution:**
```python
# Check data types
print(df.dtypes)

# Convert to numeric (coerce errors to NaN)
df['column'] = pd.to_numeric(df['column'], errors='coerce')
```

#### SettingWithCopyWarning

**Warning:**
```
SettingWithCopyWarning: A value is trying to be set on a copy of a slice
```

**Solution:**
```python
# Use .copy() to create explicit copy
df_subset = df[df['age'] > 30].copy()
df_subset['new_column'] = value

# Or use .loc for setting values
df.loc[df['age'] > 30, 'new_column'] = value
```

### Debugging Tips

#### 1. Use Print Statements

```python
print(f"DataFrame shape: {df.shape}")
print(f"Column types: {df.dtypes}")
print(f"First few rows:\n{df.head()}")
print(f"Variable value: {variable}")
print(f"Variable type: {type(variable)}")
```

#### 2. Check Data at Each Step

```python
# After loading
print(f"Loaded {len(df)} rows")

# After filtering
print(f"After filtering: {len(df_filtered)} rows")

# After transformation
print(f"Column created: {df['new_column'].describe()}")
```

#### 3. Use Jupyter Notebooks for Interactive Debugging

```python
# In Jupyter, you can run cells individually
# and inspect variables in real-time

df.head()  # See data
df.info()  # Check structure
df.describe()  # Summary stats
```

#### 4. Handle Edge Cases

```python
# Check for empty DataFrame
if df.empty:
    print("Warning: DataFrame is empty!")
    
# Check for missing values
if df.isnull().any().any():
    print("Warning: Missing values detected!")
    print(df.isnull().sum())

# Check for duplicate columns
if df.columns.duplicated().any():
    print("Warning: Duplicate column names!")
```

---

## Resources

### Official Documentation

- **Python:** [python.org/doc](https://docs.python.org/)
- **Pandas:** [pandas.pydata.org/docs](https://pandas.pydata.org/docs/)
- **NumPy:** [numpy.org/doc](https://numpy.org/doc/)
- **Scikit-learn:** [scikit-learn.org/stable](https://scikit-learn.org/stable/)
- **Matplotlib:** [matplotlib.org/stable](https://matplotlib.org/stable/)
- **Seaborn:** [seaborn.pydata.org](https://seaborn.pydata.org/)

### Learning Resources

#### Online Courses
- **Coursera:** "Applied Data Science with Python" by University of Michigan
- **DataCamp:** Python for Data Science track
- **Kaggle Learn:** Free micro-courses on Python, ML, and pandas

#### Books
- *Python for Data Analysis* by Wes McKinney (creator of pandas)
- *Hands-On Machine Learning* by Aurélien Géron
- *Introduction to Statistical Learning* (available free online)

#### Practice Platforms
- **Kaggle:** Real datasets and competitions
- **LeetCode:** Python coding practice
- **HackerRank:** Python challenges

### Cheat Sheets

- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [NumPy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
- [Scikit-learn Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)
- [Matplotlib Cheat Sheet](https://matplotlib.org/cheatsheets/)

### Community Resources

- **Stack Overflow:** [stackoverflow.com](https://stackoverflow.com/) - Q&A for coding problems
- **GitHub:** Search for example projects and code
- **Reddit:** r/learnpython, r/datascience, r/MachineLearning
- **Medium:** Articles and tutorials on data science topics

### When You Get Stuck

1. **Read the error message carefully** - Python errors are usually informative
2. **Google the exact error message** - Chances are someone has solved it
3. **Check the documentation** - Official docs are often the best resource
4. **Ask specific questions** - "How do I group by multiple columns in pandas?"
5. **Share minimal reproducible examples** - Makes it easier for others to help

---

## Next Steps

### Week 1: Python Fundamentals
- [ ] Install Python and required packages
- [ ] Complete basic Python syntax exercises
- [ ] Practice with pandas DataFrames
- [ ] Load and explore a simple dataset

### Week 2: Data Analysis
- [ ] Practice filtering, grouping, and joining data
- [ ] Create visualizations with matplotlib
- [ ] Handle missing values and outliers
- [ ] Perform basic statistical analysis

### Week 3: Machine Learning Basics
- [ ] Understand train-test split
- [ ] Implement your first classification model
- [ ] Perform K-means clustering
- [ ] Interpret model results

### Week 4: Project
- [ ] Choose a business problem
- [ ] Clean and prepare data
- [ ] Build and evaluate models
- [ ] Document findings and recommendations

### Continuous Learning
- Participate in Kaggle competitions
- Read data science blogs and papers
- Contribute to open-source projects
- Build a portfolio of analyses

---

## Contributing

This is a living document. Suggestions for improvements are welcome!

**How to contribute:**
1. Fork the repository
2. Make your changes
3. Submit a pull request
4. Provide clear description of improvements

---

## License

This guide is provided as-is for educational purposes.

---

**Last Updated:** October 2025

**Author:** Created for analysts transitioning to Python and ML

**Version:** 1.0

---

## Appendix: Quick Reference

### Essential Pandas Operations

```python
# Loading data
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')

# Exploring
df.head()                    # First 5 rows
df.info()                    # Structure and types
df.describe()                # Summary statistics
df['col'].value_counts()     # Frequency counts

# Selecting
df['col']                    # Single column
df[['col1', 'col2']]        # Multiple columns
df[df['col'] > 5]           # Filter rows
df.loc[rows, columns]        # Label-based selection
df.iloc[rows, columns]       # Position-based selection

# Creating columns
df['new'] = df['col1'] + df['col2']
df['new'] = df['col'].apply(lambda x: x * 2)

# Grouping
df.groupby('col')['value'].mean()
df.groupby(['col1', 'col2']).agg({'value': 'sum'})

# Merging
pd.merge(df1, df2, on='key', how='left')

# Missing values
df.isnull().sum()            # Count missing
df.dropna()                  # Remove missing
df.fillna(0)                 # Fill with value
```

### Essential Scikit-learn Pattern

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Prepare data
X = df[feature_columns]
y = df[target_column]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train
model = KMeans(n_clusters=5, random_state=42)
model.fit(X_train_scaled)

# 5. Predict
predictions = model.predict(X_test_scaled)

# 6. Evaluate
score = some_metric(y_test, predictions)
```

---

**Happy Learning! 🚀**
