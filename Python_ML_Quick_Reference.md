# Python & ML Quick Reference Cheat Sheet

> Fast lookup for common operations in data analysis and machine learning

---

## Table of Contents
- [Python Basics](#python-basics)
- [Pandas Operations](#pandas-operations)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning](#machine-learning)
- [Statistical Testing](#statistical-testing)
- [Visualization](#visualization)

---

## Python Basics

### Data Types
```python
x = 5                    # int
x = 5.0                  # float
x = "hello"              # string
x = True                 # boolean
x = [1, 2, 3]           # list
x = {'key': 'value'}    # dictionary
x = None                # null/missing
```

### Lists
```python
items = [1, 2, 3, 4, 5]
items[0]                # First element (0-indexed)
items[-1]               # Last element
items[1:3]              # Slice: [2, 3]
items.append(6)         # Add to end
len(items)              # Length
sum(items)              # Sum
```

### Dictionaries
```python
d = {'name': 'John', 'age': 30}
d['name']               # 'John'
d.get('salary', 0)      # 0 (default if missing)
d['email'] = 'john@...' # Add/update
d.keys()                # All keys
d.values()              # All values
```

### Control Flow
```python
# If-else
if x > 10:
    print("Big")
elif x > 5:
    print("Medium")
else:
    print("Small")

# For loop
for i in range(5):      # 0 to 4
    print(i)
    
for item in items:      # Iterate list
    print(item)
```

### Functions
```python
def calculate(x, y):
    """Function documentation"""
    result = x + y
    return result

output = calculate(5, 3)  # 8
```

---

## Pandas Operations

### Loading Data
```python
import pandas as pd

df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')
df = pd.read_sql(query, connection)
```

### Exploring Data
```python
df.head()               # First 5 rows
df.tail()               # Last 5 rows
df.shape                # (rows, columns)
df.columns              # Column names
df.dtypes               # Data types
df.info()               # Overview
df.describe()           # Statistics
df['col'].unique()      # Unique values
df['col'].nunique()     # Count unique
df['col'].value_counts() # Frequency
```

### Selecting Data
```python
# Columns
df['col']               # Single column (Series)
df[['col1', 'col2']]   # Multiple columns (DataFrame)

# Rows
df.iloc[0]              # First row by position
df.iloc[0:5]           # First 5 rows
df.loc[0:5]            # Rows 0-5 by label

# Filtering
df[df['age'] > 30]     # Single condition
df[(df['age'] > 30) & (df['status'] == 'active')]  # Multiple
df[df['status'].isin(['active', 'pending'])]        # isin
```

### Creating/Modifying Columns
```python
df['new'] = 0                           # Set all to 0
df['total'] = df['price'] * df['qty']  # Calculate
df['category'] = df['amount'].apply(lambda x: 'High' if x > 100 else 'Low')

# Conditional
import numpy as np
df['flag'] = np.where(df['value'] > 10, 'Yes', 'No')

# Rename
df = df.rename(columns={'old_name': 'new_name'})

# Drop columns
df = df.drop(columns=['col1', 'col2'])
```

### Grouping & Aggregating
```python
# Single aggregation
df.groupby('category')['amount'].mean()
df.groupby('category')['amount'].agg(['mean', 'sum', 'count'])

# Multiple aggregations
df.groupby('category').agg({
    'amount': ['mean', 'sum'],
    'age': 'median'
})

# Custom aggregation names
df.groupby('status').agg(
    avg_balance=('balance', 'mean'),
    total=('balance', 'sum'),
    count=('id', 'count')
)
```

### Sorting
```python
df.sort_values('column')                    # Ascending
df.sort_values('column', ascending=False)   # Descending
df.sort_values(['col1', 'col2'])           # Multiple
```

### Merging/Joining
```python
# Merge (like SQL JOIN)
pd.merge(df1, df2, on='key', how='left')   # Left join
pd.merge(df1, df2, on='key', how='inner')  # Inner join
pd.merge(df1, df2, left_on='id1', right_on='id2')  # Different keys

# Concatenate (stack)
pd.concat([df1, df2], axis=0)  # Vertical (rows)
pd.concat([df1, df2], axis=1)  # Horizontal (columns)
```

### Missing Values
```python
df.isnull().sum()               # Count missing per column
df.dropna()                     # Remove rows with ANY missing
df.dropna(subset=['col'])       # Remove if specific column missing
df['col'].fillna(0)            # Fill with value
df['col'].fillna(df['col'].mean())  # Fill with mean
df.fillna(method='ffill')      # Forward fill
```

### Basic Statistics
```python
df['col'].mean()                # Mean
df['col'].median()              # Median
df['col'].std()                 # Standard deviation
df['col'].min()                 # Minimum
df['col'].max()                 # Maximum
df['col'].quantile(0.25)       # 25th percentile
df['col'].sum()                 # Sum
df['col'].count()              # Count non-null
```

---

## Data Preprocessing

### One-Hot Encoding
```python
# Simple method
df_encoded = pd.get_dummies(df['category'], prefix='cat')

# Multiple columns
df_encoded = pd.get_dummies(df, columns=['col1', 'col2'])

# Scikit-learn method
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['category']])
```

### Standardization
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data
```

### Min-Max Normalization
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['encoded'] = le.fit_transform(df['category'])

# Manual mapping
mapping = {'low': 0, 'medium': 1, 'high': 2}
df['encoded'] = df['category'].map(mapping)
```

### Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for test
    random_state=42     # For reproducibility
)
```

---

## Machine Learning

### General Pattern
```python
from sklearn.some_module import SomeModel

# 1. Initialize
model = SomeModel(parameter=value)

# 2. Train
model.fit(X_train, y_train)

# 3. Predict
predictions = model.predict(X_test)

# 4. Evaluate
score = some_metric(y_test, predictions)
```

### K-Means Clustering
```python
from sklearn.cluster import KMeans

# Fit model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Add to DataFrame
df['cluster'] = cluster_labels

# Analyze clusters
df.groupby('cluster').mean()
df.groupby('cluster').size()
```

### Finding Optimal K

**Elbow Method:**
```python
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot and look for "elbow"
plt.plot(range(2, 11), inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()
```

**Silhouette Score:**
```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)
    
# Higher score = better separation
```

### Classification
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))
```

### Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")
```

---

## Statistical Testing

### T-Test (Compare Means)
```python
from scipy.stats import ttest_ind

group1 = df[df['segment'] == 'A']['metric']
group2 = df[df['segment'] == 'B']['metric']

t_stat, p_value = ttest_ind(group1, group2)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference")
```

### Two-Proportion Z-Test
```python
from scipy import stats
import numpy as np

def proportion_test(success1, n1, success2, n2):
    p1 = success1 / n1
    p2 = success2 / n2
    p_pool = (success1 + success2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p1, p2, z, p_value

# Usage
p1, p2, z, p = proportion_test(120, 1000, 95, 1000)
print(f"P-value: {p:.4f}")
```

### ANOVA (3+ Groups)
```python
from scipy.stats import f_oneway

group_a = df[df['group'] == 'A']['metric']
group_b = df[df['group'] == 'B']['metric']
group_c = df[df['group'] == 'C']['metric']

f_stat, p_value = f_oneway(group_a, group_b, group_c)
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {p_value:.4f}")
```

### Chi-Square Test
```python
from scipy.stats import chi2_contingency

contingency = pd.crosstab(df['var1'], df['var2'])
chi2, p_value, dof, expected = chi2_contingency(contingency)

print(f"Chi-square: {chi2:.3f}")
print(f"P-value: {p_value:.4f}")
```

### Interpreting P-Values
- **p < 0.01**: Highly significant (99% confidence)
- **p < 0.05**: Significant (95% confidence) ⭐ Most common threshold
- **p < 0.10**: Marginally significant (90% confidence)
- **p ≥ 0.10**: Not significant

---

## Visualization

### Matplotlib Basics
```python
import matplotlib.pyplot as plt

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Series 1')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
plt.scatter(x, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Histogram
plt.hist(data, bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Save figure
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

### Seaborn Plots
```python
import seaborn as sns

# Distribution
sns.histplot(df['column'], kde=True)

# Box plot
sns.boxplot(data=df, x='category', y='value')

# Violin plot
sns.violinplot(data=df, x='category', y='value')

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)

# Pair plot
sns.pairplot(df, hue='category')

plt.show()
```

### Quick Pandas Plots
```python
# Histogram
df['column'].hist(bins=30)

# Box plot
df.boxplot(column='value', by='category')

# Scatter
df.plot.scatter(x='col1', y='col2')

# Line plot
df.plot(x='date', y='value')

plt.show()
```

---

## Common Workflows

### Clustering Analysis Workflow
```python
# 1. Load and prepare data
df = pd.read_csv('data.csv')
X = df[['feature1', 'feature2', 'feature3']]

# 2. Encode categorical variables
X_encoded = pd.get_dummies(X)

# 3. Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 4. Find optimal K
from sklearn.metrics import silhouette_score
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}: {score:.3f}")

# 5. Fit final model
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 6. Analyze clusters
df.groupby('cluster').mean()
df.groupby('cluster').size()
```

### Statistical Testing Workflow
```python
# 1. Prepare data
test_group = df[df['group'] == 'test']['outcome']
control_group = df[df['group'] == 'control']['outcome']

# 2. Check assumptions
print(f"Test: n={len(test_group)}, mean={test_group.mean():.2f}")
print(f"Control: n={len(control_group)}, mean={control_group.mean():.2f}")

# 3. Perform test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(test_group, control_group)

# 4. Interpret
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("✓ Significant difference detected")
else:
    print("✗ No significant difference")

# 5. Calculate effect size
diff = test_group.mean() - control_group.mean()
pooled_std = np.sqrt((test_group.std()**2 + control_group.std()**2) / 2)
cohens_d = diff / pooled_std
print(f"Effect size (Cohen's d): {cohens_d:.3f}")
```

---

## Debugging Tips

### Print Debugging
```python
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Types: {df.dtypes}")
print(f"Value: {variable}")
print(f"Type: {type(variable)}")
```

### Data Quality Checks
```python
# Missing values
print(df.isnull().sum())

# Duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# Unique values
print(df['column'].nunique())

# Value range
print(f"Min: {df['column'].min()}, Max: {df['column'].max()}")

# Data types
print(df.dtypes)
```

### Common Issues

**Memory Issues:**
```python
# Check memory usage
df.memory_usage(deep=True)

# Reduce memory
df['col'] = df['col'].astype('category')  # For categoricals
df['col'] = df['col'].astype('int32')     # For integers
```

**Performance:**
```python
# Use vectorization (not loops)
df['new'] = df['col1'] * df['col2']  # Fast

# Not this:
for i in range(len(df)):  # Slow
    df.loc[i, 'new'] = df.loc[i, 'col1'] * df.loc[i, 'col2']
```

---

## Quick Tips

### Setting Random Seeds
```python
import numpy as np
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# Use random_state=SEED in sklearn functions
```

### Working with Dates
```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Date arithmetic
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
```

### String Operations
```python
df['name'].str.lower()           # Lowercase
df['name'].str.upper()           # Uppercase
df['name'].str.strip()           # Remove whitespace
df['name'].str.replace('old', 'new')  # Replace
df['name'].str.contains('pattern')    # Boolean check
df['name'].str.split(',')        # Split into list
```

### Conditional Operations
```python
# np.where (like SQL CASE)
import numpy as np
df['category'] = np.where(df['value'] > 100, 'High', 'Low')

# Multiple conditions
df['category'] = np.select(
    [df['value'] > 100, df['value'] > 50],
    ['High', 'Medium'],
    default='Low'
)

# Apply function
df['category'] = df['value'].apply(lambda x: 'High' if x > 100 else 'Low')
```

---

## Keyboard Shortcuts (Jupyter)

- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `Esc + A`: Insert cell above
- `Esc + B`: Insert cell below
- `Esc + D + D`: Delete cell
- `Esc + M`: Convert to markdown
- `Esc + Y`: Convert to code
- `Tab`: Autocomplete
- `Shift + Tab`: Function documentation

---

## Resources

**Documentation:**
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [NumPy](https://numpy.org/doc/)
- [Matplotlib](https://matplotlib.org/stable/)

**Quick Help:**
- Stack Overflow: [stackoverflow.com](https://stackoverflow.com/)
- Pandas Q&A: Search "pandas [your question]"

**Practice:**
- Kaggle Learn: [kaggle.com/learn](https://www.kaggle.com/learn)
- DataCamp: [datacamp.com](https://www.datacamp.com/)

---

**Pro Tip:** Bookmark this page and use `Ctrl+F` to quickly find operations you need!

---

Last updated: October 2025
