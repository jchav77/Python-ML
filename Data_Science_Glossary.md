# Data Science & Machine Learning Glossary

> Technical terms explained in plain English for analysts from SAS/SQL/Excel backgrounds

---

## A

**Algorithm**  
A set of step-by-step instructions to solve a problem. In ML, it's the mathematical process that learns patterns from data.  
*SAS equivalent: A PROC procedure*

**API (Application Programming Interface)**  
A way for programs to talk to each other. For example, you use pandas' API to work with DataFrames.  
*Think of it like: The menu of functions available in a library*

**Array**  
A collection of numbers in a grid. Can be 1D (list), 2D (table), or higher dimensions.  
*Excel equivalent: A range of cells*  
*NumPy arrays are the foundation of most data science in Python*

---

## B

**Bias (in ML)**  
When a model systematically makes errors in one direction. For example, always predicting values too high.  
*Not to be confused with statistical bias or fairness bias*

**Binary Variable**  
A variable with only two possible values (0/1, True/False, Yes/No).  
*SQL equivalent: Boolean type*  
*Examples: purchased (yes/no), active_flag (1/0)*

**Bootstrap**  
A resampling technique where you randomly sample with replacement from your data to estimate uncertainty.  
*Like repeatedly drawing samples from the same pool*

---

## C

**Categorical Variable**  
A variable representing categories or groups (not numbers).  
*Examples: status (active/inactive), color (red/blue/green), region (North/South/East/West)*  
*SAS: CLASS variable*

**Centroid**  
The center point of a cluster in K-means clustering.  
*Think of it as: The "average location" of all points in that cluster*

**Classification**  
Predicting which category something belongs to.  
*Examples: Will customer churn? (yes/no), What tier is this customer? (gold/silver/bronze)*  
*SAS equivalent: Predicting CLASS variables*

**Clustering**  
Grouping similar observations together without predefined labels.  
*Business use: Customer segmentation, product categorization*  
*Unlike classification, we don't know the groups beforehand*

**Coefficient**  
A number that represents the strength and direction of a relationship in a model.  
*Example: In "Sales = 2 × Advertising + 100", the coefficient is 2*  
*SAS output: Parameter estimates in PROC REG*

**Confusion Matrix**  
A table showing how well your classifier performed: true positives, false positives, true negatives, false negatives.  
*Shows what your model got right vs wrong*

**Correlation**  
A measure of how two variables move together (-1 to 1).  
*1 = perfect positive relationship, -1 = perfect negative, 0 = no relationship*  
*SAS: PROC CORR*

---

## D

**DataFrame**  
Pandas' version of a table/dataset. Has rows and columns like Excel or a SQL table.  
*SAS equivalent: DATA step dataset*  
*Most data work in Python uses DataFrames*

**Data Leakage**  
When information from the test set accidentally gets into the training set, making performance look better than it really is.  
*A major pitfall in ML that makes models fail in production*

**Data Type (dtype)**  
The kind of data in a column: int (integer), float (decimal), object (text), bool (True/False), datetime.  
*SAS equivalent: Variable type (numeric/character)*  
*Important because operations depend on type*

**Dimension Reduction**  
Techniques to reduce the number of features while keeping important information.  
*Examples: PCA, selecting most important features*  
*Useful when you have too many variables*

---

## E

**Elbow Method**  
A visual technique to find optimal number of clusters. Plot inertia vs K and look for the "elbow" (bend in the line).  
*Where the line starts to flatten out is often a good K*

**Encoding**  
Converting categories to numbers so ML algorithms can use them.  
*Example: ["red", "blue", "green"] → [0, 1, 2] or one-hot encoded*

**Epoch**  
One complete pass through the entire training dataset.  
*Mainly used in neural networks*

---

## F

**Feature**  
An input variable used to make predictions. Also called: predictor, independent variable, X variable.  
*In a table, features are usually columns*  
*Examples: age, income, purchase_history*

**Feature Engineering**  
Creating new features from existing ones to improve model performance.  
*Example: Creating "age_group" from "age", or "total_spend" from purchase history*

**Feature Scaling**  
Adjusting features to have similar ranges. Essential for algorithms like K-means and neural networks.  
*Methods: Standardization (z-scores) or Min-Max normalization (0-1 range)*

**Fit**  
Training a model on data. The model "fits" itself to the patterns in your training data.  
*"fit" is the method you call to train: model.fit(X_train, y_train)*

---

## G

**Generalization**  
A model's ability to perform well on new, unseen data.  
*The goal of ML: models should work on future data, not just training data*

**Gradient Descent**  
An optimization algorithm that finds the best parameters by taking steps in the direction that reduces error.  
*Like walking downhill to find the lowest point*

---

## H

**Hyperparameter**  
Settings you choose before training (not learned from data).  
*Examples: number of clusters in K-means, learning rate in neural networks*  
*Different from model parameters which are learned*

**Hypothesis**  
A claim you're testing statistically.  
*Null hypothesis (H₀): No effect exists*  
*Alternative hypothesis (H₁): An effect exists*

**Hypothesis Testing**  
Using statistics to determine if a result is likely real or just random chance.  
*Example: Is the difference between test and control significant?*

---

## I

**Imbalanced Data**  
When categories in your target variable are very unequal.  
*Example: 95% non-fraud, 5% fraud*  
*Can cause models to just predict the majority class*

**Inertia**  
In K-means, the sum of squared distances from points to their cluster centers.  
*Lower inertia = tighter clusters*  
*Used in elbow method to choose K*

**Iteration**  
One round of an algorithm's process. K-means runs multiple iterations to converge.  
*Each iteration: reassign points to clusters, then move centroids*

---

## K

**K (in K-means)**  
The number of clusters you want to find.  
*You choose K, then the algorithm finds the best K groups*

**K-Fold Cross-Validation**  
Splitting data into K pieces, training K times (each time using a different piece as test set).  
*Gives more reliable performance estimates than single train-test split*

---

## L

**Label**  
The correct answer or target value for supervised learning.  
*Example: In predicting churn, the label is "did churn" (yes/no)*  
*SAS: Dependent variable*

**Label Encoding**  
Converting categories to sequential integers: ["low", "medium", "high"] → [0, 1, 2]  
*Use when categories have natural order (ordinal)*

**Learning Rate**  
How big the steps are when training (mainly neural networks).  
*Too high: might miss optimal solution; Too low: training is very slow*

**Leakage** → See Data Leakage

**Library**  
A collection of pre-written code you can import and use.  
*Examples: pandas, scikit-learn, matplotlib*  
*Like SAS procedures, but you can mix and match*

**Linear Regression**  
Predicting a continuous number using a straight line relationship.  
*Example: Predict sales based on advertising spend*  
*SAS: PROC REG*

**Logistic Regression**  
Despite the name, it's for classification (predicting categories).  
*Example: Predict will customer buy (yes/no) based on features*  
*SAS: PROC LOGISTIC*

---

## M

**Machine Learning (ML)**  
Algorithms that learn patterns from data to make predictions or find insights.  
*Two main types: Supervised (predict labels) and Unsupervised (find patterns)*

**Matrix**  
A 2D array of numbers (rows and columns).  
*Your data is usually stored as a matrix*

**Mean Absolute Error (MAE)**  
Average of the absolute differences between predictions and actuals.  
*Tells you on average how far off your predictions are*

**Mean Squared Error (MSE)**  
Average of squared differences between predictions and actuals.  
*Penalizes large errors more than MAE*

**Metric**  
A number that measures model performance.  
*Examples: accuracy, MSE, R², silhouette score*

**Model**  
The mathematical representation learned from training data.  
*It encodes the patterns found in your training data*

---

## N

**NaN (Not a Number)**  
Python's representation of missing/null values in numeric data.  
*SAS equivalent: . (dot)*  
*SQL equivalent: NULL*

**Normalization**  
Scaling data to a standard range (usually 0 to 1).  
*Formula: (x - min) / (max - min)*

**Null Hypothesis**  
The assumption that there's no effect or difference.  
*We try to disprove this with statistics*  
*Example: "Test and control groups have the same conversion rate"*

**NumPy**  
Python library for numerical computing (arrays, math operations).  
*Foundation for most data science libraries*  
*Faster than pure Python for math operations*

---

## O

**Observations**  
Individual rows in your dataset. Also called: samples, records, instances.  
*Each observation represents one customer, transaction, etc.*

**Outlier**  
A data point that's very different from others.  
*Can be errors or genuine extreme values*  
*May need special handling depending on context*

**Overfitting**  
When a model learns the training data too well, including noise, and performs poorly on new data.  
*Like memorizing test questions instead of learning the material*

---

## P

**Pandas**  
Python library for data manipulation (DataFrames).  
*Equivalent to SAS DATA step + PROC SQL combined*  
*Most important library for data analysis in Python*

**Parameter**  
Values that the model learns during training.  
*Example: Coefficients in regression*  
*Different from hyperparameters (which you set)*

**Pipeline**  
A sequence of data processing steps.  
*Example: Load → Clean → Encode → Scale → Train → Predict*

**Predict**  
Using a trained model to make predictions on new data.  
*model.predict(new_data)*

**Predictor** → See Feature

**P-value**  
Probability that the observed result occurred by chance.  
*p < 0.05 typically means "statistically significant"*  
*Lower p-value = stronger evidence of real effect*

**Python**  
Programming language widely used for data science and ML.  
*Why Python? Huge ecosystem, easy to learn, powerful libraries*

---

## R

**R² (R-squared)**  
Proportion of variance explained by the model (0 to 1).  
*1 = perfect fit, 0 = model is no better than predicting the mean*  
*SAS output: In PROC REG results*

**Random Forest**  
ML algorithm that combines many decision trees.  
*Often performs well "out of the box"*

**Random State / Random Seed**  
A number that makes randomness reproducible.  
*Set it to get same results every time you run code*  
*Example: random_state=42*

**Regression**  
Predicting a continuous number (not categories).  
*Examples: Predict price, sales amount, age*  
*SAS: PROC REG, PROC GLM*

**Regularization**  
Techniques to prevent overfitting by penalizing complex models.  
*Makes models simpler and more general*

---

## S

**Scikit-learn (sklearn)**  
Python's main ML library. Provides algorithms for classification, regression, clustering, etc.  
*Consistent API across all algorithms*  
*Industry standard for traditional ML*

**Series**  
Pandas' version of a single column (1D array with labels).  
*Like one column from Excel*  
*df['column'] returns a Series*

**Silhouette Score**  
Measures how well-separated clusters are (-1 to 1).  
*Higher is better: means points are close to their cluster, far from others*  
*Used to evaluate clustering quality and choose K*

**Standardization**  
Scaling data to have mean=0 and standard deviation=1.  
*Formula: (x - mean) / std*  
*Also called: Z-score normalization*

**Statistically Significant**  
When a result is unlikely to have occurred by chance (typically p < 0.05).  
*Doesn't necessarily mean practically important!*

**Supervised Learning**  
ML where you have labeled training data (inputs + correct outputs).  
*Examples: Classification, Regression*  
*You're supervising the model by providing answers*

**Support Vector Machine (SVM)**  
ML algorithm for classification and regression.  
*Finds the best boundary between classes*

---

## T

**Target Variable**  
What you're trying to predict. Also called: dependent variable, label, Y variable.  
*Example: In predicting churn, the target is "churned" (yes/no)*

**Test Set**  
Data held back to evaluate model performance.  
*Model has never seen this data during training*  
*Typically 20-30% of total data*

**Training Set**  
Data used to train the model.  
*Model learns patterns from this data*  
*Typically 70-80% of total data*

**Transform**  
Applying a fitted preprocessing step to data.  
*Example: scaler.transform(new_data) applies the scaling learned during fit*

**T-test**  
Statistical test comparing means of two groups.  
*Example: Is average purchase amount different between segments?*  
*SAS: PROC TTEST*

**Type I Error**  
False positive: Concluding there's an effect when there isn't.  
*Example: Saying test beat control when it didn't*

**Type II Error**  
False negative: Missing a real effect.  
*Example: Saying no difference when test really does beat control*

---

## U

**Underfitting**  
When a model is too simple to capture patterns in the data.  
*Performs poorly on both training and test data*

**Unsupervised Learning**  
ML where you don't have labels, just features.  
*Examples: Clustering, Dimensionality Reduction*  
*Goal is to find structure/patterns in data*

---

## V

**Validation Set**  
A third dataset (besides train/test) used to tune hyperparameters.  
*Prevents using test set for model development*

**Variance (in ML context)**  
How much predictions change when trained on different data.  
*High variance = overfitting*  
*Low variance = model is too simple*

**Vector**  
A 1D array of numbers.  
*Example: [1, 2, 3, 4, 5]*

**Vectorization**  
Performing operations on entire arrays at once (not looping).  
*Much faster in Python than loops*  
*Example: df['new'] = df['a'] * df['b'] instead of looping through rows*

---

## X

**X**  
Convention for feature matrix (input variables).  
*Usually a 2D array/DataFrame: rows=observations, columns=features*

---

## Y

**Y**  
Convention for target variable (what you're predicting).  
*Usually a 1D array/Series*

---

## Z

**Z-score**  
Number of standard deviations away from the mean.  
*Formula: (x - mean) / std*  
*Used in standardization and detecting outliers*

**Z-test**  
Statistical test comparing proportions or means (when sample size is large).  
*Example: Compare conversion rates between test and control*

---

## Common Acronyms

**API**: Application Programming Interface  
**ANOVA**: Analysis of Variance  
**CI**: Confidence Interval  
**CSV**: Comma-Separated Values  
**EDA**: Exploratory Data Analysis  
**IDE**: Integrated Development Environment (like VS Code)  
**IQR**: Interquartile Range (Q3 - Q1)  
**JSON**: JavaScript Object Notation (data format)  
**MAE**: Mean Absolute Error  
**ML**: Machine Learning  
**MSE**: Mean Squared Error  
**NaN**: Not a Number  
**OLS**: Ordinary Least Squares  
**PCA**: Principal Component Analysis  
**RMSE**: Root Mean Squared Error  
**ROC**: Receiver Operating Characteristic  
**SVM**: Support Vector Machine  

---

## SAS/SQL/Excel → Python Translation

| SAS | SQL | Excel | Python (Pandas) |
|-----|-----|-------|-----------------|
| DATA step | CREATE TABLE | Sheet | DataFrame |
| Variable | Column | Column | Series/Column |
| Observation | Row | Row | Row |
| PROC MEANS | AVG(), SUM() | =AVERAGE() | .mean(), .sum() |
| PROC FREQ | GROUP BY | Pivot Table | .value_counts() |
| PROC SQL | SELECT | Filter | df[condition] |
| MERGE | JOIN | VLOOKUP | merge() |
| WHERE | WHERE | Filter | df[condition] |
| . (missing) | NULL | (blank) | NaN, None |
| PUT | CONCAT() | & | + or f-string |
| SUBSTR | SUBSTRING() | MID() | .str[] |
| CLASS variable | Categorical | Categories | object dtype |

---

## Quick Comparison: Traditional Analysis vs. ML

| Aspect | Traditional Analysis | Machine Learning |
|--------|---------------------|------------------|
| **Goal** | Explain relationships | Predict outcomes |
| **Approach** | Hypothesis-driven | Pattern-driven |
| **Tools** | SAS, SQL, Excel | Python, R, ML frameworks |
| **Focus** | Statistical significance | Prediction accuracy |
| **Model** | Simple, interpretable | Can be complex "black box" |
| **Output** | P-values, coefficients | Predictions, accuracy scores |
| **Example** | "Does X affect Y?" | "What will Y be for new data?" |

**You need both!** Traditional analysis explains what happened and why. ML predicts what will happen next.

---

## Tips for Learning ML Terms

1. **Don't memorize everything** - Learn as you go, when you need it
2. **Relate to what you know** - Connect to SAS/SQL/Excel concepts
3. **Use the glossary** - Bookmark this and refer back often
4. **Practice with code** - Seeing terms in action helps them stick
5. **Don't stress about math** - Libraries handle the math; focus on concepts

---

## Still Confused?

That's normal! Data science has lots of jargon. When you encounter a new term:

1. Search it with "in plain English" or "for beginners"
2. Look for "X explained using [familiar tool]" 
3. Check this glossary
4. Ask in communities (Reddit, Stack Overflow)
5. Remember: Everyone started as a beginner

The terminology will become natural with practice!

---

**Pro tip:** When reading documentation or tutorials, keep this glossary open in another tab for quick reference.

---

Last updated: October 2025
