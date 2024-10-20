import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

# Load the "mpg" dataset
df = sns.load_dataset("mpg")

# Display the first few rows of the dataset
df.head()

# Summary of the dataset
df.info()

# Display summary statistics of the dataset
df.describe()

# Check for missing values
df.isnull().sum()

# Drop rows with missing values in the 'horsepower' column
df = df.dropna(subset=['horsepower'])

# Check to ensure there are no more missing values in 'horsepower'
df.isnull().sum()

# Drop the 'name' column
df.drop(['name'], axis=1, inplace=True)

# Verify that the column has been removed
df.head()

# EDA

## Plot the distribution of mpg (target variable)
plt.figure(figsize=(8, 6))
sns.histplot(df['mpg'], kde=True, bins=20, color='blue')
plt.title('Distribution of Miles Per Gallon (mpg)')
plt.xlabel('mpg')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of mpg vs horsepower
plt.figure(figsize=(8, 6))
sns.scatterplot(x='horsepower', y='mpg', data=df)
plt.title('MPG vs Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()

# Scatter plot of mpg vs weight
plt.figure(figsize=(8, 6))
sns.scatterplot(x='weight', y='mpg', data=df)
plt.title('MPG vs Weight')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# Scatter plot of mpg vs displacement
plt.figure(figsize=(8, 6))
sns.scatterplot(x='displacement', y='mpg', data=df)
plt.title('MPG vs Displacement')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.show()

# Create a correlation heatmap to visualize relationships between numerical features
# Select only the numeric columns for the correlation matrix
numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap to visualize the correlations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Bivariate Analysis

# Countplot of model_year
plt.figure(figsize=(15, 8))
sns.countplot(x="model_year", data=df, palette="rainbow")
plt.title('Count of Cars by Model Year')
plt.show()

# Boxplot of model_year vs mpg
plt.figure(figsize=(15, 8))
sns.boxplot(x="model_year", y="mpg", data=df, palette="Spectral")
plt.title('MPG Distribution by Model Year')
plt.show()

# Multivariate Analysis 

# lmplot of horsepower vs mpg by origin
graph = sns.lmplot(x="horsepower", y="mpg", hue="origin", data=df, palette="rainbow")
graph.set(xlim=(0, 250), ylim=(0, 50))
plt.title('MPG vs Horsepower by Origin')
plt.show()

# lmplot of acceleration vs mpg by origin
graph = sns.lmplot(x="acceleration", y="mpg", hue="origin", data=df, palette="rainbow")
graph.set(ylim=(0, 50), xlim=(5, 28))
plt.title('MPG vs Acceleration by Origin')
plt.show()

# lmplot of weight vs mpg by origin
graph = sns.lmplot(x="weight", y="mpg", hue="origin", data=df, palette="rainbow")
graph.set(ylim=(0, 50), xlim=(1500, 5500))
plt.title('MPG vs Weight by Origin')
plt.show()

# lmplot of displacement vs mpg by origin
graph = sns.lmplot(x="displacement", y="mpg", hue="origin", data=df, palette="rainbow")
graph.set(ylim=(0, 50), xlim=(0, 475))
plt.title('MPG vs Displacement by Origin')
plt.show()

# Data preprocessing

# Drop the 'acceleration' and 'displacement' columns
df.drop(["acceleration", "displacement"], axis=1, inplace=True)
df.head()

from datetime import datetime

# Get the current year
today = datetime.today()
this_year = today.year

# Create a new 'age' column
df["age"] = this_year - df["model_year"]

# Drop the 'model_year' column as it's no longer needed
df.drop(["model_year"], axis=1, inplace=True)
df.head()

# Apply one-hot encoding to the 'origin' column
df = pd.get_dummies(df, drop_first=True)
df.head()

# Fitting and Evaluating the Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Traditional train-test split

# Split the dataset into training and testing sets
X = df.drop('mpg', axis=1)
y = df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model's performance on the test set
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print(f"Mean Squared Error (Test Set): {mse_test}")
print(f"R-squared (Test Set): {r2_test}")

# Cross-Validation 

from sklearn.model_selection import cross_val_score

# Perform 10-fold cross-validation
cv_scores = cross_val_score(lr_model, X, y, cv=10, scoring='neg_mean_squared_error')

# Compute the mean and standard deviation of the cross-validated MSE scores
mse_cv = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"Mean Squared Error (10-Fold Cross-Validation): {mse_cv}")
print(f"Standard Deviation of Cross-Validation Scores: {cv_std}")

# Visualizing the Model
from yellowbrick.regressor import PredictionError

# Create the PredictionError visualizer

# With the train-test split method
visualizer_train_test = PredictionError(lr_model)

# Fit the model and visualize the prediction error
visualizer_train_test = PredictionError(lr_model, title="Prediction Error LinearRegression (Train test split method)")
visualizer_train_test.fit(X_train, y_train)
visualizer_train_test.score(X_test, y_test)

# Show the plot
visualizer_train_test.show()

# With the cross-validation method
from sklearn.model_selection import cross_val_predict

# Get cross-validation predictions
y_pred_cv = cross_val_predict(lr_model, X, y, cv=10)

# Create the PredictionError visualizer for cross-validation
visualizer_cv = PredictionError(lr_model, title="Prediction Error LinearRegression (10-fold cross-validation method)")
# Fit the visualizer to the full dataset
visualizer_cv.fit(X, y)

# Use the cross-validation predictions to score and plot
visualizer_cv.score(X, y_pred_cv)

# Show the plot
visualizer_cv.show()
