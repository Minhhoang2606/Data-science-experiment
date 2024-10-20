import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from Seaborn
titanic_df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
titanic_df.head()
titanic_df.info()

# Drop the 'embarked', 'class', and 'alive' columns
titanic_df.drop(["embarked", "class", "alive"], axis = 1, inplace = True)
titanic_df.info()
# Unique values in the 'who' column
titanic_df['who'].unique()
# 1. Dataset overview

# Initial analysis of the dataset

## Basic information about the dataset
titanic_df.info()
## Checking for missing values
titanic_df.isnull().sum()

# Visualizing survival rates
sns.countplot(x='survived', data=titanic_df, palette=['red', 'green'])
plt.title('Count of Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.show()

# Survival rates by gender
sns.countplot(x='sex', hue='survived', data=titanic_df)
plt.title('Survival by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.show()

# Survival rates by passenger class
sns.countplot(x='pclass', hue='survived', data=titanic_df)
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.show()

# Distribution of age
sns.histplot(titanic_df['age'].dropna(), kde=False, bins=30)
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# Survival rates by age
sns.histplot(titanic_df[titanic_df['survived'] == 1]['age'].dropna(), kde=False, bins=30, color='green', label='Survived')
sns.histplot(titanic_df[titanic_df['survived'] == 0]['age'].dropna(), kde=False, bins=30, color='red', label='Not Survived')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

# Creating a new feature for family size
titanic_df['family_size'] = titanic_df['sibsp'] + titanic_df['parch'] + 1

# Survival rates by family size
sns.countplot(x='family_size', hue='survived', data=titanic_df)
plt.title('Survival by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Number of Passengers')
plt.show()

# Survival rates by embarkation port
sns.countplot(x='embarked', hue='survived', data=titanic_df)
plt.title('Survival by Embarkation Port')
plt.xlabel('Port of Embarkation')
plt.ylabel('Number of Passengers')
plt.show()

# Visualizing the relationship between deck and survival
sns.countplot(x='deck', hue='survived', data=titanic_df)
plt.title('Survival by Deck')
plt.xlabel('Deck')
plt.ylabel('Number of Passengers')
plt.show()

# Age distribution with KDE 
plt.figure(figsize=(10, 6))
sns.histplot(titanic_df['age'], kde=True, color='brown', bins=30)
plt.title("Age Distribution of Titanic Passengers (Before Handling Missing Values)")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()

# Fare distribution with KDE 
plt.figure(figsize=(10, 6))
sns.histplot(titanic_df['fare'], kde=True, color='darkred', bins=30)
plt.title("Fare Distribution of Titanic Passengers (Before Handling Missing Values)")
plt.xlabel("Fare")
plt.ylabel("Density")
plt.show()

# Survival rates by embarkation town
sns.countplot(x='survived', hue='embark_town', data=titanic_df)
plt.title("Survival by Embarkation Town")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Embarkation Town")
plt.show()

# Boxplots of age by passenger class
plt.figure(figsize=(10, 6))
sns.boxplot(x="pclass", y="age", data=titanic_df, palette="Set3")
plt.show()

# Automatically select only numerical columns
numerical_columns = titanic_df.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numerical_columns.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='RdPu', center=0, linewidths=1, linecolor='white')
plt.title('Correlation Heatmap of Numerical Features in Titanic Dataset')
plt.show()

# Filling missing 'age' values based on passenger class (pclass)
titanic_df.loc[(titanic_df['age'].isnull()) & (titanic_df['pclass'] == 1), 'age'] = 38
titanic_df.loc[(titanic_df['age'].isnull()) & (titanic_df['pclass'] == 2), 'age'] = 29
titanic_df.loc[(titanic_df['age'].isnull()) & (titanic_df['pclass'] == 3), 'age'] = 23

# Dropping rows where 'embark_town' has missing values
titanic_df.dropna(subset=['embark_town'], inplace=True)

# Dropping the 'deck' column due to too many missing values
titanic_df.drop(columns=['deck'], inplace=True)
titanic_df.head()

# 2. Data preprocessing
import pandas as pd

# Create Dataset 1: All features
dataset_all_features = titanic_df.copy()

# Create Dataset 2: Excluding 'age', 'sibsp', and 'parch'
dataset_reduced_features = titanic_df.drop(columns=['age', 'sibsp', 'parch'])
dataset_reduced_features.head()

# Encoding features in 2 datasets
# For Dataset 1 (with all features):
sex_1 = pd.get_dummies(dataset_all_features["sex"], drop_first=True)
who_1 = pd.get_dummies(dataset_all_features["who"], drop_first=True)
adult_male_1 = pd.get_dummies(dataset_all_features["adult_male"], drop_first=True)
embark_town_1 = pd.get_dummies(dataset_all_features["embark_town"], drop_first=True)
alone_1 = pd.get_dummies(dataset_all_features["alone"], drop_first=True)

# Concatenate the encoded columns back into the dataset
dataset_all_features = pd.concat([dataset_all_features, sex_1, who_1, adult_male_1, embark_town_1, alone_1], axis=1)

# Drop the original categorical columns
dataset_all_features.drop(columns=["sex", "who", "adult_male", "embark_town", "alone"], inplace=True)
dataset_all_features.info()

bool_columns = ['male', 'man', 'woman', 'Queenstown', 'Southampton']

# Convert boolean columns to integers (0/1)
dataset_all_features[bool_columns] = dataset_all_features[bool_columns].astype(int)
dataset_all_features.info()

# For Dataset 2 (without 'age', 'sibsp', 'parch'):
sex_2 = pd.get_dummies(dataset_reduced_features["sex"], drop_first=True)
who_2 = pd.get_dummies(dataset_reduced_features["who"], drop_first=True)
adult_male_2 = pd.get_dummies(dataset_reduced_features["adult_male"], drop_first=True)
embark_town_2 = pd.get_dummies(dataset_reduced_features["embark_town"], drop_first=True)
alone_2 = pd.get_dummies(dataset_reduced_features["alone"], drop_first=True)

# Concatenate the encoded columns back into the dataset
dataset_reduced_features = pd.concat([dataset_reduced_features, sex_2, who_2, adult_male_2, embark_town_2, alone_2], axis=1)

# Drop the original categorical columns
dataset_reduced_features.drop(columns=["sex", "who", "adult_male", "embark_town", "alone"], inplace=True)
dataset_reduced_features.info()


# 3. Model building
from sklearn.model_selection import train_test_split

# For Dataset 1 (with all features):
X_all = dataset_all_features.drop('survived', axis=1)
y_all = dataset_all_features['survived']

# For Dataset 2 (without 'age', 'sibsp', 'parch'):
X_reduced = dataset_reduced_features.drop('survived', axis=1)
y_reduced = dataset_reduced_features['survived']

# Convert feature names (column names) to strings for both datasets
X_all = X_all.rename(str, axis="columns")
X_reduced = X_reduced.rename(str, axis="columns")

# Split Dataset 1 (all features)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.25, random_state=42)

# Split Dataset 2 (without 'age', 'sibsp', 'parch')
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y_reduced, test_size=0.25, random_state=42)

# 4. Model training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train the logistic regression model on Dataset 1 (with all features)
logreg_all = LogisticRegression()
logreg_all.fit(X_train_all, y_train_all)

# Train a separate logistic regression model on Dataset 2 (without 'age', 'sibsp', and 'parch')
logreg_reduced = LogisticRegression()
logreg_reduced.fit(X_train_reduced, y_train_reduced)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predictions and evaluation for Dataset 1 (with all features)
y_pred_all = logreg_all.predict(X_test_all)
print("Accuracy with all features:", accuracy_score(y_test_all, y_pred_all))
print("Confusion Matrix:\n", confusion_matrix(y_test_all, y_pred_all))
print("Classification Report:\n", classification_report(y_test_all, y_pred_all))

# Predictions and evaluation for Dataset 2 (without 'age', 'sibsp', 'parch')
y_pred_reduced = logreg_reduced.predict(X_test_reduced)
print("Accuracy without 'age', 'sibsp', 'parch':", accuracy_score(y_test_reduced, y_pred_reduced))
print("Confusion Matrix:\n", confusion_matrix(y_test_reduced, y_pred_reduced))
print("Classification Report:\n", classification_report(y_test_reduced, y_pred_reduced))
