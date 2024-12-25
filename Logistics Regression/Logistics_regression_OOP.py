import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from Seaborn
titanic_df = sns.load_dataset('titanic')

# Drop the 'embarked', 'class', and 'alive' columns
titanic_df.drop(["embarked", "class", "alive"], axis = 1, inplace = True)
titanic_df.info()

# Filling missing 'age' values based on passenger class (pclass)
titanic_df.loc[(titanic_df['age'].isnull()) & (titanic_df['pclass'] == 1), 'age'] = 38
titanic_df.loc[(titanic_df['age'].isnull()) & (titanic_df['pclass'] == 2), 'age'] = 29
titanic_df.loc[(titanic_df['age'].isnull()) & (titanic_df['pclass'] == 3), 'age'] = 23

# Dropping rows where 'embark_town' has missing values
titanic_df.dropna(subset=['embark_town'], inplace=True)

# Dropping the 'deck' column due to too many missing values
titanic_df.drop(columns=['deck'], inplace=True)
titanic_df.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
class TitanicLogisticModel:
    def __init__(self, dataset, drop_columns=None):
        """
        Initialize the class with the dataset and columns to drop (if any).
        """
        self.dataset = dataset.copy()
        if drop_columns:
            self.dataset = self.dataset.drop(columns=drop_columns)

    def encode_features(self):
        """
        Encode categorical features in the dataset.
        """
        # Encode categorical variables using one-hot encoding (with drop_first=True to avoid multicollinearity)
        sex = pd.get_dummies(self.dataset["sex"], drop_first=True)
        who = pd.get_dummies(self.dataset["who"], drop_first=True)
        adult_male = pd.get_dummies(self.dataset["adult_male"], drop_first=True)
        embark_town = pd.get_dummies(self.dataset["embark_town"], drop_first=True)
        alone = pd.get_dummies(self.dataset["alone"], drop_first=True)

        # Concatenate the encoded columns back into the dataset
        self.dataset = pd.concat([self.dataset, sex, who, adult_male, embark_town, alone], axis=1)

        # Drop the original categorical columns
        self.dataset.drop(columns=["sex", "who", "adult_male", "embark_town", "alone"], inplace=True)

        # Convert boolean columns to integers
        bool_columns = ['male', 'man', 'woman', 'Queenstown', 'Southampton']
        self.dataset[bool_columns] = self.dataset[bool_columns].astype(int)

    def split_data(self):
        """
        Split the dataset into training and testing sets.
        """
        X = self.dataset.drop('survived', axis=1)
        y = self.dataset['survived']

        # Convert feature names to strings
        X = X.rename(str, axis="columns")

        # Split the dataset
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def train_and_evaluate(self):
        """
        Train the logistic regression model and evaluate it.
        """
        # Encode features
        self.encode_features()

        # Split the data
        X_train, X_test, y_train, y_test = self.split_data()

        # Train the model
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)

        # Make predictions
        y_pred = logreg.predict(X_test)

        # Print evaluation metrics
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


# Example usage of the refactored class

# Dataset 1: With all features
print("Evaluating Model with All Features:")
model_all = TitanicLogisticModel(titanic_df)
model_all.train_and_evaluate()

# Dataset 2: Without 'age', 'sibsp', 'parch'
print("\nEvaluating Model without 'age', 'sibsp', 'parch':")
model_reduced = TitanicLogisticModel(titanic_df, drop_columns=['age', 'sibsp', 'parch'])
model_reduced.train_and_evaluate()