import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
calories = pd.read_csv('calories.csv')
dataset= pd.read_csv('exercise.csv')


#combine the datasets
new_dataset= pd.concat ([calories['Calories'],dataset] , axis=1)
new_dataset.drop(['User_ID'], axis=1, inplace=True)

print(new_dataset.head())
# Check for missing values

#print(new_dataset.isnull().sum())

#print(new_dataset.describe())

# Visualize the data
sns.countplot(x='Gender', data=new_dataset)
plt.title('Distribution of Gender')
plt.show()

sns.displot(new_dataset['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.displot(new_dataset['Height'], kde=True)
plt.title('Height Distribution')
plt.show()

sns.displot(new_dataset['Weight'], kde=True)
plt.title('Weight Distribution')
plt.show()

#encode categorical variables
new_dataset['Gender']=LabelEncoder().fit_transform(new_dataset['Gender'])


correlation_matrix = new_dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Split the dataset into features and target variable
X= new_dataset.drop('Calories', axis=1)
y = new_dataset['Calories']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the XGBoost classifier
model = XGBRegressor()
# Fit the model on the training data
model.fit(X_train, y_train)
#Mkae predictions on training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# Evaluate the model
print("Training Predictions:", y_train_pred[:5])
print("Testing Predictions:", y_test_pred[:5])
# Calculate the accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training Accuracy: ({train_accuracy:.2f})*100")
print(f"Testing Accuracy: ({test_accuracy:.2f})*100")
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred.round())
print("Confusion Matrix:\n", conf_matrix)
# Print the classification report
class_report = classification_report(y_test, y_test_pred.round())
print("Classification Report:\n", class_report)