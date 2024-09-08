import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
import time

import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#First we take a cleaner look at the data and start with EDA
df_original = pd.read_csv('/Users/arnavchopra/Desktop/find_smokers_project/smoking.csv' , index_col = False)
print("The Original Dataframe is: \n\n", df_original.head())

#Seeing the general structure and quantity of the data
print("\n\n\nShape of the above dataset is:  ", end="")
print(df_original.shape)

print("\n\n\nDescription of the dataset : \n")
print(df_original.describe() )

#On observing the dataset carefully I decide to carry out my EDA
#on the training data before hand for convinience purposes

x_train = pd.read_csv('/Users/arnavchopra/Desktop/find_smokers_project/competition_format/x_train.csv', index_col = False)
y_train = pd.read_csv('/Users/arnavchopra/Desktop/find_smokers_project/competition_format/y_train.csv', index_col = False)
print("The dataset to be worked on:\n\n\n ", x_train.head())

#Seeing the general structure and quantity of the data
print("\n\n\nShape of the above dataset is:  ", end="")
print(x_train.shape)

print("\n\n\nDescription of the dataset : \n")
print(x_train.describe() )

#I will first set the ID column to be the index of the dataframe.
x_train = x_train.set_index('ID')
print("\n\n\nAfter id feature is set as row index: \n", x_train.head())


#Now, as we move to the next column we see that the gender,oral and tartar columns are alphabetical
#Thus I will enumerate all those columns
print("\n\nAfter Enumeration is complete: \n\n")
x_train.gender = x_train.gender.apply(lambda x: 1 if x == 'F' else 0)
x_train.oral = x_train.oral.apply(lambda x: 1 if x == 'Y' else 0)
x_train.tartar = x_train.tartar.apply(lambda x: 1 if x == 'Y' else 0)
print(x_train.head())

#Checking the smoking column for any different values
plt.hist(y_train.smoking)
plt.title('Smoking Data')
plt.show()

#Finding out exact number of smokers/non-smokers
print("\n\nExact Number of Smokers/Non-Smokers: ", y_train.groupby('smoking').size())
x_test = pd.read_csv('/Users/arnavchopra/Desktop/find_smokers_project/competition_format/x_test.csv')
y_test = pd.read_csv('/Users/arnavchopra/Desktop/find_smokers_project/competition_format/y_test.csv')

x_test = x_test.drop(columns=['ID'])
y_test = y_test.drop(columns=['ID'])

y_train = y_train.drop(columns=['ID'])

x_test.gender = x_test.gender.apply(lambda x: 1 if x == 'F' else 0)
x_test.oral = x_test.oral.apply(lambda x: 1 if x == 'Y' else 0)
x_test.tartar = x_test.tartar.apply(lambda x: 1 if x == 'Y' else 0)

#TRAINING DATA BEFORE SCALING
print("Training of the UN-SCALED data started: \n\n")
print("Still Training on Descision Tree Classifier..................\n")

model = DecisionTreeClassifier()

# Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

print("\n\nTraining Completed!\n")

print("Training of the UN-SCALED data started: \n\n")
print("Still Training on SVM..................\n")
model = SVC()

# Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

print("\n\nTraining Completed!\n")

print("Training of the UN-SCALED data started: \n\n")
print("Still Training on GaussianNB..................\n")

model = GaussianNB()

# Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

print("\n\nTraining Completed!\n")

print("Training of the UN-SCALED data started: \n\n")
print("Still Training on KNN..................\n")


model = KNeighborsClassifier()

#Train the model
model.fit(x_train, y_train)

#Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

print("\n\nTraining Completed!\n")

#Standardising the data:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x_train)
print("\n\nThe data has been succesfully scaled....\n\n")

#We'll now apply the machine learning algorithms(Decision Tree, SVM, Naive Bayes, KNN)
# to this dataset and evaluate their performance using cross-validation)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
import time

#TRAINING DATA AFTER SCALING
print("Training of the scaled data started: \n\n")
print("Still Training on Descision Tree Classifier..................\n")

model = DecisionTreeClassifier()

# Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

accuracy_df = []
accuracy_df.append(accuracy)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

print("\n\nTraining Completed!\n")

print("Training of the scaled data started: \n\n")
print("Still Training on SVM..................\n")
model = SVC()

# Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
accuracy_df.append(accuracy)
print("\n\nTraining Completed!\n")

print("Training of the scaled data started: \n\n")
print("Still Training on GaussianNB..................\n")

model = GaussianNB()

# Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy_df.append(accuracy)
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

print("\n\nTraining Completed!\n")

print("Training of the scaled data started: \n\n")
print("Still Training on KNN..................\n")


model = KNeighborsClassifier()

#Train the model
model.fit(x_train, y_train)

#Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
accuracy_df.append(accuracy)
print("\n\nTraining Completed!\n")

#Visualizing the accuracies


accuracies_before_scaling = accuracy_df


accuracies_after_scaling = accuracy_df


models = ['Decision Tree', 'SVM', 'GaussianNB', 'KNN']


x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracies_before_scaling, width, label='Before Scaling')
rects2 = ax.bar(x + width/2, accuracies_after_scaling, width, label='After Scaling')


ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Before and After Scaling')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

fig.tight_layout()

plt.show()

print(x_train)

