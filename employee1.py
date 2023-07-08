#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Read the CSV files
df1 = pd.read_csv('/Users/sudharmendragv/Downloads/Risk Assessment ML Project/inputdata/dataset1.csv')
df2 = pd.read_csv('/Users/sudharmendragv/Downloads/Risk Assessment ML Project/inputdata/dataset2.csv')

# Combine the two DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_df.to_csv('combined.csv', index=False)


# In[2]:


combined_df.head()


# In[3]:


combined_df.tail()


# In[4]:


combined_df.describe()


# In[5]:


combined_df.isnull().sum() # Checking the null values


# In[6]:


combined_df.info


# In[7]:


for i in (combined_df.columns):
    print("{}:{}".format(i, combined_df[i].value_counts().shape[0])) # Printing the unique values for each columns


# In[8]:


combined_df.columns


# In[9]:


combined_df.info()


# In[10]:


combined_df.duplicated().sum()
combined_df = combined_df.drop_duplicates()

x = combined_df.hist(figsize = (10,10)) # Histogram of all the parameters


# In[11]:


# Importing all the required libraries


import numpy as np
num_duplicates = np.sum(combined_df.duplicated())
print(f'Number of duplicates: {num_duplicates}')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve

import scipy
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
# Correlation chart colorbar
plt.matshow(combined_df.corr())
plt.colorbar()
plt.show()


# In[12]:


sns.distplot(a=combined_df["lastmonth_activity"]);


# In[13]:


sns.distplot(a=combined_df["lastyear_activity"]);


# In[14]:


# Heatmap values for all the parameters
plt.figure(figsize=(10,10))
sns.heatmap(combined_df.corr(), annot=True);


# In[15]:


sns.countplot(x="exited", data=combined_df);


# In[16]:


x = combined_df.filter(items=['lastmonth_activity', 'lastyear_activity', 'exited']).groupby(['exited']).sum()
x.plot.bar(stacked=True)
plt.show()


# In[17]:


x = combined_df['lastmonth_activity'].groupby(combined_df['exited'])
x.sum().plot(kind='bar')


# In[18]:


x = combined_df['number_of_employees'].groupby(combined_df['exited'])
x.sum().plot(kind='bar')


# In[19]:


sns.distplot(a=combined_df["number_of_employees"]);


# In[20]:


plt.figure(figsize=(15, 15))
sns.stripplot(data=combined_df, x='corporation', y='number_of_employees')
plt.title('Number of employees in specific corporation')
plt.xlabel('Corporation')
plt.ylabel('Number of employees')
plt.show()


# In[21]:


combined_df.drop(['corporation'], axis=1, inplace=True) 


# In[22]:


combined_df.columns


# In[ ]:





# In[144]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

import pandas as pd

# Read the CSV files
df1 = pd.read_csv('/Users/sudharmendragv/Downloads/Risk Assessment ML Project/inputdata/dataset1.csv')
df2 = pd.read_csv('/Users/sudharmendragv/Downloads/Risk Assessment ML Project/inputdata/dataset2.csv')

# Combine the two DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_df.to_csv('combined.csv', index=False)


# Load the combined dataset
#combined_data = pd.read_csv('merged_dataset.csv')

# Load the test data
test_data = pd.read_csv('/Users/sudharmendragv/Downloads/Risk Assessment ML Project/testdata/testdata.csv')

combined_df.drop(['corporation'], axis=1, inplace=True) 
test_data.drop(['corporation'], axis=1, inplace=True) 
# Separate the features and target variables
X = combined_df.drop('exited', axis=1)
y = combined_df['exited']
X_test = test_data.drop('exited', axis=1)
y_test = test_data['exited']

# Split the combined data into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the validation data
val_predictions = model.predict(X_val)

# Calculate accuracy and precision on the validation data
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions)

# Make predictions on the test data
test_predictions = model.predict(X_test)

# Calculate accuracy and precision on the test data
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions)

print("Validation Predictions:", val_predictions)
print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)

print("Test Predictions:", test_predictions)
print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)


# In[145]:


# Using sklearn-metrics the values of f1 socre, Recall, Precision and Acuracy are prined below. 
# The accuracy obtained is 90%
from sklearn import metrics

print("f1-score: ",metrics.f1_score(y_test,test_predictions, average='weighted'))
print("Recall: ",metrics.recall_score(y_test,test_predictions, average='weighted'))
print("Precision: ",metrics.precision_score(y_test,test_predictions, average='weighted'))
print("Accuracy: ",metrics.accuracy_score(y_test,test_predictions))
print("Confusion-Matrix :\n ", metrics.confusion_matrix(y_test,test_predictions))


import matplotlib.pyplot as plt
import seaborn as sns

# Compute the confusion matrix
cm = metrics.confusion_matrix(y_test, test_predictions)

# Plot the confusion matrix using seaborn heatmap
sns.heatmap(cm, annot=True, cmap='Oranges', fmt='g', cbar=False, 
            xticklabels=['Class 1', 'Class 2', 'Class 3'], 
            yticklabels=['Class 1', 'Class 2', 'Class 3'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('\nConfusion Matrix')
plt.show()


# In[148]:


# training a linear support vector machine (SVM) model on this dataset using scikit-learn's 
# LinearSVC class with a specific tolerance value and regularization parameter.
from sklearn.svm import LinearSVC

support_vector = LinearSVC(tol=1e-2, C=1)
support_vector.fit(X_train, y_train)


# In[149]:


# Printing the Train and Test score values.
print(f"Train score: {support_vector.score(X_train, y_train)}")
print(f"Test score: {support_vector.score(X_test, y_test)}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions
y_train_pred = support_vector.predict(X_train)
y_test_pred = support_vector.predict(X_test)

# Compute confusion matrices
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrices as heatmaps for both Train and Test
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
sns.heatmap(test_cm, annot=True, fmt='d', cmap='viridis_r', ax=ax[1])
ax[0].set_title('Train Confusion Matrix')
ax[1].set_title('Test Confusion Matrix')
ax[0].set_xlabel('Predicted Label')
ax[1].set_xlabel('Predicted Label')
ax[0].set_ylabel('True Label')
ax[1].set_ylabel('True Label')
plt.tight_layout()
plt.show()


# In[155]:


# training a decision tree classifier model on this dataset using scikit-learn's DecisionTreeClassifier class 
# with a specific maximum depth and maximum leaf nodes.
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=7)
tree.fit(X_train, y_train)


# In[157]:


# Train and Test accuracy
print(f"Train score: {tree.score(X_train, y_train)}")
print(f"Test score: {tree.score(X_test, y_test)}")


# In[209]:


# training a Random Forest classifier model on this dataset using scikit-learn's RandomForestClassifier class 
# with a specific maximum depth and maximum leaf nodes.
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=2, max_leaf_nodes=2)
forest.fit(X_train, y_train)


# In[211]:


# Printing the Train, Test accuracy scores
print(f"Train score: {forest.score(X_train, y_train)}")
print(f"Test score: {forest.score(X_test, y_test)}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




