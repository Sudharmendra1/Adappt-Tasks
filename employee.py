#!/usr/bin/env python
# coding: utf-8

# In[11]:

#Importing all the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve

# Reading the CSV files. Since there are two input files, combining both of them and named it as 'combined_df'
df1 = pd.read_csv('/Users/sudharmendragv/Downloads/Adappt/dataset1.csv')
df2 = pd.read_csv('/Users/sudharmendragv/Downloads/Adappt/dataset2.csv')

# Combine the two DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_df.to_csv('combined.csv', index=False)


# In[12]:


combined_df.head()


# In[13]:


combined_df.tail()


# In[14]:

#Describing the dataframe
combined_df.describe()


# In[15]:


combined_df.isnull().sum() # Checking the null values


# In[16]:


combined_df.info


# In[17]:


for i in (combined_df.columns):
    print("{}:{}".format(i, combined_df[i].value_counts().shape[0])) # Printing the unique values for each columns


# In[18]:


combined_df.columns


# In[19]:


combined_df.info()


# In[20]:

#Checking for duplicate values and removing them
combined_df.duplicated().sum()
combined_df = combined_df.drop_duplicates()
num_duplicates = np.sum(combined_df.duplicated())


# In[21]:


print(f'Number of duplicates: {num_duplicates}')


# In[22]:


x = combined_df.hist(figsize = (10,10)) # Histogram of all the parameters


# In[23]:


import scipy
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
# Correlation chart colorbar
plt.matshow(combined_df.corr())
plt.colorbar()
plt.show()


# In[24]:

#Univariate Distribution data for 'lastmonth_activity'
sns.distplot(a=combined_df["lastmonth_activity"]);


# In[25]:

#Univariate Distribution data for 'lastyear_activity'
sns.distplot(a=combined_df["lastyear_activity"]);


# In[26]:


# Heatmap values for all the parameters
plt.figure(figsize=(10,10))
sns.heatmap(combined_df.corr(), annot=True);


# In[27]:

#Countplot for exited
sns.countplot(x="exited", data=combined_df);


# In[28]:


x = combined_df.filter(items=['lastmonth_activity', 'lastyear_activity', 'exited']).groupby(['exited']).sum()
x.plot.bar(stacked=True)
plt.show()


# In[29]:


#Bar plot for lastmonth_activity for exited
x = combined_df['lastmonth_activity'].groupby(combined_df['exited'])
x.sum().plot(kind='bar')


# In[30]:


# Number of employees and exited
x = combined_df['number_of_employees'].groupby(combined_df['exited'])
x.sum().plot(kind='bar')


# In[31]:

#Finding number of employees
sns.distplot(a=combined_df["number_of_employees"]);


# In[32]:

# Number of employees in specific corporation
plt.figure(figsize=(15, 15))
sns.stripplot(data=combined_df, x='corporation', y='number_of_employees')
plt.title('Number of employees in specific corporation')
plt.xlabel('Corporation')
plt.ylabel('Number of employees')
plt.show()


# In[33]:

# Dropping Corporation after visualization inorder to build the model
combined_df.drop(['corporation'], axis=1, inplace=True) 


# In[34]:

# Printing number of columns after dropping the corporation columns
combined_df.columns


# In[35]:


# Dropping specific column[exited] for further modeling
combined_df1= combined_df.copy()
combined_df1.head(10)
print(combined_df1)

X=combined_df1.pop('exited')
Y= combined_df1
print (Y.shape)
print (X.shape)


# In[132]:


# Splitting the dataset as Train and Test data. Train a logistic regression model on this dataset, 
# predict the target variable for the test set, and print the coefficients of the trained model.
X_train,X_test,y_train,y_test = train_test_split(Y,X,test_size=0.3,random_state=100)
trained_model = LogisticRegression(solver='liblinear')
trained_model.fit(X_train,y_train)
print ('Printing the co-efficients :', trained_model.coef_)

y_prediction = trained_model.predict(X_test)
y_prediction_probability = trained_model.predict_proba(X_test)


# In[133]:


# Using sklearn-metrics the values of f1 socre, Recall, Precision and Acuracy are prined below. 
# The accuracy obtained is 87%
from sklearn import metrics

print("f1-score: ",metrics.f1_score(y_test,y_prediction, average='weighted'))
print("Recall: ",metrics.recall_score(y_test,y_prediction, average='weighted'))
print("Precision: ",metrics.precision_score(y_test,y_prediction, average='weighted'))
print("Accuracy: ",metrics.accuracy_score(y_test,y_prediction))
print("Confusion-Matrix :\n ", metrics.confusion_matrix(y_test,y_prediction))


import matplotlib.pyplot as plt
import seaborn as sns

# Compute the confusion matrix
cm = metrics.confusion_matrix(y_test, y_prediction)

# Plot the confusion matrix using seaborn heatmap
sns.heatmap(cm, annot=True, cmap='Oranges', fmt='g', cbar=False, 
            xticklabels=['Class 1', 'Class 2', 'Class 3'], 
            yticklabels=['Class 1', 'Class 2', 'Class 3'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('\nConfusion Matrix')
plt.show()


# In[311]:


# training a linear support vector machine (SVM) model on this dataset using scikit-learn's 
# LinearSVC class with a specific tolerance value and regularization parameter.
from sklearn.svm import LinearSVC

support_vector = LinearSVC(tol=1e-3, C=3)
support_vector.fit(X_train, y_train)


# In[312]:


# Printing the Train and Test score values.
print(f"Train score: {support_vector.score(X_train, y_train)}")
print(f"Test score: {support_vector.score(X_test, y_test)}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions for Xtrain and Xtest
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


# In[187]:


# training a decision tree classifier model on this dataset using scikit-learn's DecisionTreeClassifier class 
# with a specific maximum depth and maximum leaf nodes.
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=7)
tree.fit(X_train, y_train)


# In[188]:


# Train and Test accuracy
print(f"Train score: {tree.score(X_train, y_train)}")
print(f"Test score: {tree.score(X_test, y_test)}")

# Plotting the graph to analyze test, train score
import matplotlib.pyplot as plt

train_scores = []
test_scores = []
max_leaf_nodes = [2, 3, 4, 5, 6]

for leaf_nodes in max_leaf_nodes:
    tree = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=leaf_nodes)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

plt.plot(max_leaf_nodes, train_scores, label="Train score")
plt.plot(max_leaf_nodes, test_scores, label="Test score")
plt.xlabel("Max Leaf Nodes")
plt.ylabel("Accuracy Score")
plt.legend()
plt.show()


# In[193]:


# training a Random Forest classifier model on this dataset using scikit-learn's RandomForestClassifier class 
# with a specific maximum depth and maximum leaf nodes.
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=2, max_leaf_nodes=7)
forest.fit(X_train, y_train)


# In[194]:


# Printing the Train, Test accuracy scores
print(f"Train score: {forest.score(X_train, y_train)}")
print(f"Test score: {forest.score(X_test, y_test)}")

# Plotting the graph to analyze test, train score
import matplotlib.pyplot as plt

train_scores = []
test_scores = []
max_leaf_nodes = [2, 3, 4, 5, 6]

for leaf_nodes in max_leaf_nodes:
    forest = RandomForestClassifier(max_depth=3, max_leaf_nodes=leaf_nodes)
    forest.fit(X_train, y_train)
    train_scores.append(forest.score(X_train, y_train))
    test_scores.append(forest.score(X_test, y_test))

plt.plot(max_leaf_nodes, train_scores, label="Train score")
plt.plot(max_leaf_nodes, test_scores, label="Test score")
plt.xlabel("Max Leaf Nodes")
plt.ylabel("Accuracy Score")
plt.legend()
plt.show()


# In[313]:

#Saving the model in pkl file
import pickle
with open('model_pickle_employee', 'wb') as f:
    pickle.dump(tree,f)
    


# In[ ]:




