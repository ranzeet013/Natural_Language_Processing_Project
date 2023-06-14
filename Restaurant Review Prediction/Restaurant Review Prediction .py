#!/usr/bin/env python
# coding: utf-8

# # Restaurant Review Prediction

# The Restaurant Review Prediction project aims to develop a machine learning model that can predict the sentiment of restaurant reviews based on textual data. The goal is to classify the reviews as positive or negative to provide insights into customer satisfaction and help restaurant owners understand customer sentiments.

# # Importing Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:
# 
# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations
# 
# Scikit-learn: for machine learning algorithms

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings 
warnings.filterwarnings('ignore')


# # Importing Dataset

# In[3]:


dataframe = pd.read_csv('Restaurant_Reviews.tsv', 
                        sep = '\t', 
                        quoting = 3)


# # Exploring Dataset

# Data exploration is an essential step in any machine learning project, including spam message classification. It involves gaining a deeper understanding of the dataset, its characteristics, and the relationships between its variables.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[7]:


dataframe['Liked'].value_counts()


# # Data Cleaning 

# Data cleaning is an essential step in preparing your data for analysis or modeling. It involves identifying and handling inconsistencies, errors, missing values, outliers, and irrelevant information in your dataset. 

# In[9]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[10]:


dataframe.columns


# In[11]:


positive = dataframe[dataframe['Liked'] == 1]
negative = dataframe[dataframe['Liked'] == 0]


# In[12]:


positive.head()


# In[13]:


negative.head()


# In[14]:


corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataframe['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)

    corpus.append(review)


# In[15]:


print(corpus)


# CountVectorizer is a feature extraction technique used to convert a collection of text documents into a matrix of token counts. It is part of the text preprocessing and feature extraction module in scikit-learn, a popular machine learning library in Python.
# 
# CountVectorizer works by tokenizing the text data, building a vocabulary of known words, and representing each document as a vector of word frequencies or counts.

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer


# In[17]:


cv = CountVectorizer(max_features = 1500)


# In[18]:


x = cv.fit_transform(corpus).toarray()


# In[19]:


y = dataframe.iloc[:, 1].values


# In[20]:


x.shape, y.shape


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


# In[23]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# GaussianNB (Gaussian Naive Bayes) is a classification algorithm based on the Naive Bayes theorem and assumes that the features are normally distributed. It is part of the Naive Bayes family of algorithms and is particularly useful for solving classification problems with continuous features.

# In[24]:


from sklearn.naive_bayes import GaussianNB


# In[25]:


gaussian = GaussianNB()


# In[26]:


gaussian.fit(x_train, y_train)


# In[27]:


y_pred_gaussian = gaussian.predict(x_test)


# # Error Analysis

# Error analysis is a crucial step in evaluating and improving the performance of a spam message classification model. It involves analyzing the errors made by the model to gain insights into the types of misclassifications and identify patterns or common characteristics that contribute to these errors.

# In[28]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[29]:


accuracy_score = accuracy_score(y_test, y_pred_gaussian)


# In[30]:


accuracy_score


# In[31]:


print(classification_report(y_test, y_pred_gaussian))


# In[32]:


confusion_matrix = confusion_matrix(y_test, y_pred_gaussian)


# In[33]:


confusion_matrix


# In[34]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True,
            cmap = 'RdPu')


# # Prediction

# In[42]:


print(y_test[4]), print(y_pred_gaussian[4])


# In[44]:


print(y_test[34]), print(y_pred_gaussian[34])

