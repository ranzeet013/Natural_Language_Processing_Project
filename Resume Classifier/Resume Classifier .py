#!/usr/bin/env python
# coding: utf-8

# # Resume Classifier

# The Resume Classifier Project is an artificial intelligence (AI) project aimed at developing a model that can automatically classify resumes based on their content. The goal of this project is to streamline the hiring process by automatically identifying and categorizing resumes according to specific criteria, such as skills, experience, education, and job titles.
# 
# The project involves leveraging machine learning techniques to train a classifier that can accurately analyze the textual information present in resumes and assign appropriate categories or labels. The classifier is trained on a labeled dataset, which consists of a large number of resumes that have been manually classified by human experts

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


resumedata = pd.read_csv('UpdatedResumeDataSet.csv', encoding = 'utf-8')


# In[4]:


resumedata['cleaned_data'] = ''
resumedata.head()


# In[5]:


resumedata.tail()


# In[6]:


print(resumedata['Category'].value_counts())


# # Data Visualization

# Data visualization refers to the presentation of data in a visual format, such as charts, graphs, maps, or infographics, to effectively communicate information, patterns, and insights. It is a powerful tool for analyzing and understanding complex data sets, as it allows users to visually explore and interpret information more easily than through raw data alone.

# In the Resume Classifier Project, data visualization can play a vital role in providing insights and communicating the performance and results of the classifier. Here is a description of how data visualization can be utilized in this project:

# In[10]:


plt.figure(figsize = (15, 6))
plt.xticks(rotation = 90)
sns.countplot(y = 'Category', data = resumedata)


# In[13]:


from matplotlib.gridspec import GridSpec
targetCounts = resumedata['Category'].value_counts()
targetLabels  = resumedata['Category'].unique()

plt.figure(1, figsize=(25,25))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, 3)]
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()


# # Cleaning The Dataset

# Cleaning the dataset in the Resume Classifier Project typically involves applying various data preprocessing techniques to ensure that the textual data from resumes is in a clean and standardized format. One commonly used tool for this task is regular expressions (regex), which allows for efficient text matching and manipulation.The dataset cleaning process using regex:

# In[14]:


import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  
    resumeText = re.sub('RT|cc', ' ', resumeText)  
    resumeText = re.sub('#\S+', '', resumeText)  
    resumeText = re.sub('@\S+', '  ', resumeText)  
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText) 
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  
    return resumeText
    
resumedata['cleaned_resume'] = resumedata.Resume.apply(lambda x: cleanResume(x))


# In[16]:


resumedata.head()


# # WordCloud

# In the Resume Classifier Project, a word cloud can be a valuable visualization tool to gain insights into the most frequent and prominent words in the resumes. A word cloud visually represents the frequency or importance of words in a text corpus by displaying them in different sizes or colors, where larger or bolder words indicate higher frequency or significanc.In this way word cloud can be used in this project:

# In[17]:


import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords =[]
Sentences = resumedata['Resume'].values
cleanedSentences = ""
for i in range(0,160):
    cleanedText = cleanResume(Sentences[i])
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)
    
wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Encoding

# In the Resume Classifier Project, encoding refers to the process of transforming categorical variables into numerical representations that can be understood and processed by machine learning algorithms. Since resumes may contain categorical information such as job titles, industries, or experience levels, encoding is necessary to convert these categories into numerical values.

# In[19]:


from sklearn.preprocessing import LabelEncoder

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumedata[i] = le.fit_transform(resumedata[i])


# # Splitting Dataset

# In the Resume Classifier Project, splitting the dataset refers to dividing the available data into separate subsets for training, validation, and testing. This division is essential to evaluate the performance of the classifier and ensure its generalizability.

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


# In[21]:


requiredText = resumedata['cleaned_resume'].values
requiredTarget = resumedata['Category'].values


# # Feature Extraction

# In the Resume Classifier Project, feature extraction involves transforming the raw textual data from resumes into numerical representations that capture the relevant information for classification. The goal is to extract meaningful features that can be used as input to machine learning algorithms.

# In[23]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)
print ("Feature completed .....")


# In[24]:


X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)


# # Model Training

# Model training in the Resume Classifier Project involves using the labeled dataset and the extracted features to train a machine learning model that can accurately classify resumes. The trained model will learn patterns and relationships in the data, enabling it to make predictions on new, unseen resumes

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)


# In[28]:


print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# # Error Analysis

# Error analysis in the Resume Classifier Project involves analyzing and understanding the errors made by the trained classifier when classifying resumes. By carefully examining these errors, valuable insights can be gained to further improve the performance of the classifier.

# In[30]:


from sklearn import metrics


# In[31]:


print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))


# # Thanks !
