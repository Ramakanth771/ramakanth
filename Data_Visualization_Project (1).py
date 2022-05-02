#!/usr/bin/env python
# coding: utf-8

# In[10]:


# In[1]:

import os
print("Current Working Directory is:",os.getcwd())
os.chdir("C:\\Users\\R A M\\Downloads")
print("New Working Directory is:",os.getcwd())


# In[11]:


# In[2]:

import warnings
warnings.filterwarnings('ignore')


# In[12]:


# In[3]:

get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pylab as plt
from dmba import classificationSummary, gainsChart


# In[ ]:





# In[13]:


# In[4]:

x=pd.read_csv('DP.csv')
x.tail(3)


# In[8]:


# In[5]:

print("Number of Missing Values in Column:\n",x.isnull().sum())
print("Dimension of the Dataset is:",x.shape)


# In[11]:


# In[6]:

Newx = x.copy()
Newx.dropna(inplace=True)
print("Dimension of the Dataset is:",x.shape)
print("Number of Missing Values in Column:\n",Newx.isnull().sum())


# In[12]:


# In[7]:

x_df=pd.DataFrame(Newx)


# In[13]:


# In[8]:

Newx.info()


# In[14]:


# In[9]:

x_df.drop(['Review'], axis=1, inplace = True)
x_df.drop(['EmployeeID'], axis=1, inplace = True)
x_df.drop(['Sr.No'],axis=1,inplace=True)


# In[15]:


# In[10]:


x_df["Companyname"] = x_df["Companyname"].astype('category')
x_df["Companytype"] = x_df["Companytype"].astype('category')
x_df["Gender"] = x_df["Gender"].astype('category')
x_df["EmployeeDepartment"] = x_df["EmployeeDepartment"].astype('category')
x_df["JobRole"] = x_df["JobRole"].astype('category')
x_df["Attrition"] = x_df["Attrition"].astype('category')
x_df["location"] = x_df["location"].astype('category')
x_df["OverTime"] = x_df["OverTime"].astype('category')
x_df["Worktimingsatisfaction"] = x_df["Worktimingsatisfaction"].astype('category')

x_df.dtypes


# In[16]:


# In[11]:


x_df.drop_duplicates()


# In[17]:


# In[13]:


s = x_df.Companyname
Counts = x_df.Companyname.value_counts()
Percentage = x_df.Companyname.value_counts(normalize=True)*100
pd.DataFrame({'Counts':Counts, 'Percentage':Percentage})


# In[18]:


# In[14]:


s = x_df.Gender
Counts = x_df.Gender.value_counts()
Percentage = x_df.Gender.value_counts(normalize=True)*100
pd.DataFrame({'Counts':Counts, 'Percentage':Percentage})


# In[19]:


# In[15]:


s = x_df.EmployeeDepartment
Counts = x_df.EmployeeDepartment.value_counts()
Percentage = x_df.EmployeeDepartment.value_counts(normalize=True)*100
pd.DataFrame({'Counts':Counts, 'Percentage':Percentage})


# In[20]:


# In[16]:


s = x_df.location
Counts = x_df.location.value_counts()
Percentage = x_df.location.value_counts(normalize=True)*100
pd.DataFrame({'Counts':Counts, 'Percentage':Percentage})


# In[21]:


# In[17]:


name= x_df['Gender']
att=x_df['Attrition']
fig = plt.figure()
 
# Horizontal Bar Plot
plt.bar(name,att)
plt.title("Gender vs Attrition")
plt.ylabel("Attrition")
plt.xlabel("Gender")
# Show Plot
plt.show()


# In[22]:


# In[18]:


x_df["Companyname"] = x_df["Companyname"].cat.codes
x_df["Companytype"] = x_df["Companytype"].cat.codes
x_df["Gender"] = x_df["Gender"].cat.codes
x_df["EmployeeDepartment"] = x_df["EmployeeDepartment"].cat.codes
x_df["JobRole"] = x_df["JobRole"].cat.codes
x_df["Attrition"] = x_df["Attrition"].cat.codes
x_df["location"] = x_df["location"].cat.codes
x_df["OverTime"] = x_df["OverTime"].cat.codes
x_df["Worktimingsatisfaction"] = x_df["Worktimingsatisfaction"].cat.codes

x_df.dtypes


# In[23]:


# In[19]:


x_df.corr()


# In[24]:


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')
influential_features = ['Gender', 'JobRole', 'OverTime', 'Worktimingsatisfaction', 'Attrition']
x_df[influential_features].corr()


# In[25]:


# In[21]:


sn.heatmap(x_df[influential_features].corr(), annot=True)


# In[26]:


# In[22]:


x_df.corr()


# In[27]:


# In[23]:


sn.heatmap(x_df.corr(),annot=False)


# In[28]:


# In[24]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary


# In[29]:


# In[25]:


import sys
get_ipython().system('{sys.executable} -m pip install pydotplus')


# In[30]:


# # Random forest
# 

# In[38]:


y = x_df[influential_features]['Attrition']
X = x_df[influential_features].drop(columns=['Attrition'])


# In[31]:


# In[27]:




y = x_df['Attrition']
X = x_df.drop(columns=['Attrition'])
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

rf = RandomForestClassifier(n_estimators=500, random_state=1)
rf.fit(train_X, train_y)


# In[32]:


# In[28]:


importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

df = pd.DataFrame({'feature': train_X.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance')
print(df)

ax = df.plot(kind='barh', xerr='std', x='feature', legend=False)
ax.set_ylabel('')

plt.tight_layout()
plt.show()


# In[35]:


# In[56]:


# Predict on the test set results

y_pred = rf.predict(valid_X)
# Check accuracy score 

print('Random forest Model accuracy score  : {0:0.4f}'. format(accuracy_score(valid_y, y_pred)))
from sklearn.metrics import classification_report

print(classification_report(valid_y, y_pred))
cm = confusion_matrix(valid_y, y_pred)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))


# In[42]:


get_ipython().system('pip install mord')


# In[43]:


# # Logistic regression

# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from mord import LogisticIT
import matplotlib.pylab as plt
import seaborn as sns
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score

y = x_df['Attrition']
X = x_df.drop(columns=['Attrition'])

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
logit_reg.fit(train_X, train_y)

print('intercept ', logit_reg.intercept_[0])
print(pd.DataFrame({'coeff': logit_reg.coef_[0]}, index=X.columns).transpose())
print()
print('AIC', AIC_score(valid_y, logit_reg.predict(valid_X), df = len(train_X.columns) + 1))


# In[44]:


# In[47]:


classificationSummary(train_y, logit_reg.predict(train_X))
classificationSummary(valid_y, logit_reg.predict(valid_X))
y_pred=logit_reg.predict(valid_X)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(valid_y, y_pred)
cnf_matrix
from sklearn.metrics import classification_report

print(classification_report(valid_y, y_pred))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(valid_y, y_pred)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# In[45]:


# # Naive bayes
# 

# In[43]:


predictors = ['Worktimingsatisfaction', 'OverTime', 'EmployeeAge', 'Companyname', 'Companytype', 'YearsAtCompany', 'Gender', 'JobRole', 'Careergrowth', 'Jobsecurity', 'Worklifebalance', 'SkillDevelopment', 'Companyculture', 'WorkSatisfaction', 'Salaryandbenefits','OverTime' ]
outcome = 'Attrition'

X = pd.get_dummies(x_df[predictors])
y = x_df['Attrition']


# split into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, random_state=1)

# run naive Bayes
delays_nb = MultinomialNB(alpha=0.01)
delays_nb.fit(X_train, y_train)

# predict probabilities
predProb_train = delays_nb.predict_proba(X_train)
predProb_valid = delays_nb.predict_proba(X_valid)

# predict class membership
y_valid_pred = delays_nb.predict(X_valid)
y_train_pred = delays_nb.predict(X_train)
from sklearn.metrics import accuracy_score

print('Naive bayes Model accuracy score: {0:0.4f}'. format(accuracy_score(y_valid, y_valid_pred)))
from sklearn.metrics import classification_report

print(classification_report(y_valid, y_valid_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_valid, y_valid_pred)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# In[46]:


# # SVM

# In[37]:


from sklearn.svm import SVC
y = x_df['Attrition']
X = x_df.drop(columns=['Attrition'])

# split into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.60, random_state=0)
# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0) 

# fit classifier to training set
poly_svc.fit(X_train,y_train)

# make predictions on test set
y_pred=poly_svc.predict(X_valid)

# compute and print accuracy score
print('SVM Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_valid, y_pred)))
from sklearn.metrics import classification_report

print(classification_report(y_valid, y_pred))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_valid, y_pred)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))


# In[47]:


# In[67]:


import numpy as np
import matplotlib.pyplot as plt
location ={ ' LR': 89.84, 'Random forest':86.77, 'Naive bayes':82.88,'SVM':64.77}
names=list(location.keys()) # extracting keys from student and stored as list in name
values=list(location.values()) 
plt.barh(names,values,color='y') # Bar method is used for barplot. First parameter is c
plt.title("Accuracy of machine learning algorithm for selected variables")
plt.ylabel("Algorthim")
plt.xlabel("Percentage")
plt.grid(True)
plt.show()


# In[48]:


# # Selected variables

# # Random forest for selected variables

# In[59]:


y = x_df[influential_features]['Attrition']
X = x_df[influential_features].drop(columns=['Attrition'])
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

rf = RandomForestClassifier(n_estimators=500, random_state=1)
rf.fit(train_X, train_y)
# Predict on the test set results

y_pred = rf.predict(valid_X)
# Check accuracy score 

print('Random forest Model accuracy score  : {0:0.4f}'. format(accuracy_score(valid_y, y_pred)))
from sklearn.metrics import classification_report

print(classification_report(valid_y, y_pred))
cm = confusion_matrix(valid_y, y_pred)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))


# In[49]:


# # Logistic regression

# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from mord import LogisticIT
import matplotlib.pylab as plt
import seaborn as sns
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score

y = x_df[influential_features]['Attrition']
X = x_df[influential_features].drop(columns=['Attrition'])

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
logit_reg.fit(train_X, train_y)

print('intercept ', logit_reg.intercept_[0])
print(pd.DataFrame({'coeff': logit_reg.coef_[0]}, index=X.columns).transpose())
print()
print('AIC', AIC_score(valid_y, logit_reg.predict(valid_X), df = len(train_X.columns) + 1))
classificationSummary(train_y, logit_reg.predict(train_X))
classificationSummary(valid_y, logit_reg.predict(valid_X))
y_pred=logit_reg.predict(valid_X)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(valid_y, y_pred)
cnf_matrix
from sklearn.metrics import classification_report

print(classification_report(valid_y, y_pred))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(valid_y, y_pred)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# In[50]:


# # SVM

# In[62]:


from sklearn.svm import SVC
y = x_df[influential_features]['Attrition']
X = x_df[influential_features].drop(columns=['Attrition'])

# split into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.60, random_state=0)
# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0) 

# fit classifier to training set
poly_svc.fit(X_train,y_train)

# make predictions on test set
y_pred=poly_svc.predict(X_valid)

# compute and print accuracy score
print('SVM Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_valid, y_pred)))
from sklearn.metrics import classification_report

print(classification_report(y_valid, y_pred))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_valid, y_pred)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))


# In[51]:


# # Naive bayes

# In[64]:


predictors = ['Worktimingsatisfaction', 'OverTime', 'Gender', 'JobRole' ]
outcome = 'Attrition'

X = pd.get_dummies(x_df[predictors])
y = x_df['Attrition']


# split into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, random_state=1)

# run naive Bayes
delays_nb = MultinomialNB(alpha=0.01)
delays_nb.fit(X_train, y_train)

# predict probabilities
predProb_train = delays_nb.predict_proba(X_train)
predProb_valid = delays_nb.predict_proba(X_valid)

# predict class membership
y_valid_pred = delays_nb.predict(X_valid)
y_train_pred = delays_nb.predict(X_train)
from sklearn.metrics import accuracy_score

print('Naive bayes Model accuracy score: {0:0.4f}'. format(accuracy_score(y_valid, y_valid_pred)))
from sklearn.metrics import classification_report

print(classification_report(y_valid, y_valid_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_valid, y_valid_pred)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# In[53]:


# # Word cloud for review

# In[82]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
df = pd.read_csv("review.csv")
df
text1 = " ".join(review for review in df.Review)

wc1 = WordCloud(collocations = False, background_color = 'black').generate(text1)
# saving the image
wc1.to_file('got.png')
# Display the generated Word Cloud

plt.imshow(wc1, interpolation='bilinear')
plt.axis("off")
plt.show()

