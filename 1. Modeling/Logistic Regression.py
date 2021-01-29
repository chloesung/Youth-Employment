#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# # Import & Data preprocessing

# In[2]:


df1 = pd.read_csv('permtempdata_logit_stepwise.csv', engine='python')
df2 = pd.read_csv('bigsmalldata_logit_stepwise.csv', engine='python')


# In[3]:


df1 = df1.drop('Unnamed: 0',axis=1)
df2 = df2.drop('Unnamed: 0',axis=1)


# In[4]:


x1 = pd.get_dummies(df1['uniMajor'], prefix='uniMajor')
x2 = pd.get_dummies(df1['youthLoc'], prefix='youthLoc')
x3 = pd.get_dummies(df1['newedu'], prefix='newedu')


# In[5]:


X = df1.drop(['uniMajor','youthLoc','newedu','permIdc'],axis=1)
X = X.replace(1,0)
X = X.replace(2,1)
X1 = pd.concat([X,x1,x2,x3],axis=1)


# In[37]:


#더미변수로 만들어주기

x1 = pd.get_dummies(df2['uniType'], prefix='uniType')
x2 = pd.get_dummies(df2['uniMajor'], prefix='uniMajor')
x3 =  pd.get_dummies(df2['youthHouse'], prefix='youthHouse')

X = df2.drop(['uniMajor','uniType','corpSize','youthHouse'],axis=1)
X = X.replace(1,0)
X = X.replace(2,1)
X2 = pd.concat([X,x1,x2,x3],axis=1)


# In[7]:


X=X1
y = df1['permIdc']


# # Logistic Regression - 정규직/비정규직

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_validTest, y_train, y_validTest = train_test_split(X, y, train_size=0.6, test_size = 0.4, random_state=1234) 
X_valid, X_test, y_valid, y_test = train_test_split(X_validTest, y_validTest, train_size = 0.5, test_size = 0.5, random_state = 1234)


# In[9]:


from sklearn.linear_model import LogisticRegression
m1 = LogisticRegression(max_iter = 4000)
m1.fit(X_train, y_train)


# In[10]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=m1, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_valid, y_valid)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[11]:


model = LogisticRegression( C= 0.1, penalty= 'l2',solver= 'newton-cg')
model.fit(X_train, y_train)


# In[12]:


X.columns


# In[13]:


standard = [1,1,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4
gpa = [0,1,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4
grad = [1,0,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4
inter = [1,1,0,1,1,1]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4
abr = [1,1,1,0,1,1]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4
cert = [1,1,1,1,0,1]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4
club = [1,1,1,1,1,0]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4

test = np.array([standard, gpa, grad, inter, abr, cert, club])
print(model.predict_proba(test))


# In[14]:


a = model.predict_proba(test)


# In[15]:


dataset = pd.DataFrame({'Column1': a[:, 0], 'Column2': a[:, 1]})


# In[16]:


dataset


# In[17]:


standard = [1,1,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4
bs = [1,1,1,1,1,1]+[1]+[0]*10+[0]*1+[1]+[0]*14+[1]+[0]*4
dg = [1,1,1,1,1,1]+[1]+[0]*10+[0]*2+[1]+[0]*13+[1]+[0]*4
ic = [1,1,1,1,1,1]+[1]+[0]*10+[0]*3+[1]+[0]*12+[1]+[0]*4
gj = [1,1,1,1,1,1]+[1]+[0]*10+[0]*4+[1]+[0]*11+[1]+[0]*4
dj = [1,1,1,1,1,1]+[1]+[0]*10+[0]*5+[1]+[0]*10+[1]+[0]*4
ws = [1,1,1,1,1,1]+[1]+[0]*10+[0]*6+[1]+[0]*9+[1]+[0]*4
gg = [1,1,1,1,1,1]+[1]+[0]*10+[0]*7+[1]+[0]*8+[1]+[0]*4
gw = [1,1,1,1,1,1]+[1]+[0]*10+[0]*8+[1]+[0]*7+[1]+[0]*4
cb = [1,1,1,1,1,1]+[1]+[0]*10+[0]*9+[1]+[0]*6+[1]+[0]*4
cn = [1,1,1,1,1,1]+[1]+[0]*10+[0]*10+[1]+[0]*5+[1]+[0]*4
jb = [1,1,1,1,1,1]+[1]+[0]*10+[0]*11+[1]+[0]*4+[1]+[0]*4
jn = [1,1,1,1,1,1]+[1]+[0]*10+[0]*12+[1]+[0]*3+[1]+[0]*4
gb = [1,1,1,1,1,1]+[1]+[0]*10+[0]*13+[1]+[0]*2+[1]+[0]*4
gn = [1,1,1,1,1,1]+[1]+[0]*10+[0]*14+[1]+[0]*1+[1]+[0]*4
jj = [1,1,1,1,1,1]+[1]+[0]*10+[0]*15+[1]+[1]+[0]*4


test_location = np.array([standard, bs, dg, ic, gj, dj, ws, gg, gw, cb, cn, jb, jn, gb, gn , jj])
a = model.predict_proba(test_location)
data2 = pd.DataFrame({'Column1': a[:, 0], 'Column2': a[:, 1]})
data2


# In[18]:


standard = [1,1,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4
two = [1,1,1,1,1,1]+[0]+[1]+[0]*9+[1]+[0]*15+[1]+[0]*4
ind = [1,1,1,1,1,1]+[0]*2+[1]+[0]*8+[1]+[0]*15+[1]+[0]*4
edu = [1,1,1,1,1,1]+[0]*3+[1]+[0]*7+[1]+[0]*15+[1]+[0]*4
tel = [1,1,1,1,1,1]+[0]*4+[1]+[0]*6+[1]+[0]*15+[1]+[0]*4
cyber = [1,1,1,1,1,1]+[0]*5+[1]+[0]*5+[1]+[0]*15+[1]+[0]*4
poly = [1,1,1,1,1,1]+[0]*6+[1]+[0]*4+[1]+[0]*15+[1]+[0]*4
none = [1,1,1,1,1,1]+[0]*7+[1]+[0]*3+[1]+[0]*15+[1]+[0]*4
cyber1 = [1,1,1,1,1,1]+[0]*8+[1]+[0]*2+[1]+[0]*15+[1]+[0]*4
poly1 = [1,1,1,1,1,1]+[0]*9+[1]+[0]*1+[1]+[0]*15+[1]+[0]*4
poly3 = [1,1,1,1,1,1]+[0]*10+[1]+[1]+[0]*15+[0]*3+[1]+[0]

test_unimajor = np.array([standard, two, ind, edu, tel, cyber, poly, none,cyber1,poly1,poly3])
a = model.predict_proba(test_unimajor)
data3 = pd.DataFrame({'Column1': a[:, 0], 'Column2': a[:, 1]})
data3


# In[19]:


standard = [1,1,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[1]+[0]*4
two = [1,1,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[0]+[1]+[0]*3
ind = [1,1,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[0]*2+[1]+[0]*2
edu = [1,1,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[0]*3+[1]+[0]
tel = [1,1,1,1,1,1]+[1]+[0]*10+[1]+[0]*15+[0]*4+[1]

test_unimajor = np.array([standard, two, ind, edu, tel])
a = model.predict_proba(test_unimajor)
data4 = pd.DataFrame({'Column1': a[:, 0], 'Column2': a[:, 1]})
data4


# In[20]:


score_perm = pd.concat([dataset,data3,data2,data4],axis=0)


# In[21]:


len(score_perm)


# In[22]:


col = X.columns.tolist()


# In[23]:


col = ['standard'] + col


# In[24]:


data = pd.DataFrame({'variable': col})


# In[25]:


score_perm = score_perm.reset_index()


# In[26]:


data


# In[27]:


score_perm


# In[28]:


score_perm = pd.concat([data,score_perm],axis=1).drop('index',axis=1)


# In[29]:


score_perm.to_csv('score_perm.csv')


# # Logistic Regression - 대기업/중소기업

# In[38]:


X=X2
y = df2['corpSize']


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_validTest, y_train, y_validTest = train_test_split(X, y, train_size=0.6, test_size = 0.4, random_state=1234) 
X_valid, X_test, y_valid, y_test = train_test_split(X_validTest, y_validTest, train_size = 0.5, test_size = 0.5, random_state = 1234)


# In[40]:


from sklearn.linear_model import LogisticRegression
m1 = LogisticRegression(max_iter = 4000)
m1.fit(X_train, y_train)


# In[41]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=m1, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_valid, y_valid)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[42]:


model = LogisticRegression( C= 100, penalty= 'l2',solver= 'newton-cg')
model.fit(X_train, y_train)


# In[43]:


X.columns


# In[63]:


standard = [0,1,1,1,1,1]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3
gpa = [1,1,1,1,1,1]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3
grad = [0,0,1,1,1,1]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3
inter = [0,1,0,1,1,1]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3
abr = [0,1,1,0,1,1]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3
cert = [0,1,1,1,0,1]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3
club = [0,1,1,1,1,0]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3

test = np.array([standard, gpa, grad, inter, abr, cert, club])
a = model.predict_proba(test)
data1 = pd.DataFrame({'Column1': a[:, 0], 'Column2': a[:, 1]})
data1


# In[64]:


standard = [0,1,1,1,1,1]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3
two = [0,1,1,1,1,1]+[0]+[1]+[0]*4+[1]+[0]*9+[1]+[0]*3
ind = [0,1,1,1,1,1]+[0]*2+[1]+[0]*3+[1]+[0]*9+[1]+[0]*3
edu = [0,1,1,1,1,1]+[0]*3+[1]+[0]*2+[1]+[0]*9+[1]+[0]*3
tel = [0,1,1,1,1,1]+[0]*4+[1]+[0]*1+[1]+[0]*9+[1]+[0]*3
cyber = [0,1,1,1,1,1]+[0]*5+[1]+[0]*9+[1]+[1]+[0]*3

test_unimajor = np.array([standard, two, ind, edu, tel, cyber])
a = model.predict_proba(test_unimajor)
data2 = pd.DataFrame({'Column1': a[:, 0], 'Column2': a[:, 1]})
data2


# In[65]:


standard = [0,1,1,1,1,1]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3
two = [0,1,1,1,1,1]+[1]+[0]*5+[0]+[1]+[0]*8+[1]+[0]*3
ind = [0,1,1,1,1,1]+[1]+[0]*5+[0]*2+[1]+[0]*7+[1]+[0]*3
edu = [0,1,1,1,1,1]+[1]+[0]*5+[0]*3+[1]+[0]*6+[1]+[0]*3
tel = [0,1,1,1,1,1]+[1]+[0]*5+[0]*4+[1]+[0]*5+[1]+[0]*3
cyber = [0,1,1,1,1,1]+[1]+[0]*5+[0]*5+[1]+[0]*4+[1]+[0]*3
a1 = [0,1,1,1,1,1]+[1]+[0]*5+[0]*6+[1]+[0]*3+[1]+[0]*3
a2 = [0,1,1,1,1,1]+[1]+[0]*5+[0]*7+[1]+[0]*2+[1]+[0]*3
a3 = [0,1,1,1,1,1]+[1]+[0]*5+[0]*8+[1]+[0]+[1]+[0]*3
a4 = [0,1,1,1,1,1]+[0]*5+[1]+[0]*9+[1]+[1]+[0]*3



test_unimajor = np.array([standard, two, ind, edu, tel, cyber, a1,a2,a3,a4])
a = model.predict_proba(test_unimajor)
data3 = pd.DataFrame({'Column1': a[:, 0], 'Column2': a[:, 1]})
data3


# In[66]:


standard = [0,1,1,1,1,1]+[1]+[0]*5+[1]+[0]*9+[1]+[0]*3
two = [0,1,1,1,1,1]+[1]+[0]*5+[1]+[0]*9+[0]+[1]+[0]*2
a2 = [0,1,1,1,1,1]+[1]+[0]*5+[1]+[0]*9+[0]*2+[1]+[0]
a1 = [0,1,1,1,1,1]+[1]+[0]*5+[1]+[0]*9+[0]*3+[1]

test_unimajor = np.array([standard, two, a2, a1])
a = model.predict_proba(test_unimajor)
data4 = pd.DataFrame({'Column1': a[:, 0], 'Column2': a[:, 1]})
data4


# In[62]:


df2[df2['uniMajor']==19990920]


# In[80]:


score_corp = pd.concat([data1,data2,data3,data4],axis=0).reset_index()
col = ['standard'] + X.columns.tolist()
data = pd.DataFrame({'variable': col})
score_corp = pd.concat([data,score_corp],axis=1).drop('index',axis=1)


# In[81]:


score_corp


# In[82]:


score_corp.to_csv('score_corp.csv')

