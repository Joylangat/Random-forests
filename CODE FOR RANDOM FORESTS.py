#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# In[3]:


df = pd.read_csv("insurance.csv")


# In[4]:


# Basic statistical details

df.describe()


# In[5]:


#Missing data check

df.isnull().sum()


# In[6]:


df.dtypes 


# In[7]:


# Removing duplicate rows.

print('Duplicate Rows Count : ', df.duplicated().sum())

df=df.drop_duplicates(keep="first")


# In[8]:


# Correlation Heatmap

plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)


# In[9]:


df.head()


# In[10]:


def encoder(method, dataframe, columns_label, columns_onehot):
    
    if method == 'labelEncoder':      
    
        df_lbl = dataframe.copy()
    
        for col in columns_label:
            label = LabelEncoder()
            label.fit(list(dataframe[col].values))
            df_lbl[col] = label.transform(df_lbl[col].values)
        
        return df_lbl
    
    elif method == 'oneHotEncoder':
        
        df_oh = dataframe.copy()

        df_oh= pd.get_dummies(data = df_oh, prefix = 'ohe', prefix_sep='_',
                       columns = columns_onehot,
                       drop_first =True,
                       dtype='int8')
        
        return df_oh


# In[11]:


method =['labelEncoder', 'oneHotEncoder']
dataframe = df.copy()
columns_label = ['sex', 'smoker', 'region']
columns_onehot = ['sex', 'smoker', 'region', 'children']

df_label = encoder(method[0], dataframe, columns_label, columns_onehot)

df_onehot = encoder(method[1], dataframe, columns_label, columns_onehot)

df_label.head()


# In[12]:


df_onehot.head()


# In[13]:


def scaler(method, data, columns_scaler):
    
    if method == 'standartScaler':
        
        Standart = StandardScaler()

        df_standart = data.copy()

        df_standart[columns_scaler]=Standart.fit_transform(df_standart[columns_scaler])
        
        return df_standart
        
    elif method == 'minMaxScaler':
        
        MinMax= MinMaxScaler()

        df_minmax = data.copy()

        df_minmax[columns_scaler]=MinMax.fit_transform(df_minmax[columns_scaler])
        
        return df_minmax
    
    elif method =='npLog':
        
        df_nplog = data.copy()

        df_nplog[columns_scaler]=np.log(df_nplog[columns_scaler])
        
        return df_nplog
    
    elif method == 'default':
        
        return data


# In[14]:


method = 'minMaxScaler'
data = df_label
columns_scaler = ['bmi', 'charges']

df_scaler = scaler(method, data, columns_scaler)


# In[15]:


df_scaler .head()


# In[16]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[17]:


X = df_scaler.drop('charges',axis=1)
y = df_scaler['charges']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=16)


# In[18]:


RandomForestRegressor=RandomForestRegressor(random_state = 42)

RandomForestRegressor.fit(X_train, y_train)

train_pred = RandomForestRegressor.predict(X_train)
test_pred = RandomForestRegressor.predict(X_test)

print('Train MAE :', mean_absolute_error(y_train,train_pred))

print('Test MAE :', mean_absolute_error(y_test, test_pred))

print('Train R2 :', r2_score(y_train,train_pred))

print('Test R2 :', r2_score(y_test, test_pred))


# In[19]:


def regression_gridsearch(param_grid_data, model_params, func_input):
    last=[]
    model_params=model_params
    for params in param_grid_data:
        
        result = {}
        result['encoder'] = params['encoder']
        result['scaler'] = params['scaler']
        result['random_state'] = params['random_state']
        result['test_size'] = params['test_size']
        
        data = encoder(params['encoder'], func_input['data'], func_input['columns_label'], func_input['columns_onehot'] )
        data = scaler(params['scaler'], data,func_input['columns_scaler'])
        
        
        X = data.drop(func_input['output'],axis=1)
        y = data[func_input['output']].values.reshape(-1,)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=params['test_size'],random_state=params['random_state'])      
        
        for model_name, mp in model_params.items():
            res={}
            res=result
            clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            test_score = r2_score(y_test, pred)
            test_mape = mean_absolute_percentage_error(y_test, pred)
            res['model']=model_name
            res['best_score']=clf.best_score_
            res['test_score']=test_score
            res['test_mape']=test_mape
            res['best_params']=clf.best_params_     
            last.append(res)
    result = pd.DataFrame(last, columns=['encoder','scaler','random_state','test_size','model','best_score','test_score','test_mape','best_params'])
    
    return result 


# In[20]:


from sklearn.ensemble import RandomForestRegressor

param_grid_data = { 
    'encoder' : ['labelEncoder', 'oneHotEncoder'],
    'scaler' : ['standartScaler', 'minMaxScaler', 'npLog', 'default'],
    'random_state' : [16],
    'test_size' : [0.3]
}

param_grid_data = [dict(zip(param_grid_data.keys(), v)) for v in itertools.product(*param_grid_data.values())]

func_input = {
    'columns_label': ['sex', 'smoker', 'region'],
    'columns_onehot' : ['sex', 'smoker', 'region', 'children'],
    'columns_scaler' : ['bmi', 'charges'],
    'output' : ['charges'],
    'data' : df,
}

RandomForestRegressor_params = {
    'RFRegressor': {
    'model': RandomForestRegressor(),
    'params' : {'bootstrap': [True], # True, False
                'max_depth': [10, 20, 50], # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None
                'max_features': ['auto','sqrt'], # 'auto','sqrt'
                'min_samples_leaf': [1, 2, 4], #1, 2, 4
                'min_samples_split': [2, 5], # 2, 5, 10
                'n_estimators': [100, 200],
                'random_state' : [42]}}, # 200, 400, 800, 1000, 1200
        }


# In[ ]:


RFR_result=regression_gridsearch(param_grid_data, RandomForestRegressor_params, func_input)


# In[ ]:


RFR_result.head()


# In[ ]:


def best_params(model_name, result):
    
    best_index =np.argmax(result['test_score'])

    best_params = result['best_params'][best_index]
    best_encoder = result['encoder'][best_index]
    best_scaler = result['scaler'][best_index]
    best_random_state = result['random_state'][best_index]
    best_test_size = result['test_size'][best_index]
    
    print('\nModel Name: ', model_name, '\nBest Params: ', best_params, '\nBest Encoder: ', best_encoder, '\nBest Scaler: ', best_scaler, '\nBest Random State: ', best_random_state, '\nBest Test Size: ', best_test_size)

    best_params = {
        'params' : best_params,
        'encoder' : best_encoder,
        'scaler' : best_scaler,
        'random_state' : best_random_state,
        'test_size' : best_test_size   
        }
    
    return best_params


# In[ ]:


best_params_rfr = best_params('Random Forest Regression', RFR_result)


# In[ ]:


def best_data(best_params, func_input):

    data = encoder(best_params['encoder'], func_input['data'], func_input['columns_label'], func_input['columns_onehot'])
    data = scaler(best_params['scaler'], data, func_input['columns_scaler'])

    X = data.drop(func_input['output'],axis=1)
    y = data[func_input['output']].values.reshape(-1,)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=best_params['test_size'],random_state=best_params['random_state'])
    
    return X, y, X_train, X_test, y_train, y_test


# In[ ]:


X, y, X_train, X_test, y_train, y_test = best_data(best_params_rfr, func_input)

RandomForestRegressor=RandomForestRegressor(bootstrap = best_params_rfr['params']['bootstrap'], max_depth = best_params_rfr['params']['max_depth'],
                                            max_features = best_params_rfr['params']['max_features'], min_samples_leaf = best_params_rfr['params']['min_samples_leaf'],
                                            min_samples_split = best_params_rfr['params']['min_samples_split'], n_estimators = best_params_rfr['params']['n_estimators'],
                                            )

RandomForestRegressor.fit(X_train, y_train)

train_pred = RandomForestRegressor.predict(X_train)
test_pred = RandomForestRegressor.predict(X_test)

print('Train MAE :', mean_absolute_error(y_train,train_pred))

print('Test MAE :', mean_absolute_error(y_test, test_pred))

print('Train R2 :', r2_score(y_train,train_pred))

print('Test R2 :', r2_score(y_test, test_pred))


# In[ ]:


plt.scatter(x = y_test, y = test_pred, color = 'blue', marker = 'o', s = 70, alpha = 0.5,
          label = 'Test data')
plt.title('Test and Pred')
plt.xlabel('Test')
plt.ylabel('Pred')
plt.show()


# In[ ]:




