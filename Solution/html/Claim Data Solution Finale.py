#!/usr/bin/env python
# coding: utf-8

# #                                                                   **Solution For Simulated Claim Data** 

# ### Gaurav Modi
# ### gaurav.modi@gmail.com 
# ### M (908 265 0151)

# Main Points 
# ===========
# #### Accuracy = 0.88
# #### ROC AUC = 0.86
# 
# precision    recall  f1-score   support
# 
#            0       0.89      0.97      0.93    113138
#            1       0.83      0.53      0.65     28262
# 
#     accuracy                           0.88    141400
#    macro avg       0.86      0.75      0.79    141400
# weighted avg       0.88      0.88      0.87    141400
# 
# Profile Report is available in same folder as HTML file. 
# 
# Future work
# ===========
# Simple Imputer used because of time constrain, we can improve performance and data quality by using advance imputation algorithms. 
# Try different categorical encoding algorithm to acheive better performance and accuracy,
# more hyperparameter tuning,
# ensambles Models, bagging boosting and stacking 

# In[124]:


get_ipython().system('pip install -q klib')
get_ipython().system('pip install -q pandas-profiling')
get_ipython().system('pip install -q --pre pycaret')
get_ipython().system('pip install -q category_encoders')
get_ipython().system('pip install -q catboost')
get_ipython().system('pip install -q pandas-profiling')
get_ipython().system('pip install -q shap')


# In[126]:


import numpy as np 
import pandas as pd 
import category_encoders as ce
#pd.set_option('max_columns', None)
from pandas_profiling import ProfileReport

from sklearn.impute import SimpleImputer
import klib
import shap

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, f1_score
from pandas.api.types import is_numeric_dtype
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool


# In[50]:


# Loading Data, dropping unnessery colums, renaming columns 
data = pd.read_csv("C:/Users/gmodi/Downloads/archive/claimData.csv",index_col=None, dtype={'place_of_service_code': 'category', 'is_denial': 'category'})
data = data.drop(['Unnamed: 0','firm_id','service_year','cpt_code_description','state','payer_name','is_approved','modifier_code_1','modifier_code_2','modifier_code_3','modifier_code_4'], axis=1)
data = data.rename(columns = {'primary_insurance_policy_type':'pipt','primary_insurance_policy_relationship':'pipr','place_of_service_code':'psc'})


# ## Data Profile (EDA)

# In[127]:


cdata = klib.data_cleaning(data)
cdata.reset_index(drop=True, inplace=True)
prof = ProfileReport(cdata)
#prof.to_widgets()
prof.to_notebook_iframe()
#prof.to_file(output_file='ClaimDataProfile.html')


# ## Data Imputation

# In[130]:


data0 = data.query('is_denial == "0" ')
data0.fillna(data0.mean())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy ='most_frequent')
imputer = imputer.fit(data0)
data_with_imputed_values0  = pd.DataFrame(imputer.transform(data0))
data_with_imputed_values0.columns = data.columns


# In[ ]:


data1 = data.query('is_denial == "1" ')
data1.fillna(data1.mean())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy ='most_frequent')
imputer = imputer.fit(data1)
data_with_imputed_values1  = pd.DataFrame(imputer.transform(data1)).copy()
data_with_imputed_values1.columns = data.columns


# In[55]:


data_clean = pd.concat([data_with_imputed_values0, data_with_imputed_values1], ignore_index=True)
data_clean.shape


# In[57]:


data_clean[['unitCharge', 'units','age','diagnosis_count','modifier_count','position']] = data_clean[['unitCharge', 'units','age','diagnosis_count','modifier_count','position']].round(0).astype(int)


# In[59]:


df = klib.data_cleaning(data_clean)
df = klib.convert_datatypes(data_clean)
df.info()


# ## Preparing Data For Model

# In[61]:


mdf, data_test = train_test_split(df, stratify=data["is_denial"], test_size=0.30)


# In[71]:


X = mdf[['unitCharge', 'age', 'diagnosis_count', 'position', 'cpt_code', 'sex','itemType', 'diagnosis_code_1',
       'diagnosis_code_2', 'diagnosis_code_3', 'payer_code', 'pipt', 'pipr']]
y = mdf['is_denial']


# In[72]:


def get_categorical_indicies(X):
    cats = []
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            pass
        else:
            cats.append(col)
    cat_indicies = []
    for col in cats:
        cat_indicies.append(X.columns.get_loc(col))
    return cat_indicies

categorical_indicies = get_categorical_indicies(X)


# In[73]:


def convert_cats(X):
    cats = []
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            pass
    else:
        cats.append(col)
    cat_indicies = []
    for col in cats:
        X[col] = X[col].astype('category')
convert_cats(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

train_dataset = cb.Pool(X_train,y_train, cat_features=categorical_indicies)
test_dataset = cb.Pool(X_test,y_test, cat_features=categorical_indicies)

model = cb.CatBoostClassifier(loss_function='Logloss', eval_metric='Accuracy', verbose = False)

grid = {'learning_rate': [0.03, 0.1],'depth': [4, 6, 10],'l2_leaf_reg': [1, 3, 5,],'iterations': [50, 100, 150]}

model.grid_search(grid,train_dataset, verbose = False)


# ## Performance of the Model

# In[122]:


pred = model.predict(X_test)
print(classification_report(y_test, pred))


# In[123]:


val_data = (X_test, y_test)

def validate(model, val_data):
    y = model.predict(val_data[0])
    print('Accuracy =', accuracy_score(y, val_data[1]))
    print('ROC AUC =', roc_auc_score(y, val_data[1]))
    #print('F1 =', f1_score(y, val_data[1]))

validate(model, val_data)


# #### we got 0.88 accuracy & 0.86 on ROC AUC , precision, recall and f1-score is also reasonable, we can improve performance and accuracy more by trying different imputations algorithms, different categorical encoding algorithm, hyper parameter tuning, recursive feature selection, more feature engineering, binning age, claim amount and many more. 

# In[111]:


## saving Model on Disk
model.save_model("model")
from_file = CatBoostClassifier()
from_file.load_model("model")


# ## Analysis and Interpret Model

# In[76]:


def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
plot_feature_importance(model.get_feature_importance(),X_train.columns,'CATBOOST')


# In[68]:


shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])


# #### The above explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.
# 
# #### If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset (in the notebook this plot is interactive)

# In[77]:


# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values[0:100,:], X.iloc[0:100,:])


# ####  Catboost comes with function called calc_feature_statistics which plots the average real value and average value of our models prediction for each feature value.

# In[128]:


model.calc_feature_statistics(test_dataset,feature='unitCharge',plot=True,prediction_type='Class')


# In[41]:


model.calc_feature_statistics(test_dataset,feature='diagnosis_count',plot=True,prediction_type='Class')


# In[42]:


model.calc_feature_statistics(test_dataset,feature='age',plot=True,prediction_type='Class')


# In[44]:


model.calc_feature_statistics(test_dataset,feature='position',plot=True,prediction_type='Class')


# In[82]:


test_objects = [X.iloc[0:1], X.iloc[91:92]]

for obj in test_objects:
    print('Probability of class 1 = {:.4f}'.format(model.predict_proba(obj)[0][1]))
    print('Formula raw prediction = {:.4f}'.format(model.predict(obj, prediction_type='RawFormulaVal')[0]))
    print('\n')


# #### To get an overview of which features are most important for a model we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. The color represents the feature value (red high, blue low). This reveals for example that a high Diagnosis Counts less chances your claim will be deny.   
# 
# #### same analysis we can do for all variables age, unitCharge, position, units and modifier_count_2
# 
# 
# #### other categorical variables with high counts such as diagnosis_codes, payer_codes are encoded by catboost so hard to explain. 

# In[91]:


mdf.query('is_denial == "0" ').diagnosis_count.value_counts()


# In[90]:


mdf.query('is_denial == "1" ').diagnosis_count.value_counts()


# In[38]:


shap.summary_plot(shap_values, X)


# #### This plot is made of many dots. Each dot has three characteristics:
# 
# #### Vertical location shows what feature it is depicting
# #### Color shows whether that feature was high or low for that row of the dataset
# #### Horizontal location shows whether the effect of that value caused a higher or lower prediction.
# 

# # Converting Model To REST API (FastAPI)

# In[ ]:


# 1. Library imports
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# 2. Create the app object
app = FastAPI()

#. Load trained Pipeline
model = load_model('Final_catboost_Model_08Feb2023')

# Define predict function
@app.post('/predict')
def predict(unitCharge,units,age,diagnosis_count,modifier_count,position,using_rcm,cpt_code,sex,itemType,modifier_code_1,modifier_code_2,modifier_code_3,modifier_code_4,diagnosis_code_1,diagnosis_code_2,diagnosis_code_3,diagnosis_code_4,payer_code,pipt,pipr,psc):
    data = pd.DataFrame([[unitCharge,units,age,diagnosis_count,modifier_count,position,using_rcm,cpt_code,sex,itemType,modifier_code_1,modifier_code_2,modifier_code_3,modifier_code_4,diagnosis_code_1,diagnosis_code_2,diagnosis_code_3,diagnosis_code_4,payer_code,pipt,pipr,psc]])
    data.columns = ['unitCharge','units','age','diagnosis_count','modifier_count','position','using_rcm','cpt_code','sex','itemType','modifier_code_1','modifier_code_2','modifier_code_3','modifier_code_4','diagnosis_code_1','diagnosis_code_2','diagnosis_code_3','diagnosis_code_4','payer_code','pipt','pipr','psc']

    predictions = predict_model(model, data=data) 
    return {'prediction': int(predictions['Label'][0])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


# # Deploy Model as Rest API on AWS Lambda (ServerLess Architecture)

# In[ ]:


# Execution Role : FastApiLambdaRole
# RunTime : Python 3.8
# Layer1 (ARN) : arn:aws:lambda:us-west-2:446751924810:layer:python-3-8-scikit-learn-0-23-1:4
# Layer2 (Custom) : pythonpackage(FastAPI libs)

import json
import joblib
import re
import string
from bs4 import BeautifulSoup
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum
import tempfile
import boto3

s3_client = boto3.client("s3")


# Declaring our FastAPI instance
app = FastAPI()
lambda_handler = Mangum(app)

# Defining path operation for root endpoint
@app.get("/")
def main():
    return {"message": "Welcome to AI!"}

class request_body(BaseModel):
        unitCharge: string
        units: string
        age: string
        diagnosis_count: string
        modifier_count: string
        position: string: stringusing_rcm: string
        cpt_code: string
        sex: string
        itemType: string
        modifier_code_1: string
        modifier_code_2: string
        modifier_code_3: string
        modifier_code_4: string
        diagnosis_code_1: string
        diagnosis_code_2: string
        diagnosis_code_3: string
        diagnosis_code_4: string
        payer_code: string
        pipt: string
        pipr: string
        psc: string

@app.post("/SimulatedClaimData")
def ClaimData(data: request_body):
    
        # READ Model From S3
        with tempfile.TemporaryFile() as fp:
            s3_client.download_fileobj(
                Fileobj=fp,
                Bucket="###-ai-s3-dev",
                Key="./Final catboost Model 08Feb2023.pkl",
            )
            fp.seek(0)
            model = joblib.load(fp)
            prediction = predict_model(model, data=data_unseen)
            prediction.[0]
return {prediction.[0]}


# #### Other Models 

# #### Other Models 

# ![image.png](attachment:44d4d4cc-4830-41be-9209-21f35e033217.png)

# In[ ]:




