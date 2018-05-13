
---
title: "机器学习-xgboost-调参01
"
date: 2018-05-13T22:16:59+08:00
draft: True
tags: ["python","机器学习","调参"]
share: true
---

这是在使用xgboost过程中简单调参过程。

<!--more-->

```python
#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

# train = pd.read_csv(r'.\xgboost\datasource\learn1_tune_parameter\Dataset', encoding='utf-8')
train = pd.read_csv(r'D:\TEST\jupyter\xgboost\datasource\learn1_tune_parameter\Dataset\Train_nyOWmfK.csv',encoding='utf-8')

target = 'Disbursed'
IDcol = 'ID'
```

    C:\Users\M4500\Anaconda3\envs\py35_xgboost\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    C:\Users\M4500\Anaconda3\envs\py35_xgboost\lib\site-packages\sklearn\grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)
    


```python
train[predictors].info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 87020 entries, 0 to 87019
    Data columns (total 24 columns):
    Gender                   87020 non-null object
    City                     86017 non-null object
    Monthly_Income           87020 non-null int64
    DOB                      87020 non-null object
    Lead_Creation_Date       87020 non-null object
    Loan_Amount_Applied      86949 non-null float64
    Loan_Tenure_Applied      86949 non-null float64
    Existing_EMI             86949 non-null float64
    Employer_Name            86949 non-null object
    Salary_Account           75256 non-null object
    Mobile_Verified          87020 non-null object
    Var5                     87020 non-null int64
    Var1                     87020 non-null object
    Loan_Amount_Submitted    52407 non-null float64
    Loan_Tenure_Submitted    52407 non-null float64
    Interest_Rate            27726 non-null float64
    Processing_Fee           27420 non-null float64
    EMI_Loan_Submitted       27726 non-null float64
    Filled_Form              87020 non-null object
    Device_Type              87020 non-null object
    Var2                     87020 non-null object
    Source                   87020 non-null object
    Var4                     87020 non-null int64
    LoggedIn                 87020 non-null int64
    dtypes: float64(8), int64(4), object(12)
    memory usage: 15.9+ MB
    


```python
def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Gender</th>
      <th>City</th>
      <th>Monthly_Income</th>
      <th>DOB</th>
      <th>Lead_Creation_Date</th>
      <th>Loan_Amount_Applied</th>
      <th>Loan_Tenure_Applied</th>
      <th>Existing_EMI</th>
      <th>Employer_Name</th>
      <th>...</th>
      <th>Interest_Rate</th>
      <th>Processing_Fee</th>
      <th>EMI_Loan_Submitted</th>
      <th>Filled_Form</th>
      <th>Device_Type</th>
      <th>Var2</th>
      <th>Source</th>
      <th>Var4</th>
      <th>LoggedIn</th>
      <th>Disbursed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID000002C20</td>
      <td>Female</td>
      <td>Delhi</td>
      <td>20000</td>
      <td>23-May-78</td>
      <td>15-May-15</td>
      <td>300000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>CYBOSOL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID000004E40</td>
      <td>Male</td>
      <td>Mumbai</td>
      <td>35000</td>
      <td>07-Oct-85</td>
      <td>04-May-15</td>
      <td>200000.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>TATA CONSULTANCY SERVICES LTD (TCS)</td>
      <td>...</td>
      <td>13.25</td>
      <td>NaN</td>
      <td>6762.90</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID000007H20</td>
      <td>Male</td>
      <td>Panchkula</td>
      <td>22500</td>
      <td>10-Oct-81</td>
      <td>19-May-15</td>
      <td>600000.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>ALCHEMIST HOSPITALS LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S143</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID000008I30</td>
      <td>Male</td>
      <td>Saharsa</td>
      <td>35000</td>
      <td>30-Nov-87</td>
      <td>09-May-15</td>
      <td>1000000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>BIHAR GOVERNMENT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S143</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID000009J40</td>
      <td>Male</td>
      <td>Bengaluru</td>
      <td>100000</td>
      <td>17-Feb-84</td>
      <td>20-May-15</td>
      <td>500000.0</td>
      <td>2.0</td>
      <td>25000.0</td>
      <td>GLOBAL EDGE SOFTWARE</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S134</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ID000010K00</td>
      <td>Male</td>
      <td>Bengaluru</td>
      <td>45000</td>
      <td>21-Apr-82</td>
      <td>20-May-15</td>
      <td>300000.0</td>
      <td>5.0</td>
      <td>15000.0</td>
      <td>COGNIZANT TECHNOLOGY SOLUTIONS INDIA PVT LTD</td>
      <td>...</td>
      <td>13.99</td>
      <td>1500.0</td>
      <td>6978.92</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S143</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ID000011L10</td>
      <td>Female</td>
      <td>Sindhudurg</td>
      <td>70000</td>
      <td>23-Oct-87</td>
      <td>01-May-15</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>CARNIVAL CRUISE LINE</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S133</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ID000012M20</td>
      <td>Male</td>
      <td>Bengaluru</td>
      <td>20000</td>
      <td>25-Jul-75</td>
      <td>20-May-15</td>
      <td>200000.0</td>
      <td>5.0</td>
      <td>2597.0</td>
      <td>GOLDEN TULIP FLORITECH PVT. LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S159</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ID000013N30</td>
      <td>Male</td>
      <td>Kochi</td>
      <td>75000</td>
      <td>26-Jan-72</td>
      <td>02-May-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>SIIS PVT LTD</td>
      <td>...</td>
      <td>14.85</td>
      <td>26000.0</td>
      <td>30824.65</td>
      <td>Y</td>
      <td>Mobile</td>
      <td>C</td>
      <td>S122</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ID000014O40</td>
      <td>Female</td>
      <td>Mumbai</td>
      <td>30000</td>
      <td>12-Sep-89</td>
      <td>03-May-15</td>
      <td>300000.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>SOUNDCLOUD.COM</td>
      <td>...</td>
      <td>18.25</td>
      <td>1500.0</td>
      <td>10883.38</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S133</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ID000016Q10</td>
      <td>Male</td>
      <td>Mumbai</td>
      <td>25000</td>
      <td>01-Jan-76</td>
      <td>02-May-15</td>
      <td>1000000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>KRISHNA KUMAR</td>
      <td>...</td>
      <td>20.00</td>
      <td>6600.0</td>
      <td>17485.96</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S133</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ID000018S30</td>
      <td>Female</td>
      <td>Surat</td>
      <td>25000</td>
      <td>13-Oct-89</td>
      <td>02-May-15</td>
      <td>140000.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>S D JAIN MODERN SCHOOL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S122</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ID000019T40</td>
      <td>Female</td>
      <td>Pune</td>
      <td>24000</td>
      <td>22-May-90</td>
      <td>02-May-15</td>
      <td>500000.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>K.E.M. HOSPITAL RESEARCH CENTRE, PUNE</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S133</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ID000021V10</td>
      <td>Male</td>
      <td>Bhubaneswar</td>
      <td>27000</td>
      <td>24-Jun-82</td>
      <td>09-May-15</td>
      <td>200000.0</td>
      <td>5.0</td>
      <td>4600.0</td>
      <td>GI STAFFING SERVICES PVT LTD</td>
      <td>...</td>
      <td>18.00</td>
      <td>4500.0</td>
      <td>5078.69</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S133</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ID000022W20</td>
      <td>Female</td>
      <td>Howrah</td>
      <td>28000</td>
      <td>09-Feb-89</td>
      <td>13-May-15</td>
      <td>100000.0</td>
      <td>1.0</td>
      <td>1200.0</td>
      <td>MCX STOCK EXCHANGE LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S151</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ID000023X30</td>
      <td>Male</td>
      <td>Chennai</td>
      <td>42000</td>
      <td>08-May-82</td>
      <td>05-May-15</td>
      <td>500000.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>SMEC INDIA PVT LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S159</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ID000024Y40</td>
      <td>Male</td>
      <td>Ludhiana</td>
      <td>28994</td>
      <td>11-Oct-85</td>
      <td>08-May-15</td>
      <td>300000.0</td>
      <td>5.0</td>
      <td>2550.0</td>
      <td>UNIPARTS INDIA LTD</td>
      <td>...</td>
      <td>15.50</td>
      <td>6000.0</td>
      <td>7215.96</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>E</td>
      <td>S122</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ID000025Z00</td>
      <td>Female</td>
      <td>Delhi</td>
      <td>20000</td>
      <td>06-Jan-90</td>
      <td>01-May-15</td>
      <td>100000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>INTEC CAPITAL LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S122</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ID000027B20</td>
      <td>Female</td>
      <td>Bengaluru</td>
      <td>33000</td>
      <td>14-Jul-76</td>
      <td>24-May-15</td>
      <td>500000.0</td>
      <td>5.0</td>
      <td>7000.0</td>
      <td>N RAVIKUMAR</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>E</td>
      <td>S133</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ID000028C30</td>
      <td>Male</td>
      <td>Panchkula</td>
      <td>31500</td>
      <td>29-Aug-82</td>
      <td>01-May-15</td>
      <td>500000.0</td>
      <td>5.0</td>
      <td>10000.0</td>
      <td>S P SINGLA CONSTRUCTION PVT LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>E</td>
      <td>S133</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ID000029D40</td>
      <td>Male</td>
      <td>Lucknow</td>
      <td>60000</td>
      <td>14-Jul-85</td>
      <td>01-May-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>TCS AND ASSOCIATES PVT LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Mobile</td>
      <td>F</td>
      <td>S133</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ID000031F10</td>
      <td>Female</td>
      <td>Bengaluru</td>
      <td>16000</td>
      <td>01-Feb-83</td>
      <td>01-May-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>RELIANCE RETAIL LIMITED</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Mobile</td>
      <td>C</td>
      <td>S133</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ID000032G20</td>
      <td>Female</td>
      <td>Pune</td>
      <td>12000</td>
      <td>25-Jan-87</td>
      <td>01-May-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>TERNT HYPERMARKET LIMITED</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Mobile</td>
      <td>C</td>
      <td>S133</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>ID000033H30</td>
      <td>Male</td>
      <td>Bardhaman</td>
      <td>30000</td>
      <td>10-Feb-73</td>
      <td>01-May-15</td>
      <td>1000000.0</td>
      <td>5.0</td>
      <td>5000.0</td>
      <td>MD.IDRIS KHAN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>E</td>
      <td>S133</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>ID000034I40</td>
      <td>Male</td>
      <td>Indore</td>
      <td>45000</td>
      <td>12-Dec-75</td>
      <td>01-May-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>DILIP SOLANKI</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>E</td>
      <td>S133</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>ID000035J00</td>
      <td>Female</td>
      <td>Hyderabad</td>
      <td>45000</td>
      <td>11-Jan-81</td>
      <td>01-May-15</td>
      <td>300000.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>CIGNITI SOFTWARE SERVICES PVT LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S133</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>ID000037L20</td>
      <td>Male</td>
      <td>Bengaluru</td>
      <td>22843</td>
      <td>08-Jun-91</td>
      <td>01-May-15</td>
      <td>300000.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>SYNERGY BUSINESS SOLUTIONS PVT LTD</td>
      <td>...</td>
      <td>20.00</td>
      <td>2600.0</td>
      <td>13232.91</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>E</td>
      <td>S133</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ID000040O00</td>
      <td>Female</td>
      <td>Delhi</td>
      <td>2900</td>
      <td>22-Jul-82</td>
      <td>01-May-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>INVENTIV INTERNATIONAL PHARMA SERVICES P LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Mobile</td>
      <td>C</td>
      <td>S133</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>ID000041P10</td>
      <td>Female</td>
      <td>Udaipur</td>
      <td>8500</td>
      <td>20-May-94</td>
      <td>01-May-15</td>
      <td>100000.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>ARC GATE</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S133</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ID000043R30</td>
      <td>Male</td>
      <td>Mumbai</td>
      <td>200000</td>
      <td>05-Feb-85</td>
      <td>01-May-15</td>
      <td>1000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>APT BUSINESS SERVICES LLP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S159</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>86990</th>
      <td>ID124780G00</td>
      <td>Male</td>
      <td>Kochi</td>
      <td>80000</td>
      <td>16-May-73</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>CA CAMPUS</td>
      <td>...</td>
      <td>15.25</td>
      <td>28800.0</td>
      <td>40259.00</td>
      <td>Y</td>
      <td>Mobile</td>
      <td>G</td>
      <td>S122</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86991</th>
      <td>ID124782I20</td>
      <td>Male</td>
      <td>Kochi</td>
      <td>80000</td>
      <td>16-May-73</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>CA CAMPUS</td>
      <td>...</td>
      <td>23.00</td>
      <td>28800.0</td>
      <td>46154.12</td>
      <td>Y</td>
      <td>Mobile</td>
      <td>G</td>
      <td>S122</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86992</th>
      <td>ID124783J30</td>
      <td>Female</td>
      <td>Delhi</td>
      <td>25000</td>
      <td>19-Jul-92</td>
      <td>31-Jul-15</td>
      <td>80000.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>IBM CORPORATION</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86993</th>
      <td>ID124785L00</td>
      <td>Female</td>
      <td>Mumbai</td>
      <td>35000</td>
      <td>01-Dec-81</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PUNJAB AND SIND BANK (P AND SB)</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Mobile</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86994</th>
      <td>ID124786M10</td>
      <td>Male</td>
      <td>Jamshedpur</td>
      <td>15000</td>
      <td>12-Sep-84</td>
      <td>31-Jul-15</td>
      <td>200000.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>LIVELY HOOD SYSTEMS</td>
      <td>...</td>
      <td>37.00</td>
      <td>3800.0</td>
      <td>8811.27</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86995</th>
      <td>ID124787N20</td>
      <td>Male</td>
      <td>Firozpur</td>
      <td>30000</td>
      <td>27-Jul-78</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000.0</td>
      <td>CIGNA TTK HEALTH INSURANCE COMPANY LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86996</th>
      <td>ID124788O30</td>
      <td>Male</td>
      <td>Pune</td>
      <td>50000</td>
      <td>30-May-86</td>
      <td>31-Jul-15</td>
      <td>500000.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>XYZ MULTNATIONAL LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86997</th>
      <td>ID124789P40</td>
      <td>Male</td>
      <td>Delhi</td>
      <td>32000</td>
      <td>21-Oct-86</td>
      <td>31-Jul-15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86998</th>
      <td>ID124790Q00</td>
      <td>Male</td>
      <td>Ahmedabad</td>
      <td>45000</td>
      <td>31-Aug-56</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>BHARAT SANCHAR NIGAM LTD (BSNL)</td>
      <td>...</td>
      <td>14.85</td>
      <td>16200.0</td>
      <td>22481.36</td>
      <td>Y</td>
      <td>Mobile</td>
      <td>G</td>
      <td>S122</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86999</th>
      <td>ID124791R10</td>
      <td>Female</td>
      <td>Baddi</td>
      <td>13000</td>
      <td>05-Jun-90</td>
      <td>31-Jul-15</td>
      <td>200000.0</td>
      <td>2.0</td>
      <td>5000.0</td>
      <td>JOHNSON AND JOHNSON LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87000</th>
      <td>ID124792S20</td>
      <td>Female</td>
      <td>Palwal</td>
      <td>51524</td>
      <td>01-Nov-62</td>
      <td>31-Jul-15</td>
      <td>300000.0</td>
      <td>5.0</td>
      <td>23648.0</td>
      <td>DHBVN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87001</th>
      <td>ID124793T30</td>
      <td>Male</td>
      <td>Coimbatore</td>
      <td>53000</td>
      <td>29-Jun-87</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>RENAULT NISSAN TECHNOLOGY AND BUSINESS CENTRE ...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Mobile</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87002</th>
      <td>ID124794U40</td>
      <td>Female</td>
      <td>Baddi</td>
      <td>13000</td>
      <td>05-Jun-90</td>
      <td>31-Jul-15</td>
      <td>100000.0</td>
      <td>2.0</td>
      <td>5000.0</td>
      <td>JOHNSON AND JOHNSON LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87003</th>
      <td>ID124795V00</td>
      <td>Male</td>
      <td>Rajkot</td>
      <td>40000</td>
      <td>01-Jan-60</td>
      <td>31-Jul-15</td>
      <td>700000.0</td>
      <td>0.0</td>
      <td>8450.0</td>
      <td>KJO</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87004</th>
      <td>ID124796W10</td>
      <td>Male</td>
      <td>Aurangabad</td>
      <td>32000</td>
      <td>22-May-78</td>
      <td>31-Jul-15</td>
      <td>1000000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87005</th>
      <td>ID124798Y30</td>
      <td>Male</td>
      <td>Nellore</td>
      <td>116000</td>
      <td>21-Feb-69</td>
      <td>31-Jul-15</td>
      <td>1000000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>APTRANSCO</td>
      <td>...</td>
      <td>15.75</td>
      <td>15000.0</td>
      <td>36278.14</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87006</th>
      <td>ID124799Z40</td>
      <td>Female</td>
      <td>Bengaluru</td>
      <td>10000</td>
      <td>01-May-89</td>
      <td>31-Jul-15</td>
      <td>100000.0</td>
      <td>2.0</td>
      <td>3500.0</td>
      <td>FAROOQ AHMED</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87007</th>
      <td>ID124802C20</td>
      <td>Male</td>
      <td>Hyderabad</td>
      <td>15000</td>
      <td>02-May-88</td>
      <td>31-Jul-15</td>
      <td>200000.0</td>
      <td>3.0</td>
      <td>500.0</td>
      <td>FIRSTOBJECT TECHNOLOGIES LTD</td>
      <td>...</td>
      <td>31.50</td>
      <td>3800.0</td>
      <td>8222.69</td>
      <td>Y</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87008</th>
      <td>ID124803D30</td>
      <td>Female</td>
      <td>Pune</td>
      <td>35000</td>
      <td>31-Mar-81</td>
      <td>31-Jul-15</td>
      <td>1000000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>INNOBELLA MKTG AND ENTMT SOLN P L</td>
      <td>...</td>
      <td>15.25</td>
      <td>17400.0</td>
      <td>20811.58</td>
      <td>Y</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87009</th>
      <td>ID124804E40</td>
      <td>Male</td>
      <td>Delhi</td>
      <td>133000</td>
      <td>14-Aug-85</td>
      <td>31-Jul-15</td>
      <td>200000.0</td>
      <td>0.0</td>
      <td>34000.0</td>
      <td>OPERA SOLUTIONS MANAGEMENT CONSULTING SERVICES...</td>
      <td>...</td>
      <td>13.99</td>
      <td>1000.0</td>
      <td>5464.29</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87010</th>
      <td>ID124806G10</td>
      <td>Male</td>
      <td>Nagpur</td>
      <td>28000</td>
      <td>10-Jun-73</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>UTTAM VALUE STEEL LTD,WARDHA</td>
      <td>...</td>
      <td>14.85</td>
      <td>10000.0</td>
      <td>13877.39</td>
      <td>Y</td>
      <td>Mobile</td>
      <td>G</td>
      <td>S122</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87011</th>
      <td>ID124808I30</td>
      <td>Male</td>
      <td>Bengaluru</td>
      <td>15000</td>
      <td>01-Jun-90</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>AIRTEL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Mobile</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87012</th>
      <td>ID124810K00</td>
      <td>Male</td>
      <td>Bengaluru</td>
      <td>46000</td>
      <td>02-Jan-85</td>
      <td>31-Jul-15</td>
      <td>300000.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>COGNIZANT TECHNOLOGY SOLUTIONS INDIA PVT LTD</td>
      <td>...</td>
      <td>13.00</td>
      <td>2400.0</td>
      <td>10108.19</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87013</th>
      <td>ID124811L10</td>
      <td>Male</td>
      <td>Secunderabad</td>
      <td>24000</td>
      <td>01-Jan-90</td>
      <td>31-Jul-15</td>
      <td>300000.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>INDIAN AIR FORCE</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87014</th>
      <td>ID124812M20</td>
      <td>Female</td>
      <td>Pune</td>
      <td>49000</td>
      <td>31-May-82</td>
      <td>31-Jul-15</td>
      <td>400000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>INFOSYS TECHNOLOGIES</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87015</th>
      <td>ID124813N30</td>
      <td>Female</td>
      <td>Ajmer</td>
      <td>71901</td>
      <td>27-Nov-69</td>
      <td>31-Jul-15</td>
      <td>1000000.0</td>
      <td>5.0</td>
      <td>14500.0</td>
      <td>MAYO COLLEGE</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87016</th>
      <td>ID124814O40</td>
      <td>Female</td>
      <td>Kochi</td>
      <td>16000</td>
      <td>01-Dec-90</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>KERALA COMMUNICATORS CABLE LTD</td>
      <td>...</td>
      <td>35.50</td>
      <td>4800.0</td>
      <td>9425.76</td>
      <td>Y</td>
      <td>Mobile</td>
      <td>G</td>
      <td>S122</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87017</th>
      <td>ID124816Q10</td>
      <td>Male</td>
      <td>Bengaluru</td>
      <td>118000</td>
      <td>28-Jan-72</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>BANGALORE INSTITUTE OF TECHNOLOGY</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Mobile</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87018</th>
      <td>ID124818S30</td>
      <td>Male</td>
      <td>Bengaluru</td>
      <td>98930</td>
      <td>27-Apr-77</td>
      <td>31-Jul-15</td>
      <td>800000.0</td>
      <td>5.0</td>
      <td>13660.0</td>
      <td>FIRSTSOURCE SOLUTION LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87019</th>
      <td>ID124821V10</td>
      <td>Male</td>
      <td>Mumbai</td>
      <td>42300</td>
      <td>31-Oct-88</td>
      <td>31-Jul-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>GOVERNMENT OF INDIA</td>
      <td>...</td>
      <td>13.99</td>
      <td>3450.0</td>
      <td>18851.81</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>87020 rows × 26 columns</p>
</div>




```python
train.iloc[86997,:]
```




    ID                       ID124789P40
    Gender                          Male
    City                           Delhi
    Monthly_Income                 32000
    DOB                        21-Oct-86
    Lead_Creation_Date         31-Jul-15
    Loan_Amount_Applied              NaN
    Loan_Tenure_Applied              NaN
    Existing_EMI                     NaN
    Employer_Name                    NaN
    Salary_Account                   NaN
    Mobile_Verified                    Y
    Var5                               0
    Var1                            HBXX
    Loan_Amount_Submitted         500000
    Loan_Tenure_Submitted              5
    Interest_Rate                    NaN
    Processing_Fee                   NaN
    EMI_Loan_Submitted               NaN
    Filled_Form                        N
    Device_Type              Web-browser
    Var2                               G
    Source                          S122
    Var4                               1
    LoggedIn                           0
    Disbursed                          0
    Name: 86997, dtype: object




```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Gender</th>
      <th>City</th>
      <th>Monthly_Income</th>
      <th>DOB</th>
      <th>Lead_Creation_Date</th>
      <th>Loan_Amount_Applied</th>
      <th>Loan_Tenure_Applied</th>
      <th>Existing_EMI</th>
      <th>Employer_Name</th>
      <th>...</th>
      <th>Interest_Rate</th>
      <th>Processing_Fee</th>
      <th>EMI_Loan_Submitted</th>
      <th>Filled_Form</th>
      <th>Device_Type</th>
      <th>Var2</th>
      <th>Source</th>
      <th>Var4</th>
      <th>LoggedIn</th>
      <th>Disbursed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID000002C20</td>
      <td>Female</td>
      <td>Delhi</td>
      <td>20000</td>
      <td>23-May-78</td>
      <td>15-May-15</td>
      <td>300000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>CYBOSOL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID000004E40</td>
      <td>Male</td>
      <td>Mumbai</td>
      <td>35000</td>
      <td>07-Oct-85</td>
      <td>04-May-15</td>
      <td>200000.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>TATA CONSULTANCY SERVICES LTD (TCS)</td>
      <td>...</td>
      <td>13.25</td>
      <td>NaN</td>
      <td>6762.9</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>G</td>
      <td>S122</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID000007H20</td>
      <td>Male</td>
      <td>Panchkula</td>
      <td>22500</td>
      <td>10-Oct-81</td>
      <td>19-May-15</td>
      <td>600000.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>ALCHEMIST HOSPITALS LTD</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S143</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID000008I30</td>
      <td>Male</td>
      <td>Saharsa</td>
      <td>35000</td>
      <td>30-Nov-87</td>
      <td>09-May-15</td>
      <td>1000000.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>BIHAR GOVERNMENT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S143</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID000009J40</td>
      <td>Male</td>
      <td>Bengaluru</td>
      <td>100000</td>
      <td>17-Feb-84</td>
      <td>20-May-15</td>
      <td>500000.0</td>
      <td>2.0</td>
      <td>25000.0</td>
      <td>GLOBAL EDGE SOFTWARE</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>Web-browser</td>
      <td>B</td>
      <td>S134</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Monthly_Income</th>
      <th>Loan_Amount_Applied</th>
      <th>Loan_Tenure_Applied</th>
      <th>Existing_EMI</th>
      <th>Var5</th>
      <th>Loan_Amount_Submitted</th>
      <th>Loan_Tenure_Submitted</th>
      <th>Interest_Rate</th>
      <th>Processing_Fee</th>
      <th>EMI_Loan_Submitted</th>
      <th>Var4</th>
      <th>LoggedIn</th>
      <th>Disbursed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8.702000e+04</td>
      <td>8.694900e+04</td>
      <td>86949.000000</td>
      <td>8.694900e+04</td>
      <td>87020.000000</td>
      <td>5.240700e+04</td>
      <td>52407.000000</td>
      <td>27726.000000</td>
      <td>27420.000000</td>
      <td>27726.000000</td>
      <td>87020.000000</td>
      <td>87020.000000</td>
      <td>87020.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.884997e+04</td>
      <td>2.302507e+05</td>
      <td>2.131399</td>
      <td>3.696228e+03</td>
      <td>4.961503</td>
      <td>3.950106e+05</td>
      <td>3.891369</td>
      <td>19.197474</td>
      <td>5131.150839</td>
      <td>10999.528377</td>
      <td>2.949805</td>
      <td>0.029350</td>
      <td>0.014629</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.177511e+06</td>
      <td>3.542068e+05</td>
      <td>2.014193</td>
      <td>3.981021e+04</td>
      <td>5.670385</td>
      <td>3.082481e+05</td>
      <td>1.165359</td>
      <td>5.834213</td>
      <td>4725.837644</td>
      <td>7512.323050</td>
      <td>1.697720</td>
      <td>0.168785</td>
      <td>0.120062</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>5.000000e+04</td>
      <td>1.000000</td>
      <td>11.990000</td>
      <td>200.000000</td>
      <td>1176.410000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.650000e+04</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>2.000000e+05</td>
      <td>3.000000</td>
      <td>15.250000</td>
      <td>2000.000000</td>
      <td>6491.600000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.500000e+04</td>
      <td>1.000000e+05</td>
      <td>2.000000</td>
      <td>0.000000e+00</td>
      <td>2.000000</td>
      <td>3.000000e+05</td>
      <td>4.000000</td>
      <td>18.000000</td>
      <td>4000.000000</td>
      <td>9392.970000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000e+04</td>
      <td>3.000000e+05</td>
      <td>4.000000</td>
      <td>3.500000e+03</td>
      <td>11.000000</td>
      <td>5.000000e+05</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>6250.000000</td>
      <td>12919.040000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.445544e+08</td>
      <td>1.000000e+07</td>
      <td>10.000000</td>
      <td>1.000000e+07</td>
      <td>18.000000</td>
      <td>3.000000e+06</td>
      <td>6.000000</td>
      <td>37.000000</td>
      <td>50000.000000</td>
      <td>144748.280000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#问题：希望检查两个list中各名称的缺失情况
predictors = ['Gender', 'City', 'Monthly_Income', 'DOB', 'Lead_Creation_Date', 'Loan_Amount_Applied', 'Loan_Tenure_Applied', 'Existing_EMI', 'Employer_Name', 'Salary_Account', 'Mobile_Verified', 'Var5', 'Var1', 'Loan_Amount_Submitted', 'Loan_Tenure_Submitted', 'Interest_Rate', 'Processing_Fee', 'EMI_Loan_Submitted', 'Filled_Form', 'Device_Type', 'Var2', 'Source', 'Var4', 'LoggedIn']
train_s = ['ID',
 'Gender',
 'City',
 'Monthly_Income',
 'DOB',
 'Lead_Creation_Date',
 'Loan_Amount_Applied',
 'Loan_Tenure_Applied',
 'Existing_EMI',
 'Employer_Name',
 'Salary_Account',
 'Mobile_Verified',
 'Var5',
 'Var1',
 'Loan_Amount_Submitted',
 'Loan_Tenure_Submitted',
 'Interest_Rate',
 'Processing_Fee',
 'EMI_Loan_Submitted',
 'Filled_Form',
 'Device_Type',
 'Var2',
 'Source',
 'Var4',
 'LoggedIn',
 'Disbursed']
# left数据
pred_s = pd.DataFrame({"pred":list(predictors)})
# right数据
train_s =pd.DataFrame({"train":list(train.columns)})
# 根据左边pred的列'pred'，右边train的列'train'，保留right的表数据。
pd.merge(pred_s, train_s, how='right',left_on='pred',right_on='train')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred</th>
      <th>train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gender</td>
      <td>Gender</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City</td>
      <td>City</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Monthly_Income</td>
      <td>Monthly_Income</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DOB</td>
      <td>DOB</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lead_Creation_Date</td>
      <td>Lead_Creation_Date</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Loan_Amount_Applied</td>
      <td>Loan_Amount_Applied</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Loan_Tenure_Applied</td>
      <td>Loan_Tenure_Applied</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Existing_EMI</td>
      <td>Existing_EMI</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Employer_Name</td>
      <td>Employer_Name</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Salary_Account</td>
      <td>Salary_Account</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Mobile_Verified</td>
      <td>Mobile_Verified</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Var5</td>
      <td>Var5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Var1</td>
      <td>Var1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Loan_Amount_Submitted</td>
      <td>Loan_Amount_Submitted</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Loan_Tenure_Submitted</td>
      <td>Loan_Tenure_Submitted</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Interest_Rate</td>
      <td>Interest_Rate</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Processing_Fee</td>
      <td>Processing_Fee</td>
    </tr>
    <tr>
      <th>17</th>
      <td>EMI_Loan_Submitted</td>
      <td>EMI_Loan_Submitted</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Filled_Form</td>
      <td>Filled_Form</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Device_Type</td>
      <td>Device_Type</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Var2</td>
      <td>Var2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Source</td>
      <td>Source</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Var4</td>
      <td>Var4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LoggedIn</td>
      <td>LoggedIn</td>
    </tr>
    <tr>
      <th>24</th>
      <td>NaN</td>
      <td>ID</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NaN</td>
      <td>Disbursed</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(train.columns)
```




    ['ID',
     'Gender',
     'City',
     'Monthly_Income',
     'DOB',
     'Lead_Creation_Date',
     'Loan_Amount_Applied',
     'Loan_Tenure_Applied',
     'Existing_EMI',
     'Employer_Name',
     'Salary_Account',
     'Mobile_Verified',
     'Var5',
     'Var1',
     'Loan_Amount_Submitted',
     'Loan_Tenure_Submitted',
     'Interest_Rate',
     'Processing_Fee',
     'EMI_Loan_Submitted',
     'Filled_Form',
     'Device_Type',
     'Var2',
     'Source',
     'Var4',
     'LoggedIn',
     'Disbursed']




```python
help(pd.merge)
```

    Help on function merge in module pandas.core.reshape.merge:
    
    merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
        Merge DataFrame objects by performing a database-style join operation by
        columns or indexes.
        
        If joining columns on columns, the DataFrame indexes *will be
        ignored*. Otherwise if joining indexes on indexes or indexes on a column or
        columns, the index will be passed on.
        
        Parameters
        ----------
        left : DataFrame
        right : DataFrame
        how : {'left', 'right', 'outer', 'inner'}, default 'inner'
            * left: use only keys from left frame, similar to a SQL left outer join;
              preserve key order
            * right: use only keys from right frame, similar to a SQL right outer join;
              preserve key order
            * outer: use union of keys from both frames, similar to a SQL full outer
              join; sort keys lexicographically
            * inner: use intersection of keys from both frames, similar to a SQL inner
              join; preserve the order of the left keys
        on : label or list
            Field names to join on. Must be found in both DataFrames. If on is
            None and not merging on indexes, then it merges on the intersection of
            the columns by default.
        left_on : label or list, or array-like
            Field names to join on in left DataFrame. Can be a vector or list of
            vectors of the length of the DataFrame to use a particular vector as
            the join key instead of columns
        right_on : label or list, or array-like
            Field names to join on in right DataFrame or vector/list of vectors per
            left_on docs
        left_index : boolean, default False
            Use the index from the left DataFrame as the join key(s). If it is a
            MultiIndex, the number of keys in the other DataFrame (either the index
            or a number of columns) must match the number of levels
        right_index : boolean, default False
            Use the index from the right DataFrame as the join key. Same caveats as
            left_index
        sort : boolean, default False
            Sort the join keys lexicographically in the result DataFrame. If False,
            the order of the join keys depends on the join type (how keyword)
        suffixes : 2-length sequence (tuple, list, ...)
            Suffix to apply to overlapping column names in the left and right
            side, respectively
        copy : boolean, default True
            If False, do not copy data unnecessarily
        indicator : boolean or string, default False
            If True, adds a column to output DataFrame called "_merge" with
            information on the source of each row.
            If string, column with information on source of each row will be added to
            output DataFrame, and column will be named value of string.
            Information column is Categorical-type and takes on a value of "left_only"
            for observations whose merge key only appears in 'left' DataFrame,
            "right_only" for observations whose merge key only appears in 'right'
            DataFrame, and "both" if the observation's merge key is found in both.
        
            .. versionadded:: 0.17.0
        
        validate : string, default None
            If specified, checks if merge is of specified type.
        
            * "one_to_one" or "1:1": check if merge keys are unique in both
              left and right datasets.
            * "one_to_many" or "1:m": check if merge keys are unique in left
              dataset.
            * "many_to_one" or "m:1": check if merge keys are unique in right
              dataset.
            * "many_to_many" or "m:m": allowed, but does not result in checks.
        
            .. versionadded:: 0.21.0
        
        Examples
        --------
        
        >>> A              >>> B
            lkey value         rkey value
        0   foo  1         0   foo  5
        1   bar  2         1   bar  6
        2   baz  3         2   qux  7
        3   foo  4         3   bar  8
        
        >>> A.merge(B, left_on='lkey', right_on='rkey', how='outer')
           lkey  value_x  rkey  value_y
        0  foo   1        foo   5
        1  foo   4        foo   5
        2  bar   2        bar   6
        3  bar   2        bar   8
        4  baz   3        NaN   NaN
        5  NaN   NaN      qux   7
        
        Returns
        -------
        merged : DataFrame
            The output type will the be same as 'left', if it is a subclass
            of DataFrame.
        
        See also
        --------
        merge_ordered
        merge_asof
    
    


```python
a=list(predictors)
print(a+['0','0'])

```

    ['Gender', 'City', 'Monthly_Income', 'DOB', 'Lead_Creation_Date', 'Loan_Amount_Applied', 'Loan_Tenure_Applied', 'Existing_EMI', 'Employer_Name', 'Salary_Account', 'Mobile_Verified', 'Var5', 'Var1', 'Loan_Amount_Submitted', 'Loan_Tenure_Submitted', 'Interest_Rate', 'Processing_Fee', 'EMI_Loan_Submitted', 'Filled_Form', 'Device_Type', 'Var2', 'Source', 'Var4', 'LoggedIn', '0', '0']
    


```python
pd.DataFrame({"pred":list(predictors)+['0','0'],"train":list(train.columns)})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred</th>
      <th>train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gender</td>
      <td>ID</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City</td>
      <td>Gender</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Monthly_Income</td>
      <td>City</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DOB</td>
      <td>Monthly_Income</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lead_Creation_Date</td>
      <td>DOB</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Loan_Amount_Applied</td>
      <td>Lead_Creation_Date</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Loan_Tenure_Applied</td>
      <td>Loan_Amount_Applied</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Existing_EMI</td>
      <td>Loan_Tenure_Applied</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Employer_Name</td>
      <td>Existing_EMI</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Salary_Account</td>
      <td>Employer_Name</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Mobile_Verified</td>
      <td>Salary_Account</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Var5</td>
      <td>Mobile_Verified</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Var1</td>
      <td>Var5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Loan_Amount_Submitted</td>
      <td>Var1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Loan_Tenure_Submitted</td>
      <td>Loan_Amount_Submitted</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Interest_Rate</td>
      <td>Loan_Tenure_Submitted</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Processing_Fee</td>
      <td>Interest_Rate</td>
    </tr>
    <tr>
      <th>17</th>
      <td>EMI_Loan_Submitted</td>
      <td>Processing_Fee</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Filled_Form</td>
      <td>EMI_Loan_Submitted</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Device_Type</td>
      <td>Filled_Form</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Var2</td>
      <td>Device_Type</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Source</td>
      <td>Var2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Var4</td>
      <td>Source</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LoggedIn</td>
      <td>Var4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>LoggedIn</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>Disbursed</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
```


```python
modelfit(gbm0, train, predictors)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-56-97073696ff95> in <module>()
    ----> 1 modelfit(gbm0, train, predictors)
    

    <ipython-input-3-3b07bb6753d9> in modelfit(alg, dtrain, predictors, performCV, printFeatureImportance, cv_folds)
          1 def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
          2     #Fit the algorithm on the data
    ----> 3     alg.fit(dtrain[predictors], dtrain['Disbursed'])
          4 
          5     #Predict training set:
    

    ~\Anaconda3\envs\py35_xgboost\lib\site-packages\sklearn\ensemble\gradient_boosting.py in fit(self, X, y, sample_weight, monitor)
        977 
        978         # Check input
    --> 979         X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE)
        980         n_samples, self.n_features_ = X.shape
        981         if sample_weight is None:
    

    ~\Anaconda3\envs\py35_xgboost\lib\site-packages\sklearn\utils\validation.py in check_X_y(X, y, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)
        571     X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
        572                     ensure_2d, allow_nd, ensure_min_samples,
    --> 573                     ensure_min_features, warn_on_dtype, estimator)
        574     if multi_output:
        575         y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
    

    ~\Anaconda3\envs\py35_xgboost\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        431                                       force_all_finite)
        432     else:
    --> 433         array = np.array(array, dtype=dtype, order=order, copy=copy)
        434 
        435         if ensure_2d:
    

    ValueError: could not convert string to float: 'S122'


### 独热向量编码的处理


```python
# # 根据表格信息中的类型，选取objection的类型（str）转换
# train[predictors].info()
# object_name_str = 'Gender,City,DOB,Lead_Creation_Date,Employer_Name,Salary_Account,Mobile_Verified,Var1,Filled_Form,Device_Type,Var2,Source'
# object_name = object_name_str.split(",")
# train_onehot = pd.get_dummies(data=train, columns=object_name)

# 先提取部分number列
number_columnname = [i for i in train.columns if i not in object_name]
train_use = train[number_columnname]
predictors_use = [i for i in number_columnname if i not in ['ID','Disbursed']]
train_use = train_use.dropna()

```


```python
modelfit(gbm0, train_use, predictors_use)
```

    
    Model Report
    Accuracy : 0.9859
    AUC Score (Train): 0.993837
    CV Score : Mean - 0.9837965 | Std - 0.01172338 | Min - 0.9607132 | Max - 0.9919025
    


![png](/img/xgboost_tune_parameter_files/xgboost_tune_parameter_16_1.png)



```python
train[number_columnname].info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 87020 entries, 0 to 87019
    Data columns (total 14 columns):
    ID                       87020 non-null object
    Monthly_Income           87020 non-null int64
    Loan_Amount_Applied      86949 non-null float64
    Loan_Tenure_Applied      86949 non-null float64
    Existing_EMI             86949 non-null float64
    Var5                     87020 non-null int64
    Loan_Amount_Submitted    52407 non-null float64
    Loan_Tenure_Submitted    52407 non-null float64
    Interest_Rate            27726 non-null float64
    Processing_Fee           27420 non-null float64
    EMI_Loan_Submitted       27726 non-null float64
    Var4                     87020 non-null int64
    LoggedIn                 87020 non-null int64
    Disbursed                87020 non-null int64
    dtypes: float64(8), int64(5), object(1)
    memory usage: 9.3+ MB
    

### tune parameter，开始调参提高模型准确率

#### 1、Fix learning rate and number of estimators for tuning tree-based parameters


```python
param_test1 = {'n_estimators':list(range(20,81,10))}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
                                                               min_samples_split=200,
                                                               min_samples_leaf=50,
                                                               max_depth=8,
                                                               max_features='sqrt',
                                                               subsample=0.8,
                                                               random_state=10), 
                        param_grid = param_test1,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)

gsearch1.fit(train_use[predictors_use],train_use[target])
print(
"grid_scores_:", gsearch1.grid_scores_, "\n"
    "best_params_:",gsearch1.best_params_, "\n"
    "best_score_:",gsearch1.best_score_, "\n"
)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=8,
                  max_features='sqrt', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=50, min_samples_split=200,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=10, subsample=0.8, verbose=0,
                  warm_start=False),
           fit_params={}, iid=False, n_jobs=4,
           param_grid={'n_estimators': [20, 30, 40, 50, 60, 70, 80]},
           pre_dispatch='2*n_jobs', refit=True, scoring='roc_auc', verbose=0)



#### 2、Tuning tree-specific parameters

有四个参数对模型的输出结果影响最大，先考虑对它们进行调整。四个参数如下：
max_depth、num_samples_split、min_samples_leaf、max_features  
同时，这里的调整将给CV带来巨大的计算量，max_features耗费时间最长。  
思路如下：  
1、**max_depth**：选取max_depth范围，min_sample_split范围，其他按照某一个值先走（如何选取默认值？）  
2、**num_samples_split**、**min_samples_leaf**：通过1中得到确定的max_depth值，选取num_samples_split、min_samples_leaf  
3、**max_features**：通过2中，得到确定的max_depth值、num_samples_split、min_samples_leaf值  
4、通过3中，得到确定的max_depth、num_samples_split、min_samples_leaf、max_features  


```python
param_test2 = {'max_depth':list(range(5,16,2)), 'min_samples_split':list(range(100,500,100))}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
                                                               n_estimators=60,
                                                               max_features='sqrt',
                                                               subsample=0.8,
                                                               random_state=10),
                        param_grid = param_test2,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch2.fit(train_use[predictors_use],train_use[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

# print(
# "grid_scores_:", gsearch1.grid_scores_, "\n"
#     "best_params_:",gsearch1.best_params_, "\n"
#     "best_score_:",gsearch1.best_score_, "\n"
# )
```




    ([mean: 0.98534, std: 0.00818, params: {'min_samples_split': 100, 'max_depth': 5},
      mean: 0.98522, std: 0.00769, params: {'min_samples_split': 200, 'max_depth': 5},
      mean: 0.98599, std: 0.00676, params: {'min_samples_split': 300, 'max_depth': 5},
      mean: 0.98523, std: 0.00902, params: {'min_samples_split': 400, 'max_depth': 5},
      mean: 0.98515, std: 0.00876, params: {'min_samples_split': 100, 'max_depth': 7},
      mean: 0.98496, std: 0.01031, params: {'min_samples_split': 200, 'max_depth': 7},
      mean: 0.98654, std: 0.00619, params: {'min_samples_split': 300, 'max_depth': 7},
      mean: 0.98651, std: 0.00713, params: {'min_samples_split': 400, 'max_depth': 7},
      mean: 0.98746, std: 0.00463, params: {'min_samples_split': 100, 'max_depth': 9},
      mean: 0.98673, std: 0.00635, params: {'min_samples_split': 200, 'max_depth': 9},
      mean: 0.98639, std: 0.00609, params: {'min_samples_split': 300, 'max_depth': 9},
      mean: 0.98708, std: 0.00624, params: {'min_samples_split': 400, 'max_depth': 9},
      mean: 0.98594, std: 0.00570, params: {'min_samples_split': 100, 'max_depth': 11},
      mean: 0.98598, std: 0.00556, params: {'min_samples_split': 200, 'max_depth': 11},
      mean: 0.98775, std: 0.00347, params: {'min_samples_split': 300, 'max_depth': 11},
      mean: 0.98874, std: 0.00342, params: {'min_samples_split': 400, 'max_depth': 11},
      mean: 0.98664, std: 0.00620, params: {'min_samples_split': 100, 'max_depth': 13},
      mean: 0.98520, std: 0.00658, params: {'min_samples_split': 200, 'max_depth': 13},
      mean: 0.98627, std: 0.00649, params: {'min_samples_split': 300, 'max_depth': 13},
      mean: 0.98805, std: 0.00416, params: {'min_samples_split': 400, 'max_depth': 13},
      mean: 0.98705, std: 0.00483, params: {'min_samples_split': 100, 'max_depth': 15},
      mean: 0.98849, std: 0.00278, params: {'min_samples_split': 200, 'max_depth': 15},
      mean: 0.98719, std: 0.00472, params: {'min_samples_split': 300, 'max_depth': 15},
      mean: 0.98624, std: 0.00682, params: {'min_samples_split': 400, 'max_depth': 15}],
     {'max_depth': 11, 'min_samples_split': 400},
     0.9887399266152318)



选取得到最优的max_depth=11，因上面的设定值当中400是最小的，需要在>400的情况进行进一步cv。
保留max_depth=11,还可考虑将min_samples_leaf加入调试


```python
param_test3 = {'min_samples_split':list(range(100,1000,100)), 'min_samples_leaf':list(range(30,71,10))}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
                                                               n_estimators=60,
                                                               max_depth=11,
                                                               max_features='sqrt',
                                                               subsample=0.8,
                                                               random_state=10),
                        param_grid = param_test3,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch3.fit(train_use[predictors_use],train_use[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

```


```python
param_test3 = {'max_depth':list(range(5,16,2)), 'min_samples_split':list(range(100,1000,100)), 'min_samples_leaf':list(range(30,71,10))}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
                                                               n_estimators=60,
                                                               max_features='sqrt',
                                                               subsample=0.8,
                                                               random_state=10),
                        param_grid = param_test3,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch3.fit(train_use[predictors_use],train_use[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

```




    ([mean: 0.98760, std: 0.00423, params: {'min_samples_split': 100, 'max_depth': 5, 'min_samples_leaf': 30},
      mean: 0.98628, std: 0.00503, params: {'max_depth': 5, 'min_samples_split': 200, 'min_samples_leaf': 30},
      mean: 0.98357, std: 0.01142, params: {'max_depth': 5, 'min_samples_split': 300, 'min_samples_leaf': 30},
      mean: 0.98563, std: 0.00767, params: {'min_samples_split': 400, 'max_depth': 5, 'min_samples_leaf': 30},
      mean: 0.98759, std: 0.00446, params: {'min_samples_split': 500, 'max_depth': 5, 'min_samples_leaf': 30},
      mean: 0.98637, std: 0.00689, params: {'max_depth': 5, 'min_samples_split': 600, 'min_samples_leaf': 30},
      mean: 0.98785, std: 0.00471, params: {'max_depth': 5, 'min_samples_split': 700, 'min_samples_leaf': 30},
      mean: 0.98772, std: 0.00444, params: {'max_depth': 5, 'min_samples_split': 800, 'min_samples_leaf': 30},
      mean: 0.98771, std: 0.00437, params: {'max_depth': 5, 'min_samples_split': 900, 'min_samples_leaf': 30},
      mean: 0.98705, std: 0.00519, params: {'min_samples_split': 100, 'max_depth': 5, 'min_samples_leaf': 40},
      mean: 0.98546, std: 0.00632, params: {'max_depth': 5, 'min_samples_split': 200, 'min_samples_leaf': 40},
      mean: 0.98761, std: 0.00400, params: {'max_depth': 5, 'min_samples_split': 300, 'min_samples_leaf': 40},
      mean: 0.98750, std: 0.00466, params: {'min_samples_split': 400, 'max_depth': 5, 'min_samples_leaf': 40},
      mean: 0.98629, std: 0.00666, params: {'max_depth': 5, 'min_samples_split': 500, 'min_samples_leaf': 40},
      mean: 0.98672, std: 0.00605, params: {'min_samples_split': 600, 'max_depth': 5, 'min_samples_leaf': 40},
      mean: 0.98824, std: 0.00380, params: {'max_depth': 5, 'min_samples_split': 700, 'min_samples_leaf': 40},
      mean: 0.98756, std: 0.00501, params: {'max_depth': 5, 'min_samples_split': 800, 'min_samples_leaf': 40},
      mean: 0.98804, std: 0.00382, params: {'max_depth': 5, 'min_samples_split': 900, 'min_samples_leaf': 40},
      mean: 0.98713, std: 0.00549, params: {'min_samples_split': 100, 'max_depth': 5, 'min_samples_leaf': 50},
      mean: 0.98674, std: 0.00538, params: {'max_depth': 5, 'min_samples_split': 200, 'min_samples_leaf': 50},
      mean: 0.98525, std: 0.00834, params: {'min_samples_split': 300, 'max_depth': 5, 'min_samples_leaf': 50},
      mean: 0.98680, std: 0.00559, params: {'max_depth': 5, 'min_samples_split': 400, 'min_samples_leaf': 50},
      mean: 0.98576, std: 0.00762, params: {'max_depth': 5, 'min_samples_split': 500, 'min_samples_leaf': 50},
      mean: 0.98685, std: 0.00568, params: {'max_depth': 5, 'min_samples_split': 600, 'min_samples_leaf': 50},
      mean: 0.98848, std: 0.00361, params: {'min_samples_split': 700, 'max_depth': 5, 'min_samples_leaf': 50},
      mean: 0.98751, std: 0.00478, params: {'max_depth': 5, 'min_samples_split': 800, 'min_samples_leaf': 50},
      mean: 0.98799, std: 0.00386, params: {'max_depth': 5, 'min_samples_split': 900, 'min_samples_leaf': 50},
      mean: 0.98868, std: 0.00348, params: {'max_depth': 5, 'min_samples_split': 100, 'min_samples_leaf': 60},
      mean: 0.98753, std: 0.00521, params: {'max_depth': 5, 'min_samples_split': 200, 'min_samples_leaf': 60},
      mean: 0.98652, std: 0.00640, params: {'max_depth': 5, 'min_samples_split': 300, 'min_samples_leaf': 60},
      mean: 0.98729, std: 0.00484, params: {'min_samples_split': 400, 'max_depth': 5, 'min_samples_leaf': 60},
      mean: 0.98677, std: 0.00645, params: {'max_depth': 5, 'min_samples_split': 500, 'min_samples_leaf': 60},
      mean: 0.98668, std: 0.00685, params: {'max_depth': 5, 'min_samples_split': 600, 'min_samples_leaf': 60},
      mean: 0.98852, std: 0.00344, params: {'max_depth': 5, 'min_samples_split': 700, 'min_samples_leaf': 60},
      mean: 0.98822, std: 0.00374, params: {'min_samples_split': 800, 'max_depth': 5, 'min_samples_leaf': 60},
      mean: 0.98751, std: 0.00442, params: {'max_depth': 5, 'min_samples_split': 900, 'min_samples_leaf': 60},
      mean: 0.98774, std: 0.00464, params: {'max_depth': 5, 'min_samples_split': 100, 'min_samples_leaf': 70},
      mean: 0.98668, std: 0.00641, params: {'max_depth': 5, 'min_samples_split': 200, 'min_samples_leaf': 70},
      mean: 0.98568, std: 0.00780, params: {'max_depth': 5, 'min_samples_split': 300, 'min_samples_leaf': 70},
      mean: 0.98580, std: 0.00745, params: {'max_depth': 5, 'min_samples_split': 400, 'min_samples_leaf': 70},
      mean: 0.98636, std: 0.00711, params: {'min_samples_split': 500, 'max_depth': 5, 'min_samples_leaf': 70},
      mean: 0.98649, std: 0.00705, params: {'max_depth': 5, 'min_samples_split': 600, 'min_samples_leaf': 70},
      mean: 0.98836, std: 0.00343, params: {'max_depth': 5, 'min_samples_split': 700, 'min_samples_leaf': 70},
      mean: 0.98767, std: 0.00434, params: {'max_depth': 5, 'min_samples_split': 800, 'min_samples_leaf': 70},
      mean: 0.98811, std: 0.00353, params: {'min_samples_split': 900, 'max_depth': 5, 'min_samples_leaf': 70},
      mean: 0.98771, std: 0.00426, params: {'max_depth': 7, 'min_samples_split': 100, 'min_samples_leaf': 30},
      mean: 0.98597, std: 0.00747, params: {'max_depth': 7, 'min_samples_split': 200, 'min_samples_leaf': 30},
      mean: 0.98624, std: 0.00705, params: {'min_samples_split': 300, 'max_depth': 7, 'min_samples_leaf': 30},
      mean: 0.98787, std: 0.00433, params: {'max_depth': 7, 'min_samples_split': 400, 'min_samples_leaf': 30},
      mean: 0.98900, std: 0.00311, params: {'max_depth': 7, 'min_samples_split': 500, 'min_samples_leaf': 30},
      mean: 0.98744, std: 0.00539, params: {'min_samples_split': 600, 'max_depth': 7, 'min_samples_leaf': 30},
      mean: 0.98765, std: 0.00472, params: {'max_depth': 7, 'min_samples_split': 700, 'min_samples_leaf': 30},
      mean: 0.98873, std: 0.00317, params: {'min_samples_split': 800, 'max_depth': 7, 'min_samples_leaf': 30},
      mean: 0.98802, std: 0.00398, params: {'max_depth': 7, 'min_samples_split': 900, 'min_samples_leaf': 30},
      mean: 0.98710, std: 0.00409, params: {'max_depth': 7, 'min_samples_split': 100, 'min_samples_leaf': 40},
      mean: 0.98760, std: 0.00455, params: {'max_depth': 7, 'min_samples_split': 200, 'min_samples_leaf': 40},
      mean: 0.98592, std: 0.00751, params: {'max_depth': 7, 'min_samples_split': 300, 'min_samples_leaf': 40},
      mean: 0.98688, std: 0.00465, params: {'max_depth': 7, 'min_samples_split': 400, 'min_samples_leaf': 40},
      mean: 0.98790, std: 0.00484, params: {'max_depth': 7, 'min_samples_split': 500, 'min_samples_leaf': 40},
      mean: 0.98768, std: 0.00487, params: {'max_depth': 7, 'min_samples_split': 600, 'min_samples_leaf': 40},
      mean: 0.98696, std: 0.00589, params: {'max_depth': 7, 'min_samples_split': 700, 'min_samples_leaf': 40},
      mean: 0.98882, std: 0.00319, params: {'min_samples_split': 800, 'max_depth': 7, 'min_samples_leaf': 40},
      mean: 0.98815, std: 0.00383, params: {'max_depth': 7, 'min_samples_split': 900, 'min_samples_leaf': 40},
      mean: 0.98822, std: 0.00421, params: {'max_depth': 7, 'min_samples_split': 100, 'min_samples_leaf': 50},
      mean: 0.98833, std: 0.00365, params: {'max_depth': 7, 'min_samples_split': 200, 'min_samples_leaf': 50},
      mean: 0.98688, std: 0.00596, params: {'max_depth': 7, 'min_samples_split': 300, 'min_samples_leaf': 50},
      mean: 0.98604, std: 0.00704, params: {'max_depth': 7, 'min_samples_split': 400, 'min_samples_leaf': 50},
      mean: 0.98659, std: 0.00702, params: {'min_samples_split': 500, 'max_depth': 7, 'min_samples_leaf': 50},
      mean: 0.98677, std: 0.00637, params: {'max_depth': 7, 'min_samples_split': 600, 'min_samples_leaf': 50},
      mean: 0.98760, std: 0.00452, params: {'max_depth': 7, 'min_samples_split': 700, 'min_samples_leaf': 50},
      mean: 0.98875, std: 0.00339, params: {'max_depth': 7, 'min_samples_split': 800, 'min_samples_leaf': 50},
      mean: 0.98835, std: 0.00362, params: {'min_samples_split': 900, 'max_depth': 7, 'min_samples_leaf': 50},
      mean: 0.98732, std: 0.00507, params: {'max_depth': 7, 'min_samples_split': 100, 'min_samples_leaf': 60},
      mean: 0.98597, std: 0.00708, params: {'max_depth': 7, 'min_samples_split': 200, 'min_samples_leaf': 60},
      mean: 0.98706, std: 0.00486, params: {'min_samples_split': 300, 'max_depth': 7, 'min_samples_leaf': 60},
      mean: 0.98828, std: 0.00341, params: {'max_depth': 7, 'min_samples_split': 400, 'min_samples_leaf': 60},
      mean: 0.98764, std: 0.00501, params: {'max_depth': 7, 'min_samples_split': 500, 'min_samples_leaf': 60},
      mean: 0.98699, std: 0.00579, params: {'min_samples_split': 600, 'max_depth': 7, 'min_samples_leaf': 60},
      mean: 0.98779, std: 0.00463, params: {'max_depth': 7, 'min_samples_split': 700, 'min_samples_leaf': 60},
      mean: 0.98858, std: 0.00350, params: {'max_depth': 7, 'min_samples_split': 800, 'min_samples_leaf': 60},
      mean: 0.98742, std: 0.00472, params: {'min_samples_split': 900, 'max_depth': 7, 'min_samples_leaf': 60},
      mean: 0.98800, std: 0.00441, params: {'max_depth': 7, 'min_samples_split': 100, 'min_samples_leaf': 70},
      mean: 0.98815, std: 0.00380, params: {'max_depth': 7, 'min_samples_split': 200, 'min_samples_leaf': 70},
      mean: 0.98622, std: 0.00675, params: {'min_samples_split': 300, 'max_depth': 7, 'min_samples_leaf': 70},
      mean: 0.98798, std: 0.00427, params: {'max_depth': 7, 'min_samples_split': 400, 'min_samples_leaf': 70},
      mean: 0.98837, std: 0.00366, params: {'max_depth': 7, 'min_samples_split': 500, 'min_samples_leaf': 70},
      mean: 0.98816, std: 0.00388, params: {'max_depth': 7, 'min_samples_split': 600, 'min_samples_leaf': 70},
      mean: 0.98825, std: 0.00405, params: {'max_depth': 7, 'min_samples_split': 700, 'min_samples_leaf': 70},
      mean: 0.98769, std: 0.00469, params: {'max_depth': 7, 'min_samples_split': 800, 'min_samples_leaf': 70},
      mean: 0.98839, std: 0.00349, params: {'max_depth': 7, 'min_samples_split': 900, 'min_samples_leaf': 70},
      mean: 0.98778, std: 0.00395, params: {'max_depth': 9, 'min_samples_split': 100, 'min_samples_leaf': 30},
      mean: 0.98696, std: 0.00497, params: {'min_samples_split': 200, 'max_depth': 9, 'min_samples_leaf': 30},
      mean: 0.98678, std: 0.00623, params: {'max_depth': 9, 'min_samples_split': 300, 'min_samples_leaf': 30},
      mean: 0.98741, std: 0.00521, params: {'min_samples_split': 400, 'max_depth': 9, 'min_samples_leaf': 30},
      mean: 0.98612, std: 0.00741, params: {'max_depth': 9, 'min_samples_split': 500, 'min_samples_leaf': 30},
      mean: 0.98788, std: 0.00407, params: {'max_depth': 9, 'min_samples_split': 600, 'min_samples_leaf': 30},
      mean: 0.98769, std: 0.00450, params: {'min_samples_split': 700, 'max_depth': 9, 'min_samples_leaf': 30},
      mean: 0.98854, std: 0.00328, params: {'max_depth': 9, 'min_samples_split': 800, 'min_samples_leaf': 30},
      mean: 0.98835, std: 0.00369, params: {'min_samples_split': 900, 'max_depth': 9, 'min_samples_leaf': 30},
      mean: 0.98746, std: 0.00375, params: {'max_depth': 9, 'min_samples_split': 100, 'min_samples_leaf': 40},
      mean: 0.98735, std: 0.00358, params: {'max_depth': 9, 'min_samples_split': 200, 'min_samples_leaf': 40},
      mean: 0.98682, std: 0.00597, params: {'min_samples_split': 300, 'max_depth': 9, 'min_samples_leaf': 40},
      mean: 0.98869, std: 0.00369, params: {'max_depth': 9, 'min_samples_split': 400, 'min_samples_leaf': 40},
      mean: 0.98830, std: 0.00386, params: {'max_depth': 9, 'min_samples_split': 500, 'min_samples_leaf': 40},
      mean: 0.98728, std: 0.00527, params: {'min_samples_split': 600, 'max_depth': 9, 'min_samples_leaf': 40},
      mean: 0.98780, std: 0.00477, params: {'max_depth': 9, 'min_samples_split': 700, 'min_samples_leaf': 40},
      mean: 0.98776, std: 0.00425, params: {'max_depth': 9, 'min_samples_split': 800, 'min_samples_leaf': 40},
      mean: 0.98814, std: 0.00368, params: {'min_samples_split': 900, 'max_depth': 9, 'min_samples_leaf': 40},
      mean: 0.98820, std: 0.00359, params: {'max_depth': 9, 'min_samples_split': 100, 'min_samples_leaf': 50},
      mean: 0.98839, std: 0.00381, params: {'max_depth': 9, 'min_samples_split': 200, 'min_samples_leaf': 50},
      mean: 0.98784, std: 0.00402, params: {'max_depth': 9, 'min_samples_split': 300, 'min_samples_leaf': 50},
      mean: 0.98766, std: 0.00457, params: {'max_depth': 9, 'min_samples_split': 400, 'min_samples_leaf': 50},
      mean: 0.98719, std: 0.00514, params: {'max_depth': 9, 'min_samples_split': 500, 'min_samples_leaf': 50},
      mean: 0.98704, std: 0.00549, params: {'max_depth': 9, 'min_samples_split': 600, 'min_samples_leaf': 50},
      mean: 0.98784, std: 0.00462, params: {'max_depth': 9, 'min_samples_split': 700, 'min_samples_leaf': 50},
      mean: 0.98795, std: 0.00360, params: {'max_depth': 9, 'min_samples_split': 800, 'min_samples_leaf': 50},
      mean: 0.98832, std: 0.00365, params: {'min_samples_split': 900, 'max_depth': 9, 'min_samples_leaf': 50},
      mean: 0.98646, std: 0.00601, params: {'max_depth': 9, 'min_samples_split': 100, 'min_samples_leaf': 60},
      mean: 0.98705, std: 0.00540, params: {'max_depth': 9, 'min_samples_split': 200, 'min_samples_leaf': 60},
      mean: 0.98806, std: 0.00353, params: {'min_samples_split': 300, 'max_depth': 9, 'min_samples_leaf': 60},
      mean: 0.98797, std: 0.00449, params: {'max_depth': 9, 'min_samples_split': 400, 'min_samples_leaf': 60},
      mean: 0.98867, std: 0.00332, params: {'max_depth': 9, 'min_samples_split': 500, 'min_samples_leaf': 60},
      mean: 0.98740, std: 0.00466, params: {'max_depth': 9, 'min_samples_split': 600, 'min_samples_leaf': 60},
      mean: 0.98824, std: 0.00333, params: {'min_samples_split': 700, 'max_depth': 9, 'min_samples_leaf': 60},
      mean: 0.98800, std: 0.00375, params: {'max_depth': 9, 'min_samples_split': 800, 'min_samples_leaf': 60},
      mean: 0.98789, std: 0.00419, params: {'max_depth': 9, 'min_samples_split': 900, 'min_samples_leaf': 60},
      mean: 0.98785, std: 0.00413, params: {'min_samples_split': 100, 'max_depth': 9, 'min_samples_leaf': 70},
      mean: 0.98717, std: 0.00524, params: {'max_depth': 9, 'min_samples_split': 200, 'min_samples_leaf': 70},
      mean: 0.98802, std: 0.00360, params: {'max_depth': 9, 'min_samples_split': 300, 'min_samples_leaf': 70},
      mean: 0.98835, std: 0.00337, params: {'min_samples_split': 400, 'max_depth': 9, 'min_samples_leaf': 70},
      mean: 0.98815, std: 0.00389, params: {'max_depth': 9, 'min_samples_split': 500, 'min_samples_leaf': 70},
      mean: 0.98656, std: 0.00605, params: {'max_depth': 9, 'min_samples_split': 600, 'min_samples_leaf': 70},
      mean: 0.98838, std: 0.00347, params: {'max_depth': 9, 'min_samples_split': 700, 'min_samples_leaf': 70},
      mean: 0.98796, std: 0.00364, params: {'max_depth': 9, 'min_samples_split': 800, 'min_samples_leaf': 70},
      mean: 0.98807, std: 0.00443, params: {'max_depth': 9, 'min_samples_split': 900, 'min_samples_leaf': 70},
      mean: 0.98720, std: 0.00467, params: {'max_depth': 11, 'min_samples_split': 100, 'min_samples_leaf': 30},
      mean: 0.98644, std: 0.00405, params: {'max_depth': 11, 'min_samples_split': 200, 'min_samples_leaf': 30},
      mean: 0.98750, std: 0.00459, params: {'max_depth': 11, 'min_samples_split': 300, 'min_samples_leaf': 30},
      mean: 0.98742, std: 0.00407, params: {'max_depth': 11, 'min_samples_split': 400, 'min_samples_leaf': 30},
      mean: 0.98836, std: 0.00388, params: {'max_depth': 11, 'min_samples_split': 500, 'min_samples_leaf': 30},
      mean: 0.98806, std: 0.00396, params: {'min_samples_split': 600, 'max_depth': 11, 'min_samples_leaf': 30},
      mean: 0.98848, std: 0.00350, params: {'max_depth': 11, 'min_samples_split': 700, 'min_samples_leaf': 30},
      mean: 0.98790, std: 0.00408, params: {'max_depth': 11, 'min_samples_split': 800, 'min_samples_leaf': 30},
      mean: 0.98863, std: 0.00335, params: {'max_depth': 11, 'min_samples_split': 900, 'min_samples_leaf': 30},
      mean: 0.98829, std: 0.00334, params: {'max_depth': 11, 'min_samples_split': 100, 'min_samples_leaf': 40},
      mean: 0.98831, std: 0.00307, params: {'max_depth': 11, 'min_samples_split': 200, 'min_samples_leaf': 40},
      mean: 0.98695, std: 0.00612, params: {'max_depth': 11, 'min_samples_split': 300, 'min_samples_leaf': 40},
      mean: 0.98818, std: 0.00408, params: {'min_samples_split': 400, 'max_depth': 11, 'min_samples_leaf': 40},
      mean: 0.98871, std: 0.00320, params: {'max_depth': 11, 'min_samples_split': 500, 'min_samples_leaf': 40},
      mean: 0.98833, std: 0.00342, params: {'min_samples_split': 600, 'max_depth': 11, 'min_samples_leaf': 40},
      mean: 0.98779, std: 0.00450, params: {'max_depth': 11, 'min_samples_split': 700, 'min_samples_leaf': 40},
      mean: 0.98821, std: 0.00362, params: {'max_depth': 11, 'min_samples_split': 800, 'min_samples_leaf': 40},
      mean: 0.98792, std: 0.00394, params: {'max_depth': 11, 'min_samples_split': 900, 'min_samples_leaf': 40},
      mean: 0.98858, std: 0.00322, params: {'min_samples_split': 100, 'max_depth': 11, 'min_samples_leaf': 50},
      mean: 0.98706, std: 0.00487, params: {'min_samples_split': 200, 'max_depth': 11, 'min_samples_leaf': 50},
      mean: 0.98644, std: 0.00547, params: {'max_depth': 11, 'min_samples_split': 300, 'min_samples_leaf': 50},
      mean: 0.98766, std: 0.00405, params: {'max_depth': 11, 'min_samples_split': 400, 'min_samples_leaf': 50},
      mean: 0.98867, std: 0.00346, params: {'max_depth': 11, 'min_samples_split': 500, 'min_samples_leaf': 50},
      mean: 0.98816, std: 0.00360, params: {'max_depth': 11, 'min_samples_split': 600, 'min_samples_leaf': 50},
      mean: 0.98805, std: 0.00389, params: {'max_depth': 11, 'min_samples_split': 700, 'min_samples_leaf': 50},
      mean: 0.98777, std: 0.00376, params: {'min_samples_split': 800, 'max_depth': 11, 'min_samples_leaf': 50},
      mean: 0.98814, std: 0.00365, params: {'max_depth': 11, 'min_samples_split': 900, 'min_samples_leaf': 50},
      mean: 0.98784, std: 0.00347, params: {'max_depth': 11, 'min_samples_split': 100, 'min_samples_leaf': 60},
      mean: 0.98811, std: 0.00332, params: {'min_samples_split': 200, 'max_depth': 11, 'min_samples_leaf': 60},
      mean: 0.98706, std: 0.00486, params: {'max_depth': 11, 'min_samples_split': 300, 'min_samples_leaf': 60},
      mean: 0.98789, std: 0.00395, params: {'min_samples_split': 400, 'max_depth': 11, 'min_samples_leaf': 60},
      mean: 0.98836, std: 0.00368, params: {'max_depth': 11, 'min_samples_split': 500, 'min_samples_leaf': 60},
      mean: 0.98862, std: 0.00309, params: {'max_depth': 11, 'min_samples_split': 600, 'min_samples_leaf': 60},
      mean: 0.98839, std: 0.00331, params: {'max_depth': 11, 'min_samples_split': 700, 'min_samples_leaf': 60},
      mean: 0.98766, std: 0.00399, params: {'min_samples_split': 800, 'max_depth': 11, 'min_samples_leaf': 60},
      mean: 0.98795, std: 0.00357, params: {'max_depth': 11, 'min_samples_split': 900, 'min_samples_leaf': 60},
      mean: 0.98786, std: 0.00357, params: {'max_depth': 11, 'min_samples_split': 100, 'min_samples_leaf': 70},
      mean: 0.98685, std: 0.00506, params: {'min_samples_split': 200, 'max_depth': 11, 'min_samples_leaf': 70},
      mean: 0.98825, std: 0.00328, params: {'max_depth': 11, 'min_samples_split': 300, 'min_samples_leaf': 70},
      mean: 0.98764, std: 0.00473, params: {'max_depth': 11, 'min_samples_split': 400, 'min_samples_leaf': 70},
      mean: 0.98814, std: 0.00389, params: {'min_samples_split': 500, 'max_depth': 11, 'min_samples_leaf': 70},
      mean: 0.98862, std: 0.00352, params: {'max_depth': 11, 'min_samples_split': 600, 'min_samples_leaf': 70},
      mean: 0.98833, std: 0.00368, params: {'max_depth': 11, 'min_samples_split': 700, 'min_samples_leaf': 70},
      mean: 0.98810, std: 0.00389, params: {'min_samples_split': 800, 'max_depth': 11, 'min_samples_leaf': 70},
      mean: 0.98864, std: 0.00340, params: {'max_depth': 11, 'min_samples_split': 900, 'min_samples_leaf': 70},
      mean: 0.98830, std: 0.00360, params: {'min_samples_split': 100, 'max_depth': 13, 'min_samples_leaf': 30},
      mean: 0.98679, std: 0.00532, params: {'max_depth': 13, 'min_samples_split': 200, 'min_samples_leaf': 30},
      mean: 0.98875, std: 0.00306, params: {'max_depth': 13, 'min_samples_split': 300, 'min_samples_leaf': 30},
      mean: 0.98840, std: 0.00341, params: {'max_depth': 13, 'min_samples_split': 400, 'min_samples_leaf': 30},
      mean: 0.98805, std: 0.00394, params: {'max_depth': 13, 'min_samples_split': 500, 'min_samples_leaf': 30},
      mean: 0.98798, std: 0.00416, params: {'max_depth': 13, 'min_samples_split': 600, 'min_samples_leaf': 30},
      mean: 0.98743, std: 0.00508, params: {'max_depth': 13, 'min_samples_split': 700, 'min_samples_leaf': 30},
      mean: 0.98771, std: 0.00423, params: {'max_depth': 13, 'min_samples_split': 800, 'min_samples_leaf': 30},
      mean: 0.98851, std: 0.00335, params: {'max_depth': 13, 'min_samples_split': 900, 'min_samples_leaf': 30},
      mean: 0.98819, std: 0.00296, params: {'min_samples_split': 100, 'max_depth': 13, 'min_samples_leaf': 40},
      mean: 0.98773, std: 0.00370, params: {'max_depth': 13, 'min_samples_split': 200, 'min_samples_leaf': 40},
      mean: 0.98892, std: 0.00288, params: {'max_depth': 13, 'min_samples_split': 300, 'min_samples_leaf': 40},
      mean: 0.98785, std: 0.00426, params: {'max_depth': 13, 'min_samples_split': 400, 'min_samples_leaf': 40},
      mean: 0.98830, std: 0.00358, params: {'min_samples_split': 500, 'max_depth': 13, 'min_samples_leaf': 40},
      mean: 0.98839, std: 0.00383, params: {'max_depth': 13, 'min_samples_split': 600, 'min_samples_leaf': 40},
      mean: 0.98846, std: 0.00389, params: {'max_depth': 13, 'min_samples_split': 700, 'min_samples_leaf': 40},
      mean: 0.98796, std: 0.00407, params: {'min_samples_split': 800, 'max_depth': 13, 'min_samples_leaf': 40},
      mean: 0.98808, std: 0.00380, params: {'min_samples_split': 900, 'max_depth': 13, 'min_samples_leaf': 40},
      mean: 0.98847, std: 0.00343, params: {'max_depth': 13, 'min_samples_split': 100, 'min_samples_leaf': 50},
      mean: 0.98770, std: 0.00389, params: {'max_depth': 13, 'min_samples_split': 200, 'min_samples_leaf': 50},
      mean: 0.98777, std: 0.00461, params: {'max_depth': 13, 'min_samples_split': 300, 'min_samples_leaf': 50},
      mean: 0.98701, std: 0.00482, params: {'min_samples_split': 400, 'max_depth': 13, 'min_samples_leaf': 50},
      mean: 0.98723, std: 0.00568, params: {'max_depth': 13, 'min_samples_split': 500, 'min_samples_leaf': 50},
      mean: 0.98774, std: 0.00425, params: {'min_samples_split': 600, 'max_depth': 13, 'min_samples_leaf': 50},
      mean: 0.98824, std: 0.00349, params: {'max_depth': 13, 'min_samples_split': 700, 'min_samples_leaf': 50},
      mean: 0.98851, std: 0.00314, params: {'max_depth': 13, 'min_samples_split': 800, 'min_samples_leaf': 50},
      mean: 0.98777, std: 0.00382, params: {'max_depth': 13, 'min_samples_split': 900, 'min_samples_leaf': 50},
      mean: 0.98690, std: 0.00576, params: {'min_samples_split': 100, 'max_depth': 13, 'min_samples_leaf': 60},
      mean: 0.98727, std: 0.00503, params: {'max_depth': 13, 'min_samples_split': 200, 'min_samples_leaf': 60},
      mean: 0.98829, std: 0.00335, params: {'max_depth': 13, 'min_samples_split': 300, 'min_samples_leaf': 60},
      mean: 0.98767, std: 0.00366, params: {'max_depth': 13, 'min_samples_split': 400, 'min_samples_leaf': 60},
      mean: 0.98828, std: 0.00371, params: {'max_depth': 13, 'min_samples_split': 500, 'min_samples_leaf': 60},
      mean: 0.98830, std: 0.00383, params: {'min_samples_split': 600, 'max_depth': 13, 'min_samples_leaf': 60},
      mean: 0.98851, std: 0.00307, params: {'max_depth': 13, 'min_samples_split': 700, 'min_samples_leaf': 60},
      mean: 0.98788, std: 0.00369, params: {'max_depth': 13, 'min_samples_split': 800, 'min_samples_leaf': 60},
      mean: 0.98793, std: 0.00343, params: {'min_samples_split': 900, 'max_depth': 13, 'min_samples_leaf': 60},
      mean: 0.98762, std: 0.00443, params: {'min_samples_split': 100, 'max_depth': 13, 'min_samples_leaf': 70},
      mean: 0.98739, std: 0.00473, params: {'max_depth': 13, 'min_samples_split': 200, 'min_samples_leaf': 70},
      mean: 0.98708, std: 0.00498, params: {'max_depth': 13, 'min_samples_split': 300, 'min_samples_leaf': 70},
      mean: 0.98818, std: 0.00386, params: {'max_depth': 13, 'min_samples_split': 400, 'min_samples_leaf': 70},
      mean: 0.98810, std: 0.00393, params: {'min_samples_split': 500, 'max_depth': 13, 'min_samples_leaf': 70},
      mean: 0.98741, std: 0.00530, params: {'max_depth': 13, 'min_samples_split': 600, 'min_samples_leaf': 70},
      mean: 0.98872, std: 0.00292, params: {'max_depth': 13, 'min_samples_split': 700, 'min_samples_leaf': 70},
      mean: 0.98830, std: 0.00350, params: {'min_samples_split': 800, 'max_depth': 13, 'min_samples_leaf': 70},
      mean: 0.98850, std: 0.00369, params: {'max_depth': 13, 'min_samples_split': 900, 'min_samples_leaf': 70},
      mean: 0.98902, std: 0.00296, params: {'min_samples_split': 100, 'max_depth': 15, 'min_samples_leaf': 30},
      mean: 0.98821, std: 0.00327, params: {'max_depth': 15, 'min_samples_split': 200, 'min_samples_leaf': 30},
      mean: 0.98785, std: 0.00443, params: {'max_depth': 15, 'min_samples_split': 300, 'min_samples_leaf': 30},
      mean: 0.98720, std: 0.00524, params: {'min_samples_split': 400, 'max_depth': 15, 'min_samples_leaf': 30},
      mean: 0.98820, std: 0.00401, params: {'max_depth': 15, 'min_samples_split': 500, 'min_samples_leaf': 30},
      mean: 0.98814, std: 0.00374, params: {'max_depth': 15, 'min_samples_split': 600, 'min_samples_leaf': 30},
      mean: 0.98853, std: 0.00352, params: {'max_depth': 15, 'min_samples_split': 700, 'min_samples_leaf': 30},
      mean: 0.98806, std: 0.00375, params: {'min_samples_split': 800, 'max_depth': 15, 'min_samples_leaf': 30},
      mean: 0.98817, std: 0.00345, params: {'max_depth': 15, 'min_samples_split': 900, 'min_samples_leaf': 30},
      mean: 0.98731, std: 0.00439, params: {'max_depth': 15, 'min_samples_split': 100, 'min_samples_leaf': 40},
      mean: 0.98694, std: 0.00523, params: {'min_samples_split': 200, 'max_depth': 15, 'min_samples_leaf': 40},
      mean: 0.98876, std: 0.00290, params: {'max_depth': 15, 'min_samples_split': 300, 'min_samples_leaf': 40},
      mean: 0.98810, std: 0.00397, params: {'max_depth': 15, 'min_samples_split': 400, 'min_samples_leaf': 40},
      mean: 0.98853, std: 0.00396, params: {'max_depth': 15, 'min_samples_split': 500, 'min_samples_leaf': 40},
      mean: 0.98802, std: 0.00394, params: {'min_samples_split': 600, 'max_depth': 15, 'min_samples_leaf': 40},
      mean: 0.98832, std: 0.00389, params: {'max_depth': 15, 'min_samples_split': 700, 'min_samples_leaf': 40},
      mean: 0.98784, std: 0.00409, params: {'max_depth': 15, 'min_samples_split': 800, 'min_samples_leaf': 40},
      mean: 0.98853, std: 0.00308, params: {'min_samples_split': 900, 'max_depth': 15, 'min_samples_leaf': 40},
      mean: 0.98737, std: 0.00463, params: {'max_depth': 15, 'min_samples_split': 100, 'min_samples_leaf': 50},
      mean: 0.98799, std: 0.00360, params: {'max_depth': 15, 'min_samples_split': 200, 'min_samples_leaf': 50},
      mean: 0.98654, std: 0.00637, params: {'min_samples_split': 300, 'max_depth': 15, 'min_samples_leaf': 50},
      mean: 0.98691, std: 0.00548, params: {'max_depth': 15, 'min_samples_split': 400, 'min_samples_leaf': 50},
      mean: 0.98855, std: 0.00339, params: {'max_depth': 15, 'min_samples_split': 500, 'min_samples_leaf': 50},
      mean: 0.98834, std: 0.00353, params: {'max_depth': 15, 'min_samples_split': 600, 'min_samples_leaf': 50},
      mean: 0.98772, std: 0.00429, params: {'max_depth': 15, 'min_samples_split': 700, 'min_samples_leaf': 50},
      mean: 0.98848, std: 0.00331, params: {'max_depth': 15, 'min_samples_split': 800, 'min_samples_leaf': 50},
      mean: 0.98847, std: 0.00317, params: {'max_depth': 15, 'min_samples_split': 900, 'min_samples_leaf': 50},
      mean: 0.98803, std: 0.00406, params: {'min_samples_split': 100, 'max_depth': 15, 'min_samples_leaf': 60},
      mean: 0.98711, std: 0.00449, params: {'max_depth': 15, 'min_samples_split': 200, 'min_samples_leaf': 60},
      mean: 0.98830, std: 0.00365, params: {'max_depth': 15, 'min_samples_split': 300, 'min_samples_leaf': 60},
      mean: 0.98840, std: 0.00325, params: {'max_depth': 15, 'min_samples_split': 400, 'min_samples_leaf': 60},
      mean: 0.98706, std: 0.00534, params: {'min_samples_split': 500, 'max_depth': 15, 'min_samples_leaf': 60},
      mean: 0.98846, std: 0.00342, params: {'max_depth': 15, 'min_samples_split': 600, 'min_samples_leaf': 60},
      mean: 0.98858, std: 0.00314, params: {'min_samples_split': 700, 'max_depth': 15, 'min_samples_leaf': 60},
      mean: 0.98776, std: 0.00407, params: {'max_depth': 15, 'min_samples_split': 800, 'min_samples_leaf': 60},
      mean: 0.98861, std: 0.00298, params: {'max_depth': 15, 'min_samples_split': 900, 'min_samples_leaf': 60},
      mean: 0.98787, std: 0.00442, params: {'min_samples_split': 100, 'max_depth': 15, 'min_samples_leaf': 70},
      mean: 0.98738, std: 0.00513, params: {'max_depth': 15, 'min_samples_split': 200, 'min_samples_leaf': 70},
      mean: 0.98717, std: 0.00452, params: {'max_depth': 15, 'min_samples_split': 300, 'min_samples_leaf': 70},
      mean: 0.98732, std: 0.00505, params: {'min_samples_split': 400, 'max_depth': 15, 'min_samples_leaf': 70},
      mean: 0.98796, std: 0.00397, params: {'max_depth': 15, 'min_samples_split': 500, 'min_samples_leaf': 70},
      mean: 0.98816, std: 0.00368, params: {'min_samples_split': 600, 'max_depth': 15, 'min_samples_leaf': 70},
      mean: 0.98821, std: 0.00368, params: {'min_samples_split': 700, 'max_depth': 15, 'min_samples_leaf': 70},
      mean: 0.98829, std: 0.00354, params: {'max_depth': 15, 'min_samples_split': 800, 'min_samples_leaf': 70},
      mean: 0.98896, std: 0.00308, params: {'max_depth': 15, 'min_samples_split': 900, 'min_samples_leaf': 70}],
     {'max_depth': 15, 'min_samples_leaf': 30, 'min_samples_split': 100},
     0.9890228504557201)




```python
gsearch3.best_params_
```




    {'max_depth': 15, 'min_samples_leaf': 30, 'min_samples_split': 100}



从而获得最佳的参数{'max_depth': 15, 'min_samples_leaf': 30, 'min_samples_split': 100}，



```python
modelfit(gsearch3.best_estimator_, train_use, predictors_use)
```

    
    Model Report
    Accuracy : 0.9936
    AUC Score (Train): 0.998578
    CV Score : Mean - 0.9890229 | Std - 0.002963774 | Min - 0.9842912 | Max - 0.9922054
    


![png](/img/xgboost_tune_parameter_files/xgboost_tune_parameter_29_1.png)


剩下最后一个参数max_features需要调整，即


```python
param_test4 = {'max_features':list(range(5,13,2))}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
                                                               n_estimators=60,
#                                                                max_features='sqrt',
                                                               max_depth=15,
                                                               min_samples_leaf=30,
                                                               min_samples_split=100,
                                                               subsample=0.8,
                                                               random_state=10),
                        param_grid = param_test4,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch4.fit(train_use[predictors_use],train_use[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

```




    ([mean: 0.98612, std: 0.00630, params: {'max_features': 5},
      mean: 0.98685, std: 0.00537, params: {'max_features': 7},
      mean: 0.98670, std: 0.00546, params: {'max_features': 9},
      mean: 0.98354, std: 0.01104, params: {'max_features': 11}],
     {'max_features': 7},
     0.9868547163366094)



选取得到最佳的max_features=7

#### 3、ubsample and making models with lower learning rate

这一部分调试subsample、learning_rate、estimators.  
调试思路：  
1、控制其他固定，首先subsample
2、确定得到subsample,控制其他固定，调试learning_rate，相应调整estimators


```python
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}

gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
                                                               n_estimators=60,
                                                               max_features=7,
                                                               max_depth=15,
                                                               min_samples_leaf=30,
                                                               min_samples_split=100,
#                                                                subsample=0.8,
                                                               random_state=10),
                        param_grid = param_test5,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch5.fit(train_use[predictors_use],train_use[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

```




    ([mean: 0.98721, std: 0.00411, params: {'subsample': 0.6},
      mean: 0.98543, std: 0.00529, params: {'subsample': 0.7},
      mean: 0.98499, std: 0.00648, params: {'subsample': 0.75},
      mean: 0.98685, std: 0.00537, params: {'subsample': 0.8},
      mean: 0.98708, std: 0.00519, params: {'subsample': 0.85},
      mean: 0.98211, std: 0.01470, params: {'subsample': 0.9}],
     {'subsample': 0.6},
     0.9872142441445778)




```python
选取得到subsample=0.6
```


```python
param_test6 = {'learning_rate':[0.01, 0.05, 0.1]}

gsearch6 = GridSearchCV(estimator = GradientBoostingClassifier(
                                                               n_estimators=60,
                                                               max_features=7,
                                                               max_depth=15,
                                                               min_samples_leaf=30,
                                                               min_samples_split=100,
                                                               subsample=0.6,
                                                               random_state=10),
                        param_grid = param_test6,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch6.fit(train_use[predictors_use],train_use[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

```




    ([mean: 0.98540, std: 0.00935, params: {'learning_rate': 0.01},
      mean: 0.98622, std: 0.00682, params: {'learning_rate': 0.05},
      mean: 0.98721, std: 0.00411, params: {'learning_rate': 0.1}],
     {'learning_rate': 0.1},
     0.9872142441445778)




```python
选取得到learning_rate=0.1
```


```python
modelfit(gsearch6.best_estimator_, train_use, predictors_use)
```

    
    Model Report
    Accuracy : 0.9898
    AUC Score (Train): 0.996605
    CV Score : Mean - 0.9872142 | Std - 0.004112248 | Min - 0.9812374 | Max - 0.9922994
    


![png](/img/xgboost_tune_parameter_files/xgboost_tune_parameter_39_1.png)


情形一：learning_rate=0.05，n_estimators=120


```python
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05,
                                         n_estimators=120,
                                         max_depth=15,
                                         min_samples_split=100,
                                         min_samples_leaf=30,
                                         subsample=0.65,
                                         random_state=10,
                                         max_features=7)
modelfit(gbm_tuned_1, train_use, predictors_use)
```

    
    Model Report
    Accuracy : 0.9912
    AUC Score (Train): 0.997203
    CV Score : Mean - 0.9877034 | Std - 0.004135146 | Min - 0.9815896 | Max - 0.9919535
    


![png](/img/xgboost_tune_parameter_files/xgboost_tune_parameter_41_1.png)


情形二：learning_rate=0.01，n_estimators=600


```python
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.01,
                                         n_estimators=600,
                                         max_depth=15,
                                         min_samples_split=100,
                                         min_samples_leaf=30,
                                         subsample=0.65,
                                         random_state=10,
                                         max_features=7)
modelfit(gbm_tuned_1, train_use, predictors_use)
```

    
    Model Report
    Accuracy : 0.9916
    AUC Score (Train): 0.997242
    CV Score : Mean - 0.9878723 | Std - 0.004478299 | Min - 0.980673 | Max - 0.9923074
    


![png](/img/xgboost_tune_parameter_files/xgboost_tune_parameter_43_1.png)


情形三：learning_rate=0.005，n_estimators=1200


```python
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.005,
                                         n_estimators=1200,
                                         max_depth=15,
                                         min_samples_split=100,
                                         min_samples_leaf=30,
                                         subsample=0.65,
                                         random_state=10,
                                         max_features=7)
modelfit(gbm_tuned_1, train_use, predictors_use)
```

    
    Model Report
    Accuracy : 0.9914
    AUC Score (Train): 0.997147
    CV Score : Mean - 0.9884338 | Std - 0.003558152 | Min - 0.9837379 | Max - 0.9922452
    


![png](/img/xgboost_tune_parameter_files/xgboost_tune_parameter_45_1.png)



```python

```
