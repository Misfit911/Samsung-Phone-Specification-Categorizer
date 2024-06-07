<<<<<<< HEAD
# **Samsung Phone Specification Categorizer**
---
<img src="https://github.com/Misfit911/Samsung-Phone-Specification-Categorizer/assets/127237815/ee873031-85dc-4d6b-834d-2774174a038a" width="1100" height="500">

##### Author: Bradley Ouko
---

# **Overview**
---
By addressing the real-world challenge of understanding mobile phone success factors and setting an affordable price, Samsung can enhance its market position and customer satisfaction. The choice of the Spec_score_binary as the target variable aligns well with the project goals.

The evaluation results indicate strong model performance, especially with the decision tree and random forest classifiers. The perfect accuracy achieved by these models suggests that they are suitable for predicting whether a phone is high-spec or low-spec. The implications for Samsung include optimizing marketing strategies, resource allocation, and overall customer satisfaction.

# **Business Understanding**
---
# **Business Problem**
Samsung wants to launch a new product in the market. As a data scientist working for Samsung, I have been tasked to gain valuable insights from the dataset related to mobile phones. That way, it can gain an understanding of the factors that contribute to a phone’s success or failure and hence set an affordable price that will increase sales. The goal is to drive sales up📈 by offering a high spec product at an affordable price. Samsung wants to know which features will cost less to produce and have a high profit margin as well as when sold.

## **Stakeholders**
1. **Samsung Product Team:**
    * They can use the insights to enhance product features, prioritize improvements, and optimize marketing strategies.

    * **Team Goal:** Improve Samsung’s market share and customer satisfaction.

2. **Sales and Marketing Teams:**
    * They can leverage the findings to tailor advertising campaigns, target specific customer segments, and highlight key features.

    * **Team Goal:** Increase sales and revenue.

3. **Consumers and Potential Buyers:**
    * They benefit indirectly from better products and informed purchasing decisions.

    * **Team Goal:** Make informed choices when buying a phone.

### **Conclusion**
The project’s implications lie in improving Samsung’s competitiveness, understanding customer preferences, and driving innovation. By addressing this problem, Samsung is able to enhance user experiences and hence its success.✅

# **Data Understanding**
---

## **Dataset Suitability**
The dataset is suitable for the project because it contains relevant information about mobile phones, including specifications, ratings, prices, and features. It covers various aspects that impact a phone’s success in the market.

## **Feature Justification**
1. **Rating**: Captures user satisfaction, which directly affects sales and brand reputation.

2. **Spec_score**: Reflects technical specifications, influencing perceived value.

3. **Price**: Critical for consumer decisions.

4. **Company**: Brand reputation plays a significant role.

5. **Android_version**: Relevant for software compatibility and user experience.

6. **Battery, Camera, RAM, Display**: Key features affecting user satisfaction.

7. **Fast Charging**: A desirable feature.

8. **Processor**: Influences performance.

9. **External_Memory, Inbuilt_memory**: Storage options matter to users.

10. **Screen_resolution**: Affects display quality.

### **Limitations**
1. **Subjectivity**: Ratings are subjective and may not fully represent technical quality.

2. **Missing Data**: Check for missing values and handle them appropriately.

3. **Market Trends**: The dataset might not capture recent trends.

## **Project Objectives**
### **Main Objective:**
Create a machine learning model that predicts the specification rating based on the features. This can comes in handy when the stakeholders want to know which product features are ideal for their new product launch campaign.🚀

**Specific Objectives:**
1. **Feature Selection and Exploration:**
    * Explore relationships between features and the chosen target variable.
    * Identify relevant features for model training.

2. **Model Building and Evaluation:**
    * Develop and evaluate machine learning models (e.g., logistic regression, decision trees, random forests) using the selected target variable.
    * Optimize model performance using appropriate metrics (accuracy, precision, recall, etc.).

3. **Interpretability and Insights:**
    * Interpret model results to understand feature importance.
    * Provide actionable insights for Samsung’s product team.

**Target Variable**

- `Spec_score_binary` is the target variable.

- This binary variable classifies phones as either `High-Spec`(spec_score >= 70) or `Low-Spec`. It directly reflects user satisfaction and guides on product improvements and Quality Control.

*Let's dive into it!*

![dive ](https://github.com/Misfit911/Samsung-Phone-Specification-Categorizer/assets/127237815/7213b699-4bc3-4b40-b4a1-6d8421af0cd8)

*Import modules*


```python
#import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary modules for df for Modelling
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Imbalanced correction
from imblearn.over_sampling import SMOTE

# Models
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Import modeling toolkit
import mods
from mods import DataSourcing, DataPreprocessing, DataAnalysis, DataModeling

# Evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score

# Plot_roc_curves
from IPython.display import display, HTML
from sklearn.metrics import roc_curve, auc

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

```

*Load the data set.*


```python
# Call the DataSourcing() class
load = DataSourcing()
data = load.open_file('data\mobile_phone_data.csv')
load.dataframe_details(data)
```

    ==============================
    DATAFRAME SHAPE: (1370, 18)
    ============================== 
    
    
    ===============
    DATAFRAME HEAD:
    ===============
       Index                                  Name  Rating  Spec_score  \
    0      0                 Samsung Galaxy F14 5G    4.65          68   
    1      1                    Samsung Galaxy A11    4.20          63   
    2      2                    Samsung Galaxy A13    4.30          75   
    3      3                    Samsung Galaxy F23    4.10          73   
    4      4  Samsung Galaxy A03s (4GB RAM + 64GB)    4.10          69   
    
                           No_of_sim       Ram            Battery     Display  \
    0  Dual Sim, 3G, 4G, 5G, VoLTE,   4 GB RAM  6000 mAh Battery   6.6 inches   
    1      Dual Sim, 3G, 4G, VoLTE,   2 GB RAM  4000 mAh Battery   6.4 inches   
    2      Dual Sim, 3G, 4G, VoLTE,   4 GB RAM  5000 mAh Battery   6.6 inches   
    3      Dual Sim, 3G, 4G, VoLTE,   4 GB RAM   6000 mAh Battery  6.4 inches   
    4      Dual Sim, 3G, 4G, VoLTE,   4 GB RAM  5000 mAh Battery   6.5 inches   
    
                                                  Camera  \
    0    50 MP + 2 MP Dual Rear &amp; 13 MP Front Camera   
    1  13 MP + 5 MP + 2 MP Triple Rear &amp; 8 MP Fro...   
    2            50 MP Quad Rear &amp; 8 MP Front Camera   
    3           48 MP Quad Rear &amp; 13 MP Front Camera   
    4  13 MP + 2 MP + 2 MP Triple Rear &amp; 5 MP Fro...   
    
                          External_Memory Android_version   Price  company  \
    0    Memory Card Supported, upto 1 TB              13   9,999  Samsung   
    1  Memory Card Supported, upto 512 GB              10   9,990  Samsung   
    2    Memory Card Supported, upto 1 TB              12  11,999  Samsung   
    3    Memory Card Supported, upto 1 TB              12  11,999  Samsung   
    4    Memory Card Supported, upto 1 TB              11  11,999  Samsung   
    
        Inbuilt_memory       fast_charging  \
    0   128 GB inbuilt   25W Fast Charging   
    1    32 GB inbuilt   15W Fast Charging   
    2    64 GB inbuilt   25W Fast Charging   
    3    64 GB inbuilt                 NaN   
    4    64 GB inbuilt   15W Fast Charging   
    
                                   Screen_resolution             Processor  \
    0   2408 x 1080 px Display with Water Drop Notch   Octa Core Processor   
    1          720 x 1560 px Display with Punch Hole     1.8 GHz Processor   
    2   1080 x 2408 px Display with Water Drop Notch       2 GHz Processor   
    3                                  720 x 1600 px             Octa Core   
    4    720 x 1600 px Display with Water Drop Notch             Octa Core   
    
      Processor_name  
    0    Exynos 1330  
    1      Octa Core  
    2      Octa Core  
    3      Helio G88  
    4      Helio P35  
    ====================================================================== 
    
    
    ==============================
    DATAFRAME COLUMNS INFO:
    ==============================
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1370 entries, 0 to 1369
    Data columns (total 18 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Index              1370 non-null   int64  
     1   Name               1370 non-null   object 
     2   Rating             1370 non-null   float64
     3   Spec_score         1370 non-null   int64  
     4   No_of_sim          1370 non-null   object 
     5   Ram                1370 non-null   object 
     6   Battery            1370 non-null   object 
     7   Display            1370 non-null   object 
     8   Camera             1370 non-null   object 
     9   External_Memory    1370 non-null   object 
     10  Android_version    927 non-null    object 
     11  Price              1370 non-null   object 
     12  company            1370 non-null   object 
     13  Inbuilt_memory     1351 non-null   object 
     14  fast_charging      1281 non-null   object 
     15  Screen_resolution  1368 non-null   object 
     16  Processor          1342 non-null   object 
     17  Processor_name     1370 non-null   object 
    dtypes: float64(1), int64(2), object(15)
    memory usage: 192.8+ KB
    None
    ====================================================================== 
    
    
    ==============================
    DATAFRAME KEY STATISTICS:
    ==============================
                 count        mean         std    min     25%    50%      75%  \
    Index       1370.0  684.500000  395.629246   0.00  342.25  684.5  1026.75   
    Rating      1370.0    4.374416    0.230176   3.75    4.15    4.4     4.55   
    Spec_score  1370.0   80.234307    8.373922  42.00   75.00   82.0    86.00   
    
                    max  
    Index       1369.00  
    Rating         4.75  
    Spec_score    98.00  
    ====================================================================== 
    
    
    

# **Data Preparation**
---
## **Data Cleaning**

### Dealing with missing values
*Check for missing values*


```python
# Initialize data preprocessing
dp = DataPreprocessing()

# Use the methods to perform the desired data processing tasks
dp.check_null_values(data)
```

    Index                  0
    Name                   0
    Rating                 0
    Spec_score             0
    No_of_sim              0
    Ram                    0
    Battery                0
    Display                0
    Camera                 0
    External_Memory        0
    Android_version      443
    Price                  0
    company                0
    Inbuilt_memory        19
    fast_charging         89
    Screen_resolution      2
    Processor             28
    Processor_name         0
    dtype: int64
    
    Total number of null values in the data: 581
    
    ========================================
    List of columns with missing values:
    ========================================
    




    ['Android_version',
     'Inbuilt_memory',
     'fast_charging',
     'Screen_resolution',
     'Processor']



*Impute missing values*


```python
# Before imputing we first clean up the columns with missing values
# to get the values to impute with
dp.create_version_category(data)
dp.create_fast_charging_column(data)

# Impute using the mean and most_frequent starategies
dp.impute_missing_values(data)
dp.check_null_values(data)
```

    There are no null values in the data.
    

*Check for duplicates*


```python
dp.check_duplicates(data)
```

    There are 0 duplicates in the data.
    

*Feature Engineering* & *Encoding Columns*


```python
# Convert 'Rating' to binary (good/bad) based on a 
# threshold (4 stars or higher = good)
dp.create_rating_category(data)

# Convert 'Spec_score' to binary (high-spec/low-spec)
# based on a threshold (high-spec >= 70)
dp.create_spec_score_category(data)

# Convert 'Price' to binary (affordable/expensive) 
# based on a threshold (e.g., median price)
dp.create_price_category(data)

# Create the Company categories: Samsung, Other Brands
dp.create_company_category(data)
```

# **Data Analysis**
---

*Create box plots and histograms for all numeric columns*


```python
# Initialize the Data Visualization and Analysis class
eda= DataAnalysis(data)

numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = data.select_dtypes(include=['object', 'bool']).columns
numeric_col_li = data.select_dtypes(include=['int64', 'float64']).columns.to_list()
numeric_cols = [col for col in numeric_col_li if col != 'Index']
eda.plot_boxplots(data, numeric_cols)
eda.plot_histograms(data, numeric_cols)
```


    
![png](uno_files/uno_19_0.png)
    



    
![png](uno_files/uno_19_1.png)
    


*Plot count plots to visualize column categories*


```python
relevant_cols = ['fast_charging', 'Version_category', 'Company_category',
                 'rating_category', 'Spec_score_category', 'Price_category']

eda.plot_countplots(data, relevant_cols)
```


    
![png](uno_files/uno_21_0.png)
    



```python
# List of columns to exclude
exclude = ['Index', 'fast_charging', 'Version_category', 'Company_category',
            'Rating', 'Spec_score', 'Android_version_numeric', 'Name',
            'Battery', 'Display', 'Camera', 'External_Memory', 'Screen_resolution',
            'Processor_name', 'Price_binary', 'Spec_score_binary', 'Rating_binary',
            'Spec_score_category', 'rating_category','Price_category', 'Android_version_cleaned']

# Call the function with the DataFrame and the list of columns to exclude
eda.check_normal_distribution(data, exclude_columns=exclude)

```


    
![png](uno_files/uno_22_0.png)
    



    
![png](uno_files/uno_22_1.png)
    



    
![png](uno_files/uno_22_2.png)
    



    
![png](uno_files/uno_22_3.png)
    



    
![png](uno_files/uno_22_4.png)
    



    
![png](uno_files/uno_22_5.png)
    



    
![png](uno_files/uno_22_6.png)
    


*Further feature Encoding for correlation analysis purposes*


```python
# One-hot encode categorical variables
new_data = eda.encode_categorical_features(data)
```


```python
new_data.columns
```




    Index(['Index', 'Name', 'Rating', 'Spec_score', 'Display', 'Camera',
           'External_Memory', 'Android_version', 'Price', 'fast_charging',
           ...
           'Processor_ 2 GHz Processor', 'Processor_ 2.3 GHz Processor',
           'Processor_ Deca Core', 'Processor_ Deca Core Processor',
           'Processor_ Nine Core', 'Processor_ Nine Cores',
           'Processor_ Nine-Cores', 'Processor_ Octa Core',
           'Processor_ Octa Core Processor', 'Processor_ Quad Core'],
          dtype='object', length=187)



*Correlation Analysis*


```python
column_of_interest = 'Spec_score_binary'
spec_score_binary_corr = eda.correlation(new_data, column=column_of_interest, rank=False)

# Plotting the heatmap
if spec_score_binary_corr is not None:
    fig, ax = plt.subplots(figsize=(8, 15))
    sns.heatmap(spec_score_binary_corr.to_frame().sort_values(by=column_of_interest, ascending=False), annot=False, ax=ax)
    ax.set_title(f'Variables Correlating with {column_of_interest}')
    plt.show()
else:
    print(f'Correlation calculation failed for column: {column_of_interest}')
```

    The following columns were dropped due to being object types:
    ['Name', 'Display', 'Camera', 'External_Memory', 'Android_version', 'fast_charging', 'Screen_resolution', 'Processor_name', 'Android_version_cleaned', 'Version_category', 'rating_category', 'Spec_score_category', 'Price_category']
    Correlation to Spec_score_binary
    


    
![png](uno_files/uno_27_1.png)
    



```python
column_of_interest = 'Price'
price_corr = eda.correlation(new_data, column=column_of_interest, rank=False)

# Plotting the heatmap
if price_corr is not None:
    fig, ax = plt.subplots(figsize=(8, 15))
    sns.heatmap(price_corr.to_frame().sort_values(by=column_of_interest, ascending=False), annot=False, ax=ax)
    ax.set_title(f'Variables Correlating with {column_of_interest}')
    plt.show()
else:
    print(f'Correlation calculation failed for column: {column_of_interest}')
```

    The following columns were dropped due to being object types:
    ['Name', 'Display', 'Camera', 'External_Memory', 'Android_version', 'fast_charging', 'Screen_resolution', 'Processor_name', 'Android_version_cleaned', 'Version_category', 'rating_category', 'Spec_score_category', 'Price_category']
    Correlation to Price
    


    
![png](uno_files/uno_28_1.png)
    



```python
column_of_interest = 'Price_binary'
price_corr = eda.correlation(new_data, column=column_of_interest, rank=False)

# Plotting the heatmap
if price_corr is not None:
    fig, ax = plt.subplots(figsize=(8, 15))
    sns.heatmap(price_corr.to_frame().sort_values(by=column_of_interest, ascending=False), annot=False, ax=ax)
    ax.set_title(f'Variables Correlating with {column_of_interest}')
    plt.show()
else:
    print(f'Correlation calculation failed for column: {column_of_interest}')
```

    The following columns were dropped due to being object types:
    ['Name', 'Display', 'Camera', 'External_Memory', 'Android_version', 'fast_charging', 'Screen_resolution', 'Processor_name', 'Android_version_cleaned', 'Version_category', 'rating_category', 'Spec_score_category', 'Price_category']
    Correlation to Price_binary
    


    
![png](uno_files/uno_29_1.png)
    


# **Data Modelling**
---


*Drop columns for modelling purposes*


```python
# Drop columns no longer applicable in this section
object_columns = new_data.select_dtypes(include=['object'])
obj_col_list = object_columns.columns.tolist()

eda.drop_columns(new_data, obj_col_list)
```

*Model the data*
### **Base Model - Phone Specification Categorizer (High-Spec or Low-Spec)**
---

*Initialize the Data Modeling class*


```python
mod = DataModeling(new_data)
```

*Split the data into training and testing sets*


```python
# Base Model
# Split the data into features and Spec_score_binary
X_train, X_test, y_train, y_test= mod.split_data(new_data, 'Spec_score_binary')
```


```python
X_train.shape
```




    (959, 173)




```python
y_train.shape
```




    (959,)




```python
mod.modelplotting_evaluation(X_train, X_test, y_train, y_test)
```

    Model: LogisticRegression(random_state=42)
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 27   7]
     [  7 370]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       0.79      0.79      0.79        34
               1       0.98      0.98      0.98       377
    
        accuracy                           0.97       411
       macro avg       0.89      0.89      0.89       411
    weighted avg       0.97      0.97      0.97       411
    
    Accuracy: 0.9659367396593674
    Precision: 0.9814323607427056
    Recall: 0.9814323607427056
    F1 score: 0.9814323607427056
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [0.95833333 0.92708333 0.94791667 0.953125   0.96335079]
    Mean CV Accuracy: 0.9499618237347296
    
    --------------------------------------------------------------------------------
    Model: DecisionTreeClassifier(random_state=42)
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 34   0]
     [  0 377]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        34
               1       1.00      1.00      1.00       377
    
        accuracy                           1.00       411
       macro avg       1.00      1.00      1.00       411
    weighted avg       1.00      1.00      1.00       411
    
    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1 score: 1.0
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [1. 1. 1. 1. 1.]
    Mean CV Accuracy: 1.0
    
    --------------------------------------------------------------------------------
    Model: RandomForestClassifier(random_state=42)
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 34   0]
     [  0 377]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        34
               1       1.00      1.00      1.00       377
    
        accuracy                           1.00       411
       macro avg       1.00      1.00      1.00       411
    weighted avg       1.00      1.00      1.00       411
    
    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1 score: 1.0
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [1. 1. 1. 1. 1.]
    Mean CV Accuracy: 1.0
    
    --------------------------------------------------------------------------------
    Model: KNeighborsClassifier()
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 16  18]
     [ 14 363]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       0.53      0.47      0.50        34
               1       0.95      0.96      0.96       377
    
        accuracy                           0.92       411
       macro avg       0.74      0.72      0.73       411
    weighted avg       0.92      0.92      0.92       411
    
    Accuracy: 0.9221411192214112
    Precision: 0.952755905511811
    Recall: 0.9628647214854111
    F1 score: 0.9577836411609498
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [0.91145833 0.890625   0.953125   0.89583333 0.92146597]
    Mean CV Accuracy: 0.9145015270506109
    
    --------------------------------------------------------------------------------
    




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
      <th>Model</th>
      <th>Accuracy</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression(random_state=42)</td>
      <td>0.965937</td>
      <td>0.981432</td>
      <td>0.981432</td>
      <td>0.981432</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DecisionTreeClassifier(random_state=42)</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestClassifier(random_state=42)</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier()</td>
      <td>0.922141</td>
      <td>0.962865</td>
      <td>0.952756</td>
      <td>0.957784</td>
    </tr>
  </tbody>
</table>
</div>



*Apply SMOTE to deal with class imbalance*


```python
X_train_resampled, y_train_resampled = mod.apply_smote(X_train, y_train)
```


```python
mod.modelplotting_evaluation(X_train_resampled, X_test, y_train_resampled, y_test)
```

    Model: LogisticRegression(random_state=42)
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 31   3]
     [ 13 364]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       0.70      0.91      0.79        34
               1       0.99      0.97      0.98       377
    
        accuracy                           0.96       411
       macro avg       0.85      0.94      0.89       411
    weighted avg       0.97      0.96      0.96       411
    
    Accuracy: 0.9610705596107056
    Precision: 0.9918256130790191
    Recall: 0.9655172413793104
    F1 score: 0.9784946236559139
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [0.90816327 0.94557823 0.95238095 0.94880546 0.96928328]
    Mean CV Accuracy: 0.9448422372361913
    
    --------------------------------------------------------------------------------
    Model: DecisionTreeClassifier(random_state=42)
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 34   0]
     [  0 377]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        34
               1       1.00      1.00      1.00       377
    
        accuracy                           1.00       411
       macro avg       1.00      1.00      1.00       411
    weighted avg       1.00      1.00      1.00       411
    
    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1 score: 1.0
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [1. 1. 1. 1. 1.]
    Mean CV Accuracy: 1.0
    
    --------------------------------------------------------------------------------
    Model: RandomForestClassifier(random_state=42)
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 34   0]
     [  0 377]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        34
               1       1.00      1.00      1.00       377
    
        accuracy                           1.00       411
       macro avg       1.00      1.00      1.00       411
    weighted avg       1.00      1.00      1.00       411
    
    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1 score: 1.0
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [0.99319728 1.         1.         1.         1.        ]
    Mean CV Accuracy: 0.9986394557823128
    
    --------------------------------------------------------------------------------
    Model: KNeighborsClassifier()
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 23  11]
     [ 42 335]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       0.35      0.68      0.46        34
               1       0.97      0.89      0.93       377
    
        accuracy                           0.87       411
       macro avg       0.66      0.78      0.70       411
    weighted avg       0.92      0.87      0.89       411
    
    Accuracy: 0.8710462287104623
    Precision: 0.9682080924855492
    Recall: 0.8885941644562334
    F1 score: 0.9266943291839558
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [0.8537415  0.83333333 0.88095238 0.84982935 0.90443686]
    Mean CV Accuracy: 0.8644586844976899
    
    --------------------------------------------------------------------------------
    




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
      <th>Model</th>
      <th>Accuracy</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression(random_state=42)</td>
      <td>0.961071</td>
      <td>0.965517</td>
      <td>0.991826</td>
      <td>0.978495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DecisionTreeClassifier(random_state=42)</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestClassifier(random_state=42)</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier()</td>
      <td>0.871046</td>
      <td>0.888594</td>
      <td>0.968208</td>
      <td>0.926694</td>
    </tr>
  </tbody>
</table>
</div>




```python
models = [mod.lr, mod.dc, mod.rf, mod.knn]

mod.plot_roc_curves(models,X_train_resampled, y_train_resampled, X_test, y_test)
```


    
![png](uno_files/uno_44_0.png)
    


### **Model 2 - Price Categorizer (Expensive or Affordable)**
---


```python
X_train, X_test, y_train, y_test= mod.split_data(new_data, 'Price_binary')
mod.modelplotting_evaluation(X_train, X_test, y_train, y_test)
```

    Model: LogisticRegression(random_state=42)
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 92   2]
     [  5 312]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       0.95      0.98      0.96        94
               1       0.99      0.98      0.99       317
    
        accuracy                           0.98       411
       macro avg       0.97      0.98      0.98       411
    weighted avg       0.98      0.98      0.98       411
    
    Accuracy: 0.9829683698296837
    Precision: 0.9936305732484076
    Recall: 0.9842271293375394
    F1 score: 0.9889064976228209
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [0.99479167 0.99479167 0.99479167 0.98958333 0.9947644 ]
    Mean CV Accuracy: 0.9937445462478186
    
    --------------------------------------------------------------------------------
    Model: DecisionTreeClassifier(random_state=42)
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 94   0]
     [  0 317]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        94
               1       1.00      1.00      1.00       317
    
        accuracy                           1.00       411
       macro avg       1.00      1.00      1.00       411
    weighted avg       1.00      1.00      1.00       411
    
    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1 score: 1.0
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [1. 1. 1. 1. 1.]
    Mean CV Accuracy: 1.0
    
    --------------------------------------------------------------------------------
    Model: RandomForestClassifier(random_state=42)
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 94   0]
     [  0 317]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        94
               1       1.00      1.00      1.00       317
    
        accuracy                           1.00       411
       macro avg       1.00      1.00      1.00       411
    weighted avg       1.00      1.00      1.00       411
    
    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1 score: 1.0
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [1. 1. 1. 1. 1.]
    Mean CV Accuracy: 1.0
    
    --------------------------------------------------------------------------------
    Model: KNeighborsClassifier()
    --------------------------------------------------------------------------------
    Confusion matrix:
    
     [[ 93   1]
     [  1 316]]
    
    Classification report:               precision    recall  f1-score   support
    
               0       0.99      0.99      0.99        94
               1       1.00      1.00      1.00       317
    
        accuracy                           1.00       411
       macro avg       0.99      0.99      0.99       411
    weighted avg       1.00      1.00      1.00       411
    
    Accuracy: 0.9951338199513382
    Precision: 0.9968454258675079
    Recall: 0.9968454258675079
    F1 score: 0.9968454258675079
    --------------------------------------------------------------------------------
    
    Cross-Validation Scores: [0.99479167 0.99479167 0.98958333 1.         0.9947644 ]
    Mean CV Accuracy: 0.9947862129144852
    
    --------------------------------------------------------------------------------
    




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
      <th>Model</th>
      <th>Accuracy</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression(random_state=42)</td>
      <td>0.982968</td>
      <td>0.984227</td>
      <td>0.993631</td>
      <td>0.988906</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DecisionTreeClassifier(random_state=42)</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestClassifier(random_state=42)</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier()</td>
      <td>0.995134</td>
      <td>0.996845</td>
      <td>0.996845</td>
      <td>0.996845</td>
    </tr>
  </tbody>
</table>
</div>




```python
models= [mod.lr, mod.dc, mod.rf, mod.knn]

mod.plot_roc_curves(models, X_train, y_train, X_test, y_test)
```


    
![png](uno_files/uno_47_0.png)
    


# **Evaluation**
---
### **Base Model - Phone Specification Categorizer (High-Spec or Low-Spec)**
---
* The base model includes `logistic regression`, `decision tree`, `random forest`, and `k-neighbors classifiers`.

* `All models` perform well, with high accuracy and recall.

* The `decision tree` and `random forest` achieve perfect accuracy (1.0).
### **Model 2 - Price Categorizer (Expensive or Affordable)**
---
- Similar to the base model, `all classifiers` perform exceptionally well.

- `Decision tree` and `random forest` achieve perfect accuracy (1.0).

### **Metrics Justification:**
- **Accuracy**: Measures overall correctness but may not be sensitive to class imbalances.

- **Recall**: Important for identifying true positives (e.g., correctly predicting high-spec phones).

- **Precision**: Relevant for minimizing false positives (e.g., not misclassifying low-spec phones as high-spec).

- **F1 Score**: Balances precision and recall.

### **Final Model Recommendation:**
- Considering the business context, I recommend using the **Decision Tree Classifier** for both the **Phone Specification Categorizer** and **Price Categorizer**.

- It achieves perfect accuracy and is interpretable.

### **Implications:**
---
- By predicting whether a phone is high-spec or low-spec, Samsung can:

    - Optimize marketing strategies for each category.

    - Allocate resources effectively for the upcoming new product.📱

    - Enhance customer satisfaction by focusing on key features.


=======
# Samsung Phone Specification Categorizer
<p align="center">
  <img src="https://github.com/Misfit911/Samsung-Phone-Specification-Categorizer/assets/127237815/ee873031-85dc-4d6b-834d-2774174a038a" width="900" height="450">
</p>
<p align="center"></p>
This repository aims to address the challenge of understanding mobile phone success factors and setting an affordable price for the Samsung customers.
>>>>>>> 814e71056d69f47057c8fd8750e35123d2457191
