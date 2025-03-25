#!/usr/bin/env python
# coding: utf-8

# > ## **<center>Problem Statement:</center>**
# >   Marketing teams constantly strive to optimize their promotions, pricing,
# personalization, and campaigns to increase customer acquisition, retention, and
# revenue. However, identifying the most effective strategies can be challenging.
# Machine learning algorithms can be used to analyze past customer behavior and
# predict future outcomes based on various marketing strategies.
# 
# >The aim of this project is to develop a machine learning model that can predict whether a customer visiting an online shopping website will make a purchase or not. This prediction can help marketing teams in optimizing their promotions, pricing, personalization, and campaigns to increase the likelihood of purchase and ultimately, revenue.
# 

# In[5]:


# Assuming you want to read the file using pandas
import pandas as pd

df = pd.read_csv("online_shoppers_intention.csv")


# In[ ]:





# ## **<center> Justification and Source of Dataset :  </center>**
# 
# >The "Online Shoppers Purchasing Intention Dataset" from UCI Machine Learning
# Repository is a suitable dataset for this problem statement. This dataset contains various features related to user behavior on an online shopping website, such as the number of pages visited, the duration of the visit, and the type of traffic source. The dataset also includes a binary label indicating whether the user made a purchase or not.
# 
# >This dataset is suitable for solving this problem because it provides insights into
# various factors that influence the purchasing decision of users on an online
# shopping website. By analyzing this data, machine learning models can learn to
# identify the most effective marketing strategies for increasing the likelihood of purchase
# 
# 

# ### Importing libraries and files : ###

# In[6]:


#pip install ydata-profiling
#pip install lazypredict


# In[7]:


get_ipython().system('pip install ydata-profiling')
get_ipython().system('pip install lazypredict # This was also commented out, included it here for completeness.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


len(df.columns)


# >The decision to drop certain columns from the dataset depends on the specific analysis goals and the relevance of each column to those goals.
# 
# >In this case, I decided to drop the 'Administrative', 'Informational', and 'ProductRelated' columns because they represent the number of pages visited by the user in each of these categories, and the total number of pages visited is already captured by the 'PageValues' column. Therefore, these columns were considered redundant and not useful for the analysis.

# In[12]:


df = df.drop(['Administrative', 'Informational', 'ProductRelated'], axis=1)


# In[13]:


len(df.columns)


# >In the dataset, there are some columns with categorical variables, such as 'Month', 'VisitorType', 'OperatingSystem', and 'Browser'.
# 
# >Machine learning algorithms generally require numerical inputs, so we need to convert these categorical variables into numerical format.
# 
# >**Label encoding** and **one-hot encoding** are two techniques to achieve this conversion

# >**Label encoding** assigns a unique numerical value to each category of a variable. For example, for the 'Month' column, we can assign a numerical value of 1 for January, 2 for February, and so on. Label encoding is suitable for categorical variables that have a natural ordering, such as 'Month' and 'VisitorType'.
# 
# >One-hot encoding, on the other hand, creates a new binary column for each category of a variable.
# 

# <div class="alert alert-block alert-info">
# <b>Note:</b> Here, we will be using <b>Label encoding </b>             
# </div>

# In[14]:


from sklearn.preprocessing import LabelEncoder


# We have imported LabelEncoding above

# In[15]:


categorical_columns=['Weekend','Revenue']
for col in categorical_columns:
    encoder = LabelEncoder()
    encoder.fit(df[col])
    print('Column:', col)
    print('Original categories:', encoder.classes_)
    print('Encoded values:', encoder.transform(encoder.classes_))
    print('\n')
    df[col] = encoder.fit_transform(df[col])


# In[16]:


df['Month'] = df['Month'].map({'Feb': 2, 'Mar': 3, 'May': 5,'June':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12})


# In[17]:


df.head()


# Next, let's explore the duplicated data.

# In[18]:


df.duplicated().value_counts()  #to see the count of duplicated rows


# <div class="alert alert-block alert-info">
# <b>Note:</b> <b> False:</b> implies number of rows without any duplicates.<br>
#            <b> True:</b> implies number of rows with duplicates
# </div>

# ### Now, we will see the duplicated rows:

# In[19]:


# Use the `duplicated` function to identify duplicated rows
duplicated_rows = df[df.duplicated()]

# Print the duplicated rows
print(duplicated_rows)


# > We need to drop duplicated values from the dataset as they can affect the accuracy of the model. Duplicated values can cause bias in the data, which can lead to incorrect predictions. Hence, we will drop them before performing any analysis or building a model.

# <div class="alert alert-block alert-warning">
# We will be using the drop_duplicates() method from pandas to drop the duplicated values.
# </div>

# In[20]:


df.drop_duplicates(inplace=True)


# <div class="alert alert-block alert-info">
# <b>Note:</b>  Here, in order to make changes in the original dataframe, we have set the "inplace" parameter to "True" while dropping duplicates.
# </div>

# In[21]:


cols_to_scale = ['Administrative_Duration','Informational_Duration','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


# The MinMaxScaler is a preprocessing technique that scales all the values in a given feature to be in the range of 0 and 1. This is done to bring all the features to a common scale and avoid one feature dominating the others in a model that uses distance-based algorithms. The fit_transform() method of the scaler object fits the scaler to the data and transforms the data using the scaler in one step.

# In[31]:


df.drop_duplicates(inplace=True)

# Save cleaned data to CSV locally
df.to_csv('cleaned_label_encoded_data.csv', index=False)
print("Data saved to 'cleaned_label_encoded_data.csv'")


# # EXPLORATORY DATA ANALYSIS
#  To effectively communicate insights and patterns in data to facilitate understanding and decision-making.

# In[37]:


import pandas as pd

# Group the data by traffic type
grouped_data = df.groupby('TrafficType')

# Calculate average revenue per traffic type
average_revenue = grouped_data['Revenue'].mean()

# Calculate total revenue per traffic type
total_revenue = grouped_data['Revenue'].sum()

# Compare revenue across traffic types
revenue_comparison = pd.DataFrame({'Average Revenue': average_revenue, 'Total Revenue': total_revenue})

# Print the revenue comparison
print(revenue_comparison)


# In[33]:


# Sort the revenue comparison dataframe in descending order based on the revenue metric
revenue_comparison.sort_values(by='Total Revenue', ascending=False, inplace=True)

# Visualize the sorted data
plt.figure(figsize=(10, 6))
sns.barplot(data=revenue_comparison, x=df['TrafficType'], y='Total Revenue')
plt.title('Total Revenue by Traffic Type')
plt.xlabel('Traffic Type')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()



# Traffic Type 2 and Traffic Type 3 generate the highest total revenue, indicating that these two traffic sources are driving the <b> most valuable </b>  traffic to the website.<br>
# It is important to focus on optimizing and maximizing the traffic from Traffic Type 2 and Traffic Type 3, as they have shown to be the most valuable sources of revenue for the website

# In[34]:


# Filter the dataset for Traffic Type 2 visitors
traffic_type_2_data = df[df['TrafficType'] == 2]

# Filter the dataset for Traffic Type 3 visitors
traffic_type_3_data = df[df['TrafficType'] == 3]

# Demographic analysis
demographic_variables = ['VisitorType']

for variable in demographic_variables:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=traffic_type_2_data, x=variable, palette='viridis')
    plt.title(f'{variable} Distribution for Traffic Type 2 Visitors')
    plt.xlabel(variable)
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=traffic_type_3_data, x=variable, palette='viridis')
    plt.title(f'{variable} Distribution for Traffic Type 3 Visitors')
    plt.xlabel(variable)
    plt.ylabel('Count')
    plt.show()

# Behavioral analysis
behavioral_variables = ['PageValues', 'BounceRates', 'ExitRates']

for variable in behavioral_variables:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=traffic_type_2_data, x=variable, palette='viridis')
    plt.title(f'{variable} Distribution for Traffic Type 2 Visitors')
    plt.xlabel(variable)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=traffic_type_3_data, x=variable, palette='viridis')
    plt.title(f'{variable} Distribution for Traffic Type 3 Visitors')
    plt.xlabel(variable)
    plt.show()


# From the demographic analysis:
# 
# - For Traffic Type 2 visitors,Visitors from Traffic Type 2 are primarily Returning Visitors.
# 
# - For Traffic Type 3 visitors,Visitors from Traffic Type 3 are also primarily Returning Visitors.
# 
# From the behavioral analysis:
# 
# - The average page values for Traffic Type 2 visitors are higher compared to Traffic Type 3 visitors, indicating that visitors from Traffic Type 2 are more likely to generate revenue on the website.
# 
# -The bounce rates and exit rates for Traffic Type 2 visitors are relatively low compared to other traffic types. This suggests that visitors coming from Traffic Type 2 have a higher level of engagement and are more likely to explore multiple pages before leaving the website.
# This makes them a valuable traffic source for revenue generation.
# 
# -The bounce rates and exit rates for Traffic Type 3 visitors are relatively high compared to other traffic types.
# This suggests that visitors coming from Traffic Type 3 may have a lower engagement level with the website, leading to a higher likelihood of leaving the website without further interaction..
# 
# 
# 
# 
# 
# 

# In[35]:


# Group the data by 'SpecialDay' and calculate the average revenue or visitor count
special_day_analysis = df.groupby('SpecialDay')['Revenue'].mean()  # Replace 'Revenue' with the appropriate metric

# Sort the data in descending order based on the average revenue or visitor count
special_day_analysis = special_day_analysis.sort_values(ascending=False)

# Visualize the impact of special days on customer engagement
plt.figure(figsize=(10, 6))
sns.barplot(x=special_day_analysis.index, y=special_day_analysis.values)
plt.title("Impact of Special Days on Customer Engagement")
plt.xlabel("Special Day")
plt.ylabel("Average Revenue" )
plt.xticks(rotation=45)
plt.show()

# Identify the special days with the highest impact on customer engagement
top_special_days = special_day_analysis.head(3)  # Replace '3' with the desired number of top special days

print("Special Days with the Highest Impact on Customer Engagement:")
for day, impact in top_special_days.items():
    print(f"- {day}: {impact}")


# In[36]:


corr=df.corr()
sns.set(style='white')
plt.figure(figsize=(8, 6))
sns.heatmap(corr,vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), linewidths=0.5)


# From the above correlation matrix of the online shopper dataset, we can observe that:
# 
# The 'ExitRates' and 'BounceRates' features are moderately correlated, which makes sense as both are related to the visitor leaving the website.
# 
# The 'PageValues' feature is weakly correlated with the other features, which suggests that it may not have a strong impact on predicting whether a visitor will make a purchase or not.

# In[ ]:


plt.figure(figsize=(9, 3))
plt.hist(df['Revenue'],color='navy')

plt.title('Revenue Class Distribution')
plt.xlabel('Revenue')
plt.ylabel('Count')
plt.show()


# The above visualization helps to understand the distribution of the target variable, which is the revenue class. In the case of a binary classification problem like this one, it is important to have a balanced distribution of the classes. We can observe that there is a class imbalance here which could lead to a biased model that performs poorly on the minority class.

# In[ ]:


N=len(df)
colors = np.random.rand(N)
plt.scatter(df['PageValues'], df['BounceRates'],c=colors)
plt.title('Page Values vs. Bounce Rates')
plt.xlabel('Page Values')
plt.ylabel('Bounce Rates')
plt.show()


# We can observe that there is a general trend where higher page values tend to have lower bounce rates. This could indicate that users are more likely to stay on a website if the page provides them with more valuable information or products. However, there are also many data points with low page values and low bounce rates, suggesting that there may be other factors at play as well.

# # Handling Class Imbalance

# In[ ]:


df.head()


# In[ ]:


df['Revenue'].value_counts()


# In[ ]:


X=df.drop('Revenue',axis=1)
y=df['Revenue']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# In[ ]:


y_train.value_counts()


# In[ ]:


y.value_counts()


# In[ ]:


y_test.value_counts()


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_train[:10]


#  We'll be using *SMOTE*  technique to handle the class imbalance

# In[ ]:


from imblearn.over_sampling import SMOTE

# Perform one-hot encoding on the categorical features
X_encoded = pd.get_dummies(X)

# Apply SMOTE on the encoded features and target variable
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X_encoded, y)

# Convert the resampled target variable to a pandas Series
y_sm = pd.Series(y_sm)



# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)


# In[ ]:


# Number of classes in training Data
y_train.value_counts()


# # Comparing Machine learning models

# In[ ]:


import pandas as pd
from lazypredict.Supervised import LazyClassifier


# In[ ]:


get_ipython().system('pip install lazypredict')


# In[ ]:


clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)


# In[ ]:


models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)



# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf = RandomForestClassifier(n_estimators=1000, random_state=1)
rf.fit(X_train, y_train)

# Make predictions on the test set and evaluate model performance
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification


# In[ ]:


et = ExtraTreesClassifier(n_estimators=100, random_state=42)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
accuracy_et = accuracy_score(y_test, y_pred)
print('Extra Trees Accuracy:', accuracy_et)

print(f'Accuracy: {accuracy_et}')
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# We choose the ExtraTrees Classfier predictive model as it provides maximum accuracy ( approx 94%).

# # Time to Test!

# In[ ]:


df.columns


# We will take inputs from user , and predict whether the person will buy or not.
# 

# In[ ]:


import pandas as pd
import numpy as np

# Define user input as a dictionary
user_input = {'Administrative_Duration': 50,
              'Informational_Duration': 100,
              'ProductRelated_Duration': 200,
              'BounceRates': 0.05,
              'ExitRates': 0.1,
              'PageValues': 20,
              'SpecialDay': 0,
              'Month': 7,
              'OperatingSystems': 0,
              'Browser': 0,
              'Region': 0,
              'TrafficType': 1,
              'VisitorType': 'New_Visitor',
              'Weekend': 1}

# Create a DataFrame from the user input dictionary
user_df = pd.DataFrame.from_dict(user_input, orient='index').T

# Map VisitorType to binary columns
visitor_type_mapping = {
    'New_Visitor': 1,
    'Other': 0,
    'Returning_Visitor': 0
}
user_df['VisitorType_New_Visitor'] = user_df['VisitorType'].map(visitor_type_mapping)
user_df['VisitorType_Other'] = user_df['VisitorType'].map(visitor_type_mapping)
user_df['VisitorType_Returning_Visitor'] = user_df['VisitorType'].map(visitor_type_mapping)

# Drop the original VisitorType column
user_df.drop('VisitorType', axis=1, inplace=True)

# Make a prediction for the user input
prediction = et.predict(user_df)
print(prediction)




# In[ ]:


import pandas as pd
import numpy as np

# Define user input as a dictionary
user_input = {'Administrative_Duration': 50,
              'Informational_Duration': 100,
              'ProductRelated_Duration': 200,
              'BounceRates': 0.05,
              'ExitRates': 0.1,
              'PageValues': 20,
              'SpecialDay': 0,
              'Month': 7,
              'OperatingSystems': 0,
              'Browser': 0,
              'Region': 0,
              'TrafficType': 1,
              'VisitorType': 'New_Visitor',
              'Weekend': 1}

# Create a DataFrame from the user input dictionary
user_df = pd.DataFrame.from_dict(user_input, orient='index').T

# Map VisitorType to binary columns and create new columns for each category
visitor_type_mapping = {
    'New_Visitor': 1,
    'Other': 0,
    'Returning_Visitor': 0
}
user_df['VisitorType_New_Visitor'] = user_df['VisitorType'].map(visitor_type_mapping).astype(int)
user_df['VisitorType_Other'] = 0  # Setting to 0 as it's not a New_Visitor
user_df['VisitorType_Returning_Visitor'] = 0  # Setting to 0 as it's not a New_Visitor


# Instead of mapping, directly assign based on user input
# if user_input['VisitorType'] == 'New_Visitor':
#     user_df['VisitorType_New_Visitor'] = 1
#     user_df['VisitorType_Other'] = 0
#     user_df['VisitorType_Returning_Visitor'] = 0
# elif user_input['VisitorType'] == 'Other':
#     user_df['VisitorType_New_Visitor'] = 0
#     user_df['VisitorType_Other'] = 1
#     user_df['VisitorType_Returning_Visitor'] = 0
# elif user_input['VisitorType'] == 'Returning_Visitor':
#     user_df['VisitorType_New_Visitor'] = 0
#     user_df['VisitorType_Other'] = 0
#     user_df['VisitorType_Returning_Visitor'] = 1

# Drop the original VisitorType column
user_df.drop('VisitorType', axis=1, inplace=True)

# Make a prediction for the user input
prediction = et.predict(user_df)
print(prediction)


# In[ ]:


import pandas as pd
import numpy as np

# Define user input as a dictionary
user_input = {'Administrative_Duration': 50,
              'Informational_Duration': 100,
              'ProductRelated_Duration': 200,
              'BounceRates': 0.05,
              'ExitRates': 0.1,
              'PageValues': 20,
              'SpecialDay': 0,
              'Month': 7,
              'OperatingSystems': 0,
              'Browser': 0,
              'Region': 0,
              'TrafficType': 1,
              'VisitorType': 'New_Visitor',
              'Weekend': 1}

# Create a DataFrame from the user input dictionary
user_df = pd.DataFrame.from_dict(user_input, orient='index').T

# Perform one-hot encoding for 'VisitorType' to match training data
# Get all unique values of 'VisitorType' from the original dataframe (df)
all_visitor_types = df['VisitorType'].unique()

# Create new columns for each visitor type and initialize to 0
for visitor_type in all_visitor_types:
    user_df[f'VisitorType_{visitor_type}'] = 0

# Set the appropriate visitor type column to 1 based on user input
user_df[f'VisitorType_{user_input["VisitorType"]}'] = 1

# Drop the original 'VisitorType' column
user_df.drop('VisitorType', axis=1, inplace=True)


# Make a prediction for the user input
prediction = et.predict(user_df)
print(prediction)


# In[ ]:


import pandas as pd
import numpy as np

# Define user input as a dictionary
user_input = {'Administrative_Duration': 50,
              'Informational_Duration': 100,
              'ProductRelated_Duration': 200,
              'BounceRates': 0.05,
              'ExitRates': 0.1,
              'PageValues': 20,
              'SpecialDay': 0,
              'Month': 7,
              'OperatingSystems': 0,
              'Browser': 0,
              'Region': 0,
              'TrafficType': 1,
              'VisitorType': 'New_Visitor',
              'Weekend': 1}

# Create a DataFrame from the user input dictionary
user_df = pd.DataFrame.from_dict(user_input, orient='index').T

# Perform one-hot encoding for 'VisitorType' to match training data
# Get all unique values of 'VisitorType' from the original dataframe (df)
# Assuming 'VisitorType' was one-hot encoded during training
# Get the one-hot encoded column names from the training data (X_train)
visitor_type_columns = [col for col in X_train.columns if col.startswith('VisitorType_')]

# Create new columns for each visitor type in user_df and initialize to 0
for col in visitor_type_columns:
    user_df[col] = 0

# Set the appropriate visitor type column to 1 based on user input
user_df[f'VisitorType_{user_input["VisitorType"]}'] = 1

# Drop the original 'VisitorType' column
user_df.drop('VisitorType', axis=1, inplace=True)

# Make a prediction for the user input
prediction = et.predict(user_df)
print(prediction)


# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np

# Define user input as a dictionary
user_input = {'Administrative_Duration': 50,
              'Informational_Duration': 100,
              'ProductRelated_Duration': 200,
              'BounceRates': 0.05,
              'ExitRates': 0.1,
              'PageValues': 20,
              'SpecialDay': 0,
              'Month': 7,
              'OperatingSystems': 0,
              'Browser': 0,
              'Region': 0,
              'TrafficType': 1,
              'VisitorType': 'New_Visitor',
              'Weekend': 1}

# Create a DataFrame from the user input dictionary
user_df = pd.DataFrame(user_input, index=[0])

# One-hot encode categorical variables to match training data
user_df = pd.get_dummies(user_df, columns=['VisitorType'])

# Ensure all expected columns are present
for col in X_train.columns:
    if col not in user_df.columns:
        user_df[col] = 0

# Reorder columns to match the training data
user_df = user_df[X_train.columns]

# Make a prediction for the user input
prediction = et.predict(user_df)
print(prediction)


# It is evident that for the given set of inputs, the customer is not likely to make a purchase online.

# Similarly, we can give the model different sets of input and predict whether a customer visiting an online shopping website will make a purchase or not.

# <div class="alert alert-block alert-warning">
#     <b>Note: </b> Our prediction is subject to our model's accuracy which is <b> approximately </b> 94%.
# </div>

# In[ ]:


# Evaluate the model's accuracy on the test set
accuracy = et.score(X_test, y_test) # Changed best_et to et
print(f"Model accuracy: {accuracy:.2%}")


# In[ ]:




