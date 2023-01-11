# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import requests

import warnings
warnings.filterwarnings('ignore')

# Importing the preprocessed Australian Weather Dataset
data=pd.read_csv("C:\\Users\\adilv\\Desktop\\Project Report\\Preprocessed aus_data.csv")

# Splitting X and y into features and target...y is the dependent variable
X=data.drop('RainTomorrow',axis=1)
y=data['RainTomorrow']

# Splitting the data into train and test
import sklearn
from sklearn.model_selection import train_test_split
# Model
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.2)

# Creating Random Forest Classification Model
from sklearn.ensemble import RandomForestClassifier
rand_for_model=RandomForestClassifier().fit(X_train,y_train)
y_pred_rand=rand_for_model.predict(X_test)
# Predicting the Test set Result
y_pred_rand=rand_for_model.predict(X_test)

# Fine Tuning using RandomizedSearchCV
model_random=RandomForestClassifier(n_estimators=250,max_depth=None)
model2=model_random.fit(X_train,y_train)
y_pred_random=model2.predict(X_test)

# Training the preprocessed data with the best Hyperparameters
model_random=RandomForestClassifier(n_estimators=250,max_depth=None)
model_random.fit(X,y)

# Saving model using pickle
pickle.dump(model_random,open('randomcv_model.pkl','wb'))

# Loading model to compare the result
model=pickle.load(open('randomcv_model.pkl','rb'))

print(model.predict([[2,22.9,13,44,13,14,22,1007.1,0]]))
