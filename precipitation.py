import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from google.colab import drive
drive.mount('/content/drive')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
data = pd.read_csv("/content/weatherdata.csv")
data.dropna(axis=0)
data
target_feature = data.Precipitation
# input features
features = ['Longitude', 'Latitude', 'Max Temperature',
            'Min Temperature','Wind','Relative Humidity']
input_features = data[features]

train_x,X,train_y,Y = train_test_split(input_features,target_feature,random_state=0)
regFunc = RandomForestRegressor(random_state=1)
regFunc.fit(train_x,train_y)
# Saving model to disk,
pickle.dump(regFunc,open('PRECIP_MODEL.pkl','wb'))