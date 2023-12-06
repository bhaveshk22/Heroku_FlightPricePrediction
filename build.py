import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

df=pd.read_csv('deploy_df')

df.drop('Unnamed: 0',axis=1,inplace=True)

x=df.drop('Price',axis=1)
print(x.head())

y=df['Price']

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)



#Using the saved model 

import pickle
import gzip

# loading model
def load_zipped_model(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

model = load_zipped_model('cat_model.pkl')

