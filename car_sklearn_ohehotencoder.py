import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
from sklearn.linear_model import LinearRegression

# data
file1 = pd.read_csv('cardekho.csv')
file1['Age'] = 2021-file1['year']
file1.drop(['name','year'],axis=1,inplace = True)
#print(file1.head())

# CHECKING NAN values
#print(file1.info())
# no null values

# Label encoding
le = LabelEncoder()
file = file1
file.fuel = le.fit_transform(file.fuel)
file.seller_type = le.fit_transform(file.seller_type)
file.transmission = le.fit_transform(file.transmission)
file.owner = le.fit_transform(file.owner)

# checking unique values in each column
# print(file.head())
# print(file['fuel'].unique())
# print(file['seller_type'].unique())
# print(file['transmission'].unique())
# print(file['owner'].unique())

# converting Features and labels into array from dataframe
x_encode = file[['fuel','seller_type','transmission','owner']].values
x_no_encode = file[['km_driven','Age']]
y = file.selling_price


# One Hot encoding
ohe = OneHotEncoder(drop='first')
encoded = ohe.fit_transform(x_encode).toarray()
#print(x_no_encode)
final_x = np.concatenate((x_no_encode,encoded),axis=1)



# model
model = LinearRegression()
model.fit(final_x,y)
print(model.score(final_x,y))
#print(model.predict([[68000,14,0,0,0,0,0,0,0,0,0,0,0]]))







