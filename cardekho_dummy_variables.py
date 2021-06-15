import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# data
data1 = pd.read_csv('cardekho.csv')

#checking null values
data = data1.dropna()

# removing unnecessary column
data = data.drop(['name'],axis = 1)
data['Used for']= 2020 -data['year']
data = data.drop(['year'],axis=1)

# ENCODING
data = pd.get_dummies(data,drop_first=True)

# separating X and Y
x_variables = data.drop(['selling_price'],axis =1)

# splitting train and test
train_x,test_x,train_y,test_y = train_test_split(x_variables,data['selling_price'],test_size=0.1,random_state=2)

# Model
clf = LinearRegression()
clf.fit(train_x,train_y)
print(clf.score(test_x,test_y))











