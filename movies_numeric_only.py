# in this code, i have only learnt to use multivariate linear regression.
# i had some problems with the vectorization so i took only numeric values.

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import  neighbors
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import GridSearchCV

# Importing data and removing excess columns
file1 = pd.read_csv('movie.csv')
file1.drop(['Rank','Title','Genre','Description','Director','Actors','Year','Metascore'],axis =1,inplace = True)
# checking to see null values
    # found null values in revenue
    # removing them
file = file1.dropna()
x_values = file.drop('Revenue',axis =1)

# Outlier analysis
# here i wont be needing outlier analysis, because a movie can gross any amount of money

# Multi collinearity
# plt.figure(figsize=(6,5))
# sn.heatmap(file.corr(),annot=True)
# plt.show()
# no collinearity above 0.7 were found

# for multivariate, separate x and y
x = sm.add_constant(x_values)
y = file.iloc[:,3]
train_x,test_x,train_y,test_y = train_test_split(x,
                                                 y,
                                                 test_size=0.1,
                                                 random_state=42)

# Model
clf = LinearRegression()
#clf = neighbors.KNeighborsRegressor(n_neighbors=20,weights='uniform')
# parameters = {'C':(9*(10**20),9*(10**10)),'gamma':(0.01,0.0001,0.00001)}
# sv = svm.SVR(kernel = 'rbf')
# clf = GridSearchCV(sv,parameters,cv=5)
clf.fit(train_x,train_y)
print(clf.score(test_x,test_y))



