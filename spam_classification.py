import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

# importing dataset
data = pd.read_csv('spam.csv')
x = data['EmailText']
y = data['Label']

# splitting data set into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=20)

# vectorization
vectorizer = TfidfVectorizer()
x_train_vectors = vectorizer.fit_transform(x_train)
x_test_vectors = vectorizer.transform(x_test)

# training the model with classifier
parameters = {'C':(1,4),'gamma':(0.01,0.1)}
sv = svm.SVC()
clf = GridSearchCV(sv,parameters,cv=3)
clf.fit(x_train_vectors,y_train)
prediction = clf.predict(x_test_vectors)
accuracy = metrics.accuracy_score(y_test,prediction)
print('Accuracy: '+str(accuracy*100)+"%")
print("F1 Score(Spam,Ham): ",f1_score(y_test,clf.predict(x_test_vectors),average=None,labels= ['spam','ham']))

# Qualitative testing
test_set = ['an You Give Away Other Pegple&rsquo;s Money?&nbsp;&nbsp;What a ridiculous question! Of course you can. I do it all the time.&nbsp;&nbsp;The guys I work with, Martin and Roger, are business partners who have made it possible for me and YOU to give away their money to other people who are trying to make money online.&nbsp;They actually formed a club, made up of people who have been paid to join and who are also paid when they refer other people. It&rsquo;s called the Guarantee Downline Club and if you&rsquo;re not a member, you&rsquo;re missing out on free money and free training to help you make more money.&nbsp;&nbsp;Click the link below to learn how you can get your share...&nbsp;https://trafficzipper.com/track/225267-31-1500&nbsp;See you on the inside,Earl Savage&nbsp;P.S., After you complete your Guarantee Downline Club signup, here&rsquo;s another opportunity to make even more money! There&rsquo;s nothing to buy and nothing to sell; just click the link below and get paid. http://MarketAnyLink.com/dosh.php?id=EARLS21 &nbsp;&nbsp;Click here to earn credits http://www.listbonus.com/earn/232a0d370ca2139310e5df13f8cfa5c9/adhikarimila--------------------------------------------------------------------------------------LIST ADMINISTRATION--------------------------------------------------------------------------------------This is not SPAM. You have received this email because you are a member at Listbonus.com.','FREE Report Reveals The Traffic Sources That I Use to Get 500-1000 Referrals EASILY...Get more Referrals, Leads, and Sign-ups at ANY PROGRAM...https://ads-messenger.com/t/list/trafficbonus Click here to earn credits http://www.trafficbonus.com/earn/0f2f8996744f2e1fbf6fe1a6f49607d0/adhikarimilan-------------------------------------------------------------------------------------LIST ADMINISTRATION--------------------------------------------------------------------------------------This is not SPAM. You have received this email because you are a member at TrafficBonus.com.']
test_set_vectors = vectorizer.transform(test_set)
print(clf.predict(test_set_vectors))

