import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import metrics
class reviewclass:
    def __init__(self,text,score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
    def get_sentiment(self):
        if self.score>3:
            return 'POSITIVE'
        elif self.score<3:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
reviews = []

file_name = 'Books_small.json'
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(reviewclass(review['reviewText'],review['overall']))

#splitting
train,test = train_test_split(reviews,test_size=0.40,random_state=95)

x_train = [x.text for x in train]
y_train = [x.sentiment for x in train]

x_test = [x.text for x in test]
y_test = [x.sentiment for x in test]

# bag of words vectorization
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(x_train)
test_x_vectors = vectorizer.transform(x_test)


# classifier
parameters = {'C':(1,10),'gamma':(0.1,0.01)}
sv = svm.SVC()
clf = GridSearchCV(sv,parameters,cv=2)
clf.fit(train_x_vectors,y_train)
print(clf.best_params_)
prediction = clf.predict(test_x_vectors)
accuracy = metrics.accuracy_score(y_test,prediction)
print(accuracy)
# print(x_test[200])
# print(clf.predict(test_x_vectors[200]))



# clf_gnb = GaussianNB()
# clf_gnb.fit(train_x_vectors,y_train)
# clf_gnb.predict(test_x_vectors[0])





