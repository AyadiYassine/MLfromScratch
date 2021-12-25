import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import Knn



breast_cancer=datasets.load_breast_cancer()
X,y=breast_cancer.data,breast_cancer.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape)
#print(X_train[0])

print(y_train.shape)
#print(y_train[0])

clf=Knn(k=3)
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)

acc=np.sum(predictions==y_test)/ len(y_test)
print(acc*100)