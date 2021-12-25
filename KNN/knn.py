import numpy as np
from collections import Counter
#from KNN.test import X_train

def euclidean_distance(x1,x2):
    d=np.sqrt(np.sum((x1-x2)**2))
    return d
class Knn:
    def __init__(self,k=3):
        self.k=k 
    
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
        pass
    

    def predict(self,X):
        predicted_lables=[self._predict(x) for x in X]
        return np.array(predicted_lables)

    def _predict(self,x):
        #compute distances
        distances=[euclidean_distance(x,x_train) for x_train in self.X_train]
        #get k nearst samples,labes
        k_indices=np.argsort(distances)[0:self.k]
        k_nearest_lables=[self.y_train[i] for i in k_indices]
        #most common class label
        most_common=Counter(k_nearest_lables).most_common(1)

        return most_common[0][0]