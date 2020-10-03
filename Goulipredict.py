# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:35:13 2017

@author: Panagiotis
"""

import sklearn.naive_bayes as nb
import sklearn.svm as svm
from sklearn.neighbors import KNeighborsClassifier


def predictions(a_train,a_test,b_train,b_test):
    #clf = KNeighborsClassifier(n_neighbors=1)
    clf=nb.GaussianNB()
    #clf=svm.SVC(kernel='rbf')
    clf.fit(a_train,b_train)
    pred=clf.predict(a_test)
    #pred=pred.reshape(len(pred),1)
    k=sum(pred==b_test)
    print('Predictions')       
    print(pred)
    print('Origin')
    print(b_test)
    miss=len(b_test)-k
    failure=float(miss)/float(len(b_test))
    failure=failure*100
    print('Failure rate')
    print(failure)
    
    
def newpredict(x_new,x_train,y_train):
    #clf = KNeighborsClassifier(n_neighbors=1)
    clf=nb.GaussianNB()
    #clf=svm.SVC(kernel='rbf')
    clf.fit(x_train,y_train)
    newpred=clf.predict(x_new)
    
    return newpred