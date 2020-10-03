# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:26:30 2017

@author: Panagiotis
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import mglearn
from Goulipredict import predictions,newpredict

data=load_iris()
target_name=data['target_names']
target=data['target']
features_name=data['feature_names']
features=data['data']

#print(target_name)
#print(target)
#print(features_name)
#print(features)

f_train,f_test,label_train,label_test=train_test_split(features,target,
                                                       random_state=0)
#print(f_train)

f_df=pd.DataFrame(f_train,columns=features_name)
#label_df=pd.DataFrame(label_train,columns=target_name)
#print(f_df)
pl= pd.plotting.scatter_matrix(f_df, c=label_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)



predictions(f_train,f_test,label_train,label_test)
f_new=np.array([6,3.5,1,2.45])
new=newpredict(f_new,f_train,label_train)
print('0 : setosa , 1: versicolor ')
print('2 : virginica')
print(new)
