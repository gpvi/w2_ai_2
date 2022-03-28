# -*- coding: utf-8 -*-
# @Time : 2022/3/17 23:29
# @Author: Niuzhuoqun
# @FileName: picture_number.py
# @Software: PyCharm
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import os
import joblib
from sklearn.model_selection import cross_val_score# K折交叉验证模块
from sklearn.model_selection import train_test_split

#1.数据处理

data  = np.loadtxt(open("train.csv", 'r'), delimiter=",", skiprows=1)
# print(data)
#train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
rtrain,ctrain = np.shape(data)
xtrain = data[:,1:ctrain]
# xtrain_col_avg = np.mean(xtrain)
# xtrain = (xtrain-xtrain_col_avg)/255
std_scaler = StandardScaler()
std_scaler.fit(xtrain)
xtrain = std_scaler.transform(xtrain)
ytrain = data[:,0]



#2.训练模型

knn=KNeighborsClassifier(n_neighbors=15)


#使用K折交叉验证模块(把样本分成5份，每一份都为训练集，得到精确度再求平均值）
scores = cross_val_score(knn, xtrain, ytrain, cv=5, scoring='accuracy')

#3.测试模型

print(scores)

print(scores.mean())

