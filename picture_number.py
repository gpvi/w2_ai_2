# -*- coding: utf-8 -*-            
# @Time : 2022/3/17 23:29
# @Author: Niuzhuoqun
# @FileName: picture_number.py
# @Software: PyCharm
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import joblib
from sklearn.model_selection import train_test_split

#1.数据处理

data  = np.loadtxt(open("train.csv", 'r'), delimiter=",", skiprows=1)
# print(np.shape(data))
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
# print(np.shape(train_set[:,0]))
rtrain,ctrain = np.shape(train_set)
xtrain = train_set[:,1:ctrain]
# xtrain_col_avg = np.mean(xtrain)
# xtrain = (xtrain-xtrain_col_avg)/255
std_scaler = StandardScaler()
std_scaler.fit(xtrain)
xtrain = std_scaler.transform(xtrain)
ytrain = train_set[:,0]

#2.训练模型

model = LogisticRegression(solver='lbfgs',multi_class="multinomial",max_iter=100)


model.fit(xtrain,ytrain)
    # print("valueerror")

#3.测试模型

testdata = np.loadtxt(open('test.csv', 'r'), delimiter=",", skiprows = 1)
rtest,ctest = np.shape(test_set)
print("测试集大小：",rtest,ctest)
xtest = test_set[:,1:ctest]
# xtest = (xtest-xtrain_col_avg)/255
# xtest_col_avg = np.mean(xtest,axis = 0)

std_scaler2 = StandardScaler()
std_scaler2.fit(xtest)
xtest =std_scaler2.transform(xtest)
ytest = test_set[:,0]
ypredic = model.predict(xtest)
errors = np.count_nonzero(ytest-ypredic)
print("预测完毕。错误：", errors, "条")
print("测试数据正确率:", (rtest - errors) / rtest)

# dirs = 'testModel'
# if not os.path.exists(dirs):
#     os.makedirs(dirs)
# joblib.dump(model, dirs+'/model.pkl')
# print("模型已保存")