
from os import lstat

import scipy
from scipy import io
import cv2 as cv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def mat2csv(mat_pathname,data_features,csv_pathname):
    df=scipy.io.loadmat(mat_pathname)
    features=df[data_features]
    dfdata=pd.DataFrame(data=features)
    dfdata.to_csv(csv_pathname,index=False,header=False)

def csv2list(mat_pathname):
    with open(mat_pathname, 'r') as f:
         reader = csv.reader(f)
         list=[]
         for row in reader:
             list+=row
    return list

def get_density(img_path_name):
    #导入
    print(img_path_name)
    img = cv.imread(img_path_name,2)
    img_width=len(img[0])
    img_height=len(img)
    print('原尺寸:',img_width,img_height)
    #标准化尺寸
    img_out=cv.resize(img,(461,347))
    img_width_out=len(img_out[0])
    img_height_out=len(img_out)
    print('标准化尺寸：',img_width_out,img_height_out)
    #扁平化输出
    img_out=img_out.flatten()
    print('扁平化数组长度：',len(img_out),'\n')
    return img_out

def build_sample():
    dict={'ah2':ah2,'lb2':lb2,'population':population,\
        'terrain':terrain,'forest_height':forest_height,\
        'impervious_surface_percentage':impervious_surface_percentage,\
        'LST':LST}
    df=pd.DataFrame(dict)
    df.to_csv('sample.csv',index=False)


mat2csv('lb2.mat','lb2','lb2.csv')
mat2csv('ah2.mat','ah2','ah2.csv')
#mat2csv('X2.mat','X2','X2.csv')
#mat2csv('Y2.mat','Y2','Y2.csv')

lb2=csv2list('lb2.csv')
ah2=csv2list('ah2.csv')

LST=get_density('LST.tif')
forest_height=get_density('Forest_height_2019_SASIAutm49n.tif')
terrain=get_density('terrain.tif')
impervious_surface_percentage=get_density('impervious_surface_percentage_geographic_1000mutm49.tif')
population=get_density('population.tif')

build_sample()

path = 'sample.csv'
data = pd.read_csv(path)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X.values,y.values,test_size=0.2)

#linear regression
print('################### linear regression ##########')
from sklearn import linear_model
start=time.time()

#fit 回归模型
model=linear_model.LinearRegression()
model.fit(X_train,y_train)
#print('intercept_:',model.intercept_)
#print('coef_:','\n',model.coef_)

from sklearn.metrics import mean_squared_error,r2_score
print('Mean squared error:',mean_squared_error(y_test,model.predict(X_test)))
print ('Variance score：',r2_score(y_test,model.predict(X_test)))
print('score：',model.score(X_test,y_test))
print('time usage:',time.time()-start,'s')


#light GBM regression
import lightgbm as lgb
print('################### light GBM regression ##########')
start=time.time()

#model = lgb.LGBMRegressor(num_leaves=31,learning_rate=0.05,n_estimators=20)
model = lgb.LGBMRegressor()
model.fit (X_train,y_train.ravel())

print('Mean squared error；',mean_squared_error(y_test,model.predict(X_test)))
print ('Variance score：',r2_score(y_test,model.predict(X_test)))
print('score：',model.score(X_test,y_test))
print('time usage:',time.time()-start,'s')


import torch 
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

loss = nn.MSELoss()
in_features = X_train.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net