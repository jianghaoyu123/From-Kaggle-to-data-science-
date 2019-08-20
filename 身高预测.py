import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from matplotlib import pyplot as plt

height = pd.read_csv('height_train.csv')
test = pd.read_csv('height_test.csv')

test.head()

test.shape[0]
test['prediction']=np.zeros(test.shape[0])
test.head()

test['F2']=test.apply(lambda x: x['father_height']**2,axis=1)
test['M2']=test.apply(lambda x: x['mother_height']**2,axis=1)
height['F2']=height.apply(lambda x: x['father_height']**2,axis=1)
height['M2']=height.apply(lambda x: x['mother_height']**2,axis=1)

height0=height[height['boy_dummy']==0]
height0.head()

height1=height[height['boy_dummy']==1]
height1.head()

model0 = HuberRegressor()
model0.fit(X=height0.loc[:,['father_height','mother_height','F2']],y=height0.child_height)
model1 = HuberRegressor()
model1.fit(X=height1.loc[:,['father_height','mother_height','M2']],y=height1.child_height)

for i in range(test.shape[0]):
    if test.loc[i,'boy_dummy']==0:
        test.loc[i,'prediction']=model0.predict(test.loc[i,['father_height','mother_height','F2']].values.reshape(-1,3))
    else:
        test.loc[i,'prediction']=model1.predict(test.loc[i,['father_height','mother_height','M2']].values.reshape(-1,3))

test.head()

final=test.loc[:,['id','prediction']]

final.to_csv('F:/python×÷Òµ/work1-height prediction/½­ºÆÓê_HuberRegressor_twoorder.csv',index=False)