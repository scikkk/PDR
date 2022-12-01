from sklearn import svm
import numpy as np
import pandas as pd
from os import path as osp
# 用于绘图
import sys

from read_data import dataReader

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from save_res import output_result

all_data = dataReader()
X = np.array(all_data[['gyro_x','gyro_y','gyro_z','acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])
y_dx = np.array(all_data['Location_delta_x'])
y_dy = np.array(all_data['Location_delta_y'])
y_dir = np.array(all_data['Location_dir'])



regr1 = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=1)
regr1.fit(X, y_dx)
score1 = regr1.score(X, y_dx)
print("Latitude: ", score1)

regr2 = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=1)
regr2.fit(X, y_dy)
score2 = regr2.score(X, y_dy)
print("Longitude: ", score2)

X = np.array(all_data[['gyro_x','gyro_y','gyro_z','magnet_x','magnet_y','magnet_z']])
regr3 = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=1)
regr3.fit(X, y_dir)
score3 = regr3.score(X, y_dir)
print("Direction: ", score3)



# 测试
test = pd.read_csv("./test_case0/processed/test_case0.csv")
test_X = np.array(test[['gyro_x','gyro_y','gyro_z','acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])

pred_dx = regr1.predict(test_X)
pred_dy = regr2.predict(test_X)
test_X = np.array(test[['gyro_x','gyro_y','gyro_z','magnet_x','magnet_y','magnet_z']])
pred_dir = regr3.predict(test_X)


import joblib
#lr是一个LogisticRegression模型
joblib.dump(regr1, './models/svr/x.model')
joblib.dump(regr2, './models/svr/y.model')
joblib.dump(regr3, './models/svr/dir.model')
output_result("./test_case0/svr_Location_output.csv", "./test_case0/Location_input.csv",\
        test['time'], pred_dx, pred_dy, pred_dir )

        


