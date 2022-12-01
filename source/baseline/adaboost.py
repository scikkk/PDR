from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import pandas as pd
from os import path as osp
# 用于绘图
import sys

from read_data import dataReader


sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from save_res import output_result


all_data = dataReader()
X = np.array(all_data[['acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])
y_dx = np.array(all_data['Location_delta_x'])
y_dy = np.array(all_data['Location_delta_y'])
y_dir = np.array(all_data['Location_dir'])

regr_x = AdaBoostRegressor(n_estimators=50, learning_rate=0.3)
regr_x.fit(X, y_dx)
scorex = regr_x.score(X, y_dx)
print("Latitude: ", scorex)

regr_y = AdaBoostRegressor(n_estimators=50, learning_rate=0.3)
regr_y.fit(X, y_dy)
scorey = regr_x.score(X, y_dy)
print("Longitude: ", scorey)

X = np.array(all_data[['gyro_x','gyro_y','gyro_z','magnet_x','magnet_y','magnet_z']])
regr_dir = AdaBoostRegressor(n_estimators=50, learning_rate=0.3)
regr_dir.fit(X, y_dir)
score3 = regr_dir.score(X, y_dir)
print("Direction: ", score3)



# 测试
test = pd.read_csv("./test_case0/processed/test_case0.csv")
test_X = np.array(test[['acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])


pred_dx = regr_x.predict(test_X)
pred_dy = regr_y.predict(test_X)
test_X = np.array(test[['gyro_x','gyro_y','gyro_z','magnet_x','magnet_y','magnet_z']])
pred_dir = regr_dir.predict(test_X)

import joblib
#lr是一个LogisticRegression模型
joblib.dump(regr_x, './models/adaboost/x.model')
joblib.dump(regr_y, './models/adaboost/y.model')
joblib.dump(regr_dir, './models/adaboost/dir.model')

output_result("./test_case0/adaboost_Location_output.csv", "./test_case0/Location_input.csv",\
        test['time'], pred_dx, pred_dy, pred_dir )


