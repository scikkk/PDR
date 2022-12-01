from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd




import joblib
from read_data import dataReader
import sys
from os import path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from save_res import output_result

def getYaw(accVals, magVals) ->float:
    roll = np.arctan2(accVals[:,0],accVals[:,2])
    return roll[:,np.newaxis]*180/np.pi
    # print(roll*180/np.pi)
    pitch = -np.arctan(accVals[:,1]/(accVals[:,0]*np.sin(roll)+accVals[:,2]*np.cos(roll)))
    pitch[np.isnan(pitch)] = 0
    return pitch[:,np.newaxis]*180/np.pi
    # print(pitch*180/np.pi)
    yaw = np.arctan2(magVals[:,0]*np.sin(roll)*np.sin(pitch)+magVals[:,2]*np.cos(roll)*np.sin(pitch)+magVals[:,1]*np.cos(pitch), magVals[:,0]*np.cos(roll)-magVals[:,2]*np.sin(roll))
    # print(np.sum(np.isnan(yaw)))
    return np.concatenate((roll[:,np.newaxis],pitch[:,np.newaxis]),axis=1)*180/np.pi

# train
all_data = dataReader()
X = np.array(all_data[['acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])
y_dx = np.array(all_data['Location_delta_x'])
y_dy = np.array(all_data['Location_delta_y'])
y_dir = np.array(all_data['Location_dir'])
y_dxy = np.array(all_data[['Location_delta_x', 'Location_delta_y']])
mag = np.array(all_data[['magnet_x','magnet_y','magnet_z']])
acc = np.array(all_data[['acce_x','acce_y','acce_z']])
yaw = getYaw(mag,acc)
# exit(0)

regr_xy = RandomForestRegressor(n_estimators=4000,
                            max_depth=10,
                            max_samples=6000,
                            verbose=1)
regr_xy.fit(X, y_dxy)
score0 = regr_xy.score(X, y_dxy)
print("Latitude&Longitude: ", score0)


X = np.array(all_data[['gyro_x','gyro_y','magnet_x','magnet_y','magnet_z']])
regr_dir = RandomForestRegressor(n_estimators=1000,
                            max_depth=6,
                            max_samples=10000,
                            criterion='absolute_error',
                            # criterion='squared_error',
                            random_state=3470,
                            verbose=0)
# X = np.concatenate((X,yaw),axis=1)
regr_dir.fit(X, y_dir)
score3 = regr_dir.score(X, y_dir)
print("Direction: ", score3)

# # load model
# regr_xy = joblib.load('models/rf/xy_400_8_8000.model')
# regr_dir = joblib.load('models/rf/dir.model')

# 测试
test = pd.read_csv("./test_case0/processed/test_case0.csv")
test_X = np.array(test[['acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])
test_mag = np.array(test[['magnet_x','magnet_y','magnet_z']])
test_acc = np.array(test[['acce_x','acce_y','acce_z']])
test_yaw = getYaw(test_mag,test_acc)

pred_dxy = regr_xy.predict(test_X)
test_X = np.array(test[['gyro_x','gyro_y','magnet_x','magnet_y','magnet_z']])
# test_X = np.concatenate((test_X,test_yaw),axis=1)
pred_dir = regr_dir.predict(test_X)


#lr是一个LogisticRegression模型
joblib.dump(regr_xy, './models/rf/xy.model')
joblib.dump(regr_dir, './models/rf/dir.model')

output_result("./test_case0/rf_Location_output.csv", "./test_case0/Location_input.csv",\
        test['time'], pred_dxy[:,0], pred_dxy[:,1], pred_dir )


