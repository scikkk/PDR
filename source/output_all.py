import numpy as np
import pandas as pd
import joblib
import sys
import torch
import os
from os import path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from save_res import output_result
def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden1 = torch.nn.Linear(12,32)
        self.hidden2 = torch.nn.Linear(32,32)
        self.predict = torch.nn.Linear(32,2)
        # self.fc3 = torch.nn.Linear(128,10)
        self.apply(weights_init_)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden1(x))
        x = torch.nn.functional.relu(self.hidden2(x))
        # x = torch.nn.functional.relu(self.predict(x))
        output = self.predict(x)
        return output

# load model
mlp_model = torch.load('./models/mlp/1.pt')
# mlp_model.cuda('cuda:1,2')

# regr_xy = joblib.load('./models/rf/xy.model')
regr_dir = joblib.load('./models/rf/dir_nogc.model')
# regr_dir = joblib.load('./models/rf/dir_1000_6_10000_nogc.model')
# regr_dir = joblib.load('./models/rf/dir_400_8_8000.model')
# regr_dir = joblib.load('./models/rf/dir_400_8_8000_seed.model')
regr_dir.verbose=0
# load list
dataset_list = []
with open('./lists/test_list.txt') as f:
    for s in f.readlines():
        if s[0] != '#':
            dataset_list.append(s.strip('\n'))
    
''' test_case0
# 测试
test = pd.read_csv(f"./test_case0/processed/test_case0.csv")
# test_dir = np.array(test[['gyro_x','gyro_y','gyro_z','magnet_x','magnet_y','magnet_z']])
test_dir = np.array(test[['gyro_x','gyro_y','magnet_x','magnet_y','magnet_z']])
pred_dir = regr_dir.predict(test_dir)


test_xy = np.array(test[['gyro_x','gyro_y','gyro_z','acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])
# pred_xy = regr_dir.predict(test_xy)
# pred_dx = pred_xy[:,0]
# pred_dy = pred_xy[:,1]
test_xy = torch.from_numpy(test_xy.astype(np.float32)).cuda()
x_y_dir = mlp_model(test_xy)
x_y_dir = x_y_dir.cpu().detach().numpy()
pred_dx = x_y_dir[:,0]
pred_dy = x_y_dir[:,1]
# pred_dir = x_y_dir[:,1]
output_result(f"./test_case0/Location_output.csv", f"./test_case0/Location_input.csv",test['time'], pred_dx, pred_dy, pred_dir )
print(f"Output: ./test_case0/Location_output.csv")
'''


# predict
print('Begin prediction!')
for test_data in dataset_list:
    # 测试
    test = pd.read_csv(f"./TestSet/processed/{test_data}.csv")
    # test_dir = np.array(test[['gyro_x','gyro_y','gyro_z','magnet_x','magnet_y','magnet_z']])
    test_dir = np.array(test[['gyro_x','gyro_y','magnet_x','magnet_y','magnet_z']])
    pred_dir = regr_dir.predict(test_dir)%360

    test_xy = np.array(test[['gyro_x','gyro_y','gyro_z','acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])
   
    test_xy = torch.from_numpy(test_xy.astype(np.float32)).cuda()
    x_y_dir = mlp_model(test_xy)
    x_y_dir = x_y_dir.cpu().detach().numpy()
    pred_dx = x_y_dir[:,0]
    pred_dy = x_y_dir[:,1]
    # pred_dir = x_y_dir[:,1]
    output_result(f"./TestSet/{test_data}/Location_output.csv", f"./TestSet/{test_data}/Location_input.csv",test['time'], pred_dx, pred_dy, pred_dir )
    print(f"Output: ./TestSet/{test_data}/Location_output.csv")


