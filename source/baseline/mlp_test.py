import torch
import numpy as np
import pandas as pd

# import torch.utils.data as Data


from os import path as osp
# 用于绘图
import sys
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

def model_checker(i):
    model = torch.load("source/baseline/model/"+str(i+1)+ ".pt")
    df = pd.read_csv("test_case0/processed/test_case0.csv")
    X = np.array(df[['gyro_x','gyro_y','gyro_z','acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])#,
    # Y = np.array(df[['Location_delta_x','Location_delta_y','Location_dir']])

    X = torch.from_numpy(X.astype(np.float32)).cuda()
    # Y = torch.from_numpy(Y.astype(np.float32)).cuda()

    Y_0 = model(X)
    Y_0 = Y_0.cpu().detach().numpy()
    pred_dx = Y_0[:,0]
    pred_dy = Y_0[:,1]
    pred_dir = Y_0[:,1]
    # print(pred_dx,pred_dy,pred_dir)
    output_result("test_case0/mlp_res/"+ str(i + 1) +"mlp_Location_output.csv","test_case0/Location_input.csv", df['time'],pred_dx,pred_dy,pred_dir)

if __name__ == "__main__":
    for i in range(200):
        model = torch.load("source/baseline/model/"+str(i + 1)+ ".pt")
        df = pd.read_csv("test_case0/processed/test_case0.csv")
        X = np.array(df[['gyro_x','gyro_y','gyro_z','acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])#,
        # Y = np.array(df[['Location_delta_x','Location_delta_y','Location_dir']])

        X = torch.from_numpy(X.astype(np.float32)).cuda()
        # Y = torch.from_numpy(Y.astype(np.float32)).cuda()

        Y_0 = model(X)
        Y_0 = Y_0.cpu().detach().numpy()
        pred_dx = Y_0[:,0]
        pred_dy = Y_0[:,1]
        pred_dir = Y_0[:,1]
        # print(pred_dx,pred_dy,pred_dir)
        output_result("test_case0/mlp_res/"+ str(i + 1) +"mlp_Location_output.csv","test_case0/Location_input.csv", df['time'],pred_dx,pred_dy,pred_dir)
    # print(Y_0)