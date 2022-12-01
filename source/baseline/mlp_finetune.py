import random
import torch
import torch.nn.functional as F   # 激励函数的库
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset, TensorDataset
import read_data
from tqdm import tqdm
from mlp_test import model_checker
import os
from geopy.distance import geodesic

def eval_model(test_path, model=''):
    if model != '':
        model = model+'_'
    gt = pd.read_csv(os.path.join(test_path, "Location.csv"))
    pred = pd.read_csv(os.path.join(test_path, f"{model}Location_output.csv"))
    dist_error = get_dist_error(gt, pred)
    # dir_error = get_dir_error(gt, pred)
    print(f"{model}Distances error: ", dist_error)
    # print(f"{model}Direction error: ", dir_error)
    return dist_error

def get_dist_error(gt, pred):
    # print("local_error")
    dist_list = []
    for i in range(int(len(gt) * 0.1), len(gt)):
        dist = geodesic((gt[gt.columns[1]][i], gt[gt.columns[2]][i]), (pred[pred.columns[1]][i], pred[pred.columns[2]][i])).meters
        dist_list.append(dist)
    error = sum(dist_list) / len(dist_list)
    return error

def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(3407)

torch.device("cuda")
df = read_data.testReader()
X = np.array(df[['gyro_x','gyro_y','gyro_z','acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])#,
Y = np.array(df[['Location_delta_x','Location_delta_y']])

X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1,random_state=100)

X_train = torch.from_numpy(X_train.astype(np.float32)).cuda()
Y_train = torch.from_numpy(Y_train.astype(np.float32)).cuda()
X_test = torch.from_numpy(X_test.astype(np.float32)).cuda()
Y_test = torch.from_numpy(Y_test.astype(np.float32)).cuda()

train_data = TensorDataset(X_train,Y_train)
test_data = TensorDataset(X_test,Y_test)
# from torch.utils.data import Dataloader
train_loader = DataLoader(train_data, batch_size = 128, shuffle = True)
# print(X.shape)
# 创建加载器
# train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0)

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


if __name__ == '__main__':
    model = torch.load("source/baseline/model/1.pt")
    # eval_model("test_case0/mlp_res",str(epoch+1) +'mlp')
    # model = MLP().cuda()
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = torch.nn.MSELoss()
    train_loss_all = []
    test_loss = []
    for epoch in range(100):
        # print(epoch)
        train_loss = 0
        train_num = 0
        for step, (b_x,b_y) in enumerate(tqdm(train_loader)):
            # print(step,b_x,b_y)
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
            train_num += b_x.size(0)
        train_loss_all.append(train_loss/train_num)
        
        torch.save(model,"source/baseline/model/"+ str(epoch+1) +"_1.pt")
        pred = model.forward(X_test)
        pred = torch.squeeze(pred)
        loss_test = loss_func(pred, Y_test).item() 
        test_loss.append(loss_test)
        print("epoch:{}, loss_test:{}".format(epoch+1, loss_test))
        model_checker(epoch)
        eval_model("test_case0/mlp_res",str(epoch+1) +'mlp')

    # plt.figure(figsize = (8, 6))
    plt.plot(train_loss_all, 'ro-', label = 'Train loss')

    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.savefig('./mlp2.jpg')
    plt.clf()
    plt.plot(test_loss,'bo-', label = 'Test loss')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.savefig('./mlp_test2.jpg')
