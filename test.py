import os
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt

def eval_model(test_path, model=''):
    print(f'{test_path}:')
    if model != '':
        model = model+'_'
    gt = pd.read_csv(os.path.join(test_path, "Location.csv"))
    pred = pd.read_csv(os.path.join(test_path, f"{model}Location_output.csv"))
    dist_error = get_dist_error(gt, pred)
    dir_error = get_dir_error(gt, pred)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.subplot(211)
    plt.plot(gt['Latitude (°)'],gt['Longitude (°)'],marker='o',markersize=2,markeredgecolor='r')
    plt.plot(pred['Latitude (°)'],pred['Longitude (°)'],marker='o',markersize=2,markeredgecolor='b')
    plt.title('Latitude-Lontitude')
    plt.subplot(212)
    plt.ylim((0, 400))
    plt.plot(gt['Time (s)'],gt['Direction (°)'])
    plt.plot(pred['Time (s)'],pred['Direction (°)'])
    plt.title('DireCtion')
    name = test_path.split('/')[-1].split('\\')[-1]
    plt.savefig(f'./image/testres/{model}{name}.png')
    plt.savefig(f'./image/testres/{model}{name}.pdf')
    print(f'savefig: ./image/testres/{model}{name}.png')
    plt.clf()
    print(f"{model}Distances error: ", dist_error)
    print(f"{model}Direction error: ", dir_error)
    return dist_error


def get_dir_error(gt, pred):
    dir_list = []
    for i in range(int(len(gt) * 0.1), len(gt)):
        if gt[gt.columns[5]][i] == -1:
            continue
        dir = min(abs(gt[gt.columns[5]][i] - pred[pred.columns[5]][i]%360), 360 - abs(gt[gt.columns[5]][i] - pred[pred.columns[5]][i]%360))
        dir_list.append(dir)
    error = sum(dir_list) / len(dir_list)
    return error


def get_dist_error(gt, pred):
    dist_list = []
    for i in range(int(len(gt) * 0.1), len(gt)):
        dist = geodesic((gt[gt.columns[1]][i], gt[gt.columns[2]][i]), (pred[pred.columns[1]][i], pred[pred.columns[2]][i])).meters
        dist_list.append(dist)
    error = sum(dist_list) / len(dist_list)
    return error


if __name__ == "__main__":    
    # eval_model("test_case0",'rf')
    # eval_model("test_case0",'svr')
    # eval_model("test_case0",'adaboost')
    # eval_model("test_case0",'none')
    eval_model("test_case0")

    '''
    如有需要, 将 'Location.csv' 放入 ./outputs/test* 下即可进行测试
    for fname in [1,2,3,5,6,7,8,9,10,11]:
        eval_model(f"./outputs/{fname}")
    '''
    
