import matplotlib.pyplot as plt
import numpy as np

def draw_raw():
    dataset_list = []
    with open('./lists/data_list.txt') as f:
                for s in f.readlines():
                    if s[0] != '#':
                        dataset_list.append(s.strip('\n'))

    for dataset in dataset_list:
        t_X_Y_dir = np.loadtxt(f'./data/raw/{dataset}/Location.csv', usecols=(0,1,2,5),skiprows=1, dtype=float,delimiter =',')
        time = t_X_Y_dir[:,0]
        latitude = t_X_Y_dir[:,1]
        Longitude = t_X_Y_dir[:,2]
        dir = t_X_Y_dir[:,3]
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        plt.subplot(211)
        plt.plot(latitude,Longitude,marker='o',markersize=2,markeredgecolor='r')
        plt.title('Latitude-Longitude')
        plt.subplot(212)
        plt.ylim((0, 400))
        plt.plot(time,dir)
        plt.title('Direction')
        plt.savefig(f'./image/rawdata/{dataset}.png')
        plt.savefig(f'./image/rawdata/{dataset}.pdf')
        print(f'savefig: ./image/rawdata/{dataset}.png')
        plt.clf()
def draw_processed():
    dataset_list = []
    with open('./lists/data_list.txt') as f:
                for s in f.readlines():
                    if s[0] != '#':
                        dataset_list.append(s.strip('\n'))

    for dataset in dataset_list:
        t_X_Y_dir = np.loadtxt(f'./data/processed/{dataset}.csv', usecols=(1,14,15,16),skiprows=1, dtype=float,delimiter =',')
        time = t_X_Y_dir[:,0]
        latitude = np.cumsum(t_X_Y_dir[:,1])

        Longitude = np.cumsum(t_X_Y_dir[:,2])
        dir = t_X_Y_dir[:,3]
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        plt.subplot(211)
        plt.plot(latitude,Longitude,marker='o',markersize=2,markeredgecolor='r')
        plt.title('Latitude-Longitude')
        plt.subplot(212)
        plt.ylim((0, 400))
        plt.plot(time,dir)
        plt.title('Direction')
        plt.savefig(f'./image/processeddata/{dataset}.png')
        plt.savefig(f'./image/processeddata/{dataset}.pdf')
        print(f'savefig: ./image/processeddata/{dataset}.png')
        plt.clf()

def draw_pred_ground():
    
    dataset_list = ['./test_case0']
    with open('./lists/test_list.txt') as f:
        for s in f.readlines():
            if s[0] != '#':
                dataset_list.append('TestSet/'+s.strip('\n')) 
    for test_data in dataset_list:
        t_X_Y_dir = np.loadtxt(f"./{test_data}/Location_output.csv", usecols=(0,1,2,5),skiprows=1, dtype=float,delimiter =',')
        time = t_X_Y_dir[:,0]
        latitude = t_X_Y_dir[:,1]
        Longitude = t_X_Y_dir[:,2]
        dir = t_X_Y_dir[:,3]

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        plt.subplot(211)
        
        plt.plot(latitude,Longitude,marker='o',markersize=2,markeredgecolor='r')
        plt.title('Latitude-Longitude')
        plt.subplot(212)
        plt.ylim((0, 400))
        plt.plot(time,dir)
        plt.title('Direction')
        name = test_data.split('/')[-1].split('\\')[-1]
        plt.savefig(f'./image/prediction/{name}.png')
        plt.savefig(f'./image/prediction/{name}.pdf')
        print(f'savefig: ./image/prediction/{name}.png')
        plt.clf()

if __name__ == '__main__':
    draw_raw()
    draw_pred_ground()
    draw_processed()