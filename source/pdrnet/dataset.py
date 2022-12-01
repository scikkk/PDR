import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from IPython import embed

def load_cached_sequences(root_dir, data_list, in_dim, out_dim):
    data = np.array(pd.read_csv(root_dir+ data_list[0] +".csv"))[:, 2:]
    features_all = data[:, :in_dim]
    labels_all = data[:, in_dim:]

    for l in data_list[1:]:
        data = np.array(pd.read_csv(root_dir+ l +".csv"))[:, 2:]
        feature = data[:, :in_dim]
        label = data[:, in_dim:]
        features_all = np.concatenate((features_all, feature), axis=0)
        labels_all = np.concatenate((labels_all, label), axis=0)

    return features_all, labels_all


class MyDataset(Dataset):
    def __init__(self, root_dir, data_list, in_dim, out_dim, shuffle):
        self.feature_dim = in_dim
        self.target_dim = out_dim
        self.features, self.labels = load_cached_sequences(root_dir, data_list, in_dim, out_dim)

    def __getitem__(self, item):
        return self.features[item].astype(np.float32), self.labels[item].astype(np.float32)

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    load_cached_sequences('../data/processed/', ['0','11'])