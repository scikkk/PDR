import random
import numpy as np
from torch.utils.data import Dataset
from os import path as osp

class GlobSpeedSequence():
    """
    Dataset :- RoNIN (can be downloaded from http://ronin.cs.sfu.ca/)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 12
    target_dim = 3

    def __init__(self, data_path=None, **kwargs):
        self.features, self.targets = None, None
        if data_path is not None:
            if data_path[-1] != 'v':
                data_path += '.csv'
            self.load(data_path)

    def load(self, data_path):
        self.times = np.loadtxt(data_path, skiprows=1, dtype=float, usecols=range(1),delimiter=',')
        self.features = np.loadtxt(data_path, skiprows=1, dtype=float, usecols=range(2,14),delimiter=',')
        # self.features = features[:int(features.shape[0]/50)*50].reshape(-1,600)
        self.targets = np.loadtxt(data_path, skiprows=1, dtype=float, usecols=(14,15,16),delimiter=',')
        # targets = targets[range(0, targets.shape[0],50)]
        # self.targets = targets[1:]-targets[:-1]


    def get_feature(self):
        # np.savetxt(X=self.features,fname='1.csv',delimiter=',')
        return self.features

    def get_target(self):
        # np.savetxt(X=self.targets,fname='2.csv',delimiter=',')
        return self.targets

    def get_time(self):
        # np.savetxt(X=self.targets,fname='2.csv',delimiter=',')
        return self.times

def load_cached_sequences(seq_type, root_dir, data_list, **kwargs):
    features_all, targets_all = [], []
    for i in range(len(data_list)):
        seq = seq_type(osp.join(root_dir, data_list[i]), **kwargs)
        feat, targ = seq.get_feature(), seq.get_target()
        features_all.append(feat)
        targets_all.append(targ)
    return features_all, targets_all

class StridedSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list,  step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(StridedSequenceDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size)

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.features, self.targets = load_cached_sequences(
            seq_type, root_dir, data_list, interval=self.interval, **kwargs)
        for i in range(len(data_list)):
            self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


if __name__ == '__main__':
    dataset = GlobSpeedSequence("./test_case0/test_case0/processed/test_case0.csv")
    dataset.get_feature()
    dataset.get_target()
