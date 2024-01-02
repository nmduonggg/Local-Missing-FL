from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import random

class PTBXLReduceDataset(Dataset):
    def __init__(self, root, download=True, standard_scaler=True, train=True, crop_length=250, valid=False):
        self.root = root
        self.standard_scaler = standard_scaler
        self.train = train
        self.crop_length = crop_length

        if not os.path.exists(self.root):
            if download:
                print('Downloading PTBXL Dataset...', end=' ')
                os.makedirs(root, exist_ok=True)
                os.system('bash ./benchmark/ptbxl_classification/download.sh')
                print('done!')
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it')
        if self.train:
            self.x = np.load(os.path.join(self.root, 'x_train.npy'))        # (3170, 1000, 12)
            self.y = np.load(os.path.join(self.root, 'y_train.npy'))        # (3170,)
            # if valid:
            #     self.x = self.x[2400:,:,:]
            #     self.y = self.y[2400:]
            # else:
            #     self.x = self.x[:2400,:,:]
            #     self.y = self.y[:2400]

            # import pdb; pdb.set_trace()
        else:
            self.x = np.load(os.path.join(self.root, 'x_test.npy'))
            self.y = np.load(os.path.join(self.root, 'y_test.npy'))
        if self.standard_scaler:
            self.ss = pickle.load(open(os.path.join(self.root, 'standard_scaler.pkl'), 'rb'))
            x_tmp = list()
            for x in self.x:
                x_shape = x.shape
                x_tmp.append(self.ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
            self.x = np.array(x_tmp)
            # import pdb; pdb.set_trace()
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        start_idx = random.randint(0, x.shape[0] - self.crop_length - 1)
        x = x[start_idx:start_idx + self.crop_length].transpose().astype(np.float32)    # 12xcrop_length
        y = y.astype(np.float32)
        return x, y
    
if __name__ == '__main__':
    dataset = PTBXLReduceDataset(root='./benchmark/RAW_DATA/PTBXL_REDUCE', standard_scaler=True, train=True)