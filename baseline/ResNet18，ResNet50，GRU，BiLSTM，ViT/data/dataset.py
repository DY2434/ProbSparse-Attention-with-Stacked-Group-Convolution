import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform
        self.data_list = glob.glob(root_dir + '/*/*.mat')
        self.folder = glob.glob(root_dir + '/*/')
        self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]

        # normalize
        x = (x - 42.3199) / 4.9802

        # sampling: 2000 -> 500
        # x = x[:, ::4]  #为严谨比较复杂度，输入与我们相同，不再降维
        x = x.reshape(3, 114, 2000)

        if self.transform:
            x = self.transform(x)

        x = torch.FloatTensor(x)

        return x, y