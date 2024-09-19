import numpy as np
import glob
import os
import scipy.io as sio
from torch.utils.data import Dataset

class CSI_Dataset():
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = glob.glob(os.path.join(root_dir, '*', '*.mat'))
        # 首先获取所有唯一的文件夹名（类别名）
        categories = list(set([file.split('/')[-2] for file in self.data_list]))

        # 然后创建一个映射，将每个类别映射到一个整数标签
        self.category = {categories[i]: i for i in range(len(categories))}

        # 使用这个映射来创建标签
        self.labels = [self.category[file.split('/')[-2]] for file in self.data_list]
        self.data = []
        self.load_data()

    def load_data(self):
        # 预加载整个数据集
        for sample_dir in self.data_list:
            x = sio.loadmat(sample_dir)['CSIamp']  # 加载指定模态的数据
            x = np.array(x)
            x = (x - 42.3199) / 4.9802  # 标准化
            # sampling: 2000 -> 500
            #x = x[:, ::2]  # 降维
            x = x.transpose()  # 转置

            if self.transform:
                x = self.transform(x)

            self.data.append(x)

        # 将数据转换为numpy数组
        self.data = np.array(self.data)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 由于已经预加载，直接返回对应索引的数据和标签
        return self.data[idx], self.labels[idx]

    def get_sorted_categories(self):
        # 根据映射字典的值（整数标签）对键（类别名称）进行排序
        sorted_categories = sorted(self.category, key=self.category.get)
        return sorted_categories