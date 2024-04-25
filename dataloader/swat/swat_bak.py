import os
import os.path as osp

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def read_files_in_folder(folder_path):
    num_to_remove = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    num_to_remove.append(int(line))

    return num_to_remove


class Swat(Dataset):

    def __init__(self, root, train=True, transform=None,
                 index_path=None, index=None, base_sess=None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.transform = transform
        self.np_inputs = np.load(self.root + '/swat/ieee754_inputs.npy')
        self.np_labels = np.load(self.root + '/swat/labels.npy').astype(np.int64)
        # self.np_inputs = np.transpose(self.np_inputs, (0, 2, 1))
        self.np_inputs = self.np_inputs.reshape(self.np_inputs.shape[0], 1, -1)
        
        num_to_remove = read_files_in_folder('data/index_list/swat')

        # 找到标签为0的索引
        zero_indices = np.where(self.np_labels == 0)[0]

        # 从中选择要去除的数量
        indices_to_remove = np.random.choice(zero_indices, len(zero_indices)-2000, replace=False)

        num_to_remove = np.concatenate((num_to_remove,indices_to_remove),axis=0)

        # 创建条件索引，标记要保留的数据
        index_to_keep = np.ones(len(self.np_labels), dtype=bool)
        index_to_keep[num_to_remove] = False

        # 使用条件索引，保留标签不为0的数据
        filtered_np_inputs = self.np_inputs[index_to_keep]
        filtered_np_labels = self.np_labels[index_to_keep]
        np_inputs_train, np_inputs_test, np_labels_train, np_labels_test = train_test_split(filtered_np_inputs, filtered_np_labels, test_size=0.2, random_state=42)
        
        if train:
            self.data = np_inputs_train
            self.targets = np_labels_train
            # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(index_path)
        else:
            self.data = np_inputs_test
            self.targets = np_labels_test
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def SelectfromTxt(self, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            data_tmp.append(self.np_inputs[int(i)])           
            targets_tmp.append(self.np_labels[int(i)])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        count = 0
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
                count = count + 1
                # if count == 200:
                #   break

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        datas, targets = self.data[i], self.targets[i]
        return datas, targets