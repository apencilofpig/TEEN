import os
import os.path as osp

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def remove_unused_index(inputs, labels, index_to_remove):
    # 创建条件索引，标记要保留的数据
    index_to_keep = np.ones(len(labels), dtype=bool)
    index_to_keep[index_to_remove] = False

    return inputs[index_to_keep], labels[index_to_keep]

def get_class_items(inputs, labels, cls_idx):
    if not hasattr(cls_idx, '__iter__'):
        index = np.where(labels == cls_idx)[0]
    else:
        index = None
        for each in cls_idx:
            index = np.where(labels == each)[0] if index is None else np.append(index, np.where(labels == each)[0])
    return index, inputs[index], labels[index] 

def get_few_shot_from_txt():
    index2 = open('data/index_list/swat/session_2.txt').read().splitlines()
    index3 = open('data/index_list/swat/session_3.txt').read().splitlines()
    index_all = np.array([int(x) for x in (index2 + index3)])
    return index_all, inputs[index_all], labels[index_all]

def generate_few_shot(inputs, labels, shot, cls_idx):
    index_all = None
    for idx in cls_idx:
        index, cls_inputs, cls_labels = get_class_items(inputs, labels, idx)
        idx_to_keep = np.random.choice(index, shot, replace=False)
        index_all = np.concatenate((index_all, idx_to_keep), axis=0) if index_all is not None else idx_to_keep
    return index_all, inputs[index_all], labels[index_all]

def generate_all_dataset(inputs, labels):
    # incremental_index_train, incremental_inputs_train, incremental_labels_train = generate_few_shot(inputs, labels, 5, range(26, 36))
    incremental_index_train, incremental_inputs_train, incremental_labels_train = get_few_shot_from_txt()
    inputs, labels = remove_unused_index(inputs, labels, incremental_index_train)
    _, base_inputs, base_labels = get_class_items(inputs, labels, range(26))
    _, incremental_inputs_test, incremental_labels_test = get_class_items(inputs, labels, range(26, 36))
    
    
    # 找到标签为0的索引
    zero_indices = np.where(base_labels == 0)[0]

    # 从中选择要去除的数量
    indices_to_remove = np.random.choice(zero_indices, len(zero_indices)-2000, replace=False)

    base_inputs, base_labels = remove_unused_index(base_inputs, base_labels, indices_to_remove)

    base_inputs_train, base_inputs_test, base_labels_train, base_labels_test = train_test_split(base_inputs, base_labels, test_size=0.2, random_state=3407)

    print(incremental_index_train)

    return base_inputs_train, base_labels_train, base_inputs_test, base_labels_test, incremental_inputs_train, incremental_labels_train, incremental_inputs_test, incremental_labels_test

df = pd.read_csv('data/swat/swat_ieee754.csv')
inputs = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values
# inputs = inputs / 256.0
# inputs = np.pad(inputs, ((0,0), (0,144-126)), mode='constant', constant_values=0)
# inputs = inputs.reshape(inputs.shape[0], 1, 12, 12)
inputs = inputs.reshape(inputs.shape[0], 1, -1)
base_inputs_train, base_labels_train, base_inputs_test, base_labels_test, incremental_inputs_train, incremental_labels_train, incremental_inputs_test, incremental_labels_test = generate_all_dataset(inputs, labels)


class Swat(Dataset):

    def __init__(self, root, train=True, transform=None,
                 index_path=None, index=None, base_sess=None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.transform = transform
        
        if train:
            if base_sess:
                self.data, self.targets = base_inputs_train, base_labels_train
            else:
                _, self.data, self.targets = get_class_items(incremental_inputs_train, incremental_labels_train, index)
        else:
            self.data = np.concatenate((base_inputs_test, incremental_inputs_test), axis=0)
            self.targets = np.concatenate((base_labels_test, incremental_labels_test), axis=0)
            _, self.data, self.targets = get_class_items(self.data, self.targets, index)

        self.data = torch.from_numpy(self.data).long()
        self.targets = torch.from_numpy(self.targets)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        datas, targets = self.data[i], self.targets[i]
        return datas, targets
    
