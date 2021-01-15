from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2

class _OWN(data.Dataset):
    def __init__(self, config, is_train=True, labelPath=None):

        if config.DATASET.ROOT == '':
            self.root = None
        else:
            self.root = config.DATASET.ROOT

        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        if labelPath is not None:
            txt_file = labelPath
        else:
            txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        with open(txt_file, 'r', encoding='utf-8') as file:
            self.labels = [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels[idx].keys())[0]

        if self.root is None:
            img = cv2.imdecode(np.fromfile(img_name, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imdecode(np.fromfile(os.path.join(self.root, img_name), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        img_h, img_w = img.shape

        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx








