# sys
from multiprocessing.pool import IMapIterator
from multiprocessing.sharedctypes import Value
import os
import sys
from matplotlib.pyplot import subplots_adjust
import numpy as np
import random
import pickle

# image preprocess
import cv2

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# operation
from . import feeder_utils

subject_ID_set = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
                  'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020',
                  'F021', 'F022', 'F023', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 
                  'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018']

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        image_path: the path to file which contains image path ('.pkl' data)
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """
    def __init__(
        self,
        data_path,
        label_path,
        random_choose=False,
        random_move=False,
        window_size=-1,
        image=False,
        imagepath=False,
        image_path=None,
        image_size=256,
        debug=False,
        mmap=False,        
        istrain=False):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.image = image        
        self.imagepath = imagepath
        self.image_size = image_size
        self.image_path = image_path
        self.istrain = istrain

        print("dataset: istrain =", istrain, ":")
        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C T V

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        N, C, T, V = self.data.shape

        if self.image or self.imagepath:    # True False
            # load image by image_path
            with open(self.image_path, 'rb') as f:
                _, self.imagepaths = pickle.load(f)

        if len(self.data.shape) == 3:
            self.data = self.data[:, :, np.newaxis, :]

        self.N, self.C, self.T, self.V = self.data.shape
        print("  data.shape:", self.data.shape)    # data: (total_frames=95928/50604, 2, 1, 22)

        print("  imagepaths len:", len(self.imagepaths), len(self.imagepaths[0]))  # imagepaths: (total_frames, 1)
        self.imagepaths = np.array(self.imagepaths).reshape(-1).tolist()
        print("  new imagepaths len:", len(self.imagepaths))  # imagepaths: (total_frames)

        print("  label len:", len(self.label), len(self.label[0]), len(self.label[0][0]))   # label: (total_frames, 1, 12)
        self.label = torch.tensor(self.label).view(-1, 12).tolist()
        print("  new label len:", len(self.label), len(self.label[0]))  # label: (total_frames, 12)

        self.subID = []
        for imgpath in self.imagepaths:
            subject_idx1 = imgpath.find('F')
            subject_idx2 = imgpath.find('M')
            if subject_idx1 != -1:
                # FXXX
                subid = imgpath[subject_idx1:subject_idx1+4]
                subid_int = int(subid[1:])
            elif subject_idx2 != -1:
                # MXXX
                subid = imgpath[subject_idx2:subject_idx2+4]
                subid_int = int(subid[1:]) + 23
            self.subID.append(subid_int)
        print("  subID len:", len(self.subID))  # subID: (total_frames)
        

    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):
        # get both frame image and neutral-face image
        data_numpy = np.array(self.data[index, :, :, :])
        label = np.array(self.label[index])
        subid = np.array(self.subID[index])
        # print('data_numpy:', data_numpy.shape, 'label:', label.shape)   # data(2,1,22) label:(12,)

        if self.image:
            image = []
            imgpath = self.imagepaths[index].replace('/home/lair/data', '/Extra/Dataset')
            # print('imgpath:', imgpath)
            img = cv2.imread(imgpath)
            face = cv2.resize(img, (self.image_size, self.image_size))
            face = face.transpose((2, 0, 1))
            
            sub_idx = 0
            if subid <= 23:
                sub_idx = imgpath.find('F')
            else:
                sub_idx = imgpath.find('M')
            neutral_path = imgpath[:sub_idx+4] + '/T3/001.jpg'
            # print('neutral_path:', neutral_path)

            neutral_img = cv2.imread(neutral_path)
            neutral_face = cv2.resize(neutral_img, (self.image_size, self.image_size))
            neutral_face = neutral_face.transpose((2, 0, 1))
                
            image = [neutral_face, face]
                
        image = np.array(image)
        # print('image:', image.shape) # image:(2,3,256,256)
        
        if self.imagepath:
            imagepath = self.imagepaths[index]

        # processing data
        # if self.random_choose:
        #     data_numpy = feeder_utils.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = feeder_utils.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = feeder_utils.random_move(data_numpy)

        if self.image and not self.imagepath:
            return data_numpy, label, image, subid  # add subid
        elif self.image and self.imagepath:
            return data_numpy, label, image, imagepath
        else:
            return data_numpy, label