from operator import index
import numpy as np
import torch
import glob
import os.path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2
import torch.nn as nn

from seed import setup_seed
from data_func import *
setup_seed()

import numpy as np
class dice_coefficient(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(dice_coefficient, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.shape[0]
        logits[logits>=0.5] = 1
        logits[logits<0.5] = 0
        logits = logits.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)
        intersection = (logits * targets).sum(-1)
#         dice_score = 2. * intersection + self.epsilon / ((logits + targets).sum(-1) + self.epsilon)
#         dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        dice_score = (2. * intersection + self.epsilon) / ((logits + targets).sum(-1) + self.epsilon)
#         print(dice_score)
        return np.mean(dice_score)

class Data:
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        # self.unlabeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def supervised_training_labels(self):
        # used for supervised learning baseline, put all data labeled
        tmp_idxs = np.arange(self.n_pool)
        self.labeled_idxs[tmp_idxs[:]] = True

    def initialize_labels_random(self, num):
        # generate initial labeled pool
        # use idx to distinguish labeled and unlabeled data取1000张有target的sample
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        count = 0
        for i in tmp_idxs:
            if np.sum(self.Y_train[i])!=0:
                self.labeled_idxs[i] = True
                count+=1
                if count == num:
                    break

    def initialize_labels_K(self, k):# 每个病人有多少slice
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        start_idx = 0
        cumulative_counts = []
        cumulative_sum = 0
        selected_slices = np.arange(self.n_pool, dtype=int)[::k]
        self.labeled_idxs[selected_slices] = True
        print(len(selected_slices))
        # print(self.labeled_idxs)
        # for num_slices in num_slices_per_patient:
        #     num_full_segments = (num_slices // k) + 1
        #     last_segment_size = num_slices % k
        #     selected_slices = [start_idx + k*j for j in range(num_full_segments)]
        #     start_idx += num_slices
        #     # selected_slices.append(start_idx-1)  # Add the last slice
        #     self.labeled_idxs[selected_slices] = True #[0, 20, 40, 60, 80, 100, 120, 140]

        return len(np.arange(self.n_pool, dtype=int)[self.labeled_idxs]) # 8*50

    # def initialize_labels(self, dataset, num): #最开始initialize过程 随机选
    #     # generate initial labeled pool 根据label index来确认 labeled_idxs就是label data反之是unlabel data（x_train+y_train)
    #     tmp_idxs = np.arange(self.n_pool)
    #     np.random.shuffle(tmp_idxs)
    #     net_init = get_net(args.dataset_name, device, init=True)
    #     strategy_init = get_strategy("KMeansSampling")(dataset, net_init, init=True)
    #     rd = 0
    #     strategy_init.train(rd, init=True)
    #     query_idxs = strategy_init.query(args.n_init_labeled)
    #     strategy.update(query_idxs)

    def get_labeled_data(self):
        # get labeled data for training
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
        # print("labeled data", labeled_idxs.shape)
        # print("labeled_idxs ", labeled_idxs)
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],mode="train")

    

    def delete_black_patch(self, unlabeled_idxs, label):
        # unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#(24537)
        index = []
        # used for generated adversarial image expansion. Adding generated adversarial image with label to training dataset
        # print("label",label.shape)
        for i in range(label.shape[0]):#24537
            if torch.sum(label[i])==0:
                index.append(unlabeled_idxs[i])
        return index
        # self.X_train= np.delete(self.X_train, index, 0)
        # print("X_train",self.X_train.shape)
        # self.Y_train = np.delete(self.Y_train, index, 0)
        # unlabeled_idxs = np.delete(unlabeled_idxs, index)
        # print(unlabeled_idxs)
        # self.n_pool = len(self.X_train)
        # print("n_pool",self.n_pool)

    def get_unlabeled_data(self):
        # get unlabeled data for active learning selection process
        unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#24537
        # print("unlabeled_idxs",unlabeled_idxs.shape)

        # if index!=None:
        #     self.labeled_idxs[index] = True #5486
        #     unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#19051 19255
        #     self.labeled_idxs[index] = False
            # print("unlabeled_idxs",unlabeled_idxs.shape)
            # print("labeled_idxs",len(self.labeled_idxs))#25537 不对 应该是1k
            # unlabeled_idxs = np.arange(self.n_pool, dtype=int)
            # # exclude_index = np.concatenate((self.labeled_idxs,index))
            # print("exclude_index",len(exclude_index))
            # # unlabeled_idxs = np.delete(unlabeled_idxs, exclude_index)
            # print("deleted_unlabeled_idxs",unlabeled_idxs[-30:])
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs],mode="val")

    def get_val_data(self):
        # get validation dataset if exist
        return self.handler(self.X_val, self.Y_val,mode="val")

    def get_test_data(self):
        # get test dataset if exist
        return self.handler(self.X_test, self.Y_test,mode="val")

    def cal_test_acc(self, logits, targets):
        # calculate accuracy for test dataset
        # 各类指标
        dscs = []
        dice_coeff = dice_coefficient()
        dsc = dice_coeff(targets, logits)
        return dsc

    def cal_train_acc(self, preds):
        # calculate accuracy for train dataset for early stopping
        return 1.0 * (self.Y_train == preds).sum().item() / self.n_pool

    def add_labeled_data(self, data, label):
        # used for generated adversarial image expansion. Adding generated adversarial image with label to training dataset
        data = torch.reshape(data, (len(data),1,128,128))
        # data = torch.unsqueeze(data, 1)
        self.X_train = torch.tensor(self.X_train)#([25537, 128, 128])
        self.Y_train = torch.tensor(self.Y_train)
        self.X_train = torch.cat((self.X_train, data), 0)#([26037, 128, 128])
        self.Y_train = torch.cat((self.Y_train, label), 0)
        # print("labeled_idxs",self.labeled_idxs.shape)
        array = np.ones(len(data),dtype=bool)
        self.labeled_idxs = np.append(self.labeled_idxs, array)
        # print("changed_labeled_idxs",self.labeled_idxs.shape)

        self.n_pool += len(data)

    def update_pseudo_label(self, idx, label):
        # used for pseudo labeling, change the correct label to pseudo label
        self.X_train = torch.tensor(self.X_train)
        self.Y_train[idx] = label

    #     def (self,data,label):
    #         data = tadd_labeled_dataorch.reshape(data, (-1,512,512,3))
    # #         data = torch.unsqueeze(data, 0)
    #         self.X_train = torch.tensor(self.X_train)
    #         self.Y_train = torch.tensor(self.Y_train)
    #         self.X_train = torch.cat((self.X_train, data), 0)
    #         self.Y_train = torch.cat((self.Y_train, label), 0)
    #         for i in range(len(data)):
    #             self.labeled_idxs=np.append(self.labeled_idxs,True)
    #             self.n_pool+=1

    def get_label(self, idx):
        # Get the real label (share lable) for adversarial samples
        self.Y_train = np.array(self.Y_train)
        label = torch.tensor(self.Y_train[idx])
        return label

    # # efficient training method
    # def get_efficient_training_data(self, idx, new_idx):
    #     all_idx = np.concatenate((idx, new_idx), axis=None)
    #     labeled_idxs = np.arange(self.n_pool)[all_idx]
    #     return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])


def get_MSSEG(handler,supervised = False):
    #both 2d and 3d 
    # train_dir_name = "../MSSEG/Training/"
    # test_dir_name = "../MSSEG/Testing/"

    # # ps=get_path(train_dir_name)
    # # train_images_path = np.stack([name for name in [
    # # #     [os.path.join(train_dir_name + patient + '/Raw_Data/FLAIR.nii.gz') for patient in ps],
    # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/FLAIR_preprocessed.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/DP_preprocessed.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/T2_preprocessed.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/T1_preprocessed.nii.gz') for patient in ps]
    # # ] if name is not None], axis=1)

    # # train_masks_path = np.stack([name for name in [
    # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # ] if name is not None], axis=1)

    # # train_brain_masks_path = np.stack([name for name in [
    # #     [os.path.join(train_dir_name + patient + '/Masks/Brain_Mask.nii.gz') for patient in ps],
    # # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # ] if name is not None], axis=1)

    # ps=get_path(test_dir_name)
    # test_images_path = np.stack([name for name in [
    # #     [os.path.join(test_dir_name + patient + '/Raw_Data/FLAIR.nii.gz') for patient in ps],
    #     [os.path.join(test_dir_name + patient + '/Preprocessed_Data/FLAIR_preprocessed.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Preprocessed_Data/T2_preprocessed.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Preprocessed_Data/T1_preprocessed.nii.gz') for patient in ps]
    # ] if name is not None], axis=1)

    # test_masks_path = np.stack([name for name in [
    #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # ] if name is not None], axis=1)

    # test_brain_masks_path = np.stack([name for name in [
    #     [os.path.join(test_dir_name + patient + '/Masks/Brain_Mask.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # ] if name is not None], axis=1)
   
   
    # # train_images = get_image(train_images_path,label=False)
    # # train_masks = get_image(train_masks_path,label=True) 
    # # train_brain_masks = get_image(train_brain_masks_path,label=True)

    # test_images = get_image(test_images_path,label=False)  
    # test_masks = get_image(test_masks_path,label=True)
    # test_brain_area_masks = get_image(test_brain_masks_path,label=True)

    # # print(np.array(train_images).shape)
    # # print(np.array(train_masks).shape)
    # # print(np.array(train_brain_masks).shape)

    # print(np.array(test_images).shape)
    # print(np.array(test_masks).shape)
    # print(np.array(test_brain_area_masks).shape)

    # # train_brain_images,train_brain_masks, train_brain_area_masks = get_brain_area(train_images,train_brain_masks,train_masks)
    # # print(train_brain_images.shape)
    # # print(train_brain_masks.shape)

    # test_brain_images,test_brain_masks,test_brain_area_masks = get_brain_area(test_images,test_brain_area_masks,test_masks)
    # print(test_brain_images.shape)
    # print(test_brain_masks.shape)
    # print(test_brain_area_masks.shape)

    # # #2d
    # # #切分2d slice
    # # x_train_slice = get_2d_slice(train_brain_images,train_brain_masks,restrict=True)
    # # y_train_slice = get_2d_slice(train_brain_masks,train_brain_masks,restrict=True)
    # # print(x_train_slice.shape,y_train_slice.shape)

    # # # x_val_slice = get_2d_slice(val_x,val_y,restrict=False)
    # # # y_val_slice = get_2d_slice(val_y,val_y,restrict=False)
    # # # print(x_val.shape,y_val.shape)

    # x_test_slice = get_2d_slice(test_brain_images,test_brain_masks,restrict=False)
    # y_test_slice = get_2d_slice(test_brain_masks,test_brain_masks,restrict=False)
    # print(x_test_slice.shape,y_test_slice.shape)

    # # #为切分2d patch 防止有的无法整除
    # # full_train_imgs_list = paint_border_overlap(x_train_slice,stride=32)
    # # print(np.array(full_train_imgs_list).shape)
    # # full_train_masks_list = paint_border_overlap(y_train_slice,stride=32)
    # # print(np.array(full_train_masks_list).shape)

    # # # full_val_imgs_list = paint_border_overlap(x_val_slice,stride=64)
    # # # print(np.array(full_val_imgs_list).shape)
    # # # full_val_masks_list = paint_border_overlap(y_val_slice,stride=64)
    # # # print(np.array(full_val_masks_list).shape)

    # full_test_imgs_list = paint_border_overlap(x_test_slice,stride=96)
    # print(np.array(full_test_imgs_list).shape)
    # full_test_masks_list = paint_border_overlap(y_test_slice,stride=96)
    # print(np.array(full_test_masks_list).shape)

    # # #得到64*64 2d patch
    # # x_train,y_train = extract_ordered_overlap(np.array(full_train_imgs_list),label=full_train_masks_list,stride=32,train=True)
    # # print(np.array(x_train).shape,np.array(y_train).shape)

    # x_test,y_test = extract_ordered_overlap(np.array(full_test_imgs_list),label=full_test_masks_list,stride=96,train=False)
    # print(np.array(x_test).shape,np.array(y_test).shape)

    # train_x,val_x,train_y,val_y = train_test_split(x_train,y_train,test_size=0.2,random_state=42)
    # print(np.array(train_x).shape)
    # print(np.array(val_x).shape)




    # 3d 补充原本的图像
    # full_train_imgs_list = paint_border_overlap_3d(train_x,stride=16)
    # print(np.array(full_train_imgs_list).shape)
    # full_train_masks_list = paint_border_overlap_3d(train_y,stride=16)
    # print(np.array(full_train_masks_list).shape)

    # full_val_imgs_list = paint_border_overlap_3d(val_x,stride=48)
    # print(np.array(full_val_imgs_list).shape)
    # full_val_masks_list = paint_border_overlap_3d(val_y,stride=48)
    # print(np.array(full_val_masks_list).shape)

    # full_imgs_list = paint_border_overlap_3d(test_images,stride=48)
    # print(np.array(full_imgs_list).shape)
    # full_masks_list = paint_border_overlap_3d(test_masks,stride=48)
    # print(np.array(full_masks_list).shape)

    # # x_train = extract_ordered_overlap_3d(np.array(full_train_imgs_list),label=full_train_masks_list,stride=16,train=True)
    # # print(np.array(x_train).shape)
    # # y_train = extract_ordered_overlap_3d(np.array(full_train_masks_list),label=full_train_masks_list,stride=16,train=True)
    # # print(np.array(y_train).shape)

    # # x_val = extract_ordered_overlap_3d(np.array(full_val_imgs_list),label=full_val_masks_list,stride=48,train=False)
    # # print(np.array(x_val).shape)
    # # y_val = extract_ordered_overlap_3d(np.array(full_val_masks_list),label=full_val_masks_list,stride=48,train=False)
    # # print(np.array(y_val).shape)

    # # x_test = extract_ordered_overlap_3d(np.array(full_imgs_list),stride=48,train=False)
    # # print(np.array(x_test).shape)
    # # y_test = extract_ordered_overlap_3d(np.array(full_masks_list),stride=48,train=False)
    # # print(np.array(y_test).shape)

    # x_train = train_x
    # y_train = train_y
    # x_val = val_x
    # y_val = val_y


    # x_train = torch.load('../MSSEG/x_train_2d.pt')
    # y_train = torch.load('../MSSEG/y_train_2d.pt')
    # x_val = torch.load('../MSSEG/x_val_2d.pt')
    # y_val = torch.load('../MSSEG/y_val_2d.pt')
    # x_test = torch.load('../MSSEG/x_test_2d.pt')
    # y_test = torch.load('../MSSEG/x_test_2d.pt')

    # x_train = np.load('/home/siteng/3D_analysis/BraTS2019/train_image.npy')
    # y_train = np.load('/home/siteng/3D_analysis/BraTS2019/train_label.npy')
    # x_val = np.load('/home/siteng/3D_analysis/BraTS2019/val_image.npy')
    # y_val = np.load('/home/siteng/3D_analysis/BraTS2019/val_label.npy')
    # x_test = np.load('/home/siteng/3D_analysis/BraTS2019/test_image.npy')
    # y_test = np.load('/home/siteng/3D_analysis/BraTS2019/test_label.npy')

    x_train = np.load('../Task09_Spleen/train_image.npy')
    y_train = np.load('../Task09_Spleen/train_label.npy')
    x_val = np.load('../Task09_Spleen/val_image.npy')
    y_val = np.load('../Task09_Spleen/val_label.npy')
    x_test = np.load('../Task09_Spleen/test_image.npy')
    y_test = np.load('../Task09_Spleen/test_label.npy')

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, handler
