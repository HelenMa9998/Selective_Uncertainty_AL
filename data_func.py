import nibabel as nib
import os
import numpy as np
import cv2
import torch
import SimpleITK as sitk

from seed import setup_seed

setup_seed()
#both 2d and 3d 
def get_path(dir_name):
    centers = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    ps = []
    # print(centers)
    for i in centers:
    #     print(dir_name+i)
        patients = [f for f in sorted(os.listdir(dir_name+i)) if os.path.isdir(os.path.join(dir_name+i, f))]
        for j in patients:
            ps.append(i+"/"+j)
    return ps

def get_image(image_path,label=False):
    images = []
    for i in range(len(image_path)):
        for j in range(len(image_path[i])):
            itk_img = sitk.ReadImage(image_path[i][j])
            image = sitk.GetArrayFromImage(itk_img)
            images.append(image)
    return np.array(images)


# 预处理 只获取大脑部分
import matplotlib.pyplot as plt
def get_brain_area(images,masks,gts):
    brain_images = []
    brain_masks = []
    brain_area_masks = []
    for x in range(np.array(images).shape[0]):
        image = images[x]
        mask = masks[x]
        gt = gts[x]
#         print(mask.shape)
        target_indexs = np.where(mask == 1)
        w_maxs = np.max(np.array(target_indexs[0]))
        w_mins = np.min(np.array(target_indexs[0]))
        h_maxs = np.max(np.array(target_indexs[1]))
        h_mins = np.min(np.array(target_indexs[1]))
        d_maxs = np.max(np.array(target_indexs[2]))
        d_mins = np.min(np.array(target_indexs[2]))
#         print(w_maxs,w_mins,h_maxs,h_mins,d_maxs,d_mins)
        brain_image = image[w_mins:w_maxs, h_mins:h_maxs, d_mins:d_maxs]
        brain_mask = gt[w_mins:w_maxs, h_mins:h_maxs, d_mins:d_maxs]
        brain_area_mask = mask[w_mins:w_maxs, h_mins:h_maxs, d_mins:d_maxs]
#         print(brain_image.shape)
        brain_images.append(brain_image)
        brain_masks.append(brain_mask)
        brain_area_masks.append(brain_area_mask)
    return np.array(brain_images),np.array(brain_masks),np.array(brain_area_masks)

#切 2d slice
def get_2d_slice(images,labels,restrict=True):
    slices = []
    for x in range(np.array(images).shape[0]):
#         print(images[x].shape)
        for n_slice in range(np.array(images[x]).shape[0]):
            cbct_slice = images[x][n_slice,:,:]
            imgtlabel = labels[x][n_slice,:,:]
            prob = np.sum(imgtlabel)/(imgtlabel.shape[0]*imgtlabel.shape[1])
            if restrict==True:
                if prob==0:# 不加入
                    continue
            slices.append(np.array(cbct_slice))
    return np.array(slices)

#get 2d overlap patches
def paint_border_overlap(full_imgs_all,patch_size=[128,128],stride=32):
    patch_w,patch_h = patch_size
    full_imgs_list = []
    for full_imgs in full_imgs_all:
        img_w = full_imgs.shape[0] #width of the image
        img_h = full_imgs.shape[1]  #height of the image

        leftover_w = (img_w-patch_w)%stride  #leftover on the w dim
        leftover_h = (img_h-patch_h)%stride  #leftover on the h dim
    
        if (leftover_w != 0):   #change dimension of img_w
#             print("the side W is not compatible with the selected stride of " +str(stride))
#             print("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
#             print("So the W dim will be padded with additional " +str(stride - leftover_w) + " pixels")
            tmp_full_imgs = np.zeros((img_w+(stride - leftover_w),img_h))
            tmp_full_imgs[0:img_w,0:img_h] = full_imgs
            full_imgs = tmp_full_imgs#in（144,512,512） out(160, 512, 512)
            
        if (leftover_h != 0):  #change dimension of img_h
#             print("\nthe side H is not compatible with the selected stride of " +str(stride))
#             print("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
#             print("So the H dim will be padded with additional " +str(stride - leftover_h) + " pixels")
            tmp_full_imgs = np.zeros((full_imgs.shape[0],img_h+(stride - leftover_h)))
            tmp_full_imgs[0:full_imgs.shape[0],0:img_h] = full_imgs
            full_imgs = tmp_full_imgs#out (160, 520, 512)
            
        full_imgs_list.append(full_imgs)
#         print("new padded images shape: " +str(full_imgs.shape))
    return full_imgs_list


def extract_ordered_overlap(full_imgs_all,label=None,patch_size=[128,128],stride=32,train=True):
    patch_w,patch_h = patch_size
#     w,h,d = patch_size
    full_imgs_list = []
    patches = []
    label_patches = []
    target_center = 0
    non_target_center = 0
    for x in range(np.array(full_imgs_all).shape[0]):
        full_imgs = full_imgs_all[x]
        img_w = full_imgs.shape[1] #width of the image
        img_h = full_imgs.shape[0]  #height of the image
        
        for j in range((img_h-patch_h)//stride+1):    
            for i in range((img_w-patch_w)//stride+1):
                    imgt = full_imgs[j*stride:j*stride+patch_h,i*stride:i*stride+patch_w]
                    imgtlabel = label[x][j*stride:j*stride+patch_h,i*stride:i*stride+patch_w]
                    
                    if train == True: 
                        prob = np.sum(imgtlabel)/(imgtlabel.shape[0]*imgtlabel.shape[1])
                        if prob==0:# 不加入
                            continue
                            
                        patches.append(imgt)
                        label_patches.append(imgtlabel)
                    else: 
                        patches.append(imgt)
                        label_patches.append(imgtlabel)
    return patches,label_patches  #array with all the full_imgs divided in patches


#3D function
#判断原图是否能完整切patch 不够的话补上
def paint_border_overlap_3d(full_imgs_all,patch_size=[64,64,64],stride=16):
    patch_w,patch_h,patch_d = patch_size
    full_imgs_list = []
    for full_imgs in full_imgs_all:
        img_w = full_imgs.shape[0] #width of the image
        img_h = full_imgs.shape[1]  #height of the image
        img_d = full_imgs.shape[2]  #depth of the image

        leftover_w = (img_w-patch_w)%stride  #leftover on the w dim
        leftover_h = (img_h-patch_h)%stride  #leftover on the h dim
        leftover_d = (img_d-patch_d)%stride  #leftover on the h dim
    
        if (leftover_w != 0):   #change dimension of img_w
#             print("the side W is not compatible with the selected stride of " +str(stride))
#             print("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
#             print("So the W dim will be padded with additional " +str(stride - leftover_w) + " pixels")
            tmp_full_imgs = np.zeros((img_w+(stride - leftover_w),img_h,img_d))
            tmp_full_imgs[0:img_w,0:img_h,0:img_d] = full_imgs
            full_imgs = tmp_full_imgs#in（144,512,512） out(160, 512, 512)
            
        if (leftover_h != 0):  #change dimension of img_h
#             print("\nthe side H is not compatible with the selected stride of " +str(stride))
#             print("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
#             print("So the H dim will be padded with additional " +str(stride - leftover_h) + " pixels")
            tmp_full_imgs = np.zeros((full_imgs.shape[0],img_h+(stride - leftover_h),img_d))
            tmp_full_imgs[0:full_imgs.shape[0],0:img_h,0:img_d] = full_imgs
            full_imgs = tmp_full_imgs#out (160, 520, 512)
            
        if (leftover_d != 0):   #change dimension of img_w
#             print("the side W is not compatible with the selected stride of " +str(stride))
#             print("(img_w - patch_w) MOD stride_w: " +str(leftover_d))
#             print("So the W dim will be padded with additional " +str(stride - leftover_d) + " pixels")
            tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_d+(stride - leftover_d)))
            tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_d] = full_imgs
            full_imgs = tmp_full_imgs
        full_imgs_list.append(full_imgs)
#         print("new padded images shape: " +str(full_imgs.shape))
    return full_imgs_list


#overlap + 百分比判断
def extract_ordered_overlap_3d(full_imgs_all,label=None,patch_size=[64,64,64],stride=16,train=True):
    patch_w,patch_h,patch_d = patch_size
#     w,h,d = patch_size
    full_imgs_list = []
    patches = []
    for x in range(np.array(full_imgs_all).shape[0]):
        full_imgs = full_imgs_all[x]
        img_w = full_imgs.shape[0] #width of the image
        img_h = full_imgs.shape[1]  #height of the image
        img_d = full_imgs.shape[2]  #depth of the image
        for i in range((img_w-patch_w)//stride+1):
            for j in range((img_h-patch_h)//stride+1):
                for k in range((img_d-patch_d)//stride+1):
                    imgt = full_imgs[i*stride:i*stride+patch_w,j*stride:j*stride+patch_h,k*stride:k*stride+patch_d]
                    if train == True: 
                        imgtlabel = label[x][i*stride:i*stride+patch_w,j*stride:j*stride+patch_h,k*stride:k*stride+patch_d]
                        prob = np.sum(imgtlabel)/(imgtlabel.shape[0]*imgtlabel.shape[1]*imgtlabel.shape[2])
                        if prob<0.005:# 不加入
                            continue
                        patches.append(imgt)
                    else: 
                        patches.append(imgt)
    return patches  #array with all the full_imgs divided in patches


