import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import TensorDataset,DataLoader,Dataset
from monai import transforms
from seed import setup_seed
import torch
import albumentations as A


#data augumentation
setup_seed()
# get dataloader
class Messidor_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomRotation(15),
                                             transforms.RandomResizedCrop(size=512, scale=(0.9, 1)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.uint8(x))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

#data augumentation

keys = ("image", "label")
class aug():
    def __init__(self):
        self.random_rotated = transforms.Compose([
            transforms.AddChanneld(keys),  # 增加通道，monai所有Transforms方法默认的输入格式都是[C, W, H, ...],第一维一定是通道维
            transforms.RandRotate90d(keys, prob=1, max_k=3, spatial_axes=(0, 1), allow_missing_keys=False),
            transforms.RandFlipd(keys, prob=1, spatial_axis=(0, 1), allow_missing_keys=False),
            transforms.RandGaussianNoised(keys, prob=0.1, mean=0.0, std=0.1, allow_missing_keys=False),
#             transforms.NormalizeIntensityd(keys, allow_missing_keys=False),
            transforms.ToTensord(keys)
        ])
    
    def forward(self,x):
        x = self.random_rotated(x)
        return x
    
# class MSSEG_Handler(Dataset):
#     def __init__(self,image,label,mode="train"):
#         self.image=image
#         self.label=label
        
#         if mode=="train":
#             self.transform = True        
#         else:
#             self.transform=None
#     def __len__(self):
#         return len(self.label)
#     def __getitem__(self,index):#通过index获取某个数据
#         img = self.image[index]
#         label = self.label[index]
#         # if self.transform:
#         #     m = {"image": img,"label":label}
#         #     augs = aug()
#         #     data_dict = augs.forward(m)
#         #     img = data_dict["image"]
#         #     label = data_dict["label"]
#         return img,label, index

class MSSEG_Handler_2d(Dataset):
    def __init__(self,image,label,mode="train"):
        self.image=np.array(image)
        self.label=np.array(label)
        if mode=="train":
            self.transform = A.Compose([
                A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, always_apply=False, p=0.5),
                A.Flip(p=0.5),
                A.Rotate (limit=90, interpolation=1,always_apply=False, p=0.5),
#                 A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
#                 A.Resize(336,336, interpolation=3, always_apply=True, p=1),
                # A.ToTensor(),
        ]) 
        else:
            self.transform=None
            
    def __len__(self):
        return len(self.label)
    def __getitem__(self,index):#通过index获取某个数据
        img = self.image[index].astype(np.float32)
        label = self.label[index].astype(np.uint8)
        if self.transform!=None: 
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
        img = torch.tensor(img)
        label = torch.tensor(label).unsqueeze(0)
        return img, label, index



