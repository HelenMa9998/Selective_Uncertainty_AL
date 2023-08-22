import math
import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nib
from PIL import Image
from .strategy import Strategy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.util import change_root
import numpy as np
import torch
from .strategy import Strategy
from tqdm import tqdm

# This is our proposed method
# including pseudo labeling expansion, generated sample expansion and adaptive selection

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

criterion = FocalLoss(alpha=0.9, gamma=2,logits=False)

def count(nx):
    zeros=0
    ones=0
    for i in nx:
        if i==0:
            zeros += 1
        elif i==1:
            ones += 1
    return zeros,ones

class AdversarialAttack(Strategy):
    def __init__(self, dataset, net, eps=0.05, max_iter=1):
        super(AdversarialAttack, self).__init__(dataset, net)
        self.eps = eps
        self.max_iter = max_iter

    def cal_dis(self, x, unlabeled_idxs):
        nx = torch.unsqueeze(x, 0)
        # nx = torch.unsqueeze(nx, 0)
        nx = nx.cuda()
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)
        eta = eta.cuda()
        out = self.net.clf(nx+eta)
        out_copy = out.clone()
        out_binary = (out_copy > 0.5).int() # pred pseudo labels
        py = out_binary
        ny = out_binary
        i_iter = 0
        change_pixel_num = 0
        # while change_pixel_num < 10 and i_iter < self.max_iter:

        while i_iter < self.max_iter:#assign iterations
            # print("i_iter",i_iter)
            # loss = F.binary_cross_entropy(out.float(), ny.float())
            loss = criterion(out.float(), ny.float())
            
            # torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            # model-based uncertainty
            eta += self.eps * nx.grad.data/nx.grad.data.max()# using the size and orientation of gradient
            # print("eta",eta)
            nx.grad.data.zero_()
            # data-based stability
            out_change = self.net.clf(nx + eta)

            out_copy_change = out_change.clone()
            # py = (out_copy_change > 0.5).int()
            # print(out_copy)
            # print(out_copy_change)
            # change_pixel = torch.ne(py.flatten(),ny.flatten())
            change_pixel = torch.ne(out_copy_change.flatten(),out_copy.flatten())
            change_pixel_num = change_pixel.tolist().count(True)
            # print("i_iter",i_iter,"change_pixel_num",change_pixel_num)
            # print("(eta*eta).sum()",(eta*eta).sum())
            i_iter += 1
        # generated adversarial sample
        image = (nx + eta).cpu().detach()  

        return (eta*eta).sum(), change_pixel_num, image


    def query(self, n, index):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(index = index)
        print("unlabeled_idxs",len(unlabeled_idxs))
        self.net.clf = torch.load('./result/model.pth')
        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)
        changed = np.zeros(unlabeled_idxs.shape)
        generated_image = {}
        image = torch.zeros((len(unlabeled_idxs),1,128,128))#([1, 1, 128, 128])
        final_image = torch.zeros((500,1,128,128))
        diction = {}
        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):#从1开始
            x, y, idx = unlabeled_data[i]
            # dis[i] = self.cal_dis(x)
            dis[i],changed[i],image[i] = self.cal_dis(x,i)
            generated_image[unlabeled_idxs[i]] = image[i]

        idx_dis = np.array(unlabeled_idxs[dis.argsort()])# model-based index
        idx_changed = np.array(unlabeled_idxs[changed.sort()][0])#data-based index
        # combine model- and data-based index
        for i in range(len(idx_dis)):
            # print(idx_changed[i])
            # print(np.where(idx_dis==idx_changed[i])[0])#[2626 2734 5976]
            position = int(np.where(idx_changed==idx_dis[i])[0])+i
            diction[position] = idx_dis[i]

        dict_sort = sorted(diction.keys())
        keys = dict_sort[:n]
        final = [value for key, value in diction.items() if key in keys]
        # print("final_max",list(final).max())
        for i in range(len(final)):
            final_image[i] = generated_image[final[i]]
        # print(final)
        return final, final_image