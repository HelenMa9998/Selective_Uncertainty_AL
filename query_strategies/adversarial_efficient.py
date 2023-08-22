import math
import numpy as np
import torch
from .strategy import Strategy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from distutils.util import change_root
import numpy as np
import torch
from .strategy import Strategy
from tqdm import tqdm

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

criterion = FocalLoss(alpha=0.97, gamma=2,logits=False)

def predict_prob(data,net):
    net = torch.load('./model.pth')
    net.eval()
    probs = torch.zeros([len(data), 1, 128, 128])
    loader = DataLoader(data, shuffle=False, **self.params['test_args'])
    with torch.no_grad():
        for x, y, idxs in loader:
            x, y = x.unsqueeze(1), y.unsqueeze(1)
            x, y = x.cuda(), y.cuda()
            x.requires_grad_()
            prob = net(x) # torch.Size([8, 2, 64, 64, 64])
            probs[idxs] = prob.cpu() 
            eta = torch.zeros(x.shape)
            eta = eta.cuda()
    return probs,x,eta

class AdversarialAttack_efficient(Strategy):
    def __init__(self, dataset, net, eps=0.05, max_iter=50):
        super(AdversarialAttack_efficient, self).__init__(dataset, net)
        self.eps = eps
        self.max_iter = max_iter
        self.net = net

    def cal_dis(self, nx):
        out,nx,eta = predict_prob(nx,self.net)#([12384, 1, 128, 128])
        out_binary = out.clone()
        out_binary = (out_binary > 0.5).int() 
        py = out_binary#segmentation predicted by network batchsize,w,h,d ([1, 1, 128, 128])
        ny = out_binary#([1, 1, 128, 128])
        i_iter = 0
        change_pixel_num = np.zeros(nx.shape)
        while i_iter < self.max_iter:#设置最多次iter
            # loss = F.binary_cross_entropy(out.float(), ny.float())
            loss = criterion(out.float(), ny.float())
            # torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            # print("grad",nx.grad.data)
            norm = torch.norm(nx.grad.data)
            # eta = self.eps * nx.grad.data/norm#用了gradient的大小+符号
            eta += self.eps * torch.sign(nx.grad.data)#nx.grad.data是10-8量级 -9 -10 只是用了符号
            nx.grad.data.zero_()
            out = self.predict_prob(nx+eta)#应该是([12384, 1, 128, 128])
            out_binary_change = out.clone()
            py = (out_binary_change > 0.5).int()
            for i in range(py.shape[0]): 
                change_pixel = torch.ne(py[i].flatten(),ny[i].flatten())#整体的改变的pixel
                change_pixel_num[i] = change_pixel.tolist().count(True)
            # print("i_iter",i_iter,"change_pixel_num",change_pixel_num)
            i_iter += 1
        return (eta*eta).sum()


    def query(self, n, index):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(index = index)
        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)
        # x, y, idx = unlabeled_data
        dis = self.cal_dis(unlabeled_data)
        return unlabeled_idxs[dis.argsort()[:n]]