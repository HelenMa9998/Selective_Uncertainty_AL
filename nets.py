from turtle import shape
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from collections import OrderedDict
from tqdm import tqdm
from seed import setup_seed

setup_seed()
# used for getting prediction results for data
def recompone_overlap(preds,full_imgs_list,test_images,stride=96, image_num=8251):
    patch_w = preds.shape[2] # pred (157341, 64, 64)
    patch_h = preds.shape[1]
#     print(preds.shape)
    final_full_prob = []
    a = 0
    for x in range(image_num):
        input_array = full_imgs_list[x] # 
        # print(input_array.shape)
        ori_image = test_images[x] # 
        # print(ori_image.shape)
        img_h = np.array(input_array).shape[0]
        img_w = np.array(input_array).shape[1]
        
        full_prob = np.zeros((img_h,img_w)) #(352, 160)
        full_sum = np.zeros((img_h,img_w))#

        for i in range((img_h-patch_h)//stride+1):
            for j in range((img_w-patch_w)//stride+1):
                full_prob[i*stride:(i*stride)+patch_h,j*stride:(j*stride)+patch_w]+=preds[a] # Accumulate predicted values
                full_sum[i*stride:(i*stride)+patch_h,j*stride:(j*stride)+patch_w]+=1  # Accumulate the number of predictions
                a+=1
        final_avg = full_prob/full_sum # Take the average
        final_avg = final_avg[0:ori_image.shape[0],0:ori_image.shape[1]]
        # print("final_avg",final_avg.max())
        assert(np.max(final_avg)<=1.0) # max value for a pixel is 1.0
        assert(np.min(final_avg)>=0.0) # min value for a pixel is 0.0
        final_avg[final_avg>=0.5] = 1
        final_avg[final_avg<0.5] = 0
        final_full_prob.append(final_avg)
#         print("final_full_prob",np.array(final_full_prob).shape)
    return final_full_prob       

def recompone_overlap_3d(preds, test_images, image_num=38):
    final_full_prob = []
    a = 0
    for x in range(image_num):
        input_array = test_images[x]
        img_d = np.array(input_array).shape[0]
        img_h = np.array(input_array).shape[1]
        img_w = np.array(input_array).shape[2]
        full_prob = np.zeros((img_d,img_h,img_w))
        for i in range(img_d):
#             print(preds[a].max())
            full_prob[i]=preds[a]
            a+=1
        assert(np.max(full_prob)<=1.0) # max value for a pixel is 1.0
        assert(np.min(full_prob)>=0.0) # min value for a pixel is 0.0
        final_full_prob.append(full_prob)
    return final_full_prob

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

class dice_coefficient(nn.Module):
    def __init__(self, epsilon=0.0001):
        super(dice_coefficient, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.shape[0]
        logits = (logits > 0.5).float()
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
#         dice_score = 2. * (intersection + self.epsilon) / ((logits + targets).sum(-1) + self.epsilon)
        dice_score = (2. * intersection+ self.epsilon) / ((logits.sum(-1) + targets.sum(-1)) + self.epsilon)
        return torch.mean(dice_score)

# including different training method for active learning process (train acc=1, val loss, val acc, epoch)
class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    # epoch as a stopping criterion
    
    # Validation loss as a stopping criterion
    def supervised_val_loss(self, data, val_data,rd):
        n_epoch = 100
        trigger = 0
        best = {'epoch': 1, 'loss': 10}
        train_loss=0
        validation_loss = 0
        train_dice=0
        val_dice=0
        self.clf = self.net().to(self.device)
        self.clf.train()
        if rd==0:
            self.clf = self.clf
        else:
            self.clf = torch.load('./result/model.pth')
        # optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        #         optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        #         optimizer = optim.SGD(self.clf.parameters(), momentum=0.9, lr=0.0003, weight_decay=0.01, nesterov=True)
        optimizer = optim.Adam(self.clf.parameters(), lr=0.0001)
        criterion = nn.BCELoss()
        # criterion = FocalLoss(alpha=0.7, gamma=2,logits=False)
        # loader = DataLoader(data, shuffle=True, **self.params['train_args'],num_workers=0,drop_last=False)
        # val_loader = DataLoader(val_data, shuffle=False, **self.params['val_args'],num_workers=0,drop_last=False)

        loader=DataLoader(data, **self.params['train_args'])
        val_loader=DataLoader(val_data, **self.params['val_args'])

        dice = dice_coefficient()
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                # x, y = x.unsqueeze(1), y.unsqueeze(1)
                # print(x.shape,y.shape)
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                loss = criterion(out.float(),y.float())
                loss.backward()
                optimizer.step()
                # train_dice += dice(y,out)
                # train_loss += loss#一个epoch的loss
            # print("\n epoch",epoch,"train loss: ",train_loss/(batch_idx+1),"train_dice: ",train_dice/(batch_idx+1))
            # clear loss and auc for training
            # train_loss=0
            # train_dice=0

            with torch.no_grad():
                self.clf.eval()
                for valbatch_idx, (valinputs, valtargets, idxs) in enumerate(val_loader):
                    # valinputs, valtargets = valinputs.unsqueeze(1), valtargets.unsqueeze(1)
                    valinputs, valtargets = valinputs.to(self.device), valtargets.to(self.device)
                    valoutputs = self.clf(valinputs)
                    validation_loss += criterion(valoutputs.float(), valtargets.float())
                    # val_dice += dice(valtargets,valoutputs)
            # print(" epoch: ",epoch,"val loss: ",validation_loss/(valbatch_idx+1),"val_dice: ",val_dice/(valbatch_idx+1))
            
            trigger += 1
            # early stopping condition: if the acc not getting larger for over 10 epochs, stop
            if validation_loss / (valbatch_idx + 1) < best['loss']:
                trigger = 0
                best['epoch'] = epoch
                best['loss'] = validation_loss / (valbatch_idx + 1)
                # print(best['epoch'],best['loss'])
                torch.save(self.clf, './result/model.pth')
            # print("\n best performance at Epoch :{}, loss :{}".format(best['epoch'],best['loss']))
            validation_loss = 0
            # val_dice=0
            if trigger >= 5:
                break
        torch.cuda.empty_cache()

## restore to original dimensions
    def predict(self, data):
        self.clf = torch.load('./result/model.pth')
        self.clf.eval()
        preds = []
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad(): 
            for x, y, idxs in loader:
                # x, y = x.unsqueeze(1), y.unsqueeze(1)
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                outputs = out.data.cpu().numpy()
                preds.append(outputs)

        predictions = np.concatenate(preds, axis=0)#(40617, 1, 128, 128)
        # pred_patches = np.expand_dims(predictions,axis=1)#(40617, 1, 1, 128, 128)
        # # pred_patches[pred_patches>=0.5] = 1
        # # pred_patches[pred_patches<0.5] = 0
        # pred_imgs = recompone_overlap(pred_patches.squeeze(), full_test_imgs_list, x_test_slice, stride=96, image_num=8251)
        # pred_imgs_3d = recompone_overlap_3d(np.array(pred_imgs), test_brain_images, image_num=38)
        # pred_imgs_3d = np.array(pred_imgs_3d)
        return predictions


    # def get_underfit_idx(self, data):  # no use for underfit_idx if train_acc=1
    #     unfit_idxs = []
    #     self.clf.eval()
    #     preds = torch.zeros(len(data), dtype=data.Y.dtype)
    #     loader = DataLoader(data, shuffle=False, **self.params['test_args'])
    #     with torch.no_grad():
    #         for x, y, idxs in loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             out, e1 = self.clf(x)
    #             pred = out.max(1)[1]
    #             preds[idxs] = pred.cpu()
    #             if preds[idxs] != y:
    #                 unfit_idxs.append(idxs)
    #     return unfit_idxs

    def predict_black_patch(self, data):
        self.clf = torch.load('./result/model.pth')
        self.clf.eval()
        probs = torch.zeros([len(data), 1, 512, 512])
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                # x, y = x.unsqueeze(1), y.unsqueeze(1)
                x, y = x.to(self.device), y.to(self.device)
                prob = self.clf(x) # torch.Size([8, 2, 64, 64, 64])
                probs[idxs] = prob.cpu() 
                labels = (probs > 0.5).int()
        return labels


    # Calculating probability for prediction, used as uncertainty
    def predict_prob(self, data):
        self.clf = torch.load('./result/model.pth')
        self.clf.eval()
        probs = torch.zeros([len(data), 1, 512, 512])
        loader = DataLoader(data, **self.params['test_args'])
        sigmoid = nn.Sigmoid()
        with torch.no_grad():
            for x, y, idxs in loader:
                # x, y = x.unsqueeze(1), y.unsqueeze(1)
                x, y = x.to(self.device), y.to(self.device)
                prob = self.clf(x) # torch.Size([8, 2, 64, 64, 64])
                # prob = sigmoid(prob) # torch.Size([8, 2, 64, 64, 64])
                probs[idxs] = prob.cpu() 

        return probs

    # Calculating 10 times probability for prediction, the mean used as uncertainty
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf = torch.load('./result/model.pth')
        self.clf.train()
        probs = torch.zeros([len(data), 1, 512, 512])
        loader = DataLoader(data, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    # x, y = x.unsqueeze(1), y.unsqueeze(1)
                    x, y = x.to(self.device), y.to(self.device)
                    prob = self.clf(x)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    # Used for Bayesian sampling
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf = torch.load('./result/model.pth')
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), 1, 512, 512])
        loader = DataLoader(data, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    # x, y = x.unsqueeze(1), y.unsqueeze(1)
                    x, y = x.to(self.device), y.to(self.device)
                    prob = self.clf(x)
                    probs[i][idxs] += prob.cpu()
        return probs

    # def get_autofeature(self, data):
    #     self.clf = torch.load('./autoencoder.pth')
    #     self.clf.eval()
    #     embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
    #     loader = DataLoader(data, shuffle=False, **self.params['test_args'])
    #     with torch.no_grad():
    #         for x, y, idxs in loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             out, e1 = self.clf(x)
    #             embeddings[idxs] = e1.cpu()
    #     return embeddings

class MSSEG_model(nn.Module):
    def __init__(self):
        super(MSSEG_model, self).__init__()
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=False)
        self.model.encoder1.enc1conv1=nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x):
        x = self.model(x)
        return x
        