import numpy as np
import torch
from .strategy import Strategy

# The parameter space is obtained by Bayesian model (MC dropout), making the uncertainty of the parameter space as small as possible
# class BALDDropout(Strategy):
#     def __init__(self, dataset, net, n_drop=10):
#         super(BALDDropout, self).__init__(dataset, net)
#         self.n_drop = n_drop

#     def query(self, n, param1,param2):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)# 10 times summation
#         pb = probs.mean(0) # 10 times mean
#         entropy1 = (-pb*torch.log(pb)).sum((1,2,3)) # Average mean summation
#         entropy2 = (-probs*torch.log(probs)).sum((2,3,4)).mean(0) # Summation followed by mean
#         uncertainties = entropy2 - entropy1 # getting variance
#         return unlabeled_idxs[uncertainties.sort()[1][:n]]

class BALDDropout(Strategy):
    def __init__(self, dataset, net, n_drop=10):
        super(BALDDropout, self).__init__(dataset, net)
        self.n_drop = n_drop

    def query(self, n,param1,param2):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)# ([10, 7250, 1, 240, 240])
        pb = probs.mean(0) # ([7250, 1, 240, 240])

        entropy1 = (-pb*torch.log(pb)) # Average mean summation
        entropy2 = (-probs*torch.log(probs)).mean(0) # Summation followed by mean
        # print(entropy1.shape)
        # print(entropy2.shape)

        uncertainties = entropy2 - entropy1 # getting variance
        # print(uncertainties.shape)
        pb = pb.view(pb.size(0), -1)#([7250, 57600])
        uncertainties = uncertainties.view(uncertainties.size(0), -1)#([7250, 1])

        target_1_pixels_per_sample = [torch.where(labels.float() >= param1)[0] for labels in pb]# 7250
        uncertain_1_pixels_per_sample = [torch.where(torch.abs(labels-0.5) <= param2)[0] for labels in pb]# 7250


        target_uncertainty_avg_per_sample = [torch.nan_to_num(uncertainty[target_1_pixels].mean(), nan=0.0) for uncertainty, target_1_pixels in zip(uncertainties, target_1_pixels_per_sample)]
        uncertain_uncertainty_avg_per_sample = [torch.nan_to_num(uncertainty[target_1_pixels].mean(), nan=0.0) for uncertainty, target_1_pixels in zip(uncertainties, uncertain_1_pixels_per_sample)]

        target_sorted_indices = sorted(range(len(target_uncertainty_avg_per_sample)), key=lambda k: target_uncertainty_avg_per_sample[k])
        uncertain_sorted_indices = sorted(range(len(uncertain_uncertainty_avg_per_sample)), key=lambda k: uncertain_uncertainty_avg_per_sample[k])

        concatenated_indices = []

        for i in range(len(unlabeled_idxs)):
            if i % 2 == 0:
                current_indices = target_sorted_indices.pop(0)
            else:
                current_indices = uncertain_sorted_indices.pop(0)
            # print("current_indices",current_indices)
            if current_indices not in concatenated_indices: 
                concatenated_indices.append(current_indices)
            # print(len(concatenated_indices))
            if len(concatenated_indices) == n:
                break
        return unlabeled_idxs[concatenated_indices]