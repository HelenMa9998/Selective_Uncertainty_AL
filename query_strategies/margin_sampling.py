import numpy as np
import torch
from .strategy import Strategy

# class MarginSampling(Strategy):
#     def __init__(self, dataset, net):
#         super(MarginSampling, self).__init__(dataset, net)

#     def query(self, n, param1,param2):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data).sum((1,2,3))
#         probs_sorted, _ = probs.sort(descending=True)
#         uncertainties = probs_sorted - (1-probs_sorted)#([7250])
#         return unlabeled_idxs[uncertainties.sort()[1][:n]]


class MarginSampling(Strategy):
    def __init__(self, dataset, net):
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n,param1,param2):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        probs_sorted, _ = probs.sort(descending=True)
        print(probs_sorted.shape)

        probs_sorted = probs_sorted.view(probs_sorted.size(0), -1)
        uncertainties = probs_sorted - (1-probs_sorted)#([7250])
        print(uncertainties.shape)

        target_1_pixels_per_sample = [torch.where(labels.float() >= param1)[0] for labels in probs]# 所有样本符合要求的pixel
        uncertain_1_pixels_per_sample = [torch.where(torch.abs(labels-0.5) <= param2)[0] for labels in probs]

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

