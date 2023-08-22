import numpy as np
from .strategy import Strategy

#Get a random portion of the unlabel data to label
class RandomSampling(Strategy):
    def __init__(self, dataset, net):
        super(RandomSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        return np.random.choice(unlabeled_idxs, n, replace=False)
        # return np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)
