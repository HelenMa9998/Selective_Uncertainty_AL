import torch
from utils import get_dataset, get_net, get_strategy
from data import Data
from config import parse_args

from seed import setup_seed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# fix random seed
# setup_seed(42)
#supervised learning baseline
args = parse_args()
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get dataset
X_train, Y_train, X_val, Y_val, X_test, Y_test, handler = get_dataset(args.dataset_name,supervised=False)
dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test, handler)

print(f"number of testing pool: {dataset.n_test}")
print()
# get network
net = get_net(args.dataset_name, device)

# start supervised learning baseline
dataset.supervised_training_labels()
labeled_idxs, labeled_data = dataset.get_labeled_data()
val_data = dataset.get_val_data()
print(f"number of labeled pool: {len(labeled_idxs)}")
net.supervised_val_loss(labeled_data,val_data,rd=0)
preds = net.predict(dataset.get_test_data())
print(f"testing dice: {dataset.cal_test_acc(preds,Y_test)}")