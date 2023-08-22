import argparse
import numpy as np
import torch
import pandas as pd
from pprint import pprint

from data import Data
from utils import get_dataset, get_net, get_strategy
from config import parse_args
from seed import setup_seed

# args = parse_args.parse_known_args()[0]



# device
def main(param1,param2,param3):
    args = parse_args()
    pprint(vars(args))
    print()

    # fix random seed
    setup_seed()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test, handler = get_dataset(args.dataset_name,supervised=False)
    dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test, handler)

    net = get_net(args.dataset_name, device) # load network
    strategy = get_strategy(param3)(dataset, net) # load strategy

    # start experiment
    dataset.initialize_labels_random(args.n_init_labeled)
    # init_num = dataset.initialize_labels_K(k=15)
    # print(init_num)
    print("Round 0")
    rd = 0
    strategy.train(rd, args.training_name)
    accuracy = []
    size = []
    preds= strategy.predict(dataset.get_test_data()) # get model prediction for test dataset
    acc = dataset.cal_test_acc(preds,Y_test)
    print(f"Round 0 testing accuracy: {acc}")  # get model performance for test dataset
    accuracy.append(acc)
    size.append(args.n_init_labeled)
    testing_accuracy = 0


    for rd in range(1, args.n_round + 1):
        print(f"Round {rd}")
        # query
        query_idxs = strategy.query(args.n_query,param1,param2)  # query_idxs为active learning请求标签的数据

        # update labels
        strategy.update(query_idxs)  # update training dataset and unlabeled dataset for active learning
        strategy.train(rd, args.training_name)

        # unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data(index = None)
        # labels = net.predict_black_patch(unlabeled_data)
        # index = dataset.delete_black_patch(unlabeled_idxs, labels)

        # efficient training
        # strategy.efficient_train(rd,dataset.get_train_data())

        # calculate accuracy
        preds= strategy.predict(dataset.get_test_data())
        acc = dataset.cal_test_acc(preds,Y_test)
        print(f"Round {rd} testing accuracy: {acc}")

        accuracy.append(acc)
        labeled_idxs, _ = dataset.get_labeled_data()
        size.append(len(labeled_idxs))
        print(len(labeled_idxs))
        # unlabeled_idxs, _ = dataset.get_unlabeled_data(index = index)
        # if len(unlabeled_idxs) < 300:
        #     break

    # save the result
    dataframe = pd.DataFrame(
        {'model': 'Unet', 'Method': args.strategy_name, 'Training dataset size': size, 'Accuracy': accuracy})
    dataframe.to_csv(f"./{param3}{param1}{param2}.csv", index=False, sep=',') 

experiment_parameters = [
    # {'param1': 0.1, 'param2': 0.09, 'param3': "MarginSampling"},
    {'param1': 0.17, 'param2': 0.09, 'param3': "MarginSampling"},
    {'param1': 0.17, 'param2': 0.11, 'param3': "MarginSampling"},
    {'param1': 0.19, 'param2': 0.11, 'param3': "MarginSampling"},
    {'param1': 0.19, 'param2': 0.09, 'param3': "MarginSampling"},



    # # {'param1': 0.2, 'param2': 0.09, 'param3': "EntropySamplingDropout"},
    # # {'param1': 0.2, 'param2': 0.08, 'param3': "EntropySamplingDropout"},
    # # {'param1': 0.2, 'param2': 0.11, 'param3': "EntropySamplingDropout"},
    # {'param1': 0.19, 'param2': 0.1, 'param3': "EntropySamplingDropout"},
    # # {'param1': 0.25, 'param2': 0.08, 'param3': "EntropySamplingDropout"},



    {'param1': 0.29, 'param2': 0.1, 'param3': "BALDDropout"},
    {'param1': 0.31, 'param2': 0.1, 'param3': "BALDDropout"},
    # # {'param1': 0.31, 'param2': 0.1, 'param3': "BALDDropout"},

    # Add more parameter combinations as needed
]

for params in experiment_parameters:
    main(params['param1'],params['param2'],params['param3'])
