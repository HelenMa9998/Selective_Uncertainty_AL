import argparse
# configurations

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2024, help="random seed")
    parser.add_argument('--n_init_labeled', type=int, default=150, help="number of init labeled samples")
    parser.add_argument('--n_query', type=int, default=50, help="number of queries per round")
    parser.add_argument('--n_round', type=int, default=15, help="number of rounds")
    parser.add_argument('--dataset_name', type=str, default="MSSEG", choices=["Messidor", "MSSEG"], help="dataset")
    parser.add_argument('--early-stop', default=10, type=int, help='early stopping')
    parser.add_argument('--training_name', type=str, default="supervised_val_loss",
                        choices=["supervised_train_acc",
                                 "supervised_val_loss",
                                 "supervised_val_acc",
                                 "supervised_train_epoch",], help="training method")
    parser.add_argument('--strategy_name', type=str, default="MarginSampling",
                        choices=["RandomSampling",
                                 "LeastConfidence", 
                                 "MarginSampling", 
                                 "EntropySampling",
                                 "EntropySamplingDropout",
                                 "BALDDropout",
                                 "AdversarialAttack",
                                 "AdversarialAttack_efficient",], help="query strategy")

    args = parser.parse_args()
    return args

