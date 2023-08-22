from data import get_MSSEG
from handlers import Messidor_Handler, MSSEG_Handler_2d
from nets import Net, MSSEG_model
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, EntropySamplingDropout, BALDDropout, AdversarialAttack,AdversarialAttack_efficient
from seed import setup_seed

# important settings
setup_seed()
params = {
    'MSSEG':
        {'n_epoch': 200,
         'train_args': {'batch_size': 16,'shuffle':True, 'num_workers': 4,'drop_last':False},
         'val_args': {'batch_size': 64,'shuffle':False, 'num_workers': 4,'drop_last':False},
         'test_args': {'batch_size': 64,'shuffle':False, 'num_workers': 4,'drop_last':False},
         'optimizer_args': {'lr': 0.001}},  
}


# Get data loader
def get_handler(name):
    if name == 'Messidor':
        return Messidor_Handler
    elif name == 'MSSEG':
        return MSSEG_Handler_2d


# Get dataset
def get_dataset(name,supervised):
    if name == 'Messidor':
        return get_Messidor(get_handler(name))
    elif name == 'MSSEG':
        if supervised == True:
            return get_MSSEG(get_handler(name),supervised = True)
        else:
            return get_MSSEG(get_handler(name))
    else:
        raise NotImplementedError


# define network for specific dataset
def get_net(name, device, init=False):
    if name == 'Messidor':
        # return Net(Res_Net, params[name], device)
        # if init==False:
        return Net(Inception_V3, params[name], device)
        # return Net(Dense_Net, params[name], device)
    #         return Net(Res_Net, params[name], device)
    # else:
    #     return Net(Autoencoder, params[name], device)
    elif name == 'Breast':
        # return Net(Res_Net, params[name], device)
        # if init==False:
        return Net(Inception_V3, params[name], device)

    elif name == 'MSSEG':
        return Net(MSSEG_model, params[name], device)

    else:
        raise NotImplementedError

# get strategies
def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialAttack":
        return AdversarialAttack
    elif name == "AdversarialAttack_efficient":
        return AdversarialAttack_efficient
    else:
        raise NotImplementedError