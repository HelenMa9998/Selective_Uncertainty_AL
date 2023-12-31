## Adaptive adversarial 
### Introduction
This project is the code for paper Adaptive Adversarial samples based Active learning for medical image classification based on python and pytorch framework.
  

### Requirements  
The main package and version of the python environment are as follows
```
# Name                    Version         
python                    3.8.5                    
pytorch                   1.10.1         
torchvision               0.15.2         
cudatoolkit               10.2.89       
cudnn                     8.5.0.96           
matplotlib                3.3.2              
numpy                     1.19.2        
opencv                    4.7.0.68         
pandas                    1.1.3               
scikit-learn              0.23.2                
tqdm                      4.50.2             
```  

The above environment is successful when running the code of the project. Pytorch has very good compatibility. Thus, I suggest that try to use the existing pytorch environment firstly.

---  
## Usage 
### 1) Download Project 

Running```git clone https://github.com/HelenMa9998/Selective_Uncertainty_Active_Learning.git```  
The project structure and intention are as follows : 
```
Adversarial active learning			# Source code		
    ├── seed.py			 	                                          # Set up random seed
    ├── query_strategies		                                    # All query_strategies
    │   ├── least_confidence.py                                 # least_confidence query method
    │   ├── margin_sampling.py                                  # margin_sampling query method
    │   ├── bayesian_active_learning_disagreement_dropout.py	  # Deep bayesian query method
    │   ├── entropy_sampling.py		                              # Entropy based query method
    │   ├── entropy_sampling_dropout.py		                      # Entropy based MC dropout query method
    │   ├── random_sampling.py		                              # Random selection
    │   ├── strategy.py                                         # Functions needed for query strategies
    ├── data.py	                                                # Prepare the dataset & initialization and update for training dataset
    ├── config.py	                                              # Configuration
    ├── seed.py	                                                # Random seed setting
    ├── handlers.py                                             # Get dataloader for the dataset
    ├── main.py			                                            # An example for code utilization, including the whole process of active learning
    ├── nets.py		                                              # Training models and methods needed for query method
    ├── supervised_baseline.py	                                # An example for supervised learning traning process
    └── utils.py			                                          # Important setups including network, dataset, hyperparameters...
```
### 2) Datasets preparation 
1. Download the datasets from the official address:
   
   BraTS 2019 Dataset: https://www.med.upenn.edu/cbica/brats2019/data.html
   
   Medical Segmentation Decathlon Dataset (MSD): http://medicaldecathlon.com

   
2. Modify the data folder path for specific dataset in `data.py`

### 3) Run Active learning process 
Please confirm the configuration information in the [`utils.py`]
```
  python main.py \
      --n_round 34 \
      --n_query 20 \
      --n_init_labeled 100 \
      --dataset_name Messidor \
      --traning_method supervised_train_acc \
      --strategy_name RandomSampling \
      --seed 2022
```
The training results will be saved to the corresponding directory(save name) in `performance.csv`.  
You can also run `supervised_baseline.py` by
```
python supervised_baseline.py
```

