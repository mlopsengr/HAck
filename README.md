# TRAFFIC SIGN CLASSIFICATION
This project builds a machine learning model in python using a convolutional neural network
for classification of 57 distinct traffic signs

## Data Preparation
- The data for this project can be found on [data](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification?resource=download&select=labels.csv). It contains
57 distinct traffic signs resulting in 57 labels to be classified
- The data in the DATA folder is split into two seperate folders, 'train' and 'val', holding 80% and 20% respectively, of the total set.
- Folders 'train' and 'val' are then imported to the python script

## Python script
- The traffic.py file contains the python code used to develop this model.
- Within this script, the PyTorch framework is used
```python
    import torch as torch
```
## Neural network
- A pretrained CNN, RESNET-50 is used to train the model via transfer learning - output size is adjusted to meet the number of classes intended for classification
```python
    resnet50 = models.resnet50(pretrained=True)
```
- The Adam optimizer is used with this network and its learning rate is kept at a standard value of 0.001 but can be adjusted 
```python
optimizer = optim.Adam(resnet50.parameters(), lr=0.001)
```
- The model is then saved in folder named 'models'

## CI/CD Pipeline
- For integrating and deploying this model automatically, the azure DevOps tool is utilised
- A script named 'start_experiment.py' defines the environment, model, python version, running image, compute instance and retrieves the python script, 'traffic.py', for retraining the model
- Wihin this script, the azure compute instance created is defined as well as the workspace, subscription ID, and resoure-group. This can always be modified.
```python

compute_name = 'hackcompute22'

experiment_name = 'my-experiment' 
environment_name = 'my-environment'

model_name = 'my-model'
model_path = 'outputs/model.pkl'


source_directory = '.'

script_path = 'Hack Partners/traffic.py'


subscription_id = '4f67948b-2ff9-49ee-bf1f-90c32dc7545e'
resource_group  = 'cloud-shell-storage-westeurope'
workspace_name  = 'Hack_Workspace'
```

- The Yml file, azure-pipelines.yml then runs this script, 'start_experiment.py'

