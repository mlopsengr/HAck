#%%
import torch as torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import os
torch.manual_seed(0) # Sets random seed for reproducibility

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # Randomly crops and resize image to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet mean and std
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

dataset =  './traffic_Data/DATA'

# splitting the dataset to train and validation
#train_data_size = int(0.8 * len(os.listdir(dataset + '/train')))
#valid_data_size = len(os.listdir(dataset + '/val'))

#train_directory = dataset + '/train'
#valid_directory = dataset + '/val'

train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'val')

batch_size = 25
num_classes = 58

data = {
    'train': datasets.ImageFolder(train_directory, transform=image_transforms['train']),
    'val': datasets.ImageFolder(valid_directory, transform=image_transforms['val'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['val'])

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['val'], batch_size=batch_size, shuffle=True)

print('Train data size: ', train_data_size)
print('Valid data size: ', valid_data_size)

resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

fc_input_size = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_input_size, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes),  # recheck
    nn.LogSoftmax(dim=1)
)

#resnet50 = resnet50.to('cuda:0')  for NVIDIA GPU

loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.001)

#%%
def train_and_valid(model, loss_function, optimizer, epochs):
    loss_function = loss_function
    optimizer = optimizer
    model = model
    epochs = epochs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            
        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        train_loss = train_loss / train_data_size
        train_acc = train_acc / train_data_size

        valid_loss = valid_loss / valid_data_size
        valid_acc = valid_acc / valid_data_size

        history.append([train_loss, valid_loss, train_acc, valid_acc])

        if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print('Train loss: {:.4f}, Train acc: {:.4f}, Valid loss: {:.4f}, Valid acc: {:.4f}, Time: {:.4f}'.format(train_loss, train_acc * 100, valid_loss, valid_acc * 100, epoch_end - epoch_start))
        
        print("Best accuracy fo validation set: {:.4f} at epoch {}".format(best_acc, best_epoch))

        torch.save(model, 'models/model_epoch_{}.pt'.format(epoch + 1))

        return model, history

#%%
epochs = 5
model, history = train_and_valid(resnet50, loss_func, optimizer, epochs)
torch.save(history, 'history/ _history.pt')

history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Train loss', 'Valid loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig(dataset + '/loss_curve.png')

plt.plot(history[:,2:4])
plt.legend(['Train acc', 'Valid acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(dataset + '/accuracy_curve.png')
plt.show()
##%%
# %%

import azureml.core
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.run import Run
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.image import Image

# pickles the model to be uploaded to Azure
torch.save(model, 'outputs/model.pkl')
