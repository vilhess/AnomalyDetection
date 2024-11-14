import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import os

def get_dataset_by_digit(dataset):
    dic = {i:[] for i in range(10)}
    for img, lab in dataset:
        dic[lab].append(img)
    dic = {i:torch.stack(dic[i]) for i in range(10)}
    return dic


def load_datasets(transform=None):
    generator = torch.Generator().manual_seed(42)

    if not transform:
        dataset = MNIST(root="../../../coding/Dataset/", train=True, download=False, transform=ToTensor())
        testset = MNIST(root="../../../coding/Dataset/", train=False, download=False, transform=ToTensor())
    else:
        dataset = MNIST(root="../../../coding/Dataset/", train=True, download=False, transform=transform)
        testset = MNIST(root="../../../coding/Dataset/", train=False, download=False, transform=transform)
    
    train_size = 40000
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_dic_dataset = get_dataset_by_digit(trainset)
    val_dic_dataset = get_dataset_by_digit(valset)
    test_dic_dataset = get_dataset_by_digit(testset)
    
    return train_dic_dataset, val_dic_dataset, test_dic_dataset

def prepare_data_loaders(train_dic_dataset, val_dic_dataset, normal_digits, batch_size):
    normal_trainset = torch.cat([train_dic_dataset[i] for i in normal_digits])
    normal_val = torch.cat([val_dic_dataset[i] for i in normal_digits])
    
    if batch_size==None:
        return normal_val
    
    else:
        trainloader = DataLoader(normal_trainset, batch_size=batch_size, shuffle=True)
        return trainloader