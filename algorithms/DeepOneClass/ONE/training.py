import sys
sys.path.append("/Users/samy/Desktop/thèse/code/AnomalyDetection/data")
sys.path.append("/Users/samy/Desktop/thèse/code/AnomalyDetection/algorithms/DeepOneClass")

import torch
import torch.optim as optim
from tqdm import trange
import os
from mnist import load_datasets, prepare_data_loaders
from model import CNN
from utils import transform

DEVICE="mps"

def training(anomaly_digits):

    ANOMALY_DIGITS = anomaly_digits
    NORMAL_DIGITS = [i for i in range(10)]
    NORMAL_DIGITS.remove(ANOMALY_DIGITS)

    BATCH_SIZE=128
    EPOCHS=150

    train_dic_dataset, val_dic_dataset, _ = load_datasets(transform=transform)
    trainloader = prepare_data_loaders(train_dic_dataset, val_dic_dataset, NORMAL_DIGITS, BATCH_SIZE)

    model = CNN().to(DEVICE)

    all_preds = []
    model.eval()

    with torch.no_grad():
        for imgs in trainloader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs)
            all_preds.append(preds) 

    all_preds = torch.cat(all_preds)
    center_0 = torch.mean(all_preds, 0)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    train_iterator = trange(EPOCHS, desc="epochs")

    for epoch in train_iterator:
        epoch_loss = 0

        for imgs in trainloader:

            imgs = imgs.to(DEVICE)
            preds = model(imgs)

            current_batch_size = imgs.size(0)
            c_batch = center_0.repeat(current_batch_size, 1)

            dist = torch.sum((preds - c_batch)**2, dim=1)

            loss = torch.mean(dist)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        train_iterator.set_description(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    checkpoint = {
            "model_state_dict":model.state_dict(),
            "center":center_0.detach().cpu().numpy(),
        }

    os.makedirs(f'models_infos/DeepOneClass', exist_ok=True)
    os.makedirs(f'models_infos/DeepOneClass/ONE', exist_ok=True)
        
    
    torch.save(checkpoint, f'models_infos/DeepOneClass/One/model_anomaly_{ANOMALY_DIGITS}.pkl')

    return 

if __name__=="__main__":
    for i in range(10):
        training(anomaly_digits=i)