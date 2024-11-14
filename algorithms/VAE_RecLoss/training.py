import sys
sys.path.append("/Users/samy/Desktop/thèse/code/AnomalyDetection/data")
sys.path.append("/Users/samy/Desktop/thèse/code/AnomalyDetection/algorithms/VAE_RecLoss")

import torch
import torch.optim as optim
from tqdm import tqdm
import os
from mnist import load_datasets, prepare_data_loaders
from model import VAE
from loss import LossVAE

DEVICE="mps"

def training(anomaly_digits):

    ANOMALY_DIGITS = anomaly_digits
    NORMAL_DIGITS = [i for i in range(10)]
    NORMAL_DIGITS.remove(ANOMALY_DIGITS)

    BATCH_SIZE=128
    LEARNING_RATE=3e-4
    EPOCHS=20

    train_dic_dataset, val_dic_dataset, _ = load_datasets()
    trainloader = prepare_data_loaders(train_dic_dataset, val_dic_dataset, NORMAL_DIGITS, BATCH_SIZE)

    model = VAE(in_dim=784, hidden_dim=[512, 256], latent_dim=2).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = LossVAE()

    for epoch in range(EPOCHS):
        epoch_loss=0
        for inputs in tqdm(trainloader):
            inputs = inputs.flatten(start_dim=1).to(DEVICE)
            reconstructed, mu, logvar = model(inputs)
            loss = criterion(inputs, reconstructed, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        print(f"For epoch {epoch+1}/{EPOCHS} ; loss is {epoch_loss}")

    checkpoints = {'state_dict':model.state_dict()}

    os.makedirs(f'models_infos/VAE_RecLoss', exist_ok=True)
        
    torch.save(checkpoints, f'models_infos/VAE_RecLoss/model_anomaly_{ANOMALY_DIGITS}.pkl')
    print("Model trained, checkpoint saved")

    return 

if __name__=="__main__":
    for i in range(10):
        training(anomaly_digits=i)