import sys
sys.path.append("/Users/samy/Desktop/thèse/code/AnomalyDetection/data")
sys.path.append("/Users/samy/Desktop/thèse/code/AnomalyDetection/algorithms/VAE_RecLoss")

import torch
import json
import os

from mnist import load_datasets, prepare_data_loaders
from model import VAE

def compute_p_values_test(anomaly_digit):

    ANOMALY_DIGITS = anomaly_digit
    NORMAL_DIGITS = [i for i in range(10)]
    NORMAL_DIGITS.remove(ANOMALY_DIGITS)

    if os.path.isfile(f"models_infos/VAE_RecLoss/model_anomaly_{ANOMALY_DIGITS}.pkl"):

        if not os.path.isfile(f"models_infos/VAE_RecLoss/p_values_{ANOMALY_DIGITS}.json"):

            checkpoint = torch.load(f"models_infos/VAE_RecLoss/model_anomaly_{ANOMALY_DIGITS}.pkl")
            model = VAE(in_dim=784, hidden_dim=[512, 256], latent_dim=2)
            model.load_state_dict(checkpoint["state_dict"])

            train_dic_dataset, val_dic_dataset, test_dic_dataset = load_datasets()
            normal_val = prepare_data_loaders(train_dic_dataset, val_dic_dataset, NORMAL_DIGITS, batch_size=None)

            inputs_val = normal_val.flatten(start_dim=1)
            with torch.no_grad():
                reconstructed_val, _, _ = model(inputs_val)

            val_scores = -torch.sum(((inputs_val - reconstructed_val)**2), dim=1)
            val_scores_sorted, indices = val_scores.sort()


            final_results = {i:[None, None] for i in range(10)}

            for digit in range(10):

                inputs_test = test_dic_dataset[digit].flatten(start_dim=1)
                with torch.no_grad():
                    test_reconstructed, _, _ = model(inputs_test)

                test_scores = -torch.sum(((inputs_test - test_reconstructed)**2), dim=1)

                test_p_values = (1 + torch.sum(test_scores.unsqueeze(1) >= val_scores_sorted, dim=1)) / (len(val_scores_sorted) + 1)

                final_results[digit][0] = test_p_values.tolist()
                final_results[digit][1] = len(inputs_test)

            with open(f"models_infos/VAE_RecLoss/p_values_{ANOMALY_DIGITS}.json", "w") as file:
                json.dump(final_results, file)

        else:
            with open(f"models_infos/VAE_RecLoss/p_values_{ANOMALY_DIGITS}.json", "r") as file:
                final_results = json.load(file)

        return final_results

    else:
        print(f"Model still not trained : training required")
        return None
    

if __name__=="__main__":

    for i in range(10):
        test_p_values = compute_p_values_test(anomaly_digit=i)