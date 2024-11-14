import sys
sys.path.append("/Users/samy/Desktop/thèse/code/AnomalyDetection/data")
sys.path.append("/Users/samy/Desktop/thèse/code/AnomalyDetection/algorithms/DeepOneClass")

import torch
import json
import os

from mnist import load_datasets, prepare_data_loaders
from model import CNN
from utils import get_radius

def compute_p_values_test(anomaly_digit):

    ANOMALY_DIGITS = anomaly_digit
    NORMAL_DIGITS = [i for i in range(10)]
    NORMAL_DIGITS.remove(ANOMALY_DIGITS)

    if os.path.isfile(f"models_infos/DeepOneClass/SOFT/model_anomaly_{ANOMALY_DIGITS}.pkl"):

        if not os.path.isfile(f"models_infos/DeepOneClass/SOFT/p_values_{ANOMALY_DIGITS}.json"):

            checkpoint = torch.load(f"models_infos/DeepOneClass/SOFT/model_anomaly_{ANOMALY_DIGITS}.pkl")
            center = checkpoint["center"]
            center = torch.from_numpy(center)
            R = checkpoint["radius"]
            
            model = CNN()
            model.load_state_dict(checkpoint["model_state_dict"])

            train_dic_dataset, val_dic_dataset, test_dic_dataset = load_datasets()
            normal_val = prepare_data_loaders(train_dic_dataset, val_dic_dataset, NORMAL_DIGITS, batch_size=None)

            inputs_val = normal_val
            with torch.no_grad():
                preds_val = model(inputs_val)

            current_batch_size = preds_val.size(0)
            c_val = center.repeat(current_batch_size, 1)

            dist = torch.sum((preds_val - c_val)**2, dim=1)
            scores = dist - R**2
            scores = -dist

            val_scores_sorted, indices = scores.sort()

            final_results = {i:[None, None] for i in range(10)}

            for digit in range(10):

                inputs_test = test_dic_dataset[digit]
                with torch.no_grad():
                    preds_test = model(inputs_test)

                current_batch_size = preds_test.size(0)
                c_test = center.repeat(current_batch_size, 1)

                dist = torch.sum((preds_test - c_test)**2, dim=1)

                scores = dist - R**2
                scores = -scores

                test_p_values = (1 + torch.sum(scores.unsqueeze(1) >= val_scores_sorted, dim=1)) / (len(val_scores_sorted) + 1)

                final_results[digit][0] = test_p_values.tolist()
                final_results[digit][1] = len(inputs_test)

            with open(f"models_infos/DeepOneClass/SOFT/p_values_{ANOMALY_DIGITS}.json", "w") as file:
                json.dump(final_results, file)

        else:
            with open(f"models_infos/DeepOneClass/SOFT/p_values_{ANOMALY_DIGITS}.json", "r") as file:
                final_results = json.load(file)

        return final_results

    else:
        print(f"Model still not trained : training required")
        return None


if __name__=="__main__":

    for i in range(10):
        test_p_values = compute_p_values_test(anomaly_digit=i)