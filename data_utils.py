import json
import os
import random


def load_data(dataset_name, dataset_dir = "./dataset"):
    with open(os.path.join(dataset_dir, dataset_name, "train.json"), "r") as f:
        train_dataset = json.load(f)
    with open(os.path.join(dataset_dir, dataset_name, "valid.json"), "r") as f:
        valid_dataset = json.load(f)
    with open(os.path.join(dataset_dir, dataset_name, "test.json"), "r") as f:
        test_dataset = json.load(f)
    if os.path.exists(os.path.join(dataset_dir, dataset_name, "label_list.json")):
        with open(os.path.join(dataset_dir, dataset_name, "label_list.json"), "r") as f:
            label_list = json.load(f)
    else:
        label_list=None
    return train_dataset, valid_dataset, test_dataset, label_list