import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


with open('./leroy-brown/data/D_training.pkl', 'rb') as f:
    train = pickle.load(f)
    data = train['language']
    print(train.keys())
    # 'features', 'language', 'frame_ids', 'task_names', 'field_names', 'metadata'
    print(train['features'].shape)
    print(train['language'].shape)
    print(train['frame_ids'].shape)
    print(train['language'])

train_data, val_data = train_test_split(data, test_size=0.10, stratify=data[:, 2], random_state=42) 
print(train_data.shape)
print(val_data.shape)
print(val_data)

tasks_train = Counter([x[2] for x in train_data])
tasks_val = Counter([x[2] for x in val_data])

train_dataset = {
    'features': train['features'],
    'language': train_data,
    'frame_ids': train['frame_ids'],
    'task_names': train['task_names'],
    'field_names': train['field_names'],
    'metadata': train['metadata'],
}

val_dataset = {
    'features': train['features'],
    'language': val_data,
    'frame_ids': train['frame_ids'],
    'task_names': train['task_names'],
    'field_names': train['field_names'],
    'metadata': train['metadata'],
}

with open(file=f"D_training_ours.pkl", mode="wb") as f:
    pickle.dump(train_dataset, f)

with open(file=f"D_validation_ours.pkl", mode="wb") as f:
    pickle.dump(val_dataset, f)