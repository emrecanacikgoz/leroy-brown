import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from blind_robot.dataset.data_utils import AddGaussianNoise
from blind_robot.dataset.data_utils import map_features
from blind_robot.dataset.data_utils import mytransform


class CalvinDataset(Dataset):
    def __init__(
        self,
        path=None,
        input_features=None,
        target_features=None,
        window=None,
        target_vocabs=None,
        num_bins=None,
        add_gaussian_noise=None,
    ):
        super().__init__()
        self.path = path
        self.input_features = input_features
        self.target_features = target_features
        self.window = window
        self.target_vocabs = target_vocabs
        self.num_bins = num_bins
        self.add_gaussian_noise = add_gaussian_noise

        self.gaussian_noise = AddGaussianNoise(mean=0.0, std=0.01) # TODO: make mu, std as config arg
        self._load_data(path=path)


    def _load_state(self, path=None):
        print(f"Loading {path}...", file=sys.stderr)

        # load data
        with open(path, "rb") as handle:
            data = pickle.load(handle)

        frame_ids = data["frame_ids"]
        frame_id_to_index = np.full(1 + max(frame_ids), -1)
        frame_id_to_index[frame_ids] = np.arange(len(frame_ids))

        return data, frame_ids, frame_id_to_index
    
    def _load_language(self, data):
        vocabulary = {label: index for index, label in enumerate(data["task_names"])}

        language_data = data["language"]

        for label in language_data:
            assert label[2] in vocabulary, "task not found"

        return language_data, vocabulary


    def _load_data(self, path):
        # load data
        data, frame_ids, self.frame_id_to_index = self._load_state(path=path)
        self.episode_start_end_ids = data["metadata"]["ep_start_end_ids"]
        
        # get desired features
        all_features = map_features(data["features"])
        self.input_feature_data, self.input_feature_lengths = self._get_features(all_features, self.input_features)
        self.target_feature_data, self.target_feature_lengths = self._get_features(all_features, self.target_features)

        # fix gripper
        minus_one_indices = np.where(self.target_feature_data[:, 6] == -1.0)
        self.target_feature_data[:, 6][minus_one_indices[0]] = 0

        # binarization
        if self.num_bins is not None or self.target_vocabs is not None:
            target_vocabs = []
            for i in range(self.target_feature_data.shape[1]):
                q = self.target_vocabs[i] if self.target_vocabs is not None else self.num_bins
                target_feature_data_i, target_vocab_i = pd.cut(
                    self.target_feature_data[:, i].flatten(), bins=q, retbins=True, labels=False,
                )
                self.target_feature_data[:, i] = target_feature_data_i
                target_vocabs.append(target_vocab_i)
            self.target_vocabs = target_vocabs
            self.target_feature_data = self.target_feature_data.astype(np.int64)
        
        # get language data and vocab
        self.language_data, self.vocabulary = self._load_language(data)


    def __len__(self):
        return len(self.language_data)


    def __getitem__(self, index):
        start_frame_id, stop_frame_id, task_label, instruction = self.language_data[
            index
        ]
        start_frame_id = int(start_frame_id)
        stop_frame_id = int(stop_frame_id)

        start_index = self.frame_id_to_index[start_frame_id]
        stop_index = self.frame_id_to_index[stop_frame_id]
        stop_index = min(stop_index, len(self.input_feature_data) - 1)

        context_idx = range(start_index, stop_index)
        current_data_input = self.input_feature_data[context_idx]
        current_data_target = self.target_feature_data[context_idx] # TODO: Should we shift targets by 1?
        
        episode = {
            "input": current_data_input,
            "target": current_data_target,
            "label": self.vocabulary[task_label],
            "initial_state": np.expand_dims(current_data_input[0, :], axis=0),
            "final_state": np.expand_dims(current_data_input[-1, :], axis=0),
            "index": index,
            "instruction": instruction,
            "start_end_ids": (start_frame_id, stop_frame_id),
        }

        source = torch.tensor(episode["input"], dtype=torch.float)
        # add random noise
        if self.add_gaussian_noise:
            source = self.gaussian_noise(source)

        if self.target_vocabs is not None:
            target = torch.tensor(episode["target"], dtype=torch.long)
        else:
            target = torch.tensor(episode["target"], dtype=torch.float)

        label = torch.tensor(episode["label"], dtype=torch.long)

        initial_state = torch.tensor(episode["initial_state"], dtype=torch.float)
        final_state = torch.tensor(episode["final_state"], dtype=torch.float) 

        return source, target, initial_state, final_state, label


    def collate_fn(self, batch):

        source, target, source_start_state, source_end_state, label = zip(*batch)

        # collate language data; pad to max length
        source = torch.nn.utils.rnn.pad_sequence(
            source, batch_first=True, padding_value=0
        )
        target = torch.nn.utils.rnn.pad_sequence(
            target, batch_first=True, padding_value=-100
        )
        source_start_state = torch.nn.utils.rnn.pad_sequence(
            source_start_state, batch_first=True, padding_value=0
        )
        source_end_state = torch.nn.utils.rnn.pad_sequence(
            source_end_state, batch_first=True, padding_value=0
        )

        mask = target != -100

        label = torch.stack(label, dim=0)

        return source, target, source_start_state, source_end_state, mask, label


    def input_dim(self):
        return sum(self.input_feature_lengths)


    def _transform(self, data, transform):
        # FIXME: generic transform
        return mytransform(data, transform)


    def _get_features(self, data, feature_names):
        selected_features = []
        selected_feature_lengths = []
        for feature_name in feature_names:
            if feature_name.endswith("_sincos"):
                feature_data = data[feature_name.replace("_sincos", "")]
                feature_data = self._transform(feature_data, "sincos")
            else:
                feature_data = data[feature_name]
            selected_features.append(feature_data)
            selected_feature_lengths.append(feature_data.shape[-1])
        selected_features = np.concatenate(selected_features, axis=1)
        return selected_features, selected_feature_lengths

