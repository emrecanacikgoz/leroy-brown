from typing import Any

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from blind_robot.dataset.dataset import CalvinDataset
from blind_robot.dataset.data_utils import map_features


class CalvinDatasetGoalUnsupervised(CalvinDataset):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

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
        
        print(f"input: {self.input_feature_data}, \nshape: {self.input_feature_data.shape}") # (99022, 13)

        self.source, self.target = self._get_random_window()


    def __len__(self):
        return len(self.source)
    

    def __getitem__(self, index):

        # get episode
        current_data_input = self.source[index]
        current_data_target = self.target[index]
        
        episode = {
            "input": current_data_input,
            "target": current_data_target,
            "initial_state": np.expand_dims(current_data_input[0, :], axis=0),
            "final_state": np.expand_dims(current_data_input[-1, :], axis=0),
            "index": index,
        }

        source = torch.tensor(episode["input"], dtype=torch.float)
        # add random noise
        if self.add_gaussian_noise:
            source = self.gaussian_noise(source)

        if self.target_vocabs is not None:
            target = torch.tensor(episode["target"], dtype=torch.long)
        else:
            target = torch.tensor(episode["target"], dtype=torch.float)


        initial_state = torch.tensor(episode["initial_state"], dtype=torch.float)
        final_state = torch.tensor(episode["final_state"], dtype=torch.float) 

        return source, target, initial_state, final_state


    def collate_fn(self, batch):

        source, target, source_start_state, source_end_state = zip(*batch)

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


        return source, target, source_start_state, source_end_state, mask


    def _get_random_window(self, min_window_length=32, max_window_length=64, num_samples_to_generate=10_000):
        if num_samples_to_generate > len(self.input_feature_data):
            print(f"WARNING: num_samples_to_generate ({num_samples_to_generate}) > len(self.input_feature_data) ({len(self.input_feature_data)}). Setting num_samples_to_generate to len(self.input_feature_data).")
            num_samples_to_generate = len(self.input_feature_data)

        print(f"Generating {num_samples_to_generate} random windows...")

        random_windows_source, random_windows_target = [], []
        for i in tqdm(range(num_samples_to_generate)):

            random_start_index = np.random.randint(0, len(self.input_feature_data))
            window_length = np.random.randint(min_window_length, max_window_length)
            random_stop_index = random_start_index + window_length
            source_episode = self.input_feature_data[random_start_index:random_stop_index, :]
            target_episode = self.target_feature_data[random_start_index:random_stop_index, :]
            random_windows_source.append(source_episode)
            random_windows_target.append(target_episode)

        return random_windows_source, random_windows_target
