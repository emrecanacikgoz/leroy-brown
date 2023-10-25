from typing import Any

import numpy as np
import torch

from blind_robot.dataset.dataset import CalvinDataset
from blind_robot.dataset.data_utils import AddGaussianNoise


class CalvinDatasetGoalSupervised(CalvinDataset):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

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

        # random window
        rand_current_data_input, rand_current_data_target = self._get_random_window(context_idx)
        
        episode = {
            "input": current_data_input,
            "target": current_data_target,
            "label": self.vocabulary[task_label],
            "random_input": rand_current_data_input,
            "random_target": rand_current_data_target,
            "initial_state": np.expand_dims(current_data_input[0, :], axis=0),
            "final_state": np.expand_dims(current_data_input[-1, :], axis=0),
            "random_initial_state": np.expand_dims(rand_current_data_input[0, :], axis=0),
            "random_final_state": np.expand_dims(rand_current_data_input[-1, :], axis=0),
            "index": index,
            "instruction": instruction,
            "start_end_ids": (start_frame_id, stop_frame_id),
        }

        source = torch.tensor(episode["input"], dtype=torch.float)
        source_rand = torch.tensor(episode["random_input"], dtype=torch.float)
        # add random noise
        if self.add_gaussian_noise:
            source = self.gaussian_noise(source)
            source_rand = self.gaussian_noise(source_rand)

        if self.target_vocabs is not None:
            target = torch.tensor(episode["target"], dtype=torch.long)
        else:
            target = torch.tensor(episode["target"], dtype=torch.float)

        label = torch.tensor(episode["label"], dtype=torch.long)

        initial_state = torch.tensor(episode["initial_state"], dtype=torch.float)
        final_state = torch.tensor(episode["final_state"], dtype=torch.float) 
        initial_state_rand = torch.tensor(episode["random_initial_state"], dtype=torch.float)
        final_state_rand = torch.tensor(episode["random_final_state"], dtype=torch.float)

        return source, target, initial_state, final_state, label, source_rand, initial_state_rand, final_state_rand


    def collate_fn(self, batch):

        source, target, source_start_state, source_end_state, label, source_rand, initial_state_rand, final_state_rand = zip(*batch)

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

        # rands
        source_rand = torch.nn.utils.rnn.pad_sequence(
            source_rand, batch_first=True, padding_value=0
        )
        initial_state_rand = torch.nn.utils.rnn.pad_sequence(
            initial_state_rand, batch_first=True, padding_value=0
        )
        final_state_rand = torch.nn.utils.rnn.pad_sequence(
            final_state_rand, batch_first=True, padding_value=0
        )

        mask = target != -100

        label = torch.stack(label, dim=0)

        return source, target, source_start_state, source_end_state, mask, label, source_rand, initial_state_rand, final_state_rand


    def _get_random_window(self, context_idx, min_window_length=32, max_window_length=64):

        window_length = np.random.randint(min_window_length, max_window_length)
        contexts = list(context_idx)

        if len(contexts) <= window_length:
            start_index = 0
        else:
            start_index = np.random.randint(0, len(contexts) - window_length)

        stop_index = start_index + window_length

        random_contex_idx = range(start_index, stop_index)
        rand_current_data_input = self.input_feature_data[random_contex_idx]
        rand_current_data_target = self.target_feature_data[random_contex_idx]
        
        return rand_current_data_input, rand_current_data_target
