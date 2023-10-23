import numpy as np
import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.std = torch.tensor(std)
        self.mean = torch.tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


def map_features(all_data):
    # FIXME: verify! this mapping should be fixed and read from data file itself
    # The data can have a field data["mappings"] == {"actions": 1:7, ...}
    # otherwise typo in the code will introduce wrong mappings.
    all_features = {
        "actions": all_data[:, :7],
        "actions_tcp": all_data[:, :6],
        "actions_xyz": all_data[:, :3],
        "actions_exeyez": all_data[:, 3:6],
        "actions_g": all_data[:, 6:7],
        "rel_actions": all_data[:, 7:14],
        "rel_actions_tcp": all_data[:, 7:13],
        "rel_actions_xyz": all_data[:, 7:10],
        "rel_actions_exeyez": all_data[:, 10:13],
        "rel_actions_g": all_data[:, 13:14],
        "robot_obs": all_data[:, 14:29],
        "robot_obs_tcp": all_data[:, 14:20],
        "robot_obs_xyz": all_data[:, 14:17],
        "robot_obs_exeyez": all_data[:, 17:20],
        "robot_obs_gw": all_data[:, 20:21],
        "robot_obs_arm_joints": all_data[:, 21:28],
        "robot_obs_g": all_data[:, 28:29],
        "scene_obs": all_data[:, 29:53],
        "scene_obs_object_joints": all_data[:, 29:33],
        "scene_obs_lights": all_data[:, 33:35],
        "scene_obs_blocks_tcp": all_data[:, 35:53],
        "scene_obs_red_xyz": all_data[:, 35:38],
        "scene_obs_red_exeyez": all_data[:, 38:41],
        "scene_obs_blue_xyz": all_data[:, 41:44],
        "scene_obs_blue_exeyez": all_data[:, 44:47],
        "scene_obs_pink_xyz": all_data[:, 47:50],
        "scene_obs_pink_exeyez": all_data[:, 50:53],
        "tactile": all_data[:, 53:61],
        "controller": all_data[:, 61:73],
        "diffs": all_data[:, 73:97],
    }
    return all_features


def mytransform(data, transform):
    if transform == "sincos":
        sin = np.sin(data)
        cos = np.cos(data)
        data = np.concatenate((sin, cos), axis=-1)
    else:
        raise ValueError("Unknown transform type")
    return data


int2task = [
    "close_drawer",
    "lift_blue_block_drawer",
    "lift_blue_block_slider",
    "lift_blue_block_table",
    "lift_pink_block_drawer",
    "lift_pink_block_slider",
    "lift_pink_block_table",
    "lift_red_block_drawer",
    "lift_red_block_slider",
    "lift_red_block_table",
    "move_slider_left",
    "move_slider_right",
    "open_drawer",
    "place_in_drawer",
    "place_in_slider",
    "push_blue_block_left",
    "push_blue_block_right",
    "push_into_drawer",
    "push_pink_block_left",
    "push_pink_block_right",
    "push_red_block_left",
    "push_red_block_right",
    "rotate_blue_block_left",
    "rotate_blue_block_right",
    "rotate_pink_block_left",
    "rotate_pink_block_right",
    "rotate_red_block_left",
    "rotate_red_block_right",
    "stack_block",
    "turn_off_led",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_on_lightbulb",
    "unstack_block",
]