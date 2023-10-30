from typing import List
from blind_robot.dataset.data_utils import mytransform as transform
from evaluation.utils import join_vis_lang
from evaluation.utils import task2int as TASK2INT
import numpy as np
from termcolor import colored
import torch


# FIXME: If possible read this from env.observation_space()
FEATURE_MAPPINGS = {
    "robot_obs": {
        "xyz": [0, 1, 2],
        "exeyez": [3, 4, 5],
        "gw": [6],
        "joints": [7, 8, 9, 10, 11, 12, 13],
        "g": [14],
    },
    "scene_obs": {
        "object_joints": [0, 1, 2, 3],
        "lights": [4, 5],
        "red_xyz": [6, 7, 8],
        "red_exeyez": [9, 10, 11],
        "blue_xyz": [12, 13, 14],
        "blue_exeyez": [15, 16, 17],
        "pink_xyz": [18, 19, 20],
        "pink_exeyez": [21, 22, 23],
    },
}


def get_features(
    scene_obs: torch.tensor, robot_obs: torch.tensor, input_features: List[str]
):
    assert (
        len(scene_obs.shape) == 3
    ), f"scene_obs should be 3D, but it is ({scene_obs.shape})"
    features = []
    for feature_name in input_features:
        # TODO: generalize for other transformations
        sincos = feature_name.endswith("sincos")
        feature_name = feature_name.replace("_sincos", "")
        if feature_name.startswith("robot_obs"):
            feature_name = feature_name.replace("robot_obs_", "")
            indices = FEATURE_MAPPINGS["robot_obs"][feature_name]
            feature = robot_obs[:, :, indices]
        elif feature_name.startswith("scene_obs"):
            feature_name = feature_name.replace("scene_obs_", "")
            indices = FEATURE_MAPPINGS["scene_obs"][feature_name]
            feature = scene_obs[:, :, indices]
        else:
            raise ValueError(f"Unknown feature name: {feature_name}")
        if sincos:
            feature = transform(feature, "sincos")
        features.append(feature)

    features = np.concatenate(features, axis=-1)
    return features


def rollout(
    env,
    model,
    episode,
    task_oracle,
    task,
    lang_embeddings,
    val_annotations,
    add_noise_during_inference,
    out,
    i,
    input_features,
    device="cuda:0",
    gold=None,
):
    task_id = TASK2INT[task]
    task_id = torch.tensor([task_id]).long().to(device)
    lang_annotation = val_annotations[task][0]

    goal_encoder = model["goal_encoder"]
    transformer = model["transformer"]

    reset_info = episode["state_info"]

    obs = env.reset(
        robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0]
    )

    start_info = env.get_info()

    camera_obs = env.get_camera_obs()

    assert len(obs["robot_obs"].shape) == 3
    random_window_lenth = np.random.randint(32, 64)
    start_index = 64 - random_window_lenth
    # get random split
    robot_obs = episode["state_info"]["robot_obs"][None, start_index:start_index+1, :]
    scene_obs = episode["state_info"]["scene_obs"][None, start_index:start_index+1, :]
            
    robot_obs_end = episode["state_info"]["robot_obs"][None, -1:, :]
    scene_obs_end = episode["state_info"]["scene_obs"][None, -1:, :]
    end_state = torch.tensor(get_features(scene_obs_end, robot_obs_end, input_features), device=device).float()

    success = False
    for step in range(84):
        inputs = get_features(scene_obs, robot_obs, input_features)
        inputs = torch.tensor(inputs, device=device).float()

        goal_embeddings = goal_encoder(inputs, end_state)

        action = transformer.step(inputs, goal_embeddings)

        obs, _, _, current_info = env.step(action)

        state_obs = env.get_state_obs()
        camera_obs = env.get_camera_obs()

        assert len(state_obs["robot_obs"].shape) == 1
        robot_obs_new = state_obs["robot_obs"][None, None, :]
        scene_obs_new = state_obs["scene_obs"][None, None, :]
        robot_obs = np.concatenate((robot_obs, robot_obs_new), axis=1)
        scene_obs = np.concatenate((scene_obs, scene_obs_new), axis=1)
        robot_obs = robot_obs[:, -64:, :]
        scene_obs = scene_obs[:, -64:, :]


        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {task}
        )
        if len(current_task_info) > 0:
            succes_flag = "Success"
            img = env.render(mode="rgb_array")
            join_vis_lang(
                img,
                lang_annotation,
                out,
                i,
                succes_flag,
                action,
                state_obs["robot_obs"],
                state_obs["scene_obs"],
            )

        else:
            succes_flag = "Fail"
            img = env.render(mode="rgb_array")
            join_vis_lang(
                img,
                lang_annotation,
                out,
                i,
                succes_flag,
                action,
                state_obs["robot_obs"],
                state_obs["scene_obs"],
            )

        if len(current_task_info) > 0:
                print(colored("T", "green"), end=" ")
                return True
        
    print(colored("F", "red"), end=" ")
    return False
