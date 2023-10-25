from pathlib import Path

from calvin_agent.utils.utils import format_sftp_path
import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from scipy.spatial.transform import Rotation as R

def get_default_env(
    train_folder, dataset_path, env=None, lang_embeddings=None, device_id=0, subset="validation"
):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    print(f"train_cfg_path: {train_cfg_path}")
    cfg = OmegaConf.load(train_cfg_path)

    # cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace("calvin_models.", ""))
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    # print(cfg)
    # print(f"language folder: {lang_folder}")

    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        datasets_path = "calvin/calvin_models/conf/datamodule/datasets"
        # print(f"Dataset path: {datasets_path}")
        hydra.initialize(datasets_path)

    # we don't want to use shm dataset for evaluation
    # datasets_cfg = hydra.compose("vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder])

    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.datasets = {
        "vision_dataset": {
            "_target_": "calvin_agent.datasets.disk_dataset.DiskDataset",
            "key": "vis",
            "save_format": "npz",
            "batch_size": 32,
            "min_window_size": 16,
            "max_window_size": 32,
            "proprio_state": "${datamodule.proprioception_dims}",
            "obs_space": "${datamodule.observation_space}",
            "pad": True,
            "lang_folder": "lang_annotations",
            "num_workers": 2,
            # "datasets_dir": os.path.join(dataset_path, subset),
        },
        "lang_dataset": {
            "_target_": "calvin_agent.datasets.disk_dataset.DiskDataset",
            "key": "lang",
            "save_format": "npz",
            "batch_size": 32,
            "min_window_size": 16,
            "max_window_size": 32,
            "proprio_state": "${datamodule.proprioception_dims}",
            "obs_space": "${datamodule.observation_space}",
            "skip_frames": 1,
            "pad": True,
            "lang_folder": "lang_annotations",
            "num_workers": 2,
            # "datasets_dir": os.path.join(dataset_path, subset),
        },
    }
    cfg.datamodule.root_data_dir = dataset_path
    # print(dataset_path)
    # switch to rel_action mode at output
    cfg.datamodule["observation_space"]["actions"] = ["rel_actions"]

    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    # print(cfg.datamodule)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device(f"cuda:{device_id}")
    # print(f"Abs-datasets-dir: {dataset.abs_datasets_dir}")
    if lang_embeddings is None:
        lang_embeddings = LangEmbeddings(
            dataset.abs_datasets_dir, lang_folder, device=device
        )

    if env is None:
        rollout_cfg = OmegaConf.load(
            "evaluation/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
        )
        env = hydra.utils.instantiate(
            rollout_cfg.env_cfg, dataset, device, show_gui=False
        )

    return env, data_module, lang_embeddings


class LangEmbeddings:
    def __init__(self, val_dataset_path, lang_folder, device=torch.device("cuda:0")):
        print()
        embeddings = np.load(
            Path(val_dataset_path) / lang_folder / "embeddings.npy", allow_pickle=True
        ).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
        self.device = device

    def get_lang_goal(self, task):
        return {
            "lang": (
                torch.from_numpy(self.lang_embeddings[task])
                .to(self.device)
                .squeeze(0)
                .float()
            )
        }
    

def load_checkpoint(language_encoder, transformer, checkpoint, checkpoint_path, device):
    language_encoder.to(device)
    transformer.to(device)
    print(next(transformer.parameters()).device)
    print(f"Loading checkpoints from {checkpoint_path}")
    transformer.load_state_dict(checkpoint["model"])
    transformer.eval()
    transformer.to(device)
    language_encoder.load_state_dict(checkpoint["language_encoder"])
    language_encoder.eval()
    language_encoder.to(device)
    print("Successfully loaded...")
    
    return {"language_encoder": language_encoder, "transformer": transformer}


def join_vis_lang(img, lang_text, out, id, succes_flag, action, robot_obs, scene_obs):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    lang_text = lang_text + f"_{id}: {succes_flag}"
    add_text(img, lang_text)

    actions = action.tolist()
    robot_obs = robot_obs.tolist()
    scene_obs = scene_obs.tolist()

    lang_text = (
        f"action:     [{actions[0]:.3f}, {actions[1]:.3f}, {actions[2]:.3f},"
        f" {actions[3]:.3f}, {actions[4]:.3f}, {actions[5]:.3f}, {actions[6]:.1f}]"
    )
    add_text_fields(img, lang_text, coord=(1, 13))

    lang_text = (
        f"tcp_action: [{robot_obs[0]:.3f}, {robot_obs[1]:.3f}, {robot_obs[2]:.3f},"
        f" {robot_obs[3]:.3f}, {robot_obs[4]:.3f}, {robot_obs[5]:.3f},"
        f" {robot_obs[14]:.1f}, {robot_obs[6]:.3f}]"
    )
    add_text_fields(img, lang_text, coord=(1, 26))

    lang_text = (
        f"scene: [{scene_obs[0]:.3f}, {scene_obs[1]:.3f}, {scene_obs[2]:.3f},"
        f" {scene_obs[3]:.3f}, {scene_obs[4]:.1f}, {scene_obs[5]:.1f}]"
    )
    add_text_fields(img, lang_text, coord=(1, 39))

    lang_text = (
        f"red-block:  [{scene_obs[6]:.3f}, {scene_obs[7]:.3f}, {scene_obs[8]:.3f},"
        f" {scene_obs[9]:.3f}, {scene_obs[10]:.3f}, {scene_obs[11]:.3f}]"
    )
    add_text_fields(img, lang_text, coord=(1, 52))
    lang_text = (
        f"blue-block: [{scene_obs[12]:.3f}, {scene_obs[13]:.3f}, {scene_obs[14]:.3f},"
        f" {scene_obs[15]:.3f}, {scene_obs[16]:.3f}, {scene_obs[17]:.3f}]"
    )
    add_text_fields(img, lang_text, coord=(1, 65))
    lang_text = (
        f"pink-block: [{scene_obs[18]:.3f}, {scene_obs[19]:.3f}, {scene_obs[20]:.3f},"
        f" {scene_obs[21]:.3f}, {scene_obs[22]:.3f}, {scene_obs[23]:.3f}]"
    )
    add_text_fields(img, lang_text, coord=(1, 78))

    out.write(img)


def add_text_fields(img, lang_text, coord=None):
    height, width, _ = img.shape
    if lang_text != "":
        font_scale = (0.35 / 500) * width
        thickness = 1
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=thickness * 3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def add_text(img, lang_text):
    height, width, _ = img.shape
    if lang_text != "":
        coord = (1, int(height - 10))
        font_scale = (0.5 / 500) * width
        thickness = 1
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=thickness * 3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


TASK_TO_ID_DICT_ROTATE_VAL_64 = {
    "rotate_red_block_right": [
        403887,
        403896,
        16795,
        36345,
        36338,
        8310,
        12983,
        16382,
        36344,
        403883,
        12974,
        46599,
        403888,
        16375,
        16374,
        12981,
        403889,
        36346,
        8303,
        8308,
        16794,
        46596,
        8309,
        403886,
        16384,
        403882,
        36343,
        36350,
        12982,
    ],
}

TASK_TO_ID_DICT_ROTATE_TRAIN_64 = {
    "rotate_red_block_right": [
        113783,
        73269,
        77302,
        113780,
        87103,
        57352,
        181006,
        153997,
        154005,
        77288,
        87112,
        133249,
        154345,
        77284,
        191588,
        77294,
        57348,
        181016,
        132055,
        77298,
        87209,
        135559,
        57347,
        135553,
        133253,
        154346,
        154001,
        195772,
        113797,
        99992,
        99984,
        135548,
        77281,
        57357,
        113781,
        87105,
        100481,
        154349,
        191593,
        87092,
        135547,
        87102,
        191592,
        100484,
        191586,
        132056,
        191583,
        94331,
        195760,
        132064,
        57358,
        57350,
        132050,
        181025,
        181009,
    ],
}

task2int = {
    "close_drawer": 0,
    "lift_blue_block_drawer": 1,
    "lift_blue_block_slider": 2,
    "lift_blue_block_table": 3,
    "lift_pink_block_drawer": 4,
    "lift_pink_block_slider": 5,
    "lift_pink_block_table": 6,
    "lift_red_block_drawer": 7,
    "lift_red_block_slider": 8,
    "lift_red_block_table": 9,
    "move_slider_left": 10,
    "move_slider_right": 11,
    "open_drawer": 12,
    "place_in_drawer": 13,
    "place_in_slider": 14,
    "push_blue_block_left": 15,
    "push_blue_block_right": 16,
    "push_into_drawer": 17,
    "push_pink_block_left": 18,
    "push_pink_block_right": 19,
    "push_red_block_left": 20,
    "push_red_block_right": 21,
    "rotate_blue_block_left": 22,
    "rotate_blue_block_right": 23,
    "rotate_pink_block_left": 24,
    "rotate_pink_block_right": 25,
    "rotate_red_block_left": 26,
    "rotate_red_block_right": 27,
    "stack_block": 28,
    "turn_off_led": 29,
    "turn_off_lightbulb": 30,
    "turn_on_led": 31,
    "turn_on_lightbulb": 32,
    "unstack_block": 33,
}
