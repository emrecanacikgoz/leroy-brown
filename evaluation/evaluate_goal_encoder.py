import argparse
from collections import Counter
import os
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch
import wandb

from blind_robot.utils.helpers import set_seed
from blind_robot.model.action_decoder import ActionDecoder
from blind_robot.model.encoders.goal_encoder import StateGoalEncoder
from blind_robot.dataset.dataset import CalvinDataset
from evaluation.rollout_goal_encoder import rollout
from evaluation.utils import get_default_env
from evaluation.utils import load_checkpoint
from evaluation.utils import TASK_TO_ID_DICT_ROTATE_VAL_64, TASK_TO_ID_DICT_ROTATE_TRAIN_64


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.wandb.project, config=config_dict)

    checkpoint = torch.load(cfg.evaluation.checkpoint)
    subset = cfg.evaluation.calvin.subset
    device = cfg.evaluation.device

    train_data = CalvinDataset(
        path=cfg.data.train_path,
        input_features=cfg.data.input_features,
        target_features=cfg.data.target_features,
        window=cfg.data.window,
        target_vocabs=checkpoint["output_vocabs"],
        add_gaussian_noise=False,
    )

    valid_data = CalvinDataset(
        path=cfg.data.val_path,
        input_features=cfg.data.input_features,
        target_features=cfg.data.target_features,
        window=cfg.data.window,
        target_vocabs=checkpoint["output_vocabs"],
        add_gaussian_noise=False,
    )

    env, data_module, lang_embeddings = get_default_env(
        cfg.evaluation.calvin.train_folder,
        cfg.evaluation.calvin.dataset_path,
        env=None,
        lang_embeddings=None,
        device_id=0,
    )
    print(data_module)
    print("Step1 - loading env: DONE")

    # set task set-up
    conf_dir = Path(cfg.evaluation.calvin.conf_dir)
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )

    if subset == "validation":
        task_to_id_dict = TASK_TO_ID_DICT_ROTATE_VAL_64  # take from their checkpoint
        dataset = data_module.val_dataloader().dataset.datasets["vis"]
        gold_data = valid_data
    elif subset == "training":
        task_to_id_dict = TASK_TO_ID_DICT_ROTATE_TRAIN_64 # take from their checkpoint
        dataset = data_module.train_dataloader()["vis"].dataset
        gold_data = train_data
    else:
        raise ValueError(f"Unknown subset {subset}")
    # gold_data_map = {}
    # for d in gold_data:
    #     inp, tgt, ss, es, m, lbl = d
    #     gold_data_map [sid] = (inp, tgt, lbl)
    print("Step2 - task set-up: DONE")
    n_input = valid_data.input_dim()

    # set encoders
    goal_encoder = StateGoalEncoder(
        in_features=n_input,
        hidden_size=cfg.model.goal_encoder.hidden_size,
        latent_goal_encoder_features=cfg.model.goal_encoder.latent_features,
        l2_normalize_embeddings=cfg.model.goal_encoder.l2_normalize_embeddings,
    ).to(device)
    
    # set policy
    transformer = ActionDecoder(
        n_input=n_input,
        context_size=cfg.data.window*2,
        n_heads=cfg.model.n_heads,
        n_layer=cfg.model.n_layer,
        hidden_size=cfg.model.hidden_size,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        num_tasks=cfg.data.num_tasks,
        embedding_size=cfg.model.goal_encoder.latent_features,
        output_vocabs=train_data.target_vocabs,
    ).to(device)

    # load weights
    device = cfg.evaluation.device
    model = load_checkpoint(language_encoder=None, goal_encoder=goal_encoder, transformer=transformer, checkpoint=checkpoint, checkpoint_path=cfg.evaluation.checkpoint, device=device)
    print("Step3 - loading model: DONE")

    # create folder for videos
    epoch_num = cfg.evaluation.checkpoint.split("/")[-1].split(".")[0].split("-")[-1]
    vidoe_path = f"videos/{cfg.name}/{subset}"
    if not os.path.exists(vidoe_path):
        os.makedirs(vidoe_path)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")

    # evaluate-train
    results_false = {}
    results = Counter()
    res, len_ids = 0, 0
    for task, ids in task_to_id_dict.items():
        output_filename = os.path.join(vidoe_path, f"{task}-epoch_{epoch_num}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        frame_width, frame_height = 500, 500
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
        for i in ids:
            episode = dataset[int(i)]
            #gold = gold_data_map[int(i)]
            res_task = rollout(
                env,
                model,
                episode,
                task_oracle,
                task,
                lang_embeddings,
                val_annotations,
                cfg.evaluation.add_noise_during_inference,
                out,
                i,
                cfg.data.input_features,
                device=device,
                gold=None
            )
            results[task] += res_task
            if not res_task:
                results_false[task] = id
        print(f"{task}: {results[task]} / {len(ids)}")
        wandb.log({f"accuracy/{subset}/{task}": results[task] / len(ids)})
        out.release()
        res += results[task]
        len_ids += len(ids)
        total_eval_acc = res / len_ids * 100
        # wandb.log({"evaluation_accuracy_val": total_eval_acc})

    total_eval_acc = (
        sum(results.values()) / sum(len(x) for x in task_to_id_dict.values()) * 100
    )

    print(
        f"ACC: {total_eval_acc:.2f}"
    )

    wandb.log({f"accuracy/{subset}/total": total_eval_acc})

    print("\nStep4 - evaluate tasks: DONE")
    print(results_false)
    print("Evaluation Done... Terminating...")
    return total_eval_acc


if __name__ == "__main__":
    main()  # pylint: disable=E1120