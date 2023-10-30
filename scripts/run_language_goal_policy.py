import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from blind_robot.utils.helpers import set_seed
from blind_robot.model.action_decoder import ActionDecoder
from blind_robot.model.encoders.language_encoder import LanguageEncoder
from blind_robot.model.encoders.goal_encoder import StateGoalEncoder
from blind_robot.training.training_lang_goal import train_batch_supervised
from blind_robot.training.training_lang_goal import valid_batch_supervised
from blind_robot.training.utils import CheckpointSaver
from blind_robot.dataset.dataset_goal_supervised import CalvinDatasetGoalSupervised as CalvinDataset
from evaluation.evaluate import main as evaluate


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.wandb.project, config=config_dict, entity="akyurek")
    device = cfg.training.device

    train_data = CalvinDataset(
        path=cfg.data.train_path,
        input_features=cfg.data.input_features,
        target_features=cfg.data.target_features,
        window=cfg.data.window,
        num_bins=cfg.data.num_bins,
        add_gaussian_noise=cfg.data.add_gaussian_noise,
    )

    valid_data = CalvinDataset(
        path=cfg.data.val_path,
        input_features=cfg.data.input_features,
        target_features=cfg.data.target_features,
        window=cfg.data.window,
        target_vocabs=train_data.target_vocabs,
        add_gaussian_noise=False,
    )

    #train_data = torch.utils.data.ConcatDataset([train_data, valid_data])

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle_train,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=True,
        drop_last=False,
        collate_fn=train_data.collate_fn,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle_val,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=True,
        drop_last=False,
        collate_fn=valid_data.collate_fn,
    )
    n_input = train_data.input_dim()

    # set encoders
    language_encoder = LanguageEncoder(
        in_features=n_input,
        task_embedding_size=cfg.model.language_encoder.task_embedding_size,
        hidden_size=cfg.model.language_encoder.language_encoder_hidden_size,
        latent_language_encoder_features=cfg.model.language_encoder.latent_language_encoder_features,
        num_tasks=cfg.data.num_tasks,
        l2_normalize_language_embeddings=cfg.model.language_encoder.l2_normalize_language_embeddings,
    ).to(device)

    goal_encoder = StateGoalEncoder(
        in_features=n_input,
        hidden_size=cfg.model.goal_encoder.hidden_size,
        latent_goal_encoder_features=cfg.model.goal_encoder.latent_features,
        l2_normalize_embeddings=cfg.model.goal_encoder.l2_normalize_embeddings,
    ).to(device)
    
    # set policy
    policy = ActionDecoder(
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
    
    print(f"policy architecture ------- \n {policy}")

    # set optimizer
    opt = torch.optim.AdamW(
        params=list(policy.parameters()) + list(language_encoder.parameters()) + list(goal_encoder.parameters()),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    opt.grad_clip = cfg.optimizer.grad_clip
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt, T_max=cfg.training.num_epochs, eta_min=1e-6
    )

    # set checkpoints
    checkpoint_saver = CheckpointSaver(cfg.name)

    # training loop
    for epoch in tqdm(range(cfg.training.num_epochs)):

        train_loss = train_batch_supervised(policy, language_encoder, goal_encoder, opt, cfg.optimizer.lamda_auxiliary, train_loader, device)
        valid_loss = valid_batch_supervised(policy, language_encoder, goal_encoder, opt, cfg.optimizer.lamda_auxiliary, valid_loader, device)
        scheduler.step()

        wandb.log(
            {
                "train": {k: l.mean() for k, l in train_loss.items()},
                "val": {k: l.mean() for k, l in valid_loss.items()},
                "lr": scheduler.get_last_lr()[0],
            }
        )

        if (epoch % 10) == 0:
            _ = checkpoint_saver.save_last_supervised(policy, language_encoder, goal_encoder, opt, scheduler, epoch)
            cfg.evaluation.checkpoint = checkpoint_saver.get_save_path(epoch)
            cfg.evaluation.calvin.subset = "validation"
            val_acc = evaluate(cfg)
            cfg.evaluation.calvin.subset = "training"
            train_acc = evaluate(cfg)
            wandb.log({"val_sim_acc": val_acc, "train_sim_acc": train_acc})



if __name__ == "__main__":
    main()  # pylint: disable=E1120
