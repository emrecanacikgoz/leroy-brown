import csv
import os

import torch


class CSVWriter:
    def __init__(self, cfg) -> None:
        experiments_path = "experiments"
        experiment_path = os.path.join(experiments_path, cfg.experiment)

        if not os.path.exists(os.path.join(experiment_path, "logs")):
            os.makedirs(os.path.join(experiment_path, "logs"))

        self.log_path = os.path.join(experiment_path, "logs", "logs.csv")
        self.f = open(self.log_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.f)
        self.csv_writer.writerow(
            [
                "epoch",
                "train_eef_pos",
                "train_eef_rot",
                "train_gripper",
                "valid_eef_pos",
                "valid_eef_rot",
                "valid_gripper",
            ]
        )

    def log(self, row):
        self.csv_writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


class CheckpointSaver:
    def __init__(self, experiment_name) -> None:
        experiments_path = "experiments"
        experiment_path = os.path.join(experiments_path, experiment_name)
        if not os.path.exists(os.path.join(experiment_path, "weights")):
            os.makedirs(os.path.join(experiment_path, "weights"))
        self.weight_path = os.path.join(experiment_path, "weights")

    def get_save_path(self, epoch):
        return os.path.join(self.weight_path, f"last-{epoch}.pt")

    def save_last(self, policy, language_encoder, opt, scheduler, epoch):
        path = self.get_save_path(epoch)
        checkpoint = {
            "model": policy.state_dict(),
            "language_encoder": language_encoder.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "output_vocabs": policy.output_vocabs,
        }
        torch.save(checkpoint, path)
        return path

    def save_last_supervised(self, policy, language_encoder, goal_encoder, opt, scheduler, epoch):
        path = self.get_save_path(epoch)
        checkpoint = {
            "model": policy.state_dict(),
            "language_encoder": language_encoder.state_dict(),
            "goal_encoder": goal_encoder.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "output_vocabs": policy.output_vocabs,
        }
        torch.save(checkpoint, path)
        return path
    
    def save_last_unsupervised(self, policy, goal_encoder, opt, scheduler, epoch):
        path = self.get_save_path(epoch)
        checkpoint = {
            "goal_encoder": goal_encoder.state_dict(),
            "model": policy.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "output_vocabs": policy.output_vocabs,
        }
        torch.save(checkpoint, path)
        return path

    def load_last(self, device="cpu"):
        torch.load(os.path.join(self.weight_path, "last.pt"), map_location=device)

    def save_best(self, model, epoch):
        path = os.path.join(self.weight_path, f"best.pt")
        torch.save(model.state_dict(), path)
        return path
 