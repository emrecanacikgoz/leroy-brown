import numpy as np
import torch
import torch.nn.functional as F


def process_batch_supervised(model, language_encoder, goal_encoder, opt, lambda_aux, loader, device, train=False):

    loss_eef_pos_value = []
    loss_eef_rot_value = []
    loss_gripper_value = []
    loss_aux_value = []
    for batch in loader:
        inputs, targets, start_states, end_states, mask, label, inputs_rand, _, end_states_rand = batch

        # send to device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        start_states = start_states.to(device, non_blocking=True)
        end_states = end_states.to(device, non_blocking=True) 
        mask = mask.to(device, non_blocking=True) 
        label = label.to(device, non_blocking=True)
        
        inputs_rand = inputs_rand.to(device, non_blocking=True)
        end_states_rand = end_states_rand.to(device, non_blocking=True)

        # encoders
        embed_l = language_encoder(inputs, start_states, label)

        # goal encoder
        embed_g = goal_encoder(inputs_rand, end_states_rand)

        # last-frame padding for auxilary loss
        if embed_g.shape[1] != embed_l.shape[1]:
            last_frame = embed_g[:, -1, :].unsqueeze(1)
            window_diff = embed_l.shape[1] - embed_g.shape[1]
            last_frame = last_frame.repeat(1, window_diff, 1)
            embed_g = torch.cat([embed_g, last_frame], dim=1)

        # auxilary loss
        auxilary_loss = F.mse_loss(embed_g, embed_l, reduction="none").mean()

        # policy
        actions_pred = model(inputs, embed_l)

        if model.output_vocabs is None:
            # FIXME: how can we assert the ordering of pos, rot, grasp is in the same order
            # with the data.
            loss_eef_pos = (
                F.mse_loss(actions_pred[:, :, :3], targets[:, :, :3], reduction="none")
                * mask[:, :, :3]
            ).mean()

            loss_eef_rot = (
                F.mse_loss(actions_pred[:, :, 3:6], targets[:, :, 3:6], reduction="none")
                * mask[:, :, 3:6]
            ).mean()

            loss_eef_grasp = (
                F.binary_cross_entropy_with_logits(
                    actions_pred[:, :, 6:7], targets[:, :, 6:7], reduction="none"
                )
                * mask[:, :, 6:7]
            ).mean()
        else:
            losses = []
            for i in range(actions_pred.shape[2]):
                loss = F.cross_entropy(
                    actions_pred[:, :, i, :].reshape(-1, len(model.output_vocabs[0])-1),
                    targets[:, :, i].reshape(-1),
                    ignore_index=-100
                )
                losses.append(loss)

            losses = torch.stack(losses, dim=0)

            loss_eef_pos, loss_eef_rot, loss_eef_grasp = losses[:3].sum(), losses[3:6].sum(), losses[6:7].sum()

        loss = loss_eef_pos + loss_eef_rot + loss_eef_grasp + lambda_aux*auxilary_loss

        loss_eef_pos_value.append(loss_eef_pos.item())
        loss_eef_rot_value.append(loss_eef_rot.item())
        loss_gripper_value.append(loss_eef_grasp.item())
        loss_aux_value.append(auxilary_loss.item())

        if train:
            opt.zero_grad()
            loss.backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            opt.step()

    loss_dict = {
        "eef_pos": np.array(loss_eef_pos_value),
        "eef_rot": np.array(loss_eef_rot_value),
        "gripper": np.array(loss_gripper_value),
        "auxilary": np.array(loss_aux_value),
    }
    return loss_dict


def train_batch_supervised(model, language_encoder, goal_encoder, opt, lambda_aux, train_loader, device):
    model.train()
    language_encoder.train()
    train_loss = process_batch_supervised(model, language_encoder, goal_encoder, opt, lambda_aux, train_loader, device, train=True)
    return train_loss


def valid_batch_supervised(model, language_encoder, goal_encoder, opt, lambda_aux, valid_loader, device):
    model.eval()
    language_encoder.eval()
    with torch.no_grad():
        valid_loss = process_batch_supervised(model, language_encoder, goal_encoder, opt, lambda_aux, valid_loader, device, train=False)

    return valid_loss