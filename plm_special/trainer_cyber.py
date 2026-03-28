import time
import numpy as np
import torch

from torch.utils.data import DataLoader

from plm_special.utils.utils_cyber import masked_cross_entropy_loss, process_batch_cyber


class CyberTrainer:
    def __init__(self, args, model, optimizer, exp_dataset, device, action_dim: int, batch_size=1, grad_accum_steps=1, lr_scheduler=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.exp_dataset = exp_dataset
        self.device = device
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr_scheduler = lr_scheduler
        self.dataloader = DataLoader(exp_dataset, batch_size, shuffle=True, pin_memory=True)

    def train_epoch(self, report_loss_per_steps=100):
        train_losses = []
        logs = {}
        train_start = time.time()
        dataset_size = len(self.dataloader)

        self.model.train()
        for step, batch in enumerate(self.dataloader):
            loss = self.train_step(batch)
            train_losses.append(loss.item())

            loss = loss / self.grad_accum_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            if ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == dataset_size):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if step % report_loss_per_steps == 0:
                print(f"Step {step} - mean train loss {np.mean(train_losses):>9f}")

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = float(np.mean(train_losses))
        logs["training/train_loss_std"] = float(np.std(train_losses))
        return logs, train_losses

    def train_step(self, batch):
        states, actions_in, returns, timesteps, labels, masks = process_batch_cyber(batch, device=self.device, action_dim=self.action_dim)
        logits = self.model(states, actions_in, returns, timesteps)  # (1,T,A)
        loss = masked_cross_entropy_loss(logits, labels, masks)
        return loss

