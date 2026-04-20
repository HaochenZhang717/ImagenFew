import os
import time
import numpy as np
import torch
from torch.optim import Adam

import data.data
# from torch.utils.tensorboard import SummaryWriter
from data import GenerationDataset
from evaluation.base_evaluator import BaseEvaluator
import matplotlib.pyplot as plt
import wandb
import copy
from tqdm import tqdm


class Trainer:
    """
    Trainer for conditional generation model.
    """

    def __init__(self, configs, eval_configs, dataset, model):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_data(dataset)
        self._init_opt()
        # self._init_eval(eval_configs)
        self._best_valid_loss = 1e10
        wandb.init(
            project="verbalts",  # 你可以改名字
            name=os.getenv("WANDB_NAME", "no_name"),  # 自动用实验名
            config=self.configs
        )

    def update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _init_eval(self, eval_configs):
        dataset = GenerationDataset(eval_configs["data"])
        self.evaluator = BaseEvaluator(eval_configs["eval"], dataset, None)
        self.eval_configs = eval_configs

    def _init_cfgs(self, configs):
        self.configs = configs

        self.n_epochs = self.configs["epochs"]
        self.itr_per_epoch = self.configs["itr_per_epoch"]
        self.valid_epoch_interval = self.configs["val_epoch_interval"]
        self.display_epoch_interval = self.configs["display_interval"]

        self.lr = self.configs["lr"]
        self.batch_size = self.configs["batch_size"]

        self.model_path = self.configs["model_path"]
        self.output_folder = configs["output_folder"]
        os.makedirs(self.output_folder, exist_ok=True)

    def _init_model(self, model):
        self.model = model
        self.ema_model = copy.deepcopy(self.model)
        if self.model_path != "":
            print("Loading pretrained model from {}".format(self.model_path))
            ckpt = torch.load(self.model_path)
            if "ema_model" in ckpt:
                self.model.load_state_dict(ckpt["model"])
                self.ema_model.load_state_dict(ckpt["ema_model"])
            else:
                self.model.load_state_dict(ckpt)

        self.ema_model.eval()
        self.ema_decay = 0.999  # 可以调 0.999~0.9999

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params}")

    def _init_opt(self):
        self.opt = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)

        if os.getenv("SCHEDULER") == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                T_max=self.n_epochs,
                eta_min=0.5 * self.lr
            )
        elif os.getenv("SCHEDULER") == "MULTISTEP":
            p1 = int(0.75 * self.n_epochs)
            p2 = int(0.9 * self.n_epochs)
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.opt, milestones=[p1, p2], gamma=0.1)
        else:
            raise NotImplementedError

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.opt,
        #     T_0=100,  # 第一个周期长度
        #     T_mult=2,  # 每个周期长度倍增
        #     eta_min=0.5 * self.lr  # 最小学习率
        # )

    def _init_data(self, dataset):
        self.dataset = dataset
        # if isinstance(self.dataset.dataset, data.data.V7Dataset):
        folder = self.dataset.configs['folder']
        self.long_clip_embeds_train = torch.load(
            os.path.join(folder, f"train_embeds_long_clip.pt"),
            map_location="cuda"
        )
        self.long_clip_embeds_valid = torch.load(
            os.path.join(folder, f"valid_embeds_long_clip.pt"),
            map_location="cuda"
        )

        self.train_loader = dataset.get_loader(split="train", batch_size=self.batch_size, shuffle=True,
                                               include_self=True)
        self.valid_loader = dataset.get_loader(split="valid", batch_size=self.batch_size, shuffle=False,
                                               include_self=True)

    def _reset_train(self):
        self._best_valid_loss = 1e10
        self._global_batch_no = 0

    """
    Train.
    """
    def train(self):
        self._reset_train()
        for epoch_no in range(self.n_epochs):
            self._train_epoch(epoch_no)
            if self.valid_loader is not None and (epoch_no + 1) % self.valid_epoch_interval == 0:
                self.valid(epoch_no)

            if (epoch_no + 1) % 100 == 0:
                self.evaluate(epoch_no)

    def evaluate(self, epoch_no):
        self.ema_model.eval()
        self.evaluator.model = self.ema_model
        self.evaluator.n_samples = 10
        _, samples = self.evaluator.evaluate(mode="cond_gen", sampler="ddim", save_pred=False)
        path = os.path.join(fr"{self.output_folder}", f"samples_during_training_Epoch{epoch_no}.pt")
        torch.save(samples, path)

    def _train_epoch(self, epoch_no):
        start_time = time.time()
        avg_loss = 0
        self.model.train()

        for batch_no, train_batch in tqdm(enumerate(self.train_loader)):
            self._global_batch_no += 1
            self.opt.zero_grad()
            if hasattr(self, "long_clip_embeds_train"):
                train_batch["cap_embed"] = self.long_clip_embeds_train["embeddings"][train_batch["indices"]]
                train_batch["cap_embed_mask"] = self.long_clip_embeds_train["all_masks"][train_batch["indices"]]
            loss_dict = self.model(train_batch, is_train=True)

            loss_dict["all"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.update_ema()

            avg_loss += loss_dict["all"].item()

            if batch_no >= self.itr_per_epoch:
                break

        avg_loss /= (batch_no + 1)
        # self.tf_writer.add_scalar("Train/epoch_loss", avg_loss, epoch_no)
        # self.tf_writer.add_scalar("Train/lr", self.opt.param_groups[0]['lr'], epoch_no)
        end_time = time.time()
        self.lr_scheduler.step()

        if (epoch_no + 1) % self.display_epoch_interval == 0:
            print("Epoch:", epoch_no,
                  "Loss:", avg_loss,
                  "Time: {:.2f}".format(end_time - start_time))

            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/learning_rate": self.opt.param_groups[0]['lr'],
                "epoch": epoch_no
            })

    """
    Valid.
    """

    def valid(self, epoch_no=-1):
        self.ema_model.eval()
        avg_loss_valid = 0
        with torch.no_grad():
            for batch_no, valid_batch in enumerate(self.valid_loader):
                if hasattr(self, "long_clip_embeds_valid"):
                    # valid_batch["my_cap_embed"] = self.long_clip_embeds_train["embeddings"][valid_batch["indices"]]
                    valid_batch["cap_embed"] = self.long_clip_embeds_valid["embeddings"][valid_batch["indices"]]
                    valid_batch["cap_embed_mask"] = self.long_clip_embeds_valid["all_masks"][valid_batch["indices"]]

                loss_dict = self.ema_model(valid_batch, is_train=False)
                avg_loss_valid += loss_dict["all"].item()

        avg_loss_valid = avg_loss_valid / (batch_no + 1)
        # self.tf_writer.add_scalar("Valid/epoch_loss", avg_loss_valid, epoch_no)
        wandb.log({
            "valid/loss": avg_loss_valid,
            "epoch": epoch_no
        })
        if self._best_valid_loss > avg_loss_valid:
            self._best_valid_loss = avg_loss_valid
            print(f"\n*** Best loss is updated to {avg_loss_valid} at {epoch_no}.\n")
            self.save_model("model_best_loss")
        if (epoch_no + 1) % 100 == 0:
            self.save_model(f"model_epoch_{epoch_no}")

    """
    Save.
    """

    def save_model(self, comment):
        os.makedirs(fr"{self.output_folder}/ckpts", exist_ok=True)
        path = os.path.join(fr"{self.output_folder}/ckpts", f"{comment}.pth")

        torch.save({
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "comment": comment,
        }, path)

