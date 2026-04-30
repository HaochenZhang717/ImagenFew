import torch
import torch.nn as nn

from models.diffusion.DiT_image import DiT_Tiny

from samplers import DDPMSampler, DDIMSampler
import numpy as np
import time
import random

class ImageUnconditionalLatentGenerator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device = configs["device"]
        self.configs = configs
        self._init_diff(configs["diffusion"])

    def _init_diff(self, configs):
        input_dim = configs.get("n_var", 1)
        configs["device"] = self.device

        if configs["type"] == "Text2Ts":
            self.diff_model = DiT_Tiny(configs, input_dim).to(self.device)
        
        self.num_steps = configs["num_steps"]
        self.ddpm = DDPMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)
        self.ddim = DDIMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)

    def _expand_to_x(self, coef, x):
        while coef.dim() < x.dim():
            coef = coef.unsqueeze(-1)
        return coef

    def _ddpm_forward(self, x0, t, noise):
        alpha_bar_sqrt = self._expand_to_x(self.ddpm.alpha_bar_sqrt[t], x0)
        one_minus_alpha_bar_sqrt = self._expand_to_x(self.ddpm.one_minus_alpha_bar_sqrt[t], x0)
        return alpha_bar_sqrt * x0 + one_minus_alpha_bar_sqrt * noise

    def _ddpm_reverse(self, xt, pred_noise, t, noise):
        coef1 = self._expand_to_x(self.ddpm.reverse_coef1[t], xt)
        coef2 = self._expand_to_x(self.ddpm.reverse_coef2[t], xt)
        x_prev = coef1 * (xt - coef2 * pred_noise)

        mask = self._expand_to_x((t > 0).float(), xt)
        sigma = self._expand_to_x(self.ddpm.sigma[t - 1], xt)
        return x_prev + mask * (sigma * noise)

    def _ddim_predict_x0(self, xt, pred_noise, t):
        mask = self._expand_to_x((t == -1).float(), xt)
        coef1 = self._expand_to_x(self.ddim.one_minus_alpha_bar_sqrt[t], xt)
        coef2 = self._expand_to_x(self.ddim.alpha_bar_sqrt_inverse[t], xt)
        pred_x0 = (xt - coef1 * pred_noise) * coef2
        return mask * xt + (1 - mask) * pred_x0

    def _ddim_reverse(self, xt, pred_noise, t, noise, is_determin=False):
        pred_x0 = self._ddim_predict_x0(xt, pred_noise, t)
        mask = self._expand_to_x(t == 0, xt)
        coef1 = self._expand_to_x(self.ddim.alpha_bar_sqrt[t - 1], xt)

        if is_determin:
            coef2 = self._expand_to_x(self.ddim.reverse_coef2_determin[t - 1], xt)
            coef3 = 0
        else:
            coef2 = self._expand_to_x(self.ddim.reverse_coef2[t - 1], xt)
            coef3 = self._expand_to_x(self.ddim.sigma[t - 1], xt)

        x_prev = coef1 * pred_x0 + coef2 * pred_noise + coef3 * noise
        return mask * pred_x0 + (~mask) * x_prev
    
    def _noise_estimation_loss(self, x, tp, attr_emb, t):        
        noise = torch.randn_like(x)
        noisy_x = self._ddpm_forward(x, t, noise)
        pred_noise, loss_dict = self.predict_noise(noisy_x, tp, attr_emb, t)
        residual = noise - pred_noise
        loss_dict["noise_loss"] = (residual ** 2).mean()
        all_loss = torch.zeros_like(loss_dict["noise_loss"])
        for k in loss_dict.keys():
            all_loss += loss_dict[k]
        loss_dict["all"] = all_loss
        return loss_dict
    
    """
    Pretrain.
    """
    def forward(self, batch, is_train):
        x, tp = self._unpack_data_uncond_gen(batch)
        B = x.shape[0]

        if is_train:
            t = torch.randint(0, self.num_steps, [B], device=self.device)
            loss = self._noise_estimation_loss(x, tp, None, t)
            return loss
        
        loss_dict = {}
        for t in range(self.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()
            tmp_loss_dict = self._noise_estimation_loss(x, tp, None, t)
            for k in tmp_loss_dict:
                if k in loss_dict.keys():
                    loss_dict[k] += tmp_loss_dict[k]
                else:
                    loss_dict[k] = tmp_loss_dict[k]
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / self.num_steps
        return loss_dict

    def _unpack_data_uncond_gen(self, batch):
        ts = batch["ts"].to(self.device).float()
        tp = batch["tp"].to(self.device).float()
        ts = ts.permute(0, 2, 1)
        return ts, tp

    """
    Generation.
    """
    @torch.no_grad()
    def generate(self, batch, n_samples, sampler="ddim"):
        ts, tp = self._unpack_data_uncond_gen(batch)
        samples = []
        B = ts.shape[0]
        for i in range(n_samples):
            x = torch.randn_like(ts)
            for t in range(self.num_steps-1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                pred_noise, _ = self.predict_noise(x, tp, None, t)
                if sampler == "ddpm":
                    x = self._ddpm_reverse(x, pred_noise, t, noise)
                else:
                    x = self._ddim_reverse(x, pred_noise, t, noise, is_determin=True)
            samples.append(x)
        return torch.stack(samples)

    def predict_noise(self, xt, tp, attr_emb, t):
        # noisy_x = torch.unsqueeze(xt, 1)
        pred_noise, loss_dict = self.diff_model(xt, attr_emb, t)
        return pred_noise, loss_dict
