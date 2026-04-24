import numpy as np
import torch


class DiffusionProcess:
    def __init__(self, args, diffusion_fn, shape):
        self.args = args
        self.device = args.device
        self.shape = shape
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(
            1 - torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps),
            dim=0,
        ).to(device=self.device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=self.device), self.alpha_bars[:-1]])
        self.deterministic = args.deterministic
        self.net = diffusion_fn.to(device=self.device)
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float("inf")
        self.S_noise = 1
        self.num_steps = args.diffusion_steps
        self.guidance_scale = float(getattr(args, "guidance_scale", 1.0))

    def _denoise(self, x, sigma, text_condition=None):
        if text_condition is None or self.guidance_scale == 1.0:
            return self.net(x, sigma, text_condition).to(torch.float64)

        cond_denoised = self.net(x, sigma, text_condition).to(torch.float64)
        uncond_denoised = self.net(x, sigma, None).to(torch.float64)
        return uncond_denoised + self.guidance_scale * (cond_denoised - uncond_denoised)

    def sample(self, latents, text_condition=None):
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self._denoise(x_hat, t_hat, text_condition)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised = self._denoise(x_next, t_next, text_condition)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    @torch.no_grad()
    def sampling(self, sampling_number=16, impute=False, xT=None, text_condition=None):
        if xT is None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        return self.sample(xT, text_condition=text_condition)
