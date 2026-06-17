import torch
import numpy as np
import torch.nn.functional as F

from torch import allclose, argmax, autograd, exp, linspace, nn, sigmoid, sqrt
from torch.special import expm1
from tqdm import tqdm, trange

from model_utils import *
from model_losses import normal_kl, discretized_gaussian_log_likelihood
from model_noise import FixedLinearSchedule, LearnedLinearSchedule


class VDM_Tools():
    def __init__(
                self, 
                noise_steps: int, 
                conditioned_prior: bool, 
                noise_schedule: str, 
                loss: str, 
                device: torch.DeviceObjType, 
                gamma_min=-13.3, 
                gamma_max=5.0
                ):
        self.noise_steps = noise_steps
        self.conditioned_prior = conditioned_prior
        self.vocab_size = 256
        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(gamma_min, gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(gamma_min, gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {noise_schedule}")
        self.loss = loss
        self.device = device

    def init_prior_mean_variance(self, dataloader):
        all_images = []
        for i, (images, _, _, _, _, _, _) in enumerate(dataloader):
            all_images.append(images)

        all_images = torch.cat(all_images, dim=0)
        mean = torch.mean(all_images, dim=0)
        variance = torch.var(all_images, dim=0)

        self.prior_mean = mean
        self.prior_variance = variance
        print("Priors Initialized")

    def sample_timesteps(self, n):
        times = torch.rand(n, device=self.device).requires_grad_(True)
        return times
    
    def get_specific_timesteps(self, timesteps, n):
        """
        Get a tensor of specific timestep.

        Args:
            timesteps: List of timestep to be converted to tensor.
            n: Batch size.

        Returns:
            Tensor of shape (n, len(timesteps)) on the correct device.
        """
        timesteps_tensor = torch.tensor(timesteps / self.noise_steps, dtype=torch.long).to(self.device)
        return timesteps_tensor.repeat(n)

    @torch.no_grad()
    def sample_p_s_t(self, model, n, z, c, t, s):
        """Samples from p(z_s | z_t, x). Used for standard ancestral sampling."""
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        gap = -expm1(gamma_s - gamma_t)
        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        gamma_t_expand = (torch.ones(n) * gamma_t).to(self.device)
        pred_noise = model(z, c, gamma_t_expand)
        
        mean = alpha_s / alpha_t * (z - gap * sigma_t * pred_noise)
        scale = sigma_s * sqrt(gap)
        return mean + scale * torch.randn_like(gap)

    @torch.no_grad()
    def p_sample_loop(self, model, n, c, m, resolution: int = 6):
        if c.shape[0] != n:
            c = concat_to_batchsize(c, n)
        with torch.no_grad():
            z = torch.randn((n, 1, resolution, 6), device=self.device)
            c.to(self.device)
            steps = linspace(1.0, 0.0, self.noise_steps + 1, device=self.device)
            for i in trange(self.noise_steps, desc="sampling"):
                z = self.sample_p_s_t(model, n, z, c, steps[i], steps[i + 1])
            logprobs = self.log_probs_x_z0(z_0=z)  # (B, C, H, W, vocab_size)
            x = argmax(logprobs, dim=-1)  # (B, C, H, W)
            x = x.float() / (self.vocab_size - 1)
            x[:,:, :, -4:] = F.one_hot(torch.argmax(F.softmax(x[:, :, :, -4:], dim=-1), dim=-1), num_classes=4).float()
            x = x
        return x, c

    def sample_q_t_0(self, x, times, noise=None):
        """Samples from the distributions q(x_t | x_0) at the given time steps."""
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(times)
        gamma_t_padded = unsqueeze_right(gamma_t, x.ndim - gamma_t.ndim)
        mean = x * sqrt(sigmoid(-gamma_t_padded))  # x * alpha
        scale = sqrt(sigmoid(gamma_t_padded))
        if noise is None:
            noise = torch.randn_like(x)
        return mean + noise * scale, gamma_t

    def training_losses(self, model, x_start, c, m, t):
        bpd_factor = 1 / (np.prod(x_start.shape[1:]) * np.log(2))
        assert 0.0 <= x_start.min() and x_start.max() <= 1.0

        img_int = torch.round(x_start * (self.vocab_size - 1)).long()
        x_start = 2 * ((img_int + 0.5) / self.vocab_size) - 1

        # Sample from q(x_t | x_0) with random t.
        noise = torch.randn_like(x_start)
        x_t, gamma_t = self.sample_q_t_0(x=x_start, times=t, noise=noise)

        # Forward through model
        model_out = model(x_t, c, gamma_t)

        # *** Diffusion loss (bpd)
        gamma_grad = autograd.grad(  # gamma_grad shape: (B, )
            gamma_t,  # (B, )
            t,  # (B, )
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_loss = ((model_out - noise) ** 2).sum((1, 2, 3))  # (B, )
        diffusion_loss = 0.5 * pred_loss * gamma_grad * bpd_factor

        # *** Latent loss (bpd): KL divergence from N(0, 1) to q(z_1 | x)
        gamma_1 = self.gamma(torch.tensor([1.0], device=self.device))
        sigma_1_sq = sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x_start**2  # (alpha_1 * x)**2
        latent_loss = kl_std_normal(mean_sq, sigma_1_sq).sum((1, 2, 3)) * bpd_factor

        # *** Reconstruction loss (bpd): - E_{q(z_0 | x)} [log p(x | z_0)].
        log_probs = self.log_probs_x_z0(x_start)  # (B, C, H, W, vocab_size)
        x_one_hot = torch.zeros((*x_start.shape, self.vocab_size), device=self.device)
        x_one_hot.scatter_(4, img_int.unsqueeze(-1), 1)  # one-hot over last dim
        log_probs = (x_one_hot * log_probs).sum(-1)  # (B, C, H, W)
        recons_loss = -log_probs.sum((1, 2, 3)) * bpd_factor

        # *** Overall loss in bpd. Shape (B, ).
        loss = diffusion_loss + latent_loss + recons_loss

        with torch.no_grad():
            gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))

        assert self.loss == "vlb"
        metrics = {
            "loss": loss.mean(),
            "diff_loss": diffusion_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "loss_recon": recons_loss.mean(),
            "gamma_0": gamma_0.item(),
            "gamma_1": gamma_1.item(),
        }
        return metrics

    def log_probs_x_z0(self, x=None, z_0=None):
        """Computes log p(x | z_0) for all possible values of x.

        Args:
            x: Input image, shape (B, C, H, W).
            z_0: z_0 to be decoded, shape (B, C, H, W).

        Returns:
            log_probs: Log probabilities of shape (B, C, H, W, vocab_size).
        """
        gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        if x is None and z_0 is not None:
            z_0_rescaled = z_0 / sqrt(sigmoid(-gamma_0))  # z_0 / alpha_0
        elif z_0 is None and x is not None:
            # Equal to z_0/alpha_0 with z_0 sampled from q(z_0 | x)
            z_0_rescaled = x + exp(0.5 * gamma_0) * torch.randn_like(x)  # (B, C, H, W)
        else:
            raise ValueError("Must provide either x or z_0, not both.")
        z_0_rescaled = z_0_rescaled.unsqueeze(-1)  # (B, C, H, W, 1)
        x_lim = 1 - 1 / self.vocab_size
        x_values = linspace(-x_lim, x_lim, self.vocab_size, device=self.device)
        logits = -0.5 * exp(-gamma_0) * (z_0_rescaled - x_values) ** 2  # broadcast x
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, C, H, W, vocab_size)
        return log_probs

def kl_std_normal(mean_squared, var):
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)


class DDPM_Tools:
    def __init__(
                self, 
                noise_steps: int, 
                conditioned_prior: bool, 
                noise_schedule: str,
                loss: str,
                physics_constraint: bool,
                lambda_physics: float,
                device: torch.DeviceObjType, 
                beta_start=1e-4, 
                beta_end=0.02
                ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.conditioned_prior = conditioned_prior
        self.loss = loss
        self.physics_constraint = physics_constraint
        self.lambda_physics = lambda_physics
        self.device = device

        self.prior_mean = None
        self.prior_variance = None

        self.betas = self.prepare_noise_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    def prepare_noise_schedule(self):
        scale = 1000 / self.noise_steps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def load_prior_mean_variance(self, model_path):
        self.prior_mean = torch.load(os.path.join(model_path, "prior_mean.pth"), weights_only=True)
        self.prior_variance = torch.load(os.path.join(model_path, "prior_variance.pth"), weights_only=True)
        print("Priors Initialized")

    def init_prior_mean_variance(self, dataloader, model_path):
        all_images = []
        for i, (images, _) in enumerate(dataloader):
            all_images.append(images)
        all_images = torch.cat(all_images, dim=0)
        
        mean = torch.mean(all_images, dim=0)
        variance = torch.var(all_images, dim=0)

        self.prior_mean = mean
        self.prior_variance = variance
        torch.save(mean, os.path.join(model_path, "prior_mean.pth"))
        torch.save(variance, os.path.join(model_path, "prior_variance.pth"))

    def noise_images(self, x_start, t):
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alphas_hat[t])[:, None, None, None]

        if self.conditioned_prior == True:
            if self.prior_mean == None or self.prior_variance == None:
                raise ValueError("Priors not initialized")
            else:
                mean = self.prior_to_batchsize(self.prior_mean, x_start.shape[0])
                variance = self.prior_to_batchsize(self.prior_variance, x_start.shape[0])
                assert mean.shape == variance.shape == x_start.shape
                noise = torch.randn_like(x_start)
                return (
                    sqrt_alpha_hat * (x_start-mean) + sqrt_one_minus_alpha_hat * noise, noise
                )

        noise = torch.randn_like(x_start)
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def get_specific_timesteps(self, timesteps, n):
        """
        Get a tensor of specific timesteps.

        Args:
            timesteps: List of timesteps to be converted to tensor.
            n: Batch size.

        Returns:
            Tensor of shape (n, len(timesteps)) on the correct device.
        """
        timesteps_tensor = torch.tensor(timesteps, dtype=torch.long).to(self.device)
        return timesteps_tensor.repeat(n)

    def p_sample_loop(self, model, n, c, resolution: int):

        if c.shape[0] != n:
            c = concat_to_batchsize(c, n)

        if self.conditioned_prior == True:
            variance = self.prior_to_batchsize(self.prior_variance, n)
            mean = self.prior_to_batchsize(self.prior_mean, n)

        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, resolution, resolution)).to(self.device)
            c.to(self.device)
            batch_size, channels, height, width = c.shape
            circular_mask = create_circular_mask(height, width).to(c.device)
            circular_mask = circular_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, -1, -1)

            # steps = list(reversed(range(1, self.noise_steps)))
            # for i in tqdm(steps,  desc="sampling loop", leave=False):
            #     t = (torch.ones(n) * i).long().to(self.device)

            #     x = x * circular_mask
            #     s = model(x, c, t)
            #     s = s * circular_mask

            #     alpha = self.alphas[t][:, None, None, None]
            #     alpha_hat = self.alphas_hat[t][:, None, None, None]
            #     beta = self.betas[t][:, None, None, None]

            #     if i > 1:
            #         noise = torch.randn_like(x)
            #         # if self.conditioned_prior == True:
            #         #     noise = noise * torch.sqrt(variance)
            #         x = (1 / torch.sqrt(alpha) * (x- ((1 - alpha)/ (torch.sqrt(1-alpha_hat))) * s) + torch.sqrt(beta) * noise)
            #     else:
            #         x = (1 / torch.sqrt(alpha) * (x- ((1 - alpha)/ (torch.sqrt(1-alpha_hat))) * s))

            t = (torch.ones(n)).long().to(self.device)
            x = x * circular_mask
            x = model(x, c, t)
            x = x * circular_mask
        model.train()

        if self.conditioned_prior == True:
            x = x + mean

        return x, c

    def training_losses(self, model, x_start, c, t):
        """
        Calculate the training losses for a single timestep
        Args:
            model: neural network denoiser
            x_start: original image
            c: conditions
            t: timestep

        Returns:
            loss
        """
        batch_size, channels, height, width = c.shape
        
        circular_mask = create_circular_mask(height, width).to(c.device)
        circular_mask = circular_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, -1, -1)

        x_t, noise = self.noise_images(x_start=x_start, t=t)

        x_t = x_t * circular_mask
        s = model(x_t, c, t)
        s = s * circular_mask
        noise = noise * circular_mask

        assert self.loss == "l2"
        mse_loss = nn.MSELoss()
        loss = mse_loss(s, noise)

        if self.physics_constraint == True:

            # generations, _ = self.p_sample_loop(model, batch_size, c=c, resolution=height)
            # generations = generations * circular_mask

            max_error = torch.mean(torch.abs(torch.max(s) - torch.max(noise)))
            min_error = torch.mean(torch.abs(torch.min(s) - torch.min(noise)))
            var_error = torch.mean(torch.abs(torch.var(s) - torch.var(noise)))
            mean_error = torch.mean(torch.abs(torch.mean(s) - torch.mean(noise)))

            physics_loss = max_error + min_error

            loss += self.lambda_physics * physics_loss

            metrics = {
                "loss": loss,
                "mse_loss": loss,
                "physics_loss": physics_loss
            }
        else:
            metrics = {
                "loss": loss,
                "mse_loss": loss
            }

        return metrics

    def prior_to_batchsize(self, prior, batchsize):
        return prior.unsqueeze(0).expand(batchsize, *prior.shape).to(self.device)