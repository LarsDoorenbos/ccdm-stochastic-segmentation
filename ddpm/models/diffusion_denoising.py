
from typing import Optional, Tuple, cast, Union
import logging
import math

import torch
from torch import Tensor
from torch import nn
import numpy as np

from .one_hot_categorical import OneHotCategoricalBCHW

LOGGER = logging.getLogger(__name__)

__all__ = ["DiffusionModel", "DenoisingModel"]


def linear_schedule(time_steps: int, start=1e-2, end=0.2) -> Tuple[Tensor, Tensor, Tensor]:
    betas = torch.linspace(start, end, time_steps)
    alphas = 1 - betas
    cumalphas = torch.cumprod(alphas, dim=0)
    return betas, alphas, cumalphas


def cosine_schedule(time_steps: int, s: float = 8e-3) -> Tuple[Tensor, Tensor, Tensor]:
    t = torch.arange(0, time_steps)
    s = 0.008
    cumalphas = torch.cos(((t / time_steps + s) / (1 + s)) * (math.pi / 2)) ** 2

    def func(t): return math.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2

    betas_ = []
    for i in range(time_steps):
        t1 = i / time_steps
        t2 = (i + 1) / time_steps
        betas_.append(min(1 - func(t2) / func(t1), 0.999))
    betas = torch.tensor(betas_)
    alphas = 1 - betas
    return betas, alphas, cumalphas


class DiffusionModel(nn.Module):
    betas: Tensor
    alphas: Tensor
    cumalphas: Tensor

    def __init__(self, schedule: str, time_steps: int, num_classes: int, schedule_params=None):
        super().__init__()

        schedule_func = {
            "linear": linear_schedule,
            "cosine": cosine_schedule
        }[schedule]
        if schedule_params is not None:
            LOGGER.info(f"noise schedule '{schedule}' with params {schedule_params} with time steps={time_steps}")
            betas, alphas, cumalphas = schedule_func(time_steps, **schedule_params)
        else:
            LOGGER.info(f"noise schedule '{schedule}' with default params (schedule_params = {schedule_params})"
                        f" with time steps={time_steps}")
            betas, alphas, cumalphas = schedule_func(time_steps)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("cumalphas", cumalphas)

        self.num_classes = num_classes

    @property
    def time_steps(self):
        return len(self.betas)

    def q_xt_given_xtm1(self, xtm1: Tensor, t: Tensor) -> OneHotCategoricalBCHW:
        t = t - 1

        betas = self.betas[t]
        betas = betas[..., None, None, None]
        probs = (1 - betas) * xtm1 + betas / self.num_classes
        return OneHotCategoricalBCHW(probs)

    def q_xt_given_x0(self, x0: Tensor, t: Tensor) -> OneHotCategoricalBCHW:
        t = t - 1

        cumalphas = self.cumalphas[t]
        cumalphas = cumalphas[..., None, None, None]
        probs = cumalphas * x0 + (1 - cumalphas) / self.num_classes
        return OneHotCategoricalBCHW(probs)

    def theta_post(self, xt: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        t = t - 1

        alphas_t = self.alphas[t][..., None, None, None]
        cumalphas_tm1 = self.cumalphas[t - 1][..., None, None, None]
        alphas_t[t == 0] = 0.0
        cumalphas_tm1[t == 0] = 1.0
        theta = ((alphas_t * xt + (1 - alphas_t) / self.num_classes) *
                 (cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes))
        return theta / theta.sum(dim=1, keepdim=True)

    def theta_post_prob(self, xt: Tensor, theta_x0: Tensor, t: Tensor) -> Tensor:
        """
        This is equivalent to calling theta_post with all possible values of x0
        from 0 to C-1 and multiplying each answer times theta_x0[:, c].

        This should be used when x0 is unknown and what you have is a probability
        distribution over x0. If x0 is one-hot encoded (i.e., only 0's and 1's),
        use theta_post instead.
        """
        t = t - 1

        alphas_t = self.alphas[t][..., None, None, None]
        cumalphas_tm1 = self.cumalphas[t - 1][..., None, None, None, None]
        alphas_t[t == 0] = 0.0
        cumalphas_tm1[t == 0] = 1.0

        # We need to evaluate theta_post for all values of x0
        x0 = torch.eye(self.num_classes, device=xt.device)[None, :, :, None, None]
        # theta_xt_xtm1.shape == [B, C, H, W]
        theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) / self.num_classes
        # theta_xtm1_x0.shape == [B, C1, C2, H, W]
        theta_xtm1_x0 = cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes

        aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
        # theta_xtm1_xtx0 == [B, C1, C2, H, W]
        theta_xtm1_xtx0 = aux / aux.sum(dim=1, keepdim=True)

        # theta_x0.shape = [B, C, H, W]

        return torch.einsum("bcdhw,bdhw->bchw", theta_xtm1_xtx0, theta_x0)


class DenoisingModel(nn.Module):

    def __init__(self, diffusion: DiffusionModel, unet: nn.Module, dataset_file: str, step_T_sample:str = "majority"):
        super().__init__()
        self.diffusion = diffusion
        self.unet = unet
        self.dataset_file = dataset_file
        self.step_T_sample = step_T_sample

    @property
    def time_steps(self):
        return self.diffusion.time_steps

    def forward(self, x: Tensor, condition: Tensor, feature_condition: Tensor = None, t: Optional[Tensor] = None, label_ref_logits: Optional[Tensor] = None,
                validation: bool = False) -> Union[Tensor, dict]:

        if self.training:
            if not isinstance(t, Tensor):
                raise ValueError("'t' needs to be a Tensor at training time")
            if not isinstance(x, Tensor):
                raise ValueError("'x' needs to be a Tensor at training time")
            return self.forward_step(x, condition, feature_condition, t) 
        else:
            if validation:
                return self.forward_step(x, condition, feature_condition, t)
            if t is None:
                return self.forward_denoising(x, condition, feature_condition, label_ref_logits=label_ref_logits)

            return self.forward_denoising(x, condition, feature_condition, cast(int, t.item()), label_ref_logits)

    def forward_step(self, x: Tensor, condition: Tensor, feature_condition: Tensor, t: Tensor) -> Tensor:
        return self.unet(x, condition, feature_condition=feature_condition, timesteps=t)

    def forward_denoising(self, x: Optional[Tensor], condition: Tensor, feature_condition: Tensor, init_t: Optional[int] = None,
                          label_ref_logits: Optional[Tensor] = None) -> dict:

        if init_t is None:
            init_t = self.time_steps

        xt = x

        if label_ref_logits is not None:
            weights = self.guidance_scale_weights(label_ref_logits)
            label_ref = label_ref_logits.argmax(dim=1)

        shape = xt.shape

        if init_t > 10000:
            K = init_t % 10000
            assert 0 < K <= self.time_steps
            if K == self.time_steps:
                t_values = range(K, 0, -1)
            else:
                t_values = [round(t_val) for t_val in np.linspace(self.time_steps, 1, K)]
                LOGGER.warning(f"Override default {self.time_steps} time steps with {len(t_values)}.")
        else:
            t_values = range(init_t, 0, -1)

        for t in t_values:
            # Auxiliary values
            t_ = torch.full(size=(shape[0],), fill_value=t, device=xt.device)

            # Predict the noise of x_t
            ret = self.unet(xt, condition, feature_condition, t_.float())
            x0pred = ret["diffusion_out"]

            probs = self.diffusion.theta_post_prob(xt, x0pred, t_)

            if label_ref_logits is not None:
                if self.guidance_scale > 0:
                    gradients = self.guidance_fn(probs, label_ref if self.guidance_loss_fn_name == 'CE' else label_ref_logits, weights)
                    probs = probs - gradients

            probs = torch.clamp(probs, min=1e-12)

            if t > 1:
                xt = OneHotCategoricalBCHW(probs=probs).sample()
            else:
                if self.step_T_sample is None or self.step_T_sample == "majority":
                    xt = OneHotCategoricalBCHW(probs=probs).max_prob_sample()
                elif self.step_T_sample == "confidence":
                    xt = OneHotCategoricalBCHW(probs=probs).prob_sample()

        ret = {"diffusion_out": xt}
        return ret

    def _check_tensor(self, tensor: Tensor) -> list:

        invalid_values = []

        if torch.isnan(tensor).any():
            LOGGER.error("nan found in tensor!!")
            invalid_values.append("nan")

        if torch.isinf(tensor).any():
            LOGGER.error("inf found in tensor!!")
            invalid_values.append("inf")

        if (tensor.sum(dim=1) < -1e-3).any():
            LOGGER.error("negative KL divergence in tensor!!")
            invalid_values.append("neg")

        return invalid_values