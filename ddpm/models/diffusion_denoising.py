from typing import Optional, Tuple, cast, Union
import logging

import torch
import torch as th
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import math

from torch.distributions import Categorical

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


def kl_div_guidance(logits, label_ref):
    background_probs = torch.ones_like(label_ref[:, 0]) * -100
    background_probs = background_probs[:, None]

    label_ref = torch.cat((background_probs, label_ref), dim=1)

    dist = th.log_softmax(logits, dim=1)
    label_dist = th.log_softmax(label_ref, dim=1)

    return F.kl_div(dist, label_dist, log_target=True)


class DenoisingModel(nn.Module):

    def __init__(self, diffusion: DiffusionModel, unet: nn.Module, guidance_scale: float, guidance_scale_weighting: str,
                 guidance_loss_fn: str, dataset_file: str, label_smoothing: float = 0.0, step_T_sample:str = "majority"):
        super().__init__()
        self.diffusion = diffusion
        self.unet = unet
        self.guidance_scale = guidance_scale
        self.guidance_scale_weighting = guidance_scale_weighting
        self.guidance_loss_fn_name = guidance_loss_fn
        self.dataset_file = dataset_file
        self.step_T_sample = step_T_sample

        self._ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if guidance_loss_fn == "CE":
            self.guidance_loss_fn = self._ce_loss_fn
        elif guidance_loss_fn == "CE-2prob":
            self.guidance_loss_fn = self._cross_entropy_2probs
        elif guidance_loss_fn == "KLDiv":
            self.guidance_loss_fn = kl_div_guidance
        else:
            raise NotImplementedError(f"unsupported: {guidance_loss_fn}")

    @property
    def num_classes(self):
        return self.diffusion.num_classes

    @property
    def time_steps(self):
        return self.diffusion.time_steps

    def forward(self, x: Tensor, condition: Tensor,
                t: Optional[Tensor] = None,
                label_ref_logits: Optional[Tensor] = None,
                validation: bool = False,
                condition_features: Optional[Tensor] = None,
                get_multiscale_predictions: Optional[bool] = False) -> Union[Tensor, dict]:

        if self.training:
            if not isinstance(t, Tensor):
                raise ValueError("'t' needs to be a Tensor at training time")
            if not isinstance(x, Tensor):
                raise ValueError("'x' needs to be a Tensor at training time")
            return self.forward_step(x, condition, t, condition_features=condition_features,
                                     get_multiscale_predictions=get_multiscale_predictions)  # returns dictionary with {"diffusion_out":tensor, ...}
        else:
            if validation:
                return self.forward_step(x, condition, t, condition_features=condition_features)
            if t is None:

                return self.forward_denoising(x, condition, label_ref_logits=label_ref_logits,
                                              condition_features=condition_features)

            return self.forward_denoising(x, condition, cast(int, t.item()), label_ref_logits,
                                          condition_features=condition_features)

    def forward_step(self, x: Tensor, condition: Tensor, t: Tensor,
                     condition_features: Optional[Tensor]=None,
                     get_multiscale_predictions=False) -> Tensor:

        ret = self.unet(x, condition, t, condition_features, get_multiscale_predictions=get_multiscale_predictions)
        return ret

    def forward_denoising(self, x: Optional[Tensor], condition: Tensor, init_t: Optional[int] = None,
                          label_ref_logits: Optional[Tensor] = None,
                          condition_features: Optional[Tensor] = None) -> dict:

        if init_t is None:
            init_t = self.time_steps

        # if x is None:
        #     label_shape = (img.shape[0], self.diffusion.num_classes, *img.shape[2:])
        #     x = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=img.device)).sample()

        xt = x

        if label_ref_logits is not None:
            weights = self.guidance_scale_weights(label_ref_logits)
            label_ref = label_ref_logits.argmax(dim=1)

        shape = xt.shape

        for t in range(init_t, 0, -1):
            # Auxiliary values
            t_ = torch.full(size=(shape[0],), fill_value=t, device=xt.device)

            # Predict the noise of x_t
            ret = self.unet(xt, condition, t_.float(), condition_features)
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

            # if t == int(init_t / 2) + 1 or t == int(init_t / 2) or t == int(init_t / 2) - 15 or t == 1:
            # from ddpm.utils import save_x
            #     save_x(xt, "./logs/output/", timestep=str(t-1))
        ret = {"diffusion_out": xt}
        return ret

    def guidance_fn(self, x, label_ref, guidance_scale_weights):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = x_in  # think identity function

            loss = self.guidance_loss_fn(logits, label_ref)
            gradients_unscaled = th.autograd.grad(outputs=loss, inputs=x_in)[0]

            scales = guidance_scale_weights * self.guidance_scale
            gradients = gradients_unscaled * scales[:, None, :, :]
            assert len(self._check_tensor(gradients)) == 0
            return gradients

    def guidance_scale_weights(self, label_ref_logits):
        label_ref = label_ref_logits.argmax(dim=1)

        if self.guidance_scale_weighting == "binary":
            guidance_scale_weights = th.zeros_like(label_ref)
            label_ref_sm = th.softmax(label_ref_logits, dim=1)

            prob_threshold = 0.5
            for cls in range(label_ref_sm.shape[1]):
                guidance_scale_weights = th.max(guidance_scale_weights, th.where(label_ref_sm[:, cls, :, :] >= prob_threshold, 1, 0))

        elif self.guidance_scale_weighting == "maxprob":
            label_ref_sm = th.softmax(label_ref_logits, dim=1)
            guidance_scale_weights = th.max(label_ref_sm, dim=1)[0]

        elif self.guidance_scale_weighting == "entropy":
            label_ref_sm = th.softmax(label_ref_logits, dim=1)
            label_ref_sm = OneHotCategoricalBCHW.channels_last(label_ref_sm)
            entropy = Categorical(probs=label_ref_sm).entropy()
            guidance_scale_weights = []  # th.zeros_like(label_ref.argmax(dim=1))
            for img_b in range(label_ref_sm.shape[0]):
                img = entropy[img_b]
                img = img.max() + img.min() - img
                guidance_scale_weights.append(img)
            guidance_scale_weights = torch.stack(guidance_scale_weights, dim=0)

        elif self.guidance_scale_weighting == "kldiv":
            label_ref_sm = th.log_softmax(label_ref_logits, dim=1)
            uniform_dist = th.ones_like(label_ref_sm) * 1 / label_ref_sm.shape[1]

            guidance_scale_weights = th.sum(F.kl_div(label_ref_sm, uniform_dist, reduction='none'), dim=1)
        elif self.guidance_scale_weighting == "default":
            guidance_scale_weights = th.ones_like(label_ref)
        else:
            raise NotImplementedError

        return guidance_scale_weights

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

    def _cross_entropy_2probs(self, probs, label_ref_logits):
        is_cityscapes = 'cityscapes' in self.dataset_file
        label_ref = th.softmax(label_ref_logits, dim=1)
        if is_cityscapes:
            background_probs = torch.zeros_like(label_ref_logits[:, 0])
            background_probs = background_probs[:, None]
            label_ref = torch.cat((label_ref, background_probs), dim=1)
            return self._ce_loss_fn(probs, label_ref)
        elif 'voc' in self.dataset_file:
            # add 22nd class that corresponds to 'void' or unlabelled.
            unlabelled_probs = torch.zeros_like(label_ref_logits[:, 0])
            unlabelled_probs = unlabelled_probs[:, None]
            label_ref = torch.cat((label_ref, unlabelled_probs), dim=1)
            return self._ce_loss_fn(probs, label_ref)
        else:
            return self._ce_loss_fn(probs, label_ref)


if __name__ == '__main__':

    from datasets.cityscapes_config import decode_target_to_color
    from PIL import Image
    from datasets.cityscapes import encode_target
    from torchvision import transforms
    from PIL import Image
    import numpy as np


    def to_numpy(tensor):
        """Tensor to numpy, calls .cpu() if necessary"""
        with torch.no_grad():
            if tensor.device.type == 'cuda':
                tensor = tensor.cpu()
            return tensor.numpy()

    def show_tensor_with_pil_label(x, save=False, name=''):
        # x B,C,H,W
        assert len(x.shape) == 4
        b,c,h,w = x.shape

        if c == 1: # integer valued labels
            x_int = x.squeeze(1).long()
        else:  # assume onehot otherwise
            x_int = torch.argmax(x, dim=1, keepdim=True).squeeze(1).long()

        # x_int is B1HW -> x_rgb is B,H,W,3
        x_rgb = decode_target_to_color(x_int)  # BCHW
        x_rgb_numpy = to_numpy(x_rgb).astype(np.uint8)

        # only show 1st image in the batch
        x_pil = Image.fromarray(x_rgb_numpy[0])
        if save:
            # x_pil.show()
            x_pil.save(name)
        return x


    # image_pil = Image.fromarray(np.array(Image.open('2008_000026.png')))
    image_pil = Image.fromarray(np.array(Image.open('aachen_000008_000019_gtFine_labelIds.png')))
    # image_pil = Image.open('aachen_000008_000019_gtFine_labelIds.png')
    # image_pil = np.array(image_pil) + 50
    image_pil = image_pil.resize(size=(256, 128), resample=Image.NEAREST)
    image_pil.show()
    im = torch.tensor(encode_target(transforms.PILToTensor()(image_pil))).long()
    x0 = torch.nn.functional.one_hot(im, 20).permute(0,3,1,2).to("cuda")
    # x0 (B,C,H,W)
    steps = 250

    _steps = list(np.arange(1, steps, 1, dtype=int))
    _, C, H, W = x0.shape

    # for s in [0.008, 0.1, 0.2, 0.3, 0.4]:
    for s in [0.5, 0.6, 0.7, 0.8]:
        noise_params = {"s": s}
        model = DiffusionModel(schedule='cosine', time_steps=steps, num_classes=20, schedule_params=noise_params).to(
            "cuda")

        for time_step in _steps:
            t = torch.tensor([time_step], dtype=torch.long).cuda()
            xt = model.q_xt_given_x0(x0, t).sample()
            if time_step % 25 == 0:
                show_tensor_with_pil_label(xt, save=True, name=f'..\\..\\logs\\noise_schedules\\s={noise_params["s"]}_{time_step}_{H}x{W}.png')
        a = 1

