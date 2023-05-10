import wandb
import importlib
import logging
import os
import pprint
from dataclasses import dataclass
from typing import Union, Optional, List, Tuple, Callable, Dict, Any
from types import FunctionType

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

import ignite.distributed as idist
from ignite.contrib.handlers import ProgressBar, WandBLogger
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Frequency, ConfusionMatrix, mIoU, IoU
from ignite.utils import setup_logger

import torch.backends.cudnn as cudnn
from torch.nn.functional import one_hot

# Local imports
from datasets.pipelines import build_transforms
from .models import DenoisingModel, build_model
from .models.one_hot_categorical import OneHotCategoricalBCHW
from .polyak import PolyakAverager
from .models.condition_encoder import build_cond_encoder
from .optimizer import build_optimizer
from .utils import WithStateDict, archive_code, expanduservars, worker_init_fn, _flatten, _loader_subset, \
    grid_of_predictions, save_image
from .utils import pil_from_bchw_tensor_label  # for debug only

__all__ = ["run_train", "Trainer", "load"]

LOGGER = logging.getLogger(__name__)
Model = Union[DenoisingModel, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


@dataclass
class Trainer:
    polyak: PolyakAverager
    optimizer: torch.optim.Optimizer
    lr_scheduler: Union[torch.optim.lr_scheduler.LambdaLR, None]
    class_weights: torch.Tensor
    save_debug_state: Callable[[Engine, Dict[str, Any]], None]
    cond_polyak: PolyakAverager
    train_encoder: bool
    params: dict
    use_ms_loss: bool

    @property
    def flat_model(self):
        """View of the model without DataParallel wrappers."""
        return _flatten(self.model)

    @property
    def model(self):
        return self.polyak.model

    @property
    def average_model(self):
        return self.polyak.average_model

    @property
    def cond_encoder(self):
        return self.cond_polyak.model

    @property
    def average_cond_encoder(self):
        return self.cond_polyak.average_model

    @property
    def time_steps(self):
        return self.flat_model.time_steps

    @property
    def diffusion_model(self):
        return self.flat_model.diffusion

    def get_loss(self, x0, xt, t, x0pred):
        batch_size = x0.shape[0]
        prob_xtm1_given_xt_x0 = self.diffusion_model.theta_post(xt, x0, t)
        prob_xtm1_given_xt_x0pred = self.diffusion_model.theta_post_prob(xt, x0pred, t)

        loss_diffusion = nn.functional.kl_div(
            torch.log(torch.clamp(prob_xtm1_given_xt_x0pred, min=1e-12)),
            prob_xtm1_given_xt_x0,
            reduction='none'
        )

        # Look for nan and inf in loss and save debug state if found
        self._check_loss(loss_diffusion, locals())
        mask = self.class_weights[x0.argmax(dim=1)]
        loss_diffusion = loss_diffusion.sum(dim=1) * mask
        loss_diffusion = torch.sum(loss_diffusion) / batch_size
        return loss_diffusion

    def train_step_categorical(self, engine: Engine, batch) -> dict:
        image, x0 = batch
        self.model.train()
        if self.train_encoder:
            self.cond_encoder.train()

        self.shape = x0.shape[1:]

        device = idist.device()
        image = image.to(device, non_blocking=True)
        x0 = x0.to(device, non_blocking=True)

        if self.params['conditioning'] in ['concat_pixels_concat_features', 'concat_pixels_attend_features']:
            condition = image
            condition_features = self.cond_encoder(image)
        else:
            condition = self.cond_encoder(image)
            condition_features = None

        batch_size = x0.shape[0]

        # Sample a random step and generate gaussian noise
        t = torch.randint(1, self.time_steps + 1, size=(batch_size,), device=device)
        xt = self.diffusion_model.q_xt_given_x0(x0, t).sample()

        # Estimate the noise with the model
        use_ms_loss = self.params.get("use_ms_loss", False)

        if isinstance(condition_features, torch.Tensor) or condition_features is None:
            # is tensor or None with multiscale features
            ret = self.model(xt.contiguous(), condition.contiguous(), t,
                             condition_features=condition_features.contiguous() if condition_features is not None else None,
                             get_multiscale_predictions=self.use_ms_loss)

        else:  # is list with multiscale features
            ret = self.model(xt.contiguous(), condition.contiguous(), t,
                             condition_features=[f.contiguous() for f in condition_features],
                             get_multiscale_predictions=self.use_ms_loss)

        x0pred = ret["diffusion_out"]  # assumed to be at original resolution of x0
        # logits = ret.get("logits", None)  # todo pass to get_loss if ever used again, for now omitted for simplicity

        loss = self.get_loss(x0, xt, t, x0pred)
        extra_dict = {}
        if self.use_ms_loss:
            extra_dict.update({"loss_res_1": loss.item()})
            for res in self.flat_model.unet.multiscale_prediction_resolutions:
                x0pred_res = ret[f"diffusion_out_{int(res)}"]
                x0_res = torch.nn.functional.interpolate(x0.float(), (x0.shape[2] // res, x0.shape[3] // res),
                                                         mode="nearest").long()
                xt_res = torch.nn.functional.interpolate(xt.float(), (x0.shape[2] // res, x0.shape[3] // res),
                                                         mode="nearest").long()
                loss_res = self.get_loss(x0_res, xt_res, t, x0pred_res)  # todo specify wieghts
                loss += loss_res
                extra_dict.update({f"loss_res_{int(res)}": loss_res.item()})

        # gradient computation
        loss.backward()

        # model update
        if self.params["grad_accumulation"]:
            if engine.state.iteration % self.params["grad_accumulation_step"] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                _ = self.lr_scheduler.step() if self.lr_scheduler is not None else None
                self.polyak.update()
                self.cond_polyak.update()
                # LOGGER.info(f"update {engine.state.iteration}")
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.defaults['lr']
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
            _ = self.lr_scheduler.step() if self.lr_scheduler is not None else None
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.defaults['lr']
            self.polyak.update()
            self.cond_polyak.update()

        # logging
        # pil_from_bchw_tensor_label(x0).show(), pil_from_bchw_tensor_label(xt).show(), pil_from_bchw_tensor_label(x0pred).show()
        ret_dict = {"num_items": batch_size, "loss": loss.item(), "lr": lr}
        ret_dict.update(extra_dict)
        return ret_dict

    def _debug_state(self, locals: dict, debug_names: Optional[List[str]] = None) -> Dict[str, Any]:

        if debug_names is None:
            debug_names = [
                "image", "t", "x0", "xt", "x0pred",
                "prob_xtm1_given_xt_x0", "prob_xtm1_given_xt_x0pred", "loss"
            ]

        to_save = self.objects_to_save(locals["engine"])
        debug_tensors = {k: locals[k] for k in debug_names}
        to_save["tensors"] = WithStateDict(**debug_tensors)
        return to_save

    def _check_loss(self, loss: Tensor, locals: dict) -> None:

        invalid_values = []

        if torch.isnan(loss).any():
            LOGGER.error("nan found in loss!!")
            invalid_values.append("nan")

        if torch.isinf(loss).any():
            LOGGER.error("inf found in loss!!")
            invalid_values.append("inf")

        if (loss.sum(dim=1) < -1e-3).any():
            LOGGER.error("negative KL divergence in loss!!")
            invalid_values.append("neg")

        if invalid_values:
            LOGGER.error("Saving debug state...")
            self.save_debug_state(locals["engine"], self._debug_state(locals))
            raise ValueError(f"Invalid value {invalid_values} found in loss. Debug state has been saved.")

    @torch.no_grad()
    def test_step_categorical(self, _: Engine, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        image, label = batch
        image = image.to(idist.device())
        label = label.to(idist.device())

        label_shape = (label.shape[0], self.flat_model.diffusion.num_classes, *label.shape[2:])
        x = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=label.device)).sample()

        label = label.argmax(dim=1)
        prediction = self.predict(x, image)

        return {'y': label, 'y_pred': prediction}

    @torch.no_grad()
    def predict(self, xt: Tensor, condition: Tensor, label_ref: Optional[Tensor] = None) -> Tensor:
        self.average_model.eval()
        self.average_cond_encoder.eval()
        if self.params['conditioning'] == 'concat_pixels_concat_features':
            condition_features = self.average_cond_encoder(condition)
        else:
            condition_features = None
            condition = self.average_cond_encoder(condition)
        ret = self.average_model(xt, condition, condition_features=condition_features)
        return ret["diffusion_out"]

    def objects_to_save(self,
                        engine: Optional[Engine] = None,
                        weights_only: bool = False,
                        polyak_enabled: bool = True,
                        skip: Optional[list] = None) -> Dict[str, Any]:

        to_save: Dict[str, Any] = {
            "model": self.flat_model.unet,
            "cond_encoder": self.cond_encoder
        }

        if polyak_enabled:
            to_save["average_model"] = _flatten(self.average_model).unet
            to_save["average_cond_encoder"] = self.average_cond_encoder

        if not weights_only:
            to_save["optimizer"] = self.optimizer
            to_save["scheduler"] = self.lr_scheduler
            if engine is not None:
                to_save["engine"] = engine

        # ommiting specified keys from checkpoint
        if skip is not None:
            for skip_key in skip:
                if skip_key in to_save:
                    del to_save[skip_key]
                    LOGGER.info(f'"{skip_key}" will not be loaded from "inin_from" chkpt (will load:{to_save.keys()})')
        return to_save


def attach_train_step(trainer: Trainer, diffusion_type: str):
    if diffusion_type == 'categorical':
        engine = Engine(trainer.train_step_categorical)
    else:
        raise ValueError(f'unknown diffusion type: {diffusion_type}')
    return engine


def attach_test_step(trainer: Trainer, diffusion_type: str):
    if diffusion_type == 'categorical':
        engine = Engine(trainer.test_step_categorical)
    else:
        raise ValueError(f'unknown diffusion type: {diffusion_type}')
    return engine


def build_engine(trainer: Trainer,
                 output_path: str,
                 train_loader: DataLoader,
                 validation_loader: DataLoader,
                 cond_vis_fn: FunctionType,
                 num_classes: int,
                 ignore_class: int,
                 params: dict,
                 train_ids_to_class_names: dict) -> Engine:
    diffusion_type = diffusion_type = params.get("diffusion_type", 'categorical')

    engine = attach_train_step(trainer, diffusion_type)
    frequency_metric = Frequency(output_transform=lambda x: x["num_items"])
    frequency_metric.attach(engine, "imgs/s", Events.ITERATION_COMPLETED)
    GpuInfo().attach(engine, "gpu")
    # control some settings from params

    validation_freq = params["validation_freq"] if "validation_freq" in params else 5000
    save_freq = params["save_freq"] if "save_freq" in params else 1000
    display_freq = params['display_freq'] if "display_freq" in params else 500
    n_validation_predictions = params["n_validation_predictions"] if "n_validation_predictions" in params else 4
    n_validation_images = params["n_validation_images"] if "n_validation_images" in params else 5

    engine_test = attach_test_step(trainer, diffusion_type)
    cm = ConfusionMatrix(num_classes=num_classes)
    LOGGER.info(f"Ignore class {ignore_class} in IoU evaluation...")
    IoU(cm, ignore_index=ignore_class).attach(engine_test, "IoU")
    mIoU(cm, ignore_index=ignore_class).attach(engine_test, "mIoU")

    engine_train = attach_test_step(trainer, diffusion_type)
    cm_train = ConfusionMatrix(num_classes=num_classes)
    IoU(cm_train, ignore_index=ignore_class).attach(engine_train, "IoU")
    mIoU(cm_train, ignore_index=ignore_class).attach(engine_train, "mIoU")

    if idist.get_local_rank() == 0:
        ProgressBar(persist=True).attach(engine_test)

        if params["wandb"]:
            tb_logger = WandBLogger(project=params["wandb_project"], entity='cdm', config=params)

            tb_logger.attach_output_handler(
                engine,
                Events.ITERATION_COMPLETED(every=50),
                tag="training",
                output_transform=lambda x: x,
                metric_names=["imgs/s"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

            tb_logger.attach_output_handler(
                engine_test,
                Events.EPOCH_COMPLETED,
                tag="testing",
                metric_names=["mIoU", "IoU"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

        checkpoint_handler = ModelCheckpoint(
            output_path,
            "model",
            n_saved=3,
            require_empty=False,
            score_function=None,
            score_name=None
        )

        checkpoint_best = ModelCheckpoint(
            output_path,
            "best",
            n_saved=3,
            require_empty=False,
            score_function=lambda engine: engine.state.metrics['mIoU'],
            score_name='mIoU',
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )

    @engine.on(Events.EPOCH_COMPLETED(every=1))
    def epoch_completed_and_set_epoch(engine: Engine):
        # ALL gpu processes must reach this point
        # note: all gpus reach this point but LOGGER.info only displays the msg from process with rank = 0
        if isinstance(engine.state.dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            engine.state.dataloader.sampler.set_epoch(engine.state.epoch)
            LOGGER.info("DDP sampler: set_epoch=%d (iter=%d) completed rank=%d ",
                        engine.state.epoch,
                        engine.state.iteration,
                        idist.get_local_rank())

    # Display some info every 100 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=display_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_info(engine: Engine):
        if "loss_ce" in engine.state.output and "loss_diffusion" in engine.state.output:
            LOGGER.info(
                "epoch=%d, iter=%d, speed=%.2fimg/s, loss=%.4g, loss_ce=%.4g, loss_diff=%.4g, lr=%.6g, gpu:0 util=%.2f%%",
                engine.state.epoch,
                engine.state.iteration,
                engine.state.metrics["imgs/s"],
                engine.state.output["loss"],
                engine.state.output["loss_ce"],
                engine.state.output["loss_diffusion"],
                engine.state.output["lr"],
                engine.state.metrics["gpu:0 util(%)"]
            )
        elif "loss_res_1" in engine.state.output:
            extra_msg_loss = ''
            for res in [1, 2, 4, 8, 16, 32]:
                if f"loss_res_{res}" in engine.state.output:
                    extra_msg_loss += f'{f"loss_res_{res}"} : {engine.state.output[f"loss_res_{res}"] :.3f} '

            LOGGER.info(
                "epoch=%d, iter=%d, speed=%.2fimg/s, loss=%.4g, lr=%.6g,  gpu:0 util=%.2f%%",
                engine.state.epoch,
                engine.state.iteration,
                engine.state.metrics["imgs/s"],
                engine.state.output["loss"],
                engine.state.output["lr"],
                engine.state.metrics["gpu:0 util(%)"]
            )
            LOGGER.info(extra_msg_loss)

        else:
            LOGGER.info(
                "epoch=%d, iter=%d, speed=%.2fimg/s, loss=%.4g, lr=%.6g,  gpu:0 util=%.2f%%",
                engine.state.epoch,
                engine.state.iteration,
                engine.state.metrics["imgs/s"],
                engine.state.output["loss"],
                engine.state.output["lr"],
                engine.state.metrics["gpu:0 util(%)"]
            )

    # Save model every 1000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=save_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_model(engine: Engine):
        checkpoint_handler(engine,
                           trainer.objects_to_save(engine, weights_only=False, polyak_enabled=params['polyak_enabled']))

    # Generate and save a few segmentations every 1000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=validation_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_qualitative_results(_: Engine, num_images=n_validation_images, num_predictions=n_validation_predictions):
        LOGGER.info("Generating images...")
        loader = _loader_subset(validation_loader, num_images, randomize=False)
        grid = grid_of_predictions(_flatten(trainer.average_model), trainer.average_cond_encoder, loader,
                                   num_predictions, cond_vis_fn, params)
        loader = _loader_subset(validation_loader, num_images, randomize=True)
        grid_shuffle = grid_of_predictions(_flatten(trainer.average_model), trainer.average_cond_encoder, loader,
                                           num_predictions, cond_vis_fn, params)
        grid = torch.concat([grid, grid_shuffle], dim=0)
        filename = os.path.join(output_path, f"images_{engine.state.iteration:06}.png")
        LOGGER.info("Saving images to %s...", filename)
        os.makedirs(output_path, exist_ok=True)
        img = save_image(grid, filename, nrow=grid.shape[0] // (num_images * 2))
        if params["wandb"]:
            images = wandb.Image(img, caption=f"Iteration {engine.state.iteration}")
            wandb.log({"examples": images})

    # Compute the mIoU score every 5000 iterations (calls trainer.test_step() for images in validation_loader)
    @engine.on(Events.ITERATION_COMPLETED(every=validation_freq))
    def test(_: Engine):
        LOGGER.info("mIoU computation...")
        engine_test.run(validation_loader, max_epochs=1)

    # Save the best models by mIoU score (runs once every len(validation_loader))
    @engine_test.on(Events.EPOCH_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_iou(engine_test: Engine):
        LOGGER.info("val mIoU score: %.4g", engine_test.state.metrics["mIoU"])
        if isinstance(train_ids_to_class_names, dict):
            per_class_ious = [(train_ids_to_class_names[i], iou) for i, iou in
                              enumerate(engine_test.state.metrics["IoU"])]
        else:
            per_class_ious = [(i, iou) for i, iou in enumerate(engine_test.state.metrics["IoU"])]
        for iou in per_class_ious:
            LOGGER.info("val IoU scores per class: %.2g  %.4g", iou[0], iou[1].item())
        if params["wandb"]:
            wandb.log({"mIoU_val": engine_test.state.metrics["mIoU"]})
        checkpoint_best(engine_test,
                        trainer.objects_to_save(engine, weights_only=False, polyak_enabled=params['polyak_enabled']))

    @engine.on(Events.ITERATION_COMPLETED(every=validation_freq))
    def train_mIoU(_: Engine):
        LOGGER.info("train mIoU computation...")
        train_loader_ss = _loader_subset(train_loader, 50, randomize=False)
        engine_train.run(train_loader_ss, max_epochs=1)

    @engine_train.on(Events.EPOCH_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_train_mIoU(engine_train: Engine):
        LOGGER.info("train mIoU score: %.4g", engine_train.state.metrics["mIoU"])
        if params["wandb"]:
            wandb.log({"mIoU_train": engine_train.state.metrics["mIoU"]})

    return engine


def load(filename: str, trainer: Trainer, engine: Engine,
         weights_only: bool = False,
         polyak_enabled: bool = True,
         skip_keys: Optional[list] = None):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    if skip_keys is not None:
        assert type(skip_keys) == list
    to_load = trainer.objects_to_save(engine, weights_only, polyak_enabled, skip=skip_keys)
    ModelCheckpoint.load_objects(to_load, state)


def _build_model(params: dict, input_shapes: List[Tuple[int, int, int]], cond_encoded_shape) -> Model:
    model: Model = build_model(
        time_steps=params["time_steps"],
        schedule=params["beta_schedule"],
        schedule_params=params.get("beta_schedule_params", None),
        guidance_scale=params.get("guidance_scale", None),
        guidance_scale_weighting=params.get("guidance_scale_weighting", None),
        guidance_loss_fn=params.get("guidance_loss_fn", 'CE'),
        label_smoothing=params.get("label_smoothing", None),
        conditioning=params['conditioning'],
        cond_encoded_shape=cond_encoded_shape,
        input_shapes=input_shapes,
        backbone=params["backbone"],
        backbone_params=params[params["backbone"]],
        dataset_file=params['dataset_file'],
        step_T_sample=params.get('evaluation_vote_strategy', None),
        diffusion_type=params.get('diffusion_type', 'categorical'),
        bits=params.get('bits', None),
        analog_bits_scale=params.get('analog_bits_scale', 1.0),
        params=params
    ).to(idist.device())
    # Wrap the model in DataParallel or DistributedDataParallel for parallel processing
    if params["distributed"]:
        local_rank = idist.get_local_rank()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif params["multigpu"]:
        model = nn.DataParallel(model)
    return model


def _build_datasets(params: dict) -> Tuple[DataLoader, DataLoader, torch.Tensor, int, dict]:
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_ids_to_class_names = None

    if dataset_file == 'datasets.cityscapes_c':
        train_dataset = dataset_module.training_dataset(params["corruption"],
                                                        params["corruption_strength"])  # type: ignore
        validation_dataset = dataset_module.validation_dataset(params['dataset_val_max_size'], params["corruption"],
                                                               params["corruption_strength"])  # type: ignore
        train_ids_to_class_names = dataset_module.train_ids_to_class_names()
    # required in params.yml for using datasets/pipelines/transforms
    elif (dataset_file == 'datasets.cityscapes') and all(['dataset_pipeline_train' in params,
                                                          'dataset_pipeline_train_settings' in params,
                                                          'dataset_pipeline_val' in params,
                                                          'dataset_pipeline_val_settings' in params]):

        transforms_names_train = params["dataset_pipeline_train"]
        transforms_settings_train = params["dataset_pipeline_train_settings"]
        transforms_dict_train = build_transforms(transforms_names_train, transforms_settings_train, num_classes=20)

        transforms_names_val = params["dataset_pipeline_val"]
        transforms_settings_val = params["dataset_pipeline_val_settings"]
        transforms_dict_val = build_transforms(transforms_names_val, transforms_settings_val, num_classes=20)

        args = {'transforms_dict_train': transforms_dict_train}
        train_dataset = dataset_module.training_dataset(**args)  # type: ignore

        validation_dataset = dataset_module.validation_dataset(max_size=params['dataset_val_max_size'],
                                                               transforms_dict_val=transforms_dict_val)  # type: ignore
    else:
        train_dataset = dataset_module.training_dataset()  # type: ignore
        validation_dataset = dataset_module.validation_dataset(max_size=params['dataset_val_max_size'])  # type: ignore

    if "mmseg" in dataset_file:
        # again since mmseg changed it...
        setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    LOGGER.info("%d images in dataset '%s'", len(train_dataset), dataset_file)
    LOGGER.info("%d images in validation dataset '%s'", len(validation_dataset), dataset_file)

    # If there is no 'get_weights' function in the dataset module, create a tensor full of ones.
    get_weights = getattr(dataset_module, 'get_weights', lambda _: torch.ones(train_dataset[0][1].shape[0]))
    class_weights = get_weights(params["class_weights"])

    #  worker_init_function not used: each worker has the same random seed every epoch
    #  workers (threads) are re-initialized every epoch and their seeding is the same every time
    #  see https://discuss.pytorch.org/t/does-getitem-of-dataloader-reset-random-seed/8097
    #  https://github.com/pytorch/pytorch/issues/5059

    if params['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        rank=idist.get_local_rank(),
                                                                        num_replicas=params['num_gpus'])
        batch_size = params['batch_size'] // params['num_gpus']  # batch_size of each process

    else:
        train_sampler = None
        batch_size = params['batch_size']  # if single_gpu or non-DDP

    dataset_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                drop_last=True,
                                pin_memory=True,
                                sampler=train_sampler,
                                shuffle=train_sampler is None,
                                num_workers=params["mp_loaders"],
                                worker_init_fn=worker_init_fn)

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   num_workers=params["mp_loaders"],
                                   shuffle=False,
                                   worker_init_fn=worker_init_fn)

    return dataset_loader, validation_loader, class_weights, dataset_module.get_ignore_class(), train_ids_to_class_names


def _build_debug_checkpoint(output_path: str) -> ModelCheckpoint:
    return ModelCheckpoint(output_path, "debug_state", require_empty=False, score_function=None, score_name=None)


def run_train(local_rank: int, params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    LOGGER.info("experiment dir: %s", output_path)

    LOGGER.info("Training params:\n%s", pprint.pformat(params))

    # num_gpus = torch.cuda.device_count()
    LOGGER.info("%d GPUs available", torch.cuda.device_count())

    cudnn.benchmark = params['cudnn']['benchmark']  # this set to true usually slightly accelerates training
    LOGGER.info(f'*** setting cudnn.benchmark to {cudnn.benchmark} ***')
    LOGGER.info(f"*** cudnn.enabled {cudnn.enabled}")
    LOGGER.info(f"*** cudnn.deterministic {cudnn.deterministic}")

    # Load the datasets
    train_loader, validation_loader, class_weights, ignore_class, train_ids_to_class_names = _build_datasets(params)

    # Build the model, optimizer, trainer and training engine
    input_shapes = [i.shape for i in train_loader.dataset[0] if hasattr(i, 'shape')]
    LOGGER.info("Input shapes: " + str(input_shapes))

    cond_encoder, cond_vis_fn = build_cond_encoder(params)
    if params['polyak_enabled']:
        average_cond_encoder, _ = build_cond_encoder(params)
        cond_polyak = PolyakAverager(cond_encoder, average_cond_encoder, alpha=params["polyak_alpha"])
    else:
        cond_polyak = PolyakAverager(cond_encoder, None)

    # cond_encoded_shape = cond_encoder(train_loader.dataset[0][0][None].to(idist.device())).shape
    # LOGGER.info("Encoded condition shape: " + str(cond_encoded_shape))
    cond_encoder_shape = None

    if hasattr(train_loader.dataset, 'num_classes'):
        num_classes = train_loader.dataset.num_classes
    else:
        num_classes = input_shapes[1][0]

        assert len(
            class_weights) == num_classes, f"len(class_weights) != num_classes: {len(class_weights)} != {num_classes}"

    if params['polyak_enabled']:
        model, average_model = [_build_model(params, input_shapes, cond_encoder_shape) for _ in range(2)]
        # initialize both the "model" and "average_model" with the same weights
        average_model.load_state_dict(model.state_dict())
        polyak = PolyakAverager(model, average_model, alpha=params["polyak_alpha"])
    else:
        print("WARN: Polyak averaging disabled.")
        model = _build_model(params, input_shapes, cond_encoder_shape)
        polyak = PolyakAverager(model, None)

    train_encoder = params.get("train_encoder", False)
    if train_encoder:
        optimizer_staff = build_optimizer(params, model, train_loader, cond_encoder)
    else:
        optimizer_staff = build_optimizer(params, model, train_loader, debug=False)
    optimizer = optimizer_staff['optimizer']
    lr_scheduler = optimizer_staff['lr_scheduler']
    # optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    debug_checkpoint = _build_debug_checkpoint(output_path)

    # scaler = torch.cuda.amp.GradScaler()

    trainer = Trainer(polyak, optimizer, lr_scheduler, class_weights.to(idist.device()), debug_checkpoint, cond_polyak,
                      params["train_encoder"],
                      params, use_ms_loss=params.get("use_ms_loss", False))
    engine = build_engine(trainer, output_path, train_loader, validation_loader, cond_vis_fn,
                          num_classes=num_classes, ignore_class=ignore_class,
                          params=params, train_ids_to_class_names=train_ids_to_class_names)

    init_from = params.get('init_from', None)
    if init_from is not None:
        init_from = expanduservars(init_from)
        skip_keys = params.get("init_skip_keys", None)
        load(init_from, trainer=trainer, engine=engine, weights_only=True, polyak_enabled=params['polyak_enabled'],
             skip_keys=skip_keys)

    # Load a model (if requested in params.yml) to continue training from it
    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        load(load_from, trainer=trainer, engine=engine, polyak_enabled=params['polyak_enabled'],
             skip_keys=params.get("load_skip_keys", None))
        optimizer.param_groups[0]['capturable'] = True

    if params.get("grad_accumulation", False):
        mult = params.get('grad_accumulation_step', 1)
    else:
        mult = 1

    if params["diffusion_type"] == "continuous_analog_bits":
        if hasattr(train_loader.dataset, "apply_one_hot"):
            train_loader.dataset.apply_one_hot = False
            LOGGER.info(f"train loader {train_loader.dataset} will NOT apply one hot to labels")

    if params["diffusion_type"] == "continuous_analog_bits":
        if hasattr(validation_loader.dataset, "apply_one_hot"):
            validation_loader.dataset.apply_one_hot = True
            LOGGER.info(f"valid loader {validation_loader.dataset} will apply one hot to labels")

    else:
        LOGGER.info(f"dataset {train_loader.dataset} will apply one hot to labels")

    archive_code(output_path, params["params_file"])  # save params after it has been processed/modified is fixed
    # Run the training engine for the requested number of epochs
    engine.run(train_loader, max_epochs=mult * params["max_epochs"])
