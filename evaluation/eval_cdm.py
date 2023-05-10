import importlib
import logging
import wandb
import pathlib
from dataclasses import dataclass
import os
from typing import Union, Tuple
from PIL import Image
import pprint
import ignite.distributed as idist
import numpy as np
import torch

from datasets.pipelines import build_transforms

from ignite.contrib.handlers import ProgressBar, WandBLogger
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine
from ignite.metrics import ConfusionMatrix, mIoU, IoU
from ignite.utils import setup_logger
from ignite.contrib.metrics import GpuInfo

import torch.backends.cudnn as cudnn
from torch import nn, Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from ddpm.models import DenoisingModel
from ddpm.models.one_hot_categorical import OneHotCategoricalBCHW
from ddpm.trainer import _build_model, build_cond_encoder
from ddpm.utils import expanduservars, archive_code, worker_init_fn, _flatten
# from ddpm.utils import pil_from_bchw_tensor_label, pil_from_bchw_tensor_image, _onehot_to_color_image  # debug only
from .utils import create_new_directory

from .cs_eval import evaluateImgLists, args

LOGGER = logging.getLogger(__name__)
Model = Union[DenoisingModel, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _build_datasets(params: dict) -> Tuple[DataLoader, torch.Tensor, int, int, dict]:
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    # required in params_eval.yml for using datasets/pipelines/transforms
    train_ids_to_class_names = None

    if ((dataset_file == 'datasets.cityscapes') or (dataset_file == 'datasets.ade20k'))\
            and all(['dataset_pipeline_val' in params, 'dataset_pipeline_val_settings' in params]):
        transforms_names_val = params["dataset_pipeline_val"]
        transforms_settings_val = params["dataset_pipeline_val_settings"]
        transforms_dict_val = build_transforms(transforms_names_val, transforms_settings_val, num_classes=dataset_module.get_num_classes())
        validation_dataset = dataset_module.validation_dataset(max_size=params['dataset_val_max_size'],
                                                               transforms_dict_val=transforms_dict_val)  # type: ignore

        train_ids_to_class_names = dataset_module.train_ids_to_class_names()

    else:
        train_dataset = dataset_module.training_dataset()  # type: ignore
        validation_dataset = dataset_module.validation_dataset(max_size=params['dataset_val_max_size'])  # type: ignore

    LOGGER.info("%d images in validation dataset '%s'", len(validation_dataset), dataset_file)

    # If there is no 'get_weights' function in the dataset module, create a tensor full of ones.
    get_weights = getattr(dataset_module, 'get_weights', lambda _: torch.ones(train_dataset[0][1].shape[0]))
    class_weights = get_weights(params["class_weights"])

    batch_size = params['batch_size']  # if single_gpu or non-DDP
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   num_workers=params["mp_loaders"],
                                   shuffle=False,
                                   worker_init_fn=worker_init_fn)

    return validation_loader, class_weights, dataset_module.get_ignore_class(), dataset_module.get_num_classes(), train_ids_to_class_names


@dataclass
class Evaluator:
    eligible_eval_resolutions = ["original", "dataloader"]
    eligible_eval_vote_strategies = ["majority", "confidence"]
    # original => compute miou wrt to original labels
    # dataloader => compute miou wrt to dataloader (potentially resized, cropped etc) labels

    def __init__(self,
                 model: Model,
                 average_model: Model,
                 cond_encoder: Model,
                 params: dict,
                 num_classes: int,
                 ignore: int):

        self.params = params
        self.cond_encoder = cond_encoder
        self.model = _flatten(model)
        self.average_model = _flatten(average_model)
        self.checkpoint_dir = None
        self.dataset_module = importlib.import_module(params['dataset_file'])
        self.dataset_module_config = importlib.import_module(params['dataset_file']+'_config')
        self.pred_list = []
        self.label_list = []
        self.images_cnt = 0
        self.diffusion_type = self.params.get("diffusion_type", "categorical")
        self.num_classes = num_classes  # USED CLASSES + 1 (IGNORE)
        self.ignore = ignore  # ASSUMED TO BE num_classes-1
        assert self.ignore == (self.num_classes - 1), f"Invalid ignore or num_classes" \
                                                      f" assumed ignore = num_classes-1" \
                                                      f" but got ignore = {ignore} and num_classes = {num_classes}"

        self.eval_settings = self.params.get("evaluation", {})
        self.eval_resolution = self.eval_settings.get("resolution", "dataloader")
        self.eval_voting_strategy = self.eval_settings.get("evaluation_vote_strategy", 'confidence')
        self.num_evaluations = self.eval_settings.get("evaluations", 1)

        assert(self.eval_resolution in self.eligible_eval_resolutions), f"eval_resolution={self.eval_resolution} " \
                                                                        f"in not in {self.eligible_eval_resolutions}"
        assert(self.eval_voting_strategy in self.eligible_eval_vote_strategies), f"eval_voting_strategy={self.eval_voting_strategy} " \
                                                                        f"in not in {self.eligible_eval_vote_strategies}"

        self.average_model.step_T_sample = self.eval_voting_strategy
        LOGGER.info(f"Evaluation with diffusion_type: {self.diffusion_type}")
        LOGGER.info(f"Evaluation settings {self.eval_settings}")

    def load(self, filename: str):
        LOGGER.info("Loading state from %s...", filename)
        checkpoint = torch.load(filename, map_location=idist.device())
        self.load_objects(checkpoint)
        v = pathlib.Path(filename)
        self.checkpoint_dir = str(v.parent)

    def load_objects(self, checkpoint: dict, strict=True):
        LOGGER.info("Loading checkpoint sanity check")
        LOGGER.info(f"model parameters :{sum([p.shape.numel() for p in self.model.unet.parameters()])}")
        LOGGER.info(f"checkpoint parameters: {sum([checkpoint['model'][v].shape.numel() for v in checkpoint['model']])}")
        self.model.unet.load_state_dict(checkpoint["model"], strict)
        try:
            self.cond_encoder.load_state_dict(checkpoint["cond_encoder"], strict)
        except:
            LOGGER.info(f"no cond_encoder found in checkpoint with entries {checkpoint.keys()}")
        self.average_model.unet.load_state_dict(checkpoint["average_model"], strict)
        # ret_average_cond_encoder = self.average_cond_encoder.load_state_dict(checkpoint["average_cond_encoder"])

    @property
    def time_steps(self):
        return self.model.time_steps

    @property
    def diffusion_model(self):
        return self.model.diffusion

    @torch.no_grad()
    def predict_condition(self, x: Tensor) -> dict:
        # x BCHW -> (B,D,...)
        ret = {"condition_features": None, "condition": None}
        self.cond_encoder.eval()
        if self.params["conditioning"] == 'concat_pixels_concat_features':
            assert(self.params["cond_encoder"] in ["dino_vits8"])
            ret.update({"condition_features": self.cond_encoder(x)})
            ret.update({"condition": x})
        else:
            # PSA calling this can be a dummy_function that just return x.
            ret.update({"condition": self.cond_encoder(x)})
        return ret

    @torch.no_grad()
    def predict_single(self, image, condition, label_ref_logits=None, condition_features=None):
        # predict a single segmentation (BNHW) for image (BCHW) where N = num_classes
        label_shape = (image.shape[0], self.num_classes, *image.shape[2:])
        xt = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=image.device)).sample()

        self.average_model.eval()
        self.cond_encoder.eval()
        ret = self.average_model(x=xt,
                                 condition=condition,
                                 label_ref_logits=label_ref_logits,
                                 condition_features=condition_features)

        # ret is dict {"diffusion_out" : unet's softmaxed output, "logits" : None or output of parallel unet head}
        assert("diffusion_out" in ret), "forward method of self.average_model must always return a dict with 'diffusion_out' as key"
        return ret["diffusion_out"]

    @torch.no_grad()
    def predict_multiple(self, image, condition, condition_features=None):
        assert(self.num_evaluations > 1), f'predict_multiple assumes evaluations > 1 instead got {self.num_evaluations}'
        # predict a params['evaluations'] * segmentations each of shape (BNHW) for image (BCHW) where N = num_classes
        for i in range(self.num_evaluations):
            prediction_onehot_i = self.predict_single(image, condition, condition_features=condition_features)
            if self.eval_voting_strategy == 'confidence':
                if i == 0:
                    prediction_onehot_total = torch.zeros_like(prediction_onehot_i)
                prediction_onehot_total += prediction_onehot_i * (1/self.num_evaluations)

            elif self.eval_voting_strategy == 'majority':
                if i == 0:
                    votes = torch.zeros_like(prediction_onehot_i)
                votes += prediction_onehot_i
            else:
                raise ValueError(f"{self.eval_voting_strategy} is not a valid voting strategy")

        if self.eval_voting_strategy == 'majority':
            prediction_onehot_total = votes.argmax(dim=1)
            prediction_onehot_total = one_hot(prediction_onehot_total.to(torch.int64),
                                              self.num_classes).squeeze(1).permute(0, 3, 1, 2)
        return prediction_onehot_total

    @torch.no_grad()
    def infer_step_categorical(self, engine: Engine, batch: Tensor): # -> Tuple[Tensor]:
        # cdm only inference step

        # prep data
        image, label, label_orig = batch
        image = image.to(idist.device())
        label_onehot = label.to(idist.device())
        label = label_onehot.argmax(dim=1).long()  # one_hot to int, (BHW)

        # forward step
        condition_dict = self.predict_condition(image)
        # PSA condition_dict has keys "feature_condition" (optional, default to None) and "condition" (always a tensor)
        if self.num_evaluations == 1:
            prediction_onehot = self.predict_single(image, **condition_dict)  # (BNHW)
        else:
            prediction_onehot = self.predict_multiple(image, **condition_dict)

        # debug only shows 1st element of the batch
        # pil_from_bchw_tensor_label(prediction_onehot).show()
        # pil_from_bchw_tensor_label(label_onehot).show()
        # pil_from_bchw_tensor_image(Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)).show()

        # prep prediction and label for IOU/MIOU computation
        # prediction = prediction_onehot.argmax(dim=1).long()
        if self.eval_resolution == 'original':
            # replace label with original labels of shape (B H_orig W_orig)
            # upsample prediction_onehot to (H_orig,W_orig) with bilinear interpolation
            label = label_orig.to(idist.device())
            b, h_orig, w_orig = label.shape
            prediction_onehot = torch.nn.functional.interpolate(prediction_onehot, (h_orig, w_orig), mode='bilinear')

        prediction_onehot = prediction_onehot[:, 0:self.num_classes-1, ...]  # removing ignore class channel
        self.save_preds(label.long(), prediction_onehot.argmax(dim=1).long())
        self.images_cnt += image.shape[0]

        return {"y": label, "y_pred": prediction_onehot}

    def save_preds(self, label: Tensor, pred:Tensor):
        # pred is in train_id format
        # this function saves predictions for cityscapes script to use them aftewards
        assert label.dtype == torch.long
        assert pred.dtype == torch.long
        assert self.checkpoint_dir is not None, f'saving preds in checkpoint_dir but it is {self.checkpoint_dir}'

        if self.num_evaluations > 1:
            mode = f'@{self.eval_voting_strategy}'
        else:
            mode = ''
        path_submit = str(pathlib.Path(self.checkpoint_dir) / f'outputs@{self.num_evaluations}{mode}' / 'val' / 'submit')
        path_debug = str(pathlib.Path(self.checkpoint_dir) / f'outputs@{self.num_evaluations}{mode}' / 'val' / 'debug')
        path_label = str(pathlib.Path(self.checkpoint_dir) / f'outputs@{self.num_evaluations}{mode}' / 'val' / 'label')
        path_label_debug = str(pathlib.Path(self.checkpoint_dir) / f'outputs@{self.num_evaluations}{mode}' / 'val' / 'label_debug')

        create_new_directory(path_submit)
        create_new_directory(path_debug)
        create_new_directory(path_label)
        create_new_directory(path_label_debug)

        pred_submit = self.dataset_module_config.map_train_id_to_id(pred.cpu().clone())
        pred_debug = self.dataset_module_config.decode_target_to_color(pred.cpu().clone())
        label_submit = self.dataset_module_config.map_train_id_to_id(label.cpu().clone())
        label_debug = self.dataset_module_config.decode_target_to_color(label.cpu().clone())
        #
        # Image.fromarray(pred_submit[0].cpu().numpy().astype(np.uint8)).show()
        # Image.fromarray(pred_debug[0].cpu().numpy().astype(np.uint8)).show()
        # Image.fromarray(label_submit[0].cpu().numpy().astype(np.uint8)).show()

        for i in range(pred_submit.shape[0]):
            path_filename_submit = \
                str(pathlib.Path(path_submit) / f'{self.images_cnt + i+1}_id.png')
            path_filename_debug = \
                str(pathlib.Path(path_debug) / f'{self.images_cnt + i+1}_rgb.png')
            path_filename_label_submit = \
                str(pathlib.Path(path_label) / f'{self.images_cnt + i+1}_label.png')
            path_filename_label_debug = \
                str(pathlib.Path(path_label_debug) / f'{self.images_cnt + i+1}_label.png')

            Image.fromarray(pred_submit[i].cpu().numpy().astype(np.uint8)).save(path_filename_submit)
            Image.fromarray(pred_debug[i].cpu().numpy().astype(np.uint8)).save(path_filename_debug)
            Image.fromarray(label_submit[i].cpu().numpy().astype(np.uint8)).save(path_filename_label_submit)

            Image.fromarray(label_debug[i].cpu().numpy().astype(np.uint8)).save(path_filename_label_debug)

            LOGGER.info(f"saved pred {i} from batch shape {pred_submit.shape} with "
                        f"id format [{path_filename_submit} "
                        f"and color at {path_filename_debug}")

            LOGGER.info(f"saved label {i} from batch shape {label_submit.shape} with "
                        f"id format [{path_filename_label_submit} ")

            self.pred_list.append(path_filename_submit)
            self.label_list.append(path_filename_label_submit)


def attach_test_step(evaluator: Evaluator, diffusion_type: str):
    if diffusion_type == 'categorical':
        engine = Engine(evaluator.infer_step_categorical)
    elif diffusion_type == 'continuous_analog_bits':
        engine = Engine(evaluator.infer_step_abc)
    else:
        raise ValueError(f'unknown diffusion type: {diffusion_type}')
    return engine


def build_engine(evaluator: Evaluator,
                 num_classes: int,
                 ignore_class: int,
                 params: dict,
                 train_ids_to_class_names: Union[None, dict] = None) -> Engine:

    diffusion_type = params["diffusion_type"]
    engine_test = attach_test_step(evaluator, diffusion_type=diffusion_type)
    GpuInfo().attach(engine_test, "gpu")
    LOGGER.info(f"Ignore class {ignore_class} in IoU evaluation...")
    cm = ConfusionMatrix(num_classes=num_classes-1)  # 0-18
    cm.attach(engine_test, 'cm')
    IoU(cm).attach(engine_test, "IoU")
    mIoU(cm).attach(engine_test, "mIoU")
    from cityscapesscripts.evaluation import evalPixelLevelSemanticLabeling
    if idist.get_local_rank() == 0:
        ProgressBar(persist=True).attach(engine_test)
        if params["wandb"]:
            tb_logger = WandBLogger(project=params["wandb_project"], entity='cdm', config=params)
            tb_logger.attach_output_handler(
                engine_test,
                Events.EPOCH_COMPLETED,
                tag="testing",
                metric_names=["mIoU", "IoU"],
                global_step_transform=global_step_from_engine(engine_test, Events.ITERATION_COMPLETED)
            )

    @engine_test.on(Events.ITERATION_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_eval(et: Engine):
        LOGGER.info(f"{et.state.metrics}")

    @engine_test.on(Events.EPOCH_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_iou(et: Engine):
        LOGGER.info("mIoU score: %.4g", et.state.metrics["mIoU"])
        if isinstance(train_ids_to_class_names, dict):
            per_class_ious = [(train_ids_to_class_names[i], iou.item()) for i, iou in
                              enumerate(engine_test.state.metrics["IoU"])]

        else:
            per_class_ious = [(i, iou.item()) for i, iou in enumerate(engine_test.state.metrics["IoU"])]

        LOGGER.info(f"val IoU scores per class:{per_class_ious}")
        if params["wandb"]:
            wandb.log({"mIoU_val": et.state.metrics["mIoU"]})
    return engine_test


def run_inference(params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)
    LOGGER.info("%d GPUs available", torch.cuda.device_count())
    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    LOGGER.info("experiment dir: %s", output_path)
    archive_code(output_path, params["params_file"])

    LOGGER.info("Inference params:\n%s", pprint.pformat(params))
    cudnn.benchmark = params['cudnn']['benchmark']  # this set to true usually slightly accelerates training
    LOGGER.info(f'*** setting cudnn.benchmark to {cudnn.benchmark} ***')
    LOGGER.info(f"*** cudnn.enabled {cudnn.enabled}")
    LOGGER.info(f"*** cudnn.deterministic {cudnn.deterministic}")

    # Load the datasets
    data_loader, _, ignore_class, num_classes, train_ids_to_class_names = _build_datasets(params)
    assert(hasattr(data_loader, "dataset"))
    if hasattr(data_loader.dataset, "return_metadata"):
        data_loader.dataset.return_metadata = True
    elif hasattr(data_loader.dataset.dataset, "return_metadata"):  # in case Subset of dataset is used
        data_loader.dataset.dataset.return_metadata = True
    else:
        raise ValueError()
    # build evaluator
    cdm_only = params["cdm_only"]

    eval_h_model = params['dataset_pipeline_val_settings']['target_size'][0]
    eval_w_model = params['dataset_pipeline_val_settings']['target_size'][1]

    LOGGER.info(f"Expecting image resolution of {(eval_h_model, eval_w_model)} to build model.")
    input_shapes = [(3, eval_h_model, eval_w_model), (num_classes, eval_h_model, eval_w_model)]

    cond_encoder, _ = build_cond_encoder(params)
    cond_encoded_shape = None if params["conditioning"] != 'x-attention' \
        else cond_encoder(data_loader.dataset[0][0][None].to(idist.device())).shape

    model, average_model = [_build_model(params, input_shapes, cond_encoded_shape) for _ in range(2)]
    evaluator = Evaluator(model, average_model, cond_encoder, params, num_classes, ignore_class)
    engine_test = build_engine(evaluator, num_classes, ignore_class, params, train_ids_to_class_names)

    # load checkpoint
    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        evaluator.load(load_from)

    engine_test.run(data_loader, max_epochs=1)

    # evaluate using cityscapes official script
    args.evalInstLevelScore = False
    args.evalPixelAccuracy = True
    args.JSONOutput = False
    results = evaluateImgLists(sorted(evaluator.pred_list), sorted(evaluator.label_list), args, lambda x: torch.as_tensor(x))
    print(results)
    import json
    results_json = json.dumps(results, indent=2, sort_keys=True)
    with open(str(pathlib.Path(evaluator.checkpoint_dir) / 'cts_script_results.json'), 'w') as json_file:
        json_file.write(results_json)
