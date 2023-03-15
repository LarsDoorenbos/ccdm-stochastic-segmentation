
import importlib
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import ignite.distributed as idist
import numpy as np
import torch
# Ignite imports
from ignite.engine import Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import ConfusionMatrix, mIoU, IoU, DiceCoefficient
from ignite.utils import setup_logger
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.utils.data import DataLoader

from ddpm.models.one_hot_categorical import OneHotCategoricalBCHW
from ddpm.polyak import PolyakAverager
from ddpm.trainer import _build_model, _flatten
from ddpm.utils import expanduservars

LOGGER = logging.getLogger(__name__)


def iou(x, y, axis=-1):
    iou_ = (x & y).sum(axis) / (x | y).sum(axis)
    iou_[np.isnan(iou_)] = 1.
    return iou_


# exclude background
def batched_distance(x, y):
    try:
        per_class_iou = iou(x[:, :, None], y[:, None, :], axis=-2)
    except MemoryError:
        raise NotImplementedError

    return 1 - per_class_iou[..., 1:].mean(-1)


def calc_batched_generalised_energy_distance(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.reshape(*samples_dist_0.shape[:2], -1)
    samples_dist_1 = samples_dist_1.reshape(*samples_dist_1.shape[:2], -1)

    eye = np.eye(num_classes)

    samples_dist_0 = eye[samples_dist_0].astype(np.bool)
    samples_dist_1 = eye[samples_dist_1].astype(np.bool)
    
    cross = np.mean(batched_distance(samples_dist_0, samples_dist_1), axis=(1,2))
    diversity_0 = np.mean(batched_distance(samples_dist_0, samples_dist_0), axis=(1,2))
    diversity_1 = np.mean(batched_distance(samples_dist_1, samples_dist_1), axis=(1,2))
    return 2 * cross - diversity_0 - diversity_1, diversity_0, diversity_1


def batched_hungarian_matching(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.reshape((*samples_dist_0.shape[:2], -1))
    samples_dist_1 = samples_dist_1.reshape((*samples_dist_1.shape[:2], -1))

    eye = np.eye(num_classes)

    samples_dist_0 = eye[samples_dist_0].astype(np.bool)
    samples_dist_1 = eye[samples_dist_1].astype(np.bool)
    
    cost_matrix = batched_distance(samples_dist_0, samples_dist_1)

    h_scores = []
    for i in range(samples_dist_0.shape[0]):
        h_scores.append((1-cost_matrix[i])[linear_sum_assignment(cost_matrix[i])].mean())

    return h_scores


@dataclass
class Tester:

    polyak: PolyakAverager
    num_samples: int
    num_classes: int
    geds: List
    similarity_samples: List
    similarity_experts: List
    hm_ious: List
    nonzero: int
    counter: int

    @torch.no_grad()
    def test_step(self, _: Engine, batch: Tensor) -> Dict[str, Any]:
        image, labels, _ = batch

        max_num_samples = np.max(self.num_samples)

        image = image.to(idist.device())
        image = image.repeat_interleave(max_num_samples, dim=0)

        self.polyak.average_model.eval()

        x = OneHotCategoricalBCHW(logits=torch.zeros(labels[:, 0].repeat_interleave(max_num_samples, dim=0).shape, device=labels.device)).sample().to(idist.device())

        prediction = self.polyak.average_model(x, image)['diffusion_out']
        prediction = prediction.reshape(labels.shape[0], -1, *labels.shape[2:])

        labels = labels.to(idist.device())
        labels = labels.argmax(dim=2)

        for idx, samples in enumerate(self.num_samples):
            ged, similarity_experts, similarity_samples = calc_batched_generalised_energy_distance(labels.cpu().numpy(), prediction[:, :samples].argmax(dim=2).cpu().numpy(), self.num_classes)

            self.geds[idx] += np.sum(ged)
            self.similarity_experts[idx] += np.sum(similarity_experts)
            self.similarity_samples[idx] += np.sum(similarity_samples)

        for idx, samples in enumerate(self.num_samples):
            lcm = np.lcm(samples, labels.shape[1])

            hm_labels = labels.repeat_interleave(lcm // labels.shape[1], dim=1).cpu().numpy()
            predictions = prediction[:, :samples].repeat_interleave(lcm // samples, dim=1).argmax(dim=2).cpu().numpy()

            hm_iou = batched_hungarian_matching(hm_labels, predictions, self.num_classes)

            self.hm_ious[idx] += np.sum(hm_iou)

        mean_prediction = torch.log(prediction).mean(dim=(1))
        
        nonzero = torch.count_nonzero(labels, dim=(2,3)) > 0
        self.nonzero += torch.sum(nonzero)
        
        mean_prediction = mean_prediction.repeat_interleave(torch.sum(nonzero, dim=1), dim=0)
        
        self.counter += 1
        
        LOGGER.info("Test batch %d with mean GED (%d) %.2f and mean HM IoU %.2f", self.counter, self.num_samples[-1], np.mean(ged), np.mean(hm_iou))
        
        return {'y': labels[nonzero], 'y_pred': mean_prediction}

    def objects_to_save(self, engine: Optional[Engine] = None) -> Dict[str, Any]:
        to_save: Dict[str, Any] = {
            "average_model": _flatten(self.polyak.average_model).unet,
        }

        return to_save

def build_engine(tester: Tester, validation_loader: DataLoader, num_classes: int) -> Engine:
    engine_test = Engine(tester.test_step)

    cm = ConfusionMatrix(num_classes=num_classes)

    IoU(cm).attach(engine_test, "IoU")
    mIoU(cm).attach(engine_test, "mIoU")
    DiceCoefficient(cm).attach(engine_test, "Dice")

    return engine_test


def load(filename: str, trainer, engine: Engine):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    to_load = trainer.objects_to_save(engine)
    ModelCheckpoint.load_objects(to_load, state)


def eval_lidc_uncertainty(params):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    LOGGER.info("%d GPUs available", torch.cuda.device_count())

    # Load the datasets
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)

    test_dataset = dataset_module.test_dataset(params["dataset_val_max_size"])  # type: ignore

    LOGGER.info("%d images in test dataset '%s'", len(test_dataset), dataset_file)

    test_loader = idist.auto_dataloader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["mp_loaders"]
    )

    # Build the model, optimizer, trainer and training engine
    input_shapes = [test_loader.dataset[0][0].shape, test_loader.dataset[0][1].shape]
    input_shapes[1] = input_shapes[1][1:]
    LOGGER.info("Input shapes: " + str(input_shapes))
    
    num_classes = input_shapes[1][0]
    model, average_model = [_build_model(params, input_shapes, input_shapes[0]) for _ in range(2)]
    polyak = PolyakAverager(model, average_model, alpha=params["polyak_alpha"])

    tester = Tester(polyak, params["evaluations"], num_classes, np.zeros(len(params["evaluations"])), np.zeros(len(params["evaluations"])), np.zeros(len(params["evaluations"])), np.zeros(len(params["evaluations"])), 0, 0)
    engine = build_engine(tester, test_loader, num_classes=num_classes)

    # Load a model (if requested in params.yml) to continue training from it
    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        load(load_from, trainer=tester, engine=engine)

    engine.state.max_epochs = None
    engine.run(test_loader, max_epochs=1)
    
    LOGGER.info(str(params['time_steps']) + ' ' + str(params['load_from']))

    LOGGER.info("Nonzero: %.4g", tester.nonzero / (len(test_dataset)*4))
    LOGGER.info("mIoU scores: %.4g", engine.state.metrics["mIoU"])
    LOGGER.info("IoU scores: %.4g and %.4g", *engine.state.metrics["IoU"])
    LOGGER.info("Dice scores: %.4g and %.4g", *engine.state.metrics["Dice"])
    LOGGER.info("Diversity experts: %.4g", tester.similarity_experts[0] / len(test_dataset))

    for i in range(4):
        LOGGER.info("GED (%d): %.4g", params["evaluations"][i], tester.geds[i] / len(test_dataset))
        LOGGER.info("Diversity samples (%d): %.4g", params["evaluations"][i], tester.similarity_samples[i] / len(test_dataset))
        LOGGER.info("HM IoU (%d):%.4g", params["evaluations"][i], tester.hm_ious[i] / len(test_dataset))
