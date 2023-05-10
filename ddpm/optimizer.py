import torch
from torch import nn
from .lr_functions import LRFcts
from torch.utils.data.dataloader import DataLoader
import logging
from typing import Optional
LOGGER = logging.getLogger(__name__)
import ignite.distributed as idist


def build_optimizer(params: dict, model: nn.Module, train_dataloader: DataLoader,
                    condition_encoder: Optional[nn.Module] = None, debug=False) -> dict:

    d = {'optimizer': None, 'lr_scheduler': None}  # to be returned

    parameters = model.parameters()
    if condition_encoder is not None:
        encoder_parameters = condition_encoder.parameters()
        parameters = list(encoder_parameters) + list(parameters)
        total_num_of_parameters = sum(map(torch.numel, parameters))
        LOGGER.info(f"adding condition encoder parameters to optimizer total parameters {total_num_of_parameters}")
    # this is the setting used throughout
    if 'optim' not in params:
        print('defaulting to adam optimiser with lr=1.0e-4')
        d['optimizer'] = torch.optim.Adam(parameters, lr=1.0e-4)  # default
        return d
    else:
        p_opt = params['optim']
        assert ('name' in p_opt) and ('learning_rate' in p_opt)  # todo clarify requirement

    if p_opt['name'] == 'SGD':
        wd = p_opt.get('weight_decay', 0.0005)  # usually only used with cnn on cityscapes
        momentum = p_opt.get('momentum', 0.9)
        d['optimizer'] = torch.optim.SGD(parameters, lr=p_opt['learning_rate'], momentum=momentum, weight_decay=wd)

    elif p_opt['name'] == 'Adam':
        d['optimizer'] = torch.optim.Adam(parameters, lr=p_opt['learning_rate'])

    elif p_opt['name'] == 'AdamW':  # usually only used with swin
        wd = p_opt.get('weight_decay', 0.01)
        betas = p_opt.get('betas', (0.9, 0.999))
        d['optimizer'] = torch.optim.AdamW(parameters, lr=p_opt['learning_rate'], betas=betas, weight_decay=wd)
    else:
        raise ValueError(f"optimizer {p_opt['name']} not recognized")

    if 'lr_function' in p_opt:
        # bs = params['batch_size']
        # polynomial learning rate requires assumes a predetermined number total steps (or equivalently epochs)
        # if no specific number of epochs was specified then use max_epochs (i.e train forever)
        epochs = p_opt.get('epochs', params['max_epochs'])

        using_grad_accumulation = params.get("grad_accumulation", False)
        if using_grad_accumulation:
            num_batches_per_epoch_mult = 1/params["grad_accumulation_step"]
        else:
            num_batches_per_epoch_mult = 1

        num_batches_per_epoch = len(train_dataloader)
        lr_total_steps = int(num_batches_per_epoch_mult * num_batches_per_epoch) * epochs

        lr_restart_steps = []

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d['optimizer'], lr_lambda=LRFcts(p_opt, lr_total_steps,
                                                                                          lr_restart_steps))

        d['lr_scheduler'] = lr_scheduler
        LOGGER.info("lr_schedule: '{}' starting lr {} and lr_params {} with "
                    "num_steps_per_epoch {} and total steps {}".format(p_opt['lr_function'],
                                                                       p_opt['learning_rate'],
                                                                       p_opt['lr_params'],
                                                                       num_batches_per_epoch,
                                                                       lr_total_steps))
        LOGGER.info(f"Using gradient accumulation: {using_grad_accumulation} -- "
                    f"over {params.get('grad_accumulation_step', None)} steps")
        if debug:
            debug_lr_plot(lr_total_steps, lr_scheduler)
    return d


def debug_lr_plot(lr_total_steps, lr_scheduler):
    lrs = []
    for i in range(lr_total_steps):
        lrs.append(lr_scheduler.get_lr())

        if i <= lr_total_steps - 2:
            lr_scheduler.step()

    import matplotlib.pyplot as plt
    lr_funct = plt.plot(lrs)
    plt.show()
    a = 1


if __name__ == '__main__':
    par = {
        'optim': {'name': 'sgd',
                  'learning_rate': 1.0e-4,
                  'weight_decay': 0.0005,

                  }
    }